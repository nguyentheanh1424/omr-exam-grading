#include "omr_api.h"
#include "marker_detect.h"
#include "warp_global.h"
#include "idw_refine.h"
#include "region_refine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

struct OMR_Handle {
    uint32_t magic;
};

namespace {

constexpr uint32_t kHandleMagic = 0x4F4D5231u;  // "OMR1"
constexpr float kLargeNegative = -1e9f;

void copy_error_message(char* dst, const char* src) {
    if (dst == nullptr) {
        return;
    }
    if (src == nullptr) {
        dst[0] = '\0';
        return;
    }
    std::snprintf(dst, OMR_ERROR_MESSAGE_CAPACITY, "%s", src);
}

int32_t set_error(OMR_Result* out_result, int32_t code, const char* message) {
    if (out_result != nullptr) {
        out_result->err_code = code;
        copy_error_message(out_result->error_message, message);
    }
    return code;
}

bool is_finite(float x) {
    return std::isfinite(static_cast<double>(x)) != 0;
}

bool checked_mul_size_t(size_t a, size_t b, size_t* out) {
    if (out == nullptr) {
        return false;
    }
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > (std::numeric_limits<size_t>::max() / b)) {
        return false;
    }
    *out = a * b;
    return true;
}

bool validate_image(
    const OMR_ImageView* image,
    char* err,
    size_t err_cap
) {
    (void)err_cap;
    if (image == nullptr) {
        copy_error_message(err, "image is null");
        return false;
    }
    if (image->data == nullptr) {
        copy_error_message(err, "image.data is null");
        return false;
    }
    if (image->width <= 0 || image->height <= 0) {
        copy_error_message(err, "image width/height must be > 0");
        return false;
    }
    if (!(image->channels == 1 || image->channels == 3)) {
        copy_error_message(err, "image channels must be 1 (gray) or 3 (BGR)");
        return false;
    }
    const int32_t min_stride = image->width * image->channels;
    if (image->stride < min_stride) {
        copy_error_message(err, "image stride is too small");
        return false;
    }
    if (image->stride <= 0) {
        copy_error_message(err, "image stride must be > 0");
        return false;
    }
    return true;
}

bool validate_form(
    const OMR_FormSpec* form,
    const OMR_ImageView* image,
    const OMR_RuntimeOptions* runtime_options,
    char* err,
    size_t err_cap
) {
    (void)err_cap;
    if (form == nullptr) {
        copy_error_message(err, "form is null");
        return false;
    }
    if (form->n_questions <= 0) {
        copy_error_message(err, "form.n_questions must be > 0");
        return false;
    }
    if (form->n_options_per_question <= 0) {
        copy_error_message(err, "form.n_options_per_question must be > 0");
        return false;
    }
    if (form->output_width <= 0 || form->output_height <= 0) {
        copy_error_message(err, "form output width/height must be > 0");
        return false;
    }
    if (form->circle_rois == nullptr || form->n_circle_rois <= 0) {
        copy_error_message(err, "form circle_rois is missing");
        return false;
    }
    if (form->n_template_markers < 0 || form->n_detected_markers < 0 || form->n_region_windows < 0 ||
        form->n_metadata_fields < 0 || form->n_metadata_bubbles < 0) {
        copy_error_message(err, "marker counts must be >= 0");
        return false;
    }
    if (form->n_template_markers > 0 && form->template_markers == nullptr) {
        copy_error_message(err, "template_markers pointer is null");
        return false;
    }
    if (form->n_detected_markers > 0 && form->detected_markers == nullptr) {
        copy_error_message(err, "detected_markers pointer is null");
        return false;
    }
    if (form->n_region_windows > 0 && form->region_windows == nullptr) {
        copy_error_message(err, "region_windows pointer is null");
        return false;
    }
    if (form->n_metadata_fields > 0 && form->metadata_fields == nullptr) {
        copy_error_message(err, "metadata_fields pointer is null");
        return false;
    }
    if (form->n_metadata_bubbles > 0 && form->metadata_bubbles == nullptr) {
        copy_error_message(err, "metadata_bubbles pointer is null");
        return false;
    }
    if (form->answer_key == nullptr) {
        copy_error_message(err, "form.answer_key is null");
        return false;
    }
    if (form->n_answer_key != form->n_questions) {
        copy_error_message(err, "form.n_answer_key must equal form.n_questions");
        return false;
    }

    std::vector<int32_t> marker_ids;
    marker_ids.reserve(static_cast<size_t>(form->n_template_markers));
    for (int32_t i = 0; i < form->n_template_markers; ++i) {
        const OMR_MarkerTemplate& m = form->template_markers[i];
        if (!is_finite(m.x) || !is_finite(m.y)) {
            copy_error_message(err, "template marker coordinates must be finite");
            return false;
        }
        if (m.x < 0.0f || m.x >= static_cast<float>(form->output_width) ||
            m.y < 0.0f || m.y >= static_cast<float>(form->output_height)) {
            copy_error_message(err, "template marker is out of output bounds");
            return false;
        }
        if (std::find(marker_ids.begin(), marker_ids.end(), m.id) != marker_ids.end()) {
            copy_error_message(err, "duplicate template marker id");
            return false;
        }
        marker_ids.push_back(m.id);
    }

    std::vector<int32_t> detected_ids;
    detected_ids.reserve(static_cast<size_t>(form->n_detected_markers));
    for (int32_t i = 0; i < form->n_detected_markers; ++i) {
        const OMR_DetectedMarker& m = form->detected_markers[i];
        if (!is_finite(m.x) || !is_finite(m.y)) {
            copy_error_message(err, "detected marker coordinates must be finite");
            return false;
        }
        if (m.x < 0.0f || m.x >= static_cast<float>(image->width) ||
            m.y < 0.0f || m.y >= static_cast<float>(image->height)) {
            copy_error_message(err, "detected marker is out of input image bounds");
            return false;
        }
        if (std::find(detected_ids.begin(), detected_ids.end(), m.id) != detected_ids.end()) {
            copy_error_message(err, "duplicate detected marker id");
            return false;
        }
        detected_ids.push_back(m.id);
    }

    for (int32_t i = 0; i < form->n_region_windows; ++i) {
        const OMR_RegionWindow& w = form->region_windows[i];
        if (w.n_marker_ids != 4) {
            copy_error_message(err, "each region window must contain exactly 4 marker ids");
            return false;
        }
        for (int32_t a = 0; a < w.n_marker_ids; ++a) {
            for (int32_t b = a + 1; b < w.n_marker_ids; ++b) {
                if (w.marker_ids[a] == w.marker_ids[b]) {
                    copy_error_message(err, "duplicate marker id inside one region window");
                    return false;
                }
            }
            if (!marker_ids.empty() &&
                std::find(marker_ids.begin(), marker_ids.end(), w.marker_ids[a]) == marker_ids.end()) {
                copy_error_message(err, "region window references unknown marker id");
                return false;
            }
        }
    }

    if (runtime_options != nullptr && runtime_options->assume_aligned_input == 0) {
        if (form->template_markers == nullptr || form->n_template_markers < 4) {
            copy_error_message(err, "template_markers must have at least 4 points when warp is required");
            return false;
        }
        if (form->detected_markers != nullptr &&
            form->n_detected_markers > 0 &&
            form->n_detected_markers < 4) {
            copy_error_message(err, "detected_markers must have at least 4 points when provided");
            return false;
        }
        if (form->region_windows == nullptr || form->n_region_windows <= 0) {
            copy_error_message(err, "region_windows must be provided when warp is required");
            return false;
        }
    }

    for (int32_t i = 0; i < form->n_answer_key; ++i) {
        const int32_t key = form->answer_key[i];
        if (key < -1 || key >= form->n_options_per_question) {
            copy_error_message(err, "answer_key value out of range");
            return false;
        }
    }

    size_t seen_size = 0;
    if (!checked_mul_size_t(
            static_cast<size_t>(form->n_questions),
            static_cast<size_t>(form->n_options_per_question),
            &seen_size)) {
        copy_error_message(err, "ROI cardinality overflow");
        return false;
    }
    std::vector<uint8_t> seen(seen_size, 0);
    std::vector<int32_t> rois_per_question(static_cast<size_t>(form->n_questions), 0);

    const int32_t roi_width_limit =
        (runtime_options != nullptr && runtime_options->assume_aligned_input == 0)
            ? form->output_width
            : image->width;
    const int32_t roi_height_limit =
        (runtime_options != nullptr && runtime_options->assume_aligned_input == 0)
            ? form->output_height
            : image->height;

    for (int32_t i = 0; i < form->n_circle_rois; ++i) {
        const OMR_CircleROI& roi = form->circle_rois[i];
        if (roi.r <= 0) {
            copy_error_message(err, "ROI radius must be > 0");
            return false;
        }
        if (roi.question < 0 || roi.question >= form->n_questions) {
            copy_error_message(err, "ROI question index out of range");
            return false;
        }
        if (roi.option < 0 || roi.option >= form->n_options_per_question) {
            copy_error_message(err, "ROI option index out of range");
            return false;
        }
        if (roi.selection_mode < 0 || roi.selection_mode > 1) {
            copy_error_message(err, "ROI selection_mode is out of range");
            return false;
        }

        if (roi.cx - roi.r < 0 || roi.cx + roi.r >= roi_width_limit ||
            roi.cy - roi.r < 0 || roi.cy + roi.r >= roi_height_limit) {
            copy_error_message(err, "ROI circle exceeds image bounds");
            return false;
        }

        const size_t idx = static_cast<size_t>(roi.question) *
                           static_cast<size_t>(form->n_options_per_question) +
                           static_cast<size_t>(roi.option);
        if (seen[idx] != 0) {
            copy_error_message(err, "duplicate ROI (question, option)");
            return false;
        }
        seen[idx] = 1;
        rois_per_question[static_cast<size_t>(roi.question)] += 1;
    }

    for (int32_t q = 0; q < form->n_questions; ++q) {
        if (rois_per_question[static_cast<size_t>(q)] != form->n_options_per_question) {
            copy_error_message(err, "each question must provide exactly n_options_per_question ROIs");
            return false;
        }
    }

    std::vector<int32_t> metadata_field_ids;
    metadata_field_ids.reserve(static_cast<size_t>(form->n_metadata_fields));
    size_t metadata_total = 0;
    for (int32_t i = 0; i < form->n_metadata_fields; ++i) {
        const OMR_MetadataField& field = form->metadata_fields[i];
        if (field.n_columns <= 0 || field.n_rows <= 0) {
            copy_error_message(err, "metadata field dims must be > 0");
            return false;
        }
        if (std::find(metadata_field_ids.begin(), metadata_field_ids.end(), field.field_id) != metadata_field_ids.end()) {
            copy_error_message(err, "duplicate metadata field id");
            return false;
        }
        metadata_field_ids.push_back(field.field_id);
        if (!checked_mul_size_t(
                static_cast<size_t>(field.n_columns),
                static_cast<size_t>(field.n_rows),
                &seen_size)) {
            copy_error_message(err, "metadata field cardinality overflow");
            return false;
        }
        metadata_total += seen_size;
    }

    std::vector<uint8_t> metadata_seen(metadata_total, 0);
    std::vector<size_t> metadata_offsets(static_cast<size_t>(form->n_metadata_fields), 0);
    size_t running_offset = 0;
    for (int32_t i = 0; i < form->n_metadata_fields; ++i) {
        metadata_offsets[static_cast<size_t>(i)] = running_offset;
        const OMR_MetadataField& field = form->metadata_fields[i];
        size_t field_size = 0;
        checked_mul_size_t(
            static_cast<size_t>(field.n_columns),
            static_cast<size_t>(field.n_rows),
            &field_size
        );
        running_offset += field_size;
    }

    auto metadata_index_for_id = [&](int32_t field_id) -> int32_t {
        for (int32_t i = 0; i < form->n_metadata_fields; ++i) {
            if (form->metadata_fields[i].field_id == field_id) {
                return i;
            }
        }
        return -1;
    };

    for (int32_t i = 0; i < form->n_metadata_bubbles; ++i) {
        const OMR_MetadataBubble& bubble = form->metadata_bubbles[i];
        if (bubble.r <= 0) {
            copy_error_message(err, "metadata bubble radius must be > 0");
            return false;
        }
        const int32_t field_idx = metadata_index_for_id(bubble.field_id);
        if (field_idx < 0) {
            copy_error_message(err, "metadata bubble references unknown field id");
            return false;
        }
        const OMR_MetadataField& field = form->metadata_fields[field_idx];
        if (bubble.column < 0 || bubble.column >= field.n_columns ||
            bubble.row < 0 || bubble.row >= field.n_rows) {
            copy_error_message(err, "metadata bubble row/column out of range");
            return false;
        }
        if (bubble.cx - bubble.r < 0 || bubble.cx + bubble.r >= roi_width_limit ||
            bubble.cy - bubble.r < 0 || bubble.cy + bubble.r >= roi_height_limit) {
            copy_error_message(err, "metadata bubble exceeds image bounds");
            return false;
        }

        const size_t field_offset = metadata_offsets[static_cast<size_t>(field_idx)];
        const size_t idx = field_offset +
                           static_cast<size_t>(bubble.column) * static_cast<size_t>(field.n_rows) +
                           static_cast<size_t>(bubble.row);
        if (metadata_seen[idx] != 0) {
            copy_error_message(err, "duplicate metadata bubble (field, column, row)");
            return false;
        }
        metadata_seen[idx] = 1;
    }

    for (int32_t i = 0; i < form->n_metadata_fields; ++i) {
        const OMR_MetadataField& field = form->metadata_fields[i];
        const size_t field_offset = metadata_offsets[static_cast<size_t>(i)];
        const size_t field_size =
            static_cast<size_t>(field.n_columns) * static_cast<size_t>(field.n_rows);
        for (size_t k = 0; k < field_size; ++k) {
            if (metadata_seen[field_offset + k] == 0) {
                copy_error_message(err, "each metadata field must provide a full bubble grid");
                return false;
            }
        }
    }

    return true;
}

bool validate_warp_params(const OMR_WarpParams* p, char* err, size_t err_cap) {
    (void)err_cap;
    if (p == nullptr) {
        copy_error_message(err, "warp_params is null");
        return false;
    }
    if (!is_finite(p->global_h_ransac_thresh) || p->global_h_ransac_thresh <= 0.0f) {
        copy_error_message(err, "global_h_ransac_thresh must be finite and > 0");
        return false;
    }
    if (!is_finite(p->local_h_ransac_thresh) || p->local_h_ransac_thresh <= 0.0f) {
        copy_error_message(err, "local_h_ransac_thresh must be finite and > 0");
        return false;
    }
    if (p->region_bbox_margin_px < 0) {
        copy_error_message(err, "region_bbox_margin_px must be >= 0");
        return false;
    }
    if (p->global_idw_grid_w <= 0 || p->global_idw_grid_h <= 0) {
        copy_error_message(err, "global IDW grid must be > 0");
        return false;
    }
    if (p->patch_idw_grid_w <= 0 || p->patch_idw_grid_h <= 0) {
        copy_error_message(err, "patch IDW grid must be > 0");
        return false;
    }
    if (!is_finite(p->global_idw_power) || p->global_idw_power <= 0.0f ||
        !is_finite(p->patch_idw_power) || p->patch_idw_power <= 0.0f) {
        copy_error_message(err, "IDW power must be finite and > 0");
        return false;
    }
    if (!is_finite(p->global_idw_eps) || p->global_idw_eps <= 0.0f ||
        !is_finite(p->patch_idw_eps) || p->patch_idw_eps <= 0.0f) {
        copy_error_message(err, "IDW eps must be finite and > 0");
        return false;
    }
    if (!is_finite(p->skip_idw_if_residual_lt_px) || p->skip_idw_if_residual_lt_px < 0.0f) {
        copy_error_message(err, "skip_idw_if_residual_lt_px must be finite and >= 0");
        return false;
    }
    if (!(p->residual_breakpoints_px[0] < p->residual_breakpoints_px[1] &&
          p->residual_breakpoints_px[1] < p->residual_breakpoints_px[2])) {
        copy_error_message(err, "residual_breakpoints_px must be strictly increasing");
        return false;
    }
    return true;
}

bool validate_binarize_params(const OMR_BinarizeParams* p, char* err, size_t err_cap) {
    (void)err_cap;
    if (p == nullptr) {
        copy_error_message(err, "bin_params is null");
        return false;
    }
    if (!(p->method == OMR_BINARIZE_DUAL || p->method == OMR_BINARIZE_SKELETON)) {
        copy_error_message(err, "binarize method is invalid");
        return false;
    }
    if (p->blur_ksize <= 0 || (p->blur_ksize % 2) == 0) {
        copy_error_message(err, "blur_ksize must be odd and > 0");
        return false;
    }
    if (!is_finite(p->fill_percentile) || p->fill_percentile < 0.0f || p->fill_percentile > 100.0f) {
        copy_error_message(err, "fill_percentile must be in [0, 100]");
        return false;
    }
    if (p->thin_iterations < 0) {
        copy_error_message(err, "thin_iterations must be >= 0");
        return false;
    }
    if (p->denoise_kernel_size <= 0) {
        copy_error_message(err, "denoise_kernel_size must be > 0");
        return false;
    }
    return true;
}

bool validate_grading_params(const OMR_GradingParams* p, char* err, size_t err_cap) {
    (void)err_cap;
    if (p == nullptr) {
        copy_error_message(err, "grading_params is null");
        return false;
    }
    if (!is_finite(p->abs_th) || p->abs_th < 0.0f || p->abs_th > 1.0f) {
        copy_error_message(err, "abs_th must be in [0, 1]");
        return false;
    }
    if (!is_finite(p->rel_th) || p->rel_th < 0.0f || p->rel_th > 1.0f) {
        copy_error_message(err, "rel_th must be in [0, 1]");
        return false;
    }
    if (!is_finite(p->patch_radius_multiplier) || p->patch_radius_multiplier <= 0.0f) {
        copy_error_message(err, "patch_radius_multiplier must be > 0");
        return false;
    }
    if (!is_finite(p->fill_inner_ratio) || !is_finite(p->fill_outer_ratio) ||
        !is_finite(p->bg_inner_ratio) || !is_finite(p->bg_outer_ratio)) {
        copy_error_message(err, "ring ratios must be finite");
        return false;
    }
    if (!(p->fill_inner_ratio >= 0.0f && p->fill_inner_ratio < p->fill_outer_ratio &&
          p->fill_outer_ratio < p->bg_inner_ratio &&
          p->bg_inner_ratio < p->bg_outer_ratio)) {
        copy_error_message(err, "ring ratio ordering is invalid");
        return false;
    }
    if (p->min_valid_pixels <= 0) {
        copy_error_message(err, "min_valid_pixels must be > 0");
        return false;
    }
    if (p->clahe_tile_w <= 0 || p->clahe_tile_h <= 0) {
        copy_error_message(err, "CLAHE tile size must be > 0");
        return false;
    }
    if (p->gaussian_ksize_w <= 0 || p->gaussian_ksize_h <= 0 ||
        (p->gaussian_ksize_w % 2) == 0 || (p->gaussian_ksize_h % 2) == 0) {
        copy_error_message(err, "Gaussian kernel size must be odd and > 0");
        return false;
    }
    if (!is_finite(p->calibration_percentile) ||
        p->calibration_percentile <= 0.0f ||
        p->calibration_percentile > 1.0f) {
        copy_error_message(err, "calibration_percentile must be in (0, 1]");
        return false;
    }
    if (!(p->abs_th_min <= p->abs_th_max && p->rel_th_min <= p->rel_th_max)) {
        copy_error_message(err, "threshold min/max bounds are invalid");
        return false;
    }
    return true;
}

bool validate_runtime_options(const OMR_RuntimeOptions* p, char* err, size_t err_cap) {
    (void)err_cap;
    if (p == nullptr) {
        copy_error_message(err, "runtime_options is null");
        return false;
    }
    if (!(p->assume_aligned_input == 0 || p->assume_aligned_input == 1)) {
        copy_error_message(err, "assume_aligned_input must be 0 or 1");
        return false;
    }
    if (!(p->return_scored_image == 0 || p->return_scored_image == 1)) {
        copy_error_message(err, "return_scored_image must be 0 or 1");
        return false;
    }
    return true;
}

inline uint8_t gray_at(const OMR_ImageView& image, int32_t x, int32_t y) {
    const uint8_t* row = image.data + static_cast<size_t>(y) * static_cast<size_t>(image.stride);
    if (image.channels == 1) {
        return row[x];
    }

    const size_t idx = static_cast<size_t>(x) * static_cast<size_t>(image.channels);
    const float b = static_cast<float>(row[idx + 0]);
    const float g = static_cast<float>(row[idx + 1]);
    const float r = static_cast<float>(row[idx + 2]);
    const float gray = 0.114f * b + 0.587f * g + 0.299f * r;
    return static_cast<uint8_t>(std::round(std::clamp(gray, 0.0f, 255.0f)));
}

float bubble_score(
    const OMR_ImageView& image,
    const OMR_CircleROI& roi,
    const OMR_GradingParams& g
) {
    const int32_t r_fill_in = std::max(0, static_cast<int32_t>(g.fill_inner_ratio * roi.r));
    const int32_t r_fill_out = std::max(0, static_cast<int32_t>(g.fill_outer_ratio * roi.r));
    const int32_t r_bg_in = std::max(0, static_cast<int32_t>(g.bg_inner_ratio * roi.r));
    const int32_t r_bg_out = std::max(0, static_cast<int32_t>(g.bg_outer_ratio * roi.r));

    const int32_t r_max = std::max(r_fill_out, r_bg_out);
    const int32_t x0 = std::max(0, roi.cx - r_max);
    const int32_t y0 = std::max(0, roi.cy - r_max);
    const int32_t x1 = std::min(image.width - 1, roi.cx + r_max);
    const int32_t y1 = std::min(image.height - 1, roi.cy + r_max);

    const int32_t r_fill_in2 = r_fill_in * r_fill_in;
    const int32_t r_fill_out2 = r_fill_out * r_fill_out;
    const int32_t r_bg_in2 = r_bg_in * r_bg_in;
    const int32_t r_bg_out2 = r_bg_out * r_bg_out;

    double fill_sum = 0.0;
    double bg_sum = 0.0;
    int32_t fill_count = 0;
    int32_t bg_count = 0;

    for (int32_t y = y0; y <= y1; ++y) {
        const int32_t dy = y - roi.cy;
        for (int32_t x = x0; x <= x1; ++x) {
            const int32_t dx = x - roi.cx;
            const int32_t dist2 = dx * dx + dy * dy;

            if (dist2 >= r_fill_in2 && dist2 <= r_fill_out2) {
                fill_sum += static_cast<double>(gray_at(image, x, y));
                ++fill_count;
            } else if (dist2 >= r_bg_in2 && dist2 <= r_bg_out2) {
                bg_sum += static_cast<double>(gray_at(image, x, y));
                ++bg_count;
            }
        }
    }

    if (fill_count < g.min_valid_pixels || bg_count < g.min_valid_pixels) {
        return 0.0f;
    }

    const float fill_mean = static_cast<float>(fill_sum / static_cast<double>(fill_count));
    const float bg_mean = static_cast<float>(bg_sum / static_cast<double>(bg_count));

    const float fill_dark = 1.0f - (fill_mean / 255.0f);
    const float bg_dark = 1.0f - (bg_mean / 255.0f);
    return std::max(0.0f, fill_dark - bg_dark);
}

}  // namespace

uint32_t omr_api_version(void) {
    return OMR_API_VERSION;
}

OMR_Handle* omr_create(void) {
    OMR_Handle* handle = new (std::nothrow) OMR_Handle;
    if (handle == nullptr) {
        return nullptr;
    }
    handle->magic = kHandleMagic;
    return handle;
}

void omr_destroy(OMR_Handle* handle) {
    if (handle == nullptr) {
        return;
    }
    handle->magic = 0;
    delete handle;
}

void omr_init_result(OMR_Result* out_result) {
    if (out_result == nullptr) {
        return;
    }
    std::memset(out_result, 0, sizeof(*out_result));
    out_result->err_code = OMR_OK;
}

void omr_free_result(OMR_Result* out_result) {
    if (out_result == nullptr) {
        return;
    }
    if (out_result->answers != nullptr) {
        std::free(out_result->answers);
        out_result->answers = nullptr;
    }
    if (out_result->metadata_selected_rows != nullptr) {
        std::free(out_result->metadata_selected_rows);
        out_result->metadata_selected_rows = nullptr;
    }
    if (out_result->scored_image_data != nullptr) {
        std::free(out_result->scored_image_data);
        out_result->scored_image_data = nullptr;
    }
    out_result->n_answers = 0;
    out_result->n_metadata_selected_rows = 0;
    out_result->scored_image_width = 0;
    out_result->scored_image_height = 0;
    out_result->scored_image_stride = 0;
    out_result->scored_image_channels = 0;
}

void omr_init_default_warp_params(OMR_WarpParams* out_params) {
    if (out_params == nullptr) {
        return;
    }
    std::memset(out_params, 0, sizeof(*out_params));
    out_params->apriltag_dict = OMR_TAG_DICT_APRILTAG_16H5;
    out_params->global_h_ransac_thresh = 2.0f;
    out_params->local_h_ransac_thresh = 3.0f;
    out_params->region_bbox_margin_px = 100;
    out_params->use_global_idw = 0;
    out_params->use_region_refine = 1;

    out_params->global_idw_grid_w = 48;
    out_params->global_idw_grid_h = 36;
    out_params->global_idw_power = 2.5f;
    out_params->global_idw_eps = 1e-3f;

    out_params->patch_idw_grid_w = 24;
    out_params->patch_idw_grid_h = 24;
    out_params->patch_idw_power = 4.0f;
    out_params->patch_idw_eps = 1e-3f;

    out_params->skip_idw_if_residual_lt_px = 0.5f;
    out_params->residual_breakpoints_px[0] = 3.0f;
    out_params->residual_breakpoints_px[1] = 8.0f;
    out_params->residual_breakpoints_px[2] = 15.0f;

    out_params->residual_factors[0] = 0.25f;
    out_params->residual_factors[1] = 0.15f;
    out_params->residual_factors[2] = 0.20f;
    out_params->residual_factors[3] = 0.30f;
}

void omr_init_default_binarize_params(OMR_BinarizeParams* out_params) {
    if (out_params == nullptr) {
        return;
    }
    std::memset(out_params, 0, sizeof(*out_params));
    out_params->method = OMR_BINARIZE_DUAL;
    out_params->blur_ksize = 1;
    out_params->fill_percentile = 8.0f;
    out_params->thin_iterations = 1;
    out_params->denoise_kernel_size = 1;
}

void omr_init_default_grading_params(OMR_GradingParams* out_params) {
    if (out_params == nullptr) {
        return;
    }
    std::memset(out_params, 0, sizeof(*out_params));
    out_params->abs_th = 0.12f;
    out_params->rel_th = 0.04f;
    out_params->auto_threshold = 1;

    out_params->clahe_clip_limit = 3.0f;
    out_params->clahe_tile_w = 8;
    out_params->clahe_tile_h = 8;
    out_params->gaussian_ksize_w = 3;
    out_params->gaussian_ksize_h = 3;
    out_params->gaussian_sigma = 0.0f;

    out_params->patch_radius_multiplier = 1.6f;
    out_params->fill_inner_ratio = 0.45f;
    out_params->fill_outer_ratio = 0.85f;
    out_params->bg_inner_ratio = 1.05f;
    out_params->bg_outer_ratio = 1.45f;
    out_params->min_valid_pixels = 20;

    out_params->min_questions_for_calibration = 8;
    out_params->calibration_percentile = 0.6f;
    out_params->abs_th_mad_multiplier = 6.5f;
    out_params->rel_th_mad_multiplier = 4.5f;
    out_params->abs_th_baseline_offset = 0.015f;
    out_params->rel_th_baseline_offset = 0.004f;
    out_params->abs_th_min = 0.20f;
    out_params->abs_th_max = 0.40f;
    out_params->rel_th_min = 0.015f;
    out_params->rel_th_max = 0.25f;
}

void omr_init_default_runtime_options(OMR_RuntimeOptions* out_options) {
    if (out_options == nullptr) {
        return;
    }
    std::memset(out_options, 0, sizeof(*out_options));
    out_options->assume_aligned_input = 1;
    out_options->return_scored_image = 1;
    out_options->return_intermediate_steps = 0;
    out_options->debug_level = 0;
}

int32_t omr_process(
    OMR_Handle* handle,
    const OMR_ImageView* image,
    const OMR_FormSpec* form,
    const OMR_WarpParams* warp_params,
    const OMR_BinarizeParams* bin_params,
    const OMR_GradingParams* grading_params,
    const OMR_RuntimeOptions* runtime_options,
    OMR_Result* out_result
) {
    if (out_result == nullptr) {
        return OMR_ERR_NULL_ARGUMENT;
    }

    omr_init_result(out_result);

    if (handle == nullptr) {
        return set_error(out_result, OMR_ERR_INVALID_HANDLE, "handle is null");
    }
    if (handle->magic != kHandleMagic) {
        return set_error(out_result, OMR_ERR_INVALID_HANDLE, "handle is invalid");
    }

    if (!validate_image(image, out_result->error_message, OMR_ERROR_MESSAGE_CAPACITY)) {
        return set_error(out_result, OMR_ERR_BAD_IMAGE, out_result->error_message);
    }
    if (!validate_runtime_options(runtime_options, out_result->error_message, OMR_ERROR_MESSAGE_CAPACITY)) {
        return set_error(out_result, OMR_ERR_BAD_CONFIG, out_result->error_message);
    }
    if (!validate_form(form, image, runtime_options, out_result->error_message, OMR_ERROR_MESSAGE_CAPACITY)) {
        return set_error(out_result, OMR_ERR_INVALID_ROI_LAYOUT, out_result->error_message);
    }
    if (!validate_warp_params(warp_params, out_result->error_message, OMR_ERROR_MESSAGE_CAPACITY)) {
        return set_error(out_result, OMR_ERR_BAD_CONFIG, out_result->error_message);
    }
    if (!validate_binarize_params(bin_params, out_result->error_message, OMR_ERROR_MESSAGE_CAPACITY)) {
        return set_error(out_result, OMR_ERR_BAD_CONFIG, out_result->error_message);
    }
    if (!validate_grading_params(grading_params, out_result->error_message, OMR_ERROR_MESSAGE_CAPACITY)) {
        return set_error(out_result, OMR_ERR_BAD_CONFIG, out_result->error_message);
    }

    OMR_ImageView working_image = *image;
    std::vector<uint8_t> warped_storage;
    if (runtime_options->assume_aligned_input == 0) {
        OMR_FormSpec process_form = *form;
        std::vector<OMR_DetectedMarker> auto_detected;
        if (process_form.detected_markers == nullptr || process_form.n_detected_markers < 4) {
            if (!omr_marker::detect_markers_v1(
                    *image,
                    form->template_markers,
                    form->n_template_markers,
                    &auto_detected,
                    out_result->error_message,
                    OMR_ERROR_MESSAGE_CAPACITY)) {
                return set_error(out_result, OMR_ERR_INSUFFICIENT_MARKERS, out_result->error_message);
            }
            process_form.detected_markers = auto_detected.data();
            process_form.n_detected_markers = static_cast<int32_t>(auto_detected.size());
        }

        float h_src_to_dst[9] = {0.0f};
        int32_t inliers = 0;
        if (!omr_warp::compute_global_h_from_markers(
                process_form,
                warp_params->global_h_ransac_thresh,
                300,
                h_src_to_dst,
                &inliers,
                runtime_options->debug_level)) {
            return set_error(
                out_result,
                OMR_ERR_INSUFFICIENT_MARKERS,
                "failed to compute global homography from detected/template markers"
            );
        }

        if (!omr_warp::warp_image_bilinear(
                *image,
                form->output_width,
                form->output_height,
                h_src_to_dst,
                &warped_storage,
                &working_image,
                out_result->error_message,
                OMR_ERROR_MESSAGE_CAPACITY)) {
            return set_error(out_result, OMR_ERR_WARP_FAILED, out_result->error_message);
        }

        std::vector<uint8_t> idw_storage;
        OMR_ImageView idw_view{};
        if (warp_params->use_global_idw != 0) {
            if (!omr_idw::refine_global_idw(
                    working_image,
                    process_form,
                    *warp_params,
                    h_src_to_dst,
                    &idw_storage,
                    &idw_view,
                    out_result->error_message,
                    OMR_ERROR_MESSAGE_CAPACITY)) {
                return set_error(out_result, OMR_ERR_WARP_FAILED, out_result->error_message);
            }
            warped_storage.swap(idw_storage);
            working_image = idw_view;
            working_image.data = warped_storage.data();
        }

        std::vector<uint8_t> region_storage;
        OMR_ImageView region_view{};
        if (warp_params->use_region_refine != 0) {
            if (!omr_region::refine_regions_local(
                    working_image,
                    process_form,
                    *warp_params,
                    *bin_params,
                    h_src_to_dst,
                    runtime_options->debug_level,
                    &region_storage,
                    &region_view,
                    out_result->error_message,
                    OMR_ERROR_MESSAGE_CAPACITY)) {
                return set_error(out_result, OMR_ERR_WARP_FAILED, out_result->error_message);
            }
            warped_storage.swap(region_storage);
            working_image = region_view;
            working_image.data = warped_storage.data();
        }
    }

    std::vector<float> best_val(static_cast<size_t>(form->n_questions), kLargeNegative);
    std::vector<float> second_val(static_cast<size_t>(form->n_questions), kLargeNegative);
    std::vector<int32_t> best_opt(static_cast<size_t>(form->n_questions), -1);

    for (int32_t i = 0; i < form->n_circle_rois; ++i) {
        const OMR_CircleROI& roi = form->circle_rois[i];
        const float s = bubble_score(working_image, roi, *grading_params);

        const size_t q_idx = static_cast<size_t>(roi.question);
        if (s > best_val[q_idx]) {
            second_val[q_idx] = best_val[q_idx];
            best_val[q_idx] = s;
            best_opt[q_idx] = roi.option;
        } else if (s > second_val[q_idx]) {
            second_val[q_idx] = s;
        }
    }

    const size_t answers_count = static_cast<size_t>(form->n_questions);
    int32_t* answers = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * answers_count));
    if (answers == nullptr) {
        return set_error(out_result, OMR_ERR_ALLOCATION_FAILED, "failed to allocate answers");
    }
    for (int32_t q = 0; q < form->n_questions; ++q) {
        answers[q] = -1;
    }

    for (int32_t q = 0; q < form->n_questions; ++q) {
        const size_t q_idx = static_cast<size_t>(q);
        if (best_opt[q_idx] >= 0 &&
            best_val[q_idx] >= grading_params->abs_th &&
            (best_val[q_idx] - second_val[q_idx]) >= grading_params->rel_th) {
            answers[q] = best_opt[q_idx];
        }
    }

    size_t metadata_selection_count = 0;
    for (int32_t i = 0; i < form->n_metadata_fields; ++i) {
        metadata_selection_count += static_cast<size_t>(form->metadata_fields[i].n_columns);
    }

    int32_t* metadata_selected_rows = nullptr;
    if (metadata_selection_count > 0) {
        metadata_selected_rows = static_cast<int32_t*>(
            std::malloc(sizeof(int32_t) * metadata_selection_count)
        );
        if (metadata_selected_rows == nullptr) {
            std::free(out_result->answers);
            out_result->answers = nullptr;
            out_result->n_answers = 0;
            return set_error(out_result, OMR_ERR_ALLOCATION_FAILED, "failed to allocate metadata selections");
        }
        for (size_t i = 0; i < metadata_selection_count; ++i) {
            metadata_selected_rows[i] = -1;
        }

        std::vector<size_t> metadata_offsets(static_cast<size_t>(form->n_metadata_fields), 0);
        std::vector<float> metadata_best_val(metadata_selection_count, kLargeNegative);
        std::vector<float> metadata_second_val(metadata_selection_count, kLargeNegative);
        std::vector<int32_t> metadata_best_row(metadata_selection_count, -1);

        size_t running_offset = 0;
        for (int32_t i = 0; i < form->n_metadata_fields; ++i) {
            metadata_offsets[static_cast<size_t>(i)] = running_offset;
            running_offset += static_cast<size_t>(form->metadata_fields[i].n_columns);
        }

        auto metadata_index_for_id = [&](int32_t field_id) -> int32_t {
            for (int32_t i = 0; i < form->n_metadata_fields; ++i) {
                if (form->metadata_fields[i].field_id == field_id) {
                    return i;
                }
            }
            return -1;
        };

        for (int32_t i = 0; i < form->n_metadata_bubbles; ++i) {
            const OMR_MetadataBubble& bubble = form->metadata_bubbles[i];
            const int32_t field_idx = metadata_index_for_id(bubble.field_id);
            if (field_idx < 0) {
                continue;
            }
            const OMR_CircleROI roi{
                bubble.cx,
                bubble.cy,
                bubble.r,
                bubble.column,
                bubble.row,
                0,
            };
            const float s = bubble_score(working_image, roi, *grading_params);
            const size_t selection_idx =
                metadata_offsets[static_cast<size_t>(field_idx)] + static_cast<size_t>(bubble.column);
            if (s > metadata_best_val[selection_idx]) {
                metadata_second_val[selection_idx] = metadata_best_val[selection_idx];
                metadata_best_val[selection_idx] = s;
                metadata_best_row[selection_idx] = bubble.row;
            } else if (s > metadata_second_val[selection_idx]) {
                metadata_second_val[selection_idx] = s;
            }
        }

        for (int32_t i = 0; i < form->n_metadata_fields; ++i) {
            const OMR_MetadataField& field = form->metadata_fields[i];
            const size_t field_offset = metadata_offsets[static_cast<size_t>(i)];
            for (int32_t col = 0; col < field.n_columns; ++col) {
                const size_t selection_idx = field_offset + static_cast<size_t>(col);
                if (metadata_best_row[selection_idx] >= 0 &&
                    metadata_best_val[selection_idx] >= grading_params->abs_th &&
                    (metadata_best_val[selection_idx] - metadata_second_val[selection_idx]) >= grading_params->rel_th) {
                    metadata_selected_rows[selection_idx] = metadata_best_row[selection_idx];
                }
            }
        }
    }

    int32_t score = 0;
    int32_t graded_questions = 0;
    for (int32_t q = 0; q < form->n_questions; ++q) {
        const int32_t gt = form->answer_key[q];
        if (gt < 0) {
            continue;
        }
        ++graded_questions;
        if (answers[q] == gt) {
            ++score;
        }
    }

    out_result->answers = answers;
    out_result->n_answers = form->n_questions;
    out_result->metadata_selected_rows = metadata_selected_rows;
    out_result->n_metadata_selected_rows = static_cast<int32_t>(metadata_selection_count);
    out_result->score = score;
    out_result->total_questions = form->n_questions;
    out_result->graded_questions = graded_questions;
    out_result->used_abs_th = grading_params->abs_th;
    out_result->used_rel_th = grading_params->rel_th;

    if (runtime_options->return_scored_image != 0) {
        size_t bytes = 0;
        if (!checked_mul_size_t(
                static_cast<size_t>(working_image.stride),
                static_cast<size_t>(working_image.height),
                &bytes)) {
            std::free(out_result->answers);
            out_result->answers = nullptr;
            out_result->n_answers = 0;
            if (out_result->metadata_selected_rows != nullptr) {
                std::free(out_result->metadata_selected_rows);
                out_result->metadata_selected_rows = nullptr;
                out_result->n_metadata_selected_rows = 0;
            }
            return set_error(out_result, OMR_ERR_BAD_IMAGE, "image size overflow");
        }

        out_result->scored_image_data = static_cast<uint8_t*>(std::malloc(bytes));
        if (out_result->scored_image_data == nullptr) {
            std::free(out_result->answers);
            out_result->answers = nullptr;
            out_result->n_answers = 0;
            if (out_result->metadata_selected_rows != nullptr) {
                std::free(out_result->metadata_selected_rows);
                out_result->metadata_selected_rows = nullptr;
                out_result->n_metadata_selected_rows = 0;
            }
            return set_error(out_result, OMR_ERR_ALLOCATION_FAILED, "failed to allocate scored image");
        }

        std::memcpy(out_result->scored_image_data, working_image.data, bytes);
        out_result->scored_image_width = working_image.width;
        out_result->scored_image_height = working_image.height;
        out_result->scored_image_stride = working_image.stride;
        out_result->scored_image_channels = working_image.channels;
    }

    out_result->err_code = OMR_OK;
    copy_error_message(out_result->error_message, "");
    return OMR_OK;
}

const char* omr_error_code_to_string(int32_t code) {
    switch (code) {
        case OMR_OK:
            return "OMR_OK";
        case OMR_ERR_NULL_ARGUMENT:
            return "OMR_ERR_NULL_ARGUMENT";
        case OMR_ERR_INVALID_HANDLE:
            return "OMR_ERR_INVALID_HANDLE";
        case OMR_ERR_BAD_IMAGE:
            return "OMR_ERR_BAD_IMAGE";
        case OMR_ERR_BAD_CONFIG:
            return "OMR_ERR_BAD_CONFIG";
        case OMR_ERR_INVALID_ROI_LAYOUT:
            return "OMR_ERR_INVALID_ROI_LAYOUT";
        case OMR_ERR_ALLOCATION_FAILED:
            return "OMR_ERR_ALLOCATION_FAILED";
        case OMR_ERR_INSUFFICIENT_MARKERS:
            return "OMR_ERR_INSUFFICIENT_MARKERS";
        case OMR_ERR_WARP_FAILED:
            return "OMR_ERR_WARP_FAILED";
        case OMR_ERR_NOT_IMPLEMENTED:
            return "OMR_ERR_NOT_IMPLEMENTED";
        case OMR_ERR_INTERNAL:
            return "OMR_ERR_INTERNAL";
        default:
            return "OMR_ERR_UNKNOWN";
    }
}
