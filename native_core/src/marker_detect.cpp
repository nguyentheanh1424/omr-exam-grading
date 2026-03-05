#include "marker_detect.h"
#include "warp_global.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <unordered_set>
#include <vector>

namespace omr_marker {

namespace {

struct Candidate {
    float cx;
    float cy;
    int min_x;
    int min_y;
    int max_x;
    int max_y;
    int area;
};

inline void copy_error(char* dst, size_t cap, const char* msg) {
    if (dst == nullptr || cap == 0) {
        return;
    }
    std::snprintf(dst, cap, "%s", (msg == nullptr) ? "" : msg);
}

inline uint8_t gray_at(const OMR_ImageView& image, int x, int y) {
    const uint8_t* row = image.data + static_cast<size_t>(y) * static_cast<size_t>(image.stride);
    if (image.channels == 1) {
        return row[x];
    }
    const size_t idx = static_cast<size_t>(x) * static_cast<size_t>(image.channels);
    const float b = static_cast<float>(row[idx + 0]);
    const float g = static_cast<float>(row[idx + 1]);
    const float r = static_cast<float>(row[idx + 2]);
    const float gray = 0.114f * b + 0.587f * g + 0.299f * r;
    return static_cast<uint8_t>(std::clamp(gray, 0.0f, 255.0f));
}

int otsu_threshold(const OMR_ImageView& image) {
    int hist[256] = {0};
    const int total = image.width * image.height;
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            hist[gray_at(image, x, y)] += 1;
        }
    }

    double sum_all = 0.0;
    for (int i = 0; i < 256; ++i) {
        sum_all += static_cast<double>(i) * static_cast<double>(hist[i]);
    }

    double sum_b = 0.0;
    int w_b = 0;
    int w_f = 0;
    double max_var = -1.0;
    int best_t = 100;

    for (int t = 0; t < 256; ++t) {
        w_b += hist[t];
        if (w_b == 0) {
            continue;
        }
        w_f = total - w_b;
        if (w_f == 0) {
            break;
        }
        sum_b += static_cast<double>(t) * static_cast<double>(hist[t]);
        const double m_b = sum_b / static_cast<double>(w_b);
        const double m_f = (sum_all - sum_b) / static_cast<double>(w_f);
        const double var = static_cast<double>(w_b) * static_cast<double>(w_f) * (m_b - m_f) * (m_b - m_f);
        if (var > max_var) {
            max_var = var;
            best_t = t;
        }
    }
    return best_t;
}

std::vector<Candidate> find_candidates(const OMR_ImageView& image) {
    const int w = image.width;
    const int h = image.height;
    const int threshold = std::min(120, otsu_threshold(image));

    std::vector<uint8_t> mask(static_cast<size_t>(w) * static_cast<size_t>(h), 0u);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            mask[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] =
                (gray_at(image, x, y) < threshold) ? 1u : 0u;
        }
    }

    std::vector<uint8_t> visited(mask.size(), 0u);
    std::vector<int> queue;
    queue.reserve(8192);
    std::vector<Candidate> out;

    for (int y0 = 0; y0 < h; ++y0) {
        for (int x0 = 0; x0 < w; ++x0) {
            const size_t start_idx = static_cast<size_t>(y0) * static_cast<size_t>(w) + static_cast<size_t>(x0);
            if (mask[start_idx] == 0u || visited[start_idx] != 0u) {
                continue;
            }

            visited[start_idx] = 1u;
            queue.clear();
            queue.push_back(static_cast<int>(start_idx));

            int area = 0;
            long long sum_x = 0;
            long long sum_y = 0;
            int min_x = x0, max_x = x0;
            int min_y = y0, max_y = y0;

            for (size_t qi = 0; qi < queue.size(); ++qi) {
                const int idx = queue[qi];
                const int y = idx / w;
                const int x = idx - y * w;

                area += 1;
                sum_x += x;
                sum_y += y;
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);

                const int nx[4] = {x - 1, x + 1, x, x};
                const int ny[4] = {y, y, y - 1, y + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nx[k] < 0 || nx[k] >= w || ny[k] < 0 || ny[k] >= h) {
                        continue;
                    }
                    const size_t nidx = static_cast<size_t>(ny[k]) * static_cast<size_t>(w) + static_cast<size_t>(nx[k]);
                    if (mask[nidx] == 0u || visited[nidx] != 0u) {
                        continue;
                    }
                    visited[nidx] = 1u;
                    queue.push_back(static_cast<int>(nidx));
                }
            }

            const int bw = max_x - min_x + 1;
            const int bh = max_y - min_y + 1;
            const int bbox_area = bw * bh;
            if (area < 250 || bbox_area <= 0) {
                continue;
            }
            if (bw < 18 || bh < 18) {
                continue;
            }
            const float aspect = static_cast<float>(bw) / static_cast<float>(bh);
            if (aspect < 0.6f || aspect > 1.4f) {
                continue;
            }
            const float fill_ratio = static_cast<float>(area) / static_cast<float>(bbox_area);
            if (fill_ratio < 0.08f || fill_ratio > 0.85f) {
                continue;
            }

            Candidate c{};
            c.cx = static_cast<float>(sum_x) / static_cast<float>(area);
            c.cy = static_cast<float>(sum_y) / static_cast<float>(area);
            c.min_x = min_x;
            c.min_y = min_y;
            c.max_x = max_x;
            c.max_y = max_y;
            c.area = area;
            out.push_back(c);
        }
    }

    // Deduplicate nearby candidates.
    std::vector<Candidate> dedup;
    for (const Candidate& c : out) {
        bool merged = false;
        for (Candidate& d : dedup) {
            const float dx = c.cx - d.cx;
            const float dy = c.cy - d.cy;
            if ((dx * dx + dy * dy) < 16.0f * 16.0f) {
                if (c.area > d.area) {
                    d = c;
                }
                merged = true;
                break;
            }
        }
        if (!merged) {
            dedup.push_back(c);
        }
    }
    return dedup;
}

int nearest_template_id(
    float x,
    float y,
    const OMR_MarkerTemplate* templates,
    int32_t n_templates,
    const std::unordered_set<int32_t>& used,
    float max_dist
) {
    int best_id = -1;
    float best_d2 = max_dist * max_dist;
    for (int32_t i = 0; i < n_templates; ++i) {
        if (used.find(templates[i].id) != used.end()) {
            continue;
        }
        const float dx = x - templates[i].x;
        const float dy = y - templates[i].y;
        const float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best_id = templates[i].id;
        }
    }
    return best_id;
}

}  // namespace

bool detect_markers_v1(
    const OMR_ImageView& image,
    const OMR_MarkerTemplate* template_markers,
    int32_t n_template_markers,
    std::vector<OMR_DetectedMarker>* out_detected,
    char* err_message,
    size_t err_cap
) {
    if (template_markers == nullptr || n_template_markers < 4 || out_detected == nullptr) {
        copy_error(err_message, err_cap, "marker detect requires template markers and output buffer");
        return false;
    }

    const std::vector<Candidate> candidates = find_candidates(image);
    if (candidates.size() < 4) {
        copy_error(err_message, err_cap, "marker detect found fewer than 4 candidates");
        return false;
    }

    // Find corner-like candidates.
    auto by_min_sum = std::min_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return (a.cx + a.cy) < (b.cx + b.cy); });
    auto by_max_sum = std::max_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return (a.cx + a.cy) < (b.cx + b.cy); });
    auto by_min_x_minus_y = std::min_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return (a.cx - a.cy) < (b.cx - b.cy); });
    auto by_max_x_minus_y = std::max_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return (a.cx - a.cy) < (b.cx - b.cy); });

    std::array<Candidate, 4> corners = {*by_min_sum, *by_min_x_minus_y, *by_max_x_minus_y, *by_max_sum};

    // Corner IDs from template geometry.
    auto t_min_sum = std::min_element(template_markers, template_markers + n_template_markers,
        [](const OMR_MarkerTemplate& a, const OMR_MarkerTemplate& b) { return (a.x + a.y) < (b.x + b.y); });
    auto t_max_sum = std::max_element(template_markers, template_markers + n_template_markers,
        [](const OMR_MarkerTemplate& a, const OMR_MarkerTemplate& b) { return (a.x + a.y) < (b.x + b.y); });
    auto t_min_x_minus_y = std::min_element(template_markers, template_markers + n_template_markers,
        [](const OMR_MarkerTemplate& a, const OMR_MarkerTemplate& b) { return (a.x - a.y) < (b.x - b.y); });
    auto t_max_x_minus_y = std::max_element(template_markers, template_markers + n_template_markers,
        [](const OMR_MarkerTemplate& a, const OMR_MarkerTemplate& b) { return (a.x - a.y) < (b.x - b.y); });

    std::array<OMR_MarkerTemplate, 4> t_corners = {*t_min_sum, *t_min_x_minus_y, *t_max_x_minus_y, *t_max_sum};

    std::vector<OMR_DetectedMarker> detected;
    detected.reserve(static_cast<size_t>(candidates.size()));
    for (int i = 0; i < 4; ++i) {
        detected.push_back(OMR_DetectedMarker{
            t_corners[static_cast<size_t>(i)].id,
            corners[static_cast<size_t>(i)].cx,
            corners[static_cast<size_t>(i)].cy
        });
    }

    // Build temporary form for initial global H.
    OMR_FormSpec tmp_form{};
    tmp_form.template_markers = template_markers;
    tmp_form.n_template_markers = n_template_markers;
    tmp_form.detected_markers = detected.data();
    tmp_form.n_detected_markers = static_cast<int32_t>(detected.size());

    float h_src_to_dst[9] = {0.0f};
    int32_t inliers = 0;
    if (!omr_warp::compute_global_h_from_markers(tmp_form, 8.0f, 100, h_src_to_dst, &inliers)) {
        copy_error(err_message, err_cap, "initial corner-based homography failed");
        return false;
    }

    // Assign IDs for all candidates by nearest template in warped space.
    std::unordered_set<int32_t> used_ids;
    for (const OMR_DetectedMarker& d : detected) {
        used_ids.insert(d.id);
    }

    auto apply_h = [&](float x, float y, float* u, float* v) {
        const double w = static_cast<double>(h_src_to_dst[6]) * x +
                         static_cast<double>(h_src_to_dst[7]) * y +
                         static_cast<double>(h_src_to_dst[8]);
        if (std::abs(w) < 1e-12) {
            *u = std::numeric_limits<float>::quiet_NaN();
            *v = std::numeric_limits<float>::quiet_NaN();
            return;
        }
        *u = static_cast<float>((static_cast<double>(h_src_to_dst[0]) * x +
                                 static_cast<double>(h_src_to_dst[1]) * y +
                                 static_cast<double>(h_src_to_dst[2])) / w);
        *v = static_cast<float>((static_cast<double>(h_src_to_dst[3]) * x +
                                 static_cast<double>(h_src_to_dst[4]) * y +
                                 static_cast<double>(h_src_to_dst[5])) / w);
    };

    for (const Candidate& c : candidates) {
        float ux = 0.0f;
        float uy = 0.0f;
        apply_h(c.cx, c.cy, &ux, &uy);
        if (!std::isfinite(ux) || !std::isfinite(uy)) {
            continue;
        }

        const int id = nearest_template_id(ux, uy, template_markers, n_template_markers, used_ids, 200.0f);
        if (id < 0) {
            continue;
        }
        used_ids.insert(id);
        detected.push_back(OMR_DetectedMarker{
            id,
            c.cx,
            c.cy
        });
    }

    // Keep unique IDs and sorted order.
    std::sort(detected.begin(), detected.end(), [](const OMR_DetectedMarker& a, const OMR_DetectedMarker& b) {
        return a.id < b.id;
    });
    detected.erase(std::unique(detected.begin(), detected.end(), [](const OMR_DetectedMarker& a, const OMR_DetectedMarker& b) {
        return a.id == b.id;
    }), detected.end());

    if (detected.size() < 4) {
        copy_error(err_message, err_cap, "marker detect produced fewer than 4 labeled markers");
        return false;
    }

    *out_detected = std::move(detected);
    copy_error(err_message, err_cap, "");
    return true;
}

}  // namespace omr_marker
