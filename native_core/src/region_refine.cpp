#include "region_refine.h"
#include "marker_detect.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <unordered_map>
#include <vector>

namespace omr_region {

namespace {

constexpr float kMaxTrustedLocalMarkerDeltaPx = 24.0f;

struct Pt2 {
    float x;
    float y;
};

inline void copy_error(char* dst, size_t cap, const char* msg) {
    if (dst == nullptr || cap == 0) {
        return;
    }
    std::snprintf(dst, cap, "%s", (msg == nullptr) ? "" : msg);
}

inline Pt2 apply_h(const float h[9], const Pt2& p) {
    const double x = static_cast<double>(p.x);
    const double y = static_cast<double>(p.y);
    const double w = static_cast<double>(h[6]) * x + static_cast<double>(h[7]) * y + static_cast<double>(h[8]);
    if (std::abs(w) < 1e-12) {
        return Pt2{0.0f, 0.0f};
    }
    const double u = (static_cast<double>(h[0]) * x + static_cast<double>(h[1]) * y + static_cast<double>(h[2])) / w;
    const double v = (static_cast<double>(h[3]) * x + static_cast<double>(h[4]) * y + static_cast<double>(h[5])) / w;
    return Pt2{static_cast<float>(u), static_cast<float>(v)};
}

bool invert_3x3(const float m[9], float out_inv[9]) {
    const double a = m[0], b = m[1], c = m[2];
    const double d = m[3], e = m[4], f = m[5];
    const double g = m[6], h = m[7], i = m[8];
    const double A = e * i - f * h;
    const double B = -(d * i - f * g);
    const double C = d * h - e * g;
    const double D = -(b * i - c * h);
    const double E = a * i - c * g;
    const double F = -(a * h - b * g);
    const double G = b * f - c * e;
    const double H = -(a * f - c * d);
    const double I = a * e - b * d;
    const double det = a * A + b * B + c * C;
    if (std::abs(det) < 1e-12) {
        return false;
    }
    const double s = 1.0 / det;
    out_inv[0] = static_cast<float>(A * s);
    out_inv[1] = static_cast<float>(D * s);
    out_inv[2] = static_cast<float>(G * s);
    out_inv[3] = static_cast<float>(B * s);
    out_inv[4] = static_cast<float>(E * s);
    out_inv[5] = static_cast<float>(H * s);
    out_inv[6] = static_cast<float>(C * s);
    out_inv[7] = static_cast<float>(F * s);
    out_inv[8] = static_cast<float>(I * s);
    return true;
}

bool solve_linear(
    std::vector<double>* a,
    std::vector<double>* b,
    int n,
    std::vector<double>* x
) {
    if (a == nullptr || b == nullptr || x == nullptr) {
        return false;
    }
    x->assign(static_cast<size_t>(n), 0.0);
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = std::abs((*a)[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(col)]);
        for (int row = col + 1; row < n; ++row) {
            const double v = std::abs((*a)[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)]);
            if (v > best) {
                best = v;
                pivot = row;
            }
        }
        if (best < 1e-12) {
            return false;
        }
        if (pivot != col) {
            for (int j = col; j < n; ++j) {
                std::swap(
                    (*a)[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(j)],
                    (*a)[static_cast<size_t>(pivot) * static_cast<size_t>(n) + static_cast<size_t>(j)]
                );
            }
            std::swap((*b)[static_cast<size_t>(col)], (*b)[static_cast<size_t>(pivot)]);
        }
        const double diag = (*a)[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(col)];
        for (int j = col; j < n; ++j) {
            (*a)[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(j)] /= diag;
        }
        (*b)[static_cast<size_t>(col)] /= diag;
        for (int row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double f = (*a)[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)];
            if (std::abs(f) < 1e-18) {
                continue;
            }
            for (int j = col; j < n; ++j) {
                (*a)[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(j)] -=
                    f * (*a)[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(j)];
            }
            (*b)[static_cast<size_t>(row)] -= f * (*b)[static_cast<size_t>(col)];
        }
    }
    for (int i = 0; i < n; ++i) {
        (*x)[static_cast<size_t>(i)] = (*b)[static_cast<size_t>(i)];
    }
    return true;
}

bool estimate_homography(
    const std::vector<Pt2>& src,
    const std::vector<Pt2>& dst,
    float out_h[9]
) {
    if (out_h == nullptr || src.size() != dst.size() || src.size() < 4) {
        return false;
    }
    std::vector<double> ata(64, 0.0);
    std::vector<double> atb(8, 0.0);

    auto acc = [&](const std::array<double, 8>& r, double rhs) {
        for (int i = 0; i < 8; ++i) {
            atb[static_cast<size_t>(i)] += r[static_cast<size_t>(i)] * rhs;
            for (int j = 0; j < 8; ++j) {
                ata[static_cast<size_t>(i) * 8u + static_cast<size_t>(j)] +=
                    r[static_cast<size_t>(i)] * r[static_cast<size_t>(j)];
            }
        }
    };

    for (size_t i = 0; i < src.size(); ++i) {
        const double x = src[i].x;
        const double y = src[i].y;
        const double u = dst[i].x;
        const double v = dst[i].y;
        const std::array<double, 8> r1 = {x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y};
        const std::array<double, 8> r2 = {0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y};
        acc(r1, u);
        acc(r2, v);
    }

    std::vector<double> x(8, 0.0);
    if (!solve_linear(&ata, &atb, 8, &x)) {
        return false;
    }
    out_h[0] = static_cast<float>(x[0]);
    out_h[1] = static_cast<float>(x[1]);
    out_h[2] = static_cast<float>(x[2]);
    out_h[3] = static_cast<float>(x[3]);
    out_h[4] = static_cast<float>(x[4]);
    out_h[5] = static_cast<float>(x[5]);
    out_h[6] = static_cast<float>(x[6]);
    out_h[7] = static_cast<float>(x[7]);
    out_h[8] = 1.0f;
    return true;
}

inline uint8_t sample_bilinear(const OMR_ImageView& img, float sx, float sy, int c) {
    if (sx < 0.0f || sy < 0.0f || sx > static_cast<float>(img.width - 1) || sy > static_cast<float>(img.height - 1)) {
        return 255u;
    }
    const int x0 = static_cast<int>(std::floor(sx));
    const int y0 = static_cast<int>(std::floor(sy));
    const int x1 = std::min(x0 + 1, img.width - 1);
    const int y1 = std::min(y0 + 1, img.height - 1);
    const float wx = sx - static_cast<float>(x0);
    const float wy = sy - static_cast<float>(y0);

    const uint8_t* row0 = img.data + static_cast<size_t>(y0) * static_cast<size_t>(img.stride);
    const uint8_t* row1 = img.data + static_cast<size_t>(y1) * static_cast<size_t>(img.stride);
    const size_t i00 = static_cast<size_t>(x0) * static_cast<size_t>(img.channels) + static_cast<size_t>(c);
    const size_t i01 = static_cast<size_t>(x1) * static_cast<size_t>(img.channels) + static_cast<size_t>(c);
    const float v00 = static_cast<float>(row0[i00]);
    const float v01 = static_cast<float>(row0[i01]);
    const float v10 = static_cast<float>(row1[i00]);
    const float v11 = static_cast<float>(row1[i01]);
    const float top = v00 * (1.0f - wx) + v01 * wx;
    const float bot = v10 * (1.0f - wx) + v11 * wx;
    const float val = top * (1.0f - wy) + bot * wy;
    return static_cast<uint8_t>(std::round(std::clamp(val, 0.0f, 255.0f)));
}

std::pair<float, float> sample_grid_bilinear(
    const std::vector<float>& dx,
    const std::vector<float>& dy,
    int gw,
    int gh,
    float gx,
    float gy
) {
    const int x0 = std::max(0, std::min(gw, static_cast<int>(std::floor(gx))));
    const int y0 = std::max(0, std::min(gh, static_cast<int>(std::floor(gy))));
    const int x1 = std::max(0, std::min(gw, x0 + 1));
    const int y1 = std::max(0, std::min(gh, y0 + 1));
    const float wx = gx - static_cast<float>(x0);
    const float wy = gy - static_cast<float>(y0);
    auto idx = [gw](int x, int y) -> size_t {
        return static_cast<size_t>(y) * static_cast<size_t>(gw + 1) + static_cast<size_t>(x);
    };
    const float dx00 = dx[idx(x0, y0)];
    const float dx01 = dx[idx(x1, y0)];
    const float dx10 = dx[idx(x0, y1)];
    const float dx11 = dx[idx(x1, y1)];
    const float dy00 = dy[idx(x0, y0)];
    const float dy01 = dy[idx(x1, y0)];
    const float dy10 = dy[idx(x0, y1)];
    const float dy11 = dy[idx(x1, y1)];
    const float dx_top = dx00 * (1.0f - wx) + dx01 * wx;
    const float dx_bot = dx10 * (1.0f - wx) + dx11 * wx;
    const float dy_top = dy00 * (1.0f - wx) + dy01 * wx;
    const float dy_bot = dy10 * (1.0f - wx) + dy11 * wx;
    return {dx_top * (1.0f - wy) + dx_bot * wy, dy_top * (1.0f - wy) + dy_bot * wy};
}

float choose_residual_factor(const OMR_WarpParams& p, float max_residual) {
    const float b0 = p.residual_breakpoints_px[0];
    const float b1 = p.residual_breakpoints_px[1];
    const float b2 = p.residual_breakpoints_px[2];
    if (max_residual > b2) {
        return p.residual_factors[3];
    }
    if (max_residual > b1) {
        return p.residual_factors[2];
    }
    if (max_residual > b0) {
        return p.residual_factors[1];
    }
    return p.residual_factors[0];
}

uint8_t gray_from_patch_pixel(
    const std::vector<uint8_t>& patch,
    int pw,
    int channels,
    int x,
    int y
) {
    const size_t base =
        static_cast<size_t>(y) * static_cast<size_t>(pw) * static_cast<size_t>(channels) +
        static_cast<size_t>(x) * static_cast<size_t>(channels);
    if (channels <= 1) {
        return patch[base];
    }
    const float b = static_cast<float>(patch[base + 0]);
    const float g = static_cast<float>(patch[base + 1]);
    const float r = static_cast<float>(patch[base + 2]);
    const float gray = 0.114f * b + 0.587f * g + 0.299f * r;
    return static_cast<uint8_t>(std::round(std::clamp(gray, 0.0f, 255.0f)));
}

uint8_t percentile_u8(std::vector<uint8_t> values, float percentile) {
    if (values.empty()) {
        return 255u;
    }
    const float clamped = std::clamp(percentile, 0.0f, 100.0f);
    const size_t idx = static_cast<size_t>(
        (clamped / 100.0f) * static_cast<float>(values.size() - 1)
    );
    std::nth_element(values.begin(), values.begin() + static_cast<ptrdiff_t>(idx), values.end());
    return values[idx];
}

void erode_cross_mask(std::vector<uint8_t>* mask, int pw, int ph, int iterations) {
    if (mask == nullptr || iterations <= 0) {
        return;
    }
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<uint8_t> prev = *mask;
        for (int y = 0; y < ph; ++y) {
            for (int x = 0; x < pw; ++x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(pw) + static_cast<size_t>(x);
                if (prev[idx] == 0u) {
                    (*mask)[idx] = 0u;
                    continue;
                }
                const bool keep =
                    x > 0 && x + 1 < pw &&
                    y > 0 && y + 1 < ph &&
                    prev[idx - 1] != 0u &&
                    prev[idx + 1] != 0u &&
                    prev[idx - static_cast<size_t>(pw)] != 0u &&
                    prev[idx + static_cast<size_t>(pw)] != 0u;
                (*mask)[idx] = keep ? 255u : 0u;
            }
        }
    }
}

std::vector<uint8_t> binarize_patch_dual_like_python(
    const std::vector<uint8_t>& patch,
    int pw,
    int ph,
    int channels,
    const OMR_BinarizeParams& bin_params
) {
    std::vector<uint8_t> gray(static_cast<size_t>(pw) * static_cast<size_t>(ph), 255u);
    for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
            gray[static_cast<size_t>(y) * static_cast<size_t>(pw) + static_cast<size_t>(x)] =
                gray_from_patch_pixel(patch, pw, channels, x, y);
        }
    }

    const uint8_t fill_th = percentile_u8(gray, bin_params.fill_percentile);
    std::vector<uint8_t> mask(static_cast<size_t>(pw) * static_cast<size_t>(ph), 0u);
    for (size_t i = 0; i < gray.size(); ++i) {
        mask[i] = (gray[i] < fill_th) ? 255u : 0u;
    }

    erode_cross_mask(&mask, pw, ph, std::max(0, bin_params.thin_iterations));
    return mask;
}

void warp_patch(
    const OMR_ImageView& src,
    int x0,
    int y0,
    int pw,
    int ph,
    const float h_local[9],
    std::vector<uint8_t>* out_patch
) {
    out_patch->assign(static_cast<size_t>(pw) * static_cast<size_t>(ph) * static_cast<size_t>(src.channels), 255u);
    float inv_h[9] = {0.0f};
    if (!invert_3x3(h_local, inv_h)) {
        return;
    }
    for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
            const Pt2 src_local = apply_h(inv_h, Pt2{static_cast<float>(x), static_cast<float>(y)});
            const float sx = std::clamp(static_cast<float>(x0) + src_local.x, 0.0f, static_cast<float>(src.width - 1));
            const float sy = std::clamp(static_cast<float>(y0) + src_local.y, 0.0f, static_cast<float>(src.height - 1));
            for (int c = 0; c < src.channels; ++c) {
                (*out_patch)[static_cast<size_t>(y) * static_cast<size_t>(pw) * static_cast<size_t>(src.channels) +
                             static_cast<size_t>(x) * static_cast<size_t>(src.channels) +
                             static_cast<size_t>(c)] = sample_bilinear(src, sx, sy, c);
            }
        }
    }
}

void refine_patch_idw(
    const std::vector<uint8_t>& patch,
    int pw,
    int ph,
    int channels,
    const std::vector<Pt2>& src_local,
    const std::vector<Pt2>& dst_local,
    int grid_w,
    int grid_h,
    float power,
    float eps,
    std::vector<uint8_t>* out_patch
) {
    if (grid_w <= 0 || grid_h <= 0 || src_local.size() != dst_local.size() || src_local.empty()) {
        *out_patch = patch;
        return;
    }

    std::vector<Pt2> residuals(src_local.size());
    for (size_t i = 0; i < src_local.size(); ++i) {
        residuals[i] = Pt2{src_local[i].x - dst_local[i].x, src_local[i].y - dst_local[i].y};
    }

    std::vector<float> dx_coarse(static_cast<size_t>(grid_w + 1) * static_cast<size_t>(grid_h + 1), 0.0f);
    std::vector<float> dy_coarse(static_cast<size_t>(grid_w + 1) * static_cast<size_t>(grid_h + 1), 0.0f);
    auto idx = [grid_w](int x, int y) -> size_t {
        return static_cast<size_t>(y) * static_cast<size_t>(grid_w + 1) + static_cast<size_t>(x);
    };

    for (int gy = 0; gy <= grid_h; ++gy) {
        const float y = (static_cast<float>(gy) / static_cast<float>(grid_h)) * static_cast<float>(ph - 1);
        for (int gx = 0; gx <= grid_w; ++gx) {
            const float x = (static_cast<float>(gx) / static_cast<float>(grid_w)) * static_cast<float>(pw - 1);
            double sw = 0.0;
            double sdx = 0.0;
            double sdy = 0.0;
            for (size_t k = 0; k < dst_local.size(); ++k) {
                const float dx = dst_local[k].x - x;
                const float dy = dst_local[k].y - y;
                const float dist = std::sqrt(dx * dx + dy * dy) + eps;
                const double w = 1.0 / std::pow(static_cast<double>(dist), static_cast<double>(power));
                sw += w;
                sdx += w * static_cast<double>(residuals[k].x);
                sdy += w * static_cast<double>(residuals[k].y);
            }
            if (sw > 0.0) {
                dx_coarse[idx(gx, gy)] = static_cast<float>(sdx / sw);
                dy_coarse[idx(gx, gy)] = static_cast<float>(sdy / sw);
            }
        }
    }

    out_patch->assign(patch.size(), 255u);
    auto patch_sample = [&](float sx, float sy, int c) -> uint8_t {
        if (sx < 0.0f || sy < 0.0f || sx > static_cast<float>(pw - 1) || sy > static_cast<float>(ph - 1)) {
            return 255u;
        }
        const int x0 = static_cast<int>(std::floor(sx));
        const int y0 = static_cast<int>(std::floor(sy));
        const int x1 = std::min(x0 + 1, pw - 1);
        const int y1 = std::min(y0 + 1, ph - 1);
        const float wx = sx - static_cast<float>(x0);
        const float wy = sy - static_cast<float>(y0);
        const size_t i00 = static_cast<size_t>(y0) * static_cast<size_t>(pw) * static_cast<size_t>(channels) +
                           static_cast<size_t>(x0) * static_cast<size_t>(channels) + static_cast<size_t>(c);
        const size_t i01 = static_cast<size_t>(y0) * static_cast<size_t>(pw) * static_cast<size_t>(channels) +
                           static_cast<size_t>(x1) * static_cast<size_t>(channels) + static_cast<size_t>(c);
        const size_t i10 = static_cast<size_t>(y1) * static_cast<size_t>(pw) * static_cast<size_t>(channels) +
                           static_cast<size_t>(x0) * static_cast<size_t>(channels) + static_cast<size_t>(c);
        const size_t i11 = static_cast<size_t>(y1) * static_cast<size_t>(pw) * static_cast<size_t>(channels) +
                           static_cast<size_t>(x1) * static_cast<size_t>(channels) + static_cast<size_t>(c);
        const float v00 = static_cast<float>(patch[i00]);
        const float v01 = static_cast<float>(patch[i01]);
        const float v10 = static_cast<float>(patch[i10]);
        const float v11 = static_cast<float>(patch[i11]);
        const float top = v00 * (1.0f - wx) + v01 * wx;
        const float bot = v10 * (1.0f - wx) + v11 * wx;
        const float val = top * (1.0f - wy) + bot * wy;
        return static_cast<uint8_t>(std::round(std::clamp(val, 0.0f, 255.0f)));
    };

    for (int y = 0; y < ph; ++y) {
        const float gy = (static_cast<float>(y) / static_cast<float>(ph - 1)) * static_cast<float>(grid_h);
        for (int x = 0; x < pw; ++x) {
            const float gx = (static_cast<float>(x) / static_cast<float>(pw - 1)) * static_cast<float>(grid_w);
            const auto d = sample_grid_bilinear(dx_coarse, dy_coarse, grid_w, grid_h, gx, gy);
            const float sx = std::clamp(static_cast<float>(x) + d.first, 0.0f, static_cast<float>(pw - 1));
            const float sy = std::clamp(static_cast<float>(y) + d.second, 0.0f, static_cast<float>(ph - 1));
            for (int c = 0; c < channels; ++c) {
                (*out_patch)[static_cast<size_t>(y) * static_cast<size_t>(pw) * static_cast<size_t>(channels) +
                             static_cast<size_t>(x) * static_cast<size_t>(channels) +
                             static_cast<size_t>(c)] = patch_sample(sx, sy, c);
            }
        }
    }
}

}  // namespace

bool refine_regions_local(
    const OMR_ImageView& warped,
    const OMR_FormSpec& form,
    const OMR_WarpParams& params,
    const OMR_BinarizeParams& bin_params,
    const float h_src_to_dst[9],
    std::vector<uint8_t>* out_storage,
    OMR_ImageView* out_view,
    char* err_message,
    size_t err_cap
) {
    if (out_storage == nullptr || out_view == nullptr || h_src_to_dst == nullptr) {
        copy_error(err_message, err_cap, "region refine output pointers are null");
        return false;
    }
    if (form.template_markers == nullptr || form.detected_markers == nullptr || form.region_windows == nullptr) {
        copy_error(err_message, err_cap, "region refine requires template/detected markers and windows");
        return false;
    }

    out_storage->assign(
        static_cast<size_t>(warped.stride) * static_cast<size_t>(warped.height),
        255u
    );
    std::copy(
        warped.data,
        warped.data + static_cast<size_t>(warped.stride) * static_cast<size_t>(warped.height),
        out_storage->begin()
    );

    OMR_ImageView work = warped;
    work.data = out_storage->data();

    std::unordered_map<int32_t, Pt2> template_map;
    template_map.reserve(static_cast<size_t>(form.n_template_markers));
    for (int32_t i = 0; i < form.n_template_markers; ++i) {
        template_map[form.template_markers[i].id] = Pt2{form.template_markers[i].x, form.template_markers[i].y};
    }

    std::unordered_map<int32_t, Pt2> src_after_global;
    src_after_global.reserve(static_cast<size_t>(form.n_detected_markers));
    for (int32_t i = 0; i < form.n_detected_markers; ++i) {
        const OMR_DetectedMarker& dm = form.detected_markers[i];
        src_after_global[dm.id] = apply_h(h_src_to_dst, Pt2{dm.x, dm.y});
    }

    std::unordered_map<int32_t, Pt2> src_after_local_detect;
    {
        std::vector<OMR_DetectedMarker> warped_detected;
        char detect_err[OMR_ERROR_MESSAGE_CAPACITY] = {0};
        if (omr_marker::detect_markers_v1(
                warped,
                form.template_markers,
                form.n_template_markers,
                &warped_detected,
                detect_err,
                sizeof(detect_err))) {
            src_after_local_detect.reserve(warped_detected.size());
            for (const OMR_DetectedMarker& dm : warped_detected) {
                src_after_local_detect[dm.id] = Pt2{dm.x, dm.y};
            }
        }
    }

    for (int32_t wi = 0; wi < form.n_region_windows; ++wi) {
        const OMR_RegionWindow& win = form.region_windows[wi];
        std::vector<int32_t> usable_ids;
        usable_ids.reserve(static_cast<size_t>(win.n_marker_ids));
        for (int32_t k = 0; k < win.n_marker_ids; ++k) {
            const int32_t id = win.marker_ids[k];
            if (template_map.find(id) == template_map.end()) {
                continue;
            }
            if (src_after_global.find(id) != src_after_global.end()) {
                usable_ids.push_back(id);
            }
        }
        if (usable_ids.size() < 4) {
            continue;
        }

        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();
        for (int32_t id : usable_ids) {
            const Pt2 p = template_map[id];
            min_x = std::min(min_x, p.x);
            min_y = std::min(min_y, p.y);
            max_x = std::max(max_x, p.x);
            max_y = std::max(max_y, p.y);
        }

        const int margin = std::max(0, params.region_bbox_margin_px);
        const int x0 = std::max(0, static_cast<int>(std::floor(min_x)) - margin);
        const int y0 = std::max(0, static_cast<int>(std::floor(min_y)) - margin);
        const int x1 = std::min(warped.width, static_cast<int>(std::ceil(max_x)) + margin);
        const int y1 = std::min(warped.height, static_cast<int>(std::ceil(max_y)) + margin);
        const int pw = x1 - x0;
        const int ph = y1 - y0;
        if (pw <= 2 || ph <= 2) {
            continue;
        }

        std::vector<Pt2> src_local;
        std::vector<Pt2> dst_local;
        src_local.reserve(usable_ids.size());
        dst_local.reserve(usable_ids.size());
        for (int32_t id : usable_ids) {
            const Pt2 global_s = src_after_global[id];
            const auto local_it = src_after_local_detect.find(id);
            Pt2 s = global_s;
            if (local_it != src_after_local_detect.end()) {
                const float dx = local_it->second.x - global_s.x;
                const float dy = local_it->second.y - global_s.y;
                const float dist = std::sqrt(dx * dx + dy * dy);
                if (dist <= kMaxTrustedLocalMarkerDeltaPx) {
                    s = local_it->second;
                }
            }
            const Pt2 d = template_map[id];
            src_local.push_back(Pt2{s.x - static_cast<float>(x0), s.y - static_cast<float>(y0)});
            dst_local.push_back(Pt2{d.x - static_cast<float>(x0), d.y - static_cast<float>(y0)});
        }

        float h_local[9] = {0.0f};
        if (!estimate_homography(src_local, dst_local, h_local)) {
            continue;
        }

        std::vector<uint8_t> patch_h;
        warp_patch(warped, x0, y0, pw, ph, h_local, &patch_h);

        std::vector<Pt2> src_est_after_h = dst_local;
        float max_res_before = 0.0f;
        for (size_t i = 0; i < src_local.size(); ++i) {
            const float dx = src_local[i].x - dst_local[i].x;
            const float dy = src_local[i].y - dst_local[i].y;
            max_res_before = std::max(max_res_before, std::sqrt(dx * dx + dy * dy));
        }

        const float factor = choose_residual_factor(params, max_res_before);
        float max_res_after = 0.0f;
        for (size_t i = 0; i < src_local.size(); ++i) {
            const float dx = (src_local[i].x - dst_local[i].x) * factor;
            const float dy = (src_local[i].y - dst_local[i].y) * factor;
            src_est_after_h[i] = Pt2{dst_local[i].x + dx, dst_local[i].y + dy};
            max_res_after = std::max(max_res_after, std::sqrt(dx * dx + dy * dy));
        }

        std::vector<uint8_t> patch_refined;
        if (max_res_after < params.skip_idw_if_residual_lt_px) {
            patch_refined = patch_h;
        } else {
            refine_patch_idw(
                patch_h,
                pw,
                ph,
                work.channels,
                src_est_after_h,
                dst_local,
                std::max(1, params.patch_idw_grid_w),
                std::max(1, params.patch_idw_grid_h),
                params.patch_idw_power,
                params.patch_idw_eps,
                &patch_refined
            );
        }

        const std::vector<uint8_t> ink_mask =
            binarize_patch_dual_like_python(patch_refined, pw, ph, work.channels, bin_params);

        for (int y = 0; y < ph; ++y) {
            uint8_t* dst_row = out_storage->data() +
                static_cast<size_t>(y0 + y) * static_cast<size_t>(work.stride) +
                static_cast<size_t>(x0) * static_cast<size_t>(work.channels);
            for (int x = 0; x < pw; ++x) {
                const bool is_ink =
                    ink_mask[static_cast<size_t>(y) * static_cast<size_t>(pw) + static_cast<size_t>(x)] != 0u;
                for (int c = 0; c < work.channels; ++c) {
                    dst_row[static_cast<size_t>(x) * static_cast<size_t>(work.channels) + static_cast<size_t>(c)] =
                        is_ink ? 0u : 255u;
                }
            }
        }
    }

    out_view->width = work.width;
    out_view->height = work.height;
    out_view->stride = work.stride;
    out_view->channels = work.channels;
    out_view->data = out_storage->data();
    copy_error(err_message, err_cap, "");
    return true;
}

}  // namespace omr_region
