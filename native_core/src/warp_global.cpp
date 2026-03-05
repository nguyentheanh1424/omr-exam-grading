#include "warp_global.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

namespace omr_warp {

namespace {

struct Pt2 {
    float x;
    float y;
};

struct Corr {
    Pt2 src;
    Pt2 dst;
};

bool solve_linear_system(
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
        double best_abs = std::abs((*a)[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(col)]);
        for (int row = col + 1; row < n; ++row) {
            const double v = std::abs((*a)[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)]);
            if (v > best_abs) {
                best_abs = v;
                pivot = row;
            }
        }

        if (best_abs < 1e-12) {
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

bool estimate_homography_from_corrs(
    const std::vector<Corr>& corrs,
    const std::vector<int32_t>* subset_idx,
    float out_h[9]
) {
    if (out_h == nullptr) {
        return false;
    }

    const int n = (subset_idx == nullptr) ?
        static_cast<int>(corrs.size()) :
        static_cast<int>(subset_idx->size());
    if (n < 4) {
        return false;
    }

    std::vector<double> ata(64, 0.0);
    std::vector<double> atb(8, 0.0);

    auto acc_row = [&](const std::array<double, 8>& row, double rhs) {
        for (int i = 0; i < 8; ++i) {
            atb[static_cast<size_t>(i)] += row[static_cast<size_t>(i)] * rhs;
            for (int j = 0; j < 8; ++j) {
                ata[static_cast<size_t>(i) * 8u + static_cast<size_t>(j)] +=
                    row[static_cast<size_t>(i)] * row[static_cast<size_t>(j)];
            }
        }
    };

    for (int k = 0; k < n; ++k) {
        const Corr& c = (subset_idx == nullptr) ?
            corrs[static_cast<size_t>(k)] :
            corrs[static_cast<size_t>((*subset_idx)[static_cast<size_t>(k)])];

        const double x = static_cast<double>(c.src.x);
        const double y = static_cast<double>(c.src.y);
        const double u = static_cast<double>(c.dst.x);
        const double v = static_cast<double>(c.dst.y);

        const std::array<double, 8> r1 = {x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y};
        const std::array<double, 8> r2 = {0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y};
        acc_row(r1, u);
        acc_row(r2, v);
    }

    std::vector<double> x(8, 0.0);
    if (!solve_linear_system(&ata, &atb, 8, &x)) {
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

inline Pt2 apply_h(const float h[9], const Pt2& p) {
    const double x = static_cast<double>(p.x);
    const double y = static_cast<double>(p.y);
    const double w = static_cast<double>(h[6]) * x + static_cast<double>(h[7]) * y + static_cast<double>(h[8]);
    if (std::abs(w) < 1e-12) {
        return Pt2{std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()};
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
    const double inv_det = 1.0 / det;
    out_inv[0] = static_cast<float>(A * inv_det);
    out_inv[1] = static_cast<float>(D * inv_det);
    out_inv[2] = static_cast<float>(G * inv_det);
    out_inv[3] = static_cast<float>(B * inv_det);
    out_inv[4] = static_cast<float>(E * inv_det);
    out_inv[5] = static_cast<float>(H * inv_det);
    out_inv[6] = static_cast<float>(C * inv_det);
    out_inv[7] = static_cast<float>(F * inv_det);
    out_inv[8] = static_cast<float>(I * inv_det);
    return true;
}

void copy_error(char* dst, size_t cap, const char* msg) {
    if (dst == nullptr || cap == 0) {
        return;
    }
    std::snprintf(dst, cap, "%s", (msg == nullptr) ? "" : msg);
}

}  // namespace

bool compute_global_h_from_markers(
    const OMR_FormSpec& form,
    float ransac_thresh_px,
    int ransac_iterations,
    float out_h[9],
    int32_t* out_inliers
) {
    if (out_h == nullptr) {
        return false;
    }
    if (form.template_markers == nullptr || form.detected_markers == nullptr ||
        form.n_template_markers <= 0 || form.n_detected_markers <= 0) {
        return false;
    }

    std::unordered_map<int32_t, Pt2> template_by_id;
    template_by_id.reserve(static_cast<size_t>(form.n_template_markers));
    for (int32_t i = 0; i < form.n_template_markers; ++i) {
        template_by_id[form.template_markers[i].id] = Pt2{
            form.template_markers[i].x,
            form.template_markers[i].y
        };
    }

    std::vector<Corr> corrs;
    corrs.reserve(static_cast<size_t>(form.n_detected_markers));
    for (int32_t i = 0; i < form.n_detected_markers; ++i) {
        const auto it = template_by_id.find(form.detected_markers[i].id);
        if (it == template_by_id.end()) {
            continue;
        }
        Corr c{};
        c.src = Pt2{form.detected_markers[i].x, form.detected_markers[i].y};
        c.dst = it->second;
        corrs.push_back(c);
    }

    if (corrs.size() < 4) {
        return false;
    }

    std::mt19937 rng(1337u);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(corrs.size()) - 1);

    int best_inliers = -1;
    float best_h[9] = {0.0f};
    std::vector<int32_t> best_set;

    const int iterations = std::max(50, ransac_iterations);
    for (int it = 0; it < iterations; ++it) {
        std::array<int32_t, 4> sample = {-1, -1, -1, -1};
        int filled = 0;
        while (filled < 4) {
            const int32_t idx = static_cast<int32_t>(pick(rng));
            bool exists = false;
            for (int i = 0; i < filled; ++i) {
                if (sample[static_cast<size_t>(i)] == idx) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                sample[static_cast<size_t>(filled)] = idx;
                ++filled;
            }
        }

        std::vector<int32_t> subset(sample.begin(), sample.end());
        float h[9] = {0.0f};
        if (!estimate_homography_from_corrs(corrs, &subset, h)) {
            continue;
        }

        std::vector<int32_t> inliers;
        inliers.reserve(corrs.size());
        for (int32_t i = 0; i < static_cast<int32_t>(corrs.size()); ++i) {
            const Pt2 p = apply_h(h, corrs[static_cast<size_t>(i)].src);
            if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
                continue;
            }
            const float dx = p.x - corrs[static_cast<size_t>(i)].dst.x;
            const float dy = p.y - corrs[static_cast<size_t>(i)].dst.y;
            const float err = std::sqrt(dx * dx + dy * dy);
            if (err <= ransac_thresh_px) {
                inliers.push_back(i);
            }
        }

        if (static_cast<int>(inliers.size()) > best_inliers) {
            best_inliers = static_cast<int>(inliers.size());
            std::copy(h, h + 9, best_h);
            best_set.swap(inliers);
        }
    }

    if (best_inliers < 4) {
        return false;
    }

    if (!estimate_homography_from_corrs(corrs, &best_set, out_h)) {
        std::copy(best_h, best_h + 9, out_h);
    }

    if (out_inliers != nullptr) {
        *out_inliers = static_cast<int32_t>(best_inliers);
    }
    return true;
}

bool warp_image_bilinear(
    const OMR_ImageView& src,
    int32_t out_width,
    int32_t out_height,
    const float h_src_to_dst[9],
    std::vector<uint8_t>* out_storage,
    OMR_ImageView* out_view,
    char* err_message,
    size_t err_cap
) {
    if (out_storage == nullptr || out_view == nullptr || h_src_to_dst == nullptr) {
        copy_error(err_message, err_cap, "warp output pointers are null");
        return false;
    }
    if (src.channels != 1 && src.channels != 3) {
        copy_error(err_message, err_cap, "warp supports channels 1 or 3");
        return false;
    }
    if (out_width <= 0 || out_height <= 0) {
        copy_error(err_message, err_cap, "warp output size must be > 0");
        return false;
    }

    float h_inv[9] = {0.0f};
    if (!invert_3x3(h_src_to_dst, h_inv)) {
        copy_error(err_message, err_cap, "homography inverse failed");
        return false;
    }

    const int32_t out_stride = out_width * src.channels;
    const size_t total = static_cast<size_t>(out_stride) * static_cast<size_t>(out_height);
    out_storage->assign(total, 255u);

    auto sample = [&](float sx, float sy, int c) -> uint8_t {
        if (sx < 0.0f || sy < 0.0f || sx > static_cast<float>(src.width - 1) || sy > static_cast<float>(src.height - 1)) {
            return 255u;
        }

        const int x0 = static_cast<int>(std::floor(sx));
        const int y0 = static_cast<int>(std::floor(sy));
        const int x1 = std::min(x0 + 1, src.width - 1);
        const int y1 = std::min(y0 + 1, src.height - 1);
        const float wx = sx - static_cast<float>(x0);
        const float wy = sy - static_cast<float>(y0);

        const uint8_t* row0 = src.data + static_cast<size_t>(y0) * static_cast<size_t>(src.stride);
        const uint8_t* row1 = src.data + static_cast<size_t>(y1) * static_cast<size_t>(src.stride);
        const size_t i00 = static_cast<size_t>(x0) * static_cast<size_t>(src.channels) + static_cast<size_t>(c);
        const size_t i01 = static_cast<size_t>(x1) * static_cast<size_t>(src.channels) + static_cast<size_t>(c);

        const float v00 = static_cast<float>(row0[i00]);
        const float v01 = static_cast<float>(row0[i01]);
        const float v10 = static_cast<float>(row1[i00]);
        const float v11 = static_cast<float>(row1[i01]);

        const float top = v00 * (1.0f - wx) + v01 * wx;
        const float bot = v10 * (1.0f - wx) + v11 * wx;
        const float val = top * (1.0f - wy) + bot * wy;
        const float clamped = std::clamp(val, 0.0f, 255.0f);
        return static_cast<uint8_t>(std::round(clamped));
    };

    for (int32_t y = 0; y < out_height; ++y) {
        for (int32_t x = 0; x < out_width; ++x) {
            const Pt2 src_p = apply_h(h_inv, Pt2{static_cast<float>(x), static_cast<float>(y)});
            for (int c = 0; c < src.channels; ++c) {
                (*out_storage)[static_cast<size_t>(y) * static_cast<size_t>(out_stride) +
                               static_cast<size_t>(x) * static_cast<size_t>(src.channels) +
                               static_cast<size_t>(c)] = sample(src_p.x, src_p.y, c);
            }
        }
    }

    out_view->width = out_width;
    out_view->height = out_height;
    out_view->stride = out_stride;
    out_view->channels = src.channels;
    out_view->data = out_storage->data();
    copy_error(err_message, err_cap, "");
    return true;
}

}  // namespace omr_warp
