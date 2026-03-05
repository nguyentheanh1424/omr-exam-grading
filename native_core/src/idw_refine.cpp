#include "idw_refine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <utility>
#include <vector>

namespace omr_idw {

namespace {

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

inline uint8_t sample_bilinear(
    const OMR_ImageView& img,
    float sx,
    float sy,
    int c
) {
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

inline std::pair<float, float> sample_grid_bilinear(
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

    return {
        dx_top * (1.0f - wy) + dx_bot * wy,
        dy_top * (1.0f - wy) + dy_bot * wy
    };
}

}  // namespace

bool refine_global_idw(
    const OMR_ImageView& warped,
    const OMR_FormSpec& form,
    const OMR_WarpParams& params,
    const float h_src_to_dst[9],
    std::vector<uint8_t>* out_storage,
    OMR_ImageView* out_view,
    char* err_message,
    size_t err_cap
) {
    if (out_storage == nullptr || out_view == nullptr || h_src_to_dst == nullptr) {
        copy_error(err_message, err_cap, "idw output pointers are null");
        return false;
    }
    if (form.template_markers == nullptr || form.detected_markers == nullptr ||
        form.n_template_markers <= 0 || form.n_detected_markers <= 0) {
        copy_error(err_message, err_cap, "idw requires template and detected markers");
        return false;
    }
    if (params.global_idw_grid_w <= 0 || params.global_idw_grid_h <= 0 ||
        params.global_idw_power <= 0.0f || params.global_idw_eps <= 0.0f) {
        copy_error(err_message, err_cap, "idw parameters invalid");
        return false;
    }

    std::unordered_map<int32_t, Pt2> template_map;
    template_map.reserve(static_cast<size_t>(form.n_template_markers));
    for (int32_t i = 0; i < form.n_template_markers; ++i) {
        template_map[form.template_markers[i].id] = Pt2{form.template_markers[i].x, form.template_markers[i].y};
    }

    std::vector<Pt2> anchors;
    std::vector<Pt2> residuals;
    anchors.reserve(static_cast<size_t>(form.n_detected_markers));
    residuals.reserve(static_cast<size_t>(form.n_detected_markers));

    for (int32_t i = 0; i < form.n_detected_markers; ++i) {
        const OMR_DetectedMarker& dm = form.detected_markers[i];
        const auto it = template_map.find(dm.id);
        if (it == template_map.end()) {
            continue;
        }
        const Pt2 mapped = apply_h(h_src_to_dst, Pt2{dm.x, dm.y});
        const Pt2 dst = it->second;
        anchors.push_back(dst);
        residuals.push_back(Pt2{mapped.x - dst.x, mapped.y - dst.y});
    }

    if (anchors.size() < 4) {
        copy_error(err_message, err_cap, "idw has fewer than 4 marker correspondences");
        return false;
    }

    const int w = warped.width;
    const int h = warped.height;
    const int gw = params.global_idw_grid_w;
    const int gh = params.global_idw_grid_h;
    const float power = params.global_idw_power;
    const float eps = params.global_idw_eps;

    std::vector<float> dx_coarse(static_cast<size_t>(gw + 1) * static_cast<size_t>(gh + 1), 0.0f);
    std::vector<float> dy_coarse(static_cast<size_t>(gw + 1) * static_cast<size_t>(gh + 1), 0.0f);
    auto cidx = [gw](int x, int y) -> size_t {
        return static_cast<size_t>(y) * static_cast<size_t>(gw + 1) + static_cast<size_t>(x);
    };

    for (int iy = 0; iy <= gh; ++iy) {
        const float y = (static_cast<float>(iy) / static_cast<float>(gh)) * static_cast<float>(h - 1);
        for (int ix = 0; ix <= gw; ++ix) {
            const float x = (static_cast<float>(ix) / static_cast<float>(gw)) * static_cast<float>(w - 1);

            double sw = 0.0;
            double sdx = 0.0;
            double sdy = 0.0;
            for (size_t k = 0; k < anchors.size(); ++k) {
                const float dx = anchors[k].x - x;
                const float dy = anchors[k].y - y;
                const float dist = std::sqrt(dx * dx + dy * dy) + eps;
                const double wgt = 1.0 / std::pow(static_cast<double>(dist), static_cast<double>(power));
                sw += wgt;
                sdx += wgt * static_cast<double>(residuals[k].x);
                sdy += wgt * static_cast<double>(residuals[k].y);
            }
            if (sw > 0.0) {
                dx_coarse[cidx(ix, iy)] = static_cast<float>(sdx / sw);
                dy_coarse[cidx(ix, iy)] = static_cast<float>(sdy / sw);
            }
        }
    }

    const int out_stride = w * warped.channels;
    out_storage->assign(static_cast<size_t>(out_stride) * static_cast<size_t>(h), 255u);

    for (int y = 0; y < h; ++y) {
        const float gy = (static_cast<float>(y) / static_cast<float>(h - 1)) * static_cast<float>(gh);
        for (int x = 0; x < w; ++x) {
            const float gx = (static_cast<float>(x) / static_cast<float>(w - 1)) * static_cast<float>(gw);
            const auto d = sample_grid_bilinear(dx_coarse, dy_coarse, gw, gh, gx, gy);

            const float sx = std::clamp(static_cast<float>(x) + d.first, 0.0f, static_cast<float>(w - 1));
            const float sy = std::clamp(static_cast<float>(y) + d.second, 0.0f, static_cast<float>(h - 1));
            for (int c = 0; c < warped.channels; ++c) {
                (*out_storage)[static_cast<size_t>(y) * static_cast<size_t>(out_stride) +
                               static_cast<size_t>(x) * static_cast<size_t>(warped.channels) +
                               static_cast<size_t>(c)] = sample_bilinear(warped, sx, sy, c);
            }
        }
    }

    out_view->width = w;
    out_view->height = h;
    out_view->stride = out_stride;
    out_view->channels = warped.channels;
    out_view->data = out_storage->data();
    copy_error(err_message, err_cap, "");
    return true;
}

}  // namespace omr_idw
