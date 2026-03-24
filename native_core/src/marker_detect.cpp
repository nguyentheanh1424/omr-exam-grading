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

struct Pt2 {
    float x;
    float y;
};

struct Candidate {
    float cx;
    float cy;
    int min_x;
    int min_y;
    int max_x;
    int max_y;
    int area;
    std::array<Pt2, 4> corners;
};

struct DecodedCandidate {
    OMR_DetectedMarker marker;
    int area;
    int candidate_index;
    int hamming_distance;
    float contrast;
};

constexpr int kAprilTagMarkerSize = 4;
constexpr int kAprilTagBorderSize = 1;
constexpr int kAprilTagGridSize = kAprilTagMarkerSize + 2 * kAprilTagBorderSize;
constexpr int kCellSamplesPerAxis = 5;
constexpr int kMaxDecodeHamming = 2;
constexpr int kMaxAllowedWhiteBorderCells = 2;
constexpr float kMinDecodeContrast = 40.0f;
constexpr float kTemplateNearestDistancePx = 200.0f;

constexpr std::array<uint16_t, 30> kAprilTag16h5Codes = {
    0xD8C4u, 0xA574u, 0x562Cu, 0x9DA2u, 0x659Eu, 0xD6FEu,
    0x1ACDu, 0xA2E7u, 0x9A7Fu, 0xB6A8u, 0xD01Cu, 0xD50Fu,
    0x21B0u, 0x6CE2u, 0x4E31u, 0x08F5u, 0x3C90u, 0x2DC9u,
    0xC0A5u, 0xF162u, 0xEC87u, 0xA9EAu, 0x42FBu, 0xB838u,
    0x3B97u, 0xB5CEu, 0xFAB5u, 0x0CABu, 0x53E0u, 0x74F5u,
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

bool estimate_homography_4pt(
    const std::array<Pt2, 4>& src,
    const std::array<Pt2, 4>& dst,
    float out_h[9]
) {
    if (out_h == nullptr) {
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

    for (size_t i = 0; i < src.size(); ++i) {
        const double x = static_cast<double>(src[i].x);
        const double y = static_cast<double>(src[i].y);
        const double u = static_cast<double>(dst[i].x);
        const double v = static_cast<double>(dst[i].y);

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

Pt2 apply_h(const float h[9], const Pt2& p) {
    const double x = static_cast<double>(p.x);
    const double y = static_cast<double>(p.y);
    const double w = static_cast<double>(h[6]) * x +
                     static_cast<double>(h[7]) * y +
                     static_cast<double>(h[8]);
    if (std::abs(w) < 1e-12) {
        return Pt2{
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()
        };
    }
    return Pt2{
        static_cast<float>((static_cast<double>(h[0]) * x +
                            static_cast<double>(h[1]) * y +
                            static_cast<double>(h[2])) / w),
        static_cast<float>((static_cast<double>(h[3]) * x +
                            static_cast<double>(h[4]) * y +
                            static_cast<double>(h[5])) / w),
    };
}

float sample_bilinear_gray(const OMR_ImageView& image, float x, float y) {
    if (x < 0.0f || y < 0.0f ||
        x > static_cast<float>(image.width - 1) ||
        y > static_cast<float>(image.height - 1)) {
        return 255.0f;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float wx = x - static_cast<float>(x0);
    const float wy = y - static_cast<float>(y0);

    const float v00 = static_cast<float>(gray_at(image, x0, y0));
    const float v01 = static_cast<float>(gray_at(image, x1, y0));
    const float v10 = static_cast<float>(gray_at(image, x0, y1));
    const float v11 = static_cast<float>(gray_at(image, x1, y1));

    const float top = v00 * (1.0f - wx) + v01 * wx;
    const float bot = v10 * (1.0f - wx) + v11 * wx;
    return top * (1.0f - wy) + bot * wy;
}

std::array<Pt2, 4> estimate_component_corners(
    const std::vector<std::array<int, 2>>& points,
    int bw,
    int bh
) {
    std::array<Pt2, 4> corners = {Pt2{0.0f, 0.0f}, Pt2{0.0f, 0.0f}, Pt2{0.0f, 0.0f}, Pt2{0.0f, 0.0f}};
    if (points.empty()) {
        return corners;
    }

    float min_sum = std::numeric_limits<float>::max();
    float max_sum = -std::numeric_limits<float>::max();
    float min_diff = std::numeric_limits<float>::max();
    float max_diff = -std::numeric_limits<float>::max();

    Pt2 tl_fallback{static_cast<float>(points[0][0]), static_cast<float>(points[0][1])};
    Pt2 tr_fallback = tl_fallback;
    Pt2 br_fallback = tl_fallback;
    Pt2 bl_fallback = tl_fallback;

    for (const auto& p : points) {
        const float x = static_cast<float>(p[0]);
        const float y = static_cast<float>(p[1]);
        const float sum = x + y;
        const float diff = x - y;
        if (sum < min_sum) {
            min_sum = sum;
            tl_fallback = Pt2{x, y};
        }
        if (sum > max_sum) {
            max_sum = sum;
            br_fallback = Pt2{x, y};
        }
        if (diff > max_diff) {
            max_diff = diff;
            tr_fallback = Pt2{x, y};
        }
        if (diff < min_diff) {
            min_diff = diff;
            bl_fallback = Pt2{x, y};
        }
    }

    const float tol = std::max(2.0f, 0.08f * static_cast<float>(std::max(bw, bh)));

    auto average_near_extreme = [&](float target, bool use_sum, bool pick_high, const Pt2& fallback) -> Pt2 {
        float sx = 0.0f;
        float sy = 0.0f;
        int count = 0;
        for (const auto& p : points) {
            const float x = static_cast<float>(p[0]);
            const float y = static_cast<float>(p[1]);
            const float value = use_sum ? (x + y) : (x - y);
            const bool keep = pick_high ? (value >= target - tol) : (value <= target + tol);
            if (!keep) {
                continue;
            }
            sx += x;
            sy += y;
            ++count;
        }
        if (count <= 0) {
            return fallback;
        }
        return Pt2{sx / static_cast<float>(count), sy / static_cast<float>(count)};
    };

    corners[0] = average_near_extreme(min_sum, true, false, tl_fallback);
    corners[1] = average_near_extreme(max_diff, false, true, tr_fallback);
    corners[2] = average_near_extreme(max_sum, true, true, br_fallback);
    corners[3] = average_near_extreme(min_diff, false, false, bl_fallback);
    return corners;
}

bool corners_are_valid(const std::array<Pt2, 4>& corners) {
    auto edge_len2 = [](const Pt2& a, const Pt2& b) -> float {
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        return dx * dx + dy * dy;
    };

    const float e0 = edge_len2(corners[0], corners[1]);
    const float e1 = edge_len2(corners[1], corners[2]);
    const float e2 = edge_len2(corners[2], corners[3]);
    const float e3 = edge_len2(corners[3], corners[0]);
    if (std::min(std::min(e0, e1), std::min(e2, e3)) < 16.0f) {
        return false;
    }

    float twice_area = 0.0f;
    for (size_t i = 0; i < corners.size(); ++i) {
        const Pt2& a = corners[i];
        const Pt2& b = corners[(i + 1) % corners.size()];
        twice_area += a.x * b.y - b.x * a.y;
    }
    return std::abs(twice_area) > 100.0f;
}

Pt2 center_of_corners(const std::array<Pt2, 4>& corners) {
    Pt2 center{0.0f, 0.0f};
    for (const Pt2& corner : corners) {
        center.x += corner.x;
        center.y += corner.y;
    }
    center.x *= 0.25f;
    center.y *= 0.25f;
    return center;
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
    std::vector<std::array<int, 2>> points;
    points.reserve(8192);
    std::vector<Candidate> out;

    for (int y0 = 0; y0 < h; ++y0) {
        for (int x0 = 0; x0 < w; ++x0) {
            const size_t start_idx = static_cast<size_t>(y0) * static_cast<size_t>(w) + static_cast<size_t>(x0);
            if (mask[start_idx] == 0u || visited[start_idx] != 0u) {
                continue;
            }

            visited[start_idx] = 1u;
            queue.clear();
            points.clear();
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
                points.push_back({x, y});

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
            c.corners = estimate_component_corners(points, bw, bh);
            if (!corners_are_valid(c.corners)) {
                continue;
            }
            const Pt2 center = center_of_corners(c.corners);
            c.cx = center.x;
            c.cy = center.y;
            out.push_back(c);
        }
    }

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

uint16_t bit_mask_at(int y, int x) {
    const int idx = y * kAprilTagMarkerSize + x;
    return static_cast<uint16_t>(1u << (15 - idx));
}

bool get_white_bit(uint16_t code, int y, int x) {
    return (code & bit_mask_at(y, x)) != 0;
}

uint16_t set_white_bit(uint16_t code, int y, int x, bool value) {
    const uint16_t mask = bit_mask_at(y, x);
    return value ? static_cast<uint16_t>(code | mask) : static_cast<uint16_t>(code & ~mask);
}

uint16_t rotate_code_cw(uint16_t code) {
    uint16_t rotated = 0u;
    for (int y = 0; y < kAprilTagMarkerSize; ++y) {
        for (int x = 0; x < kAprilTagMarkerSize; ++x) {
            rotated = set_white_bit(rotated, x, kAprilTagMarkerSize - 1 - y, get_white_bit(code, y, x));
        }
    }
    return rotated;
}

int popcount16(uint16_t v) {
    int count = 0;
    while (v != 0u) {
        v = static_cast<uint16_t>(v & static_cast<uint16_t>(v - 1u));
        ++count;
    }
    return count;
}

bool sample_candidate_cells(
    const OMR_ImageView& image,
    const Candidate& candidate,
    std::array<float, kAprilTagGridSize * kAprilTagGridSize>* out_cell_means,
    float* out_contrast,
    int* out_white_border_cells
) {
    if (out_cell_means == nullptr || out_contrast == nullptr || out_white_border_cells == nullptr) {
        return false;
    }

    const std::array<Pt2, 4> dst = {
        Pt2{0.0f, 0.0f},
        Pt2{static_cast<float>(kAprilTagGridSize), 0.0f},
        Pt2{static_cast<float>(kAprilTagGridSize), static_cast<float>(kAprilTagGridSize)},
        Pt2{0.0f, static_cast<float>(kAprilTagGridSize)}
    };

    float h_src_to_dst[9] = {0.0f};
    if (!estimate_homography_4pt(candidate.corners, dst, h_src_to_dst)) {
        return false;
    }

    float h_dst_to_src[9] = {0.0f};
    if (!invert_3x3(h_src_to_dst, h_dst_to_src)) {
        return false;
    }

    float min_mean = 255.0f;
    float max_mean = 0.0f;

    for (int cell_y = 0; cell_y < kAprilTagGridSize; ++cell_y) {
        for (int cell_x = 0; cell_x < kAprilTagGridSize; ++cell_x) {
            float sum = 0.0f;
            int samples = 0;
            for (int sy = 0; sy < kCellSamplesPerAxis; ++sy) {
                for (int sx = 0; sx < kCellSamplesPerAxis; ++sx) {
                    const float fx = static_cast<float>(cell_x) +
                                     0.20f +
                                     0.60f * (static_cast<float>(sx) + 0.5f) /
                                     static_cast<float>(kCellSamplesPerAxis);
                    const float fy = static_cast<float>(cell_y) +
                                     0.20f +
                                     0.60f * (static_cast<float>(sy) + 0.5f) /
                                     static_cast<float>(kCellSamplesPerAxis);
                    const Pt2 src = apply_h(h_dst_to_src, Pt2{fx, fy});
                    if (!std::isfinite(src.x) || !std::isfinite(src.y)) {
                        continue;
                    }
                    sum += sample_bilinear_gray(image, src.x, src.y);
                    ++samples;
                }
            }

            if (samples <= 0) {
                return false;
            }

            const float mean = sum / static_cast<float>(samples);
            (*out_cell_means)[static_cast<size_t>(cell_y) * kAprilTagGridSize + static_cast<size_t>(cell_x)] = mean;
            min_mean = std::min(min_mean, mean);
            max_mean = std::max(max_mean, mean);
        }
    }

    const float threshold = 0.5f * (min_mean + max_mean);
    int white_border_cells = 0;
    for (int y = 0; y < kAprilTagGridSize; ++y) {
        for (int x = 0; x < kAprilTagGridSize; ++x) {
            if (x != 0 && x != (kAprilTagGridSize - 1) &&
                y != 0 && y != (kAprilTagGridSize - 1)) {
                continue;
            }
            if ((*out_cell_means)[static_cast<size_t>(y) * kAprilTagGridSize + static_cast<size_t>(x)] > threshold) {
                ++white_border_cells;
            }
        }
    }

    *out_contrast = max_mean - min_mean;
    *out_white_border_cells = white_border_cells;
    return true;
}

bool decode_candidate_apriltag16h5(
    const OMR_ImageView& image,
    const Candidate& candidate,
    int* out_id,
    int* out_hamming,
    float* out_contrast
) {
    if (out_id == nullptr || out_hamming == nullptr || out_contrast == nullptr) {
        return false;
    }

    std::array<float, kAprilTagGridSize * kAprilTagGridSize> cell_means{};
    float contrast = 0.0f;
    int white_border_cells = 0;
    if (!sample_candidate_cells(image, candidate, &cell_means, &contrast, &white_border_cells)) {
        return false;
    }
    if (contrast < kMinDecodeContrast || white_border_cells > kMaxAllowedWhiteBorderCells) {
        return false;
    }

    float min_mean = 255.0f;
    float max_mean = 0.0f;
    for (float mean : cell_means) {
        min_mean = std::min(min_mean, mean);
        max_mean = std::max(max_mean, mean);
    }
    const float threshold = 0.5f * (min_mean + max_mean);

    uint16_t sampled_code = 0u;
    for (int y = 0; y < kAprilTagMarkerSize; ++y) {
        for (int x = 0; x < kAprilTagMarkerSize; ++x) {
            const float mean = cell_means[
                static_cast<size_t>(y + kAprilTagBorderSize) * kAprilTagGridSize +
                static_cast<size_t>(x + kAprilTagBorderSize)
            ];
            sampled_code = set_white_bit(sampled_code, y, x, mean > threshold);
        }
    }

    int best_id = -1;
    int best_hamming = 32;
    int ties = 0;
    uint16_t rotated = sampled_code;
    for (int rot = 0; rot < 4; ++rot) {
        for (size_t id = 0; id < kAprilTag16h5Codes.size(); ++id) {
            const int hamming = popcount16(static_cast<uint16_t>(rotated ^ kAprilTag16h5Codes[id]));
            if (hamming < best_hamming) {
                best_hamming = hamming;
                best_id = static_cast<int>(id);
                ties = 1;
            } else if (hamming == best_hamming) {
                ++ties;
            }
        }
        rotated = rotate_code_cw(rotated);
    }

    if (best_id < 0 || best_hamming > kMaxDecodeHamming || ties != 1) {
        return false;
    }

    *out_id = best_id;
    *out_hamming = best_hamming;
    *out_contrast = contrast;
    return true;
}

void sort_unique_detected(std::vector<OMR_DetectedMarker>* detected) {
    if (detected == nullptr) {
        return;
    }
    std::sort(detected->begin(), detected->end(), [](const OMR_DetectedMarker& a, const OMR_DetectedMarker& b) {
        return a.id < b.id;
    });
    detected->erase(std::unique(detected->begin(), detected->end(), [](const OMR_DetectedMarker& a, const OMR_DetectedMarker& b) {
        return a.id == b.id;
    }), detected->end());
}

void add_or_update_decoded(
    std::vector<DecodedCandidate>* decoded,
    const DecodedCandidate& incoming
) {
    if (decoded == nullptr) {
        return;
    }
    for (DecodedCandidate& existing : *decoded) {
        if (existing.marker.id != incoming.marker.id) {
            continue;
        }
        const bool better =
            (incoming.hamming_distance < existing.hamming_distance) ||
            (incoming.hamming_distance == existing.hamming_distance && incoming.contrast > existing.contrast) ||
            (incoming.hamming_distance == existing.hamming_distance &&
             incoming.contrast == existing.contrast &&
             incoming.area > existing.area);
        if (better) {
            existing = incoming;
        }
        return;
    }
    decoded->push_back(incoming);
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

bool fill_missing_by_template_nearest(
    const std::vector<Candidate>& candidates,
    const std::vector<uint8_t>& skip_candidates,
    const OMR_MarkerTemplate* template_markers,
    int32_t n_template_markers,
    std::vector<OMR_DetectedMarker>* io_detected
) {
    if (io_detected == nullptr || io_detected->size() < 4) {
        return false;
    }

    OMR_FormSpec tmp_form{};
    tmp_form.template_markers = template_markers;
    tmp_form.n_template_markers = n_template_markers;
    tmp_form.detected_markers = io_detected->data();
    tmp_form.n_detected_markers = static_cast<int32_t>(io_detected->size());

    float h_src_to_dst[9] = {0.0f};
    int32_t inliers = 0;
    if (!omr_warp::compute_global_h_from_markers(tmp_form, 8.0f, 100, h_src_to_dst, &inliers)) {
        return false;
    }

    std::unordered_set<int32_t> used_ids;
    for (const OMR_DetectedMarker& d : *io_detected) {
        used_ids.insert(d.id);
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i < skip_candidates.size() && skip_candidates[i] != 0u) {
            continue;
        }

        const Pt2 warped = apply_h(h_src_to_dst, Pt2{candidates[i].cx, candidates[i].cy});
        if (!std::isfinite(warped.x) || !std::isfinite(warped.y)) {
            continue;
        }

        const int id = nearest_template_id(
            warped.x,
            warped.y,
            template_markers,
            n_template_markers,
            used_ids,
            kTemplateNearestDistancePx
        );
        if (id < 0) {
            continue;
        }

        used_ids.insert(id);
        io_detected->push_back(OMR_DetectedMarker{
            id,
            candidates[i].cx,
            candidates[i].cy
        });
    }

    sort_unique_detected(io_detected);
    return io_detected->size() >= 4;
}

bool detect_markers_geometry_fallback(
    const std::vector<Candidate>& candidates,
    const OMR_MarkerTemplate* template_markers,
    int32_t n_template_markers,
    std::vector<OMR_DetectedMarker>* out_detected,
    char* err_message,
    size_t err_cap
) {
    auto by_min_sum = std::min_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return (a.cx + a.cy) < (b.cx + b.cy); });
    auto by_max_sum = std::max_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return (a.cx + a.cy) < (b.cx + b.cy); });
    auto by_min_x_minus_y = std::min_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return (a.cx - a.cy) < (b.cx - b.cy); });
    auto by_max_x_minus_y = std::max_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return (a.cx - a.cy) < (b.cx - b.cy); });

    std::array<Candidate, 4> corners = {*by_min_sum, *by_min_x_minus_y, *by_max_x_minus_y, *by_max_sum};

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

    std::unordered_set<int32_t> used_ids;
    for (const OMR_DetectedMarker& d : detected) {
        used_ids.insert(d.id);
    }

    for (const Candidate& c : candidates) {
        const Pt2 warped = apply_h(h_src_to_dst, Pt2{c.cx, c.cy});
        if (!std::isfinite(warped.x) || !std::isfinite(warped.y)) {
            continue;
        }

        const int id = nearest_template_id(
            warped.x,
            warped.y,
            template_markers,
            n_template_markers,
            used_ids,
            kTemplateNearestDistancePx
        );
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

    sort_unique_detected(&detected);
    if (detected.size() < 4) {
        copy_error(err_message, err_cap, "marker detect produced fewer than 4 labeled markers");
        return false;
    }

    *out_detected = std::move(detected);
    copy_error(err_message, err_cap, "");
    return true;
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

    std::vector<DecodedCandidate> decoded;
    std::vector<uint8_t> decoded_candidate_mask(candidates.size(), 0u);
    for (size_t i = 0; i < candidates.size(); ++i) {
        int id = -1;
        int hamming = 0;
        float contrast = 0.0f;
        if (!decode_candidate_apriltag16h5(image, candidates[i], &id, &hamming, &contrast)) {
            continue;
        }
        decoded_candidate_mask[i] = 1u;
        add_or_update_decoded(&decoded, DecodedCandidate{
            OMR_DetectedMarker{id, candidates[i].cx, candidates[i].cy},
            candidates[i].area,
            static_cast<int>(i),
            hamming,
            contrast
        });
    }

    if (decoded.size() >= 4) {
        std::vector<OMR_DetectedMarker> detected;
        detected.reserve(decoded.size());
        for (const DecodedCandidate& d : decoded) {
            detected.push_back(d.marker);
        }
        sort_unique_detected(&detected);
        if (detected.size() >= 4) {
            (void)fill_missing_by_template_nearest(
                candidates,
                decoded_candidate_mask,
                template_markers,
                n_template_markers,
                &detected
            );
            *out_detected = std::move(detected);
            copy_error(err_message, err_cap, "");
            return true;
        }
    }

    return detect_markers_geometry_fallback(
        candidates,
        template_markers,
        n_template_markers,
        out_detected,
        err_message,
        err_cap
    );
}

}  // namespace omr_marker
