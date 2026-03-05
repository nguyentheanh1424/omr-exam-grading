#ifndef OMR_WARP_GLOBAL_H
#define OMR_WARP_GLOBAL_H

#include "omr_api.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace omr_warp {

bool compute_global_h_from_markers(
    const OMR_FormSpec& form,
    float ransac_thresh_px,
    int ransac_iterations,
    float out_h[9],
    int32_t* out_inliers
);

bool warp_image_bilinear(
    const OMR_ImageView& src,
    int32_t out_width,
    int32_t out_height,
    const float h_src_to_dst[9],
    std::vector<uint8_t>* out_storage,
    OMR_ImageView* out_view,
    char* err_message,
    size_t err_cap
);

}  // namespace omr_warp

#endif
