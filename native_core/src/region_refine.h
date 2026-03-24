#ifndef OMR_REGION_REFINE_H
#define OMR_REGION_REFINE_H

#include "omr_api.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace omr_region {

bool refine_regions_local(
    const OMR_ImageView& warped,
    const OMR_FormSpec& form,
    const OMR_WarpParams& params,
    const OMR_BinarizeParams& bin_params,
    const float h_src_to_dst[9],
    int32_t debug_level,
    std::vector<uint8_t>* out_storage,
    OMR_ImageView* out_view,
    char* err_message,
    size_t err_cap
);

}  // namespace omr_region

#endif
