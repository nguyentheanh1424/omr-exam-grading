#ifndef OMR_IDW_REFINE_H
#define OMR_IDW_REFINE_H

#include "omr_api.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace omr_idw {

bool refine_global_idw(
    const OMR_ImageView& warped,
    const OMR_FormSpec& form,
    const OMR_WarpParams& params,
    const float h_src_to_dst[9],
    std::vector<uint8_t>* out_storage,
    OMR_ImageView* out_view,
    char* err_message,
    size_t err_cap
);

}  // namespace omr_idw

#endif
