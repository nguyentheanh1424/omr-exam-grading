#ifndef OMR_MARKER_DETECT_H
#define OMR_MARKER_DETECT_H

#include "omr_api.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace omr_marker {

bool detect_markers_v1(
    const OMR_ImageView& image,
    const OMR_MarkerTemplate* template_markers,
    int32_t n_template_markers,
    std::vector<OMR_DetectedMarker>* out_detected,
    char* err_message,
    size_t err_cap
);

}  // namespace omr_marker

#endif
