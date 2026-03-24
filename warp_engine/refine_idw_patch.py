import numpy as np
import cv2 as cv

# Default grid resolution for displacement field
DEFAULT_PATCH_GRID_WIDTH = 12
DEFAULT_PATCH_GRID_HEIGHT = 12

# IDW interpolation parameters
DEFAULT_PATCH_IDW_POWER = 4.0  # Higher power = more localized influence
PATCH_IDW_EPSILON = 1e-3  # Small value to prevent division by zero

# Weight sum threshold for validity check
MIN_WEIGHT_SUM_THRESHOLD = 1e-6

# Minimum markers required for refinement
MIN_MARKERS_FOR_PATCH_REFINEMENT = 1

# Coordinate bounds
MIN_PATCH_COORDINATE = 0
COORDINATE_OFFSET = 1

# OpenCV interpolation and border modes
RESIZE_INTERPOLATION = cv.INTER_LINEAR
REMAP_INTERPOLATION = cv.INTER_LINEAR
REMAP_BORDER_MODE = cv.BORDER_REPLICATE

# Numpy array axis
RESIDUAL_AXIS = 0  # Axis for summing weighted residuals


def refine_idw_patch(
        patch: np.ndarray,
        src_local: np.ndarray,  # (N,2) marker coordinates after H-local, in patch space
        dst_local: np.ndarray,  # (N,2) marker template coordinates in patch space
        grid_shape=(DEFAULT_PATCH_GRID_WIDTH, DEFAULT_PATCH_GRID_HEIGHT),
        idw_power=DEFAULT_PATCH_IDW_POWER,
        idw_eps=PATCH_IDW_EPSILON,
):
    """
    Apply IDW (Inverse Distance Weighting) interpolation for local patch refinement.

    This function computes a displacement field using IDW interpolation based on
    residuals between detected marker positions (src_local) and their expected
    positions (dst_local), then applies the field to refine the patch.

    Args:
        patch: Input image patch to refine
        src_local: (N, 2) array of detected marker coordinates in patch space
        dst_local: (N, 2) array of expected marker coordinates in patch space
        grid_shape: (width, height) tuple for displacement field grid resolution
        idw_power: Power parameter for IDW (higher = more local influence)
        idw_eps: Small epsilon to prevent division by zero in distance calculation

    Returns:
        Refined patch with IDW-based distortion correction
    """
    ph, pw = patch.shape[:2]
    N = src_local.shape[0]

    # Need at least one marker for refinement
    if N < MIN_MARKERS_FOR_PATCH_REFINEMENT:
        return patch

    # Calculate residuals (displacement vectors)
    residuals = (src_local - dst_local).astype(np.float32)  # (N, 2)

    # Initialize displacement field on coarse grid
    gx, gy = grid_shape
    dxC = np.zeros((gy + COORDINATE_OFFSET, gx + COORDINATE_OFFSET), np.float32)
    dyC = np.zeros((gy + COORDINATE_OFFSET, gx + COORDINATE_OFFSET), np.float32)

    dst = dst_local.astype(np.float32)

    # Compute displacement at each grid point using IDW
    for iy in range(gy + COORDINATE_OFFSET):
        y = (iy / gy) * (ph - COORDINATE_OFFSET)
        for ix in range(gx + COORDINATE_OFFSET):
            x = (ix / gx) * (pw - COORDINATE_OFFSET)

            # Calculate distances from current grid point to all markers
            diff = dst - np.array([x, y], np.float32)
            dist = np.linalg.norm(diff, axis=COORDINATE_OFFSET) + idw_eps

            # Compute IDW weights (closer markers have higher weight)
            w = 1.0 / (dist ** idw_power)

            w_sum = w.sum()

            # Check if weights are valid
            if w_sum < MIN_WEIGHT_SUM_THRESHOLD:
                dx = dy = 0.0
            else:
                # Normalize weights and compute weighted average of residuals
                wn = (w[:, None] / w_sum)
                delta = (wn * residuals).sum(axis=RESIDUAL_AXIS)
                dx, dy = float(delta[0]), float(delta[1])

            dxC[iy, ix] = dx
            dyC[iy, ix] = dy

    # Upsample displacement field to patch resolution
    dx = cv.resize(dxC, (pw, ph), interpolation=RESIZE_INTERPOLATION)
    dy = cv.resize(dyC, (pw, ph), interpolation=RESIZE_INTERPOLATION)

    # Create pixel coordinate grids
    xs, ys = np.meshgrid(
        np.arange(pw, dtype=np.float32),
        np.arange(ph, dtype=np.float32),
    )

    # Apply displacement field with bounds checking
    map_x = np.clip(xs + dx, MIN_PATCH_COORDINATE, pw - COORDINATE_OFFSET).astype(np.float32)
    map_y = np.clip(ys + dy, MIN_PATCH_COORDINATE, ph - COORDINATE_OFFSET).astype(np.float32)

    # Remap patch using displacement field
    refined_patch = cv.remap(
        patch,
        map_x,
        map_y,
        interpolation=REMAP_INTERPOLATION,
        borderMode=REMAP_BORDER_MODE,
    )

    return refined_patch