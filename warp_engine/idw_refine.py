import logging
import os

import cv2 as cv
import numpy as np

from .detector import detect_tags
from .utils import safe_imwrite

log = logging.getLogger(__name__)

# Grid resolution for displacement field
DEFAULT_GRID_WIDTH = 48
DEFAULT_GRID_HEIGHT = 36

# IDW interpolation parameters
DEFAULT_IDW_POWER = 2.5  # Higher = more local influence
IDW_EPSILON = 1e-3  # Small value to prevent division by zero

# Minimum number of detected tags required for refinement
MIN_TAGS_FOR_REFINEMENT = 4

# Remap interpolation method
REMAP_INTERPOLATION = cv.INTER_LINEAR

# Coordinate bounds
MIN_COORDINATE = 0
COORDINATE_OFFSET = 1  # For max bounds calculation

# Debug output filename
DEBUG_OUTPUT_FILENAME = "step3_idw_refined.png"


def idw_refine(
        warped,
        layout,
        grid=(DEFAULT_GRID_WIDTH, DEFAULT_GRID_HEIGHT),
        power=DEFAULT_IDW_POWER,
        eps=IDW_EPSILON,
        output=None,
        debug=False
):
    """
    Refine warped image using Inverse Distance Weighting (IDW) interpolation.

    Args:
        warped: Input warped image
        layout: Dictionary mapping tag IDs to their expected positions
        grid: Tuple (width, height) for displacement field grid resolution
        power: IDW power parameter (higher = more local influence)
        eps: Small epsilon to prevent division by zero
        output: Output directory for debug files
        debug: Whether to save debug images

    Returns:
        Refined image with IDW-based distortion correction
    """
    H_img, W_img = warped.shape[:2]
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    detections = detect_tags(gray)

    # Collect corresponding points
    src, dst = [], []
    for d in detections:
        if d.id in layout:
            src.append(d.center.astype(np.float32))
            dst.append(np.array(layout[d.id], np.float32))

    # Need at least 4 points for meaningful refinement
    if len(src) < MIN_TAGS_FOR_REFINEMENT:
        return warped

    src = np.array(src)
    dst = np.array(dst)
    residuals = src - dst

    # Create displacement field on coarse grid
    gx, gy = grid
    dxC = np.zeros((gy + COORDINATE_OFFSET, gx + COORDINATE_OFFSET), np.float32)
    dyC = np.zeros((gy + COORDINATE_OFFSET, gx + COORDINATE_OFFSET), np.float32)

    # Compute displacement at each grid point using IDW
    for iy in range(gy + COORDINATE_OFFSET):
        y = (iy / gy) * (H_img - COORDINATE_OFFSET)
        for ix in range(gx + COORDINATE_OFFSET):
            x = (ix / gx) * (W_img - COORDINATE_OFFSET)

            # Calculate distances from current point to all reference points
            diff = dst - np.array([x, y], np.float32)
            dist = np.linalg.norm(diff, axis=1) + eps

            # IDW weights: closer points have more influence
            w = 1.0 / (dist ** power)

            # Normalize weights and compute weighted average of residuals
            wn = (w[:, None] / w.sum())
            delta = (wn * residuals).sum(0)

            dxC[iy, ix] = delta[0]
            dyC[iy, ix] = delta[1]

    # Upsample displacement field to image resolution
    dx = cv.resize(dxC, (W_img, H_img))
    dy = cv.resize(dyC, (W_img, H_img))

    # Create pixel coordinate grids
    xs, ys = np.meshgrid(
        np.arange(W_img, dtype=np.float32),
        np.arange(H_img, dtype=np.float32)
    )

    # Apply displacement field with bounds checking
    map_x = np.clip(xs + dx, MIN_COORDINATE, W_img - COORDINATE_OFFSET)
    map_y = np.clip(ys + dy, MIN_COORDINATE, H_img - COORDINATE_OFFSET)

    # Remap image using displacement field
    refined = cv.remap(warped, map_x, map_y, REMAP_INTERPOLATION)

    if debug:
        safe_imwrite(os.path.join(output, DEBUG_OUTPUT_FILENAME), refined)

    return refined