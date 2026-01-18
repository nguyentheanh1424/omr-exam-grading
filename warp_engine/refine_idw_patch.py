import numpy as np
import cv2 as cv

def refine_idw_patch(
    patch: np.ndarray,
    src_local: np.ndarray,     # (N,2) toạ độ marker sau H-local, trong hệ patch
    dst_local: np.ndarray,     # (N,2) toạ độ marker template trong hệ patch
    grid_shape=(12, 12),
    idw_power=4.0,
    idw_eps=1e-3,
):
    """
    IDW nội suy trong patch độc lập, dùng residual = src_local - dst_local.
    """
    ph, pw = patch.shape[:2]
    N = src_local.shape[0]

    if N < 1:
        return patch

    residuals = (src_local - dst_local).astype(np.float32)   # (N,2)

    gx, gy = grid_shape
    dxC = np.zeros((gy + 1, gx + 1), np.float32)
    dyC = np.zeros((gy + 1, gx + 1), np.float32)

    dst = dst_local.astype(np.float32)

    # --- IDW coarse grid ---
    for iy in range(gy + 1):
        y = (iy / gy) * (ph - 1)
        for ix in range(gx + 1):
            x = (ix / gx) * (pw - 1)

            diff = dst - np.array([x, y], np.float32)
            dist = np.linalg.norm(diff, axis=1) + idw_eps
            w = 1.0 / (dist ** idw_power)

            w_sum = w.sum()
            if w_sum < 1e-6:
                dx = dy = 0.0
            else:
                wn = (w[:, None] / w_sum)
                delta = (wn * residuals).sum(axis=0)
                dx, dy = float(delta[0]), float(delta[1])

            dxC[iy, ix] = dx
            dyC[iy, ix] = dy

    dx = cv.resize(dxC, (pw, ph), interpolation=cv.INTER_LINEAR)
    dy = cv.resize(dyC, (pw, ph), interpolation=cv.INTER_LINEAR)

    xs, ys = np.meshgrid(
        np.arange(pw, dtype=np.float32),
        np.arange(ph, dtype=np.float32),
    )

    map_x = np.clip(xs + dx, 0, pw - 1).astype(np.float32)
    map_y = np.clip(ys + dy, 0, ph - 1).astype(np.float32)

    refined_patch = cv.remap(
        patch,
        map_x,
        map_y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE,
    )

    return refined_patch
