import logging
import os

import cv2 as cv
import numpy as np

from .detector import detect_tags
from .utils import safe_imwrite

log = logging.getLogger(__name__)


def idw_refine(warped, layout, grid=(48, 36), power=2.0, eps=1e-3, output=None, debug=False):
    H_img, W_img = warped.shape[:2]
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    detections = detect_tags(gray)

    src, dst = [], []
    for d in detections:
        if d.id in layout:
            src.append(d.center.astype(np.float32))
            dst.append(np.array(layout[d.id], np.float32))

    if len(src) < 4:
        return warped

    src = np.array(src)
    dst = np.array(dst)
    residuals = src - dst

    gx, gy = grid
    dxC = np.zeros((gy + 1, gx + 1), np.float32)
    dyC = np.zeros((gy + 1, gx + 1), np.float32)

    for iy in range(gy + 1):
        y = (iy / gy) * (H_img - 1)
        for ix in range(gx + 1):
            x = (ix / gx) * (W_img - 1)

            diff = dst - np.array([x, y], np.float32)
            dist = np.linalg.norm(diff, axis=1) + eps
            w = 1.0 / (dist ** power)

            wn = (w[:, None] / w.sum())
            delta = (wn * residuals).sum(0)

            dxC[iy, ix] = delta[0]
            dyC[iy, ix] = delta[1]

    dx = cv.resize(dxC, (W_img, H_img))
    dy = cv.resize(dyC, (W_img, H_img))

    xs, ys = np.meshgrid(np.arange(W_img, dtype=np.float32),
                         np.arange(H_img, dtype=np.float32))

    map_x = np.clip(xs + dx, 0, W_img - 1)
    map_y = np.clip(ys + dy, 0, H_img - 1)

    refined = cv.remap(warped, map_x, map_y, cv.INTER_LINEAR)

    if debug:
        safe_imwrite(os.path.join(output, "step3_idw_refined.png"), refined)

    return refined
