import cv2 as cv
import numpy as np

from .config import WINDOWS_4PTS
from .detector import detect_tags
from .refine_idw_patch import refine_idw_patch
from .binarize import binarize_patch_dual


def bbox_from_template(ids, layout, img_shape, margin=100):
    H, W = img_shape[:2]
    xs, ys = [], []

    for mid in ids:
        if mid in layout:
            x, y = layout[mid]
            xs.append(x)
            ys.append(y)

    x_min = max(0, int(min(xs)) - margin)
    y_min = max(0, int(min(ys)) - margin)
    x_max = min(W, int(max(xs)) + margin)
    y_max = min(H, int(max(ys)) + margin)

    return x_min, y_min, x_max, y_max


def refine_regions(template_img, layout, warped_src, windows=WINDOWS_4PTS, output=None, debug=False):
    base = template_img.copy()
    gray = cv.cvtColor(warped_src, cv.COLOR_BGR2GRAY)
    dets = {d.id: d for d in detect_tags(gray)}

    for wi, ids in enumerate(windows):
        ids = [i for i in ids if i in dets and i in layout]
        if len(ids) < 4:
            continue

        x0, y0, x1, y1 = bbox_from_template(ids, layout, base.shape)

        patch = warped_src[y0:y1, x0:x1]
        off = np.array([x0, y0], np.float32)

        src = np.array([dets[i].center - off for i in ids], np.float32)
        dst = np.array([np.array(layout[i], np.float32) - off for i in ids])

        H, _ = cv.findHomography(src, dst, cv.RANSAC, 3.0)
        if H is None:
            continue

        ph, pw = patch.shape[:2]

        patch_H = cv.warpPerspective(
            patch,
            H,
            (pw, ph),
            borderMode=cv.BORDER_REPLICATE,
        )

        src_before_H = src  # (N, 2)

        dst_ideal = dst

        residuals_before = src_before_H - dst_ideal

        max_residual_before = np.linalg.norm(residuals_before, axis=1).max()

        if max_residual_before > 15.0:
            h_correction_factor = 0.30
        elif max_residual_before > 8.0:
            h_correction_factor = 0.20
        elif max_residual_before > 3.0:
            h_correction_factor = 0.15
        else:
            h_correction_factor = 0.25

        residuals_after = residuals_before * h_correction_factor
        src_estimated_after_H = dst_ideal + residuals_after

        dst_local = dst_ideal

        max_residual_before = np.linalg.norm(residuals_before, axis=1).max()
        max_residual_estimated = np.linalg.norm(residuals_after, axis=1).max()

        if debug:
            with open(f"{output}/idw_residuals.txt", "a", encoding="utf-8") as f:
                f.write(f"Region {wi}:\n")
                f.write(f"  Residuals before Local H: max={max_residual_before:.3f}px\n")
                f.write(f"  Estimated after H (x{h_correction_factor}): max={max_residual_estimated:.3f}px\n")
                for i, mid in enumerate(ids):
                    res_before = np.linalg.norm(residuals_before[i])
                    res_after = np.linalg.norm(residuals_after[i])
                    f.write(f"    Marker {mid}: {res_before:.3f}px -> {res_after:.3f}px\n")

        # Skip IDW nếu residuals quá nhỏ
        if max_residual_estimated < 0.5:
            patch_refined = patch_H
            if debug:
                with open(f"{output}/idw_residuals.txt", "a", encoding="utf-8") as f:
                    f.write(f"  -> Skipped IDW (residuals < 0.5px)\n")
        else:
            patch_refined = refine_idw_patch(
                patch_H,
                src_local=src_estimated_after_H,
                dst_local=dst_local,
                grid_shape=(24, 24),
                idw_power=3.0,
            )

        base_patch = base[y0:y1, x0:x1]

        base_patch[:] = 255

        mask_ink = binarize_patch_dual(patch_refined)

        base_patch[mask_ink] = 0

        if debug:
            cv.imwrite(f"{output}/step4_region_{wi:02d}_a_original.png", patch)
            cv.imwrite(f"{output}/step4_region_{wi:02d}_b_H_warped_and_IDW_refined.png", patch_refined)
            cv.imwrite(f"{output}/step4_region_{wi:02d}_c_ink_mask.png", mask_ink.astype(np.uint8) * 255)
            cv.imwrite(f"{output}/step5_merge_region_{wi:02d}.png", base)

    return base