import cv2 as cv
import numpy as np


def binarize_patch_dual(
        patch_bgr,
        blur_ksize=1,
        thin_iterations=1,
        debug_dir=None,
        region_id=None
):
    """
    Binarize patch với tùy chọn làm mỏng nét mực.

    Args:
        patch_bgr: Patch BGR input
        blur_ksize: Kernel size cho Gaussian blur
        thin_iterations: Số lần erode để làm mỏng (0 = không làm mỏng)
        debug_dir: Thư mục lưu debug images
        region_id: ID của region (cho debug)

    Returns:
        mask_ink: Binary mask (bool) của mực
    """
    gray = cv.cvtColor(patch_bgr, cv.COLOR_BGR2GRAY)

    if debug_dir and region_id is not None:
        cv.imwrite(f"{debug_dir}/region_{region_id:02d}_1_gray.png", gray)

    # Blur để giảm noise
    if blur_ksize and blur_ksize > 1:
        gray = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        if debug_dir and region_id is not None:
            cv.imwrite(f"{debug_dir}/region_{region_id:02d}_2_blurred.png", gray)

    flat = gray.reshape(-1)

    fill_th = np.percentile(flat, 8)

    mask_fill = gray < fill_th

    # Erosion nhẹ để loại bỏ noise
    kernel_denoise = np.ones((1, 1), np.uint8)
    mask_fill = cv.erode(
        mask_fill.astype(np.uint8) * 255,
        kernel_denoise,
        iterations=1,
    ) > 0

    if debug_dir and region_id is not None:
        cv.imwrite(f"{debug_dir}/region_{region_id:02d}_3_initial_mask.png",
                   mask_fill.astype(np.uint8) * 255)

    # THINNING: Làm mỏng nét mực
    if thin_iterations > 0:
        kernel_thin = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

        mask_thin = cv.erode(
            mask_fill.astype(np.uint8) * 255,
            kernel_thin,
            iterations=thin_iterations
        ) > 0

        if debug_dir and region_id is not None:
            cv.imwrite(f"{debug_dir}/region_{region_id:02d}_4_thinned.png",
                       mask_thin.astype(np.uint8) * 255)

        mask_ink = mask_thin
    else:
        mask_ink = mask_fill

    if debug_dir and region_id is not None:
        cv.imwrite(f"{debug_dir}/region_{region_id:02d}_5_final_mask.png",
                   mask_ink.astype(np.uint8) * 255)

    return mask_ink


def binarize_patch_skeleton(
        patch_bgr,
        blur_ksize=1,
        debug_dir=None,
        region_id=None
):
    """
    Binarize với Zhang-Suen thinning (skeleton).
    Tạo ra nét mực chỉ còn 1 pixel độ dày.
    """
    gray = cv.cvtColor(patch_bgr, cv.COLOR_BGR2GRAY)

    if blur_ksize and blur_ksize > 1:
        gray = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    flat = gray.reshape(-1)
    fill_th = np.percentile(flat, 8)
    mask_fill = gray < fill_th

    kernel_denoise = np.ones((1, 1), np.uint8)
    mask_fill = cv.erode(
        mask_fill.astype(np.uint8) * 255,
        kernel_denoise,
        iterations=1,
    ) > 0

    mask_inv = (~mask_fill).astype(np.uint8) * 255

    skeleton = cv.ximgproc.thinning(mask_inv, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)

    mask_skeleton = (skeleton == 0)

    if debug_dir and region_id is not None:
        cv.imwrite(f"{debug_dir}/region_{region_id:02d}_skeleton.png",
                   mask_skeleton.astype(np.uint8) * 255)

    return mask_skeleton