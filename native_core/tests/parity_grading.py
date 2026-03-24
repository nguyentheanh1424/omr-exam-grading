from __future__ import annotations

import ctypes
from ctypes import POINTER, Structure, byref, c_char, c_float, c_int32, c_uint8, c_uint32, c_void_p
from pathlib import Path
from typing import List, Tuple

import numpy as np

from orm_engine.orm import CircleROI, OMRProcessor


OMR_OK = 0


class OMR_ImageView(Structure):
    _fields_ = [
        ("width", c_int32),
        ("height", c_int32),
        ("stride", c_int32),
        ("channels", c_int32),
        ("data", POINTER(c_uint8)),
    ]


class OMR_MarkerTemplate(Structure):
    _fields_ = [("id", c_int32), ("x", c_float), ("y", c_float)]


class OMR_DetectedMarker(Structure):
    _fields_ = [("id", c_int32), ("x", c_float), ("y", c_float)]


class OMR_RegionWindow(Structure):
    _fields_ = [("marker_ids", c_int32 * 4), ("n_marker_ids", c_int32)]


class OMR_CircleROI(Structure):
    _fields_ = [
        ("cx", c_int32),
        ("cy", c_int32),
        ("r", c_int32),
        ("question", c_int32),
        ("option", c_int32),
    ]


class OMR_MetadataField(Structure):
    _fields_ = [
        ("field_id", c_int32),
        ("n_columns", c_int32),
        ("n_rows", c_int32),
    ]


class OMR_MetadataBubble(Structure):
    _fields_ = [
        ("field_id", c_int32),
        ("column", c_int32),
        ("row", c_int32),
        ("cx", c_int32),
        ("cy", c_int32),
        ("r", c_int32),
    ]


class OMR_FormSpec(Structure):
    _fields_ = [
        ("output_width", c_int32),
        ("output_height", c_int32),
        ("template_markers", POINTER(OMR_MarkerTemplate)),
        ("n_template_markers", c_int32),
        ("detected_markers", POINTER(OMR_DetectedMarker)),
        ("n_detected_markers", c_int32),
        ("region_windows", POINTER(OMR_RegionWindow)),
        ("n_region_windows", c_int32),
        ("circle_rois", POINTER(OMR_CircleROI)),
        ("n_circle_rois", c_int32),
        ("metadata_fields", POINTER(OMR_MetadataField)),
        ("n_metadata_fields", c_int32),
        ("metadata_bubbles", POINTER(OMR_MetadataBubble)),
        ("n_metadata_bubbles", c_int32),
        ("n_questions", c_int32),
        ("n_options_per_question", c_int32),
        ("answer_key", POINTER(c_int32)),
        ("n_answer_key", c_int32),
    ]


class OMR_WarpParams(Structure):
    _fields_ = [
        ("apriltag_dict", c_int32),
        ("global_h_ransac_thresh", c_float),
        ("local_h_ransac_thresh", c_float),
        ("region_bbox_margin_px", c_int32),
        ("use_global_idw", c_int32),
        ("use_region_refine", c_int32),
        ("global_idw_grid_w", c_int32),
        ("global_idw_grid_h", c_int32),
        ("global_idw_power", c_float),
        ("global_idw_eps", c_float),
        ("patch_idw_grid_w", c_int32),
        ("patch_idw_grid_h", c_int32),
        ("patch_idw_power", c_float),
        ("patch_idw_eps", c_float),
        ("skip_idw_if_residual_lt_px", c_float),
        ("residual_breakpoints_px", c_float * 3),
        ("residual_factors", c_float * 4),
        ("reserved", c_int32 * 8),
    ]


class OMR_BinarizeParams(Structure):
    _fields_ = [
        ("method", c_int32),
        ("blur_ksize", c_int32),
        ("fill_percentile", c_float),
        ("thin_iterations", c_int32),
        ("denoise_kernel_size", c_int32),
        ("reserved", c_int32 * 8),
    ]


class OMR_GradingParams(Structure):
    _fields_ = [
        ("abs_th", c_float),
        ("rel_th", c_float),
        ("auto_threshold", c_int32),
        ("clahe_clip_limit", c_float),
        ("clahe_tile_w", c_int32),
        ("clahe_tile_h", c_int32),
        ("gaussian_ksize_w", c_int32),
        ("gaussian_ksize_h", c_int32),
        ("gaussian_sigma", c_float),
        ("patch_radius_multiplier", c_float),
        ("fill_inner_ratio", c_float),
        ("fill_outer_ratio", c_float),
        ("bg_inner_ratio", c_float),
        ("bg_outer_ratio", c_float),
        ("min_valid_pixels", c_int32),
        ("min_questions_for_calibration", c_int32),
        ("calibration_percentile", c_float),
        ("abs_th_mad_multiplier", c_float),
        ("rel_th_mad_multiplier", c_float),
        ("abs_th_baseline_offset", c_float),
        ("rel_th_baseline_offset", c_float),
        ("abs_th_min", c_float),
        ("abs_th_max", c_float),
        ("rel_th_min", c_float),
        ("rel_th_max", c_float),
        ("reserved", c_int32 * 8),
    ]


class OMR_RuntimeOptions(Structure):
    _fields_ = [
        ("assume_aligned_input", c_int32),
        ("return_scored_image", c_int32),
        ("return_intermediate_steps", c_int32),
        ("debug_level", c_int32),
        ("reserved", c_int32 * 8),
    ]


class OMR_Result(Structure):
    _fields_ = [
        ("err_code", c_int32),
        ("error_message", c_char * 256),
        ("score", c_int32),
        ("total_questions", c_int32),
        ("graded_questions", c_int32),
        ("n_answers", c_int32),
        ("answers", POINTER(c_int32)),
        ("n_metadata_selected_rows", c_int32),
        ("metadata_selected_rows", POINTER(c_int32)),
        ("used_abs_th", c_float),
        ("used_rel_th", c_float),
        ("scored_image_data", POINTER(c_uint8)),
        ("scored_image_width", c_int32),
        ("scored_image_height", c_int32),
        ("scored_image_stride", c_int32),
        ("scored_image_channels", c_int32),
    ]


def load_library() -> ctypes.CDLL:
    root = Path(__file__).resolve().parents[2]
    dll_path = root / "build" / "native_core" / "omr_core.dll"
    if not dll_path.exists():
        raise FileNotFoundError(f"Missing DLL: {dll_path}")
    lib = ctypes.CDLL(str(dll_path))

    lib.omr_create.restype = c_void_p
    lib.omr_destroy.argtypes = [c_void_p]

    lib.omr_init_result.argtypes = [POINTER(OMR_Result)]
    lib.omr_free_result.argtypes = [POINTER(OMR_Result)]

    lib.omr_init_default_warp_params.argtypes = [POINTER(OMR_WarpParams)]
    lib.omr_init_default_binarize_params.argtypes = [POINTER(OMR_BinarizeParams)]
    lib.omr_init_default_grading_params.argtypes = [POINTER(OMR_GradingParams)]
    lib.omr_init_default_runtime_options.argtypes = [POINTER(OMR_RuntimeOptions)]

    lib.omr_process.argtypes = [
        c_void_p,
        POINTER(OMR_ImageView),
        POINTER(OMR_FormSpec),
        POINTER(OMR_WarpParams),
        POINTER(OMR_BinarizeParams),
        POINTER(OMR_GradingParams),
        POINTER(OMR_RuntimeOptions),
        POINTER(OMR_Result),
    ]
    lib.omr_process.restype = c_int32
    return lib


def draw_annulus(
    gray: np.ndarray,
    cx: int,
    cy: int,
    r_in: int,
    r_out: int,
    value: int
) -> None:
    yy, xx = np.indices(gray.shape)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (d2 >= r_in * r_in) & (d2 <= r_out * r_out)
    gray[mask] = value


def run_cpp(
    lib: ctypes.CDLL,
    gray: np.ndarray,
    rois_cpp: List[Tuple[int, int, int, int, int]],
    answer_key: List[int],
    abs_th: float,
    rel_th: float,
) -> Tuple[List[int], int]:
    h, w = gray.shape
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    image = OMR_ImageView(
        width=w,
        height=h,
        stride=w,
        channels=1,
        data=gray.ctypes.data_as(POINTER(c_uint8)),
    )

    roi_arr = (OMR_CircleROI * len(rois_cpp))()
    for i, (cx, cy, r, q, opt) in enumerate(rois_cpp):
        roi_arr[i] = OMR_CircleROI(cx=cx, cy=cy, r=r, question=q, option=opt)

    key_arr = (c_int32 * len(answer_key))(*answer_key)

    form = OMR_FormSpec()
    form.output_width = w
    form.output_height = h
    form.template_markers = None
    form.n_template_markers = 0
    form.detected_markers = None
    form.n_detected_markers = 0
    form.region_windows = None
    form.n_region_windows = 0
    form.circle_rois = roi_arr
    form.n_circle_rois = len(rois_cpp)
    form.metadata_fields = None
    form.n_metadata_fields = 0
    form.metadata_bubbles = None
    form.n_metadata_bubbles = 0
    form.n_questions = len(answer_key)
    form.n_options_per_question = 2
    form.answer_key = key_arr
    form.n_answer_key = len(answer_key)

    warp = OMR_WarpParams()
    binp = OMR_BinarizeParams()
    grading = OMR_GradingParams()
    runtime = OMR_RuntimeOptions()
    result = OMR_Result()

    lib.omr_init_default_warp_params(byref(warp))
    lib.omr_init_default_binarize_params(byref(binp))
    lib.omr_init_default_grading_params(byref(grading))
    lib.omr_init_default_runtime_options(byref(runtime))
    lib.omr_init_result(byref(result))

    grading.abs_th = abs_th
    grading.rel_th = rel_th
    runtime.assume_aligned_input = 1
    runtime.return_scored_image = 0

    handle = lib.omr_create()
    if not handle:
        raise RuntimeError("omr_create failed")

    try:
        rc = lib.omr_process(
            handle,
            byref(image),
            byref(form),
            byref(warp),
            byref(binp),
            byref(grading),
            byref(runtime),
            byref(result),
        )
        if rc != OMR_OK:
            msg = bytes(result.error_message).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
            raise RuntimeError(f"omr_process failed rc={rc} msg={msg}")
        answers = [result.answers[i] for i in range(result.n_answers)]
        return answers, int(result.score)
    finally:
        lib.omr_free_result(byref(result))
        lib.omr_destroy(handle)


def run_python_ref(
    gray: np.ndarray,
    rois_py: List[CircleROI],
    answer_key: List[int],
    abs_th: float,
    rel_th: float,
) -> Tuple[List[int], int]:
    proc = OMRProcessor(
        circle_rois=rois_py,
        answer_key=answer_key,
        threshold_path="native_core/tests/.tmp_thresholds_do_not_create.json",
        auto_threshold=False,
    )
    proc.abs_th = abs_th
    proc.rel_th = rel_th

    score_cache = {}
    for roi in rois_py:
        score_cache[(roi.question, roi.option)] = proc._bubble_score(gray, roi.cx, roi.cy, roi.r)

    answers = proc._detect_answers(score_cache)
    score, _ = proc._grade(answers)
    return answers, score


def run_scenario(
    lib: ctypes.CDLL,
    name: str,
    gray: np.ndarray,
    rois_cpp: List[Tuple[int, int, int, int, int]],
    rois_py: List[CircleROI],
    answer_key: List[int],
    abs_th: float,
    rel_th: float,
) -> None:
    cpp_answers, cpp_score = run_cpp(lib, gray, rois_cpp, answer_key, abs_th, rel_th)
    py_answers, py_score = run_python_ref(gray, rois_py, answer_key, abs_th, rel_th)

    if cpp_answers != py_answers or cpp_score != py_score:
        raise AssertionError(
            f"[{name}] mismatch: C++ answers={cpp_answers}, score={cpp_score} | "
            f"Python answers={py_answers}, score={py_score}"
        )
    print(f"[PASS] {name}: answers={cpp_answers}, score={cpp_score}")


def main() -> None:
    lib = load_library()

    # Scenario 1: single marked option.
    gray1 = np.full((160, 260), 255, dtype=np.uint8)
    draw_annulus(gray1, 180, 80, 7, 14, 30)
    rois_cpp_1 = [(80, 80, 16, 0, 0), (180, 80, 16, 0, 1)]
    rois_py_1 = [CircleROI(80, 80, 16, 1, 0), CircleROI(180, 80, 16, 1, 1)]
    run_scenario(
        lib,
        "single-marked",
        gray1,
        rois_cpp_1,
        rois_py_1,
        answer_key=[1],
        abs_th=0.12,
        rel_th=0.04,
    )

    # Scenario 2: two options equally marked -> ambiguous.
    gray2 = np.full((160, 260), 255, dtype=np.uint8)
    draw_annulus(gray2, 80, 80, 7, 14, 40)
    draw_annulus(gray2, 180, 80, 7, 14, 40)
    rois_cpp_2 = [(80, 80, 16, 0, 0), (180, 80, 16, 0, 1)]
    rois_py_2 = [CircleROI(80, 80, 16, 1, 0), CircleROI(180, 80, 16, 1, 1)]
    run_scenario(
        lib,
        "ambiguous",
        gray2,
        rois_cpp_2,
        rois_py_2,
        answer_key=[1],
        abs_th=0.12,
        rel_th=0.10,
    )

    print("[DONE] parity checks passed")


if __name__ == "__main__":
    main()
