from __future__ import annotations

import ctypes
from ctypes import POINTER, Structure, byref, c_char, c_float, c_int32, c_uint8, c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2 as cv
import numpy as np

from native_core.python_adapter import NativeAdapterConfig
from postprocess_engine.bubble_field_reader import BubbleFieldConfig, build_bubble_field_cells


OMR_OK = 0
OMR_ERROR_MESSAGE_CAPACITY = 256


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
        ("error_message", c_char * OMR_ERROR_MESSAGE_CAPACITY),
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


@dataclass
class NativeRunOutput:
    answers: list[int]
    score: int
    total_questions: int
    graded_questions: int
    used_abs_th: float
    used_rel_th: float
    scored_image: np.ndarray | None
    bubble_field_selected_rows: dict[str, list[int | None]] | None = None
    bubble_field_values: dict[str, str] | None = None


class NativeCoreClient:
    def __init__(self, dll_path: str | Path | None = None):
        self._root = Path(__file__).resolve().parents[1]
        self._dll_path = Path(dll_path) if dll_path is not None else self._root / "build" / "native_core" / "omr_core.dll"
        if not self._dll_path.exists():
            raise FileNotFoundError(f"Missing DLL: {self._dll_path}")
        self.lib = ctypes.CDLL(str(self._dll_path))
        self._bind_functions()

    def _bind_functions(self) -> None:
        self.lib.omr_create.restype = c_void_p
        self.lib.omr_destroy.argtypes = [c_void_p]
        self.lib.omr_init_result.argtypes = [POINTER(OMR_Result)]
        self.lib.omr_free_result.argtypes = [POINTER(OMR_Result)]
        self.lib.omr_init_default_warp_params.argtypes = [POINTER(OMR_WarpParams)]
        self.lib.omr_init_default_binarize_params.argtypes = [POINTER(OMR_BinarizeParams)]
        self.lib.omr_init_default_grading_params.argtypes = [POINTER(OMR_GradingParams)]
        self.lib.omr_init_default_runtime_options.argtypes = [POINTER(OMR_RuntimeOptions)]
        self.lib.omr_process.argtypes = [
            c_void_p,
            POINTER(OMR_ImageView),
            POINTER(OMR_FormSpec),
            POINTER(OMR_WarpParams),
            POINTER(OMR_BinarizeParams),
            POINTER(OMR_GradingParams),
            POINTER(OMR_RuntimeOptions),
            POINTER(OMR_Result),
        ]
        self.lib.omr_process.restype = c_int32

    @staticmethod
    def _as_contiguous_image(img: np.ndarray) -> np.ndarray:
        if img.ndim not in (2, 3):
            raise ValueError("input image must be grayscale or BGR")
        if img.dtype != np.uint8:
            raise ValueError("input image must use uint8 pixels")
        return np.ascontiguousarray(img)

    @staticmethod
    def _build_form_arrays(
        config: NativeAdapterConfig,
        bubble_field_configs: Sequence[BubbleFieldConfig] | None = None,
    ):
        marker_arr = (OMR_MarkerTemplate * len(config.template_markers))()
        for idx, marker in enumerate(config.template_markers):
            marker_arr[idx] = OMR_MarkerTemplate(id=marker.marker_id, x=marker.x, y=marker.y)

        window_arr = (OMR_RegionWindow * len(config.region_windows))()
        for idx, window in enumerate(config.region_windows):
            marker_ids = (c_int32 * 4)(*window.marker_ids)
            window_arr[idx] = OMR_RegionWindow(marker_ids=marker_ids, n_marker_ids=4)

        roi_arr = (OMR_CircleROI * len(config.circle_rois))()
        for idx, roi in enumerate(config.circle_rois):
            roi_arr[idx] = OMR_CircleROI(
                cx=roi.cx,
                cy=roi.cy,
                r=roi.r,
                question=roi.question,
                option=roi.option,
            )

        answer_key_arr = (c_int32 * len(config.answer_key))(*config.answer_key)
        metadata_field_arr = None
        metadata_bubble_arr = None
        metadata_field_map: dict[str, int] = {}
        if bubble_field_configs:
            metadata_field_arr = (OMR_MetadataField * len(bubble_field_configs))()
            for idx, field in enumerate(bubble_field_configs):
                metadata_field_map[field.id] = idx
                metadata_field_arr[idx] = OMR_MetadataField(
                    field_id=idx,
                    n_columns=field.n_cols,
                    n_rows=field.n_rows,
                )

            cells = build_bubble_field_cells(bubble_field_configs)
            metadata_bubble_arr = (OMR_MetadataBubble * len(cells))()
            for idx, cell in enumerate(cells):
                metadata_bubble_arr[idx] = OMR_MetadataBubble(
                    field_id=metadata_field_map[cell.field_id],
                    column=cell.column,
                    row=cell.row,
                    cx=cell.cx,
                    cy=cell.cy,
                    r=cell.radius,
                )

        return marker_arr, window_arr, roi_arr, answer_key_arr, metadata_field_arr, metadata_bubble_arr

    @staticmethod
    def _build_detected_marker_array(
        detected_markers: Sequence[tuple[int, float, float]] | None,
    ) -> tuple[object | None, int]:
        if not detected_markers:
            return None, 0
        marker_arr = (OMR_DetectedMarker * len(detected_markers))()
        for idx, marker in enumerate(detected_markers):
            marker_id, x, y = marker
            marker_arr[idx] = OMR_DetectedMarker(id=int(marker_id), x=float(x), y=float(y))
        return marker_arr, len(detected_markers)

    @staticmethod
    def _decode_error_message(result: OMR_Result) -> str:
        return bytes(result.error_message).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

    @staticmethod
    def _copy_scored_image(result: OMR_Result) -> np.ndarray | None:
        if not bool(result.scored_image_data):
            return None
        if result.scored_image_width <= 0 or result.scored_image_height <= 0:
            return None

        total_bytes = result.scored_image_stride * result.scored_image_height
        buffer = ctypes.string_at(result.scored_image_data, total_bytes)
        array = np.frombuffer(buffer, dtype=np.uint8).copy()
        image = array.reshape(result.scored_image_height, result.scored_image_stride)
        image = image[:, : result.scored_image_width * result.scored_image_channels]
        if result.scored_image_channels == 1:
            return image.reshape(result.scored_image_height, result.scored_image_width)
        return image.reshape(
            result.scored_image_height,
            result.scored_image_width,
            result.scored_image_channels,
        )

    def run(
        self,
        img: np.ndarray,
        config: NativeAdapterConfig,
        *,
        assume_aligned_input: bool,
        return_scored_image: bool = True,
        use_global_idw: bool = False,
        use_region_refine: bool = True,
        debug_level: int = 0,
        abs_th: float | None = None,
        rel_th: float | None = None,
        auto_threshold: bool | None = None,
        detected_markers: Sequence[tuple[int, float, float]] | None = None,
        bubble_field_configs: Sequence[BubbleFieldConfig] | None = None,
    ) -> NativeRunOutput:
        img = self._as_contiguous_image(img)
        height, width = img.shape[:2]
        channels = 1 if img.ndim == 2 else img.shape[2]
        stride = int(img.strides[0])

        image = OMR_ImageView(
            width=width,
            height=height,
            stride=stride,
            channels=channels,
            data=img.ctypes.data_as(POINTER(c_uint8)),
        )

        (
            marker_arr,
            window_arr,
            roi_arr,
            answer_key_arr,
            metadata_field_arr,
            metadata_bubble_arr,
        ) = self._build_form_arrays(config, bubble_field_configs)
        detected_marker_arr, detected_marker_count = self._build_detected_marker_array(detected_markers)
        form = OMR_FormSpec(
            output_width=config.output_width,
            output_height=config.output_height,
            template_markers=marker_arr,
            n_template_markers=len(config.template_markers),
            detected_markers=detected_marker_arr,
            n_detected_markers=detected_marker_count,
            region_windows=window_arr,
            n_region_windows=len(config.region_windows),
            circle_rois=roi_arr,
            n_circle_rois=len(config.circle_rois),
            metadata_fields=metadata_field_arr,
            n_metadata_fields=0 if metadata_field_arr is None else len(bubble_field_configs),
            metadata_bubbles=metadata_bubble_arr,
            n_metadata_bubbles=0 if metadata_bubble_arr is None else len(metadata_bubble_arr),
            n_questions=config.n_questions,
            n_options_per_question=config.n_options_per_question,
            answer_key=answer_key_arr,
            n_answer_key=len(config.answer_key),
        )

        warp_params = OMR_WarpParams()
        bin_params = OMR_BinarizeParams()
        grading_params = OMR_GradingParams()
        runtime_options = OMR_RuntimeOptions()
        result = OMR_Result()

        self.lib.omr_init_default_warp_params(byref(warp_params))
        self.lib.omr_init_default_binarize_params(byref(bin_params))
        self.lib.omr_init_default_grading_params(byref(grading_params))
        self.lib.omr_init_default_runtime_options(byref(runtime_options))
        self.lib.omr_init_result(byref(result))

        warp_params.use_global_idw = 1 if use_global_idw else 0
        warp_params.use_region_refine = 1 if use_region_refine else 0
        runtime_options.assume_aligned_input = 1 if assume_aligned_input else 0
        runtime_options.return_scored_image = 1 if return_scored_image else 0
        runtime_options.debug_level = debug_level
        if abs_th is not None:
            grading_params.abs_th = float(abs_th)
        if rel_th is not None:
            grading_params.rel_th = float(rel_th)
        if auto_threshold is not None:
            grading_params.auto_threshold = 1 if auto_threshold else 0

        handle = self.lib.omr_create()
        if not handle:
            raise RuntimeError("omr_create failed")

        try:
            rc = self.lib.omr_process(
                handle,
                byref(image),
                byref(form),
                byref(warp_params),
                byref(bin_params),
                byref(grading_params),
                byref(runtime_options),
                byref(result),
            )
            if rc != OMR_OK:
                raise RuntimeError(
                    f"omr_process failed rc={rc} msg={self._decode_error_message(result)}"
                )

            answers = [result.answers[i] for i in range(result.n_answers)]
            scored_image = self._copy_scored_image(result)
            bubble_field_selected_rows = None
            bubble_field_values = None
            if bubble_field_configs:
                bubble_field_selected_rows = {}
                bubble_field_values = {}
                offset = 0
                for field in bubble_field_configs:
                    selected_rows: list[int | None] = []
                    selected_values: list[str | None] = []
                    for _ in range(field.n_cols):
                        raw_row = int(result.metadata_selected_rows[offset])
                        offset += 1
                        if raw_row < 0:
                            selected_rows.append(None)
                            selected_values.append(None)
                        else:
                            selected_rows.append(raw_row)
                            selected_values.append(field.row_values[raw_row])
                    bubble_field_selected_rows[field.id] = selected_rows
                    bubble_field_values[field.id] = "".join(
                        value if value is not None else "?" for value in selected_values
                    )
            return NativeRunOutput(
                answers=answers,
                score=int(result.score),
                total_questions=int(result.total_questions),
                graded_questions=int(result.graded_questions),
                used_abs_th=float(result.used_abs_th),
                used_rel_th=float(result.used_rel_th),
                scored_image=scored_image,
                bubble_field_selected_rows=bubble_field_selected_rows,
                bubble_field_values=bubble_field_values,
            )
        finally:
            self.lib.omr_free_result(byref(result))
            self.lib.omr_destroy(handle)


def read_image(path: str | Path) -> np.ndarray:
    img = cv.imread(str(path), cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img
