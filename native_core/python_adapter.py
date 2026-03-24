from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from orm_engine.orm import CircleROI, load_circle_rois
from warp_engine.config import A4_PX, TEMPLATE_LAYOUT_FILE, load_region_windows


DEFAULT_CIRCLE_ROIS_JSON = "config/circle_rois.json"
DEFAULT_ANSWER_KEY_JSON = "config/answer_key.json"
DEFAULT_THRESHOLDS_JSON = "config/omr_thresholds.json"


@dataclass(frozen=True)
class NativeMarkerTemplate:
    marker_id: int
    x: float
    y: float


@dataclass(frozen=True)
class NativeRegionWindow:
    marker_ids: tuple[int, int, int, int]


@dataclass(frozen=True)
class NativeCircleROI:
    cx: int
    cy: int
    r: int
    question: int
    option: int
    selection_mode: int


@dataclass(frozen=True)
class NativeAdapterConfig:
    output_width: int
    output_height: int
    n_questions: int
    n_options_per_question: int
    template_markers: tuple[NativeMarkerTemplate, ...]
    region_windows: tuple[NativeRegionWindow, ...]
    circle_rois: tuple[NativeCircleROI, ...]
    answer_key: tuple[int, ...]


@dataclass(frozen=True)
class NativeThresholdConfig:
    abs_th: float
    rel_th: float
    auto_threshold: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return _repo_root() / value


def _load_json(path: str | Path):
    resolved = _resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_template_markers(path: str | Path = TEMPLATE_LAYOUT_FILE) -> tuple[NativeMarkerTemplate, ...]:
    data = _load_json(path)
    if not isinstance(data, dict) or not data:
        raise ValueError("template marker layout must be a non-empty object")

    markers: list[NativeMarkerTemplate] = []
    seen_ids: set[int] = set()

    for raw_id, coords in sorted(data.items(), key=lambda item: int(item[0])):
        marker_id = int(raw_id)
        if marker_id in seen_ids:
            raise ValueError(f"duplicate template marker id: {marker_id}")
        if not isinstance(coords, Sequence) or len(coords) != 2:
            raise ValueError(f"template marker {marker_id} must contain exactly 2 coordinates")
        x = float(coords[0])
        y = float(coords[1])
        if x < 0 or y < 0:
            raise ValueError(f"template marker {marker_id} must stay within output bounds")
        markers.append(NativeMarkerTemplate(marker_id=marker_id, x=x, y=y))
        seen_ids.add(marker_id)

    return tuple(markers)


def build_region_windows(
    windows_4pts: Iterable[Sequence[int]] | None = None,
    template_markers: Sequence[NativeMarkerTemplate] | None = None,
) -> tuple[NativeRegionWindow, ...]:
    if windows_4pts is None:
        windows_4pts = load_region_windows()
    marker_ids = {marker.marker_id for marker in template_markers or ()}
    windows: list[NativeRegionWindow] = []

    for idx, marker_group in enumerate(windows_4pts):
        if len(marker_group) != 4:
            raise ValueError(f"region window #{idx} must contain exactly 4 marker ids")

        values = tuple(int(marker_id) for marker_id in marker_group)
        if len(set(values)) != 4:
            raise ValueError(f"region window #{idx} contains duplicate marker ids")
        if marker_ids and any(marker_id not in marker_ids for marker_id in values):
            raise ValueError(f"region window #{idx} references marker ids outside template layout")

        windows.append(NativeRegionWindow(marker_ids=values))

    return tuple(windows)


def load_answer_key(
    path: str | Path = DEFAULT_ANSWER_KEY_JSON,
    *,
    fallback_question_count: int | None = None,
) -> list[int]:
    resolved = _resolve_repo_path(path)
    if resolved.exists():
        data = _load_json(resolved)
        if not isinstance(data, list):
            raise ValueError("answer key must be stored as a JSON list")
        return [int(value) for value in data]

    if fallback_question_count is None:
        raise ValueError(f"answer key file is missing: {resolved}")
    return [0] * fallback_question_count


def load_threshold_config(
    path: str | Path = DEFAULT_THRESHOLDS_JSON,
    *,
    auto_threshold_when_missing: bool = True,
) -> NativeThresholdConfig:
    resolved = _resolve_repo_path(path)
    if not resolved.exists():
        return NativeThresholdConfig(
            abs_th=0.12,
            rel_th=0.04,
            auto_threshold=auto_threshold_when_missing,
        )

    data = _load_json(resolved)
    abs_th = float(data["abs_th"])
    rel_th = float(data["rel_th"])
    return NativeThresholdConfig(
        abs_th=abs_th,
        rel_th=rel_th,
        auto_threshold=False,
    )


def normalize_circle_rois(circle_rois: Sequence[CircleROI]) -> tuple[tuple[NativeCircleROI, ...], int, int]:
    if not circle_rois:
        raise ValueError("circle ROI list must not be empty")

    seen_pairs: set[tuple[int, int]] = set()
    by_question: dict[int, set[int]] = {}
    normalized: list[NativeCircleROI] = []
    source_questions: set[int] = set()

    for roi in circle_rois:
        if roi.question <= 0:
            raise ValueError("Python ROI question indices must be 1-based and greater than 0")
        if roi.option < 0:
            raise ValueError("ROI option indices must be >= 0")
        if roi.r <= 0:
            raise ValueError("ROI radius must be > 0")
        selection_mode_raw = getattr(roi, "selection_mode", "single")
        selection_mode_name = str(selection_mode_raw).strip().lower()
        if selection_mode_name == "single":
            selection_mode = 0
        elif selection_mode_name == "multiple":
            selection_mode = 1
        else:
            raise ValueError(f"unsupported ROI selection_mode: {selection_mode_raw!r}")

        native_question = roi.question - 1
        pair = (native_question, roi.option)
        if pair in seen_pairs:
            raise ValueError(f"duplicate ROI pair detected after normalization: {pair}")

        seen_pairs.add(pair)
        source_questions.add(roi.question)
        by_question.setdefault(native_question, set()).add(roi.option)
        normalized.append(
            NativeCircleROI(
                cx=int(roi.cx),
                cy=int(roi.cy),
                r=int(roi.r),
                question=native_question,
                option=int(roi.option),
                selection_mode=selection_mode,
            )
        )

    expected_questions = set(range(1, max(source_questions) + 1))
    if source_questions != expected_questions:
        missing = sorted(expected_questions - source_questions)
        raise ValueError(f"ROI questions must be contiguous in Python 1-based space, missing: {missing}")

    option_sets = list(by_question.values())
    expected_options = set(range(max(max(options) for options in option_sets) + 1))
    for native_question, options in sorted(by_question.items()):
        if options != expected_options:
            raise ValueError(
                f"question {native_question} must contain the same 0-based option set as the others; "
                f"expected {sorted(expected_options)}, got {sorted(options)}"
            )

    normalized.sort(key=lambda roi: (roi.question, roi.option, roi.cy, roi.cx))
    n_questions = len(by_question)
    n_options_per_question = len(expected_options)
    return tuple(normalized), n_questions, n_options_per_question


def normalize_answer_key(
    answer_key: Sequence[int],
    *,
    n_questions: int,
    n_options_per_question: int,
) -> tuple[int, ...]:
    if len(answer_key) != n_questions:
        raise ValueError(
            f"answer key length mismatch: expected {n_questions}, got {len(answer_key)}"
        )

    normalized = tuple(int(value) for value in answer_key)
    for idx, value in enumerate(normalized):
        if value < -1 or value >= n_options_per_question:
            raise ValueError(
                f"answer key at question {idx} must be in [-1, {n_options_per_question - 1}], got {value}"
            )
    return normalized


def build_native_adapter_config(
    *,
    circle_rois_path: str | Path = DEFAULT_CIRCLE_ROIS_JSON,
    answer_key_path: str | Path = DEFAULT_ANSWER_KEY_JSON,
    template_layout_path: str | Path = TEMPLATE_LAYOUT_FILE,
    output_size: tuple[int, int] = A4_PX,
    windows_4pts: Iterable[Sequence[int]] | None = None,
) -> NativeAdapterConfig:
    python_rois = load_circle_rois(str(_resolve_repo_path(circle_rois_path)))
    native_rois, n_questions, n_options_per_question = normalize_circle_rois(python_rois)
    answer_key = load_answer_key(answer_key_path, fallback_question_count=n_questions)
    native_answer_key = normalize_answer_key(
        answer_key,
        n_questions=n_questions,
        n_options_per_question=n_options_per_question,
    )
    template_markers = load_template_markers(template_layout_path)
    region_windows = build_region_windows(windows_4pts, template_markers=template_markers)

    return NativeAdapterConfig(
        output_width=int(output_size[0]),
        output_height=int(output_size[1]),
        n_questions=n_questions,
        n_options_per_question=n_options_per_question,
        template_markers=template_markers,
        region_windows=region_windows,
        circle_rois=native_rois,
        answer_key=native_answer_key,
    )


def summarize_adapter_config(config: NativeAdapterConfig) -> str:
    return (
        f"output={config.output_width}x{config.output_height}, "
        f"questions={config.n_questions}, "
        f"options_per_question={config.n_options_per_question}, "
        f"template_markers={len(config.template_markers)}, "
        f"region_windows={len(config.region_windows)}, "
        f"circle_rois={len(config.circle_rois)}"
    )
