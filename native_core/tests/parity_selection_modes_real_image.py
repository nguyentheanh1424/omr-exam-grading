from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from native_core.native_api import NativeCoreClient, read_image
from native_core.python_adapter import (
    DEFAULT_ANSWER_KEY_JSON,
    DEFAULT_BUBBLE_LAYOUT_JSON,
    build_native_adapter_config_from_data,
    load_answer_key,
    load_threshold_config,
)
from orm_engine.orm import OMRProcessor, load_circle_rois
from warp_engine.config import TEMPLATE_MARKER_POSITIONS_FILE
from warp_engine.engine import WarpEngine


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE = REPO_ROOT / "samples" / "1photo5.jpg"
TEMPLATE_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"
OUTPUT_DIR = REPO_ROOT / "results" / "parity_selection_modes_real_image"
MULTIPLE_QUESTIONS = {3, 8, 15, 22, 33, 40}


def build_mixed_mode_rois():
    circle_rois = load_circle_rois(str(REPO_ROOT / DEFAULT_BUBBLE_LAYOUT_JSON))
    return [
        replace(roi, selection_mode="multiple" if roi.question in MULTIPLE_QUESTIONS else "single")
        for roi in circle_rois
    ]


def build_python_pipeline():
    circle_rois = build_mixed_mode_rois()
    answer_key = load_answer_key(
        REPO_ROOT / DEFAULT_ANSWER_KEY_JSON,
        fallback_question_count=max(roi.question for roi in circle_rois),
    )
    warp_engine = WarpEngine(str(REPO_ROOT / TEMPLATE_MARKER_POSITIONS_FILE), str(TEMPLATE_IMAGE))
    omr = OMRProcessor(circle_rois=circle_rois, answer_key=answer_key)
    return warp_engine, omr, circle_rois, answer_key


def compare_sequences(py_values: list[Any], native_values: list[Any], *, label: str) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for index, (py_value, native_value) in enumerate(zip(py_values, native_values), start=1):
        if py_value != native_value:
            mismatches.append(
                {
                    "question": index,
                    "field": label,
                    "python": py_value,
                    "native": native_value,
                }
            )
    return mismatches


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    warp_engine, omr, circle_rois, answer_key = build_python_pipeline()
    config = build_native_adapter_config_from_data(
        circle_rois=circle_rois,
        answer_key=answer_key,
    )
    thresholds = load_threshold_config()
    client = NativeCoreClient()

    img = read_image(SAMPLE_IMAGE)
    warped = warp_engine.warp(
        img,
        output=None,
        use_global_idw=False,
        use_region_refine=True,
        debug=False,
    )
    python_result = omr.run(warped, output=None, debug=False)
    prepared_gray = omr._prep_gray(warped)
    native_result = client.run(
        prepared_gray,
        config,
        assume_aligned_input=True,
        return_scored_image=False,
        use_global_idw=False,
        use_region_refine=False,
        abs_th=thresholds.abs_th,
        rel_th=thresholds.rel_th,
        auto_threshold=thresholds.auto_threshold,
    )

    mismatches = []
    mismatches.extend(
        compare_sequences(
            [int(value) for value in python_result["answers"]],
            [int(value) for value in native_result.answers],
            label="answers",
        )
    )
    mismatches.extend(
        compare_sequences(
            [[int(option) for option in values] for values in python_result["selected_options"]],
            [[int(option) for option in values] for values in native_result.selected_options],
            label="selected_options",
        )
    )
    mismatches.extend(
        compare_sequences(
            [str(value) for value in python_result["question_statuses"]],
            [str(value) for value in native_result.question_statuses],
            label="question_statuses",
        )
    )

    report = {
        "image": str(SAMPLE_IMAGE.relative_to(REPO_ROOT)),
        "multiple_questions": sorted(MULTIPLE_QUESTIONS),
        "python": {
            "score": int(python_result["score"]),
            "answers": [int(value) for value in python_result["answers"]],
            "selected_options": [[int(option) for option in values] for values in python_result["selected_options"]],
            "question_statuses": [str(value) for value in python_result["question_statuses"]],
        },
        "native": {
            "score": int(native_result.score),
            "answers": [int(value) for value in native_result.answers],
            "selected_options": [[int(option) for option in values] for values in native_result.selected_options],
            "question_statuses": [str(value) for value in native_result.question_statuses],
            "used_abs_th": native_result.used_abs_th,
            "used_rel_th": native_result.used_rel_th,
        },
        "score_match": int(python_result["score"]) == int(native_result.score),
        "exact_match": len(mismatches) == 0 and int(python_result["score"]) == int(native_result.score),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }

    report_path = OUTPUT_DIR / "parity_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(
        "[PARITY-SELECTION-MODES]",
        report["image"],
        f"score_match={report['score_match']}",
        f"exact_match={report['exact_match']}",
        f"mismatch_count={report['mismatch_count']}",
    )
    print("[DONE] parity report written to", report_path)


if __name__ == "__main__":
    main()
