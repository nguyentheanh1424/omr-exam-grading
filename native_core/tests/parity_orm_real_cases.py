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
DATASET_PATH = REPO_ROOT / "native_core" / "tests" / "data" / "orm_real_cases.json"
OUTPUT_DIR = REPO_ROOT / "results" / "orm_native_real_cases"
TEMPLATE_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"


def load_case_set() -> list[dict[str, Any]]:
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"expected non-empty JSON list in {DATASET_PATH}")
    return data


def build_case_rois(question: int, selection_mode: str):
    base_rois = load_circle_rois(str(REPO_ROOT / DEFAULT_BUBBLE_LAYOUT_JSON))
    return [
        replace(roi, selection_mode=selection_mode if roi.question == question else roi.selection_mode)
        for roi in base_rois
    ]


def load_prepared_gray_cache() -> dict[str, Any]:
    warp_engine = WarpEngine(str(REPO_ROOT / TEMPLATE_MARKER_POSITIONS_FILE), str(TEMPLATE_IMAGE))
    base_rois = load_circle_rois(str(REPO_ROOT / DEFAULT_BUBBLE_LAYOUT_JSON))
    answer_key = load_answer_key(
        REPO_ROOT / DEFAULT_ANSWER_KEY_JSON,
        fallback_question_count=max(roi.question for roi in base_rois),
    )
    omr = OMRProcessor(circle_rois=base_rois, answer_key=answer_key)

    cache: dict[str, Any] = {}
    for image_path in sorted((REPO_ROOT / "samples").glob("1photo*.jpg")):
        img = read_image(image_path)
        warped = warp_engine.warp(
            img,
            output=None,
            use_global_idw=False,
            use_region_refine=True,
            debug=False,
        )
        cache[str(image_path.relative_to(REPO_ROOT)).replace("\\", "/")] = omr._prep_gray(warped)
    return cache


def run_python_case(prepared_gray, circle_rois, answer_key, question: int) -> dict[str, Any]:
    omr = OMRProcessor(circle_rois=circle_rois, answer_key=answer_key)
    score_cache = {
        (roi.question, roi.option): omr._bubble_score(prepared_gray, roi.cx, roi.cy, roi.r)
        for roi in omr.circle_rois
    }
    answers, selected_options, question_statuses = omr._detect_answers(score_cache)
    q_idx = question - 1
    return {
        "answer": int(answers[q_idx]),
        "selected_options": [int(value) for value in selected_options[q_idx]],
        "question_status": str(question_statuses[q_idx]),
    }


def run_native_case(client, prepared_gray, circle_rois, answer_key, question: int, thresholds) -> dict[str, Any]:
    config = build_native_adapter_config_from_data(circle_rois=circle_rois, answer_key=answer_key)
    result = client.run(
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
    q_idx = question - 1
    return {
        "answer": int(result.answers[q_idx]),
        "selected_options": [int(value) for value in result.selected_options[q_idx]],
        "question_status": str(result.question_statuses[q_idx]),
    }


def compare_case(expected: dict[str, Any], actual: dict[str, Any], source: str) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    if actual["answer"] != expected["expected_answer"]:
        mismatches.append(
            {
                "source": source,
                "field": "answer",
                "expected": expected["expected_answer"],
                "actual": actual["answer"],
            }
        )
    if actual["selected_options"] != expected["expected_selected_options"]:
        mismatches.append(
            {
                "source": source,
                "field": "selected_options",
                "expected": expected["expected_selected_options"],
                "actual": actual["selected_options"],
            }
        )
    if actual["question_status"] != expected["expected_status"]:
        mismatches.append(
            {
                "source": source,
                "field": "question_status",
                "expected": expected["expected_status"],
                "actual": actual["question_status"],
            }
        )
    return mismatches


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_case_set()
    prepared_gray_cache = load_prepared_gray_cache()
    thresholds = load_threshold_config()
    client = NativeCoreClient()
    answer_key = load_answer_key(REPO_ROOT / DEFAULT_ANSWER_KEY_JSON, fallback_question_count=40)

    summary = {
        "dataset": str(DATASET_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "total_cases": len(cases),
        "python_passed": 0,
        "native_passed": 0,
        "all_passed": 0,
    }
    report_cases: list[dict[str, Any]] = []

    for case in cases:
        image_key = case["image"]
        prepared_gray = prepared_gray_cache[image_key]
        circle_rois = build_case_rois(case["question"], case["selection_mode"])
        python_actual = run_python_case(prepared_gray, circle_rois, answer_key, case["question"])
        native_actual = run_native_case(client, prepared_gray, circle_rois, answer_key, case["question"], thresholds)
        python_mismatches = compare_case(case, python_actual, "python")
        native_mismatches = compare_case(case, native_actual, "native")
        python_ok = len(python_mismatches) == 0
        native_ok = len(native_mismatches) == 0

        if python_ok:
            summary["python_passed"] += 1
        if native_ok:
            summary["native_passed"] += 1
        if python_ok and native_ok:
            summary["all_passed"] += 1

        report_cases.append(
            {
                "id": case["id"],
                "image": case["image"],
                "question": case["question"],
                "selection_mode": case["selection_mode"],
                "expected": {
                    "answer": case["expected_answer"],
                    "selected_options": case["expected_selected_options"],
                    "question_status": case["expected_status"],
                },
                "python": python_actual,
                "native": native_actual,
                "python_ok": python_ok,
                "native_ok": native_ok,
                "mismatches": python_mismatches + native_mismatches,
                "note": case.get("note"),
            }
        )
        print(
            "[ORM-REAL-CASE]",
            case["id"],
            f"python_ok={python_ok}",
            f"native_ok={native_ok}",
            f"question={case['question']}",
            f"mode={case['selection_mode']}",
        )

    report = {
        "summary": summary,
        "cases": report_cases,
    }
    report_path = OUTPUT_DIR / "parity_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("[DONE] parity report written to", report_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
