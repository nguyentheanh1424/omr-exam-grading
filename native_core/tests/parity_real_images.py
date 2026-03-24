from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2 as cv

from native_core.native_api import NativeCoreClient, read_image
from native_core.python_adapter import (
    DEFAULT_ANSWER_KEY_JSON,
    DEFAULT_BUBBLE_LAYOUT_JSON,
    build_native_adapter_config,
    load_answer_key,
    load_threshold_config,
)
from orm_engine.orm import OMRProcessor, load_circle_rois
from warp_engine.config import TEMPLATE_MARKER_POSITIONS_FILE
from warp_engine.engine import WarpEngine


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "samples"
OUTPUT_DIR = REPO_ROOT / "results" / "parity_real_images"
TEMPLATE_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"


def build_python_pipeline() -> tuple[WarpEngine, OMRProcessor]:
    circle_rois = load_circle_rois(str(REPO_ROOT / DEFAULT_BUBBLE_LAYOUT_JSON))
    answer_key = load_answer_key(
        REPO_ROOT / DEFAULT_ANSWER_KEY_JSON,
        fallback_question_count=max(roi.question for roi in circle_rois),
    )
    warp_engine = WarpEngine(str(REPO_ROOT / TEMPLATE_MARKER_POSITIONS_FILE), str(TEMPLATE_IMAGE))
    omr = OMRProcessor(
        circle_rois=circle_rois,
        answer_key=answer_key,
    )
    return warp_engine, omr


def run_python_pipeline(
    warp_engine: WarpEngine,
    omr: OMRProcessor,
    image_path: Path,
) -> dict[str, Any]:
    img = read_image(image_path)
    warped = warp_engine.warp(
        img,
        output=None,
        use_global_idw=False,
        use_region_refine=True,
        debug=False,
    )
    result = omr.run(warped, output=None, debug=False)
    return {
        "score": int(result["score"]),
        "answers": [int(value) for value in result["answers"]],
    }


def run_native_pipeline(
    client: NativeCoreClient,
    image_path: Path,
) -> dict[str, Any]:
    config = build_native_adapter_config()
    thresholds = load_threshold_config()
    img = read_image(image_path)
    result = client.run(
        img,
        config,
        assume_aligned_input=False,
        return_scored_image=False,
        use_global_idw=False,
        use_region_refine=True,
        abs_th=thresholds.abs_th,
        rel_th=thresholds.rel_th,
        auto_threshold=thresholds.auto_threshold,
    )
    return {
        "score": result.score,
        "answers": [int(value) for value in result.answers],
        "used_abs_th": result.used_abs_th,
        "used_rel_th": result.used_rel_th,
        "configured_abs_th": thresholds.abs_th,
        "configured_rel_th": thresholds.rel_th,
    }


def compare_answers(py_answers: list[int], native_answers: list[int]) -> dict[str, Any]:
    mismatches = []
    for index, (py_value, native_value) in enumerate(zip(py_answers, native_answers)):
        if py_value != native_value:
            mismatches.append(
                {
                    "question": index,
                    "python": py_value,
                    "native": native_value,
                }
            )
    return {
        "match": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(SAMPLES_DIR.glob("1photo*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No sample images found under {SAMPLES_DIR}")

    warp_engine, omr = build_python_pipeline()
    client = NativeCoreClient()

    report: dict[str, Any] = {
        "images": [],
        "summary": {
            "total_images": len(image_paths),
            "exact_matches": 0,
            "score_matches": 0,
            "images_with_mismatches": 0,
        },
    }

    for image_path in image_paths:
        python_result = run_python_pipeline(warp_engine, omr, image_path)
        native_result = run_native_pipeline(client, image_path)
        answer_compare = compare_answers(python_result["answers"], native_result["answers"])
        score_match = python_result["score"] == native_result["score"]
        exact_match = answer_compare["match"] and score_match

        if exact_match:
            report["summary"]["exact_matches"] += 1
        if score_match:
            report["summary"]["score_matches"] += 1
        if not exact_match:
            report["summary"]["images_with_mismatches"] += 1

        image_report = {
            "image": str(image_path.relative_to(REPO_ROOT)),
            "python": python_result,
            "native": native_result,
            "score_match": score_match,
            "answer_match": answer_compare["match"],
            "mismatch_count": answer_compare["mismatch_count"],
            "mismatches": answer_compare["mismatches"],
        }
        report["images"].append(image_report)
        print(
            "[PARITY]",
            image_report["image"],
            f"score_match={score_match}",
            f"answer_match={answer_compare['match']}",
            f"mismatch_count={answer_compare['mismatch_count']}",
        )

    report_path = OUTPUT_DIR / "parity_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("[DONE] parity report written to", report_path)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

