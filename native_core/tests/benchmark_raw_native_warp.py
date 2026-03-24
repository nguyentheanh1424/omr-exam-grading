from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2 as cv

from native_core.native_api import NativeCoreClient, read_image
from native_core.python_adapter import (
    DEFAULT_ANSWER_KEY_JSON,
    DEFAULT_CIRCLE_ROIS_JSON,
    build_native_adapter_config,
    load_answer_key,
    load_threshold_config,
)
from orm_engine.orm import OMRProcessor, load_circle_rois
from warp_engine.config import TEMPLATE_LAYOUT_FILE
from warp_engine.engine import WarpEngine


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "samples"
OUTPUT_DIR = REPO_ROOT / "results" / "native_warp_benchmark"
BASELINE_DIR = OUTPUT_DIR / "baseline_best_path"
RAW_NATIVE_DIR = OUTPUT_DIR / "raw_native"
TEMPLATE_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"


def build_python_pipeline() -> tuple[WarpEngine, OMRProcessor]:
    circle_rois = load_circle_rois(str(REPO_ROOT / DEFAULT_CIRCLE_ROIS_JSON))
    answer_key = load_answer_key(
        REPO_ROOT / DEFAULT_ANSWER_KEY_JSON,
        fallback_question_count=max(roi.question for roi in circle_rois),
    )
    return (
        WarpEngine(str(REPO_ROOT / TEMPLATE_LAYOUT_FILE), str(TEMPLATE_IMAGE)),
        OMRProcessor(circle_rois=circle_rois, answer_key=answer_key),
    )


def compare_answers(expected: list[int], actual: list[int]) -> dict[str, Any]:
    mismatches = []
    for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
        if expected_value != actual_value:
            mismatches.append(
                {
                    "question": index,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    return {
        "match": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def run_best_path_ground_truth(
    image_path: Path,
    warp_engine: WarpEngine,
    omr: OMRProcessor,
    client: NativeCoreClient,
) -> dict[str, Any]:
    img = read_image(image_path)
    warped = warp_engine.warp(
        img,
        output=None,
        use_global_idw=False,
        use_region_refine=True,
        debug=False,
    )
    python_result = omr.run(warped, output=None, debug=False)
    prepared_gray = omr._prep_gray(warped)
    config = build_native_adapter_config()
    thresholds = load_threshold_config()
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
    return {
        "python_score": int(python_result["score"]),
        "python_answers": [int(value) for value in python_result["answers"]],
        "native_score": int(native_result.score),
        "native_answers": [int(value) for value in native_result.answers],
        "used_abs_th": float(native_result.used_abs_th),
        "used_rel_th": float(native_result.used_rel_th),
        "aligned_bgr": warped,
        "prepared_gray": prepared_gray,
    }


def run_raw_native(image_path: Path, client: NativeCoreClient) -> dict[str, Any]:
    config = build_native_adapter_config()
    thresholds = load_threshold_config()
    img = read_image(image_path)
    result = client.run(
        img,
        config,
        assume_aligned_input=False,
        return_scored_image=True,
        use_global_idw=False,
        use_region_refine=True,
        abs_th=thresholds.abs_th,
        rel_th=thresholds.rel_th,
        auto_threshold=thresholds.auto_threshold,
    )
    return {
        "score": int(result.score),
        "answers": [int(value) for value in result.answers],
        "used_abs_th": float(result.used_abs_th),
        "used_rel_th": float(result.used_rel_th),
        "scored_image": result.scored_image,
    }


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), image)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(SAMPLES_DIR.glob("1photo*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No sample images found under {SAMPLES_DIR}")

    warp_engine, omr = build_python_pipeline()
    dll_path = os.environ.get("OMR_DLL_PATH")
    client = NativeCoreClient(dll_path=dll_path) if dll_path else NativeCoreClient()

    report: dict[str, Any] = {
        "ground_truth": "python_warp_python_prep_native_grading",
        "raw_native_mode": {
            "assume_aligned_input": False,
            "use_global_idw": False,
            "use_region_refine": True,
        },
        "images": [],
        "summary": {
            "total_images": len(image_paths),
            "exact_matches": 0,
            "score_matches": 0,
            "images_with_mismatches": 0,
        },
    }

    for image_path in image_paths:
        image_name = image_path.stem
        baseline = run_best_path_ground_truth(image_path, warp_engine, omr, client)
        raw_native = run_raw_native(image_path, client)
        answer_compare = compare_answers(baseline["native_answers"], raw_native["answers"])
        score_match = baseline["native_score"] == raw_native["score"]
        exact_match = score_match and answer_compare["match"]

        if exact_match:
            report["summary"]["exact_matches"] += 1
        if score_match:
            report["summary"]["score_matches"] += 1
        if not exact_match:
            report["summary"]["images_with_mismatches"] += 1

        save_image(BASELINE_DIR / f"{image_name}_aligned_bgr.png", baseline["aligned_bgr"])
        save_image(BASELINE_DIR / f"{image_name}_prepared_gray.png", baseline["prepared_gray"])
        if raw_native["scored_image"] is not None:
            save_image(RAW_NATIVE_DIR / f"{image_name}_native_scored.png", raw_native["scored_image"])

        image_report = {
            "image": str(image_path.relative_to(REPO_ROOT)),
            "ground_truth_best_path": {
                "score": baseline["native_score"],
                "answers": baseline["native_answers"],
                "used_abs_th": baseline["used_abs_th"],
                "used_rel_th": baseline["used_rel_th"],
            },
            "python_pipeline_reference": {
                "score": baseline["python_score"],
                "answers": baseline["python_answers"],
            },
            "raw_native": {
                "score": raw_native["score"],
                "answers": raw_native["answers"],
                "used_abs_th": raw_native["used_abs_th"],
                "used_rel_th": raw_native["used_rel_th"],
            },
            "artifacts": {
                "baseline_aligned_bgr": str((BASELINE_DIR / f"{image_name}_aligned_bgr.png").relative_to(REPO_ROOT)),
                "baseline_prepared_gray": str((BASELINE_DIR / f"{image_name}_prepared_gray.png").relative_to(REPO_ROOT)),
                "raw_native_scored": str((RAW_NATIVE_DIR / f"{image_name}_native_scored.png").relative_to(REPO_ROOT))
                if raw_native["scored_image"] is not None
                else None,
            },
            "score_match": score_match,
            "answer_match": answer_compare["match"],
            "mismatch_count": answer_compare["mismatch_count"],
            "mismatches": answer_compare["mismatches"],
        }
        report["images"].append(image_report)
        print(
            "[RAW-NATIVE-BENCH]",
            image_report["image"],
            f"score_match={score_match}",
            f"answer_match={answer_compare['match']}",
            f"mismatch_count={answer_compare['mismatch_count']}",
        )

    report_path = OUTPUT_DIR / "benchmark_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("[DONE] raw native warp benchmark written to", report_path)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
