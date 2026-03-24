from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
OUTPUT_DIR = REPO_ROOT / "results" / "raw_native_param_sweep"
TEMPLATE_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"
TARGET_IMAGES = [
    REPO_ROOT / "samples" / "1photo5.jpg",
    REPO_ROOT / "samples" / "1photo6.jpg",
]


def build_ground_truth() -> tuple[WarpEngine, OMRProcessor]:
    circle_rois = load_circle_rois(str(REPO_ROOT / DEFAULT_BUBBLE_LAYOUT_JSON))
    answer_key = load_answer_key(
        REPO_ROOT / DEFAULT_ANSWER_KEY_JSON,
        fallback_question_count=max(roi.question for roi in circle_rois),
    )
    return (
        WarpEngine(str(REPO_ROOT / TEMPLATE_MARKER_POSITIONS_FILE), str(TEMPLATE_IMAGE)),
        OMRProcessor(circle_rois=circle_rois, answer_key=answer_key),
    )


def compare_answers(expected: list[int], actual: list[int]) -> dict[str, Any]:
    mismatches = []
    for idx, (lhs, rhs) in enumerate(zip(expected, actual)):
        if lhs != rhs:
            mismatches.append({"question": idx, "expected": lhs, "actual": rhs})
    return {
        "match": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dll_path = os.environ.get("OMR_DLL_PATH")
    client = NativeCoreClient(dll_path=dll_path) if dll_path else NativeCoreClient()
    config = build_native_adapter_config()
    thresholds = load_threshold_config()
    warp_engine, omr = build_ground_truth()

    param_sets: list[dict[str, Any]] = [
        {"name": "baseline", "overrides": {}},
        {"name": "patch_power_3", "overrides": {"patch_idw_power": 3.0}},
        {"name": "patch_power_2_5", "overrides": {"patch_idw_power": 2.5}},
        {"name": "skip_idw_1_0", "overrides": {"skip_idw_if_residual_lt_px": 1.0}},
        {"name": "bbox_margin_120", "overrides": {"region_bbox_margin_px": 120}},
        {
            "name": "factor_gentle",
            "overrides": {
                "residual_factors": [0.20, 0.12, 0.18, 0.25],
            },
        },
    ]

    report: dict[str, Any] = {"images": []}
    for image_path in TARGET_IMAGES:
        raw = read_image(image_path)
        warped = warp_engine.warp(raw, output=None, use_global_idw=False, use_region_refine=True, debug=False)
        prepared = omr._prep_gray(warped)
        gt = client.run(
            prepared,
            config,
            assume_aligned_input=True,
            return_scored_image=False,
            use_global_idw=False,
            use_region_refine=False,
            abs_th=thresholds.abs_th,
            rel_th=thresholds.rel_th,
            auto_threshold=thresholds.auto_threshold,
        )
        image_entry: dict[str, Any] = {
            "image": str(image_path.relative_to(REPO_ROOT)),
            "ground_truth": {
                "score": gt.score,
                "answers": gt.answers,
            },
            "runs": [],
        }
        for param_set in param_sets:
            result = client.run(
                raw,
                config,
                assume_aligned_input=False,
                return_scored_image=False,
                use_global_idw=False,
                use_region_refine=True,
                abs_th=thresholds.abs_th,
                rel_th=thresholds.rel_th,
                auto_threshold=thresholds.auto_threshold,
                warp_param_overrides=param_set["overrides"],
            )
            answer_compare = compare_answers(gt.answers, result.answers)
            image_entry["runs"].append(
                {
                    "name": param_set["name"],
                    "overrides": param_set["overrides"],
                    "score": result.score,
                    "score_match": result.score == gt.score,
                    "answer_match": answer_compare["match"],
                    "mismatch_count": answer_compare["mismatch_count"],
                    "mismatches": answer_compare["mismatches"],
                }
            )
            print(
                "[SWEEP]",
                image_path.name,
                param_set["name"],
                f"score={result.score}",
                f"score_match={result.score == gt.score}",
                f"mismatch_count={answer_compare['mismatch_count']}",
            )
        report["images"].append(image_entry)

    report_path = OUTPUT_DIR / "sweep_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("[DONE] sweep report written to", report_path)


if __name__ == "__main__":
    main()

