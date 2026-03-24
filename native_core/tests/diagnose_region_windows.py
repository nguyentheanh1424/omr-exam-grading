from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from native_core.native_api import NativeCoreClient, read_image
from native_core.python_adapter import (
    DEFAULT_ANSWER_KEY_JSON,
    DEFAULT_BUBBLE_LAYOUT_JSON,
    build_native_adapter_config,
    load_answer_key,
)
from orm_engine.orm import OMRProcessor, load_circle_rois
from warp_engine.config import TEMPLATE_MARKER_POSITIONS_FILE, WINDOWS_4PTS
from warp_engine.engine import WarpEngine
from warp_engine.region_warp import bbox_from_template
from warp_engine.template import load_template


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "results" / "region_window_diagnostics"
TARGET_IMAGES = [
    REPO_ROOT / "samples" / "1photo5.jpg",
    REPO_ROOT / "samples" / "1photo6.jpg",
]
TEMPLATE_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"


def build_python_pipeline() -> tuple[WarpEngine, OMRProcessor]:
    circle_rois = load_circle_rois(str(REPO_ROOT / DEFAULT_BUBBLE_LAYOUT_JSON))
    answer_key = load_answer_key(
        REPO_ROOT / DEFAULT_ANSWER_KEY_JSON,
        fallback_question_count=max(roi.question for roi in circle_rois),
    )
    return (
        WarpEngine(str(REPO_ROOT / TEMPLATE_MARKER_POSITIONS_FILE), str(TEMPLATE_IMAGE)),
        OMRProcessor(circle_rois, answer_key),
    )


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    gray_a = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
    gray_b = cv.cvtColor(b, cv.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(gray_a.astype(np.float32) - gray_b.astype(np.float32))))


def black_ratio(img: np.ndarray) -> float:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return float(np.mean(gray < 128))


def diagnose_image(image_path: Path) -> dict[str, Any]:
    image_name = image_path.stem
    image_output_dir = OUTPUT_DIR / image_name
    image_output_dir.mkdir(parents=True, exist_ok=True)

    layout = load_template(str(REPO_ROOT / TEMPLATE_MARKER_POSITIONS_FILE))
    warp_engine, omr = build_python_pipeline()
    native = NativeCoreClient()
    native_config = build_native_adapter_config()

    raw = read_image(image_path)
    python_warp = warp_engine.warp(raw, output=None, use_global_idw=False, use_region_refine=True, debug=False)
    native_result = native.run(
        raw,
        native_config,
        assume_aligned_input=False,
        return_scored_image=True,
        use_global_idw=False,
        use_region_refine=True,
        abs_th=omr.abs_th,
        rel_th=omr.rel_th,
        auto_threshold=False,
    )
    if native_result.scored_image is None:
        raise RuntimeError("native output image is missing")

    python_result = omr.run(python_warp, output=None, debug=False)
    native_py_result = omr.run(native_result.scored_image, output=None, debug=False)

    window_reports = []
    for window_index, marker_ids in enumerate(WINDOWS_4PTS):
        x0, y0, x1, y1 = bbox_from_template(marker_ids, layout, python_warp.shape, margin=100)
        py_patch = python_warp[y0:y1, x0:x1]
        native_patch = native_result.scored_image[y0:y1, x0:x1]
        cv.imwrite(str(image_output_dir / f"window_{window_index:02d}_python.png"), py_patch)
        cv.imwrite(str(image_output_dir / f"window_{window_index:02d}_native.png"), native_patch)

        questions_in_window = sorted(
            {
                roi.question
                for roi in omr.circle_rois
                if x0 <= roi.cx < x1 and y0 <= roi.cy < y1
            }
        )

        mismatches = []
        for q in questions_in_window:
            py_answer = python_result["answers"][q - 1]
            native_answer = native_result.answers[q - 1]
            if py_answer != native_answer:
                mismatches.append(
                    {
                        "question_1_based": q,
                        "python": py_answer,
                        "native": native_answer,
                        "python_on_native_output": native_py_result["answers"][q - 1],
                    }
                )

        window_reports.append(
            {
                "window_index": window_index,
                "marker_ids": list(marker_ids),
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "question_count": len(questions_in_window),
                "questions_in_window": questions_in_window,
                "mismatch_count": len(mismatches),
                "mean_abs_diff": mean_abs_diff(py_patch, native_patch),
                "python_black_ratio": black_ratio(py_patch),
                "native_black_ratio": black_ratio(native_patch),
                "mismatches": mismatches,
            }
        )

    report = {
        "image": str(image_path.relative_to(REPO_ROOT)),
        "python_score": python_result["score"],
        "native_score": native_result.score,
        "window_reports": window_reports,
    }

    with (image_output_dir / "window_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("[WINDOW]", report["image"])
    for window_report in window_reports:
        print(
            "  window",
            window_report["window_index"],
            "mad=",
            f"{window_report['mean_abs_diff']:.2f}",
            "mismatches=",
            window_report["mismatch_count"],
        )
    return report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"images": []}
    for image_path in TARGET_IMAGES:
        summary["images"].append(diagnose_image(image_path))
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[DONE] window diagnostics written to", OUTPUT_DIR / "summary.json")


if __name__ == "__main__":
    main()

