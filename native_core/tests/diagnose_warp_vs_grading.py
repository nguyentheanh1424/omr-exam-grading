from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2 as cv

from native_core.native_api import NativeCoreClient, read_image
from native_core.python_adapter import (
    DEFAULT_ANSWER_KEY_JSON,
    DEFAULT_CIRCLE_ROIS_JSON,
    build_native_adapter_config,
    load_answer_key,
)
from orm_engine.orm import OMRProcessor, load_circle_rois
from warp_engine.config import TEMPLATE_LAYOUT_FILE
from warp_engine.engine import WarpEngine


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "results" / "warp_diagnostics"
TARGET_IMAGES = [
    REPO_ROOT / "samples" / "1photo5.jpg",
    REPO_ROOT / "samples" / "1photo6.jpg",
]
TEMPLATE_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"


def build_python_processor() -> OMRProcessor:
    circle_rois = load_circle_rois(str(REPO_ROOT / DEFAULT_CIRCLE_ROIS_JSON))
    answer_key = load_answer_key(
        REPO_ROOT / DEFAULT_ANSWER_KEY_JSON,
        fallback_question_count=max(roi.question for roi in circle_rois),
    )
    return OMRProcessor(circle_rois=circle_rois, answer_key=answer_key)


def build_warp_engine() -> WarpEngine:
    return WarpEngine(str(REPO_ROOT / TEMPLATE_LAYOUT_FILE), str(TEMPLATE_IMAGE))


def compute_python_scores(omr: OMRProcessor, img) -> tuple[dict[tuple[int, int], float], list[int], int]:
    gray = omr._prep_gray(img)
    score_cache = {}
    for roi in omr.circle_rois:
        score_cache[(roi.question, roi.option)] = omr._bubble_score(gray, roi.cx, roi.cy, roi.r)
    answers = omr._detect_answers(score_cache)
    score, _ = omr._grade(answers)
    return score_cache, answers, score


def question_option_scores(score_cache: dict[tuple[int, int], float], question_1_based: int) -> list[dict[str, Any]]:
    options = []
    for option in range(5):
        options.append(
            {
                "option": option,
                "score": float(score_cache[(question_1_based, option)]),
            }
        )
    return options


def diagnose_image(image_path: Path) -> dict[str, Any]:
    image_name = image_path.stem
    image_output_dir = OUTPUT_DIR / image_name
    image_output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_image(image_path)
    warp_engine = build_warp_engine()
    omr = build_python_processor()
    native = NativeCoreClient()

    python_warp = warp_engine.warp(
        raw,
        output=None,
        use_global_idw=False,
        use_region_refine=True,
        debug=False,
    )
    cv.imwrite(str(image_output_dir / "python_warp.png"), python_warp)

    native_result = native.run(
        raw,
        config=build_native_adapter_config(),
        assume_aligned_input=False,
        return_scored_image=True,
        use_global_idw=False,
        use_region_refine=True,
        abs_th=omr.abs_th,
        rel_th=omr.rel_th,
        auto_threshold=False,
    )
    if native_result.scored_image is None:
        raise RuntimeError("native run did not return a scored image buffer")
    cv.imwrite(str(image_output_dir / "native_output.png"), native_result.scored_image)

    py_score_cache, py_answers, py_score = compute_python_scores(omr, python_warp)
    native_img_score_cache, py_on_native_answers, py_on_native_score = compute_python_scores(
        omr,
        native_result.scored_image,
    )

    mismatch_questions = []
    for idx, (py_answer, native_answer) in enumerate(zip(py_answers, native_result.answers), start=1):
        if py_answer != native_answer:
            mismatch_questions.append(
                {
                    "question_1_based": idx,
                    "python_answer": py_answer,
                    "native_answer": native_answer,
                    "python_on_native_output_answer": py_on_native_answers[idx - 1],
                    "python_option_scores_on_python_warp": question_option_scores(py_score_cache, idx),
                    "python_option_scores_on_native_output": question_option_scores(native_img_score_cache, idx),
                }
            )

    summary = {
        "image": str(image_path.relative_to(REPO_ROOT)),
        "python_thresholds": {
            "abs_th": omr.abs_th,
            "rel_th": omr.rel_th,
        },
        "python_pipeline": {
            "score": py_score,
            "answers": py_answers,
        },
        "native_pipeline": {
            "score": native_result.score,
            "answers": native_result.answers,
        },
        "python_grading_on_native_output": {
            "score": py_on_native_score,
            "answers": py_on_native_answers,
        },
        "mismatch_count": len(mismatch_questions),
        "mismatch_questions": mismatch_questions,
    }

    with (image_output_dir / "diagnostic.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "[DIAG]",
        summary["image"],
        f"mismatch_count={summary['mismatch_count']}",
        f"python_score={py_score}",
        f"native_score={native_result.score}",
        f"python_on_native_score={py_on_native_score}",
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall = {"images": []}
    for image_path in TARGET_IMAGES:
        overall["images"].append(diagnose_image(image_path))
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    print("[DONE] diagnostic summary written to", OUTPUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
