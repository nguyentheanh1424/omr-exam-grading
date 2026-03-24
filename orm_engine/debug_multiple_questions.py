from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np

from orm_engine.orm import OMRProcessor, load_circle_rois
from warp_engine.config import TEMPLATE_MARKER_POSITIONS_FILE
from warp_engine.engine import WarpEngine


DEFAULT_INPUT = "samples/1photo5.jpg"
DEFAULT_TEMPLATE = "samples/template_scan1.png"
DEFAULT_OUTPUT = "results/orm_multiple_debug"
DEFAULT_ROIS = "config/omr_bubble_layout.json"
DEFAULT_THRESHOLDS = "config/omr_bubble_thresholds.json"
DEFAULT_ANSWER_KEY = "config/answer_key.json"

TEXT_FONT = cv.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug OMR questions classified as multiple/uncertain."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--layout", default=TEMPLATE_MARKER_POSITIONS_FILE)
    parser.add_argument("--rois", default=DEFAULT_ROIS)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--answer-key", default=DEFAULT_ANSWER_KEY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["multiple"],
        help="Question statuses to export, for example: multiple uncertain",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_answer_key(path: str, rois) -> List[int]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return [int(x) for x in json.load(handle)]
    max_q = max(roi.question for roi in rois)
    return [0] * max_q


def question_bounds(rois_for_question, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    pad = max(18, max(roi.r for roi in rois_for_question))
    x0 = max(0, min(roi.cx - roi.r for roi in rois_for_question) - pad)
    y0 = max(0, min(roi.cy - roi.r for roi in rois_for_question) - pad)
    x1 = min(w, max(roi.cx + roi.r for roi in rois_for_question) + pad)
    y1 = min(h, max(roi.cy + roi.r for roi in rois_for_question) + pad)
    return x0, y0, x1, y1


def draw_question_overlay(
    image: np.ndarray,
    rois_for_question,
    sorted_scores: List[Tuple[int, float]],
    selected_options: List[int],
    status: str,
    answer: int,
) -> np.ndarray:
    vis = OMRProcessor._ensure_bgr(image)
    score_by_option = {opt: value for opt, value in sorted_scores}

    for roi in rois_for_question:
        if roi.option in selected_options:
            color = (0, 255, 255)
            thickness = 3
        elif answer == roi.option:
            color = (0, 255, 0)
            thickness = 3
        else:
            color = (160, 160, 160)
            thickness = 1

        cv.circle(vis, (roi.cx, roi.cy), roi.r, color, thickness)
        label = f"{roi.option}:{score_by_option[roi.option]:.3f}"
        cv.putText(
            vis,
            label,
            (roi.cx - 28, roi.cy - roi.r - 8),
            TEXT_FONT,
            0.35,
            (0, 0, 255),
            1,
            cv.LINE_AA,
        )

    title = f"status={status} answer={answer} selected={selected_options}"
    cv.putText(vis, title, (24, 32), TEXT_FONT, 0.7, (0, 0, 255), 2, cv.LINE_AA)
    return vis


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    rois = load_circle_rois(args.rois)
    answer_key = load_answer_key(args.answer_key, rois)

    input_img = cv.imread(args.input)
    if input_img is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    warp_engine = WarpEngine(args.layout, args.template)
    warp_artifacts = warp_engine.warp_with_artifacts(
        input_img,
        output=None,
        use_global_idw=False,
        use_region_refine=True,
        debug=False,
    )
    warped_a4 = warp_artifacts.template_merged_img

    processor = OMRProcessor(
        circle_rois=rois,
        answer_key=answer_key,
        threshold_path=args.thresholds,
        auto_threshold=False,
    )

    gray = processor._prep_gray(warped_a4)
    score_cache: Dict[Tuple[int, int], float] = {}
    for roi in rois:
        score_cache[(roi.question, roi.option)] = processor._bubble_score(gray, roi.cx, roi.cy, roi.r)

    answers, selected_options, question_statuses = processor._detect_answers(score_cache)

    questions: Dict[int, List] = {}
    for roi in rois:
        questions.setdefault(roi.question, []).append(roi)

    report = {
        "input": args.input,
        "template": args.template,
        "thresholds": {"abs_th": processor.abs_th, "rel_th": processor.rel_th},
        "statuses_filter": args.statuses,
        "questions": [],
    }

    for question in sorted(questions):
        status = question_statuses[question - 1]
        if status not in args.statuses:
            continue

        rois_for_question = sorted(questions[question], key=lambda roi: roi.option)
        sorted_scores = sorted(
            ((roi.option, score_cache[(roi.question, roi.option)]) for roi in rois_for_question),
            key=lambda item: item[1],
            reverse=True,
        )
        x0, y0, x1, y1 = question_bounds(rois_for_question, warped_a4.shape)

        overlay = draw_question_overlay(
            warped_a4,
            rois_for_question,
            sorted_scores,
            selected_options[question - 1],
            status,
            answers[question - 1],
        )
        crop = warped_a4[y0:y1, x0:x1]
        crop_overlay = overlay[y0:y1, x0:x1]

        q_dir = os.path.join(args.output_dir, f"q{question:02d}")
        ensure_dir(q_dir)
        cv.imwrite(os.path.join(q_dir, "crop.png"), crop)
        cv.imwrite(os.path.join(q_dir, "crop_overlay.png"), crop_overlay)

        top_scores = [value for _, value in sorted_scores[:2]]
        score_gap = top_scores[0] - top_scores[1] if len(top_scores) >= 2 else top_scores[0]

        question_report = {
            "question": int(question),
            "status": status,
            "answer": int(answers[question - 1]),
            "selected_options": [int(option) for option in selected_options[question - 1]],
            "correct_option": int(answer_key[question - 1]) if question - 1 < len(answer_key) else None,
            "score_gap_top2": float(score_gap),
            "roi_bounds": {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)},
            "scores_desc": [
                {"option": int(option), "score": float(value)}
                for option, value in sorted_scores
            ],
            "artifacts": {
                "crop": os.path.join(q_dir, "crop.png"),
                "crop_overlay": os.path.join(q_dir, "crop_overlay.png"),
            },
        }
        report["questions"].append(question_report)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

