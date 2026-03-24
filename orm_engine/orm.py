from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# THRESHOLD CONSTANTS
DEFAULT_ABS_TH = 0.12
DEFAULT_REL_TH = 0.04

# IMAGE PREPROCESSING CONSTANTS
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)
GAUSSIAN_KERNEL_SIZE = (3, 3)
GAUSSIAN_SIGMA = 0

# BUBBLE DETECTION CONSTANTS
# Patch extraction
PATCH_RADIUS_MULTIPLIER = 1.6

# Fill region (the actual bubble marking area)
FILL_INNER_RADIUS_RATIO = 0.45
FILL_OUTER_RADIUS_RATIO = 0.85

# Background region (annulus around the bubble)
BG_INNER_RADIUS_RATIO = 1.05
BG_OUTER_RADIUS_RATIO = 1.45

# Minimum pixel count for valid measurement
MIN_VALID_PIXELS = 20

# AUTO-CALIBRATION CONSTANTS
# Minimum questions needed for calibration
MIN_QUESTIONS_FOR_CALIBRATION = 8

# Percentile of data to use for baseline calculation
CALIBRATION_PERCENTILE = 0.6

# MAD multipliers for threshold calculation
ABS_TH_MAD_MULTIPLIER = 6.5
REL_TH_MAD_MULTIPLIER = 4.5

# Baseline offsets
ABS_TH_BASELINE_OFFSET = 0.015
REL_TH_BASELINE_OFFSET = 0.004

# Threshold bounds
ABS_TH_MIN = 0.20
ABS_TH_MAX = 0.40
REL_TH_MIN = 0.015
REL_TH_MAX = 0.25

# VISUALIZATION CONSTANTS
# Circle colors (BGR format)
COLOR_GRAY = (160, 160, 160)
COLOR_YELLOW = (0, 255, 255)  # Multiple answers detected
COLOR_RED = (0, 0, 255)        # Wrong or missing answer
COLOR_GREEN = (0, 255, 0)      # Correct answer

# Circle thickness
THICKNESS_NORMAL = 1
THICKNESS_HIGHLIGHT = 3

# Center dot size
CENTER_DOT_RATIO = 6  # radius // 6

# Score display
SCORE_FONT = cv.FONT_HERSHEY_SIMPLEX
SCORE_FONT_SCALE = 1.4
SCORE_FONT_THICKNESS = 3
SCORE_TEXT_X = 40
SCORE_TEXT_Y = 70
SCORE_PADDING = 12
SCORE_BG_COLOR = (255, 255, 255)
SCORE_BORDER_COLOR = (0, 0, 0)
SCORE_TEXT_COLOR = (0, 0, 0)
SCORE_BORDER_THICKNESS = 2

# Debug heatmap
HEATMAP_FONT = cv.FONT_HERSHEY_SIMPLEX
HEATMAP_FONT_SCALE = 0.3
HEATMAP_TEXT_COLOR = (255, 255, 255)
HEATMAP_TEXT_THICKNESS = 1
HEATMAP_TEXT_OFFSET_X = 15
HEATMAP_TEXT_OFFSET_Y = 5


@dataclass
class CircleROI:
    cx: int
    cy: int
    r: int
    question: int
    option: int
    selection_mode: str = "single"

    @staticmethod
    def from_dict(d: dict) -> "CircleROI":
        return CircleROI(
            cx=int(d["cx"]),
            cy=int(d["cy"]),
            r=int(d["r"]),
            question=int(d["question"]),
            option=int(d["option"]),
            selection_mode=str(d.get("selection_mode", "single")),
        )


def load_circle_rois(path: str) -> List[CircleROI]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [CircleROI.from_dict(d) for d in data]


class OMRProcessor:
    def __init__(
        self,
        circle_rois: List[CircleROI],
        answer_key: List[int],
        threshold_path: str = "config/omr_thresholds.json",
        auto_threshold: bool = True,
    ):
        self.circle_rois = circle_rois
        self.answer_key = answer_key

        self.threshold_path = threshold_path
        self.auto_threshold = auto_threshold

        self.abs_th = DEFAULT_ABS_TH
        self.rel_th = DEFAULT_REL_TH

        if os.path.exists(self.threshold_path):
            self._load_thresholds()

    def _load_thresholds(self):
        with open(self.threshold_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.abs_th = float(d["abs_th"])
        self.rel_th = float(d["rel_th"])

    def _save_thresholds(self):
        with open(self.threshold_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "abs_th": self.abs_th,
                    "rel_th": self.rel_th,
                    "meta": {
                        "method": "annulus_patch_darkness",
                        "r_in_ratio": FILL_INNER_RADIUS_RATIO,
                        "r_out_ratio": FILL_OUTER_RADIUS_RATIO,
                    },
                },
                f,
                indent=2,
            )

    @staticmethod
    def _prep_gray(img: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE).apply(gray)
        gray = cv.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
        return gray

    @staticmethod
    def _bubble_score(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
        h, w = gray.shape[:2]

        x0 = max(cx - int(PATCH_RADIUS_MULTIPLIER * r), 0)
        y0 = max(cy - int(PATCH_RADIUS_MULTIPLIER * r), 0)
        x1 = min(cx + int(PATCH_RADIUS_MULTIPLIER * r), w)
        y1 = min(cy + int(PATCH_RADIUS_MULTIPLIER * r), h)

        patch = gray[y0:y1, x0:x1].astype(np.float32)
        ph, pw = patch.shape[:2]

        pcx = cx - x0
        pcy = cy - y0

        r_fill_in = int(FILL_INNER_RADIUS_RATIO * r)
        r_fill_out = int(FILL_OUTER_RADIUS_RATIO * r)
        r_bg_in = int(BG_INNER_RADIUS_RATIO * r)
        r_bg_out = int(BG_OUTER_RADIUS_RATIO * r)

        mask_fill = np.zeros((ph, pw), np.uint8)
        mask_bg = np.zeros((ph, pw), np.uint8)

        cv.circle(mask_fill, (pcx, pcy), r_fill_out, 255, -1)
        cv.circle(mask_fill, (pcx, pcy), r_fill_in, 0, -1)

        cv.circle(mask_bg, (pcx, pcy), r_bg_out, 255, -1)
        cv.circle(mask_bg, (pcx, pcy), r_bg_in, 0, -1)

        fill_vals = patch[mask_fill > 0]
        bg_vals = patch[mask_bg > 0]

        if fill_vals.size < MIN_VALID_PIXELS or bg_vals.size < MIN_VALID_PIXELS:
            return 0.0

        fill_dark = 1.0 - (fill_vals.mean() / 255.0)
        bg_dark = 1.0 - (bg_vals.mean() / 255.0)

        return max(0.0, fill_dark - bg_dark)

    @staticmethod
    def _mad(x: np.ndarray) -> float:
        m = float(np.median(x))
        return float(np.median(np.abs(x - m))) + 1e-9

    def _auto_calibrate(self, score_cache: Dict[Tuple[int, int], float]):
        by_q: Dict[int, List[float]] = {}

        for (q, _), s in score_cache.items():
            by_q.setdefault(q, []).append(s)

        best_vals = []
        delta_vals = []

        for scores in by_q.values():
            if len(scores) < 2:
                continue
            scores = sorted(scores, reverse=True)
            best_vals.append(scores[0])
            delta_vals.append(scores[0] - scores[1])

        if len(best_vals) < MIN_QUESTIONS_FOR_CALIBRATION:
            return

        best_arr = np.array(best_vals, np.float32)
        delta_arr = np.array(delta_vals, np.float32)

        k = max(3, int(CALIBRATION_PERCENTILE * len(best_arr)))

        base_best = np.sort(best_arr)[:k]
        base_delta = np.sort(delta_arr)[:k]

        self.abs_th = float(np.clip(
            np.median(base_best) + ABS_TH_MAD_MULTIPLIER * self._mad(base_best) + ABS_TH_BASELINE_OFFSET,
            ABS_TH_MIN, ABS_TH_MAX
        ))
        self.rel_th = float(np.clip(
            np.median(base_delta) + REL_TH_MAD_MULTIPLIER * self._mad(base_delta) + REL_TH_BASELINE_OFFSET,
            REL_TH_MIN, REL_TH_MAX
        ))

        self._save_thresholds()

    def _detect_answers(
        self,
        score_cache: Dict[Tuple[int, int], float],
    ) -> List[int]:
        by_q: Dict[int, List[Tuple[int, float]]] = {}
        max_q = 0

        for roi in self.circle_rois:
            s = score_cache[(roi.question, roi.option)]
            by_q.setdefault(roi.question, []).append((roi.option, s))
            max_q = max(max_q, roi.question)

        answers = [-1] * max_q

        for q, items in by_q.items():
            items.sort(key=lambda x: x[1], reverse=True)
            best_opt, best_val = items[0]
            second_val = items[1][1] if len(items) > 1 else -1e9

            if best_val >= self.abs_th and (best_val - second_val) >= self.rel_th:
                answers[q - 1] = best_opt

        return answers

    def _grade(self, answers: List[int]) -> Tuple[int, List[bool]]:
        score = 0
        per = []
        for i, gt in enumerate(self.answer_key):
            ok = i < len(answers) and answers[i] == gt
            per.append(ok)
            if ok:
                score += 1
        return score, per

    def _draw_overlay(
        self,
        img: np.ndarray,
        score_cache: Dict[Tuple[int, int], float],
        answers: List[int],
    ) -> np.ndarray:
        vis = img.copy()
        h, w = img.shape[:2]

        by_q: Dict[int, List[Tuple[int, float]]] = {}
        for (q, opt), s in score_cache.items():
            by_q.setdefault(q, []).append((opt, s))

        for roi in self.circle_rois:
            q_idx = roi.question - 1
            if q_idx < 0 or q_idx >= len(answers):
                continue

            detected = answers[q_idx]
            gt = self.answer_key[q_idx] if q_idx < len(self.answer_key) else None

            items = by_q.get(roi.question, [])
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)

            best_opt, best_val = items_sorted[0]
            second_opt, second_val = items_sorted[1] if len(items_sorted) > 1 else (None, -1e9)

            is_multi = (
                best_val >= self.abs_th and
                second_val >= self.abs_th and
                (best_val - second_val) < self.rel_th
            )

            color = COLOR_GRAY
            thick = THICKNESS_NORMAL

            if is_multi:
                if roi.option == best_opt or roi.option == second_opt:
                    color, thick = COLOR_YELLOW, THICKNESS_HIGHLIGHT

            elif detected == -1:
                if gt is not None and roi.option == gt:
                    color, thick = COLOR_RED, THICKNESS_HIGHLIGHT

            elif detected == roi.option:
                if gt is not None and detected == gt:
                    color, thick = COLOR_GREEN, THICKNESS_HIGHLIGHT
                else:
                    color, thick = COLOR_RED, THICKNESS_HIGHLIGHT

            cx = int(np.clip(roi.cx, roi.r, w - roi.r - 1))
            cy = int(np.clip(roi.cy, roi.r, h - roi.r - 1))

            cv.circle(vis, (cx, cy), roi.r, color, thick)
            cv.circle(vis, (cx, cy), max(1, roi.r // CENTER_DOT_RATIO), color, -1)

        return vis

    @staticmethod
    def _draw_score(img: np.ndarray, score: int, total: int) -> np.ndarray:
        vis = img.copy()
        text = f"SCORE: {score}/{total}"

        (tw, th), _ = cv.getTextSize(
            text, SCORE_FONT, SCORE_FONT_SCALE, SCORE_FONT_THICKNESS
        )

        cv.rectangle(
            vis,
            (SCORE_TEXT_X - SCORE_PADDING, SCORE_TEXT_Y - th - SCORE_PADDING),
            (SCORE_TEXT_X + tw + SCORE_PADDING, SCORE_TEXT_Y + SCORE_PADDING),
            SCORE_BG_COLOR,
            -1
        )
        cv.rectangle(
            vis,
            (SCORE_TEXT_X - SCORE_PADDING, SCORE_TEXT_Y - th - SCORE_PADDING),
            (SCORE_TEXT_X + tw + SCORE_PADDING, SCORE_TEXT_Y + SCORE_PADDING),
            SCORE_BORDER_COLOR,
            SCORE_BORDER_THICKNESS
        )
        cv.putText(
            vis, text, (SCORE_TEXT_X, SCORE_TEXT_Y),
            SCORE_FONT, SCORE_FONT_SCALE,
            SCORE_TEXT_COLOR, SCORE_FONT_THICKNESS, cv.LINE_AA
        )
        return vis

    def run(self, a4_img: np.ndarray, output=None, debug=False) -> Dict:
        gray = self._prep_gray(a4_img)

        if debug:
            cv.imwrite(f"{output}/omr_1_preprocessed_gray.png", gray)

        score_cache: Dict[Tuple[int, int], float] = {}
        for roi in self.circle_rois:
            score_cache[(roi.question, roi.option)] = \
                self._bubble_score(gray, roi.cx, roi.cy, roi.r)

        if self.auto_threshold and not os.path.exists(self.threshold_path):
            self._auto_calibrate(score_cache)

        answers = self._detect_answers(score_cache)
        score, per = self._grade(answers)

        vis = self._draw_overlay(a4_img, score_cache, answers)
        vis = self._draw_score(vis, score, len(self.answer_key))

        if debug:
            score_img = a4_img.copy()

            for (q, opt), bubble_score_value in score_cache.items():
                roi = [r for r in self.circle_rois if r.question == q and r.option == opt][0]

                intensity = int(bubble_score_value * 255)
                color = (0, 0, intensity)

                cv.circle(score_img, (roi.cx, roi.cy), roi.r, color, -1)
                cv.putText(
                    score_img, f"{bubble_score_value:.2f}",
                    (roi.cx - HEATMAP_TEXT_OFFSET_X, roi.cy + HEATMAP_TEXT_OFFSET_Y),
                    HEATMAP_FONT, HEATMAP_FONT_SCALE, HEATMAP_TEXT_COLOR,
                    HEATMAP_TEXT_THICKNESS
                )

            cv.imwrite(f"{output}/omr_2_score_heatmap.png", score_img)

        return {
            "answers": answers,
            "score": score,
            "per_correct": per,
            "scored_img": vis,
            "thresholds": {
                "abs_th": self.abs_th,
                "rel_th": self.rel_th,
            },
        }
