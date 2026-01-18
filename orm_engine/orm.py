from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

DEFAULT_ABS_TH = 0.12
DEFAULT_REL_TH = 0.04


@dataclass
class CircleROI:
    cx: int
    cy: int
    r: int
    question: int
    option: int

    @staticmethod
    def from_dict(d: dict) -> "CircleROI":
        return CircleROI(
            cx=int(d["cx"]),
            cy=int(d["cy"]),
            r=int(d["r"]),
            question=int(d["question"]),
            option=int(d["option"]),
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
                        "r_in_ratio": 0.55,
                        "r_out_ratio": 0.90,
                    },
                },
                f,
                indent=2,
            )

    @staticmethod
    def _prep_gray(img: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.createCLAHE(3.0, (8, 8)).apply(gray)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        return gray

    @staticmethod
    def _bubble_score(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
        h, w = gray.shape[:2]

        x0 = max(cx - int(1.6 * r), 0)
        y0 = max(cy - int(1.6 * r), 0)
        x1 = min(cx + int(1.6 * r), w)
        y1 = min(cy + int(1.6 * r), h)

        patch = gray[y0:y1, x0:x1].astype(np.float32)
        ph, pw = patch.shape[:2]

        pcx = cx - x0
        pcy = cy - y0

        r_fill_in = int(0.45 * r)
        r_fill_out = int(0.85 * r)
        r_bg_in = int(1.05 * r)
        r_bg_out = int(1.45 * r)

        mask_fill = np.zeros((ph, pw), np.uint8)
        mask_bg = np.zeros((ph, pw), np.uint8)

        cv.circle(mask_fill, (pcx, pcy), r_fill_out, 255, -1)
        cv.circle(mask_fill, (pcx, pcy), r_fill_in, 0, -1)

        cv.circle(mask_bg, (pcx, pcy), r_bg_out, 255, -1)
        cv.circle(mask_bg, (pcx, pcy), r_bg_in, 0, -1)

        fill_vals = patch[mask_fill > 0]
        bg_vals = patch[mask_bg > 0]

        if fill_vals.size < 20 or bg_vals.size < 20:
            return 0.0

        fill_dark = 1.0 - (fill_vals.mean() / 255.0)
        bg_dark = 1.0 - (bg_vals.mean() / 255.0)

        # QUAN TRỌNG: so sánh tương đối
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

        if len(best_vals) < 8:
            return

        best_arr = np.array(best_vals, np.float32)
        delta_arr = np.array(delta_vals, np.float32)

        k = max(3, int(0.6 * len(best_arr)))

        base_best = np.sort(best_arr)[:k]
        base_delta = np.sort(delta_arr)[:k]

        self.abs_th = float(np.clip(
            np.median(base_best) + 6.5 * self._mad(base_best) + 0.015,
            0.05, 0.40
        ))
        self.rel_th = float(np.clip(
            np.median(base_delta) + 4.5 * self._mad(base_delta) + 0.004,
            0.015, 0.25
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

            color = (160, 160, 160)
            thick = 1

            if is_multi:
                if roi.option == best_opt or roi.option == second_opt:
                    color, thick = (0, 255, 255), 3

            elif detected == -1:
                if gt is not None and roi.option == gt:
                    color, thick = (0, 0, 255), 3

            elif detected == roi.option:
                if gt is not None and detected == gt:
                    color, thick = (0, 255, 0), 3
                else:
                    color, thick = (0, 0, 255), 3

            cx = int(np.clip(roi.cx, roi.r, w - roi.r - 1))
            cy = int(np.clip(roi.cy, roi.r, h - roi.r - 1))

            cv.circle(vis, (cx, cy), roi.r, color, thick)
            cv.circle(vis, (cx, cy), max(1, roi.r // 6), color, -1)

        return vis

    @staticmethod
    def _draw_score(img: np.ndarray, score: int, total: int) -> np.ndarray:
        vis = img.copy()
        text = f"SCORE: {score}/{total}"

        (tw, th), _ = cv.getTextSize(
            text, cv.FONT_HERSHEY_SIMPLEX, 1.4, 3
        )
        x, y, p = 40, 70, 12

        cv.rectangle(vis, (x - p, y - th - p),
                      (x + tw + p, y + p),
                      (255, 255, 255), -1)
        cv.rectangle(vis, (x - p, y - th - p),
                      (x + tw + p, y + p),
                      (0, 0, 0), 2)
        cv.putText(
            vis, text, (x, y),
            cv.FONT_HERSHEY_SIMPLEX, 1.4,
            (0, 0, 0), 3, cv.LINE_AA
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

            for (q, opt), score in score_cache.items():
                roi = [r for r in self.circle_rois if r.question == q and r.option == opt][0]

                # Color by score: white (0) → red (high)
                intensity = int(score * 255)
                color = (0, 0, intensity)

                cv.circle(score_img, (roi.cx, roi.cy), roi.r, color, -1)
                cv.putText(score_img, f"{score:.2f}",
                           (roi.cx - 15, roi.cy + 5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

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
