from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

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
BENCHMARK_REPORT = REPO_ROOT / "results" / "native_warp_benchmark" / "benchmark_report.json"
OUTPUT_DIR = REPO_ROOT / "results" / "native_warp_remaining_mismatches"
TEMPLATE_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"

THRESHOLD_ABS_EPS = 0.025
THRESHOLD_REL_EPS = 0.020
SHIFT_MAG_GEOMETRY_PX = 2.0
PATCH_MAD_GEOMETRY = 22.0
PATCH_MAD_THRESHOLD_EDGE = 14.0


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), img)


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


def compute_score_cache(omr: OMRProcessor, gray: np.ndarray) -> dict[tuple[int, int], float]:
    score_cache: dict[tuple[int, int], float] = {}
    for roi in omr.circle_rois:
        score_cache[(roi.question, roi.option)] = float(omr._bubble_score(gray, roi.cx, roi.cy, roi.r))
    return score_cache


def summarize_question(
    omr: OMRProcessor,
    score_cache: dict[tuple[int, int], float],
    question_1_based: int,
) -> dict[str, Any]:
    items: list[tuple[int, float]] = []
    for option in range(5):
        items.append((option, float(score_cache[(question_1_based, option)])))
    items.sort(key=lambda x: x[1], reverse=True)
    best_opt, best_val = items[0]
    second_opt, second_val = items[1]
    detected = -1
    if best_val >= omr.abs_th and (best_val - second_val) >= omr.rel_th:
        detected = best_opt
    return {
        "detected_answer": detected,
        "best_option": int(best_opt),
        "best_score": float(best_val),
        "second_option": int(second_opt),
        "second_score": float(second_val),
        "margin_vs_second": float(best_val - second_val),
        "close_to_abs_threshold": abs(best_val - omr.abs_th) <= THRESHOLD_ABS_EPS,
        "close_to_rel_threshold": abs((best_val - second_val) - omr.rel_th) <= THRESHOLD_REL_EPS,
        "options": [
            {"option": int(option), "score": float(score)}
            for option, score in sorted(items, key=lambda x: x[0])
        ],
    }


def get_question_rois(omr: OMRProcessor, question_1_based: int):
    return [roi for roi in omr.circle_rois if roi.question == question_1_based]


def question_bbox(question_rois: list[Any], margin_px: int) -> tuple[int, int, int, int]:
    x0 = min(roi.cx - roi.r for roi in question_rois) - margin_px
    y0 = min(roi.cy - roi.r for roi in question_rois) - margin_px
    x1 = max(roi.cx + roi.r for roi in question_rois) + margin_px
    y1 = max(roi.cy + roi.r for roi in question_rois) + margin_px
    return x0, y0, x1, y1


def crop_with_clamp(img: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def draw_question_overlay(
    img: np.ndarray,
    question_rois: list[Any],
    bbox: tuple[int, int, int, int],
    score_summary: dict[str, Any],
) -> np.ndarray:
    vis = ensure_bgr(img).copy()
    x0, y0, _, _ = bbox
    best_opt = score_summary["best_option"]
    for roi in question_rois:
        center = (roi.cx - x0, roi.cy - y0)
        color = (0, 255, 0) if roi.option == best_opt else (0, 255, 255)
        cv.circle(vis, center, roi.r, color, 2)
        cv.putText(
            vis,
            f"{roi.option}",
            (center[0] - 8, center[1] - roi.r - 6),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv.LINE_AA,
        )
    return vis


def annotate_patch(
    img: np.ndarray,
    title: str,
    question_rois: list[Any],
    bbox: tuple[int, int, int, int],
    score_summary: dict[str, Any],
) -> np.ndarray:
    vis = draw_question_overlay(img, question_rois, bbox, score_summary)
    cv.putText(vis, title, (8, 18), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    return vis


def phase_shift(base_gray: np.ndarray, other_gray: np.ndarray) -> tuple[float, float, float]:
    base = base_gray.astype(np.float32)
    other = other_gray.astype(np.float32)
    shift, response = cv.phaseCorrelate(base, other)
    dx = float(shift[0])
    dy = float(shift[1])
    return dx, dy, float(response)


def classify_mismatch(
    base_summary: dict[str, Any],
    raw_summary: dict[str, Any],
    shift_mag: float,
    patch_mad: float,
) -> str:
    threshold_edge = (
        raw_summary["close_to_abs_threshold"] or raw_summary["close_to_rel_threshold"]
    ) and shift_mag < SHIFT_MAG_GEOMETRY_PX and patch_mad < PATCH_MAD_THRESHOLD_EDGE
    geometry = shift_mag >= SHIFT_MAG_GEOMETRY_PX or patch_mad >= PATCH_MAD_GEOMETRY
    if threshold_edge and not geometry:
        return "threshold_edge"
    if geometry and not threshold_edge:
        return "minor_roi_geometry_or_warp"
    if geometry and threshold_edge:
        return "mixed_geometry_and_threshold"
    if base_summary["best_option"] != raw_summary["best_option"]:
        return "mixed_or_unclear"
    return "threshold_edge"


def load_benchmark_mismatches() -> list[dict[str, Any]]:
    data = json.loads(BENCHMARK_REPORT.read_text(encoding="utf-8"))
    targets: list[dict[str, Any]] = []
    for image_report in data["images"]:
        image_rel = image_report["image"]
        for mismatch in image_report["mismatches"]:
            targets.append(
                {
                    "image_rel": image_rel,
                    "question_0_based": int(mismatch["question"]),
                    "expected": int(mismatch["expected"]),
                    "actual": int(mismatch["actual"]),
                }
            )
    return targets


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_benchmark_mismatches()
    if not targets:
        raise RuntimeError(f"No mismatches found in {BENCHMARK_REPORT}")

    warp_engine, omr = build_python_pipeline()
    dll_path = os.environ.get("OMR_DLL_PATH")
    client = NativeCoreClient(dll_path=dll_path) if dll_path else NativeCoreClient()
    config = build_native_adapter_config()
    thresholds = load_threshold_config()

    report: dict[str, Any] = {
        "dll_path": str(dll_path) if dll_path else str(client._dll_path),
        "target_count": len(targets),
        "targets": [],
    }

    for target in targets:
        image_rel = target["image_rel"]
        image_path = REPO_ROOT / image_rel
        image_name = Path(image_rel).stem
        question_1_based = target["question_0_based"] + 1

        raw = read_image(image_path)
        baseline_aligned = warp_engine.warp(
            raw,
            output=None,
            use_global_idw=False,
            use_region_refine=True,
            debug=False,
        )
        baseline_gray = omr._prep_gray(baseline_aligned)
        baseline_scores = compute_score_cache(omr, baseline_gray)
        baseline_summary = summarize_question(omr, baseline_scores, question_1_based)

        raw_native = client.run(
            raw,
            config,
            assume_aligned_input=False,
            return_scored_image=True,
            use_global_idw=False,
            use_region_refine=True,
            abs_th=thresholds.abs_th,
            rel_th=thresholds.rel_th,
            auto_threshold=thresholds.auto_threshold,
        )
        if raw_native.scored_image is None:
            raise RuntimeError("native run did not return a scored image")

        raw_native_gray = omr._prep_gray(ensure_bgr(raw_native.scored_image))
        raw_scores = compute_score_cache(omr, raw_native_gray)
        raw_summary = summarize_question(omr, raw_scores, question_1_based)

        question_rois = get_question_rois(omr, question_1_based)
        margin_px = max(roi.r for roi in question_rois) * 3
        bbox = question_bbox(question_rois, margin_px)
        baseline_patch, clamped_bbox = crop_with_clamp(baseline_gray, bbox)
        raw_patch, _ = crop_with_clamp(raw_native_gray, bbox)
        baseline_patch_bgr, _ = crop_with_clamp(baseline_aligned, bbox)
        raw_patch_bgr, _ = crop_with_clamp(ensure_bgr(raw_native.scored_image), bbox)

        dx, dy, response = phase_shift(baseline_patch, raw_patch)
        shift_mag = float((dx * dx + dy * dy) ** 0.5)
        patch_mad = float(np.mean(np.abs(baseline_patch.astype(np.float32) - raw_patch.astype(np.float32))))
        classification = classify_mismatch(baseline_summary, raw_summary, shift_mag, patch_mad)

        question_dir = OUTPUT_DIR / image_name / f"q{question_1_based:02d}"
        question_dir.mkdir(parents=True, exist_ok=True)

        baseline_overlay = annotate_patch(
            baseline_patch_bgr,
            f"baseline q{question_1_based}",
            question_rois,
            clamped_bbox,
            baseline_summary,
        )
        raw_overlay = annotate_patch(
            raw_patch_bgr,
            f"raw_native q{question_1_based}",
            question_rois,
            clamped_bbox,
            raw_summary,
        )
        side_by_side = np.hstack([baseline_overlay, raw_overlay])

        save_image(question_dir / "baseline_patch.png", baseline_patch_bgr)
        save_image(question_dir / "raw_native_patch.png", raw_patch_bgr)
        save_image(question_dir / "baseline_overlay.png", baseline_overlay)
        save_image(question_dir / "raw_native_overlay.png", raw_overlay)
        save_image(question_dir / "side_by_side.png", side_by_side)

        option_reports = []
        for roi in sorted(question_rois, key=lambda item: item.option):
            option_bbox = (
                roi.cx - 2 * roi.r,
                roi.cy - 2 * roi.r,
                roi.cx + 2 * roi.r,
                roi.cy + 2 * roi.r,
            )
            option_baseline, _ = crop_with_clamp(baseline_patch_bgr, (
                option_bbox[0] - clamped_bbox[0],
                option_bbox[1] - clamped_bbox[1],
                option_bbox[2] - clamped_bbox[0],
                option_bbox[3] - clamped_bbox[1],
            ))
            option_raw, _ = crop_with_clamp(raw_patch_bgr, (
                option_bbox[0] - clamped_bbox[0],
                option_bbox[1] - clamped_bbox[1],
                option_bbox[2] - clamped_bbox[0],
                option_bbox[3] - clamped_bbox[1],
            ))
            save_image(question_dir / f"option_{roi.option}_baseline.png", option_baseline)
            save_image(question_dir / f"option_{roi.option}_raw_native.png", option_raw)
            option_reports.append(
                {
                    "option": int(roi.option),
                    "baseline_score": float(baseline_scores[(question_1_based, roi.option)]),
                    "raw_native_score": float(raw_scores[(question_1_based, roi.option)]),
                    "delta": float(raw_scores[(question_1_based, roi.option)] - baseline_scores[(question_1_based, roi.option)]),
                }
            )

        item = {
            "image": image_rel,
            "question_1_based": question_1_based,
            "expected_answer": target["expected"],
            "raw_native_answer": target["actual"],
            "baseline_summary": baseline_summary,
            "raw_native_summary": raw_summary,
            "phase_shift": {
                "dx": dx,
                "dy": dy,
                "magnitude": shift_mag,
                "response": response,
            },
            "patch_mad": patch_mad,
            "classification": classification,
            "option_score_deltas": option_reports,
            "artifacts": {
                "side_by_side": str((question_dir / "side_by_side.png").relative_to(REPO_ROOT)),
                "baseline_overlay": str((question_dir / "baseline_overlay.png").relative_to(REPO_ROOT)),
                "raw_native_overlay": str((question_dir / "raw_native_overlay.png").relative_to(REPO_ROOT)),
            },
        }
        report["targets"].append(item)
        with (question_dir / "diagnostic.json").open("w", encoding="utf-8") as handle:
            json.dump(item, handle, indent=2)

        print(
            "[MISMATCH-DIAG]",
            image_rel,
            f"q={question_1_based}",
            f"classification={classification}",
            f"shift={shift_mag:.2f}px",
            f"patch_mad={patch_mad:.2f}",
            f"baseline={baseline_summary['detected_answer']}",
            f"raw_native={raw_summary['detected_answer']}",
        )

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("[DONE] mismatch diagnostics written to", OUTPUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
