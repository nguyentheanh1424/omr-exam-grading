from __future__ import annotations

try:
    from native_core.tests._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np

from native_core.native_api import NativeCoreClient, read_image
from native_core.python_adapter import build_native_adapter_config, load_threshold_config


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "results" / "global_h_compare"
TARGET_IMAGES = [
    REPO_ROOT / "samples" / "1photo5.jpg",
    REPO_ROOT / "samples" / "1photo6.jpg",
]


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    mapped = (h @ pts_h.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return mapped


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dll_path = os.environ.get("OMR_DLL_PATH")
    client = NativeCoreClient(dll_path=dll_path) if dll_path else NativeCoreClient()
    config = build_native_adapter_config()
    thresholds = load_threshold_config()

    summary = {"images": []}
    for image_path in TARGET_IMAGES:
        raw = read_image(image_path)
        client.run(
            raw,
            config,
            assume_aligned_input=False,
            return_scored_image=False,
            use_global_idw=False,
            use_region_refine=False,
            debug_level=1,
            abs_th=thresholds.abs_th,
            rel_th=thresholds.rel_th,
            auto_threshold=thresholds.auto_threshold,
        )

        debug_path = REPO_ROOT / "results" / "native_global_debug" / "global_h_debug.json"
        with debug_path.open("r", encoding="utf-8") as handle:
            native_debug = json.load(handle)

        src = np.asarray([row["src"] for row in native_debug["markers"]], dtype=np.float32)
        dst = np.asarray([row["dst"] for row in native_debug["markers"]], dtype=np.float32)
        native_h = np.asarray(native_debug["homography"], dtype=np.float64).reshape(3, 3)

        cv_h, cv_mask = cv.findHomography(src, dst, cv.RANSAC, 2.0)
        if cv_h is None:
            raise RuntimeError(f"cv.findHomography failed for {image_path}")
        lmeds_h, lmeds_mask = cv.findHomography(src, dst, cv.LMEDS)
        all_h, _ = cv.findHomography(src, dst, 0)
        if lmeds_h is None or all_h is None:
            raise RuntimeError(f"additional homography fit failed for {image_path}")

        native_mapped = apply_h(native_h.astype(np.float32), src)
        cv_mapped = apply_h(cv_h.astype(np.float32), src)
        lmeds_mapped = apply_h(lmeds_h.astype(np.float32), src)
        all_mapped = apply_h(all_h.astype(np.float32), src)
        native_err = np.linalg.norm(native_mapped - dst, axis=1)
        cv_err = np.linalg.norm(cv_mapped - dst, axis=1)
        lmeds_err = np.linalg.norm(lmeds_mapped - dst, axis=1)
        all_err = np.linalg.norm(all_mapped - dst, axis=1)

        image_report = {
            "image": str(image_path.relative_to(REPO_ROOT)),
            "native": {
                "best_inliers": native_debug["best_inliers"],
                "mean_reprojection_error": float(native_err.mean()),
                "median_reprojection_error": float(np.median(native_err)),
                "max_reprojection_error": float(native_err.max()),
            },
            "opencv": {
                "inliers": int(cv_mask.sum()) if cv_mask is not None else 0,
                "mean_reprojection_error": float(cv_err.mean()),
                "median_reprojection_error": float(np.median(cv_err)),
                "max_reprojection_error": float(cv_err.max()),
            },
            "opencv_lmeds": {
                "inliers": int(lmeds_mask.sum()) if lmeds_mask is not None else 0,
                "mean_reprojection_error": float(lmeds_err.mean()),
                "median_reprojection_error": float(np.median(lmeds_err)),
                "max_reprojection_error": float(lmeds_err.max()),
            },
            "opencv_all_points": {
                "mean_reprojection_error": float(all_err.mean()),
                "median_reprojection_error": float(np.median(all_err)),
                "max_reprojection_error": float(all_err.max()),
            },
            "markers": [],
        }
        for idx, marker in enumerate(native_debug["markers"]):
            image_report["markers"].append(
                {
                    "id": marker["id"],
                    "native_error": float(native_err[idx]),
                    "opencv_error": float(cv_err[idx]),
                    "opencv_lmeds_error": float(lmeds_err[idx]),
                    "opencv_all_points_error": float(all_err[idx]),
                    "native_inlier": bool(marker["is_inlier"]),
                    "opencv_inlier": bool(cv_mask[idx][0]) if cv_mask is not None else False,
                    "opencv_lmeds_inlier": bool(lmeds_mask[idx][0]) if lmeds_mask is not None else False,
                }
            )

        summary["images"].append(image_report)
        with (OUTPUT_DIR / f"{image_path.stem}_global_h_compare.json").open("w", encoding="utf-8") as handle:
            json.dump(image_report, handle, indent=2)

        print(
            "[GLOBAL-H-COMPARE]",
            image_report["image"],
            "native_mean=",
            f"{image_report['native']['mean_reprojection_error']:.3f}",
            "opencv_ransac_mean=",
            f"{image_report['opencv']['mean_reprojection_error']:.3f}",
            "opencv_lmeds_mean=",
            f"{image_report['opencv_lmeds']['mean_reprojection_error']:.3f}",
            "opencv_all_mean=",
            f"{image_report['opencv_all_points']['mean_reprojection_error']:.3f}",
            "native_inliers=",
            image_report["native"]["best_inliers"],
            "opencv_ransac_inliers=",
            image_report["opencv"]["inliers"],
        )

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[DONE] global H compare written to", OUTPUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
