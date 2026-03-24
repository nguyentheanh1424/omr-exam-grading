from __future__ import annotations

import json
import os
from pathlib import Path

import cv2 as cv

from native_core.native_api import NativeCoreClient, read_image
from native_core.python_adapter import build_native_adapter_config, load_threshold_config
from warp_engine.detector import detect_tags


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "results" / "marker_compare"
TARGET_IMAGES = [
    REPO_ROOT / "samples" / "1photo5.jpg",
    REPO_ROOT / "samples" / "1photo6.jpg",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dll_path = os.environ.get("OMR_DLL_PATH")
    client = NativeCoreClient(dll_path=dll_path) if dll_path else NativeCoreClient()
    config = build_native_adapter_config()
    thresholds = load_threshold_config()

    report = {"images": []}
    for image_path in TARGET_IMAGES:
        image_name = image_path.stem
        raw = read_image(image_path)

        global_debug_dir = REPO_ROOT / "results" / "native_global_debug"
        global_debug_dir.mkdir(parents=True, exist_ok=True)

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

        global_debug_path = global_debug_dir / "global_h_debug.json"
        with global_debug_path.open("r", encoding="utf-8") as handle:
            native_debug = json.load(handle)

        gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        python_detections = {int(det.id): det for det in detect_tags(gray)}

        marker_rows = []
        for native_marker in native_debug["markers"]:
            marker_id = int(native_marker["id"])
            py_det = python_detections.get(marker_id)
            if py_det is None:
                marker_rows.append(
                    {
                        "id": marker_id,
                        "python_detected": False,
                        "native_src": native_marker["src"],
                        "native_reprojection_error": native_marker["reprojection_error"],
                        "is_inlier": native_marker["is_inlier"],
                    }
                )
                continue

            py_center = [float(py_det.center[0]), float(py_det.center[1])]
            dx = native_marker["src"][0] - py_center[0]
            dy = native_marker["src"][1] - py_center[1]
            marker_rows.append(
                {
                    "id": marker_id,
                    "python_detected": True,
                    "native_src": native_marker["src"],
                    "python_center": py_center,
                    "center_delta_px": (dx * dx + dy * dy) ** 0.5,
                    "native_reprojection_error": native_marker["reprojection_error"],
                    "is_inlier": native_marker["is_inlier"],
                }
            )

        image_report = {
            "image": str(image_path.relative_to(REPO_ROOT)),
            "python_detected_count": len(python_detections),
            "native_corr_count": len(native_debug["markers"]),
            "best_inliers": native_debug["best_inliers"],
            "markers": sorted(marker_rows, key=lambda item: item["id"]),
        }
        report["images"].append(image_report)

        with (OUTPUT_DIR / f"{image_name}_marker_compare.json").open("w", encoding="utf-8") as handle:
            json.dump(image_report, handle, indent=2)

        print("[MARKER-COMPARE]", image_report["image"], f"inliers={image_report['best_inliers']}")
        for row in image_report["markers"]:
            print(
                "  id",
                row["id"],
                "delta=",
                f"{row.get('center_delta_px', -1.0):.3f}",
                "repr=",
                f"{row['native_reprojection_error']:.3f}",
                "inlier=",
                row["is_inlier"],
            )

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("[DONE] marker compare written to", OUTPUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
