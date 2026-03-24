from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv

from native_core.python_adapter import DEFAULT_ANSWER_KEY_JSON, DEFAULT_BUBBLE_LAYOUT_JSON, load_answer_key
from orm_engine.orm import OMRProcessor, load_circle_rois
from postprocess_engine.handwritten_review import (
    HandwrittenRegion,
    build_manifest,
    build_review_composite,
    crop_handwritten_regions,
    draw_region_overlays,
    load_handwritten_regions,
    merge_handwritten_regions,
    save_handwritten_outputs,
)
from warp_engine.config import TEMPLATE_MARKER_POSITIONS_FILE
from warp_engine.engine import WarpEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "samples"
DEFAULT_TEMPLATE_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "handwritten_regions.json"
OUTPUT_DIR = REPO_ROOT / "results" / "handwritten_review_batch"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(SAMPLES_DIR.glob("1photo*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No sample images found under {SAMPLES_DIR}")

    warp_engine = WarpEngine(str(REPO_ROOT / TEMPLATE_MARKER_POSITIONS_FILE), str(DEFAULT_TEMPLATE_IMAGE))
    regions = load_handwritten_regions(DEFAULT_REGIONS_JSON)
    circle_rois = load_circle_rois(str(REPO_ROOT / DEFAULT_BUBBLE_LAYOUT_JSON))
    answer_key = load_answer_key(
        REPO_ROOT / DEFAULT_ANSWER_KEY_JSON,
        fallback_question_count=max(roi.question for roi in circle_rois),
    )
    omr = OMRProcessor(circle_rois=circle_rois, answer_key=answer_key)

    summary: dict[str, list[dict]] = {"images": []}

    for image_path in image_paths:
        raw = cv.imread(str(image_path), cv.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(image_path)

        image_output_dir = OUTPUT_DIR / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = warp_engine.warp_with_artifacts(
            raw,
            output=None,
            use_global_idw=False,
            use_region_refine=True,
            debug=False,
        )
        scored = omr.run(artifacts.template_merged_img, output=None, debug=False)["scored_img"]
        patches = crop_handwritten_regions(artifacts.aligned_source_img, regions)
        merged = merge_handwritten_regions(artifacts.template_merged_img, patches, regions)
        ink_mask_regions = [
            HandwrittenRegion(
                id=region.id,
                label=region.label,
                rect=region.rect,
                padding_px=region.padding_px,
                merge_mode="ink_mask",
                save_patch=region.save_patch,
            )
            for region in regions
        ]
        merged_ink_mask = merge_handwritten_regions(
            artifacts.template_merged_img,
            patches,
            ink_mask_regions,
        )
        manifest = build_manifest(str(image_path.relative_to(REPO_ROOT)), patches)
        save_handwritten_outputs(
            image_output_dir,
            patches,
            merged,
            manifest,
            merged_ink_mask_img=merged_ink_mask,
        )

        side_by_side = build_review_composite(
            template_img=artifacts.template_merged_img,
            scored_img=scored,
            merged_handwritten_img=merged,
            mode="side_by_side",
        )
        cv.imwrite(str(image_output_dir / "aligned_source_img.png"), artifacts.aligned_source_img)
        cv.imwrite(str(image_output_dir / "aligned_source_regions.png"), draw_region_overlays(artifacts.aligned_source_img, regions))
        cv.imwrite(str(image_output_dir / "template_merged_img.png"), artifacts.template_merged_img)
        cv.imwrite(str(image_output_dir / "template_merged_regions.png"), draw_region_overlays(artifacts.template_merged_img, regions))
        cv.imwrite(str(image_output_dir / "scored_img.png"), scored)
        cv.imwrite(str(image_output_dir / "review_side_by_side.png"), side_by_side)
        cv.imwrite(str(image_output_dir / "review_merged_template_ink_mask_preview.png"), merged_ink_mask)

        summary["images"].append(
            {
                "image": str(image_path.relative_to(REPO_ROOT)),
                "region_count": len(regions),
                "patch_count": len(patches),
                "output_dir": str(image_output_dir.relative_to(REPO_ROOT)),
                "manifest": str((image_output_dir / "review_manifest.json").relative_to(REPO_ROOT)),
            }
        )
        print("[HANDWRITTEN-BATCH]", image_path.name, f"patches={len(patches)}")

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[DONE] handwritten review batch written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
