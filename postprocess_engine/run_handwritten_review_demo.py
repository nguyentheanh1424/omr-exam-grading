from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv

from native_core.python_adapter import DEFAULT_ANSWER_KEY_JSON, DEFAULT_BUBBLE_LAYOUT_JSON, load_answer_key
from orm_engine.orm import OMRProcessor, load_circle_rois
from postprocess_engine.handwritten_review import (
    HandwrittenRegion,
    build_manifest,
    crop_handwritten_regions,
    draw_region_overlays,
    load_handwritten_regions,
    merge_handwritten_regions,
    save_handwritten_outputs,
)
from warp_engine.config import TEMPLATE_MARKER_POSITIONS_FILE
from warp_engine.engine import WarpEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"
DEFAULT_REGIONS_JSON = REPO_ROOT / "config" / "handwritten_regions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype handwritten review crop/merge pipeline.")
    parser.add_argument("--input", default="samples/1photo5.jpg", help="Raw input image path.")
    parser.add_argument(
        "--regions",
        default=str(DEFAULT_REGIONS_JSON),
        help="Path to handwritten regions JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/handwritten_review_demo",
        help="Directory for merged review image, patches, and manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_arg = Path(args.input)
    output_arg = Path(args.output_dir)
    image_path = input_arg if input_arg.is_absolute() else REPO_ROOT / input_arg
    output_dir = output_arg if output_arg.is_absolute() else REPO_ROOT / output_arg

    warp_engine = WarpEngine(str(REPO_ROOT / TEMPLATE_MARKER_POSITIONS_FILE), str(DEFAULT_TEMPLATE_IMAGE))
    raw = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(image_path)

    artifacts = warp_engine.warp_with_artifacts(
        raw,
        output=None,
        use_global_idw=False,
        use_region_refine=True,
        debug=False,
    )

    circle_rois = load_circle_rois(str(REPO_ROOT / DEFAULT_BUBBLE_LAYOUT_JSON))
    answer_key = load_answer_key(
        REPO_ROOT / DEFAULT_ANSWER_KEY_JSON,
        fallback_question_count=max(roi.question for roi in circle_rois),
    )
    omr = OMRProcessor(circle_rois=circle_rois, answer_key=answer_key)
    scored = omr.run(artifacts.template_merged_img, output=None, debug=False)["scored_img"]

    regions_arg = Path(args.regions)
    regions = load_handwritten_regions(regions_arg if regions_arg.is_absolute() else REPO_ROOT / regions_arg)
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
    merged_ink_mask = merge_handwritten_regions(artifacts.template_merged_img, patches, ink_mask_regions)
    manifest = build_manifest(str(Path(args.input)), patches)
    save_handwritten_outputs(
        output_dir,
        patches,
        merged,
        manifest,
        merged_ink_mask_img=merged_ink_mask,
    )

    cv.imwrite(str(output_dir / "aligned_source_img.png"), artifacts.aligned_source_img)
    cv.imwrite(str(output_dir / "aligned_source_regions.png"), draw_region_overlays(artifacts.aligned_source_img, regions))
    cv.imwrite(str(output_dir / "template_merged_img.png"), artifacts.template_merged_img)
    cv.imwrite(str(output_dir / "template_merged_regions.png"), draw_region_overlays(artifacts.template_merged_img, regions))
    cv.imwrite(str(output_dir / "scored_img.png"), scored)

    print("[OK] handwritten review demo completed")
    print("input:", image_path)
    print("regions:", len(regions))
    print("patches:", len(patches))
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
