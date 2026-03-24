from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv

from native_core.native_api import NativeCoreClient, read_image
from native_core.python_adapter import (
    DEFAULT_ANSWER_KEY_JSON,
    DEFAULT_CIRCLE_ROIS_JSON,
    build_native_adapter_config,
    load_answer_key,
    load_threshold_config,
    summarize_adapter_config,
)
from orm_engine.orm import OMRProcessor, load_circle_rois
from postprocess_engine.handwritten_review import (
    HandwrittenRegion,
    HandwrittenPatch,
    build_manifest,
    crop_handwritten_regions,
    draw_region_overlays,
    load_handwritten_regions,
    merge_handwritten_regions,
    save_handwritten_outputs,
)
from postprocess_engine.bubble_field_reader import (
    BubbleFieldResult,
    build_bubble_field_manifest,
    draw_bubble_field_overlay,
    load_bubble_field_configs,
)
from postprocess_engine.output_artifacts import (
    load_pipeline_output_config,
    save_image_if_enabled,
    save_json_if_enabled,
)
from warp_engine.config import TEMPLATE_LAYOUT_FILE
from warp_engine.detector import detect_tags
from warp_engine.engine import WarpEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_IMAGE = REPO_ROOT / "samples" / "template_scan1.png"
DEFAULT_HANDWRITTEN_REGIONS_JSON = REPO_ROOT / "config" / "handwritten_regions.json"
DEFAULT_ID_BUBBLE_FIELDS_JSON = REPO_ROOT / "config" / "id_bubble_fields.json"
DEFAULT_OUTPUT_CONFIG_JSON = REPO_ROOT / "config" / "pipeline_outputs.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native OMR core on a real repo asset.")
    parser.add_argument("--input", default="samples/1photo5.jpg", help="Path to the input image.")
    parser.add_argument(
        "--mode",
        choices=("raw", "aligned"),
        default="raw",
        help="Use raw photo input with native warp, or aligned input if the image is already warped.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/native_harness",
        help="Directory for scored image and JSON summary.",
    )
    parser.add_argument(
        "--use-global-idw",
        action="store_true",
        help="Enable the optional global IDW refinement inside native warp.",
    )
    parser.add_argument(
        "--disable-region-refine",
        action="store_true",
        help="Disable optional region refine inside native warp.",
    )
    parser.add_argument(
        "--use-python-markers",
        action="store_true",
        help="Provide detected markers from Python aruco detection instead of native auto-detect.",
    )
    parser.add_argument(
        "--python-warp-first",
        action="store_true",
        help="Warp with the current Python pipeline first, then send the aligned image to native grading.",
    )
    parser.add_argument(
        "--python-prep-gray-first",
        action="store_true",
        help="Apply Python OMR grayscale preprocessing before native grading. Implies aligned input.",
    )
    parser.add_argument(
        "--with-handwritten-review",
        action="store_true",
        help="Also export handwritten review artifacts from the Python-aligned image.",
    )
    parser.add_argument(
        "--handwritten-regions",
        default=str(DEFAULT_HANDWRITTEN_REGIONS_JSON),
        help="Path to handwritten regions JSON used when --with-handwritten-review is enabled.",
    )
    parser.add_argument(
        "--with-id-values",
        action="store_true",
        help="Read config-driven bubble fields such as Student ID and Quiz ID from the aligned image.",
    )
    parser.add_argument(
        "--id-field-config",
        default=str(DEFAULT_ID_BUBBLE_FIELDS_JSON),
        help="Path to ID bubble field config JSON used when --with-id-values is enabled.",
    )
    parser.add_argument(
        "--output-config",
        default=str(DEFAULT_OUTPUT_CONFIG_JSON),
        help="Path to unified pipeline output artifact config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_config_arg = Path(args.output_config)
    output_config_path = output_config_arg if output_config_arg.is_absolute() else REPO_ROOT / output_config_arg
    output_config = load_pipeline_output_config(output_config_path)

    config = build_native_adapter_config()
    thresholds = load_threshold_config()
    client = NativeCoreClient()
    img = read_image(args.input)
    raw_img = img.copy()
    effective_mode = args.mode
    handwritten_artifacts = None
    bubble_field_configs = None
    field_path = None
    if args.with_id_values or output_config.bubble_fields.enabled:
        field_arg = Path(args.id_field_config)
        field_path = field_arg if field_arg.is_absolute() else REPO_ROOT / field_arg
        bubble_field_configs = load_bubble_field_configs(field_path)
    if args.python_warp_first or args.with_handwritten_review or args.with_id_values or output_config.handwritten_review.enabled or output_config.bubble_fields.enabled:
        warp_engine = WarpEngine(str(REPO_ROOT / TEMPLATE_LAYOUT_FILE), str(DEFAULT_TEMPLATE_IMAGE))
        handwritten_artifacts = warp_engine.warp_with_artifacts(
            raw_img,
            output=None,
            use_global_idw=False,
            use_region_refine=True,
            debug=False,
        )
    if args.python_warp_first:
        img = handwritten_artifacts.template_merged_img.copy()
        effective_mode = "aligned"
    if args.python_prep_gray_first:
        circle_rois = load_circle_rois(DEFAULT_CIRCLE_ROIS_JSON)
        answer_key = load_answer_key(
            DEFAULT_ANSWER_KEY_JSON,
            fallback_question_count=max(roi.question for roi in circle_rois),
        )
        omr = OMRProcessor(circle_rois=circle_rois, answer_key=answer_key)
        img = omr._prep_gray(img)
        effective_mode = "aligned"
    detected_markers = None
    if args.use_python_markers and not args.python_warp_first and not args.python_prep_gray_first:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        detected_markers = [
            (int(det.id), float(det.center[0]), float(det.center[1]))
            for det in detect_tags(gray)
        ]

    result = client.run(
        img,
        config,
        assume_aligned_input=(effective_mode == "aligned"),
        return_scored_image=True,
        use_global_idw=args.use_global_idw,
        use_region_refine=(not args.disable_region_refine) and not args.python_warp_first,
        abs_th=thresholds.abs_th,
        rel_th=thresholds.rel_th,
        auto_threshold=thresholds.auto_threshold,
        detected_markers=detected_markers,
        bubble_field_configs=bubble_field_configs,
    )

    summary = {
        "input": str(Path(args.input)),
        "mode": effective_mode,
        "python_warp_first": args.python_warp_first,
        "python_prep_gray_first": args.python_prep_gray_first,
        "adapter": summarize_adapter_config(config),
        "score": result.score,
        "graded_questions": result.graded_questions,
        "total_questions": result.total_questions,
        "used_abs_th": result.used_abs_th,
        "used_rel_th": result.used_rel_th,
        "configured_abs_th": thresholds.abs_th,
        "configured_rel_th": thresholds.rel_th,
        "answers": result.answers,
        "detected_marker_count": 0 if detected_markers is None else len(detected_markers),
        "marker_source": "python" if detected_markers is not None else "native",
        "student_id": None,
        "quiz_id": None,
    }

    summary_path = output_dir / "native_result.json"
    save_json_if_enabled(output_config.summary_json, summary_path, summary)

    save_image_if_enabled(output_config.scored_image, output_dir / "native_scored.png", result.scored_image)

    if args.with_id_values or output_config.bubble_fields.enabled:
        if handwritten_artifacts is None:
            raise RuntimeError("aligned artifacts were not prepared for ID bubble reading")
        field_results = []
        for field in bubble_field_configs or []:
            selected_rows = tuple((result.bubble_field_selected_rows or {}).get(field.id, [None] * field.n_cols))
            selected_values = tuple(
                None if row is None else field.row_values[row]
                for row in selected_rows
            )
            field_results.append(
                BubbleFieldResult(
                    field_id=field.id,
                    label=field.label,
                    decoded_value=(result.bubble_field_values or {}).get(field.id, "?" * field.n_cols),
                    selected_rows=selected_rows,
                    selected_values=selected_values,
                    is_complete=all(value is not None for value in selected_values),
                    threshold_abs=thresholds.abs_th if field.abs_th is None else field.abs_th,
                    threshold_rel=thresholds.rel_th if field.rel_th is None else field.rel_th,
                )
            )
        field_dir = output_dir / "bubble_fields"
        field_dir.mkdir(parents=True, exist_ok=True)
        field_overlay = draw_bubble_field_overlay(
            handwritten_artifacts.aligned_source_img,
            bubble_field_configs or [],
            field_results,
        )
        save_image_if_enabled(
            output_config.bubble_fields.overlay_image,
            field_dir / "aligned_bubble_fields.png",
            field_overlay,
        )
        field_manifest = build_bubble_field_manifest(field_results)
        save_json_if_enabled(
            output_config.bubble_fields.values_json,
            field_dir / "bubble_field_values.json",
            field_manifest,
        )
        summary["student_id"] = (result.bubble_field_values or {}).get("student_id")
        summary["quiz_id"] = (result.bubble_field_values or {}).get("quiz_id")
        summary["bubble_fields"] = {
            "enabled": output_config.bubble_fields.enabled,
            "config_path": str(field_path),
            "output_dir": str(field_dir),
            "fields": field_manifest["fields"],
            "source": "native",
        }
        save_json_if_enabled(output_config.summary_json, summary_path, summary)

    if args.with_handwritten_review or output_config.handwritten_review.enabled:
        if handwritten_artifacts is None:
            raise RuntimeError("handwritten artifacts were not prepared")
        regions_arg = Path(args.handwritten_regions)
        regions_path = regions_arg if regions_arg.is_absolute() else REPO_ROOT / regions_arg
        regions = load_handwritten_regions(regions_path)
        patches = crop_handwritten_regions(handwritten_artifacts.aligned_source_img, regions)
        if not output_config.handwritten_review.save_patches:
            patches = [
                HandwrittenPatch(
                    region_id=patch.region_id,
                    label=patch.label,
                    rect=patch.rect,
                    image=patch.image,
                    save_patch=False,
                )
                for patch in patches
            ]
        merged = merge_handwritten_regions(handwritten_artifacts.template_merged_img, patches, regions)
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
            handwritten_artifacts.template_merged_img,
            patches,
            ink_mask_regions,
        )
        handwritten_dir = output_dir / "handwritten_review"
        manifest = build_manifest(str(Path(args.input)), patches)
        save_handwritten_outputs(
            handwritten_dir,
            patches,
            merged,
            manifest,
            merged_ink_mask_img=merged_ink_mask if output_config.handwritten_review.save_ink_mask else None,
            save_merged_template=output_config.handwritten_review.save_merged_template,
            save_ink_mask=output_config.handwritten_review.save_ink_mask,
            save_manifest=True,
        )
        save_image_if_enabled(
            output_config.handwritten_review.save_aligned_source_img,
            handwritten_dir / "aligned_source_img.png",
            handwritten_artifacts.aligned_source_img,
        )
        save_image_if_enabled(
            output_config.handwritten_review.save_aligned_source_regions,
            handwritten_dir / "aligned_source_regions.png",
            draw_region_overlays(handwritten_artifacts.aligned_source_img, regions),
        )
        save_image_if_enabled(
            output_config.handwritten_review.save_template_merged_img,
            handwritten_dir / "template_merged_img.png",
            handwritten_artifacts.template_merged_img,
        )
        save_image_if_enabled(
            output_config.handwritten_review.save_template_merged_regions,
            handwritten_dir / "template_merged_regions.png",
            draw_region_overlays(handwritten_artifacts.template_merged_img, regions),
        )
        save_image_if_enabled(
            output_config.handwritten_review.save_scored_img,
            handwritten_dir / "native_scored_img.png",
            result.scored_image,
        )
        summary["handwritten_review"] = {
            "enabled": output_config.handwritten_review.enabled,
            "regions_path": str(regions_path),
            "region_count": len(regions),
            "patch_count": len(patches),
            "output_dir": str(handwritten_dir),
            "manifest": str(handwritten_dir / "review_manifest.json"),
        }
        save_json_if_enabled(output_config.summary_json, summary_path, summary)

    print("[OK] native harness completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
