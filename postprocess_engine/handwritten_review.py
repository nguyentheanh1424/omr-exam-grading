from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np
from warp_engine.binarize import binarize_patch_dual


@dataclass(frozen=True)
class HandwrittenRegion:
    id: str
    label: str
    rect: tuple[int, int, int, int]
    padding_px: int = 0
    merge_mode: str = "replace_rect"
    save_patch: bool = True


@dataclass
class HandwrittenPatch:
    region_id: str
    label: str
    rect: tuple[int, int, int, int]
    image: np.ndarray
    save_patch: bool = True


OVERLAY_COLOR = (0, 140, 255)
OVERLAY_TEXT_COLOR = (255, 255, 255)
OVERLAY_THICKNESS = 2
OVERLAY_FONT = cv.FONT_HERSHEY_SIMPLEX
OVERLAY_FONT_SCALE = 0.7
OVERLAY_TEXT_THICKNESS = 2
SUPPORTED_MERGE_MODES = {"replace_rect", "ink_mask"}


def _normalize_rect(rect: Iterable[int]) -> tuple[int, int, int, int]:
    values = tuple(int(v) for v in rect)
    if len(values) != 4:
        raise ValueError("rect must contain exactly 4 integers")
    x0, y0, x1, y1 = values
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid rect: {values}")
    return values


def load_handwritten_regions(path: str | Path) -> list[HandwrittenRegion]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError("handwritten region config must be a non-empty list")

    regions: list[HandwrittenRegion] = []
    seen_ids: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each handwritten region entry must be an object")
        region = HandwrittenRegion(
            id=str(item["id"]),
            label=str(item.get("label", item["id"])),
            rect=_normalize_rect(item["rect"]),
            padding_px=max(0, int(item.get("padding_px", 0))),
            merge_mode=str(item.get("merge_mode", "replace_rect")),
            save_patch=bool(item.get("save_patch", True)),
        )
        if not region.id.strip():
            raise ValueError("handwritten region id must not be empty")
        if not region.label.strip():
            raise ValueError(f"handwritten region label must not be empty: {region.id}")
        if region.id in seen_ids:
            raise ValueError(f"duplicate handwritten region id: {region.id}")
        if region.merge_mode not in SUPPORTED_MERGE_MODES:
            raise ValueError(f"unsupported merge_mode for v1 prototype: {region.merge_mode}")
        seen_ids.add(region.id)
        regions.append(region)
    return regions


def validate_handwritten_regions(
    regions: Iterable[HandwrittenRegion],
    image_shape: tuple[int, ...],
) -> list[HandwrittenRegion]:
    if len(image_shape) < 2:
        raise ValueError("image_shape must include height and width")
    height, width = image_shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("image_shape must be positive")

    validated = list(regions)
    seen_ids: set[str] = set()
    for region in validated:
        if region.id in seen_ids:
            raise ValueError(f"duplicate handwritten region id: {region.id}")
        if region.merge_mode not in SUPPORTED_MERGE_MODES:
            raise ValueError(f"unsupported merge_mode for region {region.id}: {region.merge_mode}")
        x0, y0, x1, y1 = region.rect
        if x1 > width or y1 > height:
            raise ValueError(
                f"handwritten region {region.id} exceeds image bounds: "
                f"rect={region.rect}, image_shape={(height, width)}"
            )
        seen_ids.add(region.id)
    return validated


def _apply_padding(
    rect: tuple[int, int, int, int],
    padding_px: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    return (
        max(0, x0 - padding_px),
        max(0, y0 - padding_px),
        min(width, x1 + padding_px),
        min(height, y1 + padding_px),
    )


def crop_handwritten_regions(
    aligned_source_img: np.ndarray,
    regions: Iterable[HandwrittenRegion],
) -> list[HandwrittenPatch]:
    if aligned_source_img is None:
        raise ValueError("aligned_source_img must not be None")
    height, width = aligned_source_img.shape[:2]
    validated_regions = validate_handwritten_regions(regions, aligned_source_img.shape)
    patches: list[HandwrittenPatch] = []
    for region in validated_regions:
        x0, y0, x1, y1 = _apply_padding(region.rect, region.padding_px, width, height)
        patch = aligned_source_img[y0:y1, x0:x1].copy()
        patches.append(
            HandwrittenPatch(
                region_id=region.id,
                label=region.label,
                rect=(x0, y0, x1, y1),
                image=patch,
                save_patch=region.save_patch,
            )
        )
    return patches


def merge_handwritten_regions(
    base_img: np.ndarray,
    patches: Iterable[HandwrittenPatch],
    regions: Iterable[HandwrittenRegion] | None = None,
) -> np.ndarray:
    if base_img is None:
        raise ValueError("base_img must not be None")
    merged = base_img.copy()
    validated_regions = validate_handwritten_regions(regions or [], base_img.shape)
    region_by_id = {region.id: region for region in validated_regions}
    for patch in patches:
        x0, y0, x1, y1 = patch.rect
        expected_h = y1 - y0
        expected_w = x1 - x0
        if patch.image.shape[0] != expected_h or patch.image.shape[1] != expected_w:
            raise ValueError(f"patch shape mismatch for region {patch.region_id}")
        region = region_by_id.get(patch.region_id)
        merge_mode = region.merge_mode if region is not None else "replace_rect"
        if merge_mode == "replace_rect":
            merged[y0:y1, x0:x1] = patch.image
            continue

        if merge_mode == "ink_mask":
            base_patch = merged[y0:y1, x0:x1]
            ink_mask = binarize_patch_dual(patch.image)
            if base_patch.ndim == 2:
                base_patch[ink_mask] = 0
            else:
                base_patch[ink_mask] = (0, 0, 0)
            merged[y0:y1, x0:x1] = base_patch
            continue

        raise ValueError(f"unsupported merge_mode during merge: {merge_mode}")
    return merged


def build_review_composite(
    template_img: np.ndarray,
    scored_img: np.ndarray | None,
    merged_handwritten_img: np.ndarray,
    mode: str = "template",
) -> np.ndarray:
    if mode == "template":
        return merged_handwritten_img.copy()
    if mode == "scored":
        if scored_img is None:
            raise ValueError("scored_img is required for mode='scored'")
        return scored_img.copy()
    if mode == "side_by_side":
        right = scored_img if scored_img is not None else merged_handwritten_img
        return np.concatenate([template_img, right], axis=1)
    raise ValueError(f"unsupported review composite mode: {mode}")


def draw_region_overlays(
    base_img: np.ndarray,
    regions: Iterable[HandwrittenRegion],
) -> np.ndarray:
    overlay = base_img.copy()
    height, width = overlay.shape[:2]
    for region in validate_handwritten_regions(regions, overlay.shape):
        x0, y0, x1, y1 = _apply_padding(region.rect, region.padding_px, width, height)
        cv.rectangle(overlay, (x0, y0), (x1, y1), OVERLAY_COLOR, OVERLAY_THICKNESS)
        label = f"{region.label} [{region.merge_mode}]"
        text_origin = (x0 + 6, max(24, y0 - 8))
        cv.putText(
            overlay,
            label,
            text_origin,
            OVERLAY_FONT,
            OVERLAY_FONT_SCALE,
            (0, 0, 0),
            OVERLAY_TEXT_THICKNESS + 2,
            cv.LINE_AA,
        )
        cv.putText(
            overlay,
            label,
            text_origin,
            OVERLAY_FONT,
            OVERLAY_FONT_SCALE,
            OVERLAY_TEXT_COLOR,
            OVERLAY_TEXT_THICKNESS,
            cv.LINE_AA,
        )
    return overlay


def save_handwritten_outputs(
    output_dir: str | Path,
    patches: Iterable[HandwrittenPatch],
    merged_img: np.ndarray,
    manifest: dict,
    *,
    merged_ink_mask_img: np.ndarray | None = None,
    save_merged_template: bool = True,
    save_ink_mask: bool = True,
    save_manifest: bool = True,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if save_merged_template:
        cv.imwrite(str(output_path / "review_merged_template.png"), merged_img)
    if save_ink_mask and merged_ink_mask_img is not None:
        cv.imwrite(str(output_path / "review_merged_template_ink_mask.png"), merged_ink_mask_img)
    for patch in patches:
        if patch.save_patch:
            cv.imwrite(str(output_path / f"review_region_{patch.region_id}.png"), patch.image)
    if save_manifest:
        with (output_path / "review_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)


def build_manifest(
    image_path: str,
    patches: Iterable[HandwrittenPatch],
) -> dict:
    patch_entries = []
    for patch in patches:
        patch_entries.append(
            {
                "region_id": patch.region_id,
                "label": patch.label,
                "rect": list(patch.rect),
                "file": f"review_region_{patch.region_id}.png" if patch.save_patch else None,
                "save_patch": patch.save_patch,
            }
        )
    return {
        "input": image_path,
        "patches": patch_entries,
        "merged_template_file": "review_merged_template.png",
        "merged_template_ink_mask_file": "review_merged_template_ink_mask.png",
    }
