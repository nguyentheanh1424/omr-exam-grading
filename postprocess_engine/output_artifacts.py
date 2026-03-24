from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class BubbleFieldOutputConfig:
    enabled: bool = True
    overlay_image: bool = True
    values_json: bool = True


@dataclass(frozen=True)
class HandwrittenOutputConfig:
    enabled: bool = True
    save_patches: bool = True
    save_merged_template: bool = True
    save_ink_mask: bool = True
    save_aligned_source_img: bool = True
    save_aligned_source_regions: bool = True
    save_template_merged_img: bool = True
    save_template_merged_regions: bool = True
    save_scored_img: bool = True


@dataclass(frozen=True)
class PipelineOutputConfig:
    debug_intermediate: bool = True
    summary_json: bool = True
    scored_image: bool = True
    bubble_fields: BubbleFieldOutputConfig = BubbleFieldOutputConfig()
    handwritten_review: HandwrittenOutputConfig = HandwrittenOutputConfig()


def load_pipeline_output_config(path: str | Path) -> PipelineOutputConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("pipeline output config must be an object")

    bubble_fields = data.get("bubble_fields", {})
    handwritten_review = data.get("handwritten_review", {})
    if not isinstance(bubble_fields, dict):
        raise ValueError("bubble_fields config must be an object")
    if not isinstance(handwritten_review, dict):
        raise ValueError("handwritten_review config must be an object")

    return PipelineOutputConfig(
        debug_intermediate=bool(data.get("debug_intermediate", True)),
        summary_json=bool(data.get("summary_json", True)),
        scored_image=bool(data.get("scored_image", True)),
        bubble_fields=BubbleFieldOutputConfig(
            enabled=bool(bubble_fields.get("enabled", True)),
            overlay_image=bool(bubble_fields.get("overlay_image", True)),
            values_json=bool(bubble_fields.get("values_json", True)),
        ),
        handwritten_review=HandwrittenOutputConfig(
            enabled=bool(handwritten_review.get("enabled", True)),
            save_patches=bool(handwritten_review.get("save_patches", True)),
            save_merged_template=bool(handwritten_review.get("save_merged_template", True)),
            save_ink_mask=bool(handwritten_review.get("save_ink_mask", True)),
            save_aligned_source_img=bool(handwritten_review.get("save_aligned_source_img", True)),
            save_aligned_source_regions=bool(handwritten_review.get("save_aligned_source_regions", True)),
            save_template_merged_img=bool(handwritten_review.get("save_template_merged_img", True)),
            save_template_merged_regions=bool(handwritten_review.get("save_template_merged_regions", True)),
            save_scored_img=bool(handwritten_review.get("save_scored_img", True)),
        ),
    )


def save_json_if_enabled(enabled: bool, path: str | Path, payload: Any) -> None:
    if not enabled:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_image_if_enabled(enabled: bool, path: str | Path, image: np.ndarray | None) -> None:
    if not enabled or image is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), image)
