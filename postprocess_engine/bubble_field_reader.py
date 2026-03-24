from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np

from orm_engine.orm import OMRProcessor


FIELD_COLOR = (255, 180, 0)
SELECTED_COLOR = (0, 200, 0)
UNRESOLVED_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_SHADOW = (0, 0, 0)
FONT = cv.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class BubbleFieldConfig:
    id: str
    label: str
    origin: tuple[int, int]
    dx: int
    dy: int
    n_cols: int
    n_rows: int
    radius: int
    row_values: tuple[str, ...]
    abs_th: float | None = None
    rel_th: float | None = None


@dataclass(frozen=True)
class BubbleFieldCell:
    field_id: str
    column: int
    row: int
    cx: int
    cy: int
    radius: int
    value: str


@dataclass(frozen=True)
class BubbleFieldResult:
    field_id: str
    label: str
    decoded_value: str
    selected_rows: tuple[int | None, ...]
    selected_values: tuple[str | None, ...]
    is_complete: bool
    threshold_abs: float
    threshold_rel: float


def load_bubble_field_configs(path: str | Path) -> list[BubbleFieldConfig]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError("bubble field config must be a non-empty list")

    configs: list[BubbleFieldConfig] = []
    seen_ids: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each bubble field entry must be an object")
        field_id = str(item["id"]).strip()
        label = str(item.get("label", field_id)).strip()
        origin = tuple(int(v) for v in item["origin"])
        row_values = tuple(str(v) for v in item["row_values"])
        if len(origin) != 2:
            raise ValueError(f"origin must contain 2 integers: {field_id}")
        if not field_id or not label:
            raise ValueError("bubble field id/label must not be empty")
        if field_id in seen_ids:
            raise ValueError(f"duplicate bubble field id: {field_id}")

        n_cols = int(item["n_cols"])
        n_rows = int(item["n_rows"])
        dx = int(item["dx"])
        dy = int(item["dy"])
        radius = int(item["radius"])
        if n_cols <= 0 or n_rows <= 0:
            raise ValueError(f"bubble field dims must be > 0: {field_id}")
        if dx <= 0 or dy <= 0 or radius <= 0:
            raise ValueError(f"bubble field spacing/radius must be > 0: {field_id}")
        if len(row_values) != n_rows:
            raise ValueError(
                f"row_values length must equal n_rows for {field_id}: "
                f"{len(row_values)} != {n_rows}"
            )
        configs.append(
            BubbleFieldConfig(
                id=field_id,
                label=label,
                origin=(origin[0], origin[1]),
                dx=dx,
                dy=dy,
                n_cols=n_cols,
                n_rows=n_rows,
                radius=radius,
                row_values=row_values,
                abs_th=float(item["abs_th"]) if "abs_th" in item else None,
                rel_th=float(item["rel_th"]) if "rel_th" in item else None,
            )
        )
        seen_ids.add(field_id)
    return configs


def validate_bubble_field_configs(
    configs: Iterable[BubbleFieldConfig],
    image_shape: tuple[int, ...],
) -> list[BubbleFieldConfig]:
    if len(image_shape) < 2:
        raise ValueError("image_shape must include height and width")
    height, width = image_shape[:2]
    validated = list(configs)
    seen_ids: set[str] = set()
    for config in validated:
        if config.id in seen_ids:
            raise ValueError(f"duplicate bubble field id: {config.id}")
        max_x = config.origin[0] + (config.n_cols - 1) * config.dx
        max_y = config.origin[1] + (config.n_rows - 1) * config.dy
        if config.origin[0] - config.radius < 0 or config.origin[1] - config.radius < 0:
            raise ValueError(f"bubble field origin is out of bounds: {config.id}")
        if max_x + config.radius >= width or max_y + config.radius >= height:
            raise ValueError(
                f"bubble field grid exceeds image bounds: {config.id}, "
                f"bottom_right={(max_x, max_y)}, image_shape={(height, width)}"
            )
        seen_ids.add(config.id)
    return validated


def build_bubble_field_cells(configs: Iterable[BubbleFieldConfig]) -> list[BubbleFieldCell]:
    cells: list[BubbleFieldCell] = []
    for config in configs:
        ox, oy = config.origin
        for column in range(config.n_cols):
            for row in range(config.n_rows):
                cells.append(
                    BubbleFieldCell(
                        field_id=config.id,
                        column=column,
                        row=row,
                        cx=ox + column * config.dx,
                        cy=oy + row * config.dy,
                        radius=config.radius,
                        value=config.row_values[row],
                    )
                )
    return cells


def read_bubble_field_values(
    aligned_img: np.ndarray,
    configs: Iterable[BubbleFieldConfig],
    *,
    abs_th: float,
    rel_th: float,
) -> tuple[list[BubbleFieldResult], dict[tuple[str, int, int], float]]:
    if aligned_img is None:
        raise ValueError("aligned_img must not be None")
    validated = validate_bubble_field_configs(configs, aligned_img.shape)
    cells = build_bubble_field_cells(validated)
    gray = OMRProcessor._prep_gray(aligned_img)

    score_cache: dict[tuple[str, int, int], float] = {}
    for cell in cells:
        score_cache[(cell.field_id, cell.column, cell.row)] = OMRProcessor._bubble_score(
            gray,
            cell.cx,
            cell.cy,
            cell.radius,
        )

    results: list[BubbleFieldResult] = []
    for config in validated:
        selected_rows: list[int | None] = []
        selected_values: list[str | None] = []
        effective_abs = config.abs_th if config.abs_th is not None else abs_th
        effective_rel = config.rel_th if config.rel_th is not None else rel_th

        for column in range(config.n_cols):
            items = [
                (row, score_cache[(config.id, column, row)])
                for row in range(config.n_rows)
            ]
            items.sort(key=lambda item: item[1], reverse=True)
            best_row, best_val = items[0]
            second_val = items[1][1] if len(items) > 1 else -1e9
            if best_val >= effective_abs and (best_val - second_val) >= effective_rel:
                selected_rows.append(best_row)
                selected_values.append(config.row_values[best_row])
            else:
                selected_rows.append(None)
                selected_values.append(None)

        decoded_value = "".join(value if value is not None else "?" for value in selected_values)
        results.append(
            BubbleFieldResult(
                field_id=config.id,
                label=config.label,
                decoded_value=decoded_value,
                selected_rows=tuple(selected_rows),
                selected_values=tuple(selected_values),
                is_complete=all(value is not None for value in selected_values),
                threshold_abs=float(effective_abs),
                threshold_rel=float(effective_rel),
            )
        )

    return results, score_cache


def draw_bubble_field_overlay(
    base_img: np.ndarray,
    configs: Iterable[BubbleFieldConfig],
    results: Iterable[BubbleFieldResult],
) -> np.ndarray:
    overlay = base_img.copy()
    config_by_id = {config.id: config for config in configs}
    result_by_id = {result.field_id: result for result in results}
    cells = build_bubble_field_cells(config_by_id.values())
    for cell in cells:
        result = result_by_id.get(cell.field_id)
        color = FIELD_COLOR
        thickness = 1
        if result is not None:
            selected_row = result.selected_rows[cell.column]
            if selected_row is None:
                color = UNRESOLVED_COLOR
            elif selected_row == cell.row:
                color = SELECTED_COLOR
                thickness = 2
        cv.circle(overlay, (cell.cx, cell.cy), cell.radius, color, thickness)

    for config in config_by_id.values():
        result = result_by_id.get(config.id)
        label = config.label
        if result is not None:
            label = f"{label}: {result.decoded_value}"
        text_origin = (config.origin[0] - config.radius, max(30, config.origin[1] - config.dy // 2))
        cv.putText(overlay, label, text_origin, FONT, 0.7, TEXT_SHADOW, 3, cv.LINE_AA)
        cv.putText(overlay, label, text_origin, FONT, 0.7, TEXT_COLOR, 1, cv.LINE_AA)
    return overlay


def build_bubble_field_manifest(results: Iterable[BubbleFieldResult]) -> dict:
    return {
        "fields": [
            {
                "field_id": result.field_id,
                "label": result.label,
                "decoded_value": result.decoded_value,
                "selected_rows": list(result.selected_rows),
                "selected_values": list(result.selected_values),
                "is_complete": result.is_complete,
                "threshold_abs": result.threshold_abs,
                "threshold_rel": result.threshold_rel,
            }
            for result in results
        ]
    }
