from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

from postprocess_engine.bubble_field_reader import (
    build_bubble_field_manifest,
    load_bubble_field_configs,
    read_bubble_field_values,
    validate_bubble_field_configs,
)


class BubbleFieldReaderTests(unittest.TestCase):
    def test_load_bubble_field_configs_rejects_bad_row_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "id_fields.json"
            config_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "student_id",
                            "origin": [20, 20],
                            "dx": 20,
                            "dy": 20,
                            "n_cols": 2,
                            "n_rows": 3,
                            "radius": 8,
                            "row_values": ["0", "1"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "row_values length must equal n_rows"):
                load_bubble_field_configs(config_path)

    def test_validate_bubble_field_configs_rejects_out_of_bounds(self) -> None:
        configs = load_bubble_field_configs(
            _write_temp_config(
                [
                    {
                        "id": "student_id",
                        "origin": [10, 10],
                        "dx": 20,
                        "dy": 20,
                        "n_cols": 3,
                        "n_rows": 3,
                        "radius": 12,
                        "row_values": ["0", "1", "2"],
                    }
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "origin is out of bounds"):
            validate_bubble_field_configs(configs, (40, 40, 3))

    def test_read_bubble_field_values_decodes_selected_rows(self) -> None:
        config_path = _write_temp_config(
            [
                {
                    "id": "quiz_id",
                    "label": "Quiz ID",
                    "origin": [30, 30],
                    "dx": 45,
                    "dy": 45,
                    "n_cols": 2,
                    "n_rows": 3,
                    "radius": 12,
                    "row_values": ["0", "1", "2"],
                }
            ]
        )
        configs = load_bubble_field_configs(config_path)
        img = np.full((180, 180, 3), 255, dtype=np.uint8)
        _draw_bubble_grid(img, origin=(30, 30), dx=45, dy=45, n_cols=2, n_rows=3, radius=12)
        _mark_bubble(img, cx=30, cy=75, radius=12)
        _mark_bubble(img, cx=75, cy=120, radius=12)

        results, _ = read_bubble_field_values(img, configs, abs_th=0.1, rel_th=0.02)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].decoded_value, "12")
        self.assertEqual(results[0].selected_rows, (1, 2))
        self.assertTrue(results[0].is_complete)

    def test_build_bubble_field_manifest_is_stable(self) -> None:
        config_path = _write_temp_config(
            [
                {
                    "id": "student_id",
                    "label": "Student ID",
                    "origin": [30, 30],
                    "dx": 45,
                    "dy": 45,
                    "n_cols": 1,
                    "n_rows": 2,
                    "radius": 12,
                    "row_values": ["0", "1"],
                }
            ]
        )
        configs = load_bubble_field_configs(config_path)
        img = np.full((120, 120, 3), 255, dtype=np.uint8)
        _draw_bubble_grid(img, origin=(30, 30), dx=45, dy=45, n_cols=1, n_rows=2, radius=12)
        _mark_bubble(img, cx=30, cy=30, radius=12)
        results, _ = read_bubble_field_values(img, configs, abs_th=0.1, rel_th=0.02)
        manifest = build_bubble_field_manifest(results)
        self.assertEqual(manifest["fields"][0]["field_id"], "student_id")
        self.assertEqual(manifest["fields"][0]["decoded_value"], "0")


def _write_temp_config(data: list[dict]) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    config_path = tmp_dir / "id_fields.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


def _draw_bubble_grid(
    img: np.ndarray,
    *,
    origin: tuple[int, int],
    dx: int,
    dy: int,
    n_cols: int,
    n_rows: int,
    radius: int,
) -> None:
    ox, oy = origin
    for column in range(n_cols):
        for row in range(n_rows):
            cx = ox + column * dx
            cy = oy + row * dy
            cv.circle(img, (cx, cy), radius, (0, 0, 0), 1)


def _mark_bubble(img: np.ndarray, *, cx: int, cy: int, radius: int) -> None:
    cv.circle(img, (cx, cy), int(radius * 0.8), (0, 0, 0), 2)


if __name__ == "__main__":
    unittest.main()
