from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch as mock_patch

import numpy as np

from postprocess_engine.handwritten_review import (
    HandwrittenPatch,
    HandwrittenRegion,
    build_manifest,
    crop_handwritten_regions,
    load_handwritten_regions,
    merge_handwritten_regions,
    validate_handwritten_regions,
)


class HandwrittenReviewTests(unittest.TestCase):
    def test_load_handwritten_regions_rejects_duplicate_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "regions.json"
            config_path.write_text(
                json.dumps(
                    [
                        {"id": "name_line", "rect": [10, 10, 20, 20]},
                        {"id": "name_line", "rect": [30, 30, 40, 40]},
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate handwritten region id"):
                load_handwritten_regions(config_path)

    def test_validate_handwritten_regions_rejects_out_of_bounds_rect(self) -> None:
        regions = [
            HandwrittenRegion(
                id="name_line",
                label="Name",
                rect=(0, 0, 50, 50),
                padding_px=0,
                merge_mode="replace_rect",
            ),
            HandwrittenRegion(
                id="bad_region",
                label="Bad",
                rect=(10, 10, 120, 30),
                padding_px=0,
                merge_mode="replace_rect",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "exceeds image bounds"):
            validate_handwritten_regions(regions, (100, 100, 3))

    def test_crop_handwritten_regions_respects_padding(self) -> None:
        img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        regions = [
            HandwrittenRegion(
                id="quiz_line",
                label="Quiz",
                rect=(20, 30, 40, 50),
                padding_px=5,
                merge_mode="replace_rect",
            )
        ]
        patches = crop_handwritten_regions(img, regions)
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].rect, (15, 25, 45, 55))
        self.assertEqual(tuple(patches[0].image.shape), (30, 30, 3))

    def test_merge_handwritten_regions_replace_rect(self) -> None:
        base = np.full((20, 20, 3), 255, dtype=np.uint8)
        patch = HandwrittenPatch(
            region_id="name_line",
            label="Name",
            rect=(5, 5, 10, 10),
            image=np.zeros((5, 5, 3), dtype=np.uint8),
        )
        regions = [
            HandwrittenRegion(
                id="name_line",
                label="Name",
                rect=(5, 5, 10, 10),
                padding_px=0,
                merge_mode="replace_rect",
            )
        ]
        merged = merge_handwritten_regions(base, [patch], regions)
        self.assertTrue(np.all(merged[5:10, 5:10] == 0))
        self.assertTrue(np.all(merged[:5, :5] == 255))

    def test_merge_handwritten_regions_ink_mask(self) -> None:
        base = np.full((40, 40, 3), 255, dtype=np.uint8)
        patch_img = np.full((20, 20, 3), 255, dtype=np.uint8)
        patch = HandwrittenPatch(
            region_id="score_line",
            label="Score",
            rect=(5, 5, 25, 25),
            image=patch_img,
        )
        regions = [
            HandwrittenRegion(
                id="score_line",
                label="Score",
                rect=(5, 5, 25, 25),
                padding_px=0,
                merge_mode="ink_mask",
            )
        ]
        ink_mask = np.zeros((20, 20), dtype=bool)
        ink_mask[4:10, 6:12] = True
        with mock_patch("postprocess_engine.handwritten_review.binarize_patch_dual", return_value=ink_mask):
            merged = merge_handwritten_regions(base, [patch], regions)
        self.assertTrue(np.any(merged[5:25, 5:25] == 0))
        self.assertEqual(tuple(merged[5, 5]), (255, 255, 255))

    def test_build_manifest_hides_patch_file_when_save_patch_is_false(self) -> None:
        manifest = build_manifest(
            "samples/1photo5.jpg",
            [
                HandwrittenPatch(
                    region_id="class_line",
                    label="Class",
                    rect=(1, 2, 3, 4),
                    image=np.zeros((2, 2, 3), dtype=np.uint8),
                    save_patch=False,
                )
            ],
        )
        self.assertEqual(manifest["patches"][0]["file"], None)
        self.assertEqual(manifest["patches"][0]["save_patch"], False)


if __name__ == "__main__":
    unittest.main()
