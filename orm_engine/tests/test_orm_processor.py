from __future__ import annotations

import unittest

import numpy as np

from orm_engine.orm import CircleROI, OMRProcessor


def build_processor() -> OMRProcessor:
    rois = [
        CircleROI(cx=30, cy=30, r=10, question=1, option=0, selection_mode="single"),
        CircleROI(cx=60, cy=30, r=10, question=1, option=1, selection_mode="single"),
        CircleROI(cx=90, cy=30, r=10, question=1, option=2, selection_mode="single"),
        CircleROI(cx=120, cy=30, r=10, question=1, option=3, selection_mode="single"),
        CircleROI(cx=150, cy=30, r=10, question=1, option=4, selection_mode="single"),
    ]
    processor = OMRProcessor(circle_rois=rois, answer_key=[0], auto_threshold=False)
    processor.abs_th = 0.20
    processor.rel_th = 0.055
    return processor


class OMRProcessorTests(unittest.TestCase):
    def test_accepts_grayscale_input(self) -> None:
        processor = build_processor()
        img = np.full((180, 180), 255, dtype=np.uint8)

        result = processor.run(img, output=None, debug=False)

        self.assertEqual(result["answers"], [-1])
        self.assertEqual(result["question_statuses"], ["blank"])
        self.assertEqual(result["scored_img"].ndim, 3)
        self.assertEqual(result["scored_img"].shape[2], 3)

    def test_requires_output_dir_when_debug_enabled(self) -> None:
        processor = build_processor()
        img = np.full((180, 180, 3), 255, dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "output directory is required"):
            processor.run(img, output=None, debug=True)

    def test_detects_multiple_when_two_bubbles_are_strong(self) -> None:
        processor = build_processor()
        items = [(1, 0.31), (2, 0.27), (0, 0.04), (3, 0.03), (4, 0.02)]

        filled = processor._detect_filled_options(items)
        answer, status = processor._resolve_question_selection(items, filled, "single")

        self.assertEqual(filled, [1, 2])
        self.assertEqual(answer, -1)
        self.assertEqual(status, "invalid_multiple_on_single")

    def test_recovers_multiple_when_second_mark_is_below_raised_abs_threshold(self) -> None:
        processor = build_processor()
        items = [(1, 0.26), (2, 0.19), (3, 0.05), (0, 0.02), (4, 0.01)]

        filled = processor._detect_filled_options(items)
        answer, status = processor._resolve_question_selection(items, filled, "single")

        self.assertEqual(filled, [1, 2])
        self.assertEqual(answer, -1)
        self.assertEqual(status, "invalid_multiple_on_single")

    def test_keeps_single_when_second_mark_is_weak(self) -> None:
        processor = build_processor()
        items = [(1, 0.28), (2, 0.12), (3, 0.03), (0, 0.02), (4, 0.01)]

        filled = processor._detect_filled_options(items)
        answer, status = processor._resolve_question_selection(items, filled, "single")

        self.assertEqual(filled, [1])
        self.assertEqual(answer, 1)
        self.assertEqual(status, "single")

    def test_allows_multiple_selection_for_multiple_mode(self) -> None:
        processor = build_processor()
        items = [(1, 0.31), (2, 0.27), (0, 0.04), (3, 0.03), (4, 0.02)]

        filled = processor._detect_filled_options(items)
        answer, status = processor._resolve_question_selection(items, filled, "multiple")

        self.assertEqual(filled, [1, 2])
        self.assertEqual(answer, -1)
        self.assertEqual(status, "multiple")

    def test_marks_uncertain_for_multiple_mode_when_second_mark_is_only_recovered(self) -> None:
        processor = build_processor()
        items = [(1, 0.26), (2, 0.19), (3, 0.05), (0, 0.02), (4, 0.01)]

        filled = processor._detect_filled_options(items)
        answer, status = processor._resolve_question_selection(items, filled, "multiple")

        self.assertEqual(filled, [1, 2])
        self.assertEqual(answer, -1)
        self.assertEqual(status, "uncertain")

    def test_keeps_single_for_multiple_mode_when_only_one_option_is_filled(self) -> None:
        processor = build_processor()
        items = [(1, 0.28), (2, 0.12), (3, 0.03), (0, 0.02), (4, 0.01)]

        filled = processor._detect_filled_options(items)
        answer, status = processor._resolve_question_selection(items, filled, "multiple")

        self.assertEqual(filled, [1])
        self.assertEqual(answer, 1)
        self.assertEqual(status, "single")

    def test_rejects_inconsistent_selection_mode_within_question(self) -> None:
        rois = [
            CircleROI(cx=30, cy=30, r=10, question=1, option=0, selection_mode="single"),
            CircleROI(cx=60, cy=30, r=10, question=1, option=1, selection_mode="multiple"),
        ]

        with self.assertRaisesRegex(ValueError, "inconsistent selection_mode"):
            OMRProcessor(circle_rois=rois, answer_key=[0], auto_threshold=False)


if __name__ == "__main__":
    unittest.main()
