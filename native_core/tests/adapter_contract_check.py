from __future__ import annotations

from native_core.python_adapter import (
    build_native_adapter_config,
    build_native_adapter_config_from_data,
    summarize_adapter_config,
)
from orm_engine.orm import CircleROI


def check_in_memory_selection_modes() -> None:
    circle_rois = [
        CircleROI(20, 20, 8, 1, 0, "single"),
        CircleROI(40, 20, 8, 1, 1, "single"),
        CircleROI(20, 40, 8, 2, 0, "multiple"),
        CircleROI(40, 40, 8, 2, 1, "multiple"),
    ]
    config = build_native_adapter_config_from_data(
        circle_rois=circle_rois,
        answer_key=[1, -1],
    )
    assert config.n_questions == 2
    assert config.n_options_per_question == 2
    assert [roi.selection_mode for roi in config.circle_rois[:2]] == [0, 0]
    assert [roi.selection_mode for roi in config.circle_rois[2:]] == [1, 1]
    print("[OK] in-memory adapter config preserves question selection modes")

    invalid_rois = [
        CircleROI(20, 20, 8, 1, 0, "single"),
        CircleROI(40, 20, 8, 1, 1, "multiple"),
    ]
    try:
        build_native_adapter_config_from_data(
            circle_rois=invalid_rois,
            answer_key=[1],
        )
    except ValueError as exc:
        print("[OK] inconsistent selection_mode rejected:", exc)
    else:
        raise AssertionError("expected inconsistent selection_mode to raise ValueError")


def main() -> None:
    config = build_native_adapter_config()
    print("[OK] adapter config loaded")
    print(summarize_adapter_config(config))
    first_roi = config.circle_rois[0]
    last_roi = config.circle_rois[-1]
    print(
        "[OK] ROI question range:",
        f"{first_roi.question}..{last_roi.question}",
        f"(first option={first_roi.option}, last option={last_roi.option})",
    )
    print("[OK] answer_key length:", len(config.answer_key))
    check_in_memory_selection_modes()


if __name__ == "__main__":
    main()
