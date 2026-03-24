from __future__ import annotations

from native_core.python_adapter import build_native_adapter_config, summarize_adapter_config


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


if __name__ == "__main__":
    main()
