from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv


APRILTAG_DICT = cv.aruco.DICT_APRILTAG_16h5

A4_PX = (2481, 3509)

TEMPLATE_LAYOUT_FILE = "config/template_marker_layout.json"
REGION_WINDOWS_FILE = "config/region_windows.json"

DEFAULT_WINDOWS_4PTS = [
    [13, 14, 15, 16],
    [15, 16, 17, 18],
    [14, 16, 19, 20],
    [16, 18, 20, 21],
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return _repo_root() / value


def load_region_windows(path: str | Path = REGION_WINDOWS_FILE) -> list[list[int]]:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        return [list(window) for window in DEFAULT_WINDOWS_4PTS]

    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not data:
        raise ValueError("region windows config must be a non-empty list")

    windows: list[list[int]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, list) or len(item) != 4:
            raise ValueError(f"region window #{idx} must contain exactly 4 marker ids")
        marker_ids = [int(marker_id) for marker_id in item]
        if len(set(marker_ids)) != 4:
            raise ValueError(f"region window #{idx} contains duplicate marker ids")
        windows.append(marker_ids)
    return windows


WINDOWS_4PTS = load_region_windows()
