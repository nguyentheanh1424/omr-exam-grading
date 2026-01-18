from __future__ import annotations

import os
import json
import time

import cv2 as cv

from warp_engine.config import TEMPLATE_LAYOUT_FILE, A4_PX
from warp_engine.engine import WarpEngine
from warp_engine.template import extract_template
from warp_engine.utils import safe_mkdir

from orm_engine.orm import OMRProcessor, load_circle_rois


INPUT_IMAGE = "samples/1photo2.jpg"              # ảnh chụp cần warp
TEMPLATE_IMAGE = "samples/template_scan1.png"    # ảnh template
OUTPUT = "results"
DEBUG_MODE = True

CIRCLE_ROIS_JSON = "config/circle_rois.json"
ANSWER_KEY_JSON = "config/answer_key.json"

OUTPUT_SIZE = A4_PX
USE_EXISTING_TEMPLATE = True


def log_time(name: str, start: float):
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[TIME] {name}: {elapsed:.2f} ms")


def main():
    t_total = time.perf_counter()

    safe_mkdir(OUTPUT)

    t = time.perf_counter()
    if not USE_EXISTING_TEMPLATE:
        extract_template(
            TEMPLATE_IMAGE,
            TEMPLATE_LAYOUT_FILE,
            OUTPUT,
            DEBUG_MODE
        )
    else:
        if not os.path.exists(TEMPLATE_LAYOUT_FILE):
            raise FileNotFoundError(
                f"Không tìm thấy {TEMPLATE_LAYOUT_FILE}. "
                f"Bạn cần đặt USE_EXISTING_TEMPLATE=False để extract."
            )
    log_time("Extract template", t)

    t = time.perf_counter()
    warp_engine = WarpEngine(
        TEMPLATE_LAYOUT_FILE,
        TEMPLATE_IMAGE,
    )
    log_time("Init WarpEngine", t)

    t = time.perf_counter()
    img = cv.imread(INPUT_IMAGE)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh input: {INPUT_IMAGE}")
    log_time("Read input image", t)

    t = time.perf_counter()
    warped_a4 = warp_engine.warp(
        img,
        output=OUTPUT,
        use_global_idw=True,
        use_region_refine=True,
        debug=DEBUG_MODE,
    )
    log_time("Warp to A4", t)

    t = time.perf_counter()
    circle_rois = load_circle_rois(CIRCLE_ROIS_JSON)
    log_time("Load circle ROIs", t)

    t = time.perf_counter()
    if os.path.exists(ANSWER_KEY_JSON):
        with open(ANSWER_KEY_JSON, "r", encoding="utf-8") as f:
            answer_key = [int(x) for x in json.load(f)]
    else:
        max_q = max(r.question for r in circle_rois)
        answer_key = [0] * max_q
    log_time("Load answer key", t)

    t = time.perf_counter()
    omr = OMRProcessor(
        circle_rois=circle_rois,
        answer_key=answer_key,
    )
    log_time("Init OMRProcessor", t)

    t = time.perf_counter()
    omr_result = omr.run(warped_a4, output=OUTPUT, debug=DEBUG_MODE)
    log_time("Run OMR", t)

    t = time.perf_counter()
    scored_img = omr_result["scored_img"]
    if DEBUG_MODE:
        out_path = os.path.join(OUTPUT, "orm_3_scored.png")
    else:
        out_path = os.path.join(OUTPUT, "scored_img.png")
    cv.imwrite(out_path, scored_img)
    log_time("Save output image", t)

    log_time("TOTAL PIPELINE", t_total)


if __name__ == "__main__":
    main()
