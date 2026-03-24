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
from postprocess_engine.bubble_field_reader import (
    build_bubble_field_manifest,
    draw_bubble_field_overlay,
    load_bubble_field_configs,
    read_bubble_field_values,
)
from postprocess_engine.output_artifacts import (
    load_pipeline_output_config,
    save_image_if_enabled,
    save_json_if_enabled,
)


INPUT_IMAGE = "samples/1photo5.jpg"              # ảnh chụp cần warp
TEMPLATE_IMAGE = "samples/template_scan1.png"    # ảnh template
OUTPUT = "results"
DEBUG_MODE = True

CIRCLE_ROIS_JSON = "config/circle_rois.json"
ANSWER_KEY_JSON = "config/answer_key.json"
ID_BUBBLE_FIELDS_JSON = "config/id_bubble_fields.json"
OUTPUT_CONFIG_JSON = "config/pipeline_outputs.json"

OUTPUT_SIZE = A4_PX
USE_EXISTING_TEMPLATE = True


def log_time(name: str, start: float):
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[TIME] {name}: {elapsed:.2f} ms")


def main():
    t_total = time.perf_counter()

    safe_mkdir(OUTPUT)
    output_config = load_pipeline_output_config(OUTPUT_CONFIG_JSON)

    t = time.perf_counter()
    if not USE_EXISTING_TEMPLATE:
        extract_template(
            TEMPLATE_IMAGE,
            TEMPLATE_LAYOUT_FILE,
            OUTPUT,
            output_config.debug_intermediate
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
    warp_artifacts = warp_engine.warp_with_artifacts(
        img,
        output=OUTPUT if output_config.debug_intermediate else None,
        use_global_idw=False,
        use_region_refine=True,
        debug=output_config.debug_intermediate,
    )
    warped_a4 = warp_artifacts.template_merged_img
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
    omr_result = omr.run(
        warped_a4,
        output=OUTPUT if output_config.debug_intermediate else None,
        debug=output_config.debug_intermediate,
    )
    log_time("Run OMR", t)

    t = time.perf_counter()
    id_field_configs = load_bubble_field_configs(ID_BUBBLE_FIELDS_JSON)
    id_field_results, _ = read_bubble_field_values(
        warp_artifacts.aligned_source_img,
        id_field_configs,
        abs_th=omr.abs_th,
        rel_th=omr.rel_th,
    )
    id_field_manifest = build_bubble_field_manifest(id_field_results)
    bubble_overlay = draw_bubble_field_overlay(
        warp_artifacts.aligned_source_img,
        id_field_configs,
        id_field_results,
    )
    save_image_if_enabled(
        output_config.bubble_fields.enabled and output_config.bubble_fields.overlay_image,
        os.path.join(OUTPUT, "id_bubble_fields.png"),
        bubble_overlay,
    )
    save_json_if_enabled(
        output_config.bubble_fields.enabled and output_config.bubble_fields.values_json,
        os.path.join(OUTPUT, "id_bubble_values.json"),
        id_field_manifest,
    )
    student_id = next(
        (field["decoded_value"] for field in id_field_manifest["fields"] if field["field_id"] == "student_id"),
        None,
    )
    quiz_id = next(
        (field["decoded_value"] for field in id_field_manifest["fields"] if field["field_id"] == "quiz_id"),
        None,
    )
    log_time("Read ID bubbles", t)

    t = time.perf_counter()
    scored_img = omr_result["scored_img"]
    if DEBUG_MODE:
        out_path = os.path.join(OUTPUT, "orm_3_scored.png")
    else:
        out_path = os.path.join(OUTPUT, "scored_img.png")
    save_image_if_enabled(output_config.scored_image, out_path, scored_img)
    log_time("Save output image", t)

    t = time.perf_counter()
    answers = [int(x) for x in omr_result["answers"]]
    graded_questions = sum(1 for gt in answer_key if gt >= 0)
    score = sum(
        1
        for idx, gt in enumerate(answer_key)
        if gt >= 0 and idx < len(answers) and answers[idx] == gt
    )
    pipeline_result = {
        "input": INPUT_IMAGE,
        "mode": "aligned",
        "score": score,
        "graded_questions": graded_questions,
        "total_questions": len(answer_key),
        "answers": answers,
        "student_id": student_id,
        "quiz_id": quiz_id,
        "thresholds": {
            "abs_th": float(omr_result["thresholds"]["abs_th"]),
            "rel_th": float(omr_result["thresholds"]["rel_th"]),
        },
        "bubble_fields": {
            "enabled": True,
            "config_path": ID_BUBBLE_FIELDS_JSON,
            "output_dir": os.path.join(OUTPUT, "id_bubble_values.json"),
            "fields": id_field_manifest["fields"],
            "source": "python",
        },
    }
    save_json_if_enabled(
        output_config.summary_json,
        os.path.join(OUTPUT, "pipeline_result.json"),
        pipeline_result,
    )
    log_time("Save pipeline JSON", t)

    log_time("TOTAL PIPELINE", t_total)


if __name__ == "__main__":
    main()
