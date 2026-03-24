import json
import os
import cv2 as cv
from .detector import detect_tags
from .utils import  draw_detections, safe_mkdir
from .config import TEMPLATE_MARKER_POSITIONS_FILE
import logging

log = logging.getLogger(__name__)


def extract_template(path_img, path_out=TEMPLATE_MARKER_POSITIONS_FILE, output=None, debug=False):
    img = cv.imread(path_img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    detections = detect_tags(gray)

    if debug:
        vis = draw_detections(img, detections)
        cv.imwrite(f"{output}/step1_template_markers.png", vis)

    layout = {}
    for d in detections:
        layout[d.id] = d.center.tolist()

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)

    return layout


def load_template(path=TEMPLATE_MARKER_POSITIONS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}
