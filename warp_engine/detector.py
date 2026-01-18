import cv2 as cv
import numpy as np
from functools import lru_cache
from .config import APRILTAG_DICT
from .types import TagDetection
import logging

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def build_detector():
    params = cv.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv.aruco.getPredefinedDictionary(APRILTAG_DICT)
    return cv.aruco.ArucoDetector(dictionary, params)


def detect_tags(gray):
    detector = build_detector()
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []

    ids = ids.reshape(-1)
    dets = []
    for i, c in enumerate(corners):
        pts = np.asarray(c, np.float32).reshape(-1, 2)
        dets.append(TagDetection(id=int(ids[i]), corners=pts, center=pts.mean(0)))

    dets.sort(key=lambda d: d.id)
    return dets
