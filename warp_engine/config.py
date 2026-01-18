import cv2 as cv

APRILTAG_DICT = cv.aruco.DICT_APRILTAG_16h5

A4_PX = (2481, 3509)

TEMPLATE_LAYOUT_FILE = "config/template_marker_layout.json"

WINDOWS_4PTS = [
    [13, 14, 15, 16],
    [15, 16, 17, 18],
    [14, 16, 19, 20],
    [16, 18, 20, 21],
]
