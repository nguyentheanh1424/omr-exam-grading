from dataclasses import dataclass
import numpy as np


@dataclass
class TagDetection:
    id: int
    corners: np.ndarray
    center: np.ndarray
