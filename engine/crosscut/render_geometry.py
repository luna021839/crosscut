"""Clockwise renderer-space editorial geometry shared by selection and QA."""
import math
import cv2
import numpy as np


def reproject_editorial_frame(frame: np.ndarray, *, size: tuple[int, int], zoom: float,
        center_x: float, center_y: float, roll_degrees: float, maximum_zoom: float) -> np.ndarray:
    # FFmpeg: even maximum-zoom aspect-fill canvas -> clockwise rotate ->
    # integer zoompan crop -> output. No reflected/artificial edge pixels.
    width = max(size[0], int(math.ceil(size[0] * maximum_zoom / 2.0)) * 2)
    height = max(size[1], int(math.ceil(size[1] * maximum_zoom / 2.0)) * 2)
    source_height, source_width = frame.shape[:2]
    scale = max(width / source_width, height / source_height)
    scaled_width = max(width, int(round(source_width * scale)))
    scaled_height = max(height, int(round(source_height * scale)))
    working = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
    left, top = (scaled_width - width) // 2, (scaled_height - height) // 2
    working = working[top:top + height, left:left + width]
    if abs(roll_degrees) > 1e-9:
        rotation = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -roll_degrees, 1.0)
        working = cv2.warpAffine(working, rotation, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    crop_width, crop_height = max(1, int(width / zoom)), max(1, int(height / zoom))
    left = min(max(0, int(width * center_x - width / (2.0 * zoom))), width - crop_width)
    top = min(max(0, int(height * center_y - height / (2.0 * zoom))), height - crop_height)
    return cv2.resize(working[top:top + crop_height, left:left + crop_width], size, interpolation=cv2.INTER_CUBIC)
