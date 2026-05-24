from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, motion_threshold: int = 5000) -> None:
        self.motion_threshold = motion_threshold
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=25,
            detectShadows=True,
        )

    def has_motion(self, frame: np.ndarray) -> bool:
        mask = self._subtractor.apply(frame)
        _, thresholded = cv2.threshold(mask, 244, 255, cv2.THRESH_BINARY)
        thresholded = cv2.medianBlur(thresholded, 5)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moving_area = sum(cv2.contourArea(contour) for contour in contours)
        return moving_area >= self.motion_threshold
