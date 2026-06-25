from __future__ import annotations

import numpy as np

from monitor_scan.ai.motion import MotionDetector


def test_motion_detector_resizes_large_frames():
    detector = MotionDetector(resize_width=100)
    frame = np.zeros((200, 400, 3), dtype=np.uint8)

    resized = detector._resize_frame(frame)

    assert resized.shape == (50, 100, 3)


def test_motion_detector_ignores_static_frame_after_background_warmup():
    detector = MotionDetector(resize_width=64, area_ratio_threshold=0.05)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    detector.has_motion(frame)

    assert detector.has_motion(frame) is False


def test_motion_detector_detects_large_change_after_background_warmup():
    detector = MotionDetector(resize_width=64, area_ratio_threshold=0.01)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    changed = frame.copy()
    changed[16:48, 16:48] = 255

    detector.has_motion(frame)

    assert detector.has_motion(changed) is True
