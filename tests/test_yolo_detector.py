from __future__ import annotations

import numpy as np

from monitor_scan.ai.yolo_detector import YoloPersonDetector


def test_normalize_output_transposes_yolov8_shape_without_model(tmp_path):
    detector = object.__new__(YoloPersonDetector)
    output = np.zeros((1, 84, 5), dtype=np.float32)

    normalized = detector._normalize_output(output)

    assert normalized.shape == (5, 84)


def test_preprocess_uses_model_input_dtype_without_model():
    detector = object.__new__(YoloPersonDetector)
    detector.image_size = 64
    detector.input_dtype = np.dtype(np.float16)
    frame = np.zeros((16, 32, 3), dtype=np.uint8)

    tensor, scale, pad_x, pad_y = detector._preprocess(frame)

    assert tensor.dtype == np.float16
    assert tensor.shape == (1, 3, 64, 64)
    assert scale == 2.0
    assert pad_x == 0
    assert pad_y == 16


def test_input_numpy_dtype_supports_float16_and_float32_without_model():
    detector = object.__new__(YoloPersonDetector)

    assert detector._input_numpy_dtype("tensor(float16)") == np.dtype(np.float16)
    assert detector._input_numpy_dtype("tensor(float)") == np.dtype(np.float32)
