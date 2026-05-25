from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monitor_scan.ai.yolo_detector import YoloPersonDetector
from monitor_scan.types import BoundingBox


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


def test_backend_for_path_supports_onnx_pt_and_coreml_without_model():
    detector = object.__new__(YoloPersonDetector)

    assert detector._backend_for_path(Path("models/yolo26n.onnx")) == "onnx"
    assert detector._backend_for_path(Path("models/yolo26n.pt")) == "ultralytics"
    assert detector._backend_for_path(Path("models/yolo26n.mlpackage")) == "ultralytics"


def test_backend_for_path_rejects_unknown_suffix_without_model():
    detector = object.__new__(YoloPersonDetector)

    with pytest.raises(ValueError, match="暂不支持的 YOLO 模型格式"):
        detector._backend_for_path(Path("models/yolo26n.bin"))


def test_ultralytics_result_filters_people_and_clips_boxes_without_model():
    detector = object.__new__(YoloPersonDetector)
    detector.confidence_threshold = 0.5
    result = _FakeResult(
        xyxy=np.array([[-5.2, 1.2, 80.6, 100.4], [10, 10, 20, 20], [5, 5, 15, 15]], dtype=np.float32),
        conf=np.array([0.7, 0.9, 0.4], dtype=np.float32),
        cls=np.array([0, 1, 0], dtype=np.float32),
    )

    detections = detector._detections_from_ultralytics_result(result, frame_width=64, frame_height=48)

    assert len(detections) == 1
    assert detections[0].box == BoundingBox(0, 1, 63, 47)
    assert detections[0].confidence == pytest.approx(0.7)


def test_detect_with_ultralytics_passes_runtime_thresholds_without_model():
    detector = object.__new__(YoloPersonDetector)
    detector.image_size = 320
    detector.confidence_threshold = 0.6
    detector.nms_threshold = 0.3
    detector.model = _FakeModel()
    frame = np.zeros((10, 20, 3), dtype=np.uint8)

    detections = detector._detect_with_ultralytics(frame)

    assert detections == []
    assert len(detector.model.calls) == 1
    assert detector.model.calls[0]["frame"] is frame
    assert detector.model.calls[0] | {"frame": None} == {
        "frame": None,
        "imgsz": 320,
        "conf": 0.6,
        "iou": 0.3,
        "classes": [0],
        "verbose": False,
    }


class _FakeResult:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.boxes = _FakeBoxes(xyxy, conf, cls)


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class _FakeModel:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, frame, imgsz, conf, iou, classes, verbose):
        self.calls.append(
            {
                "frame": frame,
                "imgsz": imgsz,
                "conf": conf,
                "iou": iou,
                "classes": classes,
                "verbose": verbose,
            }
        )
        return [_FakeResult(np.empty((0, 4)), np.empty((0,)), np.empty((0,)))]
