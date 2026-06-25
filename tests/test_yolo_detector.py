from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from monitor_scan.ai.yolo_detector import COREML_COMPUTE_UNIT, YoloPersonDetector
from monitor_scan.types import BoundingBox


def test_detector_rejects_non_coreml_model_format(tmp_path):
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(ValueError, match="仅支持 CoreML mlpackage 模型"):
        YoloPersonDetector(model_path)


def test_detector_requires_existing_coreml_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="未找到 YOLO CoreML 模型"):
        YoloPersonDetector(tmp_path / "missing.mlpackage")


def test_detector_loads_coreml_with_neural_engine_priority(tmp_path, monkeypatch):
    model_path = tmp_path / "yolo26n-512-fp16-nms.mlpackage"
    model_path.mkdir()
    fake_coremltools = _fake_coremltools()
    monkeypatch.setitem(sys.modules, "coremltools", fake_coremltools)

    detector = YoloPersonDetector(model_path, coreml_warmup_runs=0)

    assert detector.backend == "coreml"
    assert fake_coremltools.models.calls == [(str(model_path), COREML_COMPUTE_UNIT)]
    assert detector.coreml_input_name == "image"
    assert detector.coreml_output_name == "var_1441"
    assert detector.coreml_input_width == 10
    assert detector.coreml_input_height == 10


def test_detect_with_fake_coreml_model_runs_warmup_and_maps_output(tmp_path, monkeypatch):
    model_path = tmp_path / "yolo26n-512-fp16-nms.mlpackage"
    model_path.mkdir()
    fake_coremltools = _fake_coremltools(
        output=np.array([[[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.8, 1]]], dtype=np.float32)
    )
    monkeypatch.setitem(sys.modules, "coremltools", fake_coremltools)

    detector = YoloPersonDetector(model_path, image_size=10, coreml_warmup_runs=1)
    detections = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert fake_coremltools.models.instances[0].predict_calls == 2
    assert len(detections) == 1
    assert detections[0].box == BoundingBox(0, 0, 9, 9)
    assert detections[0].confidence == pytest.approx(0.9)


def test_preprocess_coreml_letterboxes_frame_without_model():
    detector = object.__new__(YoloPersonDetector)
    detector.coreml_input_width = 64
    detector.coreml_input_height = 64
    frame = np.zeros((16, 32, 3), dtype=np.uint8)

    image, scale, pad_x, pad_y = detector._preprocess_coreml(frame)

    assert image.size == (64, 64)
    assert scale == 2.0
    assert pad_x == 0
    assert pad_y == 16


def test_coreml_nms_output_filters_people_and_maps_letterbox_without_model():
    detector = object.__new__(YoloPersonDetector)
    detector.confidence_threshold = 0.5
    predictions = np.array(
        [
            [10, 20, 110, 220, 0.8, 0],
            [10, 20, 110, 220, 0.9, 1],
            [10, 20, 110, 220, 0.4, 0],
        ],
        dtype=np.float32,
    )

    detections = detector._postprocess_coreml_nms_output(
        predictions,
        frame_width=100,
        frame_height=100,
        scale=2.0,
        pad_x=10,
        pad_y=20,
    )

    assert len(detections) == 1
    assert detections[0].box == BoundingBox(0, 0, 50, 99)
    assert detections[0].confidence == pytest.approx(0.8)


class _FakeCoreMLModels:
    def __init__(self, output: np.ndarray | None = None) -> None:
        self.calls = []
        self.instances = []
        self.output = output if output is not None else np.empty((1, 0, 6), dtype=np.float32)

    def MLModel(self, path: str, compute_units):
        self.calls.append((path, compute_units))
        model = _FakeCoreMLModel(self.output)
        self.instances.append(model)
        return model


class _FakeCoreMLModel:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.predict_calls = 0

    def get_spec(self):
        image_type = SimpleNamespace(width=10, height=10)
        input_description = SimpleNamespace(name="image", type=SimpleNamespace(imageType=image_type))
        output_description = SimpleNamespace(name="var_1441")
        return SimpleNamespace(description=SimpleNamespace(input=[input_description], output=[output_description]))

    def predict(self, inputs):
        self.predict_calls += 1
        return {"var_1441": self.output}


def _fake_coremltools(output: np.ndarray | None = None):
    return SimpleNamespace(
        ComputeUnit=SimpleNamespace(CPU_AND_NE=COREML_COMPUTE_UNIT),
        models=_FakeCoreMLModels(output),
    )
