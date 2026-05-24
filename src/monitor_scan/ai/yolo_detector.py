from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from monitor_scan.types import BoundingBox, PersonDetection

COCO_PERSON_CLASS_ID = 0


class YoloPersonDetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.5,
        image_size: int = 640,
        nms_threshold: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"未找到 YOLO ONNX 模型：{self.model_path}")
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.nms_threshold = nms_threshold
        self.session = ort.InferenceSession(str(self.model_path), providers=self._providers())
        input_metadata = self.session.get_inputs()[0]
        self.input_name = input_metadata.name
        self.input_dtype = self._input_numpy_dtype(input_metadata.type)

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        input_tensor, scale, pad_x, pad_y = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        predictions = self._normalize_output(outputs[0])
        return self._postprocess(predictions, frame.shape[1], frame.shape[0], scale, pad_x, pad_y)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(self.image_size / width, self.image_size / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        pad_x = (self.image_size - resized_width) // 2
        pad_y = (self.image_size - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = tensor.astype(self.input_dtype, copy=False)
        return np.expand_dims(tensor, axis=0), scale, pad_x, pad_y

    def _normalize_output(self, output: np.ndarray) -> np.ndarray:
        predictions = np.squeeze(output)
        if predictions.ndim != 2:
            raise ValueError("YOLO 模型输出维度不符合预期。")
        if predictions.shape[0] in (84, 85) and predictions.shape[1] not in (84, 85):
            predictions = predictions.T
        return predictions

    def _postprocess(
        self,
        predictions: np.ndarray,
        frame_width: int,
        frame_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> list[PersonDetection]:
        boxes: list[list[int]] = []
        scores: list[float] = []

        for row in predictions:
            if row.shape[0] < 6:
                continue
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if class_id != COCO_PERSON_CLASS_ID or confidence < self.confidence_threshold:
                continue

            center_x, center_y, width, height = row[:4]
            x1 = int(round((center_x - width / 2 - pad_x) / scale))
            y1 = int(round((center_y - height / 2 - pad_y) / scale))
            x2 = int(round((center_x + width / 2 - pad_x) / scale))
            y2 = int(round((center_y + height / 2 - pad_y) / scale))
            x1 = min(max(x1, 0), frame_width - 1)
            y1 = min(max(y1, 0), frame_height - 1)
            x2 = min(max(x2, 0), frame_width - 1)
            y2 = min(max(y2, 0), frame_height - 1)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)

        kept_indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.nms_threshold)
        if len(kept_indices) == 0:
            return []

        detections: list[PersonDetection] = []
        for index in np.array(kept_indices).flatten():
            x, y, width, height = boxes[int(index)]
            detections.append(
                PersonDetection(
                    box=BoundingBox(x, y, x + width, y + height),
                    confidence=scores[int(index)],
                )
            )
        return detections

    def _input_numpy_dtype(self, onnx_type: str) -> np.dtype:
        if onnx_type == "tensor(float16)":
            return np.dtype(np.float16)
        if onnx_type == "tensor(float)":
            return np.dtype(np.float32)
        raise ValueError(f"暂不支持的 YOLO 模型输入类型：{onnx_type}")

    def _providers(self) -> list[str]:
        available = ort.get_available_providers()
        preferred = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return [provider for provider in preferred if provider in available] or available
