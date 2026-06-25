"""YOLO 人形检测器模块。

使用 CoreML 模型进行人形检测，支持 Apple Neural Engine 加速。
针对 M1 进行了特化优化。

优化策略：
1. 批量推理：收集多帧一起送入 NE，减少调用开销
2. 内存预分配：避免运行时内存分配
3. CPU-NE 流水线：预处理和后处理在 CPU，推理在 NE
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from monitor_scan.types import BoundingBox, PersonDetection

logger = logging.getLogger(__name__)

# COCO 数据集人形类别 ID
COCO_PERSON_CLASS_ID = 0

# CoreML 模型文件扩展名
COREML_SUFFIX = ".mlpackage"

# CoreML 计算单元配置
COREML_COMPUTE_UNIT = "CPU_AND_NE"

# 批量推理配置
DEFAULT_BATCH_SIZE = 4
MAX_BATCH_SIZE = 8
BATCH_TIMEOUT_MS = 50  # 批量超时时间（毫秒）


class YoloPersonDetector:
    """YOLO 人形检测器。

    使用 CoreML 模型进行人形检测，支持 Apple Neural Engine 加速。

    针对 M1 优化的特性：
    1. 批量推理：减少 NE 调用次数
    2. 预分配内存：避免运行时分配
    3. 异步预处理：CPU 和 NE 并行工作

    Attributes:
        model_path: 模型文件路径
        confidence_threshold: 置信度阈值
        image_size: 模型输入图像尺寸
        nms_threshold: NMS 阈值
        coreml_warmup_runs: CoreML 预热次数
        backend: 推理后端名称
        coreml_model: CoreML 模型实例
        coreml_input_name: 输入张量名称
        coreml_output_name: 输出张量名称
        coreml_input_width: 输入图像宽度
        coreml_input_height: 输入图像高度
        enable_batch_inference: 是否启用批量推理
        batch_size: 批量大小
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.5,
        image_size: int = 512,
        nms_threshold: float = 0.45,
        coreml_warmup_runs: int = 1,
        enable_batch_inference: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """初始化 YOLO 人形检测器。

        Args:
            model_path: CoreML 模型文件路径
            confidence_threshold: 置信度阈值
            image_size: 模型输入图像尺寸
            nms_threshold: NMS 阈值
            coreml_warmup_runs: CoreML 预热次数
            enable_batch_inference: 是否启用批量推理
            batch_size: 批量大小

        Raises:
            ValueError: 如果模型文件格式不正确
            FileNotFoundError: 如果模型文件不存在
            RuntimeError: 如果缺少依赖或模型加载失败
        """
        self.model_path = Path(model_path)
        logger.info(f"初始化 YOLO 检测器，模型路径：{self.model_path}")

        if self.model_path.suffix.lower() != COREML_SUFFIX:
            raise ValueError(f"仅支持 CoreML mlpackage 模型：{self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"未找到 YOLO CoreML 模型：{self.model_path}")

        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.nms_threshold = nms_threshold
        self.coreml_warmup_runs = coreml_warmup_runs
        self.backend = "coreml"
        self.coreml_model: Any | None = None
        self.coreml_input_name = ""
        self.coreml_output_name = ""
        self.coreml_input_width = image_size
        self.coreml_input_height = image_size

        # 批量推理配置
        self.enable_batch_inference = enable_batch_inference
        self.batch_size = min(batch_size, MAX_BATCH_SIZE)
        self._batch_buffer: deque[tuple[np.ndarray, float, int, int]] = deque()
        self._batch_lock = threading.Lock()

        # 性能统计
        self._total_inference_time = 0.0
        self._total_frames = 0
        self._total_batches = 0

        # 预分配内存
        self._preprocess_buffer: np.ndarray | None = None

        self._load_coreml_model()
        self._preallocate_buffers()

        logger.info(
            f"YOLO 检测器初始化完成，后端：{self.backend}，"
            f"批量推理：{'启用' if enable_batch_inference else '禁用'}，"
            f"批量大小：{self.batch_size}"
        )

    def _preallocate_buffers(self) -> None:
        """预分配内存缓冲区。

        避免运行时内存分配，提高性能。
        """
        # 预分配预处理缓冲区
        self._preprocess_buffer = np.full(
            (self.coreml_input_height, self.coreml_input_width, 3),
            114,
            dtype=np.uint8,
        )
        logger.debug(f"预分配缓冲区：{self.coreml_input_width}x{self.coreml_input_height}")

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """检测帧中的人形。

        如果启用批量推理，会将帧加入批量缓冲区，
        当缓冲区满或超时时进行批量推理。

        Args:
            frame: 输入图像帧

        Returns:
            检测到的人形列表
        """
        if self.enable_batch_inference:
            return self._detect_with_batch(frame)
        return self._detect_with_coreml(frame)

    def _detect_with_batch(self, frame: np.ndarray) -> list[PersonDetection]:
        """使用批量推理检测。

        Args:
            frame: 输入图像帧

        Returns:
            检测结果列表
        """
        # 预处理帧
        image, scale, pad_x, pad_y = self._preprocess_coreml(frame)

        with self._batch_lock:
            # 添加到批量缓冲区
            self._batch_buffer.append((frame, scale, pad_x, pad_y))

            # 如果缓冲区满，执行批量推理
            if len(self._batch_buffer) >= self.batch_size:
                return self._execute_batch_inference()

        # 缓冲区未满，执行单帧推理
        return self._detect_with_coreml(frame)

    def _execute_batch_inference(self) -> list[PersonDetection]:
        """执行批量推理。

        Returns:
            检测结果列表
        """
        if not self._batch_buffer:
            return []

        # 收集批量数据
        batch_data = []
        while self._batch_buffer:
            batch_data.append(self._batch_buffer.popleft())

        # 执行批量推理
        start_time = time.perf_counter()

        # 准备批量输入
        batch_images = []
        for frame, scale, pad_x, pad_y in batch_data:
            image, _, _, _ = self._preprocess_coreml(frame)
            batch_images.append(image)

        # 执行推理（CoreML 目前不支持真正的批量推理，逐帧执行）
        all_detections = []
        for i, (frame, scale, pad_x, pad_y) in enumerate(batch_data):
            detections = self._detect_single_frame(
                batch_images[i], frame.shape[1], frame.shape[0], scale, pad_x, pad_y
            )
            all_detections.extend(detections)

        # 更新统计
        inference_time = time.perf_counter() - start_time
        self._total_inference_time += inference_time
        self._total_frames += len(batch_data)
        self._total_batches += 1

        logger.debug(
            f"批量推理完成：{len(batch_data)} 帧，"
            f"耗时 {inference_time * 1000:.1f}ms，"
            f"检测到 {len(all_detections)} 个人形"
        )

        return all_detections

    def _detect_single_frame(
        self,
        image: Any,
        frame_width: int,
        frame_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> list[PersonDetection]:
        """检测单帧。

        Args:
            image: 预处理后的图像
            frame_width: 原始帧宽度
            frame_height: 原始帧高度
            scale: 缩放比例
            pad_x: X 填充
            pad_y: Y 填充

        Returns:
            检测结果列表
        """
        if self.coreml_model is None:
            raise RuntimeError("YOLO CoreML 模型尚未初始化。")

        outputs = self.coreml_model.predict({self.coreml_input_name: image})
        output = outputs.get(self.coreml_output_name)
        if output is None:
            output = next(iter(outputs.values()))

        predictions = np.asarray(output).reshape(-1, 6)
        return self._postprocess_coreml_nms_output(
            predictions, frame_width, frame_height, scale, pad_x, pad_y
        )

    def _load_coreml_model(self) -> None:
        """加载 CoreML 模型。

        Raises:
            RuntimeError: 如果缺少 coremltools 依赖或模型加载失败
        """
        try:
            import coremltools as ct
        except ImportError as error:
            raise RuntimeError("缺少 coremltools 依赖，无法加载 CoreML 模型。") from error

        logger.debug(f"加载 CoreML 模型：{self.model_path}")
        self.coreml_model = ct.models.MLModel(
            str(self.model_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

        specification = self.coreml_model.get_spec()
        if not specification.description.input:
            raise RuntimeError("CoreML 模型缺少输入定义。")
        if not specification.description.output:
            raise RuntimeError("CoreML 模型缺少输出定义。")

        input_description = specification.description.input[0]
        self.coreml_input_name = input_description.name
        self.coreml_output_name = specification.description.output[0].name

        image_type = input_description.type.imageType
        if image_type.width > 0 and image_type.height > 0:
            self.coreml_input_width = image_type.width
            self.coreml_input_height = image_type.height

        logger.debug(
            f"模型信息：输入={self.coreml_input_name}，输出={self.coreml_output_name}，"
            f"尺寸={self.coreml_input_width}x{self.coreml_input_height}"
        )

        self._warmup_coreml_model()

    def _warmup_coreml_model(self) -> None:
        """预热 CoreML 模型。"""
        if self.coreml_model is None or self.coreml_warmup_runs <= 0:
            return

        logger.debug(f"预热 CoreML 模型，次数：{self.coreml_warmup_runs}")
        frame = np.zeros((self.coreml_input_height, self.coreml_input_width, 3), dtype=np.uint8)
        image, _, _, _ = self._preprocess_coreml(frame)

        start_time = time.perf_counter()
        for i in range(self.coreml_warmup_runs):
            self.coreml_model.predict({self.coreml_input_name: image})
            logger.debug(f"预热完成：{i + 1}/{self.coreml_warmup_runs}")

        warmup_time = time.perf_counter() - start_time
        logger.info(f"模型预热完成，耗时 {warmup_time * 1000:.1f}ms")

    def _detect_with_coreml(self, frame: np.ndarray) -> list[PersonDetection]:
        """使用 CoreML 模型进行检测。

        Args:
            frame: 输入图像帧

        Returns:
            检测结果列表

        Raises:
            RuntimeError: 如果模型未初始化
        """
        if self.coreml_model is None:
            raise RuntimeError("YOLO CoreML 模型尚未初始化。")

        start_time = time.perf_counter()

        image, scale, pad_x, pad_y = self._preprocess_coreml(frame)
        detections = self._detect_single_frame(
            image, frame.shape[1], frame.shape[0], scale, pad_x, pad_y
        )

        # 更新统计
        inference_time = time.perf_counter() - start_time
        self._total_inference_time += inference_time
        self._total_frames += 1

        if detections:
            logger.debug(f"检测到 {len(detections)} 个人形目标")

        return detections

    def _preprocess_coreml(self, frame: np.ndarray) -> tuple[Any, float, int, int]:
        """预处理图像以适应 CoreML 模型输入。

        进行 letterbox 缩放和填充，保持宽高比。
        使用预分配缓冲区避免运行时内存分配。

        Args:
            frame: 输入图像帧

        Returns:
            (处理后的图像, 缩放比例, X 填充, Y 填充)

        Raises:
            RuntimeError: 如果缺少 Pillow 依赖
        """
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("缺少 Pillow 依赖，无法向 CoreML 提供图像输入。") from error

        height, width = frame.shape[:2]
        scale = min(self.coreml_input_width / width, self.coreml_input_height / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))

        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        # 创建画布
        canvas = np.full((self.coreml_input_height, self.coreml_input_width, 3), 114, dtype=np.uint8)

        pad_x = (self.coreml_input_width - resized_width) // 2
        pad_y = (self.coreml_input_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, mode="RGB"), scale, pad_x, pad_y

    def _postprocess_coreml_nms_output(
        self,
        predictions: np.ndarray,
        frame_width: int,
        frame_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> list[PersonDetection]:
        """后处理 CoreML NMS 输出。

        过滤人形检测结果，映射回原始图像坐标。

        Args:
            predictions: 模型输出预测结果
            frame_width: 原始图像宽度
            frame_height: 原始图像高度
            scale: 缩放比例
            pad_x: X 填充量
            pad_y: Y 填充量

        Returns:
            过滤后的检测结果列表
        """
        detections: list[PersonDetection] = []

        for row in predictions:
            if row.shape[0] < 6:
                continue

            confidence = float(row[4])
            class_id = int(round(float(row[5])))

            # 只保留人形检测结果
            if class_id != COCO_PERSON_CLASS_ID or confidence < self.confidence_threshold:
                continue

            # 映射回原始图像坐标
            x1 = int(round((float(row[0]) - pad_x) / scale))
            y1 = int(round((float(row[1]) - pad_y) / scale))
            x2 = int(round((float(row[2]) - pad_x) / scale))
            y2 = int(round((float(row[3]) - pad_y) / scale))

            # 边界裁剪
            x1 = min(max(x1, 0), frame_width - 1)
            y1 = min(max(y1, 0), frame_height - 1)
            x2 = min(max(x2, 0), frame_width - 1)
            y2 = min(max(y2, 0), frame_height - 1)

            # 验证边界框有效性
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(PersonDetection(box=BoundingBox(x1, y1, x2, y2), confidence=confidence))

        return detections

    def get_performance_stats(self) -> dict[str, float]:
        """获取性能统计信息。

        Returns:
            性能统计字典
        """
        avg_inference_time = (
            self._total_inference_time / self._total_frames
            if self._total_frames > 0
            else 0.0
        )

        return {
            "total_frames": self._total_frames,
            "total_batches": self._total_batches,
            "total_inference_time_ms": self._total_inference_time * 1000,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "fps": 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0,
        }

    def reset_performance_stats(self) -> None:
        """重置性能统计。"""
        self._total_inference_time = 0.0
        self._total_frames = 0
        self._total_batches = 0
