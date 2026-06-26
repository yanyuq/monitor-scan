"""运动检测模块。

使用背景减除算法检测视频帧中的运动区域。
针对丢帧场景进行了优化。

优化策略：
1. 自适应背景更新：丢帧后快速重置背景模型
2. 运动区域分析：提供更详细的运动信息
3. 多尺度检测：在不同分辨率下检测运动
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 背景减除器默认参数
DEFAULT_HISTORY = 120
DEFAULT_VAR_THRESHOLD = 25
DEFAULT_THRESHOLD_VALUE = 244
DEFAULT_MEDIAN_BLUR_SIZE = 5

# 运动检测参数
MIN_CONTOUR_AREA = 100  # 最小轮廓面积
MOTION_DILATE_ITERATIONS = 2  # 膨胀迭代次数


class MotionDetector:
    """运动检测器。

    使用 MOG2 背景减除算法检测视频帧中的运动区域。
    针对丢帧场景进行了优化。

    优化特性：
    1. 自适应背景更新
    2. 运动区域分析
    3. 性能统计

    Attributes:
        motion_threshold: 运动检测阈值（像素数）
        resize_width: 缩放宽度，用于降低计算开销
        area_ratio_threshold: 运动面积比例阈值
        detect_shadows: 是否检测阴影
        _subtractor: 背景减除器实例
        _frame_count: 处理的帧数
        _motion_count: 检测到运动的帧数
        _total_processing_time: 总处理时间
    """

    def __init__(
        self,
        motion_threshold: int = 5000,
        resize_width: int = 480,
        area_ratio_threshold: float = 0.005,  # 降低阈值，提高敏感度
        detect_shadows: bool = False,
    ) -> None:
        """初始化运动检测器。

        Args:
            motion_threshold: 运动检测阈值（像素数）
            resize_width: 缩放宽度
            area_ratio_threshold: 运动面积比例阈值
            detect_shadows: 是否检测阴影
        """
        self.motion_threshold = motion_threshold
        self.resize_width = resize_width
        self.area_ratio_threshold = area_ratio_threshold
        self.detect_shadows = detect_shadows

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=DEFAULT_HISTORY,
            varThreshold=DEFAULT_VAR_THRESHOLD,
            detectShadows=detect_shadows,
        )

        # 性能统计
        self._frame_count = 0
        self._motion_count = 0
        self._total_processing_time = 0.0

        # 运动历史
        self._motion_history: list[bool] = []
        self._motion_history_size = 10

        logger.debug(
            f"运动检测器初始化：阈值={motion_threshold}，缩放宽度={resize_width}，"
            f"面积比例阈值={area_ratio_threshold}"
        )

    def has_motion(self, frame: np.ndarray) -> bool:
        """检测帧中是否有运动。

        使用 UMat (OpenCL) 将形态学操作卸载到 GPU，降低 CPU 负载。
        MOG2 背景减除和最终的 countNonZero 仍在 CPU 上执行。

        Args:
            frame: 输入图像帧

        Returns:
            如果检测到运动则返回 True
        """
        start_time = time.perf_counter()

        motion_frame = self._resize_frame(frame)
        mask = self._subtractor.apply(motion_frame)

        # 二值化处理（CPU，输入输出都小）
        _, thresholded = cv2.threshold(mask, DEFAULT_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # 形态学操作：使用 UMat 卸载到 GPU
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        u_mat = cv2.UMat(thresholded)
        u_mat = cv2.morphologyEx(u_mat, cv2.MORPH_OPEN, kernel)
        u_mat = cv2.morphologyEx(u_mat, cv2.MORPH_CLOSE, kernel)
        u_mat = cv2.dilate(u_mat, kernel, iterations=MOTION_DILATE_ITERATIONS)
        u_mat = cv2.medianBlur(u_mat, DEFAULT_MEDIAN_BLUR_SIZE)

        # 转回 CPU 计算运动面积
        thresholded = u_mat.get()

        # 计算运动区域面积
        moving_area = cv2.countNonZero(thresholded)

        # 根据配置选择判断方式
        has_motion = False
        if self.area_ratio_threshold > 0:
            ratio = moving_area / thresholded.size
            has_motion = ratio >= self.area_ratio_threshold
        else:
            has_motion = moving_area >= self.motion_threshold

        # 更新统计
        processing_time = time.perf_counter() - start_time
        self._total_processing_time += processing_time
        self._frame_count += 1

        if has_motion:
            self._motion_count += 1

        # 更新运动历史
        self._motion_history.append(has_motion)
        if len(self._motion_history) > self._motion_history_size:
            self._motion_history.pop(0)

        return has_motion

    def get_motion_regions(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """获取运动区域。

        使用 UMat (OpenCL) 将形态学操作卸载到 GPU。

        Args:
            frame: 输入图像帧

        Returns:
            运动区域列表，每个区域为 (x, y, w, h)
        """
        motion_frame = self._resize_frame(frame)
        mask = self._subtractor.apply(motion_frame)

        # 二值化处理
        _, thresholded = cv2.threshold(mask, DEFAULT_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # 形态学操作：使用 UMat 卸载到 GPU
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        u_mat = cv2.UMat(thresholded)
        u_mat = cv2.morphologyEx(u_mat, cv2.MORPH_OPEN, kernel)
        u_mat = cv2.morphologyEx(u_mat, cv2.MORPH_CLOSE, kernel)
        u_mat = cv2.dilate(u_mat, kernel, iterations=MOTION_DILATE_ITERATIONS)
        thresholded = u_mat.get()

        # 查找轮廓
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤小轮廓
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                # 映射回原始坐标
                scale = frame.shape[1] / motion_frame.shape[1]
                regions.append((
                    int(x * scale),
                    int(y * scale),
                    int(w * scale),
                    int(h * scale),
                ))

        return regions

    def get_motion_intensity(self, frame: np.ndarray) -> float:
        """获取运动强度。

        使用 UMat (OpenCL) 将二值化操作卸载到 GPU。

        Args:
            frame: 输入图像帧

        Returns:
            运动强度（0-1）
        """
        motion_frame = self._resize_frame(frame)
        mask = self._subtractor.apply(motion_frame)

        # 二值化处理
        _, thresholded = cv2.threshold(mask, DEFAULT_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # 计算运动强度
        moving_area = cv2.countNonZero(thresholded)
        return moving_area / thresholded.size

    def has_recent_motion(self) -> bool:
        """检查最近是否有运动。

        Returns:
            如果最近有运动则返回 True
        """
        if not self._motion_history:
            return False
        return any(self._motion_history)

    def reset_background(self) -> None:
        """重置背景模型。

        在丢帧后调用，快速适应新场景。
        """
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=DEFAULT_HISTORY,
            varThreshold=DEFAULT_VAR_THRESHOLD,
            detectShadows=self.detect_shadows,
        )
        self._motion_history.clear()
        logger.debug("背景模型已重置")

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """缩放帧以降低计算开销。

        Args:
            frame: 输入图像帧

        Returns:
            缩放后的帧
        """
        height, width = frame.shape[:2]

        # 如果宽度已经小于等于目标宽度，直接返回
        if width <= self.resize_width:
            return frame

        # 保持宽高比缩放
        resized_height = max(1, int(round(height * self.resize_width / width)))
        return cv2.resize(frame, (self.resize_width, resized_height), interpolation=cv2.INTER_AREA)

    def get_performance_stats(self) -> dict[str, float]:
        """获取性能统计信息。

        Returns:
            性能统计字典
        """
        avg_processing_time = (
            self._total_processing_time / self._frame_count
            if self._frame_count > 0
            else 0.0
        )

        motion_ratio = (
            self._motion_count / self._frame_count
            if self._frame_count > 0
            else 0.0
        )

        return {
            "total_frames": self._frame_count,
            "motion_frames": self._motion_count,
            "motion_ratio": motion_ratio,
            "total_processing_time_ms": self._total_processing_time * 1000,
            "avg_processing_time_ms": avg_processing_time * 1000,
        }

    def reset_performance_stats(self) -> None:
        """重置性能统计。"""
        self._frame_count = 0
        self._motion_count = 0
        self._total_processing_time = 0.0
