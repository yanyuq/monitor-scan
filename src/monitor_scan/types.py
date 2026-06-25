"""类型定义模块。

定义监控视频智能分析系统中使用的所有数据类型。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """边界框。

    表示图像中的一个矩形区域。

    Attributes:
        x1: 左上角 X 坐标
        y1: 左上角 Y 坐标
        x2: 右下角 X 坐标
        y2: 右下角 Y 坐标
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """边界框宽度。"""
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        """边界框高度。"""
        return max(0, self.y2 - self.y1)


@dataclass(frozen=True)
class PersonDetection:
    """人形检测结果。

    Attributes:
        box: 边界框
        confidence: 置信度 (0-1)
    """

    box: BoundingBox
    confidence: float


@dataclass(frozen=True)
class DetectionEvent:
    """检测事件。

    表示在视频中检测到人形的事件。

    Attributes:
        video_name: 视频文件名
        timestamp: 事件时间戳 (HH:MM:SS)
        confidence: 最高置信度 (0-1)
        snapshot_path: 截图文件路径
    """

    video_name: str
    timestamp: str
    confidence: float
    snapshot_path: str
