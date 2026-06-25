"""检测结果写入模块。

负责保存检测结果，包括截图和 CSV 报告。
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from monitor_scan.types import DetectionEvent, PersonDetection

logger = logging.getLogger(__name__)

# CSV 文件表头
CSV_HEADERS = ["视频文件名", "事件发生时间", "AI 置信度", "截图文件路径"]

# 截图文件扩展名
SNAPSHOT_EXTENSION = ".jpg"

# 日期格式
DATE_FORMAT = "%Y%m%d"

# 红框颜色 (BGR)
BOX_COLOR = (0, 0, 255)

# 文本参数
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
LINE_TYPE = cv2.LINE_AA


def format_timestamp(seconds: float) -> str:
    """格式化时间戳。

    Args:
        seconds: 秒数

    Returns:
        格式化的时间戳字符串 (HH:MM:SS)
    """
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def timestamp_for_filename(timestamp: str) -> str:
    """将时间戳转换为文件名安全格式。

    Args:
        timestamp: 时间戳字符串 (HH:MM:SS)

    Returns:
        文件名安全的时间戳 (HH-MM-SS)
    """
    return timestamp.replace(":", "-")


class ResultWriter:
    """检测结果写入器。

    负责保存检测事件，包括：
    - 带标注框的截图
    - CSV 格式的检测报告

    Attributes:
        output_directory: 输出目录
        snapshot_directory: 截图目录
        csv_path: CSV 报告路径
        _initialized: 是否已初始化
    """

    def __init__(self, output_directory: str | Path) -> None:
        """初始化结果写入器。

        Args:
            output_directory: 输出目录路径
        """
        self.output_directory = Path(output_directory)
        self.snapshot_directory = self.output_directory / "snapshots"
        today = datetime.now().strftime(DATE_FORMAT)
        self.csv_path = self.output_directory / f"检测报告_{today}.csv"
        self._initialized = False

        logger.debug(f"结果写入器初始化：输出目录={output_directory}")

    def prepare(self) -> None:
        """准备输出目录和 CSV 文件。"""
        # 创建截图目录
        self.snapshot_directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"截图目录已准备：{self.snapshot_directory}")

        # 创建 CSV 文件（如果不存在）
        if not self.csv_path.exists():
            self.output_directory.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(CSV_HEADERS)
            logger.debug(f"CSV 报告已创建：{self.csv_path}")

        self._initialized = True

    def save_event(
        self,
        video_path: str | Path,
        timestamp: str,
        frame: np.ndarray,
        detections: list[PersonDetection],
    ) -> DetectionEvent:
        """保存检测事件。

        Args:
            video_path: 视频文件路径
            timestamp: 事件时间戳
            frame: 视频帧
            detections: 检测结果列表

        Returns:
            保存的检测事件

        Raises:
            ValueError: 如果检测结果为空
            OSError: 如果截图保存失败
        """
        if not self._initialized:
            self.prepare()

        if not detections:
            raise ValueError("保存事件时至少需要一个人形检测结果。")

        video = Path(video_path)
        logger.info(f"保存检测事件：{video.name} {timestamp}")

        # 绘制标注框
        annotated = frame.copy()
        for detection in detections:
            box = detection.box
            cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), BOX_COLOR, 2)
            label = f"person {detection.confidence:.0%}"
            cv2.putText(
                annotated,
                label,
                (box.x1, max(20, box.y1 - 8)),
                FONT_FACE,
                FONT_SCALE,
                BOX_COLOR,
                FONT_THICKNESS,
                LINE_TYPE,
            )

        # 保存截图
        snapshot_path = self._next_snapshot_path(video, timestamp)
        if not cv2.imwrite(str(snapshot_path), annotated):
            raise OSError(f"截图保存失败：{snapshot_path}")

        logger.debug(f"截图已保存：{snapshot_path}")

        # 创建事件
        confidence = max(detection.confidence for detection in detections)
        snapshot_display_path = snapshot_path.relative_to(self.output_directory.parent).as_posix()
        event = DetectionEvent(
            video_name=video.name,
            timestamp=timestamp,
            confidence=confidence,
            snapshot_path=snapshot_display_path,
        )

        # 写入 CSV
        self._append_event(event)

        return event

    def _next_snapshot_path(self, video_path: Path, timestamp: str) -> Path:
        """生成下一个截图文件路径。

        如果文件已存在，自动添加序号后缀。

        Args:
            video_path: 视频文件路径
            timestamp: 时间戳

        Returns:
            截图文件路径
        """
        stem = video_path.stem
        time_part = timestamp_for_filename(timestamp)
        candidate = self.snapshot_directory / f"{stem}_{time_part}{SNAPSHOT_EXTENSION}"

        if not candidate.exists():
            return candidate

        # 添加序号后缀
        index = 2
        while True:
            candidate = self.snapshot_directory / f"{stem}_{time_part}_{index}{SNAPSHOT_EXTENSION}"
            if not candidate.exists():
                return candidate
            index += 1

    def _append_event(self, event: DetectionEvent) -> None:
        """将事件追加到 CSV 文件。

        Args:
            event: 检测事件
        """
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow([
                event.video_name,
                event.timestamp,
                f"{event.confidence:.0%}",
                event.snapshot_path,
            ])

        logger.debug(f"事件已写入 CSV：{event.video_name} {event.timestamp}")
