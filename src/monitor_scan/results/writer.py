from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from monitor_scan.types import DetectionEvent, PersonDetection

CSV_HEADERS = ["视频文件名", "事件发生时间", "AI 置信度", "截图文件路径"]


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def timestamp_for_filename(timestamp: str) -> str:
    return timestamp.replace(":", "-")


class ResultWriter:
    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)
        self.snapshot_directory = self.output_directory / "snapshots"
        today = datetime.now().strftime("%Y%m%d")
        self.csv_path = self.output_directory / f"检测报告_{today}.csv"
        self._initialized = False

    def prepare(self) -> None:
        self.snapshot_directory.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            self.output_directory.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(CSV_HEADERS)
        self._initialized = True

    def save_event(
        self,
        video_path: str | Path,
        timestamp: str,
        frame: np.ndarray,
        detections: list[PersonDetection],
    ) -> DetectionEvent:
        if not self._initialized:
            self.prepare()
        if not detections:
            raise ValueError("保存事件时至少需要一个人形检测结果。")

        video = Path(video_path)
        annotated = frame.copy()
        for detection in detections:
            box = detection.box
            cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 2)
            label = f"person {detection.confidence:.0%}"
            cv2.putText(
                annotated,
                label,
                (box.x1, max(20, box.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        snapshot_path = self._next_snapshot_path(video, timestamp)
        if not cv2.imwrite(str(snapshot_path), annotated):
            raise OSError(f"截图保存失败：{snapshot_path}")

        confidence = max(detection.confidence for detection in detections)
        snapshot_display_path = snapshot_path.relative_to(self.output_directory.parent).as_posix()
        event = DetectionEvent(
            video_name=video.name,
            timestamp=timestamp,
            confidence=confidence,
            snapshot_path=snapshot_display_path,
        )
        self._append_event(event)
        return event

    def _next_snapshot_path(self, video_path: Path, timestamp: str) -> Path:
        stem = video_path.stem
        time_part = timestamp_for_filename(timestamp)
        candidate = self.snapshot_directory / f"{stem}_{time_part}.jpg"
        if not candidate.exists():
            return candidate

        index = 2
        while True:
            candidate = self.snapshot_directory / f"{stem}_{time_part}_{index}.jpg"
            if not candidate.exists():
                return candidate
            index += 1

    def _append_event(self, event: DetectionEvent) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow([
                event.video_name,
                event.timestamp,
                f"{event.confidence:.0%}",
                event.snapshot_path,
            ])
