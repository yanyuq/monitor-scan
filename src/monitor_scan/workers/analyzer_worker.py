from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from monitor_scan.config import AppConfig
from monitor_scan.results.writer import ResultWriter
from monitor_scan.types import DetectionEvent
from monitor_scan.video.analyzer import VideoAnalyzer, VideoProgress


class AnalyzerWorker(QObject):
    log = pyqtSignal(str)
    file_status = pyqtSignal(str, str, int)
    overall_progress = pyqtSignal(int, int)
    detection = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, videos: list[Path], config: AppConfig) -> None:
        super().__init__()
        self.videos = videos
        self.config = config
        self._stop_requested = False

    def is_stopped(self) -> bool:
        return self._stop_requested

    @pyqtSlot()
    def run(self) -> None:
        done = 0
        total = len(self.videos)
        try:
            if not self.videos:
                self.log.emit("没有找到可分析的视频文件。")
                return
            if not self.config.model_path.exists():
                self.error.emit(f"未找到 YOLO ONNX 模型：{self.config.model_path}")
                return

            writer = ResultWriter(self.config.output_directory)
            writer.prepare()
            analyzer = VideoAnalyzer(self.config, result_writer=writer)
            self.overall_progress.emit(done, total)

            for video in self.videos:
                if self.is_stopped():
                    self.log.emit("分析任务已停止。")
                    break

                self.log.emit(f"正在处理：{video.name}")
                self.file_status.emit(str(video), "正在处理", 0)
                try:
                    analyzer.analyze_video(
                        video,
                        self,
                        progress_callback=self._on_video_progress,
                        detection_callback=self._on_detection,
                    )
                except Exception as exc:
                    self.file_status.emit(str(video), "分析失败", 0)
                    self.error.emit(f"{video.name} 分析失败：{exc}")
                    continue

                if self.is_stopped():
                    self.file_status.emit(str(video), "已停止", 0)
                    self.log.emit(f"已停止：{video.name}")
                    break

                done += 1
                self.file_status.emit(str(video), "已完成", 100)
                self.overall_progress.emit(done, total)
                self.log.emit(f"已完成：{video.name}")
        finally:
            self.finished.emit()

    @pyqtSlot()
    def stop(self) -> None:
        self._stop_requested = True

    def _on_video_progress(self, progress: VideoProgress) -> None:
        self.file_status.emit(str(progress.video_path), "正在处理", progress.progress)

    def _on_detection(self, event: DetectionEvent) -> None:
        self.detection.emit(event)
        self.log.emit(
            f"检测到人形：{event.video_name} {event.timestamp}，置信度 {event.confidence:.0%}，截图 {event.snapshot_path}"
        )
