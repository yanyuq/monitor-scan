from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from monitor_scan.config import AppConfig
from monitor_scan.results.writer import ResultWriter
from monitor_scan.types import DetectionEvent
from monitor_scan.video.analyzer import VideoAnalyzer, VideoProgress

logger = logging.getLogger(__name__)


class AnalyzerWorker(QObject):
    """视频分析工作线程，负责在后台执行视频分析任务。

    该类在独立线程中运行，通过 Qt 信号与主线程通信。
    使用线程锁确保停止标志的线程安全性。

    Signals:
        log: 日志消息
        file_status: 文件处理状态更新 (路径, 状态, 进度)
        overall_progress: 总体进度更新 (已完成数, 总数)
        detection: 检测到人形事件
        error: 错误消息
        finished: 任务完成信号
    """

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
        self._stop_lock = threading.Lock()

    def is_stopped(self) -> bool:
        """检查是否已请求停止（线程安全）。

        Returns:
            如果已请求停止则返回 True
        """
        with self._stop_lock:
            return self._stop_requested

    @pyqtSlot()
    def run(self) -> None:
        """执行视频分析任务。"""
        done = 0
        total = len(self.videos)
        logger.info(f"开始分析任务，共 {total} 个视频文件")

        try:
            if not self.videos:
                self.log.emit("没有找到可分析的视频文件。")
                logger.warning("没有找到可分析的视频文件")
                return

            if not self.config.model_path.exists():
                error_msg = f"未找到 YOLO 模型：{self.config.model_path}"
                self.error.emit(error_msg)
                logger.error(error_msg)
                return

            writer = ResultWriter(self.config.output_directory)
            writer.prepare()
            analyzer = VideoAnalyzer(self.config, result_writer=writer)
            self.overall_progress.emit(done, total)

            for video in self.videos:
                if self.is_stopped():
                    self.log.emit("分析任务已停止。")
                    logger.info("分析任务已被用户停止")
                    break

                logger.info(f"开始处理视频：{video.name}")
                self.log.emit(f"正在处理：{video.name}")
                self.file_status.emit(str(video), "正在处理", 0)

                try:
                    analyzer.analyze_video(
                        video,
                        self,
                        progress_callback=self._on_video_progress,
                        detection_callback=self._on_detection,
                    )
                except (OSError, cv2.error) as exc:
                    error_msg = f"{video.name} 分析失败：{exc}"
                    self.file_status.emit(str(video), "分析失败", 0)
                    self.error.emit(error_msg)
                    logger.error(error_msg, exc_info=True)
                    continue
                except Exception as exc:
                    error_msg = f"{video.name} 发生未知错误：{exc}"
                    self.file_status.emit(str(video), "分析失败", 0)
                    self.error.emit(error_msg)
                    logger.exception(error_msg)
                    continue

                if self.is_stopped():
                    self.file_status.emit(str(video), "已停止", 0)
                    self.log.emit(f"已停止：{video.name}")
                    logger.info(f"视频处理已停止：{video.name}")
                    break

                done += 1
                self.file_status.emit(str(video), "已完成", 100)
                self.overall_progress.emit(done, total)
                self.log.emit(f"已完成：{video.name}")
                logger.info(f"视频处理完成：{video.name} ({done}/{total})")

        except Exception as exc:
            error_msg = f"分析任务发生严重错误：{exc}"
            self.error.emit(error_msg)
            logger.exception(error_msg)
        finally:
            logger.info(f"分析任务结束，共完成 {done}/{total} 个视频")
            self.finished.emit()

    @pyqtSlot()
    def stop(self) -> None:
        """请求停止分析任务（线程安全）。"""
        with self._stop_lock:
            self._stop_requested = True
            logger.info("已收到停止分析任务的请求")

    def _on_video_progress(self, progress: VideoProgress) -> None:
        """处理视频进度更新回调。"""
        self.file_status.emit(str(progress.video_path), "正在处理", progress.progress)

    def _on_detection(self, event: DetectionEvent) -> None:
        """处理检测到人形事件的回调。"""
        self.detection.emit(event)
        log_msg = (
            f"检测到人形：{event.video_name} {event.timestamp}，"
            f"置信度 {event.confidence:.0%}，截图 {event.snapshot_path}"
        )
        self.log.emit(log_msg)
        logger.info(log_msg)
