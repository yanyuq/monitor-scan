from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2

from monitor_scan.ai.motion import MotionDetector
from monitor_scan.config import AppConfig
from monitor_scan.results.writer import ResultWriter, format_timestamp
from monitor_scan.types import DetectionEvent, PersonDetection


class StopToken(Protocol):
    def is_stopped(self) -> bool:
        ...


class PersonDetector(Protocol):
    def detect(self, frame) -> list[PersonDetection]:
        ...


@dataclass(frozen=True)
class VideoProgress:
    video_path: Path
    progress: int


class VideoAnalyzer:
    def __init__(
        self,
        config: AppConfig,
        motion_detector_factory: Callable[[], MotionDetector] | None = None,
        person_detector: PersonDetector | None = None,
        result_writer: ResultWriter | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.motion_detector_factory = motion_detector_factory or self._default_motion_detector
        self.person_detector = person_detector
        self.result_writer = result_writer or ResultWriter(config.output_directory)

    def analyze_video(
        self,
        video_path: str | Path,
        stop_token: StopToken,
        progress_callback: Callable[[VideoProgress], None] | None = None,
        detection_callback: Callable[[DetectionEvent], None] | None = None,
    ) -> list[DetectionEvent]:
        path = Path(video_path)
        capture = self._open_capture(path)
        if not capture.isOpened():
            raise OSError(f"视频无法打开：{path}")

        events: list[DetectionEvent] = []
        motion_detector = self.motion_detector_factory()
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = capture.get(cv2.CAP_PROP_FPS) or self.config.sample_fps
        frame_index = 0
        last_processed_slot = -1

        try:
            while not stop_token.is_stopped() and self._has_remaining_frames(frame_index, total_frames):
                ok, frame = capture.read()
                if not ok:
                    if not self._skip_failed_frame(capture, frame_index, total_frames):
                        break
                    frame_index += 1
                    if progress_callback is not None:
                        progress_callback(VideoProgress(path, self._progress(frame_index, total_frames)))
                    continue

                sample_slot = self._sample_slot(frame_index, video_fps)
                if sample_slot > last_processed_slot:
                    last_processed_slot = sample_slot
                    progress = self._progress(frame_index, total_frames)
                    if progress_callback is not None:
                        progress_callback(VideoProgress(path, progress))

                    if motion_detector.has_motion(frame):
                        detections = self._detect_people(frame)
                        if detections:
                            timestamp = format_timestamp(frame_index / video_fps)
                            event = self.result_writer.save_event(path, timestamp, frame, detections)
                            events.append(event)
                            if detection_callback is not None:
                                detection_callback(event)

                frame_index += 1
        finally:
            capture.release()

        if progress_callback is not None:
            progress_callback(VideoProgress(path, 100 if not stop_token.is_stopped() else self._progress(frame_index, total_frames)))
        return events

    def _detect_people(self, frame) -> list[PersonDetection]:
        if self.person_detector is None:
            from monitor_scan.ai.yolo_detector import YoloPersonDetector

            self.person_detector = YoloPersonDetector(
                self.config.model_path,
                self.config.confidence_threshold,
                self.config.image_size,
                self.config.nms_threshold,
            )
        return self.person_detector.detect(frame)

    def _default_motion_detector(self) -> MotionDetector:
        return MotionDetector(self.config.motion_threshold)

    def _open_capture(self, path: Path):
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if capture.isOpened():
            return capture
        capture.release()
        return cv2.VideoCapture(str(path))

    def _sample_slot(self, frame_index: int, video_fps: float) -> int:
        safe_fps = video_fps if video_fps > 0 else self.config.sample_fps
        return int(frame_index / safe_fps * self.config.sample_fps)

    def _has_remaining_frames(self, frame_index: int, total_frames: int) -> bool:
        return total_frames <= 0 or frame_index < total_frames

    def _skip_failed_frame(self, capture, frame_index: int, total_frames: int) -> bool:
        if total_frames <= 0 or frame_index + 1 >= total_frames:
            return False
        return bool(capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index + 1))

    def _progress(self, frame_index: int, total_frames: int) -> int:
        if total_frames <= 0:
            return 0
        return min(99, int(frame_index / total_frames * 100))
