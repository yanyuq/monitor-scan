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
from monitor_scan.video.remuxer import FfmpegRemuxer, PreparedVideo


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
        remuxer: FfmpegRemuxer | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.motion_detector_factory = motion_detector_factory or self._default_motion_detector
        self.person_detector = person_detector
        self.result_writer = result_writer or ResultWriter(config.output_directory)
        self.remuxer = remuxer or FfmpegRemuxer(config.ffmpeg_path, config.ffmpeg_timeout_seconds)

    def analyze_video(
        self,
        video_path: str | Path,
        stop_token: StopToken,
        progress_callback: Callable[[VideoProgress], None] | None = None,
        detection_callback: Callable[[DetectionEvent], None] | None = None,
    ) -> list[DetectionEvent]:
        source_path = Path(video_path)
        prepared_video = self._prepare_video(source_path)
        capture = self._open_capture(prepared_video.analysis_path)
        try:
            if not capture.isOpened():
                raise OSError(f"视频无法打开：{source_path}")
            return self._analyze_capture(
                capture,
                source_path,
                stop_token,
                progress_callback,
                detection_callback,
            )
        finally:
            capture.release()
            prepared_video.cleanup()

    def _analyze_capture(
        self,
        capture,
        source_path: Path,
        stop_token: StopToken,
        progress_callback: Callable[[VideoProgress], None] | None,
        detection_callback: Callable[[DetectionEvent], None] | None,
    ) -> list[DetectionEvent]:
        events: list[DetectionEvent] = []
        motion_detector = self.motion_detector_factory()
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = capture.get(cv2.CAP_PROP_FPS) or self.config.sample_fps
        frame_index = 0
        last_progress_slot = -1
        detected_slots: set[int] = set()
        scheduled_attempted_candidates: set[tuple[int, int]] = set()
        motion_attempted_candidates: set[tuple[int, int]] = set()
        consecutive_decode_failures = 0

        while not stop_token.is_stopped():
            ok, frame = capture.read()
            if not ok:
                consecutive_decode_failures += 1
                if consecutive_decode_failures >= self.config.max_consecutive_decode_failures:
                    break
                if not self._skip_failed_frame(capture, frame_index, total_frames):
                    break
                frame_index += 1
                if progress_callback is not None:
                    progress_callback(VideoProgress(source_path, self._progress(frame_index, total_frames)))
                continue

            consecutive_decode_failures = 0
            frame_time_seconds = self._frame_time_seconds(capture, frame_index, video_fps)
            sample_slot = self._sample_slot(frame_time_seconds)
            if sample_slot > last_progress_slot:
                last_progress_slot = sample_slot
                if progress_callback is not None:
                    progress_callback(VideoProgress(source_path, self._progress(frame_index, total_frames)))

            has_motion = motion_detector.has_motion(frame)
            if sample_slot not in detected_slots and self._should_try_detection(
                sample_slot,
                frame_time_seconds,
                has_motion,
                scheduled_attempted_candidates,
                motion_attempted_candidates,
            ):
                detections = self._detect_people(frame)
                if detections:
                    timestamp = format_timestamp(frame_time_seconds)
                    event = self.result_writer.save_event(source_path, timestamp, frame, detections)
                    events.append(event)
                    detected_slots.add(sample_slot)
                    if detection_callback is not None:
                        detection_callback(event)

            frame_index += 1

        if progress_callback is not None:
            progress_callback(
                VideoProgress(source_path, 100 if not stop_token.is_stopped() else self._progress(frame_index, total_frames))
            )
        return events

    def _prepare_video(self, source_path: Path) -> PreparedVideo:
        if not self.config.remux_before_analysis:
            return PreparedVideo(source_path=source_path, analysis_path=source_path)
        return self.remuxer.prepare(source_path)

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

    def _frame_time_seconds(self, capture, frame_index: int, video_fps: float) -> float:
        position_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
        if position_msec > 0:
            return position_msec / 1000
        safe_fps = video_fps if video_fps > 0 else self.config.sample_fps
        return frame_index / safe_fps

    def _sample_slot(self, frame_time_seconds: float) -> int:
        return int(frame_time_seconds * self.config.sample_fps)

    def _candidate_bucket(self, frame_time_seconds: float) -> int:
        slot_duration = 1 / self.config.sample_fps
        slot_start = self._sample_slot(frame_time_seconds) * slot_duration
        slot_offset = max(0.0, frame_time_seconds - slot_start)
        bucket = int(slot_offset / slot_duration * self.config.max_candidate_frames_per_slot)
        return min(self.config.max_candidate_frames_per_slot - 1, bucket)

    def _should_try_detection(
        self,
        sample_slot: int,
        frame_time_seconds: float,
        has_motion: bool,
        scheduled_attempted_candidates: set[tuple[int, int]],
        motion_attempted_candidates: set[tuple[int, int]],
    ) -> bool:
        candidate_key = (sample_slot, self._candidate_bucket(frame_time_seconds))
        if candidate_key not in scheduled_attempted_candidates:
            scheduled_attempted_candidates.add(candidate_key)
            return True
        if has_motion and candidate_key not in motion_attempted_candidates:
            motion_attempted_candidates.add(candidate_key)
            return True
        return False

    def _skip_failed_frame(self, capture, frame_index: int, total_frames: int) -> bool:
        return bool(capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index + 1))

    def _progress(self, frame_index: int, total_frames: int) -> int:
        if total_frames <= 0:
            return 0
        return min(99, int(frame_index / total_frames * 100))
