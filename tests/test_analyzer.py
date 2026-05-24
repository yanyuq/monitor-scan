from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from monitor_scan.config import AppConfig
from monitor_scan.types import BoundingBox, PersonDetection
from monitor_scan.video.analyzer import VideoAnalyzer


class NeverStop:
    def is_stopped(self) -> bool:
        return False


class StopImmediately:
    def is_stopped(self) -> bool:
        return True


class AlwaysMotionDetector:
    def __init__(self) -> None:
        self.calls = 0

    def has_motion(self, frame) -> bool:
        self.calls += 1
        return True


class FakePersonDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame) -> list[PersonDetection]:
        self.calls += 1
        return [PersonDetection(BoundingBox(1, 1, 20, 20), 0.9)]


def make_video(path: Path, frame_count: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_analyze_video_uses_motion_then_detector_and_writes_events(tmp_path):
    video_path = tmp_path / "sample.mp4"
    make_video(video_path)
    detector = FakePersonDetector()
    progress_values: list[int] = []
    events_seen = []
    analyzer = VideoAnalyzer(
        AppConfig(sample_fps=2.0, output_directory=tmp_path / "output_results"),
        motion_detector_factory=AlwaysMotionDetector,
        person_detector=detector,
    )

    events = analyzer.analyze_video(
        video_path,
        NeverStop(),
        progress_callback=lambda progress: progress_values.append(progress.progress),
        detection_callback=events_seen.append,
    )

    assert detector.calls >= 1
    assert len(events) >= 1
    assert events == events_seen
    assert progress_values[-1] == 100
    assert (tmp_path / "output_results" / "snapshots").exists()


def test_analyze_video_respects_stop_token(tmp_path):
    video_path = tmp_path / "sample.mp4"
    make_video(video_path)
    detector = FakePersonDetector()
    analyzer = VideoAnalyzer(
        AppConfig(sample_fps=2.0, output_directory=tmp_path / "output_results"),
        motion_detector_factory=AlwaysMotionDetector,
        person_detector=detector,
    )

    events = analyzer.analyze_video(video_path, StopImmediately())

    assert events == []
    assert detector.calls == 0


class BrokenMiddleFrameCapture:
    def __init__(self, path: str, api_preference: int | None = None) -> None:
        self.api_preference = api_preference
        self.position = 0
        self.frames = [
            np.zeros((48, 64, 3), dtype=np.uint8),
            None,
            np.full((48, 64, 3), 120, dtype=np.uint8),
        ]
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == cv2.CAP_PROP_FPS:
            return 1.0
        return 0.0

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        if frame is None:
            return False, None
        self.position += 1
        return True, frame

    def set(self, prop: int, value: float) -> bool:
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        self.position = int(value)
        return True

    def release(self) -> None:
        self.released = True


def test_analyze_video_skips_broken_frame_and_continues(tmp_path):
    detector = FakePersonDetector()
    analyzer = VideoAnalyzer(
        AppConfig(sample_fps=1.0, output_directory=tmp_path / "output_results"),
        motion_detector_factory=AlwaysMotionDetector,
        person_detector=detector,
    )

    with patch("monitor_scan.video.analyzer.cv2.VideoCapture", BrokenMiddleFrameCapture):
        events = analyzer.analyze_video(tmp_path / "broken.mp4", NeverStop())

    assert detector.calls == 2
    assert [event.timestamp for event in events] == ["00:00:00", "00:00:02"]


class BrokenFirstFrameInSecondCapture:
    def __init__(self, path: str, api_preference: int | None = None) -> None:
        self.position = 0
        self.frames = [
            None,
            np.full((48, 64, 3), 120, dtype=np.uint8),
            np.full((48, 64, 3), 140, dtype=np.uint8),
        ]

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        if frame is None:
            return False, None
        self.position += 1
        return True, frame

    def set(self, prop: int, value: float) -> bool:
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        self.position = int(value)
        return True

    def release(self) -> None:
        pass


def test_analyze_video_uses_next_valid_frame_in_same_sample_slot(tmp_path):
    detector = FakePersonDetector()
    analyzer = VideoAnalyzer(
        AppConfig(sample_fps=1.0, output_directory=tmp_path / "output_results"),
        motion_detector_factory=AlwaysMotionDetector,
        person_detector=detector,
    )

    with patch("monitor_scan.video.analyzer.cv2.VideoCapture", BrokenFirstFrameInSecondCapture):
        events = analyzer.analyze_video(tmp_path / "broken-start.mp4", NeverStop())

    assert detector.calls == 1
    assert [event.timestamp for event in events] == ["00:00:00"]


def test_open_capture_prefers_ffmpeg_backend(tmp_path):
    calls = []

    class OpenedCapture:
        def __init__(self, *args) -> None:
            calls.append(args)

        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            pass

    analyzer = VideoAnalyzer(AppConfig(output_directory=tmp_path / "output_results"))

    with patch("monitor_scan.video.analyzer.cv2.VideoCapture", OpenedCapture):
        capture = analyzer._open_capture(tmp_path / "sample.mp4")

    assert capture.isOpened()
    assert calls == [(str(tmp_path / "sample.mp4"), cv2.CAP_FFMPEG)]
