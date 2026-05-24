from __future__ import annotations

import csv

import numpy as np

from monitor_scan.results.writer import ResultWriter, format_timestamp, timestamp_for_filename
from monitor_scan.types import BoundingBox, PersonDetection


def test_format_timestamp_uses_video_time_format():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(65.9) == "00:01:05"
    assert format_timestamp(3661) == "01:01:01"


def test_timestamp_for_filename_replaces_colons():
    assert timestamp_for_filename("01:02:03") == "01-02-03"


def test_save_event_writes_csv_and_unique_snapshots(tmp_path):
    writer = ResultWriter(tmp_path / "output_results")
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    detections = [PersonDetection(BoundingBox(10, 12, 40, 50), 0.86)]

    first = writer.save_event(tmp_path / "camera1.mp4", "00:00:03", frame, detections)
    second = writer.save_event(tmp_path / "camera1.mp4", "00:00:03", frame, detections)

    assert first.snapshot_path.endswith("output_results/snapshots/camera1_00-00-03.jpg")
    assert second.snapshot_path.endswith("output_results/snapshots/camera1_00-00-03_2.jpg")

    with writer.csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    assert rows[0] == ["视频文件名", "事件发生时间", "AI 置信度", "截图文件路径"]
    assert rows[1] == ["camera1.mp4", "00:00:03", "86%", first.snapshot_path]
    assert rows[2] == ["camera1.mp4", "00:00:03", "86%", second.snapshot_path]
