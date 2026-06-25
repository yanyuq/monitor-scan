from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from monitor_scan.video.remuxer import FfmpegRemuxer


def test_build_command_uses_stream_copy_and_timestamp_repair(tmp_path):
    remuxer = FfmpegRemuxer()
    source_path = tmp_path / "camera.mp4"
    output_path = tmp_path / "camera_remuxed.mp4"

    command = remuxer._build_command("ffmpeg", source_path, output_path)

    assert command[:6] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-err_detect"]
    assert command[command.index("-err_detect") + 1] == "ignore_err"
    assert command[command.index("-fflags") + 1] == "+genpts+discardcorrupt"
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-avoid_negative_ts") + 1] == "make_zero"
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert command[-1] == str(output_path)


def test_prepare_falls_back_when_ffmpeg_is_missing(tmp_path):
    source_path = tmp_path / "camera.mp4"
    source_path.write_bytes(b"video")
    remuxer = FfmpegRemuxer()

    with patch.object(remuxer, "_resolve_ffmpeg", return_value=None):
        prepared = remuxer.prepare(source_path)

    assert prepared.source_path == source_path
    assert prepared.analysis_path == source_path
    assert prepared.temporary_directory is None
    assert "未找到 FFmpeg" in (prepared.message or "")


def test_prepare_falls_back_when_source_is_missing(tmp_path):
    source_path = tmp_path / "missing.mp4"
    remuxer = FfmpegRemuxer()

    with patch.object(remuxer, "_resolve_ffmpeg", return_value="ffmpeg"):
        prepared = remuxer.prepare(source_path)

    assert prepared.analysis_path == source_path
    assert prepared.temporary_directory is None
    assert "视频文件不存在" in (prepared.message or "")


def test_prepare_returns_temporary_remuxed_file_and_cleanup(tmp_path):
    source_path = tmp_path / "camera.mp4"
    source_path.write_bytes(b"video")
    captured: dict[str, object] = {}
    remuxer = FfmpegRemuxer(timeout_seconds=7)

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output_path = Path(command[-1])
        output_path.write_bytes(b"remuxed")
        return SimpleNamespace(returncode=0, stderr="")

    with patch.object(remuxer, "_resolve_ffmpeg", return_value="/usr/bin/ffmpeg"), patch(
        "monitor_scan.video.remuxer.subprocess.run", fake_run
    ):
        prepared = remuxer.prepare(source_path)

    assert prepared.source_path == source_path
    assert prepared.analysis_path != source_path
    assert prepared.analysis_path.exists()
    assert prepared.temporary_directory is not None
    assert prepared.temporary_directory.name.startswith(".monitor_scan_camera_")
    assert prepared.message == "FFmpeg 已完成无重编码索引重建。"
    assert captured["command"][0] == "/usr/bin/ffmpeg"
    assert captured["kwargs"]["timeout"] == 7
    assert "creationflags" not in captured["kwargs"]

    prepared.cleanup()

    assert not prepared.temporary_directory.exists()


def test_prepare_falls_back_and_removes_temporary_directory_when_ffmpeg_fails(tmp_path):
    source_path = tmp_path / "camera.mp4"
    source_path.write_bytes(b"video")
    remuxer = FfmpegRemuxer()

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=1, stderr="bad stream")

    with patch.object(remuxer, "_resolve_ffmpeg", return_value="ffmpeg"), patch(
        "monitor_scan.video.remuxer.subprocess.run", fake_run
    ):
        prepared = remuxer.prepare(source_path)

    assert prepared.analysis_path == source_path
    assert prepared.temporary_directory is None
    assert "bad stream" in (prepared.message or "")
    assert list(tmp_path.glob(".monitor_scan_camera_*")) == []
