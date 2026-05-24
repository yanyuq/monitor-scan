from __future__ import annotations

from monitor_scan.video.scanner import VideoScanner


def test_scan_filters_supported_video_extensions_and_sorts_by_name(tmp_path):
    (tmp_path / "b.MP4").write_bytes(b"")
    (tmp_path / "a.avi").write_bytes(b"")
    (tmp_path / "note.txt").write_text("忽略", encoding="utf-8")
    (tmp_path / "nested").mkdir()

    videos = VideoScanner().scan(tmp_path)

    assert [video.name for video in videos] == ["a.avi", "b.MP4"]


def test_scan_returns_empty_list_for_empty_directory(tmp_path):
    assert VideoScanner().scan(tmp_path) == []


def test_scan_rejects_missing_directory(tmp_path):
    missing = tmp_path / "missing"

    try:
        VideoScanner().scan(missing)
    except FileNotFoundError as exc:
        assert "视频目录不存在" in str(exc)
    else:
        raise AssertionError("缺失目录应抛出 FileNotFoundError。")
