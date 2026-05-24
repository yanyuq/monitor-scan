from __future__ import annotations

from monitor_scan.gui.main_window import MainWindow


def test_main_window_loads_directory_and_populates_video_table(qtbot, tmp_path):
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("忽略", encoding="utf-8")
    window = MainWindow()
    qtbot.addWidget(window)

    window.load_directory(tmp_path)

    assert window.video_table.rowCount() == 1
    assert window.video_table.item(0, 0).text() == "camera.mp4"
    assert window.video_table.item(0, 1).text() == "等待中"
    assert window.start_button.isEnabled()
