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


def test_main_window_builds_m1_optimized_config(qtbot, tmp_path):
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"")
    window = MainWindow()
    qtbot.addWidget(window)
    window.load_directory(tmp_path)
    window.sample_fps_input.setValue(1.5)
    window.confidence_input.setValue(0.65)

    config = window._analysis_config()

    assert not hasattr(window, "fast" + "_npu_mode_input")
    assert config.sample_fps == 1.5
    assert config.confidence_threshold == 0.65
    assert config.image_size == 512
    assert config.max_candidate_frames_per_slot == 2
    assert config.max_scheduled_detections_per_slot == 1
    assert config.max_motion_detections_per_slot == 1
    assert config.motion_resize_width == 480
    assert config.motion_area_ratio_threshold == 0.01
    assert config.motion_detect_shadows is False
