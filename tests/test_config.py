from __future__ import annotations

from pathlib import Path

from monitor_scan import config


def test_default_model_path_uses_only_coreml_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delattr(config.sys, "_MEIPASS", raising=False)

    assert config.default_model_path() == Path("models/yolo26n-512-fp16-nms.mlpackage")


def test_default_model_path_uses_bundle_model_root(monkeypatch, tmp_path):
    bundle_root = tmp_path / "bundle"
    monkeypatch.setattr(config.sys, "_MEIPASS", str(bundle_root), raising=False)

    assert config.default_model_path() == bundle_root / "models" / "yolo26n-512-fp16-nms.mlpackage"


def test_app_config_accepts_m1_defaults():
    app_config = config.AppConfig()

    app_config.validate()
    assert app_config.model_path == Path("models/yolo26n-512-fp16-nms.mlpackage")
    assert app_config.image_size == 512
    assert app_config.max_candidate_frames_per_slot == 4  # 优化后增加到 4
    assert app_config.max_scheduled_detections_per_slot == 2  # 优化后增加到 2
    assert app_config.max_motion_detections_per_slot == 2  # 优化后增加到 2
    assert app_config.motion_resize_width == 480
    assert app_config.motion_detect_shadows is False
