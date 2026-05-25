from __future__ import annotations

from pathlib import Path

from monitor_scan import config


def test_default_model_path_uses_coreml_on_macos_when_available(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    (model_dir / "yolo26n.mlpackage").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config.sys, "platform", "darwin")
    monkeypatch.delattr(config.sys, "_MEIPASS", raising=False)

    assert config.default_model_path() == Path("models/yolo26n.mlpackage")


def test_default_model_path_falls_back_to_pt_on_macos_without_coreml(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config.sys, "platform", "darwin")
    monkeypatch.delattr(config.sys, "_MEIPASS", raising=False)

    assert config.default_model_path() == Path("models/yolo26n.pt")


def test_default_model_path_uses_pt_on_non_macos(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    (model_dir / "yolo26n.mlpackage").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.delattr(config.sys, "_MEIPASS", raising=False)

    assert config.default_model_path() == Path("models/yolo26n.pt")


def test_default_model_path_uses_bundle_model_root(monkeypatch, tmp_path):
    bundle_root = tmp_path / "bundle"
    (bundle_root / "models" / "yolo26n.mlpackage").mkdir(parents=True)
    monkeypatch.setattr(config.sys, "platform", "darwin")
    monkeypatch.setattr(config.sys, "_MEIPASS", str(bundle_root), raising=False)

    assert config.default_model_path() == bundle_root / "models" / "yolo26n.mlpackage"
