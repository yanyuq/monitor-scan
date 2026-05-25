from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_package.py"
SPEC = importlib.util.spec_from_file_location("build_package", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
build_package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_package)


class FakeStream:
    def __init__(self) -> None:
        self.reconfigure_calls = []

    def reconfigure(self, **kwargs) -> None:
        self.reconfigure_calls.append(kwargs)


def test_configure_standard_streams_uses_utf8(monkeypatch):
    stdout = FakeStream()
    stderr = FakeStream()
    monkeypatch.setattr(build_package.sys, "stdout", stdout)
    monkeypatch.setattr(build_package.sys, "stderr", stderr)

    build_package._configure_standard_streams()

    assert stdout.reconfigure_calls == [{"encoding": "utf-8"}]
    assert stderr.reconfigure_calls == [{"encoding": "utf-8"}]


def test_macos_target_arch_accepts_supported_targets():
    assert build_package._macos_target_arch("macos-arm64") == "arm64"
    assert build_package._macos_target_arch("linux-x86_64") is None


def test_macos_target_arch_rejects_unknown_arch():
    with pytest.raises(SystemExit, match="不支持的 macOS 架构"):
        build_package._macos_target_arch("macos-x86_64")


def test_model_path_for_target_uses_coreml_for_macos(monkeypatch):
    monkeypatch.setattr(build_package.sys, "platform", "linux")
    assert build_package._model_path_for_target("macos-arm64") == build_package.YOLO_COREML_MODEL_PATH


def test_model_path_for_target_uses_coreml_for_local_on_macos(monkeypatch):
    monkeypatch.setattr(build_package.sys, "platform", "darwin")
    assert build_package._model_path_for_target("local") == build_package.YOLO_COREML_MODEL_PATH


def test_model_path_for_target_uses_pt_for_non_macos(monkeypatch):
    monkeypatch.setattr(build_package.sys, "platform", "linux")
    assert build_package._model_path_for_target("linux-x86_64") == build_package.YOLO_SOURCE_MODEL_PATH
    assert build_package._model_path_for_target("local") == build_package.YOLO_SOURCE_MODEL_PATH


def test_model_data_destination_preserves_mlpackage_directory(tmp_path):
    coreml_model = tmp_path / "yolo26n.mlpackage"
    pt_model = tmp_path / "yolo26n.pt"
    coreml_model.mkdir()
    pt_model.write_text("model", encoding="utf-8")

    assert build_package._model_data_destination(coreml_model) == "models/yolo26n.mlpackage"
    assert build_package._model_data_destination(pt_model) == "models"


def test_ensure_coreml_model_runs_export_when_missing(tmp_path, monkeypatch):
    source = tmp_path / "models" / "yolo26n.pt"
    output = tmp_path / "models" / "yolo26n.mlpackage"
    export_script = tmp_path / "scripts" / "export_coreml_model.py"
    source.parent.mkdir(parents=True)
    export_script.parent.mkdir(parents=True)
    source.write_text("model", encoding="utf-8")
    calls = []
    monkeypatch.setattr(build_package, "YOLO_SOURCE_MODEL_PATH", source)
    monkeypatch.setattr(build_package, "YOLO_COREML_MODEL_PATH", output)
    monkeypatch.setattr(build_package, "COREML_EXPORT_SCRIPT", export_script)
    monkeypatch.setattr(build_package.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    build_package._ensure_coreml_model()

    assert len(calls) == 1
    command = calls[0][0][0]
    assert command[:2] == [build_package.sys.executable, str(export_script)]
    assert "--source" in command
    assert str(source) in command
    assert "--output" in command
    assert str(output) in command
    assert calls[0][1]["cwd"] == build_package.REPO_ROOT
    assert calls[0][1]["check"] is True


def test_ensure_coreml_model_skips_current_output(tmp_path, monkeypatch):
    source = tmp_path / "models" / "yolo26n.pt"
    output = tmp_path / "models" / "yolo26n.mlpackage"
    source.parent.mkdir(parents=True)
    source.write_text("model", encoding="utf-8")
    output.mkdir()
    os.utime(source, (1, 1))
    os.utime(output, (2, 2))
    monkeypatch.setattr(build_package, "YOLO_SOURCE_MODEL_PATH", source)
    monkeypatch.setattr(build_package, "YOLO_COREML_MODEL_PATH", output)
    monkeypatch.setattr(
        build_package.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("CoreML 模型未过期时不应重新导出"),
    )

    build_package._ensure_coreml_model()


def test_write_macos_launcher_creates_executable_command(tmp_path):
    build_package._write_macos_launcher(tmp_path)

    launcher = tmp_path / "启动.command"
    assert launcher.exists()
    if os.name != "nt":
        assert launcher.stat().st_mode & 0o111
    assert "xattr -dr com.apple.quarantine" in launcher.read_text(encoding="utf-8")
    assert "监控视频智能分析系统.app" in launcher.read_text(encoding="utf-8")
