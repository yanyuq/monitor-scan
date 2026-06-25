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


def test_validate_macos_apple_silicon_build_accepts_arm64_macos(monkeypatch):
    monkeypatch.setattr(build_package.sys, "platform", "darwin")
    monkeypatch.setattr(build_package.platform, "machine", lambda: "arm64")

    build_package._validate_macos_apple_silicon_build()


def test_validate_macos_apple_silicon_build_rejects_other_platform(monkeypatch):
    monkeypatch.setattr(build_package.sys, "platform", "linux")
    monkeypatch.setattr(build_package.platform, "machine", lambda: "x86_64")

    with pytest.raises(SystemExit, match="仅支持在 macOS Apple Silicon arm64 环境构建运行包"):
        build_package._validate_macos_apple_silicon_build()


def test_package_target_normalizes_local_to_macos_arm64():
    assert build_package._package_target("local") == "macos-arm64"
    assert build_package._package_target("macos-arm64") == "macos-arm64"


def test_model_data_destination_preserves_mlpackage_directory(tmp_path):
    coreml_model = tmp_path / "yolo26n-512-fp16-nms.mlpackage"
    coreml_model.mkdir()

    assert build_package._model_data_destination(coreml_model) == "models/yolo26n-512-fp16-nms.mlpackage"


def test_main_builds_macos_coreml_only_pyinstaller_command(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "yolo26n-512-fp16-nms.mlpackage"
    model_path.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(build_package.sys, "argv", ["build_package.py", "--target", "macos-arm64"])
    monkeypatch.setattr(build_package.sys, "platform", "darwin")
    monkeypatch.setattr(build_package.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(build_package, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(build_package, "YOLO_COREML_MODEL_PATH", model_path)
    monkeypatch.setattr(build_package, "_remove", lambda path: None)
    monkeypatch.setattr(build_package, "_copy_dist_outputs", lambda package_root: None)
    monkeypatch.setattr(build_package, "_copy_if_exists", lambda source, target: None)
    monkeypatch.setattr(build_package, "_sign_macos_apps", lambda package_root: None)
    monkeypatch.setattr(build_package, "_write_macos_launcher", lambda package_root: None)
    monkeypatch.setattr(build_package.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(build_package.shutil, "make_archive", lambda *args, **kwargs: str(tmp_path / "release.tar.gz"))

    build_package.main()

    assert len(calls) == 1
    command = calls[0][0][0]
    assert command[:3] == [build_package.sys.executable, "-m", "PyInstaller"]
    assert "--add-data" in command
    assert f"{model_path}:models/yolo26n-512-fp16-nms.mlpackage" in command
    assert "--target-architecture" in command
    assert command[command.index("--target-architecture") + 1] == "arm64"
    removed_packages = ("onnx" + "runtime", "ultra" + "lytics", "tor" + "ch", "tor" + "ch" + "vision")
    for package in removed_packages:
        assert package not in command[command.index("--collect-all") : command.index("--exclude-module")]
        assert package in command
    assert "coremltools" in command
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][1]["check"] is True


def test_write_macos_launcher_creates_executable_command(tmp_path):
    build_package._write_macos_launcher(tmp_path)

    launcher = tmp_path / "启动.command"
    assert launcher.exists()
    if os.name != "nt":
        assert launcher.stat().st_mode & 0o111
    assert "xattr -dr com.apple.quarantine" in launcher.read_text(encoding="utf-8")
    assert "监控视频智能分析系统.app" in launcher.read_text(encoding="utf-8")
