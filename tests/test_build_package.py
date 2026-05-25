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


def test_write_macos_launcher_creates_executable_command(tmp_path):
    build_package._write_macos_launcher(tmp_path)

    launcher = tmp_path / "启动.command"
    assert launcher.exists()
    if os.name != "nt":
        assert launcher.stat().st_mode & 0o111
    assert "xattr -dr com.apple.quarantine" in launcher.read_text(encoding="utf-8")
    assert "监控视频智能分析系统.app" in launcher.read_text(encoding="utf-8")
