from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "monitor-scan"
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    _configure_standard_streams()
    parser = argparse.ArgumentParser(description="构建监控视频智能分析系统运行包")
    parser.add_argument("--target", required=True, help="运行包目标平台标识，例如 windows-x86_64")
    args = parser.parse_args()

    model_path = REPO_ROOT / "models" / "yolov8n.onnx"
    if not model_path.exists():
        raise SystemExit(f"缺少模型文件，无法打包：{model_path}")

    _remove(REPO_ROOT / "build")
    _remove(REPO_ROOT / "dist")
    release_dir = REPO_ROOT / "release"
    _remove(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    separator = ";" if os.name == "nt" else ":"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        APP_NAME,
        "--paths",
        str(REPO_ROOT / "src"),
        "--add-data",
        f"{model_path}{separator}models",
        "--collect-all",
        "onnxruntime",
        "--collect-all",
        "cv2",
        "--collect-all",
        "imageio_ffmpeg",
        "--collect-submodules",
        "PyQt6",
        str(REPO_ROOT / "src" / "monitor_scan" / "__main__.py"),
    ]
    if sys.platform == "darwin":
        command[command.index("--name") : command.index("--name") + 2] = ["--name", "监控视频智能分析系统"]
        command.extend(["--osx-bundle-identifier", "com.yanyuq.monitor-scan"])
        command.extend(["--codesign-identity", "-"])
        target_arch = _macos_target_arch(args.target)
        if target_arch is not None:
            command.extend(["--target-architecture", target_arch])

    subprocess.run(command, cwd=REPO_ROOT, check=True)

    package_root = release_dir / f"{APP_NAME}-{args.target}"
    package_root.mkdir(parents=True, exist_ok=True)
    _copy_dist_outputs(package_root)
    _copy_if_exists(REPO_ROOT / "README.md", package_root / "README.md")
    if sys.platform == "darwin":
        _sign_macos_apps(package_root)
        _write_macos_launcher(package_root)

    archive_base = release_dir / f"{APP_NAME}-{args.target}"
    if args.target.startswith("windows"):
        archive = shutil.make_archive(str(archive_base), "zip", release_dir, package_root.name)
    else:
        archive = shutil.make_archive(str(archive_base), "gztar", release_dir, package_root.name)

    print(f"已生成运行包：{archive}")
    return 0


def _configure_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def _copy_dist_outputs(package_root: Path) -> None:
    dist_dir = REPO_ROOT / "dist"
    outputs = [path for path in dist_dir.iterdir() if path.name.startswith(APP_NAME) or path.name.startswith("监控视频智能分析系统")]
    if not outputs:
        raise SystemExit("PyInstaller 未生成可打包产物。")

    for output in outputs:
        target = package_root / output.name
        if output.is_dir():
            shutil.copytree(output, target, symlinks=True)
        else:
            shutil.copy2(output, target)


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copy2(source, target)


def _macos_target_arch(target: str) -> str | None:
    if not target.startswith("macos-"):
        return None
    arch = target.removeprefix("macos-")
    if arch != "arm64":
        raise SystemExit(f"不支持的 macOS 架构：{arch}")
    return arch


def _sign_macos_apps(package_root: Path) -> None:
    for app in package_root.glob("*.app"):
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)


def _write_macos_launcher(package_root: Path) -> None:
    launcher = package_root / "启动.command"
    launcher.write_text(
        "#!/bin/sh\n"
        "DIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
        "APP=\"$DIR/监控视频智能分析系统.app\"\n"
        "xattr -dr com.apple.quarantine \"$APP\" 2>/dev/null || true\n"
        "open \"$APP\"\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)


def _remove(path: Path) -> None:
    if path.is_dir():
        try:
            shutil.rmtree(path)
        except OSError:
            for metadata_file in path.rglob(".DS_Store"):
                metadata_file.unlink(missing_ok=True)
            shutil.rmtree(path)
    elif path.exists():
        path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
