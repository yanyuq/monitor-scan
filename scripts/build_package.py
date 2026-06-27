from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "monitor-scan"
MACOS_APP_NAME = "监控视频智能分析系统"
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = REPO_ROOT / "models"
YOLO_COREML_MODEL_PATH = MODEL_DIRECTORY / "yolo26n-512-fp16-nms.mlpackage"
SUPPORTED_TARGETS = frozenset({"macos-arm64", "local"})
EXCLUDED_RUNTIME_MODULES = ("onnx" + "runtime", "ultra" + "lytics", "tor" + "ch", "tor" + "ch" + "vision")


def main() -> int:
    _configure_standard_streams()
    parser = argparse.ArgumentParser(description="构建监控视频智能分析系统 macOS Apple Silicon 运行包")
    parser.add_argument("--target", required=True, choices=sorted(SUPPORTED_TARGETS), help="运行包目标平台标识")
    args = parser.parse_args()

    _validate_macos_apple_silicon_build()
    if not YOLO_COREML_MODEL_PATH.exists():
        raise SystemExit(f"缺少 CoreML 模型文件，无法打包：{YOLO_COREML_MODEL_PATH}")

    _remove(REPO_ROOT / "build")
    _remove(REPO_ROOT / "dist")
    release_dir = REPO_ROOT / "release"
    _remove(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        MACOS_APP_NAME,
        "--paths",
        str(REPO_ROOT / "src"),
        "--add-data",
        f"{YOLO_COREML_MODEL_PATH}:models/{YOLO_COREML_MODEL_PATH.name}",
        "--collect-all",
        "cv2",
        "--collect-all",
        "imageio_ffmpeg",
        "--hidden-import",
        "coremltools",
        "--collect-data",
        "coremltools",
        "--collect-submodules",
        "PyQt6",
        "--osx-bundle-identifier",
        "com.yanyuq.monitor-scan",
        "--codesign-identity",
        "-",
        "--target-architecture",
        "arm64",
        str(REPO_ROOT / "src" / "monitor_scan" / "__main__.py"),
    ]

    for module in EXCLUDED_RUNTIME_MODULES:
        command.extend(["--exclude-module", module])

    subprocess.run(command, cwd=REPO_ROOT, check=True)

    package_root = release_dir / f"{APP_NAME}-{_package_target(args.target)}"
    package_root.mkdir(parents=True, exist_ok=True)
    _copy_dist_outputs(package_root)
    _copy_if_exists(REPO_ROOT / "README.md", package_root / "README.md")
    _sign_macos_apps(package_root)
    _write_macos_launcher(package_root)

    archive_base = release_dir / package_root.name
    archive = shutil.make_archive(str(archive_base), "gztar", release_dir, package_root.name)

    print(f"已生成运行包：{archive}")
    return 0


def _validate_macos_apple_silicon_build() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("仅支持在 macOS Apple Silicon arm64 环境构建运行包。")


def _package_target(target: str) -> str:
    return "macos-arm64" if target == "local" else target


def _model_data_destination(model_path: Path) -> str:
    if model_path.is_dir():
        return f"models/{model_path.name}"
    return "models"


def _configure_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def _copy_dist_outputs(package_root: Path) -> None:
    dist_dir = REPO_ROOT / "dist"
    outputs = [path for path in dist_dir.iterdir() if path.name.startswith(APP_NAME) or path.name.startswith(MACOS_APP_NAME)]
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


def _sign_macos_apps(package_root: Path) -> None:
    for app in package_root.glob("*.app"):
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)


def _write_macos_launcher(package_root: Path) -> None:
    launcher = package_root / "启动.command"
    launcher.write_text(
        "#!/bin/sh\n"
        "DIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
        f"APP=\"$DIR/{MACOS_APP_NAME}.app\"\n"
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
