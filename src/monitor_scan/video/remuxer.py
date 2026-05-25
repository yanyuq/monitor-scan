from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreparedVideo:
    source_path: Path
    analysis_path: Path
    temporary_directory: Path | None = None
    message: str | None = None

    @property
    def used_temporary_file(self) -> bool:
        return self.temporary_directory is not None and self.analysis_path != self.source_path

    def cleanup(self) -> None:
        if self.temporary_directory is not None and self.temporary_directory.exists():
            shutil.rmtree(self.temporary_directory, ignore_errors=True)


class FfmpegRemuxer:
    def __init__(self, ffmpeg_path: str = "ffmpeg", timeout_seconds: int = 1800) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.timeout_seconds = timeout_seconds

    def prepare(self, video_path: str | Path) -> PreparedVideo:
        source_path = Path(video_path)
        executable = self._resolve_ffmpeg()
        if executable is None:
            return self._fallback(source_path, "未找到 FFmpeg，已直接分析原视频。")
        if not source_path.exists():
            return self._fallback(source_path, "视频文件不存在，已跳过 FFmpeg 重封装。")

        temporary_directory = self._create_temporary_directory(source_path)
        output_path = temporary_directory / f"{source_path.stem}_remuxed{source_path.suffix.lower()}"
        command = self._build_command(executable, source_path, output_path)

        run_options = {
            "capture_output": True,
            "text": True,
            "timeout": self.timeout_seconds,
            "check": False,
        }
        if os.name == "nt":
            run_options["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            result = subprocess.run(command, **run_options)
        except (OSError, subprocess.TimeoutExpired) as exc:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            return self._fallback(source_path, f"FFmpeg 重封装失败，已直接分析原视频：{exc}")

        if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
            stderr = result.stderr.strip() or "未返回错误详情"
            shutil.rmtree(temporary_directory, ignore_errors=True)
            return self._fallback(source_path, f"FFmpeg 重封装失败，已直接分析原视频：{stderr}")

        return PreparedVideo(
            source_path=source_path,
            analysis_path=output_path,
            temporary_directory=temporary_directory,
            message="FFmpeg 已完成无重编码索引重建。",
        )

    def _build_command(self, executable: str, source_path: Path, output_path: Path) -> list[str]:
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-err_detect",
            "ignore_err",
            "-fflags",
            "+genpts+discardcorrupt",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-avoid_negative_ts",
            "make_zero",
        ]
        if output_path.suffix.lower() in {".mp4", ".mov", ".m4v"}:
            command.extend(["-movflags", "+faststart"])
        command.append(str(output_path))
        return command

    def _resolve_ffmpeg(self) -> str | None:
        resolved = shutil.which(self.ffmpeg_path)
        if resolved is not None:
            return resolved
        explicit_path = Path(self.ffmpeg_path)
        if explicit_path.exists() and explicit_path.is_file():
            return str(explicit_path)
        try:
            import imageio_ffmpeg
        except ImportError:
            return None
        return imageio_ffmpeg.get_ffmpeg_exe()

    def _create_temporary_directory(self, source_path: Path) -> Path:
        prefix = f".monitor_scan_{self._safe_stem(source_path)}_"
        try:
            temporary_directory = Path(tempfile.mkdtemp(prefix=prefix, dir=source_path.parent))
        except OSError:
            temporary_directory = Path(tempfile.mkdtemp(prefix=prefix))
        self._hide_directory(temporary_directory)
        return temporary_directory

    def _hide_directory(self, path: Path) -> None:
        if os.name != "nt":
            return
        try:
            ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x02)
        except OSError:
            return

    def _safe_stem(self, source_path: Path) -> str:
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in source_path.stem)
        return safe[:40] or "video"

    def _fallback(self, source_path: Path, message: str) -> PreparedVideo:
        return PreparedVideo(source_path=source_path, analysis_path=source_path, message=message)
