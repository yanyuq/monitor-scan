"""FFmpeg 重封装模块。

使用 FFmpeg 对视频进行无重编码重封装，重建索引并纠正时间戳。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# FFmpeg 支持的容器格式
MOVIE_CONTAINER_FORMATS = {".mp4", ".mov", ".m4v"}

# 临时目录前缀
TEMP_DIR_PREFIX = ".monitor_scan_"

# 安全文件名最大长度
SAFE_STEM_MAX_LENGTH = 40


@dataclass(frozen=True)
class PreparedVideo:
    """准备好的视频信息。

    Attributes:
        source_path: 原始视频路径
        analysis_path: 用于分析的视频路径（可能是临时文件）
        temporary_directory: 临时目录路径
        message: 处理消息
    """

    source_path: Path
    analysis_path: Path
    temporary_directory: Path | None = None
    message: str | None = None

    @property
    def used_temporary_file(self) -> bool:
        """是否使用了临时文件。"""
        return self.temporary_directory is not None and self.analysis_path != self.source_path

    def cleanup(self) -> None:
        """清理临时文件和目录。"""
        if self.temporary_directory is not None and self.temporary_directory.exists():
            try:
                shutil.rmtree(self.temporary_directory, ignore_errors=True)
                logger.debug(f"已清理临时目录：{self.temporary_directory}")
            except Exception as exc:
                logger.warning(f"清理临时目录失败：{exc}")


class FfmpegRemuxer:
    """FFmpeg 重封装器。

    使用 FFmpeg 对视频进行无重编码重封装，重建索引并纠正时间戳。

    Attributes:
        ffmpeg_path: FFmpeg 可执行文件路径
        timeout_seconds: 处理超时时间（秒）
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg", timeout_seconds: int = 1800) -> None:
        """初始化 FFmpeg 重封装器。

        Args:
            ffmpeg_path: FFmpeg 可执行文件路径
            timeout_seconds: 处理超时时间（秒）
        """
        self.ffmpeg_path = ffmpeg_path
        self.timeout_seconds = timeout_seconds
        logger.debug(f"FFmpeg 重封装器初始化：路径={ffmpeg_path}，超时={timeout_seconds}秒")

    def prepare(self, video_path: str | Path) -> PreparedVideo:
        """准备视频文件。

        使用 FFmpeg 进行无重编码重封装，如果失败则回退到原始文件。

        Args:
            video_path: 视频文件路径

        Returns:
            准备好的视频信息
        """
        source_path = Path(video_path)
        logger.info(f"准备视频文件：{source_path}")

        executable = self._resolve_ffmpeg()
        if executable is None:
            logger.warning("未找到 FFmpeg，将直接分析原视频")
            return self._fallback(source_path, "未找到 FFmpeg，已直接分析原视频。")

        if not source_path.exists():
            logger.warning(f"视频文件不存在：{source_path}")
            return self._fallback(source_path, "视频文件不存在，已跳过 FFmpeg 重封装。")

        temporary_directory = self._create_temporary_directory(source_path)
        output_path = temporary_directory / f"{source_path.stem}_remuxed{source_path.suffix.lower()}"
        command = self._build_command(executable, source_path, output_path)

        logger.debug(f"执行 FFmpeg 命令：{' '.join(command)}")

        run_options = {
            "capture_output": True,
            "text": True,
            "timeout": self.timeout_seconds,
            "check": False,
        }

        try:
            result = subprocess.run(command, **run_options)
        except subprocess.TimeoutExpired as exc:
            logger.error(f"FFmpeg 处理超时：{exc}")
            shutil.rmtree(temporary_directory, ignore_errors=True)
            return self._fallback(source_path, f"FFmpeg 重封装超时，已直接分析原视频：{exc}")
        except OSError as exc:
            logger.error(f"FFmpeg 执行失败：{exc}")
            shutil.rmtree(temporary_directory, ignore_errors=True)
            return self._fallback(source_path, f"FFmpeg 重封装失败，已直接分析原视频：{exc}")

        if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
            stderr = result.stderr.strip() or "未返回错误详情"
            logger.warning(f"FFmpeg 重封装失败：{stderr}")
            shutil.rmtree(temporary_directory, ignore_errors=True)
            return self._fallback(source_path, f"FFmpeg 重封装失败，已直接分析原视频：{stderr}")

        logger.info(f"FFmpeg 重封装成功：{output_path}")
        return PreparedVideo(
            source_path=source_path,
            analysis_path=output_path,
            temporary_directory=temporary_directory,
            message="FFmpeg 已完成无重编码索引重建。",
        )

    def _build_command(self, executable: str, source_path: Path, output_path: Path) -> list[str]:
        """构建 FFmpeg 命令。

        Args:
            executable: FFmpeg 可执行文件路径
            source_path: 源视频路径
            output_path: 输出视频路径

        Returns:
            FFmpeg 命令列表
        """
        command = [
            executable,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-err_detect", "ignore_err",
            "-fflags", "+genpts+discardcorrupt",
            "-i", str(source_path),
            "-map", "0:v:0",
            "-c:v", "copy",
            "-an",
            "-avoid_negative_ts", "make_zero",
        ]

        # 对于 MOV 容器添加 faststart 标志
        if output_path.suffix.lower() in MOVIE_CONTAINER_FORMATS:
            command.extend(["-movflags", "+faststart"])

        command.append(str(output_path))
        return command

    def _resolve_ffmpeg(self) -> str | None:
        """解析 FFmpeg 可执行文件路径。

        按优先级查找：
        1. 系统 PATH
        2. 配置的显式路径
        3. imageio-ffmpeg 内置版本

        Returns:
            FFmpeg 可执行文件路径，如果未找到则返回 None
        """
        # 尝试系统 PATH
        resolved = shutil.which(self.ffmpeg_path)
        if resolved is not None:
            logger.debug(f"在系统 PATH 中找到 FFmpeg：{resolved}")
            return resolved

        # 尝试显式路径
        explicit_path = Path(self.ffmpeg_path)
        if explicit_path.exists() and explicit_path.is_file():
            logger.debug(f"使用显式 FFmpeg 路径：{explicit_path}")
            return str(explicit_path)

        # 尝试 imageio-ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            logger.debug(f"使用 imageio-ffmpeg：{ffmpeg_path}")
            return ffmpeg_path
        except ImportError:
            logger.debug("imageio-ffmpeg 不可用")
            return None

    def _create_temporary_directory(self, source_path: Path) -> Path:
        """创建临时目录。

        优先在源文件所在目录创建，失败时使用系统临时目录。

        Args:
            source_path: 源视频路径

        Returns:
            临时目录路径
        """
        prefix = f"{TEMP_DIR_PREFIX}{self._safe_stem(source_path)}_"

        try:
            temporary_directory = Path(tempfile.mkdtemp(prefix=prefix, dir=source_path.parent))
            logger.debug(f"在源文件目录创建临时目录：{temporary_directory}")
        except OSError:
            temporary_directory = Path(tempfile.mkdtemp(prefix=prefix))
            logger.debug(f"在系统临时目录创建：{temporary_directory}")

        return temporary_directory

    def _safe_stem(self, source_path: Path) -> str:
        """生成安全的文件名前缀。

        Args:
            source_path: 源视频路径

        Returns:
            安全的文件名前缀
        """
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in source_path.stem)
        return safe[:SAFE_STEM_MAX_LENGTH] or "video"

    def _fallback(self, source_path: Path, message: str) -> PreparedVideo:
        """创建回退结果。

        Args:
            source_path: 源视频路径
            message: 回退消息

        Returns:
            使用原始文件的准备结果
        """
        logger.info(f"回退处理：{message}")
        return PreparedVideo(source_path=source_path, analysis_path=source_path, message=message)
