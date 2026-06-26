"""FFmpeg 重封装与硬件加速模块。

使用 FFmpeg 对视频进行无重编码重封装，重建索引并纠正时间戳。
支持 VideoToolbox 硬件编解码（M1/M2 Apple Silicon）。
"""

from __future__ import annotations

import logging
import os
import signal
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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

    def prepare(
        self,
        video_path: str | Path,
    ) -> PreparedVideo:
        """准备视频文件。

        使用 FFmpeg 进行无重编码重封装，修复索引和时间戳。
        ROI 裁剪在帧级别完成，不需要 FFmpeg 重编码。

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
            "start_new_session": True,  # 独立进程组，方便清理
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
            message="FFmpeg 已完成索引重建。",
        )

    def _build_command(
        self,
        executable: str,
        source_path: Path,
        output_path: Path,
    ) -> list[str]:
        """构建 FFmpeg 命令（无重编码拷贝，仅修复索引）。

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


class FfmpegFrameReader:
    """基于 FFmpeg 的硬件加速帧读取器。

    使用 FFmpeg 子进程解码视频，支持 VideoToolbox 硬件解码（Apple Silicon）。
    通过 stdout pipe 输出 rawvideo（BGR24 格式），替代 OpenCV VideoCapture 的软解码。

    用法::

        reader = FfmpegFrameReader("/path/to/video.mp4")
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            # 处理 frame
        reader.release()

    Attributes:
        width: 视频宽度
        height: 视频高度
        fps: 视频帧率
        total_frames: 总帧数
    """

    def __init__(
        self,
        video_path: str | Path,
        ffmpeg_path: str = "ffmpeg",
        hw_accel: bool = True,
    ) -> None:
        """初始化帧读取器。

        Args:
            video_path: 视频文件路径
            ffmpeg_path: FFmpeg 可执行文件路径
            hw_accel: 是否启用硬件加速解码
        """
        self._video_path = Path(video_path)
        self._ffmpeg_path = ffmpeg_path
        self._hw_accel = hw_accel
        self._process: subprocess.Popen | None = None
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 0.0
        self.total_frames: int = 0
        self.duration: float = 0.0  # 视频时长（秒）
        self._frame_size: int = 0
        self._closed: bool = True
        self._current_frame: int = 0  # 已读取帧计数
        self._hw_failed: bool = False  # 硬件解码是否已失败

        self._probe()
        self._start()

    def _probe(self) -> None:
        """使用 ffprobe 探测视频信息。"""
        executable = shutil.which(self._ffmpeg_path) or self._ffmpeg_path
        # 查找 ffprobe：优先 shutil.which，回退到替换 ffmpeg
        ffprobe = shutil.which("ffprobe") or executable.replace("ffmpeg", "ffprobe")

        # 探测流信息：宽高、帧率、帧数
        stream_cmd = [
            ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "csv=p=0",
            str(self._video_path),
        ]
        # 探测容器信息：时长
        duration_cmd = [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(self._video_path),
        ]

        try:
            result = subprocess.run(
                stream_cmd, capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                self.width = int(parts[0])
                self.height = int(parts[1])
                if "/" in parts[2]:
                    num, den = parts[2].split("/")
                    self.fps = float(num) / float(den)
                else:
                    self.fps = float(parts[2])
                if len(parts) > 3 and parts[3].strip():
                    self.total_frames = int(parts[3])
        except (OSError, ValueError, IndexError) as exc:
            logger.warning(f"ffprobe 流信息探测失败：{exc}")

        try:
            result = subprocess.run(
                duration_cmd, capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                self.duration = float(result.stdout.strip())
        except (OSError, ValueError):
            pass

        # 帧数不可用时，通过时长和帧率推算
        if self.total_frames <= 0 and self.duration > 0 and self.fps > 0:
            self.total_frames = int(self.duration * self.fps)

        self._frame_size = self.width * self.height * 3 // 2  # NV12: Y + UV
        logger.debug(
            f"视频信息：{self.width}x{self.height}，{self.fps:.2f}fps，"
            f"共 {self.total_frames} 帧，时长 {self.duration:.1f}秒"
        )

    def _start(self, seek_seconds: float = 0.0) -> None:
        """启动 FFmpeg 解码子进程。

        Args:
            seek_seconds: 跳转到指定时间位置（秒），用于断点续读
        """
        if self._frame_size <= 0:
            logger.warning("视频尺寸无效，跳过 FFmpeg 解码进程启动")
            return

        executable = shutil.which(self._ffmpeg_path) or self._ffmpeg_path
        command = [
            executable,
            "-hide_banner",
            "-loglevel", "error",
        ]

        if self._hw_accel:
            command.extend(["-hwaccel", "videotoolbox", "-hwaccel_output_format", "nv12"])

        # seek 放在 -i 之前（输入 seek，更快）
        if seek_seconds > 0:
            command.extend(["-ss", f"{seek_seconds:.3f}"])

        command.extend([
            "-i", str(self._video_path),
            "-map", "0:v:0",
            "-f", "rawvideo",
            "-pix_fmt", "nv12",  # NV12 原生输出，省去 FFmpeg 端色彩转换
            "-an",
            "pipe:1",
        ])

        accel_mode = "VideoToolbox 硬件解码" if self._hw_accel else "CPU 软解码"
        seek_info = f"，seek={seek_seconds:.1f}秒" if seek_seconds > 0 else ""
        logger.info(f"启动 FFmpeg 解码进程（{accel_mode}{seek_info}）：{self._video_path}")

        try:
            # start_new_session=True：创建独立进程组，确保 release() 时能 kill 整个子进程树
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_size * 2,  # 缓冲 2 帧
                start_new_session=True,
            )
            # 短暂检查进程是否存活
            import time
            time.sleep(0.2)
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
                logger.warning(f"FFmpeg 解码进程立即退出（{accel_mode}失败）：{stderr[:200]}")
                self._process = None
                return
            logger.info(f"FFmpeg 解码进程已启动（{accel_mode}），pid={self._process.pid}")
            self._closed = False
        except OSError as exc:
            logger.error(f"FFmpeg 解码进程启动失败：{exc}")
            self._process = None

    def read(self, timeout: float = 10.0) -> tuple[bool, np.ndarray | None]:
        """读取一帧，带超时检测和硬件解码自动回退。

        当 VideoToolbox 硬件解码卡死时，自动 kill 进程并用 CPU 软解码重启，
        从当前帧位置继续读取。

        Args:
            timeout: 读取超时时间（秒）

        Returns:
            (成功标志, 帧数据)，失败时返回 (False, None)
        """
        if self._closed:
            return False, None

        # 进程不存在或已退出，尝试重启
        if self._process is None or self._process.poll() is not None:
            if self._hw_accel and not self._hw_failed:
                return self._restart_with_cpu()
            return False, None

        import select

        stdout = self._process.stdout  # type: ignore[union-attr]
        try:
            ready, _, _ = select.select([stdout], [], [], timeout)
            if not ready:
                # 超时：检查进程状态
                if self._process.poll() is not None:
                    logger.warning("FFmpeg 解码进程已退出")
                    if self._hw_accel and not self._hw_failed:
                        return self._restart_with_cpu()
                    return False, None
                # 进程活着但无数据输出 → VideoToolbox 卡死
                logger.warning(f"FFmpeg read 超时 {timeout}秒，硬件解码可能卡死")
                if self._hw_accel and not self._hw_failed:
                    return self._restart_with_cpu()
                return False, None

            raw = stdout.read(self._frame_size)
            if len(raw) < self._frame_size:
                if self._hw_accel and not self._hw_failed:
                    return self._restart_with_cpu()
                return False, None

            # NV12 → BGR 转换（OpenCV NEON SIMD 加速，比 FFmpeg 软件转换更快）
            nv12 = np.frombuffer(raw, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
            frame = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            self._current_frame += 1
            return True, frame
        except (OSError, ValueError):
            return False, None

    def _restart_with_cpu(self) -> tuple[bool, np.ndarray | None]:
        """硬件解码卡死后，用 CPU 软解码从当前位置重启。

        Returns:
            (成功标志, 帧数据)
        """
        self._hw_failed = True
        # 计算当前帧对应的时间戳，用于 seek
        seek_seconds = self._current_frame / self.fps if self.fps > 0 else 0
        logger.warning(
            f"硬件解码失败，切换到 CPU 软解码（已读 {self._current_frame} 帧，"
            f"seek 到 {seek_seconds:.1f}秒）"
        )

        # kill 当前进程
        self._kill_process()

        # 用 CPU 软解码重启
        self._hw_accel = False
        self._start(seek_seconds=seek_seconds)

        if self._process is None:
            logger.error("CPU 软解码重启失败")
            return False, None

        logger.info(f"CPU 软解码重启成功，从第 {self._current_frame} 帧继续")
        # 重启后立即读取第一帧
        return self.read(timeout=30.0)

    def _kill_process(self) -> None:
        """强制终止当前 FFmpeg 进程。"""
        if self._process is None:
            return
        pid = self._process.pid
        try:
            self._process.stdout.close()  # type: ignore[union-attr]
        except Exception:
            pass
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                self._process.kill()
            except Exception:
                pass
        try:
            self._process.wait(timeout=3)
        except Exception:
            pass
        self._process = None
        logger.debug(f"FFmpeg 进程已终止（pid={pid}）")

    def release(self) -> None:
        """释放资源，强制关闭子进程及其整个进程组。"""
        if self._closed:
            return
        self._closed = True
        self._kill_process()

    def __del__(self) -> None:
        self.release()

    def __enter__(self) -> FfmpegFrameReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
