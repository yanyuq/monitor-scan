"""视频文件扫描模块。

扫描指定目录中的视频文件。
"""

from __future__ import annotations

import logging
from pathlib import Path

from monitor_scan.config import SUPPORTED_VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)


class VideoScanner:
    """视频文件扫描器。

    扫描指定目录中的视频文件，支持过滤和排序。

    Attributes:
        supported_extensions: 支持的视频文件扩展名集合
    """

    def __init__(self, supported_extensions: frozenset[str] = SUPPORTED_VIDEO_EXTENSIONS) -> None:
        """初始化视频扫描器。

        Args:
            supported_extensions: 支持的视频文件扩展名集合
        """
        self.supported_extensions = supported_extensions
        logger.debug(f"视频扫描器初始化，支持格式：{supported_extensions}")

    def scan(self, directory: str | Path) -> list[Path]:
        """扫描目录中的视频文件。

        Args:
            directory: 要扫描的目录路径

        Returns:
            排序后的视频文件路径列表

        Raises:
            FileNotFoundError: 如果目录不存在
            NotADirectoryError: 如果路径不是目录
        """
        root = Path(directory).expanduser()

        if not root.exists():
            raise FileNotFoundError(f"视频目录不存在：{root}")
        if not root.is_dir():
            raise NotADirectoryError(f"选择的路径不是目录：{root}")

        logger.info(f"扫描目录：{root}")

        # 扫描视频文件
        videos = [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in self.supported_extensions
        ]

        # 按文件名排序（不区分大小写）
        videos.sort(key=lambda path: path.name.lower())

        logger.info(f"扫描完成，找到 {len(videos)} 个视频文件")
        for video in videos:
            logger.debug(f"  - {video.name}")

        return videos
