from __future__ import annotations

from pathlib import Path

from monitor_scan.config import SUPPORTED_VIDEO_EXTENSIONS


class VideoScanner:
    def __init__(self, supported_extensions: frozenset[str] = SUPPORTED_VIDEO_EXTENSIONS) -> None:
        self.supported_extensions = supported_extensions

    def scan(self, directory: str | Path) -> list[Path]:
        root = Path(directory).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"视频目录不存在：{root}")
        if not root.is_dir():
            raise NotADirectoryError(f"选择的路径不是目录：{root}")

        videos = [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in self.supported_extensions
        ]
        return sorted(videos, key=lambda path: path.name.lower())
