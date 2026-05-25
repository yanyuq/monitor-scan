from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"})
MODEL_DIRECTORY = "models"
YOLO_SOURCE_MODEL_NAME = "yolo26n.pt"
YOLO_COREML_MODEL_NAME = "yolo26n.mlpackage"


def default_model_path() -> Path:
    model_root = _model_root()
    if sys.platform == "darwin":
        coreml_model_path = model_root / YOLO_COREML_MODEL_NAME
        if coreml_model_path.exists():
            return coreml_model_path
    return model_root / YOLO_SOURCE_MODEL_NAME


def _model_root() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root is not None:
        return Path(bundle_root) / MODEL_DIRECTORY
    return Path(MODEL_DIRECTORY)


@dataclass(frozen=True)
class AppConfig:
    sample_fps: float = 2.0
    confidence_threshold: float = 0.5
    motion_threshold: int = 5000
    model_path: Path = field(default_factory=default_model_path)
    output_directory: Path = Path("output_results")
    image_size: int = 640
    nms_threshold: float = 0.45
    max_candidate_frames_per_slot: int = 5
    max_consecutive_decode_failures: int = 300
    remux_before_analysis: bool = True
    ffmpeg_path: str = "ffmpeg"
    ffmpeg_timeout_seconds: int = 1800

    def validate(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("抽帧频率必须大于 0。")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("AI 置信度阈值必须位于 0 到 1 之间。")
        if self.motion_threshold <= 0:
            raise ValueError("运动检测阈值必须大于 0。")
        if self.image_size <= 0:
            raise ValueError("模型输入尺寸必须大于 0。")
        if self.max_candidate_frames_per_slot <= 0:
            raise ValueError("每个采样时间槽的候选帧数量必须大于 0。")
        if self.max_consecutive_decode_failures <= 0:
            raise ValueError("连续解码失败上限必须大于 0。")
        if self.ffmpeg_timeout_seconds <= 0:
            raise ValueError("FFmpeg 处理超时时间必须大于 0。")
