from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"})


def default_model_path() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root is not None:
        return Path(bundle_root) / "models" / "yolov8n.onnx"
    return Path("models/yolov8n.onnx")


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
