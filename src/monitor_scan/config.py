"""应用配置模块。

定义监控视频智能分析系统的所有配置参数。
针对严重丢帧场景进行了优化配置。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# 支持的视频文件扩展名
SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"})

# 模型目录名
MODEL_DIRECTORY = "models"

# YOLO CoreML 模型文件名
YOLO_COREML_MODEL_NAME = "yolo26n-512-fp16-nms.mlpackage"

# 帧签名缩放尺寸，用于检测停滞帧
FRAME_SIGNATURE_SIZE = 16

# 进度条最大值
PROGRESS_MAX = 100

# 默认配置范围常量
SAMPLE_FPS_MIN = 0.1
SAMPLE_FPS_MAX = 30.0
CONFIDENCE_THRESHOLD_MIN = 0.0
CONFIDENCE_THRESHOLD_MAX = 1.0
NMS_THRESHOLD_MIN = 0.0
NMS_THRESHOLD_MAX = 1.0
MOTION_AREA_RATIO_MIN = 0.0
MOTION_AREA_RATIO_MAX = 1.0


def default_model_path() -> Path:
    """获取默认模型路径。

    Returns:
        默认 CoreML 模型路径
    """
    return _model_root() / YOLO_COREML_MODEL_NAME


def _model_root() -> Path:
    """获取模型根目录。

    在打包环境中使用 PyInstaller 的临时目录，否则使用相对路径。

    Returns:
        模型根目录路径
    """
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root is not None:
        return Path(bundle_root) / MODEL_DIRECTORY
    return Path(MODEL_DIRECTORY)


@dataclass(frozen=True)
class AppConfig:
    """应用配置数据类。

    所有配置参数使用不可变数据类定义，确保配置在运行时不会被意外修改。

    针对严重丢帧场景的优化配置说明：
    - 丢帧检测阈值：时间戳跳跃超过此值视为丢帧
    - 丢帧采样率倍数：丢帧区域的采样率倍增系数
    - 最大采样率：防止过度采样的上限
    - 滑动窗口重叠比例：提高检测覆盖的重叠度
    - 关键帧强制检测：确保 I-frame 不被遗漏
    - 丢帧补偿帧数：丢帧后保持高采样率的帧数
    """

    # 基础配置
    sample_fps: float = 2.0  # 提高默认采样率到 2 帧/秒
    confidence_threshold: float = 0.5
    motion_threshold: int = 5000
    model_path: Path = field(default_factory=default_model_path)
    output_directory: Path = Path("output_results")

    # 模型配置
    image_size: int = 512
    nms_threshold: float = 0.45

    # 采样配置
    max_candidate_frames_per_slot: int = 4  # 增加到 4 个候选帧
    max_scheduled_detections_per_slot: int = 2  # 增加到 2 次定时检测
    max_motion_detections_per_slot: int = 2  # 增加到 2 次运动检测

    # 错误处理配置
    max_consecutive_decode_failures: int = 3000
    max_consecutive_stalled_frames: int = 30

    # FFmpeg 配置
    remux_before_analysis: bool = True
    ffmpeg_path: str = "ffmpeg"
    ffmpeg_timeout_seconds: int = 1800

    # CoreML 配置
    coreml_warmup_runs: int = 1

    # 运动检测配置
    motion_resize_width: int = 480
    motion_area_ratio_threshold: float = 0.005  # 降低阈值，提高敏感度
    motion_detect_shadows: bool = False

    # 帧处理配置
    frame_signature_size: int = FRAME_SIGNATURE_SIZE

    # ==================== 丢帧优化配置 ====================

    # 丢帧检测阈值（秒）：时间戳跳跃超过此值视为丢帧
    frame_drop_threshold_seconds: float = 0.5

    # 丢帧严重程度阈值：丢帧时间 / 期望间隔 超过此值视为严重丢帧
    frame_drop_severity_threshold: float = 2.0

    # 丢帧区域采样率倍数：丢帧时采样率 = 基础采样率 × 此倍数
    frame_drop_sample_rate_multiplier: float = 5.0

    # 最大采样率（帧/秒）：防止过度采样
    max_sample_fps: float = 10.0

    # 丢帧后保持高采样率的帧数
    frame_drop_recovery_frames: int = 30

    # ==================== 滑动窗口配置 ====================

    # 是否启用滑动窗口模式
    enable_sliding_window: bool = True

    # 滑动窗口重叠比例（0-1）
    sliding_window_overlap_ratio: float = 0.5

    # ==================== 关键帧检测配置 ====================

    # 是否启用关键帧强制检测
    enable_keyframe_detection: bool = True

    # 关键帧检测间隔（帧数）：每 N 帧强制检测一次
    keyframe_interval: int = 30

    # 丢帧后的首个完整帧是否强制检测
    force_detection_after_drop: bool = True

    # ==================== 多层过滤配置 ====================

    # 是否启用帧差预过滤
    enable_frame_diff_prefilter: bool = True

    # 帧差阈值：低于此值视为静态场景，跳过检测
    frame_diff_threshold: float = 5.0

    # 静态场景跳过帧数：连续静态帧超过此数才跳过
    static_scene_skip_frames: int = 10

    # ==================== 丢帧补偿配置 ====================

    # 是否启用双向检测补偿
    enable_bidirectional_detection: bool = True

    # 丢帧区域冗余检测次数
    frame_drop_redundant_detections: int = 3

    # 置信度衰减系数：丢帧时间越长，置信度越低
    confidence_decay_factor: float = 0.95

    # ==================== ROI 区域配置 ====================

    # 检测区域（可选，像素坐标），全部为 None 时不裁剪
    roi_x: int | None = None
    roi_y: int | None = None
    roi_width: int | None = None
    roi_height: int | None = None

    @property
    def has_roi(self) -> bool:
        """是否启用了 ROI 区域裁剪。"""
        return any(v is not None for v in (self.roi_x, self.roi_y, self.roi_width, self.roi_height))

    def validate(self) -> None:
        """验证配置参数的有效性。

        Raises:
            ValueError: 如果任何配置参数无效
        """
        # 抽帧频率验证
        if not SAMPLE_FPS_MIN <= self.sample_fps <= SAMPLE_FPS_MAX:
            raise ValueError(f"抽帧频率必须在 {SAMPLE_FPS_MIN} 到 {SAMPLE_FPS_MAX} 之间。")

        # 置信度阈值验证
        if not CONFIDENCE_THRESHOLD_MIN <= self.confidence_threshold <= CONFIDENCE_THRESHOLD_MAX:
            raise ValueError(f"AI 置信度阈值必须在 {CONFIDENCE_THRESHOLD_MIN} 到 {CONFIDENCE_THRESHOLD_MAX} 之间。")

        # 运动检测阈值验证
        if self.motion_threshold <= 0:
            raise ValueError("运动检测阈值必须大于 0。")

        # 模型输入尺寸验证
        if self.image_size <= 0:
            raise ValueError("模型输入尺寸必须大于 0。")

        # NMS 阈值验证
        if not NMS_THRESHOLD_MIN <= self.nms_threshold <= NMS_THRESHOLD_MAX:
            raise ValueError(f"NMS 阈值必须在 {NMS_THRESHOLD_MIN} 到 {NMS_THRESHOLD_MAX} 之间。")

        # 采样配置验证
        if self.max_candidate_frames_per_slot <= 0:
            raise ValueError("每个采样时间槽的候选帧数量必须大于 0。")
        if self.max_scheduled_detections_per_slot <= 0:
            raise ValueError("每个采样时间槽的定时检测次数必须大于 0。")
        if self.max_motion_detections_per_slot <= 0:
            raise ValueError("每个采样时间槽的运动补检次数必须大于 0。")

        # 错误处理配置验证
        if self.max_consecutive_decode_failures <= 0:
            raise ValueError("连续解码失败上限必须大于 0。")
        if self.max_consecutive_stalled_frames <= 0:
            raise ValueError("连续停滞帧上限必须大于 0。")

        # FFmpeg 配置验证
        if self.ffmpeg_timeout_seconds <= 0:
            raise ValueError("FFmpeg 处理超时时间必须大于 0。")

        # CoreML 配置验证
        if self.coreml_warmup_runs < 0:
            raise ValueError("CoreML 预热次数不能小于 0。")

        # 运动检测配置验证
        if self.motion_resize_width <= 0:
            raise ValueError("运动检测缩放宽度必须大于 0。")
        if not MOTION_AREA_RATIO_MIN < self.motion_area_ratio_threshold <= MOTION_AREA_RATIO_MAX:
            raise ValueError(f"运动面积比例阈值必须在 {MOTION_AREA_RATIO_MIN} 到 {MOTION_AREA_RATIO_MAX} 之间。")

        # 帧处理配置验证
        if self.frame_signature_size <= 0:
            raise ValueError("帧签名缩放尺寸必须大于 0。")

        # 丢帧优化配置验证
        if self.frame_drop_threshold_seconds <= 0:
            raise ValueError("丢帧检测阈值必须大于 0。")
        if self.frame_drop_severity_threshold <= 0:
            raise ValueError("丢帧严重程度阈值必须大于 0。")
        if self.frame_drop_sample_rate_multiplier < 1.0:
            raise ValueError("丢帧采样率倍数必须大于等于 1。")
        if self.max_sample_fps < self.sample_fps:
            raise ValueError("最大采样率必须大于等于基础采样率。")
        if self.frame_drop_recovery_frames < 0:
            raise ValueError("丢帧恢复帧数不能为负数。")

        # 滑动窗口配置验证
        if not 0.0 <= self.sliding_window_overlap_ratio < 1.0:
            raise ValueError("滑动窗口重叠比例必须在 0 到 1 之间（不含 1）。")

        # 关键帧检测配置验证
        if self.keyframe_interval <= 0:
            raise ValueError("关键帧检测间隔必须大于 0。")

        # 帧差预过滤配置验证
        if self.frame_diff_threshold < 0:
            raise ValueError("帧差阈值不能为负数。")
        if self.static_scene_skip_frames <= 0:
            raise ValueError("静态场景跳过帧数必须大于 0。")

        # 丢帧补偿配置验证
        if self.frame_drop_redundant_detections < 0:
            raise ValueError("冗余检测次数不能为负数。")
        if not 0.0 < self.confidence_decay_factor <= 1.0:
            raise ValueError("置信度衰减系数必须在 0 到 1 之间。")

        # ROI 区域配置验证
        if self.has_roi:
            if self.roi_x is None or self.roi_y is None or self.roi_width is None or self.roi_height is None:
                raise ValueError("启用 ROI 时必须同时指定 x、y、width、height 四个参数。")
            if self.roi_x < 0 or self.roi_y < 0:
                raise ValueError("ROI 的 x 和 y 坐标不能为负数。")
            if self.roi_width <= 0 or self.roi_height <= 0:
                raise ValueError("ROI 的宽度和高度必须大于 0。")
            if self.roi_width % 2 != 0 or self.roi_height % 2 != 0:
                raise ValueError("ROI 的宽度和高度必须为偶数（FFmpeg 编码要求）。")
