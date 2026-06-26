"""视频分析器模块。

针对严重丢帧场景优化的视频分析器。
实现了多层检测漏斗、丢帧感知自适应采样、关键帧强制检测等优化策略。

核心优化策略：
1. 丢帧感知自适应采样：检测到丢帧时自动增加采样密度
2. 多层检测漏斗：逐层过滤，减少 YOLO 调用次数
3. 关键帧强制检测：确保 I-frame 不被遗漏
4. 丢帧补偿策略：丢帧后保持高采样率，双向检测
5. 滑动窗口检测：提高检测覆盖的重叠度
6. 硬件加速解码：优先使用 FFmpeg VideoToolbox 解码（Apple Silicon）
7. 解码-检测并行流水线：解码线程与检测线程分离
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from monitor_scan.ai.motion import MotionDetector
from monitor_scan.config import AppConfig, PROGRESS_MAX
from monitor_scan.results.writer import ResultWriter, format_timestamp
from monitor_scan.types import DetectionEvent, PersonDetection
from monitor_scan.video.remuxer import FfmpegFrameReader, FfmpegRemuxer, PreparedVideo

logger = logging.getLogger(__name__)


class StopToken(Protocol):
    """停止令牌协议，用于检查是否应停止处理。"""

    def is_stopped(self) -> bool:
        """检查是否已请求停止。"""
        ...


class PersonDetector(Protocol):
    """人形检测器协议。"""

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """检测帧中的人形。"""
        ...


@dataclass(frozen=True)
class VideoProgress:
    """视频处理进度信息。"""

    video_path: Path
    progress: int


@dataclass
class FrameAnalysisState:
    """帧分析状态，用于跟踪丢帧和采样状态。

    Attributes:
        frame_index: 当前帧索引
        last_frame_time: 上一帧时间戳
        current_sample_slot: 当前采样时间槽
        current_slot_detected: 当前时间槽是否已检测到人形
        scheduled_attempted_buckets: 已尝试的定时检测桶
        motion_attempted_buckets: 已尝试的运动检测桶
        consecutive_decode_failures: 连续解码失败次数
        consecutive_stalled_frames: 连续停滞帧次数
        previous_frame_signature: 上一帧签名
        previous_capture_position: 上一帧捕获位置
        previous_capture_msec: 上一帧捕获时间戳
        consecutive_static_frames: 连续静态帧次数
        frame_drop_recovery_remaining: 丢帧恢复剩余帧数
        current_sample_fps: 当前采样率（可能因丢帧而调整）
        last_detection_time: 上次检测时间
        frame_diff_history: 帧差历史记录
    """

    frame_index: int = 0
    last_frame_time: float = -1.0
    current_sample_slot: int | None = None
    current_slot_detected: bool = False
    scheduled_attempted_buckets: set[int] = field(default_factory=set)
    motion_attempted_buckets: set[int] = field(default_factory=set)
    consecutive_decode_failures: int = 0
    consecutive_stalled_frames: int = 0
    previous_frame_signature: np.ndarray | None = None
    previous_capture_position: float = -1.0
    previous_capture_msec: float = -1.0
    consecutive_static_frames: int = 0
    frame_drop_recovery_remaining: int = 0
    current_sample_fps: float = 0.0
    last_detection_time: float = -1.0
    frame_diff_history: list[float] = field(default_factory=list)


@dataclass
class DropFrameInfo:
    """丢帧信息。

    Attributes:
        is_drop: 是否检测到丢帧
        gap_duration: 丢帧持续时间（秒）
        severity: 丢帧严重程度
        drop_start_time: 丢帧开始时间
        drop_end_time: 丢帧结束时间
    """

    is_drop: bool = False
    gap_duration: float = 0.0
    severity: float = 0.0
    drop_start_time: float = 0.0
    drop_end_time: float = 0.0


class VideoAnalyzer:
    """视频分析器，负责协调运动检测、人形识别和结果保存。

    针对严重丢帧场景优化的视频分析器，实现了多层检测漏斗策略。

    优化策略：
    1. 丢帧感知自适应采样
    2. 多层检测漏斗
    3. 关键帧强制检测
    4. 丢帧补偿策略
    5. 滑动窗口检测

    Attributes:
        config: 应用配置
        motion_detector_factory: 运动检测器工厂函数
        person_detector: 人形检测器实例
        result_writer: 结果写入器
        remuxer: FFmpeg 重封装器
    """

    def __init__(
        self,
        config: AppConfig,
        motion_detector_factory: Callable[[], MotionDetector] | None = None,
        person_detector: PersonDetector | None = None,
        result_writer: ResultWriter | None = None,
        remuxer: FfmpegRemuxer | None = None,
    ) -> None:
        """初始化视频分析器。

        Args:
            config: 应用配置
            motion_detector_factory: 运动检测器工厂函数，默认使用 MotionDetector
            person_detector: 人形检测器实例，默认延迟初始化 YoloPersonDetector
            result_writer: 结果写入器，默认使用 ResultWriter
            remuxer: FFmpeg 重封装器，默认使用 FfmpegRemuxer

        Raises:
            ValueError: 如果配置验证失败
        """
        config.validate()
        self.config = config
        self.motion_detector_factory = motion_detector_factory or self._default_motion_detector
        self.person_detector = person_detector
        self.result_writer = result_writer or ResultWriter(config.output_directory)
        self.remuxer = remuxer or FfmpegRemuxer(config.ffmpeg_path, config.ffmpeg_timeout_seconds)
        logger.debug(f"VideoAnalyzer 初始化完成，配置：{config}")

    def analyze_video(
        self,
        video_path: str | Path,
        stop_token: StopToken,
        progress_callback: Callable[[VideoProgress], None] | None = None,
        detection_callback: Callable[[DetectionEvent], None] | None = None,
    ) -> list[DetectionEvent]:
        """分析单个视频文件。

        优先使用 FFmpeg VideoToolbox 硬件解码 + 并行流水线，
        不可用时回退到 OpenCV 软解码。

        当使用 FFmpeg 硬件解码且无需 ROI 裁剪时，直接读取源文件，
        跳过不必要的临时文件拷贝，减少磁盘占用。

        Args:
            video_path: 视频文件路径
            stop_token: 停止令牌，用于检查是否应停止处理
            progress_callback: 进度回调函数
            detection_callback: 检测事件回调函数

        Returns:
            检测到的事件列表

        Raises:
            OSError: 如果视频无法打开
        """
        source_path = Path(video_path)
        logger.info(f"开始分析视频：{source_path}")

        reader: FfmpegFrameReader | cv2.VideoCapture | None = None
        prepared_video: PreparedVideo | None = None

        try:
            # 先重封装：生成干净的临时文件（修复索引、丢弃损坏数据包）
            # 这一步不解码帧，速度极快，能显著降低后续解码卡死的概率
            prepared_video = self._prepare_video(source_path)
            analysis_path = prepared_video.analysis_path

            # 优先 FFmpeg 硬件解码（从干净文件读取，卡死时自动回退 CPU）
            reader = self._open_ffmpeg_reader(analysis_path)
            use_ffmpeg = reader is not None

            if not use_ffmpeg:
                # 回退到 OpenCV
                reader = self._open_capture(analysis_path)
                if not reader.isOpened():
                    raise OSError(f"视频无法打开：{source_path}")

            events = self._analyze_capture(
                reader,
                source_path,
                stop_token,
                progress_callback,
                detection_callback,
                use_ffmpeg=use_ffmpeg,
            )
            logger.info(f"视频分析完成：{source_path}，检测到 {len(events)} 个事件")
            return events

        except Exception as exc:
            logger.error(f"视频分析失败：{source_path}，错误：{exc}", exc_info=True)
            raise
        finally:
            # 安全释放资源
            if reader is not None:
                try:
                    reader.release()
                except Exception as exc:
                    logger.warning(f"释放视频捕获对象时出错：{exc}")

            if prepared_video is not None:
                try:
                    prepared_video.cleanup()
                except Exception as exc:
                    logger.warning(f"清理临时文件时出错：{exc}")

    def _analyze_capture(
        self,
        reader: FfmpegFrameReader | cv2.VideoCapture,
        source_path: Path,
        stop_token: StopToken,
        progress_callback: Callable[[VideoProgress], None] | None,
        detection_callback: Callable[[DetectionEvent], None] | None,
        use_ffmpeg: bool = False,
    ) -> list[DetectionEvent]:
        """分析已打开的视频读取器。

        实现了多层检测漏斗策略：
        1. 时间戳分析 - 检测丢帧
        2. 帧差分析 - 过滤静态场景
        3. 运动检测 - 过滤无运动场景
        4. YOLO 检测 - 最终人形识别

        当 use_ffmpeg=True 时，使用解码-检测并行流水线：
        解码线程将帧放入队列，检测线程从队列取帧处理。

        Args:
            reader: FFmpeg 帧读取器或 OpenCV 视频捕获对象
            source_path: 原始视频路径（用于报告）
            stop_token: 停止令牌
            progress_callback: 进度回调
            detection_callback: 检测事件回调
            use_ffmpeg: 是否使用 FFmpeg 硬件解码模式

        Returns:
            检测到的事件列表
        """
        events: list[DetectionEvent] = []
        motion_detector = self.motion_detector_factory()

        # 获取视频信息
        if use_ffmpeg and isinstance(reader, FfmpegFrameReader):
            total_frames = reader.total_frames
            video_fps = reader.fps or self.config.sample_fps
            video_duration = reader.duration  # 秒
        else:
            total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = reader.get(cv2.CAP_PROP_FPS) or self.config.sample_fps
            video_duration = total_frames / video_fps if video_fps > 0 and total_frames > 0 else 0.0

        logger.info(
            f"视频信息：总帧数={total_frames}，FPS={video_fps:.1f}，"
            f"时长={video_duration:.1f}秒，硬件解码={'是' if use_ffmpeg else '否'}"
        )

        # 初始化状态
        state = FrameAnalysisState()
        state.current_sample_fps = self.config.sample_fps
        last_progress_value = -1

        # 发送初始进度 0%
        if progress_callback is not None:
            progress_callback(VideoProgress(source_path, 0))
            last_progress_value = 0

        # 用于双向检测的缓冲区
        pre_drop_buffer: list[tuple[float, np.ndarray]] = []
        post_drop_buffer: list[tuple[float, np.ndarray]] = []

        # 并行流水线：解码队列
        decode_queue: queue.Queue[tuple[bool, np.ndarray | None]] = queue.Queue(maxsize=3)
        decode_thread: threading.Thread | None = None

        if use_ffmpeg and isinstance(reader, FfmpegFrameReader):
            # 启动解码线程（read() 内置超时检测和硬解→软解自动回退）
            def _decode_worker() -> None:
                while not stop_token.is_stopped():
                    ok, frame = reader.read(timeout=10.0)
                    decode_queue.put((ok, frame))
                    if not ok:
                        break

            decode_thread = threading.Thread(target=_decode_worker, name="decode-worker", daemon=True)
            decode_thread.start()
            logger.debug("解码-检测并行流水线已启动")

        while not stop_token.is_stopped():
            # 读取帧：优先从队列，回退到直接读取
            if decode_thread is not None:
                try:
                    ok, frame = decode_queue.get(timeout=5.0)
                except queue.Empty:
                    # 解码线程是否还活着
                    if not decode_thread.is_alive():
                        logger.warning("解码线程已退出，停止处理")
                        break
                    continue
            else:
                ok, frame = reader.read()

            if not ok:
                state.consecutive_decode_failures += 1
                if state.consecutive_decode_failures >= self.config.max_consecutive_decode_failures:
                    logger.warning(f"连续解码失败 {state.consecutive_decode_failures} 次，停止处理")
                    break
                # 尝试跳过失败的帧
                if isinstance(reader, cv2.VideoCapture):
                    reader.set(cv2.CAP_PROP_POS_FRAMES, state.frame_index + 1)
                state.frame_index += 1
                if progress_callback is not None:
                    progress_callback(VideoProgress(source_path, self._progress_by_time(
                        state.last_frame_time, video_duration, total_frames, state.frame_index,
                    )))
                continue

            # ROI 帧级裁剪：用 numpy 切片直接裁剪，零额外开销
            if self.config.has_roi:
                frame = frame[
                    self.config.roi_y : self.config.roi_y + self.config.roi_height,
                    self.config.roi_x : self.config.roi_x + self.config.roi_width,
                ]

            # 第 1 层：时间戳分析 - 检测丢帧
            if isinstance(reader, cv2.VideoCapture):
                frame_time_seconds = self._frame_time_seconds(reader, state.frame_index, video_fps)
            else:
                frame_time_seconds = self._frame_time_seconds_by_index(state.frame_index, video_fps)
            drop_info = self._detect_frame_drop(frame_time_seconds, state.last_frame_time, video_fps)

            if drop_info.is_drop:
                logger.warning(
                    f"检测到丢帧：{drop_info.gap_duration:.2f} 秒，"
                    f"严重程度：{drop_info.severity:.1f}x"
                )
                # 丢帧补偿：增加采样率
                self._handle_frame_drop(state, drop_info)

            state.last_frame_time = frame_time_seconds

            # 第 2 层：帧差分析 - 过滤静态场景
            if self.config.enable_frame_diff_prefilter and state.previous_frame_signature is not None:
                frame_diff = self._calculate_frame_diff(frame, state.previous_frame_signature)
                if frame_diff < self.config.frame_diff_threshold:
                    state.consecutive_static_frames += 1
                    # 静态场景可以跳过，但要确保关键帧不被跳过
                    if (state.consecutive_static_frames >= self.config.static_scene_skip_frames
                            and not self._is_keyframe(state.frame_index, drop_info)):
                        state.frame_index += 1
                        continue
                else:
                    state.consecutive_static_frames = 0

            # 计算当前帧签名
            current_frame_signature = self._frame_signature(frame)

            # 检查停滞帧
            if isinstance(reader, cv2.VideoCapture):
                capture_position = reader.get(cv2.CAP_PROP_POS_FRAMES)
                capture_msec = reader.get(cv2.CAP_PROP_POS_MSEC)
            else:
                # FFmpeg 顺序读取，使用帧索引作为位置
                capture_position = float(state.frame_index)
                capture_msec = state.frame_index / video_fps * 1000 if video_fps > 0 else 0.0

            if self._is_stalled_frame(
                current_frame_signature,
                state.previous_frame_signature,
                capture_position,
                state.previous_capture_position,
                capture_msec,
                state.previous_capture_msec,
            ):
                state.consecutive_stalled_frames += 1
                if state.consecutive_stalled_frames >= self.config.max_consecutive_stalled_frames:
                    logger.debug(f"检测到连续 {state.consecutive_stalled_frames} 帧停滞，尝试跳过")
                    state.consecutive_stalled_frames = 0
                    state.previous_frame_signature = None
                    state.previous_capture_position = -1.0
                    state.previous_capture_msec = -1.0
                    state.frame_index += 1
                    if isinstance(reader, cv2.VideoCapture):
                        set_result = reader.set(cv2.CAP_PROP_POS_FRAMES, state.frame_index + 1)
                        if not set_result:
                            logger.info("无法跳过停滞帧，可能是视频结束")
                            break
                    continue
            else:
                state.consecutive_stalled_frames = 0

            # 更新帧签名
            state.previous_frame_signature = current_frame_signature

            state.previous_capture_position = capture_position
            state.previous_capture_msec = capture_msec

            # 计算当前采样率（考虑丢帧恢复）
            effective_sample_fps = self._calculate_effective_sample_fps(state, drop_info)

            # 计算采样时间槽
            sample_slot = self._sample_slot(frame_time_seconds, effective_sample_fps)

            # 检查是否需要切换时间槽
            if sample_slot != state.current_sample_slot:
                state.current_sample_slot = sample_slot
                state.current_slot_detected = False
                state.scheduled_attempted_buckets.clear()
                state.motion_attempted_buckets.clear()

            # 更新进度（基于视频时间，每次百分比变化时通知）
            current_progress = self._progress_by_time(frame_time_seconds, video_duration, total_frames, state.frame_index)
            if current_progress > last_progress_value and progress_callback is not None:
                last_progress_value = current_progress
                progress_callback(VideoProgress(source_path, current_progress))

            # 第 3 层：运动检测 - 过滤无运动场景
            has_motion = motion_detector.has_motion(frame)

            # 第 4 层：决定是否进行 YOLO 检测
            should_detect = self._should_try_detection(
                frame_time_seconds,
                has_motion,
                state,
                drop_info,
                effective_sample_fps,
            )

            if should_detect and not state.current_slot_detected:
                # YOLO 检测
                detections = self._detect_people(frame)
                if detections:
                    timestamp = format_timestamp(frame_time_seconds)
                    event = self.result_writer.save_event(source_path, timestamp, frame, detections)
                    events.append(event)
                    state.current_slot_detected = True
                    state.last_detection_time = frame_time_seconds

                    logger.info(
                        f"检测到人形：{source_path.name} {timestamp}，"
                        f"置信度 {max(d.confidence for d in detections):.0%}"
                    )
                    if detection_callback is not None:
                        detection_callback(event)

                    # 丢帧补偿：在检测到人形后，对后续帧进行冗余检测
                    # 注意：此功能暂时禁用，因为它可能导致无限循环
                    # if drop_info.is_drop and self.config.enable_bidirectional_detection:
                    #     self._handle_post_drop_detection(
                    #         capture, frame_time_seconds, source_path, events,
                    #         detection_callback, state, total_frames
                    #     )

            # 维护双向检测缓冲区
            if self.config.enable_bidirectional_detection:
                pre_drop_buffer.append((frame_time_seconds, frame.copy()))
                if len(pre_drop_buffer) > 10:  # 保留最近 10 帧
                    pre_drop_buffer.pop(0)

            state.frame_index += 1

        # 发送最终进度
        if progress_callback is not None:
            if stop_token.is_stopped():
                final_progress = self._progress_by_time(
                    state.last_frame_time, video_duration, total_frames, state.frame_index,
                )
            else:
                final_progress = PROGRESS_MAX
            progress_callback(VideoProgress(source_path, final_progress))

        return events

    def _detect_frame_drop(
        self,
        current_time: float,
        last_time: float,
        video_fps: float,
    ) -> DropFrameInfo:
        """检测丢帧。

        Args:
            current_time: 当前帧时间戳
            last_time: 上一帧时间戳
            video_fps: 视频帧率

        Returns:
            丢帧信息
        """
        if last_time < 0:
            return DropFrameInfo()

        gap_duration = current_time - last_time
        expected_interval = 1.0 / video_fps if video_fps > 0 else 1.0 / self.config.sample_fps

        # 检查是否超过丢帧阈值
        if gap_duration > self.config.frame_drop_threshold_seconds:
            severity = gap_duration / expected_interval
            return DropFrameInfo(
                is_drop=True,
                gap_duration=gap_duration,
                severity=severity,
                drop_start_time=last_time,
                drop_end_time=current_time,
            )

        return DropFrameInfo()

    def _handle_frame_drop(
        self,
        state: FrameAnalysisState,
        drop_info: DropFrameInfo,
    ) -> None:
        """处理丢帧事件。

        Args:
            state: 帧分析状态
            drop_info: 丢帧信息
        """
        # 根据丢帧严重程度调整采样率
        if drop_info.severity >= self.config.frame_drop_severity_threshold:
            # 严重丢帧：大幅增加采样率
            multiplier = min(
                self.config.frame_drop_sample_rate_multiplier,
                drop_info.severity,
            )
            state.current_sample_fps = min(
                self.config.max_sample_fps,
                self.config.sample_fps * multiplier,
            )
            logger.debug(f"严重丢帧，采样率调整为：{state.current_sample_fps:.1f} 帧/秒")
        else:
            # 轻微丢帧：适度增加采样率
            state.current_sample_fps = min(
                self.config.max_sample_fps,
                self.config.sample_fps * 2.0,
            )
            logger.debug(f"轻微丢帧，采样率调整为：{state.current_sample_fps:.1f} 帧/秒")

        # 设置丢帧恢复期
        state.frame_drop_recovery_remaining = self.config.frame_drop_recovery_frames

    def _calculate_effective_sample_fps(
        self,
        state: FrameAnalysisState,
        drop_info: DropFrameInfo,
    ) -> float:
        """计算有效采样率。

        考虑丢帧恢复期和基础采样率。

        Args:
            state: 帧分析状态
            drop_info: 丢帧信息

        Returns:
            有效采样率
        """
        # 如果在丢帧恢复期，使用调整后的采样率
        if state.frame_drop_recovery_remaining > 0:
            state.frame_drop_recovery_remaining -= 1
            return state.current_sample_fps

        # 恢复期结束，逐渐恢复正常采样率
        if state.current_sample_fps > self.config.sample_fps:
            # 逐渐降低采样率
            state.current_sample_fps = max(
                self.config.sample_fps,
                state.current_sample_fps * 0.9,
            )
            return state.current_sample_fps

        return self.config.sample_fps

    def _calculate_frame_diff(
        self,
        current_frame: np.ndarray,
        previous_signature: np.ndarray | None,
    ) -> float:
        """计算帧差。

        Args:
            current_frame: 当前帧
            previous_signature: 上一帧签名

        Returns:
            帧差值（0-255）
        """
        if previous_signature is None:
            return 255.0  # 第一帧视为有变化

        current_signature = self._frame_signature(current_frame)

        # 计算绝对差值的平均值
        diff = np.abs(current_signature.astype(float) - previous_signature.astype(float))
        return float(np.mean(diff))

    def _is_keyframe(
        self,
        frame_index: int,
        drop_info: DropFrameInfo,
    ) -> bool:
        """判断是否为关键帧。

        关键帧包括：
        1. 固定间隔的帧
        2. 丢帧后的首个帧

        Args:
            frame_index: 帧索引
            drop_info: 丢帧信息

        Returns:
            是否为关键帧
        """
        # 固定间隔关键帧
        if self.config.enable_keyframe_detection:
            if frame_index % self.config.keyframe_interval == 0:
                return True

        # 丢帧后的首个帧
        if self.config.force_detection_after_drop and drop_info.is_drop:
            return True

        return False

    def _should_try_detection(
        self,
        frame_time_seconds: float,
        has_motion: bool,
        state: FrameAnalysisState,
        drop_info: DropFrameInfo,
        effective_sample_fps: float,
    ) -> bool:
        """判断是否应该尝试检测。

        实现多层检测漏斗策略：
        1. 关键帧强制检测
        2. 定时检测
        3. 运动触发检测
        4. 丢帧补偿检测

        Args:
            frame_time_seconds: 帧时间戳
            has_motion: 是否检测到运动
            state: 帧分析状态
            drop_info: 丢帧信息
            effective_sample_fps: 有效采样率

        Returns:
            如果应该尝试检测则返回 True
        """
        # 关键帧强制检测
        if self._is_keyframe(state.frame_index, drop_info):
            logger.debug(f"关键帧强制检测：帧 {state.frame_index}")
            return True

        # 滑动窗口检测
        if self.config.enable_sliding_window:
            slot_duration = 1.0 / effective_sample_fps
            slot_start = self._sample_slot(frame_time_seconds, effective_sample_fps) * slot_duration
            slot_offset = max(0.0, frame_time_seconds - slot_start)
            overlap_offset = slot_duration * self.config.sliding_window_overlap_ratio

            # 在滑动窗口重叠区域也进行检测
            if slot_offset < overlap_offset:
                candidate_bucket = int(slot_offset / overlap_offset * self.config.max_candidate_frames_per_slot)
                if candidate_bucket not in state.scheduled_attempted_buckets:
                    state.scheduled_attempted_buckets.add(candidate_bucket)
                    return True

        # 定时检测
        candidate_bucket = self._candidate_bucket(frame_time_seconds, effective_sample_fps)
        if (
            candidate_bucket not in state.scheduled_attempted_buckets
            and len(state.scheduled_attempted_buckets) < self.config.max_scheduled_detections_per_slot
        ):
            state.scheduled_attempted_buckets.add(candidate_bucket)
            return True

        # 运动触发检测
        if (
            has_motion
            and candidate_bucket not in state.motion_attempted_buckets
            and len(state.motion_attempted_buckets) < self.config.max_motion_detections_per_slot
        ):
            state.motion_attempted_buckets.add(candidate_bucket)
            return True

        # 丢帧补偿检测
        if drop_info.is_drop and self.config.force_detection_after_drop:
            # 丢帧后强制检测
            return True

        return False

    def _handle_post_drop_detection(
        self,
        capture: cv2.VideoCapture,
        drop_end_time: float,
        source_path: Path,
        events: list[DetectionEvent],
        detection_callback: Callable[[DetectionEvent], None] | None,
        state: FrameAnalysisState,
        total_frames: int,
    ) -> None:
        """处理丢帧后的冗余检测。

        在丢帧区域进行多次检测，确保不遗漏。

        Args:
            capture: 视频捕获对象
            drop_end_time: 丢帧结束时间
            source_path: 视频路径
            events: 事件列表
            detection_callback: 检测回调
            state: 帧分析状态
            total_frames: 总帧数
        """
        redundant_count = 0
        current_pos = capture.get(cv2.CAP_PROP_POS_FRAMES)

        for i in range(self.config.frame_drop_redundant_detections):
            # 向前跳过几帧
            skip_frames = (i + 1) * 5
            target_pos = int(current_pos) + skip_frames

            if target_pos >= total_frames:
                break

            capture.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
            ok, frame = capture.read()

            if ok:
                frame_time = self._frame_time_seconds(capture, target_pos, 0)
                detections = self._detect_people(frame)

                if detections:
                    timestamp = format_timestamp(frame_time)
                    event = self.result_writer.save_event(source_path, timestamp, frame, detections)
                    events.append(event)
                    redundant_count += 1

                    logger.info(
                        f"丢帧补偿检测到人形：{source_path.name} {timestamp}，"
                        f"置信度 {max(d.confidence for d in detections):.0%}"
                    )
                    if detection_callback is not None:
                        detection_callback(event)

        # 恢复到原来的位置
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        if redundant_count > 0:
            logger.info(f"丢帧补偿检测完成，额外检测到 {redundant_count} 个事件")

    def _prepare_video(self, source_path: Path) -> PreparedVideo:
        """准备视频文件，FFmpeg 无重编码重封装（修复索引）。

        ROI 裁剪在帧级别完成，不需要 FFmpeg 重编码。

        Args:
            source_path: 原始视频路径

        Returns:
            准备好的视频信息
        """
        if not self.config.remux_before_analysis:
            logger.debug("跳过 FFmpeg 重封装，直接分析原视频")
            return PreparedVideo(source_path=source_path, analysis_path=source_path)

        logger.debug(f"开始 FFmpeg 重封装：{source_path}")
        return self.remuxer.prepare(source_path)

    def _detect_people(self, frame: np.ndarray) -> list[PersonDetection]:
        """检测帧中的人形。

        Args:
            frame: 视频帧

        Returns:
            检测结果列表
        """
        if self.person_detector is None:
            logger.debug("延迟初始化 YoloPersonDetector")
            from monitor_scan.ai.yolo_detector import YoloPersonDetector

            self.person_detector = YoloPersonDetector(
                self.config.model_path,
                self.config.confidence_threshold,
                self.config.image_size,
                self.config.nms_threshold,
                self.config.coreml_warmup_runs,
            )
        return self.person_detector.detect(frame)

    def _default_motion_detector(self) -> MotionDetector:
        """创建默认的运动检测器。

        Returns:
            配置好的运动检测器实例
        """
        return MotionDetector(
            self.config.motion_threshold,
            self.config.motion_resize_width,
            self.config.motion_area_ratio_threshold,
            self.config.motion_detect_shadows,
        )

    def _open_ffmpeg_reader(self, path: Path) -> FfmpegFrameReader | None:
        """尝试使用 FFmpeg 硬件解码打开视频。

        Args:
            path: 视频文件路径

        Returns:
            FfmpegFrameReader 实例，失败时返回 None
        """
        # 快速检查：文件必须存在且大小合理
        if not path.exists() or path.stat().st_size < 1024:
            logger.debug("文件不存在或过小，跳过 FFmpeg 硬件解码")
            return None

        try:
            reader = FfmpegFrameReader(
                path,
                ffmpeg_path=self.config.ffmpeg_path,
                hw_accel=True,  # 优先硬件解码，卡死时自动回退 CPU
            )
            # 检查进程是否成功启动
            if reader.width > 0 and reader.height > 0 and reader._process is not None:
                return reader
            reader.release()
        except Exception as exc:
            logger.debug(f"FFmpeg 硬件解码不可用，回退到 OpenCV：{exc}")
        return None

    def _frame_time_seconds_by_index(self, frame_index: int, video_fps: float) -> float:
        """根据帧索引计算时间戳（秒）。

        适用于 FFmpeg 顺序读取模式，不依赖 capture.get()。

        Args:
            frame_index: 帧索引
            video_fps: 视频帧率

        Returns:
            时间戳（秒）
        """
        if video_fps > 0:
            return frame_index / video_fps
        return frame_index * self.config.sample_fps

    def _open_capture(self, path: Path) -> cv2.VideoCapture:
        """打开视频捕获对象。

        优先使用 FFmpeg 后端，失败时回退到默认后端。

        Args:
            path: 视频文件路径

        Returns:
            OpenCV 视频捕获对象
        """
        logger.debug(f"打开视频：{path}")

        # 优先尝试 FFmpeg 后端
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if capture.isOpened():
            return capture

        # 回退到默认后端
        capture.release()
        logger.debug("FFmpeg 后端失败，尝试默认后端")
        return cv2.VideoCapture(str(path))

    def _frame_time_seconds(self, capture: cv2.VideoCapture, frame_index: int, video_fps: float) -> float:
        """计算当前帧的时间戳（秒）。

        Args:
            capture: 视频捕获对象
            frame_index: 帧索引
            video_fps: 视频帧率

        Returns:
            帧时间戳（秒）
        """
        position_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
        if position_msec > 0:
            return position_msec / 1000

        # 回退到基于帧索引的计算
        safe_fps = video_fps if video_fps > 0 else self.config.sample_fps
        return frame_index / safe_fps

    def _sample_slot(self, frame_time_seconds: float, sample_fps: float | None = None) -> int:
        """计算帧所属的采样时间槽。

        Args:
            frame_time_seconds: 帧时间戳（秒）
            sample_fps: 采样率（可选，默认使用配置值）

        Returns:
            采样时间槽索引
        """
        fps = sample_fps if sample_fps is not None else self.config.sample_fps
        return int(frame_time_seconds * fps)

    def _candidate_bucket(self, frame_time_seconds: float, sample_fps: float | None = None) -> int:
        """计算帧在时间槽内的候选桶索引。

        Args:
            frame_time_seconds: 帧时间戳（秒）
            sample_fps: 采样率（可选，默认使用配置值）

        Returns:
            候选桶索引
        """
        fps = sample_fps if sample_fps is not None else self.config.sample_fps
        slot_duration = 1 / fps
        slot_start = self._sample_slot(frame_time_seconds, fps) * slot_duration
        slot_offset = max(0.0, frame_time_seconds - slot_start)
        bucket = int(slot_offset / slot_duration * self.config.max_candidate_frames_per_slot)
        return min(self.config.max_candidate_frames_per_slot - 1, bucket)

    def _frame_signature(self, frame: np.ndarray) -> np.ndarray:
        """生成帧签名，用于检测停滞帧和帧差计算。

        Args:
            frame: 视频帧

        Returns:
            帧签名（灰度缩略图）
        """
        resized = cv2.resize(
            frame,
            (self.config.frame_signature_size, self.config.frame_signature_size),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def _is_stalled_frame(
        self,
        frame_signature: np.ndarray,
        previous_frame_signature: np.ndarray | None,
        capture_position: float,
        previous_capture_position: float,
        capture_msec: float,
        previous_capture_msec: float,
    ) -> bool:
        """检测当前帧是否为停滞帧。

        停滞帧是指解码器返回相同内容且播放位置未前进的帧。

        Args:
            frame_signature: 当前帧签名
            previous_frame_signature: 前一帧签名
            capture_position: 当前捕获位置
            previous_capture_position: 前一捕获位置
            capture_msec: 当前捕获时间戳
            previous_capture_msec: 前一捕获时间戳

        Returns:
            如果是停滞帧则返回 True
        """
        if previous_frame_signature is None:
            return False
        if not np.array_equal(frame_signature, previous_frame_signature):
            return False

        # 检查位置是否前进
        position_available = capture_position > 0 and previous_capture_position > 0
        if position_available:
            # 如果位置相同，检查时间戳
            if capture_position == previous_capture_position:
                # 位置相同，检查时间戳是否前进
                msec_available = capture_msec > 0 and previous_capture_msec > 0
                if msec_available:
                    return capture_msec <= previous_capture_msec
                # 无法判断，假设是停滞帧
                return True
            # 位置后退，是停滞帧
            return capture_position < previous_capture_position

        # 回退到时间戳检查
        msec_available = capture_msec > 0 and previous_capture_msec > 0
        return msec_available and capture_msec <= previous_capture_msec

    def _skip_failed_frame(self, capture: cv2.VideoCapture, frame_index: int, total_frames: int) -> bool:
        """尝试跳过失败的帧。

        Args:
            capture: 视频捕获对象
            frame_index: 当前帧索引
            total_frames: 总帧数

        Returns:
            如果应该继续处理则返回 True，如果应该停止则返回 False
        """
        # 尝试跳到下一帧
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index + 1)

        # 总是返回 True，让调用者通过 consecutive_decode_failures 来判断是否应该停止
        # 这样可以处理 total_frames 报告不准确的情况
        return True

    @staticmethod
    def _progress_by_time(
        current_time: float,
        duration: float,
        total_frames: int,
        frame_index: int,
    ) -> int:
        """基于视频时间计算进度百分比。

        优先使用时间/时长，回退到帧索引/总帧数。

        Args:
            current_time: 当前帧时间戳（秒）
            duration: 视频总时长（秒）
            total_frames: 总帧数
            frame_index: 当前帧索引

        Returns:
            进度百分比 (0-100)
        """
        if duration > 0:
            return min(PROGRESS_MAX, int(current_time / duration * PROGRESS_MAX))
        if total_frames > 0:
            return min(PROGRESS_MAX, int(frame_index / total_frames * PROGRESS_MAX))
        # 都不可用时，每 100 帧算 1%
        return min(PROGRESS_MAX - 1, frame_index // 100)
