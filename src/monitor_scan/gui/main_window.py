from __future__ import annotations

import logging
from pathlib import Path

import cv2
from PyQt6.QtCore import QThread, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from monitor_scan.config import AppConfig
from monitor_scan.gui.roi_selector import RoiSelectorDialog
from monitor_scan.types import DetectionEvent
from monitor_scan.video.scanner import VideoScanner
from monitor_scan.workers.analyzer_worker import AnalyzerWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """监控视频智能分析系统主窗口。

    提供图形用户界面，支持：
    - 选择视频目录
    - 配置分析参数（抽帧频率、AI 灵敏度）
    - 启动/停止分析任务
    - 显示处理进度和检测结果
    - 实时日志输出

    Attributes:
        _scanner: 视频文件扫描器
        _selected_directory: 当前选择的视频目录
        _videos: 待分析的视频文件列表
        _thread: 工作线程
        _worker: 分析工作器
        _rows_by_path: 视频路径到表格行号的映射
    """

    def __init__(self) -> None:
        """初始化主窗口。"""
        super().__init__()
        self.setWindowTitle("监控视频智能分析系统")
        self.resize(1100, 720)
        self._scanner = VideoScanner()
        self._selected_directory: Path | None = None
        self._videos: list[Path] = []
        self._thread: QThread | None = None
        self._worker: AnalyzerWorker | None = None
        self._rows_by_path: dict[str, int] = {}

        self._build_ui()
        self._set_running(False)
        logger.info("主窗口初始化完成")

    def _build_ui(self) -> None:
        """构建用户界面。"""
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # 控制面板
        controls = QGroupBox("控制与配置")
        controls_layout = QGridLayout(controls)
        self.directory_label = QLabel("未选择视频目录")
        self.choose_button = QPushButton("选择文件夹")
        self.start_button = QPushButton("开始分析")
        self.stop_button = QPushButton("停止")

        # 抽帧频率输入
        self.sample_fps_input = QDoubleSpinBox()
        self.sample_fps_input.setRange(0.1, 30.0)
        self.sample_fps_input.setSingleStep(0.5)
        self.sample_fps_input.setValue(1.0)
        self.sample_fps_input.setSuffix(" 帧/秒")

        # AI 灵敏度输入
        self.confidence_input = QDoubleSpinBox()
        self.confidence_input.setRange(0.05, 0.99)
        self.confidence_input.setSingleStep(0.05)
        self.confidence_input.setValue(0.5)
        self.confidence_input.setDecimals(2)

        # ROI 检测区域输入（留空表示不裁剪）
        self.roi_x_input = QSpinBox()
        self.roi_x_input.setRange(0, 99999)
        self.roi_x_input.setSpecialValueText("留空")
        self.roi_x_input.setPrefix("X ")
        self.roi_y_input = QSpinBox()
        self.roi_y_input.setRange(0, 99999)
        self.roi_y_input.setSpecialValueText("留空")
        self.roi_y_input.setPrefix("Y ")
        self.roi_w_input = QSpinBox()
        self.roi_w_input.setRange(0, 99999)
        self.roi_w_input.setSpecialValueText("留空")
        self.roi_w_input.setPrefix("宽 ")
        self.roi_h_input = QSpinBox()
        self.roi_h_input.setRange(0, 99999)
        self.roi_h_input.setSpecialValueText("留空")
        self.roi_h_input.setPrefix("高 ")
        self.roi_select_button = QPushButton("框选区域")
        self.roi_select_button.setToolTip("加载视频首帧，鼠标拖拽框选检测区域")

        controls_layout.addWidget(self.choose_button, 0, 0)
        controls_layout.addWidget(self.directory_label, 0, 1, 1, 6)
        controls_layout.addWidget(QLabel("抽帧频率"), 1, 0)
        controls_layout.addWidget(self.sample_fps_input, 1, 1)
        controls_layout.addWidget(QLabel("AI 灵敏度"), 1, 2)
        controls_layout.addWidget(self.confidence_input, 1, 3)
        controls_layout.addWidget(self.start_button, 1, 4)
        controls_layout.addWidget(self.stop_button, 1, 5)
        controls_layout.addWidget(QLabel("检测区域"), 2, 0)
        controls_layout.addWidget(self.roi_x_input, 2, 1)
        controls_layout.addWidget(self.roi_y_input, 2, 2)
        controls_layout.addWidget(self.roi_w_input, 2, 3)
        controls_layout.addWidget(self.roi_h_input, 2, 4)
        controls_layout.addWidget(self.roi_select_button, 2, 5)
        layout.addWidget(controls)

        # 内容区域
        content_layout = QHBoxLayout()

        # 视频列表
        self.video_table = QTableWidget(0, 3)
        self.video_table.setHorizontalHeaderLabels(["视频文件", "状态", "进度"])
        self.video_table.horizontalHeader().setStretchLastSection(True)
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.video_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        content_layout.addWidget(self.video_table, 3)

        # 右侧面板
        right_panel = QVBoxLayout()
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["视频文件", "事件时间", "置信度", "截图路径"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)

        right_panel.addWidget(QLabel("总体进度"))
        right_panel.addWidget(self.overall_progress)
        right_panel.addWidget(QLabel("检测结果"))
        right_panel.addWidget(self.result_table, 2)
        right_panel.addWidget(QLabel("实时日志"))
        right_panel.addWidget(self.log_output, 2)
        content_layout.addLayout(right_panel, 2)
        layout.addLayout(content_layout)

        # 连接信号
        self.choose_button.clicked.connect(self.choose_directory)
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.roi_select_button.clicked.connect(self._open_roi_selector)

    def choose_directory(self) -> None:
        """打开目录选择对话框。"""
        directory = QFileDialog.getExistingDirectory(self, "选择视频目录")
        if not directory:
            return
        self.load_directory(Path(directory))

    def _open_roi_selector(self) -> None:
        """打开 ROI 框选对话框，加载第一个视频的首帧。"""
        if not self._videos:
            QMessageBox.information(self, "没有视频", "请先加载包含视频文件的目录。")
            return

        video_path = self._videos[0]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            QMessageBox.warning(self, "视频无法打开", f"无法读取视频首帧：{video_path.name}")
            logger.warning(f"无法打开视频读取首帧：{video_path}")
            return

        try:
            ok, frame = capture.read()
        finally:
            capture.release()

        if not ok or frame is None:
            QMessageBox.warning(self, "读取失败", f"无法读取视频首帧：{video_path.name}")
            logger.warning(f"读取视频首帧失败：{video_path}")
            return

        dialog = RoiSelectorDialog(frame, parent=self)
        if dialog.exec() == RoiSelectorDialog.DialogCode.Accepted and dialog.roi is not None:
            x, y, w, h = dialog.roi
            self.roi_x_input.setValue(x)
            self.roi_y_input.setValue(y)
            self.roi_w_input.setValue(w)
            self.roi_h_input.setValue(h)
            self._append_log(f"已设置检测区域：X={x}, Y={y}, 宽={w}, 高={h}")
            logger.info(f"用户框选 ROI：x={x}, y={y}, width={w}, height={h}")

    def load_directory(self, directory: Path) -> None:
        """加载指定目录中的视频文件。

        Args:
            directory: 视频目录路径
        """
        try:
            videos = self._scanner.scan(directory)
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "目录不存在", str(exc))
            logger.warning(f"目录不存在：{directory}")
            return
        except NotADirectoryError as exc:
            QMessageBox.warning(self, "路径错误", str(exc))
            logger.warning(f"路径不是目录：{directory}")
            return
        except PermissionError as exc:
            QMessageBox.warning(self, "权限不足", f"无法访问目录：{exc}")
            logger.warning(f"权限不足：{directory}")
            return
        except Exception as exc:
            QMessageBox.warning(self, "目录读取失败", f"发生未知错误：{exc}")
            logger.exception(f"读取目录时发生未知错误：{directory}")
            return

        self._selected_directory = directory
        self._videos = videos
        self.directory_label.setText(str(directory))
        self._populate_videos(videos)
        self._append_log(f"已加载 {len(videos)} 个视频文件。")
        logger.info(f"已加载目录：{directory}，包含 {len(videos)} 个视频文件")

    def start_analysis(self) -> None:
        """开始分析任务。"""
        if self._thread is not None:
            return
        if self._selected_directory is None:
            QMessageBox.information(self, "请先选择目录", "请先选择包含监控视频的文件夹。")
            return
        if not self._videos:
            QMessageBox.information(self, "没有视频", "所选目录中没有可分析的视频文件。")
            return

        try:
            config = self._analysis_config()
        except ValueError as exc:
            QMessageBox.warning(self, "配置错误", str(exc))
            logger.warning(f"配置验证失败：{exc}")
            return

        if not config.model_path.exists():
            QMessageBox.warning(self, "缺少模型文件", f"请先放置模型文件：{config.model_path}")
            logger.warning(f"模型文件不存在：{config.model_path}")
            return

        self.result_table.setRowCount(0)
        self.overall_progress.setValue(0)
        self._set_running(True)

        # 创建工作线程
        self._thread = QThread(self)
        self._worker = AnalyzerWorker(self._videos, config)
        self._worker.moveToThread(self._thread)

        # 连接信号
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.file_status.connect(self._update_video_status)
        self._worker.overall_progress.connect(self._update_overall_progress)
        self._worker.detection.connect(self._add_detection)
        self._worker.error.connect(self._show_worker_error)
        self._worker.finished.connect(self._analysis_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_thread)

        self._thread.start()
        logger.info("分析任务已启动")

    def stop_analysis(self) -> None:
        """停止分析任务。"""
        if self._worker is not None:
            self._worker.stop()
            self._append_log("已请求停止，正在等待当前帧处理结束。")
            self.stop_button.setEnabled(False)
            logger.info("已请求停止分析任务")

    def _analysis_config(self) -> AppConfig:
        """创建分析配置。

        Returns:
            配置好的 AppConfig 实例

        Raises:
            RuntimeError: 如果未选择目录
            ValueError: 如果配置验证失败
        """
        if self._selected_directory is None:
            raise RuntimeError("请先选择包含监控视频的文件夹。")

        # 读取 ROI 输入（0 表示留空/未设置）
        roi_x = self.roi_x_input.value() or None
        roi_y = self.roi_y_input.value() or None
        roi_w = self.roi_w_input.value() or None
        roi_h = self.roi_h_input.value() or None

        return AppConfig(
            sample_fps=self.sample_fps_input.value(),
            confidence_threshold=self.confidence_input.value(),
            output_directory=self._selected_directory / "output_results",
            image_size=512,
            max_candidate_frames_per_slot=2,
            max_scheduled_detections_per_slot=1,
            max_motion_detections_per_slot=1,
            motion_resize_width=480,
            motion_area_ratio_threshold=0.01,
            motion_detect_shadows=False,
            roi_x=roi_x,
            roi_y=roi_y,
            roi_width=roi_w,
            roi_height=roi_h,
        )

    def _populate_videos(self, videos: list[Path]) -> None:
        """填充视频列表表格。

        Args:
            videos: 视频文件路径列表
        """
        self.video_table.setRowCount(len(videos))
        self._rows_by_path.clear()
        for row, video in enumerate(videos):
            self._rows_by_path[str(video)] = row
            self.video_table.setItem(row, 0, QTableWidgetItem(video.name))
            self.video_table.setItem(row, 1, QTableWidgetItem("等待中"))
            self.video_table.setItem(row, 2, QTableWidgetItem("0%"))

    def _update_video_status(self, path: str, status: str, progress: int) -> None:
        """更新视频处理状态。

        Args:
            path: 视频文件路径
            status: 状态文本
            progress: 进度百分比
        """
        row = self._rows_by_path.get(path)
        if row is None:
            return
        self.video_table.setItem(row, 1, QTableWidgetItem(status))
        self.video_table.setItem(row, 2, QTableWidgetItem(f"{progress}%"))

    def _update_overall_progress(self, done: int, total: int) -> None:
        """更新总体进度。

        Args:
            done: 已完成数量
            total: 总数量
        """
        percent = 0 if total == 0 else int(done / total * 100)
        self.overall_progress.setValue(percent)
        self.overall_progress.setFormat(f"{done}/{total} 个文件")

    def _add_detection(self, event: DetectionEvent) -> None:
        """添加检测结果到结果表格。

        Args:
            event: 检测事件
        """
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)
        self.result_table.setItem(row, 0, QTableWidgetItem(event.video_name))
        self.result_table.setItem(row, 1, QTableWidgetItem(event.timestamp))
        self.result_table.setItem(row, 2, QTableWidgetItem(f"{event.confidence:.0%}"))
        self.result_table.setItem(row, 3, QTableWidgetItem(event.snapshot_path))

    def _show_worker_error(self, message: str) -> None:
        """显示工作器错误消息。

        Args:
            message: 错误消息
        """
        self._append_log(f"错误：{message}")
        logger.error(f"工作器错误：{message}")

    def _analysis_finished(self) -> None:
        """分析任务完成回调。"""
        self._append_log("分析任务结束。")
        self._set_running(False)
        logger.info("分析任务已完成")

    def _clear_thread(self) -> None:
        """清理线程引用。"""
        self._thread = None
        self._worker = None

    def _set_running(self, running: bool) -> None:
        """设置运行状态，更新 UI 控件启用状态。

        Args:
            running: 是否正在运行
        """
        self.choose_button.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.sample_fps_input.setEnabled(not running)
        self.confidence_input.setEnabled(not running)
        self.roi_x_input.setEnabled(not running)
        self.roi_y_input.setEnabled(not running)
        self.roi_w_input.setEnabled(not running)
        self.roi_h_input.setEnabled(not running)
        self.roi_select_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _append_log(self, message: str) -> None:
        """追加日志消息到日志输出区域。

        Args:
            message: 日志消息
        """
        self.log_output.appendPlainText(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def closeEvent(self, event) -> None:
        """窗口关闭事件处理。

        Args:
            event: 关闭事件
        """
        logger.info("主窗口正在关闭")
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        event.accept()
