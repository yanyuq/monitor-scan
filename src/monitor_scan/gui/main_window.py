from __future__ import annotations

from pathlib import Path

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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from monitor_scan.config import AppConfig
from monitor_scan.types import DetectionEvent
from monitor_scan.video.scanner import VideoScanner
from monitor_scan.workers.analyzer_worker import AnalyzerWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
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

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        controls = QGroupBox("控制与配置")
        controls_layout = QGridLayout(controls)
        self.directory_label = QLabel("未选择视频目录")
        self.choose_button = QPushButton("选择文件夹")
        self.start_button = QPushButton("开始分析")
        self.stop_button = QPushButton("停止")

        self.sample_fps_input = QDoubleSpinBox()
        self.sample_fps_input.setRange(0.1, 30.0)
        self.sample_fps_input.setSingleStep(0.5)
        self.sample_fps_input.setValue(2.0)
        self.sample_fps_input.setSuffix(" 帧/秒")

        self.confidence_input = QDoubleSpinBox()
        self.confidence_input.setRange(0.05, 0.99)
        self.confidence_input.setSingleStep(0.05)
        self.confidence_input.setValue(0.5)
        self.confidence_input.setDecimals(2)

        controls_layout.addWidget(self.choose_button, 0, 0)
        controls_layout.addWidget(self.directory_label, 0, 1, 1, 5)
        controls_layout.addWidget(QLabel("抽帧频率"), 1, 0)
        controls_layout.addWidget(self.sample_fps_input, 1, 1)
        controls_layout.addWidget(QLabel("AI 灵敏度"), 1, 2)
        controls_layout.addWidget(self.confidence_input, 1, 3)
        controls_layout.addWidget(self.start_button, 1, 4)
        controls_layout.addWidget(self.stop_button, 1, 5)
        layout.addWidget(controls)

        content_layout = QHBoxLayout()
        self.video_table = QTableWidget(0, 3)
        self.video_table.setHorizontalHeaderLabels(["视频文件", "状态", "进度"])
        self.video_table.horizontalHeader().setStretchLastSection(True)
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.video_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        content_layout.addWidget(self.video_table, 3)

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

        self.choose_button.clicked.connect(self.choose_directory)
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)

    def choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择视频目录")
        if not directory:
            return
        self.load_directory(Path(directory))

    def load_directory(self, directory: Path) -> None:
        try:
            videos = self._scanner.scan(directory)
        except Exception as exc:
            QMessageBox.warning(self, "目录读取失败", str(exc))
            return

        self._selected_directory = directory
        self._videos = videos
        self.directory_label.setText(str(directory))
        self._populate_videos(videos)
        self._append_log(f"已加载 {len(videos)} 个视频文件。")

    def start_analysis(self) -> None:
        if self._thread is not None:
            return
        if self._selected_directory is None:
            QMessageBox.information(self, "请先选择目录", "请先选择包含监控视频的文件夹。")
            return
        if not self._videos:
            QMessageBox.information(self, "没有视频", "所选目录中没有可分析的视频文件。")
            return

        config = AppConfig(
            sample_fps=self.sample_fps_input.value(),
            confidence_threshold=self.confidence_input.value(),
            output_directory=self._selected_directory / "output_results",
        )
        if not config.model_path.exists():
            QMessageBox.warning(self, "缺少模型文件", f"请先放置模型文件：{config.model_path}")
            return

        self.result_table.setRowCount(0)
        self.overall_progress.setValue(0)
        self._set_running(True)
        self._thread = QThread(self)
        self._worker = AnalyzerWorker(self._videos, config)
        self._worker.moveToThread(self._thread)
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

    def stop_analysis(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._append_log("已请求停止，正在等待当前帧处理结束。")
            self.stop_button.setEnabled(False)

    def _populate_videos(self, videos: list[Path]) -> None:
        self.video_table.setRowCount(len(videos))
        self._rows_by_path.clear()
        for row, video in enumerate(videos):
            self._rows_by_path[str(video)] = row
            self.video_table.setItem(row, 0, QTableWidgetItem(video.name))
            self.video_table.setItem(row, 1, QTableWidgetItem("等待中"))
            self.video_table.setItem(row, 2, QTableWidgetItem("0%"))

    def _update_video_status(self, path: str, status: str, progress: int) -> None:
        row = self._rows_by_path.get(path)
        if row is None:
            return
        self.video_table.setItem(row, 1, QTableWidgetItem(status))
        self.video_table.setItem(row, 2, QTableWidgetItem(f"{progress}%"))

    def _update_overall_progress(self, done: int, total: int) -> None:
        percent = 0 if total == 0 else int(done / total * 100)
        self.overall_progress.setValue(percent)
        self.overall_progress.setFormat(f"{done}/{total} 个文件")

    def _add_detection(self, event: DetectionEvent) -> None:
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)
        self.result_table.setItem(row, 0, QTableWidgetItem(event.video_name))
        self.result_table.setItem(row, 1, QTableWidgetItem(event.timestamp))
        self.result_table.setItem(row, 2, QTableWidgetItem(f"{event.confidence:.0%}"))
        self.result_table.setItem(row, 3, QTableWidgetItem(event.snapshot_path))

    def _show_worker_error(self, message: str) -> None:
        self._append_log(f"错误：{message}")

    def _analysis_finished(self) -> None:
        self._append_log("分析任务结束。")
        self._set_running(False)

    def _clear_thread(self) -> None:
        self._thread = None
        self._worker = None

    def _set_running(self, running: bool) -> None:
        self.choose_button.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.sample_fps_input.setEnabled(not running)
        self.confidence_input.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        event.accept()
