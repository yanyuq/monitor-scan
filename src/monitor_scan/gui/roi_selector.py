"""ROI 区域框选对话框。

提供可视化方式在视频首帧上框选检测区域。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)

# 对话框最大显示尺寸
MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800


class RoiCanvas(QLabel):
    """ROI 框选画布，支持鼠标拖拽绘制矩形选区。

    Signals:
        roi_changed: 选区变化信号，参数为原始像素坐标 (x, y, w, h)，无选区时为 None
    """

    roi_changed = pyqtSignal(object)

    def __init__(self, parent: QLabel | None = None) -> None:
        super().__init__(parent)
        self._original_pixmap: QPixmap | None = None
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0
        self._origin: QPoint | None = None  # 鼠标按下起点（显示坐标）
        self._selection: QRect | None = None  # 当前选区（显示坐标）
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)

    def set_frame(self, frame: np.ndarray) -> None:
        """设置要显示的视频帧。

        Args:
            frame: BGR 格式的 numpy 数组
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        original = QPixmap.fromImage(qimage)

        # 缩放到合适大小
        scaled = original.scaled(
            MAX_DISPLAY_WIDTH,
            MAX_DISPLAY_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._scale_x = original.width() / scaled.width()
        self._scale_y = original.height() / scaled.height()
        self._original_pixmap = original
        self.setPixmap(scaled)
        self._selection = None
        self._origin = None

    def get_roi(self) -> tuple[int, int, int, int] | None:
        """获取当前选区的原始像素坐标。

        Returns:
            (x, y, width, height) 或 None
        """
        if self._selection is None:
            return None
        rect = self._selection.normalized()
        x = int(rect.x() * self._scale_x)
        y = int(rect.y() * self._scale_y)
        w = int(rect.width() * self._scale_x)
        h = int(rect.height() * self._scale_y)
        return (x, y, w, h)

    def clear_selection(self) -> None:
        """清除选区。"""
        self._selection = None
        self._origin = None
        self.roi_changed.emit(None)
        self._redraw()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.pixmap() is not None:
            self._origin = event.pos()
            self._selection = QRect(self._origin, self._origin)
            self._redraw()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._origin is not None:
            self._selection = QRect(self._origin, event.pos()).normalized()
            self._redraw()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._origin is not None:
            self._selection = QRect(self._origin, event.pos()).normalized()
            self._origin = None
            self._redraw()
            roi = self.get_roi()
            self.roi_changed.emit(roi)

    def _redraw(self) -> None:
        """重绘画布，在原始图像上叠加半透明矩形选区。"""
        pixmap = self._original_pixmap
        if pixmap is None:
            return
        # 缩放到显示尺寸
        scaled = pixmap.scaled(
            MAX_DISPLAY_WIDTH,
            MAX_DISPLAY_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        overlay = scaled.copy()
        painter = QPainter(overlay)
        painter.setBrush(Qt.GlobalColor.red)
        painter.setOpacity(0.25)
        painter.setPen(QPen(Qt.GlobalColor.red, 2))
        if self._selection is not None:
            painter.drawRect(self._selection.normalized())
        painter.end()
        self.setPixmap(overlay)


class RoiSelectorDialog(QDialog):
    """ROI 区域框选对话框。

    在视频首帧上通过鼠标拖拽框选检测区域，确认后返回原始像素坐标。
    """

    def __init__(self, frame: np.ndarray, parent=None) -> None:
        """初始化框选对话框。

        Args:
            frame: BGR 格式的视频帧（原始分辨率）
            parent: 父窗口
        """
        super().__init__(parent)
        self.setWindowTitle("框选检测区域")
        self.setMinimumSize(800, 600)

        self._roi: tuple[int, int, int, int] | None = None

        # 画布
        self._canvas = RoiCanvas()
        self._canvas.roi_changed.connect(self._on_roi_changed)

        # 坐标显示
        self._coord_label = QLabel("请在画面上拖拽鼠标框选检测区域")
        self._coord_label.setStyleSheet("font-size: 14px; padding: 4px;")

        # 按钮
        btn_layout = QHBoxLayout()
        self._clear_btn = QPushButton("清除选区")
        self._confirm_btn = QPushButton("确认")
        self._cancel_btn = QPushButton("取消")
        self._confirm_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._clear_selection)
        self._confirm_btn.clicked.connect(self.accept)
        self._cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._clear_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self._confirm_btn)
        btn_layout.addWidget(self._cancel_btn)

        # 布局
        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._coord_label)
        layout.addLayout(btn_layout)

        # 加载帧
        self._canvas.set_frame(frame)

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        """获取框选的 ROI 坐标（原始像素）。"""
        return self._roi

    def _on_roi_changed(self, roi: tuple[int, int, int, int] | None) -> None:
        """选区变化回调。"""
        self._roi = roi
        if roi is not None:
            x, y, w, h = roi
            self._coord_label.setText(f"选区：X={x}  Y={y}  宽={w}  高={h}")
            self._confirm_btn.setEnabled(True)
        else:
            self._coord_label.setText("请在画面上拖拽鼠标框选检测区域")
            self._confirm_btn.setEnabled(False)

    def _clear_selection(self) -> None:
        """清除选区。"""
        self._canvas.clear_selection()
