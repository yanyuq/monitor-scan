from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from monitor_scan.gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
