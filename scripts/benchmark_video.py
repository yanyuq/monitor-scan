from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from monitor_scan.ai.yolo_detector import COREML_COMPUTE_UNIT, YoloPersonDetector
from monitor_scan.config import AppConfig
from monitor_scan.video.analyzer import VideoAnalyzer


class NeverStop:
    def is_stopped(self) -> bool:
        return False


class CountingDetector:
    def __init__(self, detector: YoloPersonDetector) -> None:
        self.detector = detector
        self.calls = 0
        self.latencies = []

    def detect(self, frame):
        start = time.perf_counter()
        detections = self.detector.detect(frame)
        self.latencies.append(time.perf_counter() - start)
        self.calls += 1
        return detections


def main() -> int:
    parser = argparse.ArgumentParser(description="本地视频端到端性能基准")
    parser.add_argument("--video", type=Path, required=True, help="样本视频路径")
    parser.add_argument("--model", type=Path, default=AppConfig().model_path, help="模型路径")
    parser.add_argument("--mode", choices=["m1-coreml", "custom"], default="m1-coreml", help="基准模式")
    parser.add_argument("--imgsz", type=int, default=512, help="检测输入尺寸")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="抽帧频率")
    parser.add_argument("--keep-output", action="store_true", help="保留输出目录")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"样本视频不存在：{args.video}")

    output_directory = Path(tempfile.mkdtemp(prefix="monitor-scan-benchmark-"))
    try:
        config = AppConfig(
            sample_fps=args.sample_fps,
            model_path=args.model,
            output_directory=output_directory,
            image_size=args.imgsz,
            max_candidate_frames_per_slot=2,
            max_scheduled_detections_per_slot=1,
            max_motion_detections_per_slot=1,
            remux_before_analysis=False,
        )
        detector = CountingDetector(
            YoloPersonDetector(
                config.model_path,
                config.confidence_threshold,
                config.image_size,
                config.nms_threshold,
                config.coreml_warmup_runs,
            )
        )
        analyzer = VideoAnalyzer(config, person_detector=detector)
        start = time.perf_counter()
        events = analyzer.analyze_video(args.video, NeverStop())
        elapsed = time.perf_counter() - start
        result = {
            "video": str(args.video),
            "model": str(args.model),
            "mode": args.mode,
            "backend": detector.detector.backend,
            "compute_units": COREML_COMPUTE_UNIT,
            "elapsed_seconds": elapsed,
            "yolo_calls": detector.calls,
            "mean_detect_ms": _mean_ms(detector.latencies),
            "p95_detect_ms": _percentile_ms(detector.latencies, 95),
            "events_count": len(events),
            "output_directory": str(output_directory),
            "peak_rss_mb": _peak_rss_mb(),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if not args.keep_output:
            shutil.rmtree(output_directory, ignore_errors=True)
    return 0


def _mean_ms(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values) * 1000


def _percentile_ms(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
    return ordered[index] * 1000


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024 / 1024 if usage > 10_000_000 else usage / 1024


if __name__ == "__main__":
    raise SystemExit(main())
