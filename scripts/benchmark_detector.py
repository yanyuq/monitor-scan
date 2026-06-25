from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from monitor_scan.ai.yolo_detector import COREML_COMPUTE_UNIT, YoloPersonDetector
from monitor_scan.config import AppConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="本地检测器性能基准")
    parser.add_argument("--model", type=Path, default=AppConfig().model_path, help="模型路径")
    parser.add_argument("--imgsz", type=int, default=512, help="检测输入尺寸")
    parser.add_argument("--confidence", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iterations", type=int, default=50, help="稳定推理次数")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    args = parser.parse_args()

    if args.iterations <= 0:
        raise SystemExit("稳定推理次数必须大于 0。")
    if args.warmup < 0:
        raise SystemExit("预热次数不能小于 0。")

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    load_start = time.perf_counter()
    detector = YoloPersonDetector(
        args.model,
        confidence_threshold=args.confidence,
        image_size=args.imgsz,
        coreml_warmup_runs=0,
    )
    load_seconds = time.perf_counter() - load_start

    first_start = time.perf_counter()
    detector.detect(frame)
    first_seconds = time.perf_counter() - first_start

    for _ in range(args.warmup):
        detector.detect(frame)

    latencies = []
    detections_count = 0
    for _ in range(args.iterations):
        start = time.perf_counter()
        detections = detector.detect(frame)
        latencies.append(time.perf_counter() - start)
        detections_count += len(detections)

    result = {
        "model": str(args.model),
        "backend": detector.backend,
        "image_size": args.imgsz,
        "compute_units": COREML_COMPUTE_UNIT,
        "load_seconds": load_seconds,
        "first_inference_seconds": first_seconds,
        "warm_p50_ms": statistics.median(latencies) * 1000,
        "warm_p90_ms": _percentile(latencies, 90) * 1000,
        "warm_p95_ms": _percentile(latencies, 95) * 1000,
        "warm_mean_ms": statistics.fmean(latencies) * 1000,
        "iterations": args.iterations,
        "detections_count": detections_count,
        "peak_rss_mb": _peak_rss_mb(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
    return ordered[index]


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024 / 1024 if usage > 10_000_000 else usage / 1024


if __name__ == "__main__":
    raise SystemExit(main())
