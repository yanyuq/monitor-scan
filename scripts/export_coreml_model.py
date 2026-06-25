from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "models" / "yolo26n.pt"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "yolo26n-512-fp16-nms.mlpackage"


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 yolo26n CoreML 模型")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="yolo26n PyTorch 源模型路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CoreML mlpackage 输出路径")
    parser.add_argument("--imgsz", type=int, default=512, help="导出模型输入尺寸")
    parser.add_argument("--nms", dest="nms", action="store_true", default=True, help="导出时内置 NMS")
    parser.add_argument("--no-nms", dest="nms", action="store_false", help="导出时不内置 NMS")
    parser.add_argument("--half", dest="half", action="store_true", default=True, help="使用 FP16 导出 CoreML 模型")
    parser.add_argument("--no-half", dest="half", action="store_false", help="不使用 FP16 导出 CoreML 模型")
    parser.add_argument("--int8", action="store_true", help="使用 INT8 量化导出 CoreML 模型")
    args = parser.parse_args()

    export_coreml_model(args.source, args.output, args.imgsz, nms=args.nms, half=args.half, int8=args.int8)
    print(f"已导出 CoreML 模型：{args.output}")
    return 0


def export_coreml_model(
    source: Path,
    output: Path,
    image_size: int = 512,
    nms: bool = True,
    half: bool = True,
    int8: bool = False,
) -> Path:
    source = source.resolve()
    output = output.resolve()
    if not source.exists():
        raise SystemExit(f"缺少 yolo26n 源模型，无法导出 CoreML：{source}")
    if image_size <= 0:
        raise SystemExit("导出模型输入尺寸必须大于 0。")
    if half and int8:
        raise SystemExit("FP16 和 INT8 不能同时启用。")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("缺少 ultralytics 依赖，无法导出 CoreML 模型。") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        exported_path = Path(
            YOLO(str(source), task="detect").export(
                format="coreml",
                imgsz=image_size,
                nms=nms,
                half=half,
                int8=int8,
            )
        ).resolve()
    except Exception as error:
        raise SystemExit(f"导出 CoreML 模型失败：{error}") from error

    if exported_path != output:
        _remove_path(output)
        shutil.move(str(exported_path), str(output))
    if not output.exists():
        raise SystemExit(f"CoreML 导出完成但未找到输出文件：{output}")
    return output


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
