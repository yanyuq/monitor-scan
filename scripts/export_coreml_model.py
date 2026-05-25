from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "models" / "yolo26n.pt"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "yolo26n.mlpackage"


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 yolo26n CoreML 模型")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="yolo26n PyTorch 源模型路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CoreML mlpackage 输出路径")
    parser.add_argument("--imgsz", type=int, default=640, help="导出模型输入尺寸")
    parser.add_argument("--no-nms", action="store_true", help="导出时不内置 NMS")
    args = parser.parse_args()

    export_coreml_model(args.source, args.output, args.imgsz, nms=not args.no_nms)
    print(f"已导出 CoreML 模型：{args.output}")
    return 0


def export_coreml_model(source: Path, output: Path, image_size: int = 640, nms: bool = True) -> Path:
    source = source.resolve()
    output = output.resolve()
    if not source.exists():
        raise SystemExit(f"缺少 yolo26n 源模型，无法导出 CoreML：{source}")
    if image_size <= 0:
        raise SystemExit("导出模型输入尺寸必须大于 0。")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("缺少 ultralytics 依赖，无法导出 CoreML 模型。") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        exported_path = Path(YOLO(str(source), task="detect").export(format="coreml", imgsz=image_size, nms=nms)).resolve()
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
