from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_coreml_model.py"
SPEC = importlib.util.spec_from_file_location("export_coreml_model", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
export_coreml_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_coreml_model)


def test_export_coreml_model_rejects_invalid_image_size(tmp_path):
    source = tmp_path / "source.pt"
    source.write_text("model", encoding="utf-8")

    with pytest.raises(SystemExit, match="导出模型输入尺寸必须大于 0"):
        export_coreml_model.export_coreml_model(source, tmp_path / "out.mlpackage", image_size=0)


def test_export_coreml_model_rejects_half_and_int8_together(tmp_path):
    source = tmp_path / "source.pt"
    source.write_text("model", encoding="utf-8")

    with pytest.raises(SystemExit, match="FP16 和 INT8 不能同时启用"):
        export_coreml_model.export_coreml_model(source, tmp_path / "out.mlpackage", half=True, int8=True)
