# 监控视频智能分析系统

这是一个面向 macOS Apple Silicon，尤其是 MacBook M1 的监控视频人形事件分析工具。程序使用 PyQt6 提供桌面界面，使用 OpenCV 读取视频和做运动检测，并固定加载 CoreML `mlpackage` 模型，通过 `CPU_AND_NE` 让 CoreML 优先使用 Apple Neural Engine。

## 功能特性

- 支持选择视频目录并批量扫描常见视频格式：`.mp4`、`.avi`、`.mkv`、`.mov`、`.wmv`、`.flv`。
- 使用流式读取处理长视频，避免一次性把视频载入内存。
- 默认先用 FFmpeg 无重编码重封装视频，重建索引并纠正时间戳。
- 使用降采样 OpenCV 运动检测作为前置过滤，减少空场景 YOLO 调用。
- 固定使用 yolo26n CoreML 模型识别人形。
- 检测到人形后自动保存带红框截图，并生成 CSV 检测报告。
- 支持调整抽帧频率和 AI 置信度阈值。

## 运行环境

- macOS Apple Silicon arm64。
- 推荐 MacBook Air M1 16G 或更高配置。
- Python 3.12。
- 运行时模型固定为 `models/yolo26n-512-fp16-nms.mlpackage`。

本项目当前不再支持 Windows、Linux、ONNX Runtime 或 Ultralytics `.pt` 运行时推理。

## 目录结构

```text
monitor-scan/
├── models/
│   └── yolo26n-512-fp16-nms.mlpackage/
├── scripts/
│   ├── build_package.py
│   ├── benchmark_detector.py
│   ├── benchmark_video.py
│   └── export_coreml_model.py
├── src/monitor_scan/
│   ├── ai/
│   ├── gui/
│   ├── results/
│   ├── video/
│   └── workers/
├── tests/
└── pyproject.toml
```

## 模型文件

运行和打包前必须存在以下 CoreML 模型目录：

```text
models/yolo26n-512-fp16-nms.mlpackage
```

程序不会在运行时自动下载模型，也不会从 `.pt` 自动导出模型。若缺少该目录，GUI 会提示模型文件缺失，打包脚本会直接终止。

## 本地开发环境

安装依赖：

```bash
python -m pip install -e ".[dev]"
```

运行测试：

```bash
python -m pytest
```

启动桌面程序：

```bash
python -m monitor_scan
```

如果使用仓库内虚拟环境，也可以运行：

```bash
./.venv/bin/python -m monitor_scan
```

## 使用流程

1. 确认 `models/yolo26n-512-fp16-nms.mlpackage` 存在。
2. 启动程序。
3. 点击“选择文件夹”，选择包含监控视频的目录。
4. 根据需要调整“抽帧频率”和“AI 灵敏度”。
5. 点击“开始分析”。
6. 在界面中查看视频处理状态、总体进度、实时日志和检测结果。
7. 如需中断任务，点击“停止”。

## 输出结果

程序会在所选视频目录下生成：

```text
output_results/
├── 检测报告_YYYYMMDD.csv
└── snapshots/
    └── 原视频文件名_时-分-秒.jpg
```

CSV 字段包括：

- 视频文件名
- 事件发生时间
- AI 置信度
- 截图文件路径

截图会用红框标出检测到的人形目标。

## 视频索引自动修复说明

分析视频前，程序默认会先在后台调用 FFmpeg 做一次无重编码重封装：

```text
ffmpeg -c:v copy
```

该步骤不会重新压缩视频，只会重建索引、生成时间戳并丢弃明显损坏的数据包。OpenCV 随后读取生成的临时文件，报告和截图仍使用原视频文件名；分析结束后临时目录会自动删除。

如果系统没有安装 FFmpeg，程序会优先使用 `imageio-ffmpeg` 提供的内置 FFmpeg。若 FFmpeg 不可用、重封装失败或超时，程序会自动回退为直接分析原视频。

部分严重损坏的 H.264 视频仍可能在控制台出现类似以下警告：

```text
error while decoding MB ...
missing picture in access unit
no frame!
```

这些警告表示视频局部码流损坏。自动重封装可以减少卡帧、提前结束和时间戳错乱导致的漏检，但无法恢复原始文件中已经缺失或无法解码的画面数据。

## 性能基准

检测器基准：

```bash
python scripts/benchmark_detector.py --iterations 5 --warmup 1
```

视频端到端基准：

```bash
python scripts/benchmark_video.py --video /path/to/sample.mp4
```

输出会包含模型路径、CoreML backend、`CPU_AND_NE`、推理耗时、YOLO 调用次数和进程峰值内存。

## 本地打包

安装构建依赖：

```bash
python -m pip install -e ".[build]"
```

在 macOS Apple Silicon 环境执行：

```bash
python scripts/build_package.py --target macos-arm64
```

生成的运行包会位于：

```text
release/
```

macOS 运行包内包含 `启动.command`，用于移除解压后可能携带的隔离属性并启动应用。

## 手动发布

GitHub Actions workflow 位于：

```text
.github/workflows/release.yml
```

该 workflow 只能手动触发，会在 macOS Apple Silicon runner 上执行测试、打包并上传 `monitor-scan-macos-arm64.tar.gz` 到日期版本 Release。

## 开发者重新导出 CoreML 模型

导出脚本作为开发工具保留，不属于运行时依赖。需要重新从外部 `.pt` 源模型导出时，先安装导出依赖：

```bash
python -m pip install -e ".[export]"
```

然后执行：

```bash
python scripts/export_coreml_model.py --source /path/to/yolo26n.pt --output models/yolo26n-512-fp16-nms.mlpackage --imgsz 512 --half --nms
```

导出的模型需要保持 `512×512` 图像输入，并输出 `1×300×6` 的 `x1, y1, x2, y2, confidence, class_id` 结果格式。

## 注意事项

- CoreML 没有 `NE_ONLY` 选项，当前固定使用 `CPU_AND_NE`，这是 Apple Neural Engine 优先且禁用 GPU 的可用运行方式。
- 视频解码、图像缩放、颜色转换、运动检测、截图和 CSV 写入仍会使用 CPU。
- 真实识别效果取决于模型质量、视频清晰度、抽帧频率和置信度阈值。
