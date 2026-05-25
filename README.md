# 监控视频智能分析系统

这是一个跨平台桌面工具，用于批量分析监控视频中的人形事件。程序会先用 OpenCV 做画面运动检测，只有画面发生明显变化时才调用 YOLOv8n ONNX 模型进行人形识别，从而减少长视频分析时的计算量。

## 功能特性

- 支持选择视频目录并批量扫描常见视频格式。
- 支持 `.mp4`、`.avi`、`.mkv`、`.mov`、`.wmv`、`.flv`。
- 使用流式读取处理长视频，避免一次性把视频载入内存。
- 默认先用 FFmpeg 无重编码重封装视频，重建索引并纠正时间戳。
- 支持抽帧频率和 AI 置信度阈值配置。
- 使用 OpenCV 运动检测作为前置过滤。
- 使用 ONNX Runtime 运行 YOLOv8n 模型识别人形。
- 检测到人形后自动保存带红框截图。
- 自动生成 CSV 检测报告。
- 使用 PyQt6 图形界面，后台线程分析视频，避免界面卡死。

## 目录结构

```text
monitor-scan/
├── models/
│   └── yolov8n.onnx
├── src/monitor_scan/
│   ├── ai/
│   ├── gui/
│   ├── results/
│   ├── video/
│   └── workers/
├── tests/
├── scripts/build_package.py
├── .github/workflows/release.yml
└── pyproject.toml
```

## 模型文件

程序默认读取以下模型文件：

```text
models/yolov8n.onnx
```

本项目不会在运行时自动下载模型。运行或打包前，请确认该文件已经存在。

如果是通过 GitHub Actions 打包发布，也需要确保 `models/yolov8n.onnx` 已经提交到仓库，或通过 Git LFS 管理并能在 workflow 中正常拉取。

## 本地开发环境

建议使用 Python 3.12。

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

1. 准备模型文件：确认 `models/yolov8n.onnx` 存在。
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

该步骤不会重新压缩视频，只会重建索引、生成时间戳并丢弃明显损坏的数据包。OpenCV 随后读取生成的隐藏临时文件，报告和截图仍使用原视频文件名；分析结束后临时目录会自动删除。

如果系统没有安装 FFmpeg，程序会优先使用 `imageio-ffmpeg` 提供的内置 FFmpeg。若 FFmpeg 不可用、重封装失败或超时，程序会自动回退为直接分析原视频。

部分严重损坏的 H.264 视频仍可能在控制台出现类似以下警告：

```text
error while decoding MB ...
missing picture in access unit
no frame!
```

这些警告表示视频局部码流损坏。自动重封装可以减少卡帧、提前结束和时间戳错乱导致的漏检，但无法恢复原始文件中已经缺失或无法解码的画面数据。

## 手动打包发布

项目提供 GitHub Actions workflow：

```text
.github/workflows/release.yml
```

该 workflow 只能手动触发。触发后会：

1. 在 Windows x86_64、Linux x86_64、macOS Apple Silicon 三个目标分别安装依赖。
2. 执行本地测试。
3. 使用 PyInstaller 打包运行包。
4. 创建当前日期版本号的 GitHub Release，例如 `v20260525`。
5. 上传三个平台的运行包到 Release。

打包产物命名示例：

```text
monitor-scan-windows-x86_64.zip
monitor-scan-linux-x86_64.tar.gz
monitor-scan-macos-arm64.tar.gz
```

同一天重复触发 workflow 时，会更新同名 Release，并覆盖同名运行包。

## 本地打包

如需本地打包，可以安装构建依赖：

```bash
python -m pip install -e ".[build]"
```

然后执行：

```bash
python scripts/build_package.py --target local
```

生成的运行包会位于：

```text
release/
```

## 注意事项

- Windows 和 Linux 的 workflow 构建目标为 x86_64，不是 32 位 x86。
- macOS workflow 使用 `macos-14` runner，目标为 Apple Silicon arm64。
- macOS 运行包内包含 `启动.command`，用于移除解压后可能携带的隔离属性并启动应用。
- 打包前必须存在 `models/yolov8n.onnx`。
- 真实识别效果取决于模型质量、视频清晰度、抽帧频率和置信度阈值。
