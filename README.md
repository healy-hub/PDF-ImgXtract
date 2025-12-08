# PDF-ImgXtract
# PDF 图表提取工具

本工具是一个基于深度学习的PDF图表提取工具，旨在帮助用户从PDF文档中方便、快捷地提取图片、表格等元素。

项目利用了先进的文档布局分析模型 **DocLayout-YOLO**，能够智能地识别文档中的不同类型的元素，而不仅仅是提取嵌入的原始图像内容。

## ✨ 功能特性

- **智能识别**: 基于`DocLayout-YOLO`模型，能够准确识别PDF中的图片（Figures）和表格（Tables）。
- **简单易用的图形界面**: 提供了基于PySide6的图形用户界面，用户只需选择PDF文件和输出目录即可开始提取。
- **高质量输出**: 提取的图片和表格将以`.png`格式保存到指定目录。
- **跨平台**: 基于Python和Pyside6，可在Windows、macOS和Linux上运行。

## 🚀 安装指南

1. **克隆仓库**
   ```bash
   git clone https://github.com/healy-hub/PDF-ImgXtract.git
   cd PDF2PNG
   ```

2. **安装依赖**
   项目依赖项已在 `requirements.txt` 中列出。通过以下命令安装：
   ```bash
   pip install -r requirements.txt
   ```

3. **下载模型**
   本项目使用来自Hugging Face的 `DocLayout-YOLO` ONNX模型。
   - **模型地址**: [wybxc/DocLayout-YOLO-DocStructBench-onnx](https://huggingface.co/wybxc/DocLayout-YOLO-DocStructBench-onnx)
   
   请从上述地址下载`doclayout_yolo_docstructbench_imgsz1024.onnx`模型文件，并将其放置在项目根目录下的 `models` 文件夹中。

## 📖 使用方法

确保模型文件已正确放置，然后运行 `run_gui.py` 启动图形界面：

```bash
python run_gui.py
```

**操作流程**:
1. 点击“选择PDF文件”按钮，选择您要处理的PDF文档。
2. 点击“选择输出文件夹”按钮，指定提取出的图片和表格的保存位置。
3. 点击“开始提取”按钮，程序将开始处理PDF并保存结果。
4. 提取完成后，状态栏会显示“完成”信息。

## 🙏 致谢

本项目的开发受到了以下优秀项目的启发和帮助，在此表示衷心感谢：

- **[opendatalab/DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)**: 为本项目提供了核心的文档布局分析能力。
- **[opendatalab/MinerU](https://github.com/opendatalab/MinerU)**: 提供了宝贵的参考和思路。
- **[wybxc/DocLayout-YOLO-DocStructBench-onnx](https://huggingface.co/wybxc/DocLayout-YOLO-DocStructBench-onnx)**: 提供了在Hugging Face上预训练好的ONNX模型，极大地方便了项目的实现。

## 📜 许可证

本项目依赖于 **PyMuPDF** 库，该库使用 **AGPL-3.0** 许可证。因此，为了遵守其开源协议，本项目同样采用 **GNU Affero General Public License v3.0 (AGPL-3.0)**。

有关许可证的详细信息，请参阅 `LICENSE` 文件。

