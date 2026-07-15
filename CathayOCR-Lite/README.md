# CathayOCR Lite (极速版) v1.0

基于 ncnn Vulkan 的便携式 PDF OCR 工具。CathayOCR 专业版的轻量分支。

## 特性亮点

🚀 **单一引擎，一键启动**
- 仅保留 ncnn Vulkan 引擎，无需选择纠结
- GPU 模式（Vulkan 加速）/ CPU 模式（原生 ncnn）
- 支持任意品牌显卡：NVIDIA / AMD / Intel

📦 **极致便携，U 盘即用**
- 体积仅 ~1.4GB，自带便携 Python
- 拷贝到任意目录即可运行

🌍 **多语言支持**
- 中文（繁简体自动识别）、English、Français、Deutsch、日本語、한국어
- Español、Italiano、Português、Pусский、Tiếng Việt 等

📄 **PDF OCR 专业功能**
- GPU 双实例并行（提升 30~50%）
- 古籍/竖排/复杂排版支持
- 拖放文件处理，批量输出

## 系统要求

- **操作系统**：Windows 7/10/11 64位
- **GPU 模式**：支持 Vulkan 1.1+ 的显卡
- **CPU 模式**：任意 x64 处理器

## 使用方法

1. **解压**到任意文件夹
2. 双击 **`启动.bat`**
3. 🎯 简单模式：选择语言 → 点运行
4. 🔧 专业模式：更多参数可调

## 文件结构

```
portapython\          # 便携 Python 环境
ncnn\PPOCR-ncnn-Vulkan\  # ncnn 引擎 + 模型库
CathayOCR-Lite\        # 主程序
├── umi_ocr_pdf_processor_ui.py  # 主程序
└── 开启 CathayOCR Lite.bat
启动.bat              # 🏁 双击启动
_cuda_check.py        # GPU 检测
软件介绍.md           # 说明文档
```

## 故障排除

| 问题 | 解决 |
|------|------|
| GPU 模式崩溃 | 更新显卡驱动 |
| AMD 核显不稳定 | 切换到 FP32 精度 |
| 无法启动 | 安全软件加入信任区 |

