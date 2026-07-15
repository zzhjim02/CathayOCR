# PaddleOCR-ncnn-CPP (UMI-OCR Plugin)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

基于 [Avafly/PaddleOCR-ncnn-CPP](https://github.com/Avafly/PaddleOCR-ncnn-CPP) 修改的 UMI-OCR 接口插件，提供高性能的 PP-OCR 文本检测与识别能力。

## 📋 项目概述

本项目是专为 [UMI-OCR](https://github.com/hiroi-sora/Umi-OCR) 软件设计的 OCR 识别插件，基于 ncnn 推理引擎实现 PP-OCRv3/v4/v5/v6 系列模型的高效推理。

### 核心特性

- ✅ 支持 PP-OCRv3/v4/v5 Mobile 系列模型
- ✅ 支持 PP-OCRv6 Small/Medium 模型（默认 Small）
- ✅ CPU/GPU (Vulkan) 双后端支持
- ✅ 通过 PIPE 模式与 UMI-OCR 高效通信
- ✅ 多线程并行处理，性能优异
- ✅ 跨平台支持 (Windows/Linux/macOS)

## 🛠️ 技术栈

| 组件 | 版本 | 说明 |
|------|------|------|
| **ncnn** | 1.0.20241226 | 高性能神经网络推理框架 |
| **OpenCV** | 4.11.0 | 图像处理库 |
| **pybind11** | 3.0.4 | Python 绑定生成器 |
| **C++ Standard** | C++17 | 编译标准 |
| **CMake** | 3.20+ | 构建系统 |
| **OpenMP** | 2.0+ | 并行计算支持 |

## 📦 编译依赖

> **注意**: 以下依赖仅用于编译构建，不包含在最终插件包中。

### 核心依赖库

- **ncnn 1.0.20241226**
  - 路径: `libs/ncnn/` (包含 x64/arm64/x86 多架构)
  - 特性: NCNN_VULKAN=1, NCNN_AVX2=1, NCNN_INT8=1

- **OpenCV 4.11.0** (Windows VS2022 x64)
  - 路径: `libs/opencv-4.11.0-windows-vs2022-x64-md/`
  - 运行时库: MultiThreadedDLL (/MD)

- **pybind11 3.0.4**
  - 路径: `src/3rdparty/pybind11-3.0.4/`
  - Python 要求: >=3.8

### 可选依赖

- **Vulkan SDK 1.4.341.1** (GPU 加速必需)
  - 默认路径: `C:\VulkanSDK\1.4.341.1`
  - 用于启用 `ENABLE_VULKAN` 编译选项

### 第三方库（内置）

- JSON 解析: nlohmann/json
- 日志系统: plog
- 几何算法: Clipper2

## 💻 编译环境要求

- **操作系统**: Windows 10/11 (推荐), Linux, macOS
- **编译器**: Visual Studio 2022 (MSVC), GCC, Clang
- **CMake**: 3.20 或更高版本
- **Python**: 3.8 或更高版本 (用于 Python 绑定)

## 🔨 构建说明

### CPU 版本构建

```bash
# Windows PowerShell
.\build_cpu.cmd

# 输出文件:
# - build/Release/ppocr_ocr_cpu.exe
# - build/Release/ocr_engine.pyd (可选)
```

### Vulkan GPU 版本构建

```bash
# 需要先安装 Vulkan SDK
.\build_vulkan_and_deploy.cmd

# 输出文件:
# - build_vulkan/Release/ppocr_ocr_vulkan.exe
```

### CMake 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-DENABLE_VULKAN` | 启用/禁用 Vulkan 支持 | OFF |
| `-DOpenCV_DIR` | OpenCV 库路径 | - |
| `-Dncnn_DIR` | ncnn 库路径 | - |
| `-DCMAKE_BUILD_TYPE` | 构建类型 | Release |

**重要**: `libs/` 目录下的依赖库仅用于编译链接，不会打包到插件中。

## 📁 UMI-OCR 插件安装

### 插件目录位置

将插件文件夹复制到 UMI-OCR 的 plugins 目录下：

```
UMI-OCR安装目录/plugins/PPOCR-ncnn-CPU/    # CPU 版本
UMI-OCR安装目录/plugins/PPOCR-ncnn-Vulkan/ # GPU 版本
```

### 插件文件清单

```
PPOCR-ncnn-CPU/
├── ppocr_ocr.py                   # UMI-OCR 插件入口脚本 ⭐
├── ppocr_ocr_config.py            # 配置界面定义
├── __init__.py                    # Python 包初始化
├── plugin_i18n.py                 # 国际化支持
├── i18n.csv                       # 翻译文件
├── ppocr_ocr_cpu.exe              # OCR 引擎主程序 ⭐
├── ocr_engine.pyd                 # Python 扩展模块（可选）
├── config.json                    # 运行时配置文件（自动生成）
├── config_safe.json               # 安全配置模板
├── models/                        # 模型文件目录 ⭐
│   ├── PP_OCRv3_mobile_det.*      # v3 检测模型
│   ├── PP_OCRv3_mobile_rec.*      # v3 识别模型
│   ├── PP_OCRv4_mobile_det.*      # v4 检测模型
│   ├── PP_OCRv4_mobile_rec.*      # v4 识别模型
│   ├── PP_OCRv5_mobile_det.*      # v5 检测模型
│   ├── PP_OCRv5_mobile_rec.*      # v5 识别模型
│   ├── PP_OCRv6_small_det.*       # v6 检测模型（默认）
│   ├── PP_OCRv6_small_rec.*       # v6 识别模型（默认）
│   ├── PP_OCRv6_medium_det.*      # v6 检测模型（高精度）
│   ├── PP_OCRv6_medium_rec.*      # v6 识别模型（高精度）
│   ├── PP_LCNet_x0_25_textline_ori.*  # 轻量角度模型
│   ├── PP_LCNet_x1_0_textline_ori.*   # 标准角度模型
│   ├── ppocr_keys_v1.txt          # v1-v4 字符集
│   ├── ppocr_keys_v5.txt          # v5 字符集
│   └── ppocr_keys_v6.txt          # v6 字符集
├── msvcp140.dll                   # VC++ 运行时库
├── msvcp140_1.dll
├── msvcp140_2.dll
├── vcruntime140.dll
├── vcruntime140_1.dll
├── copy_dlls.bat                  # DLL 复制工具
├── diagnose.bat                   # 诊断工具
├── fix_config.bat                 # 配置修复工具
├── QUICK_FIX.md                   # 快速修复指南
├── TROUBLESHOOTING.md             # 故障排除文档
└── README.md                      # 插件说明文档
```

> ⭐ 标记的为插件核心文件，必须存在

## ⚠️ 重要限制

### Server 模型不支持

**当前插件版本仅支持 Mobile 系列模型，不支持 Server 系列模型！**

#### ✅ 支持的模型
- PP_OCRv3_mobile_det/rec
- PP_OCRv4_mobile_det/rec
- PP_OCRv5_mobile_det/rec
- PP_OCRv6_small_det/rec
- PP_OCRv6_medium_det/rec
- PP_LCNet_x0_25/x1_0_textline_ori (角度检测)

#### ❌ 不支持的模型
- PP_OCRv5_server_det/rec
- 其他 Server 系列模型

#### 不支持原因

1. **技术层面**: Server 模型尚未完全跑通，存在兼容性问题
2. **架构层面**: UMI-OCR 采用线程调用模式，Server 模型体积大、内存占用高（1.5GB+）、推理时间长（3.9秒+），不适合多线程并发调用场景
3. **性能层面**: 在插件化部署中，Server 模型的资源消耗会影响 UMI-OCR 整体响应速度和其他插件的运行

## 📊 性能参考

> **数据来源**: [Avafly/PaddleOCR-ncnn-CPP](https://github.com/Avafly/PaddleOCR-ncnn-CPP) 原项目 README

**测试环境**: Intel Xeon Platinum 2.50GHz × 2 (VPS)  
**测试图像**: `ocr_img1.png` (简单场景), `ocr_img3.png` (复杂场景)

| 模型 | 简单图像延迟/内存 | 复杂图像延迟/内存 |
|------|------------------|------------------|
| PP-OCRv3 | 88ms / 106MB | 926ms / 228MB |
| PP-OCRv4 | 90ms / 98MB | 1005ms / 212MB |
| PP-OCRv5 | 92ms / 106MB | 1063ms / 292MB |

### PP-OCRv6 性能参考

> **数据来源**: [Avafly/PaddleOCR-ncnn-CPP](https://github.com/Avafly/PaddleOCR-ncnn-CPP) 原项目 README（单图测试）
>
> 测试环境：Intel Xeon Platinum 2.50GHz × 2 (VPS)

| 模型 | CPU 延迟 | GPU (Vulkan) 延迟 |
|------|---------|------------------|
| PP-OCRv6 Tiny | 14.4ms | 5.8ms |
| PP-OCRv6 Small | 21.7ms | 8.5ms |
| PP-OCRv6 Medium | 41.1ms | 14.4ms |

> **注意**: 实际性能会因硬件配置、图像复杂度等因素而有所不同。Tiny 模型本插件包未包含。

## 🔧 使用方式

### 运行模式

UMI-OCR 插件通过子进程调用 `ppocr_ocr_cpu.exe`，使用 **PIPE 模式**进行通信：

- 通过标准输入输出传递 JSON 数据
- 支持图像路径、base64 编码等多种输入格式
- 返回结构化 OCR 结果（文本、置信度、边界框等）

**底层引擎支持的模式**（供开发者参考）:
- **CLI 模式**: 命令行直接处理单张图片
- **PIPE 模式**: 通过管道传递 JSON 数据（UMI-OCR 使用）
- **TCP 模式**: 启动 TCP 服务器监听端口（默认 18043）

### 配置参数

通过 UMI-OCR 界面可调整以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `det_thres` | 检测阈值 | 0.45 |
| `unclip_ratio` | 文本框扩展比例 | 1.4 |
| `enable_cls` | 启用角度分类 | True |
| `max_side_len` | 最大边长 | 768 |
| `model_version` | 模型版本选择 | PP_OCRv6_small |
| `num_threads` | 线程数 | 自动检测（最多 4 线程） |

## 👨‍💻 开发说明

### 插件与开发目录分离

**重要**: UMI-OCR 插件有独立的安装目录，与开发目录分离。

- **开发目录**: `d:\Code\PaddleOCR-ncnn-CPP\` (本项目)
- **插件目录**: `UMI-OCR安装目录/plugins/PPOCR-ncnn-CPU/`

修改开发目录下的文件后，需要手动复制到插件目录才能生效。

### 调试建议

优先修改 Python 层代码（`ppocr_ocr.py`），便于快速迭代和调试。

## 📝 许可证

遵循原项目许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **原始项目**: [Avafly/PaddleOCR-ncnn-CPP](https://github.com/Avafly/PaddleOCR-ncnn-CPP)
- **PaddleOCR**: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- **ncnn**: [Tencent/ncnn](https://github.com/Tencent/ncnn)
- **UMI-OCR**: [hiroi-sora/Umi-OCR](https://github.com/hiroi-sora/Umi-OCR)

### 参考实现

- [MhLiao/DB](https://github.com/MhLiao/DB) - DB 文本检测算法
- [nihui/ncnn-android-ppocrv5](https://github.com/nihui/ncnn-android-ppocrv5)
- [FeiGeChuanShu/ncnn_paddleocr](https://github.com/FeiGeChuanShu/ncnn_paddleocr)
