# CathayOCR Dev — 源码目录

> ⚠️ 此处仅为 Dev 版**部分源码**（Python GUI + C++ OCR 引擎）。
> 
> **完整可用的 Dev 版安装包**（Pro 全部内容 + C++ 源码 + VS 工程 + ncnn 源码 + 开发环境）请到主仓库下载：
> 
> → [CathayOCR 主 README](https://github.com/zzhjim02/CathayOCR#-安装包下载)

### 包含的文件
```
CathayOCR-Dev/
├── CathayOCR-Dev-UI/       # Python GUI 主程序（PyQt5）
├── ncnn/src/               # C++ OCR 引擎完整源码
│   ├── ocr_engine.cpp/h    # OCR 引擎核心
│   ├── db_net.cpp/h        # 文本检测网络
│   ├── crnn_net.cpp/h      # 文字识别网络
│   ├── angle_net.cpp/h     # 角度分类网络
│   ├── main.cpp            # TCP 服务主程序
│   ├── pybind_ocr.cpp      # Python 绑定
│   └── utils.cpp/h         # 工具函数
├── ncnn/CMakeLists.txt     # CMake 构建配置
├── ncnn/build_*.cmd        # 构建脚本
├── ppocr_v6/               # ONNX CUDA 引擎 Python 脚本
├── tools/                  # 开发工具脚本
└── _test_pdfs/             # 8 种语言测试 PDF
```

### 可修改性
- **Python 源码**：用记事本/VS Code 就能改
- **C++ 引擎**：需要 Visual Studio 2022 + CMake 编译
- 完整 Dev 安装包里还包含 VS2022 工程文件、ncnn 源码、Vulkan SDK 等

### 完整功能文档
关于 Dev 版的全部特性（构建指南、开发环境等），请参阅主仓库的 [README.md](https://github.com/zzhjim02/CathayOCR)。
