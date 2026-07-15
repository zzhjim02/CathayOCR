# 修复日志：CUDA 全语言空白问题

**日期：** 2026-07-15
**涉及版本：** PRO v1.0.0-pro, DEV v1.0.0-dev

---

## 根因 1：onnxruntime CUDA 提供者加载失败（静默回退 CPU）

`cufft64_11.dll`（cuFFT 库）缺失于 `portapython/Lib/site-packages/onnxruntime/capi/` 目录。

onnxruntime 加载 CUDA Execution Provider 时需要 `cufft64_11.dll` 作为延迟加载依赖。
因该 DLL 不在搜索路径内，provider 初始化失败，**静默回退到 CPU** 执行所有推理。

### 修复操作

1. 从 `portapython/Lib/site-packages/nvidia/cufft/bin/` 复制 `cufft64_11.dll` 至
   `portapython/Lib/site-packages/onnxruntime/capi/`
2. 同时复制 `cufftw64_11.dll`（可选但推荐）
3. 在 `ppocr_v6_server.py` 的 `_setup_nvidia_dlls()` 函数中补上
   `os.add_dll_directory(ort_dir)`，确保 onnxruntime capi 目录在 DLL 搜索路径中

**受影响范围：** PRO/DEV 所有语言（因 CUDA 全局回退 CPU）

---

## 根因 2：model_dir 导致 PaddleX 读取 Hpi 配置 → 引擎错误

`ppocr_v6_server.py` 的 V6 和 V5 分支在检测到本地 ONNX 缓存存在时，会向
`PaddleOCR()` 传入 `model_dir`。PaddleX 从 model_dir 读取 `inference.yml`，
其中包含 `Hpi` 配置段 → `use_hpip=True` → `_resolve_child_engine` 返回
`(None, True)` → 引擎变成 `None` → `_resolve_effective_engine` 返回 `"hpi"` →
HPI 后端找不到 `.pdmodel` 文件（缓存只有 `.onnx`）→ 初始化失败。

### 修复操作

- **V6 分支**：改为仅传 `model_name`，不传 `model_dir`（与 V5 分支一致）
- **V5 分支**：已验证使用 `model_name` 方式完好，无需重复修改
- PaddleX 通过 `PADDLE_PDX_CACHE_HOME` 自动从 `models/official_models/` 加载 ONNX 模型

**受影响范围：** V6 中文/英文/日文/韩文/俄文等（之前因 `use_local=False` 侥幸跳过）

---

## 根因 3（已在此次之前修复）：简易模式 V5 语系引擎映射错误

`_apply_simple_settings()` 将 V5 语系（阿拉伯文/天城文/泰文等）错误映射到
`win7_v5`（纯中文 PP-OCRv5 Paddle CPU 引擎），该引擎不包含多语言字典。
已改为映射到 `umi_plugin_v6`（PP-OCRv6 ONNX CUDA/CPU），其服务器内部路由到
PP-OCRv5 ONNX 多语言模型。

**受影响语言（12 种）：**
天城文：印地文(hi)、马拉地文(mr)、尼泊尔文(ne)、梵文(sa)
阿拉伯文：阿拉伯文(ar)、波斯文(fa)、维吾尔文(ug)、乌尔都文(ur)
泰文(th)、希腊文(el)、泰卢固文(te)、泰米尔文(ta)
多语言(v5)

---

## 修改文件清单

| 文件 | 修改内容 | 说明 |
|------|---------|------|
| `ppocr_v6/ppocr_v6_server.py` | ① `_setup_nvidia_dlls()` 补 `os.add_dll_directory(ort_dir)` | CUDA DLL 加载修复 |
| | ② V6 分支去除 model_dir，统一 model_name | 避免 Hpi 配置干扰引擎 |
| | ③ 注释中加入 GPU/CPU 状态日志 | 调试辅助 |
| `umi_ocr_pdf_processor_ui.py` | `_apply_simple_settings()` 中 V5 语系引擎映射 | 简易模式天城文空白修复 |
| `portapython/Lib/site-packages/onnxruntime/capi/` | 添加 `cufft64_11.dll` + `cufftw64_11.dll` | CUDA provider 加载必需 |

---

## 验证结果

- CUDA in providers: ✅ True
- CUDA Session 创建: ✅ OK
- PaddleOCR init: 1.3s
- 尼泊尔文识别: ✅ `[0.972] नेपाली / Nepali`
- VRAM 占用: ~3929 MiB (5% GPU)
