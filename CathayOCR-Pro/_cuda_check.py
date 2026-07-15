"""CUDA 环境检测（启动时使用，只提示不阻止）"""
import sys
import os

def check_cuda():
    """检测 CUDA 是否可用，返回状态和提示信息"""
    try:
        import sysconfig
        site_dir = sysconfig.get_paths()["purelib"]
        ort_dir = os.path.join(site_dir, "onnxruntime", "capi")
        if os.path.isdir(ort_dir):
            os.add_dll_directory(ort_dir)
    except Exception:
        pass

    # ── 先检查 NVIDIA 驱动是否存在 ──
    # 如果 nvcuda.dll 不存在，导入 onnxruntime 会因 DLL 初始化失败而崩溃
    nvidia_driver_exists = False
    try:
        import ctypes
        lib = ctypes.WinDLL("nvcuda.dll")
        del lib
        nvidia_driver_exists = True
    except Exception:
        pass

    if not nvidia_driver_exists:
        print(
            "ℹ 未检测到 NVIDIA 显卡驱动\n"
            "  本机没有 NVIDIA 显卡或驱动未安装。\n"
            "  正常使用不受影响 — 请选择「PP-OCR (ncnn Vulkan)」引擎，\n"
            "  ncnn 支持任意品牌显卡（NVIDIA/AMD/Intel）的 GPU 加速。\n"
            "  PP-OCRv6 ONNX CUDA 引擎将自动回退 CPU 模式运行。"
        )
        return False

    # ── NVIDIA 驱动存在，检测 onnxruntime 能否加载 CUDA ──
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            print("CUDA 可用 — PP-OCRv6 ONNX CUDA 引擎可正常使用 NVIDIA GPU 加速")
            return True
        else:
            print(
                "⚠ CUDA 驱动存在但 onnxruntime 无法加载 CUDA 加速\n"
                "  已检测到 NVIDIA 驱动，但 onnxruntime CUDA 组件加载失败。\n"
                "  正常使用不受影响 — 引擎自动回退 CPU。\n"
                "  若要使用 CUDA 加速，请更新 NVIDIA 显卡驱动到最新版本。"
            )
            return False
    except ImportError:
        print(
            "⚠ onnxruntime CUDA 组件加载失败\n"
            "  Python 环境异常，CUDA DLL 未正确配置。\n"
            "  请改用 ncnn Vulkan 引擎（同样支持 GPU 加速），或重新解压便携包。"
        )
        return False

if __name__ == "__main__":
    available = check_cuda()
    sys.exit(0 if available else 1)
