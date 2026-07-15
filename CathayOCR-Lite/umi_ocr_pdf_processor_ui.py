"""
CathayOCR Lite (CathayOCR极速版) v1.0 - 纯 Vulkan 引擎PDF处理器
Architecture: 预渲染所有页面到RAM -> 单实例OCR流水线 -> 组装输出
核心思想: GPU永不等待,CPU预渲染消除I/O瓶颈
=======================================================
支持的OCR引擎:
  1. PP-OCR (ncnn Vulkan) - 唯一引擎 (原生GPU+CPU)
=======================================================
"""

import sys
import os
import time
import threading
import json
import atexit
import subprocess
import base64 as _b64
import platform
from pathlib import Path
from queue import Queue, Empty, Full
from abc import ABC, abstractmethod
import socket
import tempfile

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QProgressBar, QGroupBox, QMessageBox, QSpinBox, QComboBox, QCheckBox,
    QRadioButton, QButtonGroup, QListWidget, QListWidgetItem, QTreeView,
    QTextBrowser, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSettings
from PyQt5.QtGui import QFont

import fitz
# 压制 MuPDF 的 PDF 结构语法警告（不影响识别结果）
import os, sys
import contextlib
@contextlib.contextmanager
def _suppress_mupdf_warnings():
    """临时压制MuPDF stderr输出"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
# 全局抑制 fitz 日志级别
if hasattr(fitz, 'TOOLS') and hasattr(fitz.TOOLS, 'set_log_level'):
    try:
        fitz.TOOLS.set_log_level(40)
    except Exception:
        pass


def _setup_onnx_dll_paths():
    """设置 ONNX Runtime CUDA DLL 路径（UI 进程检测用）"""
    try:
        # 方法1: 通过 sysconfig 找 site-packages
        import sysconfig
        site_dir = sysconfig.get_paths()["purelib"]
        _add_dll_dir(site_dir)
    except Exception:
        pass
    try:
        # 方法2: 直接找当前 Python 的 site-packages
        import site
        for site_dir in site.getsitepackages():
            _add_dll_dir(site_dir)
    except Exception:
        pass

def _add_dll_dir(site_dir):
    """将 site-packages 下的 nvidia DLL 目录加入搜索路径"""
    try:
        nvidia_base = os.path.join(site_dir, "nvidia")
        if os.path.isdir(nvidia_base):
            for sub in os.listdir(nvidia_base):
                dll_dir = os.path.join(nvidia_base, sub, "bin")
                if os.path.isdir(dll_dir):
                    try:
                        os.add_dll_directory(dll_dir)
                    except Exception:
                        pass
                    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        ort_dir = os.path.join(site_dir, "onnxruntime", "capi")
        if os.path.isdir(ort_dir):
            os.environ["PATH"] = ort_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


# ============================================================
# 引擎注册表 - 统一管理所有OCR引擎
# ============================================================

ENGINE_REGISTRY = {}

def register_engine(engine_id, display_name, description, plugin_rel_path,
                    entry_file, entry_type, supports_gpu, supports_cpu,
                    model_options, supported_params, priority=0):
    """注册一个OCR引擎"""
    ENGINE_REGISTRY[engine_id] = {
        'id': engine_id,
        'name': display_name,
        'desc': description,
        'plugin_rel': plugin_rel_path,
        'entry': entry_file,
        'entry_type': entry_type,
        'gpu': supports_gpu,
        'cpu': supports_cpu,
        'models': model_options,
        'params': supported_params,
        'priority': priority,
    }

register_engine(
    'ncnn_vulkan', 'PP-OCR (ncnn Vulkan)',
    '\u2b50 速度最快！\n支持NVIDIA/AMD/Intel任意显卡\nGPU模式+Vulkan加速\nCPU模式+ncnn原生\n支持多版本模型（v3~v6）\n支持中/英/法/德/日/意/西/葡/韩/俄/多语言',
    'paddle-ocr-ncnn-cpp_plugin-master/PPOCR-ncnn-Vulkan',
    'ppocr_ocr_vulkan.exe', 'ncnn_vulkan',
    supports_gpu=True, supports_cpu=True,
    model_options=[],
    supported_params=['num_threads', 'enable_fp16', 'det_thres',
                      'unclip_ratio', 'enable_cls', 'gpu_device'],
    priority=20
)





# ============================================================
# 路径自动查找 - 多引擎支持
# ============================================================

def _find_all_plugins():
    """
    扫描所有已注册引擎的插件目录。
    返回: {engine_id: {"plugin_dir": str, "entry_path": str}}
    引擎目录不存在或入口文件缺失时跳过。
    """
    script_dir = Path(__file__).parent.resolve()
    found = {}

    for eid, einfo in ENGINE_REGISTRY.items():
        rel = einfo["plugin_rel"]
        entry = einfo["entry"]

        found_path = None
        # 搜索路径1: 嵌套结构 UmiOCR-data/plugins/{rel}
        for parent in [script_dir] + list(script_dir.parents)[:6]:
            candidate = parent / "UmiOCR-data" / "plugins" / rel
            if candidate.exists() and candidate.is_dir():
                found_path = candidate
                break

        # 搜索路径2: 平铺结构 {short_rel}（短名称映射）
        if not found_path:
            _SHORT_REL_MAP = {
                "paddle-ocr-ncnn-cpp_plugin-master": "ncnn",
            }
            rel_parts = rel.split("/", 1)
            root_dir = rel_parts[0]
            sub_path = rel_parts[1] if len(rel_parts) > 1 else ""
            short_name = _SHORT_REL_MAP.get(root_dir, root_dir)
            for parent in [script_dir] + list(script_dir.parents)[:4]:
                if sub_path:
                    candidate = parent / short_name / sub_path
                else:
                    candidate = parent / short_name
                if candidate.exists() and candidate.is_dir():
                    found_path = candidate
                    break

        # 搜索路径3: 原始路径直接搜索（兼容旧的UmiOCR-data布局）
        if not found_path:
            for parent in [script_dir] + list(script_dir.parents)[:4]:
                candidate = parent / rel
                if candidate.exists() and candidate.is_dir():
                    found_path = candidate
                    break

        if not found_path:
            continue

        entry_path = found_path / entry
        if not entry_path.exists():
            print(f"[Plugin] {eid}: entry not found at {entry_path}")
            continue

        found[eid] = {
            "plugin_dir": str(found_path),
            "entry_path": str(entry_path),
        }
        print(f"[Plugin] Found {eid}: {found_path}")

    return found

_PLUGIN_DIRS = {}

def _init_plugin_dirs():
    global _PLUGIN_DIRS
    _PLUGIN_DIRS = _find_all_plugins()
    if not _PLUGIN_DIRS:
        msg = "没有找到任何可用的OCR引擎插件！请确认以下目录存在："
        for e in ENGINE_REGISTRY.values():
            rel = e["plugin_rel"]
            ent = e["entry"]
            msg += "\n  - UmiOCR-data/plugins/" + rel + "/" + ent
        raise FileNotFoundError(msg)
    print("[Plugin] 可用引擎: " + ", ".join(_PLUGIN_DIRS.keys()))

_init_plugin_dirs()

# ============================================================
# 动态模型检测
# ============================================================

def _scan_available_ncnn_models(plugin_dir):
    """
    扫描ncnn引擎的模型。返回所有有.param的模型名称列表。
    同时返回一个set指示哪些模型同时有.bin（完整可用）。
    """
    models_dir = os.path.join(plugin_dir, "models")
    if not os.path.exists(models_dir):
        return []
    param_det = set()  # 有.param的det模型
    param_rec = set()  # 有.param的rec模型
    bin_det = set()    # 有.bin的det模型
    bin_rec = set()    # 有.bin的rec模型
    for f in os.listdir(models_dir):
        if f.endswith(".param"):
            name = f[:-6]
            if name.endswith("_det"):
                param_det.add(name[:-4])
            elif name.endswith("_rec"):
                param_rec.add(name[:-4])
        elif f.endswith(".bin"):
            name = f[:-4]
            if name.endswith("_det"):
                bin_det.add(name[:-4])
            elif name.endswith("_rec"):
                bin_rec.add(name[:-4])
    # 按优先级排序: v6_server > v6_medium > v6_small > v6_tiny > v5_server > v5_mobile > v4 > v3
    _MODEL_PRIORITY = [
        "PP_OCRv6_server", "PP_OCRv6_medium", "PP_OCRv6_small", "PP_OCRv6_tiny",
        "PP_OCRv5_server", "PP_OCRv5_mobile", "PP_OCRv4_mobile", "PP_OCRv3_mobile",
    ]
    all_bases_list = list(param_det | param_rec)
    def _model_sort_key(name):
        try:
            return _MODEL_PRIORITY.index(name)
        except ValueError:
            return len(_MODEL_PRIORITY)
    all_bases = sorted(all_bases_list, key=_model_sort_key)
    # 每个模型: (base_name, has_det_bin, has_rec_bin)
    result = []
    for base in all_bases:
        if base in param_det and base in param_rec:
            has_bin = (base in bin_det and base in bin_rec)
            result.append((base, has_bin))
    return result


def _get_ncnn_model_options(engine_id):
    """获取ncnn引擎的模型选项列表 [(value, label, is_available)]"""
    info = _PLUGIN_DIRS.get(engine_id)
    if not info:
        return []
    models = _scan_available_ncnn_models(info["plugin_dir"])
    options = []
    for base_name, has_bin in models:
        label = get_model_display_name(base_name)
        if not has_bin:
            label += " [文件不完整]"
        options.append((base_name, label, has_bin))
    return options


def _get_first_valid_ncnn_model(engine_id):
    """获取ncnn引擎的第一个完整可用模型名"""
    options = _get_ncnn_model_options(engine_id)
    for value, label, has_bin in options:
        if has_bin:
            return value
    # 全都没有.bin，返回第一个
    if options:
        return options[0][0]
    return ""


def _get_available_ncnn_models(engine_id):
    """获取指定ncnn引擎的可用(有.bin)模型列表"""
    options = _get_ncnn_model_options(engine_id)
    return [v for v, l, b in options if b]


# 所有ncnn引擎注册通用的模型映射
_NCNN_MODEL_MAP = {
    "PP_OCRv6_server": "v6 Server (超高精度/最慢)",
    "PP_OCRv6_medium": "v6 Medium (高精度/推荐)",
    "PP_OCRv6_small": "v6 Small (轻量快速)",
    "PP_OCRv6_tiny": "v6 Tiny (极速/最低精度)",
    "PP_OCRv5_server": "v5 Server (高精度)",
    "PP_OCRv5_mobile": "v5 Mobile (轻量)",
    "PP_OCRv4_mobile": "v4 Mobile (旧版)",
    "PP_OCRv3_mobile": "v3 Mobile (经典)",
}

_NCNN_MODEL_DESC = {
    "PP_OCRv6_server": "v6 Server: 精度最高但速度最慢，适合对精度要求极高的文档",
    "PP_OCRv6_medium": "v6 Medium: 精度与速度的最佳平衡，推荐日常使用",
    "PP_OCRv6_small": "v6 Small: 轻量化模型，速度较快，精度尚可",
    "PP_OCRv6_tiny": "v6 Tiny: 极致轻量，速度最快但精度最低，适合快速预览",
    "PP_OCRv5_server": "v5 Server: 旧版高精度服务器模型，体积大",
    "PP_OCRv5_mobile": "v5 Mobile: 旧版轻量移动模型",
    "PP_OCRv4_mobile": "v4 Mobile: 更早期的旧版模型",
    "PP_OCRv3_mobile": "v3 Mobile: 经典模型，兼容性好",
}

def get_model_display_name(model_key):
    return _NCNN_MODEL_MAP.get(model_key, model_key)

# ============================================================
# GPU设备检测
# ============================================================

_GPU_DEVICES = None

def _detect_vulkan_gpus(force_redetect=False):
    """
    运行ncnn Vulkan exe检测可用的GPU设备。
    返回: [{"index": 0, "name": "AMD Radeon", "score": 21, "dedicated": True/False}, ...]
    缓存结果到全局变量避免重复检测。force_redetect=True时强制重新检测。
    """
    global _GPU_DEVICES
    if _GPU_DEVICES is not None and not force_redetect:
        return _GPU_DEVICES

    vk_info = _PLUGIN_DIRS.get("ncnn_vulkan")
    if not vk_info:
        _GPU_DEVICES = []
        return _GPU_DEVICES

    exe = vk_info["entry_path"]
    # 优先使用已生成的config.json（有use_vulkan:true），没有则用config_safe.json
    config = os.path.join(vk_info["plugin_dir"], "config.json")
    if not os.path.exists(config):
        config = os.path.join(vk_info["plugin_dir"], "config_safe.json")
    if not os.path.exists(config):
        _GPU_DEVICES = []
        return _GPU_DEVICES

    devices = []
    import re as _re
    try:
        proc = subprocess.Popen(
            [exe, "-m", "pipe", "--vulkan", "-c", config],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=vk_info["plugin_dir"],
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )
        # 发送空请求让exe初始化Vulkan并输出GPU信息
        stdout, stderr = proc.communicate(input=b'{"img_path":""}\n', timeout=10)
        combined = stdout.decode("utf-8", errors="ignore") + "\n" + stderr.decode("utf-8", errors="ignore")
        # 解析GPU行: [0 AMD Radeon(TM) Graphics]  queueC=  r-score=21
        for line in combined.split("\n"):
            m = _re.search(r'\[(\d+) (.+?)\]\s+.*r-score=(\d+)', line)
            if m:
                devices.append({
                    "index": int(m.group(1)),
                    "name": m.group(2).strip(),
                    "score": int(m.group(3)),
                })
    except Exception as e:
        print(f"[GPU Detect] Error: {e}")

    if not devices:
        _GPU_DEVICES = []
        return _GPU_DEVICES

    # 标记是否为独立GPU
    # 集显常见模式：AMD Radeon(TM) Graphics、Intel(R) UHD、Intel Iris Xe
    # 独显常见模式：NVIDIA GeForce RTX 5060、AMD Radeon RX 6700 XT
    integrated_keywords = [
        "radeon(tm) graphics",  # AMD 核显（带 (TM) 后缀的通常是集显）
        "intel", "uhd", "iris", "hd graphics", "vega",
    ]
    dedicated_keywords = ["geforce", "rtx", "gtx", "radeon rx", "radeon pro"]
    for d in devices:
        name_lower = d["name"].lower()
        is_dedicated = any(kw in name_lower for kw in dedicated_keywords)
        if not is_dedicated:
            is_integrated = any(kw in name_lower for kw in integrated_keywords)
            d["dedicated"] = not is_integrated
        else:
            d["dedicated"] = True
        # 标记 GPU 加速可用性：低分 GPU（含大部分核显）走 GPU 反而比 CPU 慢
        d["supported"] = d.get("score", 0) >= 30

    _GPU_DEVICES = devices
    print(f"[GPU Detect] Found devices: {devices}")
    return _GPU_DEVICES


def _select_best_gpu():
    """自动选择最佳GPU: 优先独立显卡(最高分), 无独立则选集显(最高分)"""
    devices = _detect_vulkan_gpus()
    if not devices:
        return -1, "自动"
    # 优先独立显卡
    dedicated = [d for d in devices if d.get("dedicated")]
    if dedicated:
        best = max(dedicated, key=lambda d: d["score"])
    else:
        best = max(devices, key=lambda d: d["score"])
    return best["index"], best["name"]


def get_gpu_devices_for_ui():
    """获取GPU设备列表，优先Vulkan检测，失败时回退到nvidia-smi"""
    devices = _detect_vulkan_gpus()
    if devices:
        return devices
    # 回退: 通过nvidia-smi检测NVIDIA显卡
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            timeout=5, encoding="utf-8"
        )
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    name = parts[1]
                    devices.append({
                        "index": idx,
                        "name": name,
                        "score": 50,
                        "dedicated": True,
                    })
                except:
                    pass
        if devices:
            print(f"[GPU] nvidia-smi fallback: {devices}")
    except Exception:
        pass
    return devices

# ============================================================

def _get_gpu_vram_mb():
    """
    Query NVIDIA GPU VRAM (MB) via nvidia-smi.
    Returns int or None (non-NVIDIA / detection failed).
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total",
             "--format=csv,noheader"],
            timeout=5, encoding="utf-8"
        )
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        if not lines:
            return None
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) >= 3:
            mem_str = parts[2].lower().replace("mib", "").replace("mb", "").strip()
            try:
                return int(float(mem_str))
            except ValueError:
                pass
    except Exception:
        pass
    return None


def _get_side_for_vram(doc_idx, gpu_idx):
    """
    Return safe target_side based on GPU option. 信任用户选择，不做VRAM检测。
    None = use default values (no adjustment needed).
    gpu_idx: 0=大显存独显, 1=小显存独显, 2=核显, 3=纯CPU
    """
    if gpu_idx == 0 or gpu_idx == 3:  # 大显存独显 / 纯CPU → 用默认值
        return None
    # 小显存独显(1) / 核显(2) → 保守边长防爆
    if doc_idx == 1:  # 古籍竖排, 原2880易爆显存
        return 2240
    if doc_idx == 2:  # 扫描件, 原2400保守到2000
        return 2000
    return None  # 普通文档2000不变


# 引擎适配器抽象基类
# ============================================================

class OCREngineAdapter(ABC):
    """所有OCR引擎的通用接口"""

    def __init__(self, engine_id, plugin_dir, entry_path):
        self.engine_id = engine_id
        self.plugin_dir = plugin_dir
        self.entry_path = entry_path

    @abstractmethod
    def start(self, params):
        """启动/配置引擎。返回: "" 成功, 错误信息 失败"""
        pass

    @abstractmethod
    def run_base64(self, image_base64, timeout=180):
        """OCR base64图片。返回统一格式: {"code": 100, "data": [...]}"""
        pass

    @abstractmethod
    def run_path(self, img_path, timeout=180):
        """OCR本地图片路径"""
        pass

    def stop(self):
        """停止引擎"""
        pass

    def close(self):
        """清理资源"""
        self.stop()

class NcnnVulkanAdapter(OCREngineAdapter):
    """基于 ncnn Vulkan 的通用引擎适配器（原生支持GPU+CPU双模式）"""

    def __init__(self, engine_id, plugin_dir, entry_path):
        super().__init__(engine_id, plugin_dir, entry_path)
        self.config_path = os.path.join(plugin_dir, "config.json")
        self.port = 18043
        self.port_offset = 0
        self.server_proc = None
        self.lock = threading.Lock()
        self.current_config = {}
        self._started = False
        self._use_gpu = True  # 默认为GPU模式

    def set_port_offset(self, offset):
        """设置端口偏移，支持多实例并行"""
        self.port_offset = offset

    @property
    def _server_port(self):
        return self.port + self.port_offset

    def start(self, params):
        self.stop()
        self.current_config = params
        self._use_gpu = params.get("use_gpu", True)
        try:
            if self._use_gpu:
                return self._start_gpu(params)
            else:
                return self._start_cpu(params)
        except Exception as e:
            return f"[Error] Engine start failed: {str(e)}"

    def _start_gpu(self, params):
        """GPU模式：启动TCP服务器"""
        config = self._build_gpu_config(params)
        # 多实例配置隔离：不同端口用不同config文件
        if self.port_offset:
            self.config_path = os.path.join(
                os.path.dirname(self.config_path),
                f"config_{self._server_port}.json"
            )
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        startupinfo = None
        if platform.system() == "Windows":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        port = self._server_port
        cmd = [self.entry_path, "-m", "tcp", "-c", self.config_path, "-p", str(port)]
        print(f"[Vulkan] GPU模式 config: gpu_device_index={params.get('gpu_device', -1)}")
        print(f"[Vulkan] Starting: {' '.join(cmd)}")
        self.server_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, cwd=self.plugin_dir,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            if self._server_running():
                self._started = True
                print("[Vulkan] Server is ready on port " + str(port))
                return ""
            if self.server_proc.poll() is not None:
                stderr_text = ""
                try:
                    stderr_text = self.server_proc.stderr.read().decode('utf-8', errors='ignore')[:500]
                except: pass
                self.server_proc = None
                return "[Error] Vulkan server exited during startup. stderr: " + stderr_text
            time.sleep(0.1)
        stderr_text = ""
        try:
            stderr_text = self.server_proc.stderr.read().decode('utf-8', errors='ignore')[:500]
        except: pass
        return "[Error] Vulkan server failed to start within 30s. stderr: " + stderr_text

    def _start_cpu(self, params):
        """CPU模式：仅写入CPU config，不启动TCP服务器"""
        config = self._build_cpu_config(params)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        self._started = False  # CPU模式无长驻进程
        print("[Vulkan] CPU模式 config written (no TCP server)")
        return ""

    def _build_gpu_config(self, params):
        """构建GPU模式config（含use_vulkan和gpu_device_index）"""
        base_dir = "models"
        model_version = params.get("model_version", "")
        num_threads = params.get("num_threads", -1)
        enable_fp16 = params.get("enable_fp16", False)
        det_thres = params.get("det_thres", 0.5)
        unclip_ratio = params.get("unclip_ratio", 1.58)
        enable_cls = params.get("enable_cls", True)
        max_side_len = params.get("max_side_len", 2000)
        gpu_device = params.get("gpu_device", -1)
        if num_threads <= 0:
            cpu_count = os.cpu_count() or 1
            num_threads = min(cpu_count, 6)
        available = _get_available_ncnn_models("ncnn_vulkan")
        if not available:
            raise RuntimeError("没有找到可用的ncnn Vulkan模型文件")
        if model_version and model_version in available:
            selected = model_version
        else:
            selected = available[0]
            print(f"[Vulkan] 使用检测到的模型: {selected}")
        model_map = {}
        for base in available:
            model_map[base] = (base + "_det", base + "_rec")
        if not model_map:
            raise RuntimeError("没有找到可用的ncnn Vulkan模型")
        det_model, rec_model = model_map.get(selected, list(model_map.values())[0])
        lang = params.get("lang", "chinese")
        if lang != "chinese":
            print(f"[Vulkan] Language: {lang} (PP-OCRv6 dict covers Latin/CJK/Korean/Cyrillic)")
        if "v3" in model_version:
            keys_file = "ppocr_keys_v1.txt"
        elif "v6" in model_version:
            keys_file = "ppocr_keys_v6.txt"
        else:
            keys_file = "ppocr_keys_v5.txt"
        return {
            "save": False,
            "det": {
                "infer_threads": num_threads,
                "model_path": f"{base_dir}/{det_model}",
                "padding": 50,
                "max_side_len": max_side_len,
                "box_thres": det_thres,
                "bitmap_thres": det_thres * 0.6,
                "unclip_ratio": unclip_ratio,
                "fp16": enable_fp16,
                "use_vulkan": True,
                "gpu_device_index": gpu_device
            },
            "cls": {
                "infer_threads": min(2, num_threads),
                "reco_threads": num_threads,
                "model_path": f"{base_dir}/PP_LCNet_x0_25_textline_ori",
                "enable": enable_cls,
                "most_angle": True,
                "fp16": enable_fp16,
                "use_vulkan": True,
                "gpu_device_index": gpu_device
            },
            "rec": {
                "infer_threads": min(4, num_threads),
                "reco_threads": num_threads,
                "model_path": f"{base_dir}/{rec_model}",
                "keys_path": f"{base_dir}/{keys_file}",
                "fp16": enable_fp16,
                "use_vulkan": True,
                "gpu_device_index": gpu_device
            }
        }

    def _build_cpu_config(self, params):
        """构建CPU模式config（不含use_vulkan和gpu_device_index）"""
        base_dir = "models"
        model_version = params.get("model_version", "")
        num_threads = params.get("num_threads", -1)
        enable_fp16 = params.get("enable_fp16", False)
        det_thres = params.get("det_thres", 0.5)
        unclip_ratio = params.get("unclip_ratio", 1.58)
        enable_cls = params.get("enable_cls", True)
        max_side_len = params.get("max_side_len", 2000)
        if num_threads <= 0:
            cpu_count = os.cpu_count() or 1
            num_threads = min(cpu_count, 6)
        available = _get_available_ncnn_models("ncnn_vulkan")
        if not available:
            raise RuntimeError("没有找到可用的ncnn模型文件")
        if model_version and model_version in available:
            selected = model_version
        else:
            selected = available[0]
        model_map = {}
        for base in available:
            model_map[base] = (base + "_det", base + "_rec")
        if not model_map:
            raise RuntimeError("没有找到可用的ncnn模型")
        det_model, rec_model = model_map.get(selected, list(model_map.values())[0])
        lang = params.get("lang", "chinese")
        if lang != "chinese":
            print(f"[ncnn] Language: {lang} (model already supports all characters)")
        if "v3" in model_version:
            keys_file = "ppocr_keys_v1.txt"
        elif "v6" in model_version:
            keys_file = "ppocr_keys_v6.txt"
        else:
            keys_file = "ppocr_keys_v5.txt"
        return {
            "save": False,
            "det": {
                "infer_threads": num_threads,
                "model_path": f"{base_dir}/{det_model}",
                "padding": 50,
                "max_side_len": max_side_len,
                "box_thres": det_thres,
                "bitmap_thres": det_thres * 0.6,
                "unclip_ratio": unclip_ratio,
                "fp16": enable_fp16
            },
            "cls": {
                "infer_threads": min(2, num_threads),
                "reco_threads": num_threads,
                "model_path": f"{base_dir}/PP_LCNet_x0_25_textline_ori",
                "enable": enable_cls,
                "most_angle": True,
                "fp16": enable_fp16
            },
            "rec": {
                "infer_threads": min(4, num_threads),
                "reco_threads": num_threads,
                "model_path": f"{base_dir}/{rec_model}",
                "keys_path": f"{base_dir}/{keys_file}",
                "fp16": enable_fp16
            }
        }

    def _server_running(self):
        try:
            with socket.create_connection(("127.0.0.1", self._server_port), timeout=1):
                return True
        except Exception:
            return False

    def _tcp_request(self, request, timeout=180):
        with self.lock:
            if not self._ensure_running():
                return {"code": 102, "data": "Vulkan server not available"}
            try:
                with socket.create_connection(("127.0.0.1", self._server_port), timeout=timeout) as sock:
                    sock.settimeout(timeout)
                    json_str = json.dumps(request)
                    sock.sendall(json_str.encode("utf-8"))
                    chunks = []
                    while True:
                        try:
                            chunk = sock.recv(4096)
                        except socket.timeout:
                            return {"code": 102, "data": f"OCR timeout ({timeout}s)"}
                        if not chunk:
                            break
                        chunks.append(chunk)
                    text = b"".join(chunks).decode("utf-8", errors="ignore")
                    return self._parse_json(text)
            except Exception as e:
                return {"code": 102, "data": f"TCP error: {str(e)}"}

    def _ensure_running(self):
        if self._server_running():
            return True
        self._start_gpu(self.current_config)
        return self._started

    def _run_exe(self, img_path, timeout=180):
        """CPU模式：PIPE模式启动子进程，单次请求后退出"""
        startupinfo = None
        if platform.system() == "Windows":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        try:
            proc = subprocess.Popen(
                [self.entry_path, "-m", "pipe", "-c", self.config_path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=self.plugin_dir,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )
            request = {"img_path": img_path.replace("\\", "/")}
            json_str = json.dumps(request) + "\n"
            stdout, stderr = proc.communicate(input=json_str.encode("utf-8"), timeout=timeout)
            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="ignore") if stderr else "Unknown"
                return {"code": 102, "data": f"Process error (exit {proc.returncode}): {err_msg}"}
            stdout_str = stdout.decode("utf-8", errors="ignore")
            return self._parse_json(stdout_str)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            return {"code": 102, "data": f"OCR timeout ({timeout}s)"}
        except Exception as e:
            return {"code": 102, "data": f"OCR error: {str(e)}"}

    def _parse_json(self, text):
        first = text.find("{")
        if first == -1:
            return {"code": 102, "data": "No JSON"}
        js = text[first:]
        try:
            result = json.loads(js)
        except json.JSONDecodeError:
            bc = 0
            for i, ch in enumerate(js):
                if ch == "{": bc += 1
                elif ch == "}": bc -= 1
                if bc == 0:
                    try:
                        result = json.loads(js[:i+1])
                    except json.JSONDecodeError:
                        return {"code": 102, "data": "JSON parse failed"}
                    break
            else:
                return {"code": 102, "data": "No complete JSON"}
        code = result.get("code")
        if code == 200:
            return {"code": 100, "data": result.get("data", [])}
        elif code in (300, 400):
            return {"code": 101, "data": ""}
        else:
            return {"code": 102, "data": result.get("data", result.get("error", "Unknown"))}

    def run_base64(self, image_base64, timeout=180):
        try:
            img_bytes = _b64.b64decode(image_base64)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            try:
                if self._use_gpu:
                    request = {"img_path": tmp_path.replace("\\", "/")}
                    return self._tcp_request(request, timeout)
                else:
                    return self._run_exe(tmp_path, timeout)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            return {"code": 900, "data": f"Base64 error: {str(e)}"}

    def run_path(self, img_path, timeout=180):
        if self._use_gpu:
            request = {"img_path": img_path.replace("\\", "/")}
            return self._tcp_request(request, timeout)
        else:
            return self._run_exe(img_path, timeout)

    def stop(self):
        if self._use_gpu:
            if self.server_proc is not None:
                try:
                    if self.server_proc.poll() is None:
                        self.server_proc.terminate()
                        self.server_proc.wait(timeout=5)
                except Exception:
                    try:
                        self.server_proc.kill()
                    except Exception:
                        pass
                self.server_proc = None
            self._started = False
        # CPU模式：单次子进程已结束，无操作

    def close(self):
        self.stop()

# ============================================================
# 引擎适配器工厂
# ============================================================

def create_engine_adapter(engine_id):
    """创建引擎适配器实例（仅ncnn Vulkan）"""
    if engine_id not in _PLUGIN_DIRS:
        raise ValueError("引擎 " + engine_id + " 未找到插件目录")
    info = _PLUGIN_DIRS[engine_id]
    return NcnnVulkanAdapter(engine_id, info["plugin_dir"], info["entry_path"])

# ============================================================
# 引擎参数构建器
# ============================================================

def build_engine_params(engine_id, use_gpu, vertical_text, limit_side_len,
                        model_size_or_version, use_angle_cls, extra_params=None):
    """根据引擎类型和用户设置，构建引擎启动参数字典（仅ncnn Vulkan）"""
    if extra_params is None:
        extra_params = {}
    params = {}
    params["model_version"] = model_size_or_version
    params["max_side_len"] = limit_side_len
    params["enable_cls"] = use_angle_cls
    params["num_threads"] = extra_params.get("num_threads", -1)
    params["enable_fp16"] = extra_params.get("enable_fp16", False)
    params["det_thres"] = extra_params.get("det_thres", 0.5)
    params["unclip_ratio"] = extra_params.get("unclip_ratio", 1.58)
    params["use_gpu"] = extra_params.get("use_gpu", True)
    params["lang"] = extra_params.get("lang", "chinese") or "chinese"
    if extra_params.get("use_gpu", True):
        gpu_device = extra_params.get("gpu_device", -1)
        if gpu_device < 0:
            # 强制重新检测GPU（清除缓存）
            devices = _detect_vulkan_gpus(force_redetect=True)
            print(f"[GPU] Forced re-detect: {devices}")
            auto_idx, auto_name = _select_best_gpu()
            gpu_device = auto_idx
            print(f"[GPU] 自动选择: device {auto_idx} ({auto_name})")
        params["gpu_device"] = gpu_device
    return params

# ============================================================
# OCR Client - Multi-Engine Singleton
# ============================================================

class OCRClient:
    """OCR客户端 - 单例模式（ncnn Vulkan 引擎）
    支持双实例并行OCR（dual_instance=True时启动两个独立引擎进程），
    通过轮询调度提升GPU利用率。
    """
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, engine_id="ncnn_vulkan", use_gpu=True,
                 vertical_text=True, limit_side_len=2000,
                 model_size="medium", use_angle_cls=False,
                 dual_instance=True, extra_params=None):
        if extra_params is None:
            extra_params = {}
        gpu_device = extra_params.get("gpu_device", -1)
        if hasattr(self, "_initialized") and self._initialized:
            if getattr(self, "_engine_id", None) == engine_id and \
               getattr(self, "_dual_instance", None) == dual_instance and \
               getattr(self, "_use_gpu", None) == use_gpu and \
               getattr(self, "_gpu_device", None) == gpu_device:
                return
            self.close()
        self._engine_id = engine_id
        self._use_gpu = use_gpu
        self._gpu_device = gpu_device
        self._dual_instance = dual_instance
        self._adapter_index = 0
        self._instances = []
        num_instances = 2 if dual_instance else 1
        params = build_engine_params(
            engine_id, use_gpu, vertical_text, limit_side_len,
            model_size, use_angle_cls, extra_params
        )
        for i in range(num_instances):
            adapter = create_engine_adapter(engine_id)
            # Vulkan多实例需要隔离端口
            if hasattr(adapter, 'set_port_offset'):
                adapter.set_port_offset(i * 10)  # 实例0: 18043, 实例1: 18053
            err = adapter.start(params)
            if err:
                for a in self._instances:
                    a.close()
                raise RuntimeError(f"引擎实例{i+1}启动失败: {err}")
            self._instances.append(adapter)
            print(f"[OCRClient] 实例{i+1}就绪: {engine_id} | GPU={use_gpu} | 边长={limit_side_len} | 模型={model_size}")
        self.adapter = self._instances[0]
        self._initialized = True

    def ocr_image_base64(self, image_base64, timeout_seconds=180):
        """轮询调度多个实例"""
        if self._dual_instance and len(self._instances) > 0:
            idx = self._adapter_index % len(self._instances)
            self._adapter_index += 1
            return self._instances[idx].run_base64(image_base64, timeout_seconds)
        return self._instances[0].run_base64(image_base64, timeout_seconds)

    def force_close(self):
        """强制关闭：直接 kill 子进程，立即中断阻塞的 OCR 调用"""
        for a in getattr(self, "_instances", []):
            try:
                if hasattr(a, 'server_proc') and a.server_proc:
                    a.server_proc.kill()
            except Exception:
                pass
        self._instances = []
        self._initialized = False

    def close(self):
        for a in getattr(self, "_instances", []):
            try:
                a.close()
            except Exception:
                pass
        self._instances = []
        self._initialized = False

    def __del__(self):
        self.close()
# ============================================================
# PDF Processor - Pre-render Pipeline
# ============================================================

class PDFProcessor:
    def __init__(self, ocr_client, dual_instance=True):
        self.ocr = ocr_client
        self.dual_instance = dual_instance
        with _suppress_mupdf_warnings():
            self.font = fitz.Font("cjk")
        self._paused = False
        self._cancelled = False
        self.results = {}
        self.results_lock = threading.Lock()
        self.completed_count = 0
        self.completed_lock = threading.Lock()
        self._total_done = 0
        self._start_time = 0

    def reset(self):
        self._paused = False
        self._cancelled = False
        self.results = {}
        self.completed_count = 0
        self._total_done = 0
        self._start_time = 0
        self.completed_count = 0

    @property
    def is_paused(self):
        return self._paused
    @property
    def is_cancelled(self):
        return self._cancelled
    def pause(self):
        self._paused = True
    def resume(self):
        self._paused = False
    def cancel(self):
        self._cancelled = True
        self._paused = False
        # 强制中断阻塞的OCR调用
        if hasattr(self, 'ocr') and self.ocr:
            self.ocr.force_close()
    def wait_if_paused(self):
        while self._paused and not self._cancelled:
            time.sleep(0.05)

    def calculate_font_size(self, text, width, height):
        if height > width:
            width, height = height, width
        fontsize = round(height)
        min_size = 5
        while self.font.text_length(text, fontsize=fontsize) > width and fontsize >= min_size:
            fontsize -= 1
        while self.font.text_length(text, fontsize=fontsize) < width:
            fontsize += 1
        while self.font.text_length(text, fontsize=fontsize) > width and fontsize >= min_size:
            fontsize -= 0.1
        return fontsize

    def render_page_to_bytes(self, pdf_path, page_num, scale=2.0):
        with _suppress_mupdf_warnings():
            doc = fitz.open(pdf_path)
        page = doc[page_num]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes

    def process_pdf(self, input_path, output_dir, total_pages, scale=2.0,
                    progress_callback=None):
        """处理单个PDF文件。output_dir=None时输出到源文件所在目录"""
        if output_dir is None:
            output_dir = os.path.dirname(input_path)
        if self._cancelled:
            return None, None
        print(f"\n[PDFProcessor] Processing: {input_path} ({total_pages} pages, scale={scale}x)")
        self.reset()
        all_done = threading.Event()
        _render_done_count = [0]
        _render_done_lock = threading.Lock()
        def render_worker(start_page, end_page):
            for pn in range(start_page, end_page):
                if self._cancelled:
                    return
                self.wait_if_paused()
                if self._cancelled:
                    return
                try:
                    png_bytes = self.render_page_to_bytes(input_path, pn, scale)
                    b64_data = _b64.b64encode(png_bytes).decode("ascii")
                    while not self._cancelled:
                        try:
                            render_queue.put((pn, b64_data), timeout=1)
                            break
                        except Full:
                            continue
                except Exception as e:
                    print(f"[Render] Page {pn} error: {e}")
            with _render_done_lock:
                _render_done_count[0] += 1
                if _render_done_count[0] >= self._num_workers:
                    all_done.set()
        def ocr_consumer(consumer_id):
            my_done = 0
            done_pages = [False] * total_pages
            next_to_store = 0
            t0 = time.time()
            while my_done < total_pages and not self._cancelled:
                try:
                    pn, b64_data = render_queue.get(timeout=2)
                except Empty:
                    if all_done.is_set():
                        break
                    continue
                if self._cancelled:
                    break
                try:
                    result = self.ocr.ocr_image_base64(b64_data, timeout_seconds=180)
                except Exception as e:
                    result = {"code": 900, "data": f"OCR error: {str(e)}"}
                with self.results_lock:
                    self.results[pn] = result
                    done_pages[pn] = True
                    while next_to_store < total_pages and done_pages[next_to_store]:
                        next_to_store += 1
                my_done += 1
                with self.completed_lock:
                    self._total_done += 1
                if my_done % 10 == 0:
                    elapsed = time.time() - t0
                    rate = my_done / elapsed if elapsed > 0 else 0
                    print(f"[OCR-{consumer_id}] +{my_done} ({rate:.2f} p/s)")
                # 合计每10页打一次（只有consumer-1打）
                with self.completed_lock:
                    total_done = self._total_done
                if consumer_id == 1 and total_done % 10 == 0 and total_done > 0:
                    total_time = time.time() - self._start_time
                    total_rate = total_done / total_time if total_time > 0 else 0
                    print(f"[合计] {total_done}/{total_pages} | {total_rate:.2f} p/s")
                if progress_callback:
                    progress_callback(self._total_done, total_pages, next_to_store)
        num_consumers = 2 if self.dual_instance else 1
        render_queue = Queue(maxsize=20 if num_consumers > 1 else 12)
        # 渲染线程数 ≈ CPU核数-消费者数，留足CPU给OCR进程
        cpu_cores = os.cpu_count() or 8
        n_workers = min(max(4, cpu_cores - num_consumers), total_pages, 32)
        # 队列反压：maxsize控制预渲染量，避免撑爆内存
        # 满队列时put()自动阻塞→渲染线程等待→自然调节投喂速度
        print(f"[PDFProcessor] {n_workers} render threads, queue={render_queue.maxsize} (CPU={cpu_cores}, dual={num_consumers>1})")
        workers_per_thread = (total_pages + n_workers - 1) // n_workers
        producers = []
        self._num_workers = 0
        for i in range(n_workers):
            start = i * workers_per_thread
            end = min(start + workers_per_thread, total_pages)
            if start >= total_pages:
                break
            t = threading.Thread(target=render_worker, args=(start, end))
            t.daemon = True
            t.start()
            producers.append(t)
            self._num_workers += 1
        self._start_time = time.time()
        print(f"[PDFProcessor] Starting: {self._num_workers} render threads + {num_consumers} OCR consumers")
        consumers = []
        for i in range(num_consumers):
            c = threading.Thread(target=ocr_consumer, args=(i + 1,))
            c.daemon = True
            c.start()
            consumers.append(c)
        for c in consumers:
            c.join()
        for t in producers:
            t.join()
        if self._cancelled:
            return None, None
        return self.write_results(input_path, output_dir, total_pages, scale)

    def write_results(self, input_path, output_dir, total_pages, scale):
        input_name = Path(input_path).stem
        txt_path = os.path.join(output_dir, f"{input_name}_result.txt")
        et = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write("OCR文本提取结果\n")
            txt_file.write("=" * 60 + "\n")
            txt_file.write(f"源文件: {input_name}\n")
            txt_file.write(f"总页数: {total_pages}\n")
            txt_file.write(f"渲染倍数: {scale}x\n")
            txt_file.write(f"提取时间: {et}\n")
            txt_file.write("=" * 60 + "\n\n")
            with _suppress_mupdf_warnings():
                output_pdf = fitz.open(input_path)
            output_page_count = 0
            for page_num in sorted(self.results.keys()):
                result = self.results[page_num]
                if result.get("code") != 100:
                    continue
                text_blocks = result.get("data", [])
                page_text_lines = []
                output_page = output_pdf[page_num]
                output_page.clean_contents()
                page_rotation = output_page.rotation
                is_insert_font = False
                for tb in text_blocks:
                    text = tb.get("text", "")
                    if not text.strip():
                        continue
                    page_text_lines.append(text)
                    box = tb.get("box", [[0,0],[0,0],[0,0],[0,0]])
                    x0, y0 = box[0]
                    x2, y2 = box[2]
                    w = x2 - x0
                    h = y2 - y0
                    fontsize = self.calculate_font_size(text, w, h)
                    point = fitz.Point(x0, y2) * output_page.derotation_matrix
                    if not is_insert_font:
                        output_page.insert_font(fontname="cjk", fontbuffer=self.font.buffer)
                        is_insert_font = True
                    output_page.insert_text(
                        point, text, fontsize=fontsize, fontname="cjk",
                        rotate=page_rotation, stroke_opacity=0, fill_opacity=0
                    )
                if page_text_lines:
                    txt_file.write("\n" + "=" * 60 + "\n")
                    txt_file.write(f"第 {page_num + 1} 页\n")
                    txt_file.write("=" * 60 + "\n\n")
                    for line in page_text_lines:
                        txt_file.write(line + "\n")
                    output_page_count += 1
            output_pdf_path = os.path.join(output_dir, f"{input_name}_layered.pdf")
            output_pdf.set_metadata({
                "title": f"{input_name} - OCR Layered PDF",
                "author": "CathayOCR Lite",
                "subject": f"OCR extracted on {et}",
                "creator": "CathayOCR Lite Processor",
            })
            try:
                if total_pages <= 2000:
                    output_pdf.subset_fonts()
                    output_pdf.save(output_pdf_path, deflate=True, garbage=3)
                else:
                    output_pdf.save(output_pdf_path, deflate=True, garbage=1)
            except:
                output_pdf.save(output_pdf_path)
            output_pdf.close()
        print(f"[PDFProcessor] Done: {output_pdf_path}, TXT: {txt_path}")
        return output_pdf_path, txt_path

# ============================================================
# Batch Worker Thread
# ============================================================

class BatchWorkerThread(QThread):
    file_progress = pyqtSignal(str, int, int)
    file_finished = pyqtSignal(str, str, str)
    file_error = pyqtSignal(str, str)
    file_cancelled = pyqtSignal(str)
    all_finished = pyqtSignal(int, int, int)

    def __init__(self, file_list, output_dir,
                 engine_id="ncnn_vulkan", use_gpu=True,
                 vertical_text=True, limit_side_len=2000,
                 model_size="medium", use_angle_cls=False,
                 scale=2.0, dual_instance=True, extra_params=None):
        super().__init__()
        self.file_list = file_list
        self.output_dir = output_dir
        self.engine_id = engine_id
        self.use_gpu = use_gpu
        self.vertical_text = vertical_text
        self.limit_side_len = limit_side_len
        self.model_size = model_size
        self.use_angle_cls = use_angle_cls
        self.scale = scale
        self.dual_instance = dual_instance
        self.extra_params = extra_params or {}
        self.is_cancelled = False
        self.is_paused = False
        self.processor = None
        self.current_filename = ""

    def run(self):
        try:
            ocr_client = OCRClient(
                engine_id=self.engine_id,
                use_gpu=self.use_gpu,
                vertical_text=self.vertical_text,
                limit_side_len=self.limit_side_len,
                model_size=self.model_size,
                use_angle_cls=self.use_angle_cls,
                dual_instance=self.dual_instance,
                extra_params=self.extra_params,
            )
            self.processor = PDFProcessor(ocr_client, dual_instance=self.dual_instance)
            success_count = 0
            cancelled_count = 0
            total_files = len(self.file_list)
            for idx, input_path in enumerate(self.file_list):
                if self.is_cancelled:
                    break
                self.current_filename = os.path.basename(input_path)
                while self.is_paused and not self.is_cancelled:
                    time.sleep(0.1)
                if self.is_cancelled:
                    break
                try:
                    self.processor.reset()
                    with _suppress_mupdf_warnings():
                        pdf_doc = fitz.open(input_path)
                    total_pages = len(pdf_doc)
                    pdf_doc.close()
                    result = self.processor.process_pdf(
                        input_path, self.output_dir, total_pages,
                        scale=self.scale,
                        progress_callback=lambda done, total, stored:
                            self.file_progress.emit(self.current_filename, done, total)
                    )
                    if result[0] is None:
                        cancelled_count += 1
                        self.file_cancelled.emit(self.current_filename)
                    else:
                        pdf_path, txt_path = result
                        self.file_finished.emit(self.current_filename, pdf_path, txt_path)
                        success_count += 1
                except Exception as e:
                    self.file_error.emit(self.current_filename, str(e))
            self.all_finished.emit(total_files, success_count, cancelled_count)
        except Exception as e:
            self.file_error.emit("System", str(e))

    def cancel(self):
        self.is_cancelled = True
        if self.processor:
            self.processor.cancel()
    def pause(self):
        self.is_paused = True
        if self.processor:
            self.processor.pause()
    def resume(self):
        self.is_paused = False
        if self.processor:
            self.processor.resume()

# ============================================================
# Main Window
# ============================================================

class MainWindow(QMainWindow):
    # ── 主语言列表（显示名 → 代码） ──
    _LANG_ITEMS = [
        # ── 核心语言 ──
        ("中文 (Chinese)", "ch"),
        ("English (英文)", "en"),
        ("Français (法文)", "fr"),
        ("Deutsch (德文)", "de"),
        ("日本語 (日文)", "japan"),
        ("한국어 (韩文)", "korean"),
        ("Español (西班牙文)", "es"),
        ("Italiano (意大利文)", "it"),
        ("Português (葡萄牙文)", "pt"),
        ("Pусский (俄文)", "ru"),
        ("Tiếng Việt (越南文)", "vi"),

        # ── 其他常用 ──
        ("Nederlands (荷兰文)", "nl"),
        ("Polski (波兰文)", "pl"),
        ("Magyar (匈牙利文)", "hu"),
        ("Čeština (捷克文)", "cs"),

        # ── 全字符集 ──
        ("多语言 (全部)", "ch&en&fr&de&ja&ko&..."),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CathayOCR Lite (极速版) - PDF OCR处理器")
        self.setGeometry(100, 100, 1100, 850)
        self.cfg = QSettings("QClaw", "PDFOCRProcessor")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        title = QLabel("PDF OCR 流水线处理工具 - ncnn Vulkan 单一引擎")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ════════════════════════════════════════════
        # 界面模式切换（简单模式 vs 专业模式）
        # ════════════════════════════════════════════
        ms_widget = QWidget()
        ms_layout = QHBoxLayout(ms_widget)
        ms_layout.setContentsMargins(0, 0, 0, 0)
        self.ui_simple_btn = QRadioButton("🎯 简单模式")
        self.ui_expert_btn = QRadioButton("🔧 专业模式")
        self.ui_expert_btn.setChecked(True)
        ms_layout.addWidget(self.ui_simple_btn)
        ms_layout.addWidget(self.ui_expert_btn)
        ms_layout.addStretch()
        hint_label = QLabel("简单模式：回答3个问题自动设置 | 专业模式：全部参数自由调节")
        hint_label.setStyleSheet("color: #888; font-size: 11px;")
        ms_layout.addWidget(hint_label)
        self.ui_simple_btn.toggled.connect(self._on_ui_mode_changed)
        layout.addWidget(ms_widget)

        # ════════════════════════════════════════════
        # 简单模式设置面板（选择场景自动配置）
        # ════════════════════════════════════════════
        self.simple_group = QGroupBox("简单模式 — 四步完成配置")
        sl = QVBoxLayout(self.simple_group)
        # 提示语
        tip = QLabel("根据你的文档类型、精度偏好和硬件情况，系统自动调好所有参数。小白用户直接选即可👇")
        tip.setStyleSheet("color: #555; font-style: italic; padding-bottom: 4px;")
        tip.setWordWrap(True)
        sl.addWidget(tip)
        # 问题1：文档类型
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("📄 文档类型"))
        self.simple_doc = QComboBox()
        self.simple_doc.addItems(["普通文档（横排印刷体）", "古籍竖排（繁体/竖排/复杂排版）", "扫描件/照片（可能方向不正）"])
        self.simple_doc.setMinimumWidth(300)
        def set_doc_tip(idx):
            tips = [
                "标准横排文档，大部分PDF都能用",
                "开启竖排检测，提高分辨率，适合古籍/碑帖",
                "开启方向纠正，适合扫描件/手机拍照的文档",
            ]
            self.simple_doc.setToolTip(tips[idx] if idx < len(tips) else "")
        self.simple_doc.currentIndexChanged.connect(set_doc_tip)
        set_doc_tip(0)
        r1.addWidget(self.simple_doc)
        r1.addStretch()
        sl.addLayout(r1)
        # 问题2：精度
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("🎯 精度与速度"))
        self.simple_speed = QComboBox()
        self.simple_speed.addItems(["速度优先（尽快出结果）", "标准平衡（推荐）", "精度优先（识别最准）"])
        self.simple_speed.setMinimumWidth(300)
        self.simple_speed.setToolTip("标准平衡 = 中等模型+适度参数，适合大部分场景")
        r2.addWidget(self.simple_speed)
        r2.addStretch()
        sl.addLayout(r2)
        # 问题3：显卡
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("💻 你的显卡"))
        self.simple_gpu = QComboBox()
        self.simple_gpu.addItems([
            "NVIDIA/AMD/Intel 独显 (显存>=12GB, 高性能推荐)",
            "NVIDIA 独显 (显存<=8GB) / AMD/Intel独显",
            "仅有核显",
            "纯CPU (兼容性最好, 速度最慢)",
        ])
        self.simple_gpu.setMinimumWidth(300)
        def set_gpu_tip(idx):
            tips = [
                "显存充裕，可处理高分辨率古籍/扫描件；古籍类默认2880边长",
                "<=8GB显存，古籍竖排自动降为2240边长避免爆显存",
                "集成显卡性能有限，推荐不高于2000边长",
                "纯CPU运行，兼容性最好但速度最慢",
            ]
            self.simple_gpu.setToolTip(tips[idx] if idx < len(tips) else "")
        self.simple_gpu.currentIndexChanged.connect(set_gpu_tip)
        set_gpu_tip(0)
        r3.addWidget(self.simple_gpu)
        r3.addStretch()
        sl.addLayout(r3)

        # 问题4：语言
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("🌐 文档语言"))
        self.simple_lang = QComboBox()
        self.simple_lang.addItems([item[0] for item in self._LANG_ITEMS])
        self.simple_lang.setMinimumWidth(300)
        self.simple_lang.setMaxVisibleItems(20)
        self.simple_lang.setToolTip("选择文档语言，部分引擎会自动切换对应OCR模型")
        r4.addWidget(self.simple_lang)
        r4.addStretch()
        sl.addLayout(r4)

        # 配置摘要（一行灰色小字）
        self.simple_preview = QLabel()
        self.simple_preview.setStyleSheet("color: #666; font-size: 11px; font-style: italic; padding: 0px; margin: 0px;")
        self.simple_preview.setWordWrap(True)
        sl.addWidget(self.simple_preview)

        # 监听变化自动更新预览
        self.simple_doc.currentIndexChanged.connect(self._apply_simple_settings)
        self.simple_speed.currentIndexChanged.connect(self._apply_simple_settings)
        self.simple_gpu.currentIndexChanged.connect(self._apply_simple_settings)
        self.simple_lang.currentIndexChanged.connect(self._apply_simple_settings)
        sl.addStretch()
        layout.addWidget(self.simple_group)
        self.simple_group.setVisible(False)
        # 收集专业模式的所有参数分组，用于简单模式下隐藏
        self._expert_groups = []

        # === 引擎选择（专业模式）===
        eg = QGroupBox("OCR引擎")
        self._expert_groups.append(eg)
        el = QHBoxLayout(eg)
        el.addWidget(QLabel("引擎:"))
        self.engine_combo = QLabel("PP-OCR (ncnn) — CathayOCR Lite")
        self.engine_combo.setToolTip(
            "CathayOCR Lite (极速版)\n"
            "GPU模式+Vulkan加速 | CPU模式+ncnn原生\n"
            "支持NVIDIA/AMD/Intel任意显卡\n"
            "支持中/英/法/德/日/意/西/葡/韩/俄/多语言\n"
            "支持多版本模型(v3~v6)"
        )
        self.engine_combo.currentData = lambda rid='ncnn_vulkan': rid
        el.addWidget(self.engine_combo)
        el.addStretch()
        el.addWidget(QLabel("模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumWidth(100)
        self.mode_combo.setToolTip(
            "运行模式选择:\n"
            "  自动(推荐) = 有独显自动用GPU，无独显用CPU\n"
            "  GPU模式     = 强制使用GPU加速\n"
            "  CPU模式     = 仅用CPU，省显存，适合老旧机器"
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        el.addWidget(self.mode_combo)
        el.addStretch()
        el.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setToolTip(
            "选择OCR模型版本:\n"
            "  medium (推荐) = 精度与速度最佳平衡\n"
            "  small         = 速度更快但精度稍低\n"
            "  同一引擎下，改模型不依赖网络下载"
        )
        el.addWidget(self.model_combo)
        el.addStretch()
        el.addWidget(QLabel("语言:"))
        self.lang_combo = QComboBox()
        self.lang_combo.setMinimumWidth(100)
        self.lang_combo.setMaxVisibleItems(20)
        self.lang_combo.addItems(["中文", "English", "Français", "Deutsch", "日本語", "多语言"])
        self.lang_combo.setToolTip(
            "选择识别语言:\n"
            "  ncnn PP-OCRv6 字典内置 CJK+拉丁+西里尔+韩文\n"
            "  \"中文\" = 繁简体中文自动识别\n"
            "  \"多语言 (全部)\" = 使用完整字典(18707字符)覆盖所有语种"
        )
        self.lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        el.addWidget(self.lang_combo)
        el.addStretch()
        # GPU设备选择（仅Vulkan引擎可见）
        el.addWidget(QLabel("GPU:"))
        self.gpu_combo = QComboBox()
        self.gpu_combo.setMinimumWidth(200)
        self.gpu_combo.setToolTip("选择Vulkan GPU设备。自动=优先独立显卡。仅ncnn Vulkan生效")
        self.gpu_combo.setVisible(True)
        el.addWidget(self.gpu_combo)
        layout.addWidget(eg)

        # === OCR 配置（专业模式）===
        cg = QGroupBox("OCR 配置")
        self._expert_groups.append(cg)
        cl = QHBoxLayout(cg)
        cl.addWidget(QLabel("图像边长:"))
        self.side_len_spin = QSpinBox()
        self.side_len_spin.setRange(320, 6400)
        self.side_len_spin.setSingleStep(320)
        self.side_len_spin.setValue(self.cfg.value("side_len", 2000, type=int))
        self.side_len_spin.setToolTip(
            "图像长边最大像素值 (小白推荐: 2000):\n"
            "  2000 (推荐) = 常规文档的最佳平衡点\n"
            "  >2000        = 精细识别，适合古籍/小字\n"
            "  <2000        = 更快但可能漏字\n"
            "  古籍/竖排/复杂排版建议 ≥2000"
        )
        cl.addWidget(self.side_len_spin)
        cl.addWidget(QLabel("渲染:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["1x(快速)", "2x(清晰)", "3x(超清)"])
        self.scale_combo.setCurrentIndex(self.cfg.value("scale", 1, type=int))
        self.scale_combo.setToolTip(
            "PDF页面渲染倍率 (小白推荐: 2x):\n"
            "  1x = 最快，但过小/过密文字可能识别不全\n"
            "  2x (推荐) = 清晰与速度的平衡，适用大部分文档\n"
            "  3x = 超清，适合极小号字体PDF，速度最慢"
        )
        cl.addWidget(self.scale_combo)
        cl.addWidget(QLabel("精度:"))
        self.precision_combo = QComboBox()
        self.precision_combo.addItem("FP32 (高精度)", "fp32")
        self.precision_combo.addItem("FP16 (快速)", "fp16")
        self.precision_combo.setCurrentIndex(self.cfg.value("precision_idx", 0, type=int))
        self.precision_combo.setToolTip(
            "计算精度 (仅ncnn引擎有效):\n"
            "  FP32 (推荐) = 32位浮点，精度最高\n"
            "  FP16        = 16位半精度，速度略快\n"
            "  AMD核显建议使用FP32（FP16可能不稳定）"
        )
        cl.addWidget(self.precision_combo)
        cl.addStretch()
        layout.addWidget(cg)

        # === OCR 选项（专业模式）===
        og = QGroupBox("OCR 选项")
        self._expert_groups.append(og)
        ol = QHBoxLayout(og)
        self.vertical_check = QCheckBox("竖排文字")
        self.vertical_check.setChecked(self.cfg.value("vertical", True, type=bool))
        self.vertical_check.setToolTip(
            "竖排文字检测 (小白推荐: 古籍开启，普通文档关闭):\n"
            "  古籍/碑帖/对联等竖排文档 → 建议开启\n"
            "  普通横排文档 → 关闭可提速"
        )
        ol.addWidget(self.vertical_check)
        self.angle_cls_check = QCheckBox("方向纠正")
        self.angle_cls_check.setChecked(self.cfg.value("angle_cls", False, type=bool))
        self.angle_cls_check.setToolTip(
            "自动纠正图片方向 (小白推荐: 扫描件开启):\n"
            "  自动检测 0°/90°/180°/270° 并纠正\n"
            "  会增加约10%处理时间\n"
            "  扫描件/手机拍照的PDF → 建议开启\n"
            "  确认方向正确的电子PDF → 关闭更快"
        )
        ol.addWidget(self.angle_cls_check)
        self.rec_batch_spin = QSpinBox()
        self.rec_batch_spin.setRange(1, 64)
        self.rec_batch_spin.setValue(self.cfg.value("rec_batch", 12, type=int))
        self.rec_batch_spin.setToolTip(
            "识别批处理数 (仅PP-OCRv6, 小白推荐: 保持默认):\n"
            "  批量越大GPU利用率越高，但显存占用也越大\n"
            "  GPU模式: 12~16 (推荐)\n"
            "  CPU模式: 4~8   (推荐)\n"
            "  数值过大可能导致显存溢出(OOM)"
        )
        self.rec_batch_spin.setVisible(False)
        ol.addWidget(self.rec_batch_spin)
        self.shrink_check = QCheckBox("精对齐")
        self.shrink_check.setChecked(self.cfg.value("shrink", False, type=bool))
        self.shrink_check.setToolTip(
            "检测框精对齐 (仅PP-OCRv6, 小白可忽略):\n"
            "  合并并精调相邻文本行\n"
            "  改善段落文字识别的连贯性\n"
            "  对排版对齐要求高的文档建议开启"
        )
        self.shrink_check.setVisible(False)
        ol.addWidget(self.shrink_check)
        self.tensorrt_check = QCheckBox("TensorRT")
        self.tensorrt_check.setChecked(self.cfg.value("tensorrt", False, type=bool))
        self.tensorrt_check.setToolTip(
            "启用TensorRT加速(当前未启用):\n"
            "  需额外安装TensorRT运行时")
        self.tensorrt_check.setVisible(False)
        ol.addWidget(self.tensorrt_check)
        self.dual_check = QCheckBox("双实例并行")
        self.dual_check.setChecked(self.cfg.value("dual", True, type=bool))
        self.dual_check.setToolTip(
            "双实例并行 (小白推荐: GPU开启/CPU关闭):\n"
            "  启动两个OCR进程并行处理一页PDF\n"
            "  可提升GPU利用率30%~50%\n"
            "  ⚡ CPU模式下自动禁用（双实例对CPU无增益）\n"
            "  双实例会增加约1GB显存占用"
        )
        ol.addWidget(self.dual_check)
        ol.addStretch()
        layout.addWidget(og)

        self._last_input_dir = self.cfg.value("last_input_dir", "")
        self._last_output_dir = self.cfg.value("last_output_dir", "")

        # === 输入模式 ===
        mg = QGroupBox("输入模式")
        ml = QVBoxLayout(mg)
        mr = QHBoxLayout()
        self.mode_files = QRadioButton("选择多个文件")
        self.mode_folder = QRadioButton("遍历文件夹")
        self.mode_files.setChecked(True)
        mr.addWidget(self.mode_files)
        mr.addWidget(self.mode_folder)
        mr.addStretch()
        ml.addLayout(mr)
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(100)
        ml.addWidget(QLabel("待处理文件列表:"))
        ml.addWidget(self.file_list)
        fb = QHBoxLayout()
        self.add_files_btn = QPushButton("添加文件")
        self.add_files_btn.clicked.connect(self.add_files)
        fb.addWidget(self.add_files_btn)
        self.add_folder_btn = QPushButton("添加文件夹")
        self.add_folder_btn.clicked.connect(self.add_folder)
        fb.addWidget(self.add_folder_btn)
        self.add_folders_btn = QPushButton("添加多个文件夹")
        self.add_folders_btn.clicked.connect(self.add_multiple_folders)
        fb.addWidget(self.add_folders_btn)
        self.clear_files_btn = QPushButton("清空列表")
        self.clear_files_btn.clicked.connect(self.clear_files)
        fb.addWidget(self.clear_files_btn)
        ml.addLayout(fb)
        layout.addWidget(mg)

        # === 输出 ===
        og2 = QGroupBox("输出目录")
        ol2 = QHBoxLayout(og2)
        self.output_edit = QLineEdit()
        if self._last_output_dir:
            self.output_edit.setText(self._last_output_dir)
        ol2.addWidget(self.output_edit)
        ob = QPushButton("浏览...")
        ob.clicked.connect(self.browse_output)
        ol2.addWidget(ob)
        layout.addWidget(og2)

        # === 进度 ===
        pg = QGroupBox("处理进度")
        pl = QVBoxLayout(pg)
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 10000)
        pl.addWidget(QLabel("总体进度:"))
        pl.addWidget(self.overall_progress)
        self.current_file_label = QLabel("当前文件: 无")
        self.current_file_label.setStyleSheet("font-weight: bold;")
        pl.addWidget(self.current_file_label)
        pi = QHBoxLayout()
        pi.addWidget(QLabel("当前文件进度:"))
        self.page_info_label = QLabel("0 / 0 页")
        self.page_info_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        pi.addWidget(self.page_info_label)
        pi.addStretch()
        pl.addLayout(pi)
        self.page_progress = QProgressBar()
        self.page_progress.setRange(0, 10000)
        pl.addWidget(self.page_progress)
        sr = QHBoxLayout()
        self.status_label = QLabel("等待开始...")
        sr.addWidget(self.status_label)
        sr.addStretch()
        self.pause_label = QLabel("")
        self.pause_label.setStyleSheet("color: #ff6600;")
        sr.addWidget(self.pause_label)
        pl.addLayout(sr)
        self.speed_label = QLabel("处理速度: --")
        pl.addWidget(self.speed_label)
        self.gpu_label = QLabel("GPU: --")
        pl.addWidget(self.gpu_label)
        layout.addWidget(pg)

        # === 日志 ===
        lg = QGroupBox("处理日志")
        ll = QVBoxLayout(lg)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        ll.addWidget(self.log_text)
        layout.addWidget(lg)

        # === 按钮 ===
        bt = QHBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.start_btn.clicked.connect(self.start_processing)
        bt.addWidget(self.start_btn)
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        bt.addWidget(self.pause_btn)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        bt.addWidget(self.cancel_btn)
        layout.addLayout(bt)

        self.worker = None
        self.total_files = 0
        self.processed_files = 0
        self._speed_timer = QTimer()
        self._speed_timer.timeout.connect(self._update_speed)
        self._safety_timer = QTimer()
        self._safety_timer.timeout.connect(self._safety_timeout)
        self._safety_timer.setSingleShot(True)
        self._job_start_time = 0
        self._job_completed_pages = 0
        self._last_speed_pages = 0
        self._last_speed_time = 0
        self.setAcceptDrops(True)
        self.file_list.setAcceptDrops(True)
        self._update_model_combo()
        self._update_lang_combo()
        self._update_mode_combo()
        self._populate_gpu_combo()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith('.pdf'):
                self._add_file(path)
            elif os.path.isdir(path):
                self._add_pdf_from_folder(path)

    def _add_file(self, path):
        for i in range(self.file_list.count()):
            if self.file_list.item(i).data(Qt.UserRole) == path:
                return
        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        self.file_list.addItem(item)

    def _add_pdf_from_folder(self, folder):
        for root, dirs, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith('.pdf'):
                    full = os.path.join(root, f)
                    self._add_file(full)

    def _update_model_combo(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        # ncnn_vulkan 模型检测
        options = _get_ncnn_model_options("ncnn_vulkan")
        if options:
            for value, label, has_bin in options:
                idx = self.model_combo.count()
                self.model_combo.addItem(label, value)
                desc = _NCNN_MODEL_DESC.get(value, "")
                tip = desc
                if not has_bin:
                    tip += " [模型文件不完整，无法使用]"
                self.model_combo.setItemData(idx, tip, Qt.ToolTipRole)
        else:
            self.model_combo.addItem("无可用模型", "")
        last_model = self.cfg.value("model_val", "")
        idx = self.model_combo.findData(last_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

    def _update_mode_combo(self):
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItem("自动(推荐)", "auto")
        self.mode_combo.addItem("GPU模式", "gpu")
        self.mode_combo.addItem("CPU模式", "cpu")
        idx0 = self.mode_combo.findData("auto")
        if idx0 >= 0:
            tip = ("自动:\n"
                   "检测到独显时使用独显（支持NVIDIA/AMD/Intel），否则CPU\n"
                   "CathayOCR Lite 双实例可提升35%+速度")
            self.mode_combo.setItemData(idx0, tip, Qt.ToolTipRole)
        idx1 = self.mode_combo.findData("gpu")
        if idx1 >= 0:
            tip = "GPU模式:\n强制使用已选Vulkan GPU设备 (ncnn Vulkan)"
            self.mode_combo.setItemData(idx1, tip, Qt.ToolTipRole)
        idx2 = self.mode_combo.findData("cpu")
        if idx2 >= 0:
            self.mode_combo.setItemData(idx2,
                "CPU模式:\n仅使用CPU推理 (ncnn原生)，不加载GPU模块",
                Qt.ToolTipRole)
        last_mode = self.cfg.value("mode_val", "auto")
        idx = self.mode_combo.findData(last_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        self.mode_combo.blockSignals(False)

    def _on_engine_changed(self):
        pass

    def _on_mode_changed(self):
        """模式切换时处理逻辑"""
        mode = self.mode_combo.currentData()
        is_cpu = mode == "cpu"
        if is_cpu:
            self.dual_check.setChecked(False)
        self.dual_check.setEnabled(not is_cpu)
        self.dual_check.setToolTip(
            "双实例并行 (GPU推荐开启):\n"
            "  启动两个OCR进程并行处理一页PDF\n"
            "  可提升GPU利用率30%~50%\n"
            "  双实例会增加约1GB显存占用" if not is_cpu
            else "❌ CPU模式下双实例无意义，已自动关闭"
        )

    def _auto_detect_gpu_idx(self):
        """自动检测显卡，返回最佳索引: 0=大显存独显, 1=小显存独显, 2=核显, 3=纯CPU"""
        try:
            devices = _detect_vulkan_gpus()
            if devices:
                for dev in devices:
                    if dev.get("dedicated"):
                        vram = _get_gpu_vram_mb()
                        if vram is not None and vram > 8000:
                            return 0  # >=12GB
                        return 1  # <=8GB
                # 仅有核显
                return 2
        except Exception:
            pass
        # 无GPU
        return 3

    def _on_ui_mode_changed(self, simple_mode):
        """切换简单模式/专业模式"""
        is_simple = self.ui_simple_btn.isChecked()
        self.simple_group.setVisible(is_simple)
        for w in self._expert_groups:
            w.setVisible(not is_simple)
        if is_simple:
            self._sync_simple_from_expert()

    def _sync_simple_from_expert(self):
        """专业模式切到简单模式时，将当前参数映射到简单模式下拉"""
        # 拦截信号，避免每个 setCurrentIndex 都触发 _apply_simple_settings
        self.simple_gpu.blockSignals(True)
        self.simple_speed.blockSignals(True)
        self.simple_doc.blockSignals(True)
        self.simple_lang.blockSignals(True)

        mode = self.mode_combo.currentData()
        # 显卡：根据mode推断
        if mode == "cpu":
            self.simple_gpu.setCurrentIndex(3)  # 纯CPU
        else:
            self.simple_gpu.setCurrentIndex(0)  # 独显
        # 精度：根据模型判断
        model = self.model_combo.currentText()
        if "small" in model or "tiny" in model:
            self.simple_speed.setCurrentIndex(0)  # 速度优先
        else:
            self.simple_speed.setCurrentIndex(1)  # 标准平衡
        # 文档：根据参数判断
        side = self.side_len_spin.value()
        scale = self.scale_combo.currentIndex()
        vertical = self.vertical_check.isChecked()
        angle = self.angle_cls_check.isChecked()
        if vertical or (side >= 2800 and scale == 2):
            self.simple_doc.setCurrentIndex(1)  # 古籍竖排
        elif angle:
            self.simple_doc.setCurrentIndex(2)  # 扫描件
        else:
            self.simple_doc.setCurrentIndex(0)  # 普通文档
        # 语言：直接同步（拉框内容一致）
        lang_text = self.lang_combo.currentText()
        lang_idx = self.simple_lang.findText(lang_text)
        if lang_idx >= 0:
            self.simple_lang.setCurrentIndex(lang_idx)
        else:
            self.simple_lang.setCurrentIndex(0)

        self.simple_gpu.blockSignals(False)
        self.simple_speed.blockSignals(False)
        self.simple_doc.blockSignals(False)
        self.simple_lang.blockSignals(False)

        # 刷新 GPU 悬停提示（信号被阻断后需要手动调用）
        gpu_tips = [
            "显存充裕，可处理高分辨率古籍/扫描件；古籍类默认2880边长",
            "<=8GB显存，古籍竖排自动降为2240边长避免爆显存",
            "集成显卡性能有限，推荐不高于2000边长",
            "纯CPU运行，兼容性最好但速度最慢",
        ]
        self.simple_gpu.setToolTip(gpu_tips[self.simple_gpu.currentIndex()])

        self._apply_simple_settings()

    def _apply_simple_settings(self):
        """将简单模式的3个选择应用到实际参数"""
        doc_idx = self.simple_doc.currentIndex()
        speed_idx = self.simple_speed.currentIndex()
        gpu_idx = self.simple_gpu.currentIndex()

        # ── 文档类型→参数映射 ──
        if speed_idx == 0:  # 速度优先
            target_model = "small"
            target_precision = "fp16"
        elif speed_idx == 1:  # 标准平衡
            target_model = "medium"
            target_precision = "fp32"
        else:  # 精度优先
            target_model = "medium"
            target_precision = "fp32"

        if doc_idx == 0:  # 普通文档
            target_side = 2000
            target_scale = 1  # 2x
            target_vertical = False
            target_angle = False
        elif doc_idx == 1:  # 古籍竖排
            adjusted = _get_side_for_vram(doc_idx, gpu_idx)
            target_side = adjusted if adjusted is not None else 2880
            target_scale = 2  # 3x
            target_vertical = True
            target_angle = False
        else:  # 扫描件
            adjusted = _get_side_for_vram(doc_idx, gpu_idx)
            target_side = adjusted if adjusted is not None else 2400
            target_scale = 1  # 2x
            target_vertical = False
            target_angle = True

        # ── GPU→mode映射 ──
        if gpu_idx == 3:  # 纯CPU
            target_mode = "cpu"
        else:
            target_mode = "auto"

        # ── 批量设置参数 ──
        self.mode_combo.blockSignals(True)
        self.model_combo.blockSignals(True)
        self.side_len_spin.blockSignals(True)
        self.scale_combo.blockSignals(True)
        self.vertical_check.blockSignals(True)
        self.angle_cls_check.blockSignals(True)
        self.dual_check.blockSignals(True)
        self.lang_combo.blockSignals(True)

        target_dual = (target_mode != "cpu")
        midx = self.mode_combo.findData(target_mode)
        if midx >= 0:
            self.mode_combo.setCurrentIndex(midx)
        midx2 = self.model_combo.findText(target_model, Qt.MatchContains)
        if midx2 >= 0:
            self.model_combo.setCurrentIndex(midx2)
        self.side_len_spin.setValue(target_side)
        self.scale_combo.setCurrentIndex(target_scale)
        self.vertical_check.setChecked(target_vertical)
        self.angle_cls_check.setChecked(target_angle)
        self.dual_check.setChecked(target_dual)
        # 语言同步
        lang_text = self.simple_lang.currentText()
        lang_idx = self.lang_combo.findText(lang_text)
        if lang_idx >= 0:
            self.lang_combo.setCurrentIndex(lang_idx)

        self.mode_combo.blockSignals(False)
        self.model_combo.blockSignals(False)
        self.side_len_spin.blockSignals(False)
        self.scale_combo.blockSignals(False)
        self.vertical_check.blockSignals(False)
        self.angle_cls_check.blockSignals(False)
        self.dual_check.blockSignals(False)
        self.lang_combo.blockSignals(False)

        # ── 更新配置摘要 ──
        model_label = target_model
        gpu_text = "(GPU)" if target_mode != "cpu" else "(CPU)"
        scale_label = ["1x", "2x", "3x"][target_scale]
        dual_label = "开" if target_dual else "关"
        precision_label = target_precision.upper()

        preview_parts = [f"CathayOCR Lite {gpu_text}", f"模型 {model_label}",
                        f"边长{target_side}", f"渲染{scale_label}",
                        f"精度{precision_label}", f"双实例{dual_label}"]
        if target_vertical:
            preview_parts.append("竖排开")
        if target_angle:
            preview_parts.append("方向矫正")
        preview_text = " · ".join(preview_parts)
        self.simple_preview.setText(f"当前配置：{preview_text}")

    def _on_lang_changed(self):
        eid = self.engine_combo.currentData()
        if not eid:
            return
        self.cfg.setValue("lang_val", self.lang_combo.currentText())

    def _update_lang_combo(self):
        self.lang_combo.blockSignals(True)
        self.lang_combo.clear()
        # 直接从 _LANG_ITEMS 填充（ncnn PP-OCRv6 通用字典覆盖所有语种）
        for label, code in self._LANG_ITEMS:
            self.lang_combo.addItem(label, code)
        last_lang = self.cfg.value("lang_val", "中文 (Chinese)")
        idx = self.lang_combo.findText(last_lang)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        self.lang_combo.blockSignals(False)

    def _populate_gpu_combo(self):
        """填充GPU设备下拉框"""
        self.gpu_combo.blockSignals(True)
        self.gpu_combo.clear()
        devices = get_gpu_devices_for_ui()
        if not devices:
            self.gpu_combo.addItem("无检测到GPU", -1)
            self.gpu_combo.blockSignals(False)
            return
        # 自动选项
        auto_idx, auto_name = _select_best_gpu()
        best_name = auto_name if auto_idx >= 0 else "未知"
        self.gpu_combo.addItem(f"自动 (优先 {best_name})", -1)
        for d in devices:
            is_compat = d.get("supported", True)
            compat_flag = " ✅" if is_compat else " ❌"
            gpu_type = "🖥️" if d.get("dedicated") else "💻"
            label = f"{gpu_type} [{d['index']}] {d['name']} (score:{d['score']}){compat_flag}"
            idx = self.gpu_combo.count()
            self.gpu_combo.addItem(label, d['index'])
            tip = d['name'] + (" (独立显卡)" if d.get("dedicated") else " (集成显卡)")
            if not is_compat:
                tip += "\n低分GPU，走CPU可能更快"
            else:
                tip += "\n✓ 可GPU加速"
            self.gpu_combo.setItemData(idx, tip, Qt.ToolTipRole)
        # 选中自动
        idx = self.gpu_combo.findData(-1)
        if idx >= 0:
            self.gpu_combo.setCurrentIndex(idx)
        self.gpu_combo.blockSignals(False)

    def get_selected_engine_id(self):
        return self.engine_combo.currentData()
    def get_use_gpu(self):
        mode = self.mode_combo.currentData()
        if mode == "cpu":
            return False
        # auto/gpu mode: ncnn Vulkan 自动模式下，仅当有高分独显时才走 GPU
        if mode == "auto":
            gpu_devices = _detect_vulkan_gpus()
            capable = [d for d in gpu_devices if d.get("supported") and d.get("dedicated")]
            if capable:
                return True
            # 没有合适的独显 → CPU 模式
            return False
        # gpu 模式：强制 GPU（用户手动选的）
        return True

    def add_files(self):
        sd = self._last_input_dir or os.path.expanduser("~")
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择PDF文件", sd, "PDF文件 (*.pdf);;所有文件 (*)")
        if paths:
            self._last_input_dir = os.path.dirname(paths[0])
            self._auto_set_output(os.path.dirname(paths[0]))
        for path in paths:
            self._add_file(path)
        self._update_count()

    def add_folder(self):
        sd = self._last_input_dir or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "选择包含PDF文件的文件夹", sd)
        if folder:
            self._last_input_dir = folder
            self._auto_set_output(folder)
            count_before = self.file_list.count()
            self._add_pdf_from_folder(folder)
            added = self.file_list.count() - count_before
            self._update_count()
            self.log(f"从文件夹添加了 {added} 个PDF文件")

    def add_multiple_folders(self):
        sd = self._last_input_dir or os.path.expanduser("~")
        dlg = QFileDialog(self, "选择多个文件夹(递归遍历)", sd)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        lv = dlg.findChild(QListWidget, "listView")
        if lv:
            lv.setSelectionMode(QListWidget.MultiSelection)
        tv = dlg.findChild(QTreeView)
        if tv:
            tv.setSelectionMode(QTreeView.MultiSelection)
        if dlg.exec_() == QFileDialog.Accepted:
            folders = dlg.selectedFiles()
            if folders:
                self._last_input_dir = folders[0]
                self._auto_set_output(folders[0])
            count_before = self.file_list.count()
            for folder in folders:
                self._add_pdf_from_folder(folder)
            added = self.file_list.count() - count_before
            self._update_count()
            self.log(f"从多个文件夹添加了 {added} 个PDF文件")

    def _auto_set_output(self, src_dir):
        # 留空=每个文件输出到自己的源目录，不自动填充
        pass

    def _file_exists(self, path):
        for i in range(self.file_list.count()):
            if self.file_list.item(i).data(Qt.UserRole) == path:
                return True
        return False

    def clear_files(self):
        self.file_list.clear()
        self._update_count()

    def _update_count(self):
        self.status_label.setText(f"已选择 {self.file_list.count()} 个文件")
    def browse_output(self):
        sd = self.output_edit.text().strip() or self._last_output_dir or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", sd)
        if d:
            self.output_edit.setText(d)
            self._last_output_dir = d
    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _save_settings(self):
        self.cfg.setValue("side_len", self.side_len_spin.value())
        self.cfg.setValue("scale", self.scale_combo.currentIndex())
        self.cfg.setValue("vertical", self.vertical_check.isChecked())
        self.cfg.setValue("angle_cls", self.angle_cls_check.isChecked())
        self.cfg.setValue("rec_batch", self.rec_batch_spin.value())
        self.cfg.setValue("shrink", self.shrink_check.isChecked())
        self.cfg.setValue("tensorrt", self.tensorrt_check.isChecked())
        self.cfg.setValue("dual", self.dual_check.isChecked())
        self.cfg.setValue("precision_idx", self.precision_combo.currentIndex())
        self.cfg.setValue("engine_id", self.engine_combo.currentData())
        self.cfg.setValue("model_val", self.model_combo.currentData() or "")
        self.cfg.setValue("mode_val", self.mode_combo.currentData() or "auto")
        self.cfg.setValue("gpu_device", self.gpu_combo.currentData() if self.gpu_combo.isVisible() else -2)
        out_dir = self.output_edit.text().strip()
        if out_dir:
            self.cfg.setValue("last_output_dir", out_dir)
        if self._last_input_dir:
            self.cfg.setValue("last_input_dir", self._last_input_dir)

    def _restore_engine_settings(self):
        # 单一 ncnn_vulkan 引擎，无需 restore
        pass

    def start_processing(self):
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "警告", "请添加要处理的PDF文件")
            return
        self._save_settings()
        file_list = [self.file_list.item(i).data(Qt.UserRole) for i in range(self.file_list.count())]
        output_dir = self.output_edit.text().strip() or None  # None = 每个文件输出到自己的源目录
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                QMessageBox.warning(self, "警告", f"无法创建输出目录: {e}")
                return
        engine_id = self.get_selected_engine_id()
        use_gpu = self.get_use_gpu()
        vertical_text = self.vertical_check.isChecked()
        limit_side_len = self.side_len_spin.value()
        model_size = self.model_combo.currentData() or "medium"
        lang_map = dict(self._LANG_ITEMS)
        lang_display = self.lang_combo.currentText() or "中文 (Chinese)"
        ocr_lang = lang_map.get(lang_display, "chinese")
        use_angle_cls = self.angle_cls_check.isChecked()
        scale = self.scale_combo.currentIndex() + 1
        extra_params = {}
        extra_params["enable_fp16"] = (self.precision_combo.currentData() == "fp16")
        extra_params["gpu_device"] = self.gpu_combo.currentData()
        extra_params["use_gpu"] = use_gpu
        extra_params["lang"] = ocr_lang
        dual_instance = self.dual_check.isChecked()
        self.total_files = len(file_list)
        self.processed_files = 0
        self.overall_progress.setValue(0)
        mode_str = "GPU" if use_gpu else "CPU"
        self.log(f"开始处理 {self.total_files} 个文件")
        self.log(f"  引擎: CathayOCR Lite ({mode_str})")
        self.log(f"  模型: {model_size} | 边长: {limit_side_len} | 渲染: {scale}x")
        self.worker = BatchWorkerThread(
            file_list, output_dir,
            engine_id=engine_id, use_gpu=use_gpu,
            vertical_text=vertical_text,
            limit_side_len=limit_side_len,
            model_size=model_size,
            use_angle_cls=use_angle_cls,
            scale=scale,
            dual_instance=dual_instance,
            extra_params=extra_params,
        )
        self.worker.file_progress.connect(self._on_progress)
        self.worker.file_finished.connect(self._on_finished)
        self.worker.file_error.connect(self._on_error)
        self.worker.file_cancelled.connect(self._on_cancelled)
        self.worker.all_finished.connect(self._on_all_done)
        self.worker.start()
        self._set_buttons_processing()
        self._job_start_time = time.time()
        self._job_completed_pages = 0
        self._last_speed_pages = 0
        self._last_speed_time = self._job_start_time
        self._speed_timer.start(5000)
        self._safety_timer.start(600000)

    def toggle_pause(self):
        if not self.worker:
            return
        if self.worker.is_paused:
            self.worker.resume()
            self.pause_btn.setText("暂停")
            self.pause_label.setText("")
            self.log("继续处理")
        else:
            self.worker.pause()
            self.pause_btn.setText("继续")
            self.pause_label.setText(chr(9208) + " 已暂停")
            self.log("暂停处理")
    def cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self.log("正在取消...")
            self.cancel_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.status_label.setText("正在取消...")
    def _on_progress(self, filename, ocr_completed, total):
        self.current_file_label.setText(f"当前文件: {filename}")
        pct = int((ocr_completed / total) * 100) if total > 0 else 0
        self.page_info_label.setText(f"OCR完成 {ocr_completed} / {total} 页 ({pct}%)")
        self._job_completed_pages = ocr_completed
        self._safety_timer.start(600000)
        if total > 0:
            self.page_progress.setValue(int((ocr_completed / total) * 10000))
            ov = int(((self.processed_files + ocr_completed / total) / self.total_files) * 10000)
            self.overall_progress.setValue(int(ov))
    def _on_finished(self, filename, pdf_path, txt_path):
        self.processed_files += 1
        ov = int((self.processed_files / self.total_files) * 10000)
        self.overall_progress.setValue(ov)
        self.log(chr(10003) + f" 完成: {filename}")
    def _on_error(self, filename, err):
        self.processed_files += 1
        ov = int((self.processed_files / self.total_files) * 10000)
        self.overall_progress.setValue(ov)
        self.log(chr(10007) + f" 错误 [{filename}]: {err}")
    def _on_cancelled(self, filename):
        self.log(f"已取消: {filename}")
    def _on_all_done(self, total, success, cancelled):
        self._safety_timer.stop()
        self._speed_timer.stop()
        self.log(f"\n处理完成! 成功: {success}/{total}")
        self._set_buttons_idle()
        self.status_label.setText(f"完成 - 成功 {success}/{total}")
        self.overall_progress.setValue(10000)
        self.speed_label.setText("处理速度: -- (已完成)")
        QMessageBox.information(self, "完成", f"批量处理完成!\n\n成功: {success}/{total} 个文件\n输出目录: {self.output_edit.text()}")
    def _set_buttons_processing(self):
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.add_files_btn.setEnabled(False)
        self.add_folder_btn.setEnabled(False)
        self.clear_files_btn.setEnabled(False)
        self.engine_combo.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.gpu_combo.setEnabled(False)
        self.side_len_spin.setEnabled(False)
        self.scale_combo.setEnabled(False)
        self.vertical_check.setEnabled(False)
        self.angle_cls_check.setEnabled(False)
        self.precision_combo.setEnabled(False)
        self.dual_check.setEnabled(False)
        self.status_label.setText("处理中...")
    def _set_buttons_idle(self):
        self._safety_timer.stop()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        self.cancel_btn.setEnabled(False)
        self.add_files_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        self.clear_files_btn.setEnabled(True)
        self.engine_combo.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.gpu_combo.setEnabled(True)
        self.side_len_spin.setEnabled(True)
        self.scale_combo.setEnabled(True)
        self.vertical_check.setEnabled(True)
        self.angle_cls_check.setEnabled(True)
        self.precision_combo.setEnabled(True)
        self.dual_check.setEnabled(True)
        self.pause_label.setText("")
    def _safety_timeout(self):
        self._set_buttons_idle()
        self.log("超时保护:已自动恢复控件")
    def _update_speed(self):
        now = time.time()
        since_last = now - self._last_speed_time
        if since_last < 5:
            return
        delta = self._job_completed_pages - self._last_speed_pages
        if delta <= 0:
            self.speed_label.setText("处理速度: --")
            return
        pps = delta / since_last
        self.speed_label.setText(f"处理速度: {pps:.2f} 页/秒")
        self._last_speed_pages = self._job_completed_pages
        self._last_speed_time = now
        self._update_gpu()
    def _update_gpu(self):
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                used, total = result.stdout.strip().split(",")
                used_mb = int(used.strip())
                total_mb = int(total.strip())
                pct = used_mb / total_mb * 100
                self.gpu_label.setText(f"GPU: {used_mb}MB / {total_mb}MB ({pct:.0f}%)")
        except Exception:
            pass

    def closeEvent(self, event):
        """窗口关闭时清理所有子进程"""
        print("[MainWindow] Cleaning up OCR instances...")
        try:
            ocr = OCRClient()
            ocr.close()
            # 重置单例，确保下次启动全新
            OCRClient._instance = None
        except Exception as e:
            print(f"[MainWindow] OCR cleanup: {e}")
        # 强制杀死所有残余子进程
        self._kill_orphans()
        event.accept()

    def _kill_orphans(self):
        """强制杀死所有相关子进程（os.system 在退出时更可靠）"""
        for name in ["ppocr_ocr_vulkan.exe"]:
            os.system(f'taskkill /f /im {name} >nul 2>&1')


CLEANUP_TARGETS = ["ppocr_ocr_vulkan.exe"]


def _force_cleanup():
    """atexit 强制清理"""
    for name in CLEANUP_TARGETS:
        os.system(f'taskkill /f /im {name} >nul 2>&1')


if __name__ == '__main__':
    # 启动前先杀死所有残余进程
    for name in CLEANUP_TARGETS:
        os.system(f'taskkill /f /im {name} >nul 2>&1')
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window._restore_engine_settings()
    window.show()
    sys.exit(app.exec_())
