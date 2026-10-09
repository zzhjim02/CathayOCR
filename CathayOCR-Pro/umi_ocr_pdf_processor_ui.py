"""
CathayOCR Pro (专业版) v1.3.0 - 多引擎GPU加速PDF处理器
Architecture: 预渲染所有页面到RAM -> 单实例OCR流水线 -> 组装输出
核心思想: GPU永不等待,CPU预渲染消除I/O瓶颈
=======================================================
支持的OCR引擎:
  1. PP-OCR (ncnn Vulkan) - ⭐首选推荐 (最快速度,支持任意显卡+CPU)
  2. PP-OCRv6 (ONNX CUDA) - 主力推荐 (高精度)
  3. PP-OCRv5 (Paddle CPU) - 经典CPU方案
  4. PP-OCR (ncnn CPU) - 纯CPU方案
  5. PP-OCR (Paddle CPU) - 经典CPU方案
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
import collections

# Fix Qt platform plugin path for portable Python
try:
    import PyQt5
    _pyqt_dir = os.path.dirname(PyQt5.__file__)
    _qt_bin = os.path.join(_pyqt_dir, 'Qt5', 'bin')
    _qt_plugins = os.path.join(_pyqt_dir, 'Qt5', 'plugins', 'platforms')
    if os.path.isdir(_qt_bin):
        os.environ['PATH'] = _qt_bin + os.pathsep + os.environ.get('PATH', '')
    if os.path.isdir(_qt_plugins):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = _qt_plugins
except Exception:
    pass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QProgressBar, QGroupBox, QMessageBox, QSpinBox, QComboBox, QCheckBox,
    QRadioButton, QButtonGroup, QListWidget, QListWidgetItem, QTreeView,
    QTextBrowser, QFrame, QGridLayout, QShortcut,
    QSystemTrayIcon, QMenu, QAction, QStyle
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSettings, QEvent
from PyQt5.QtGui import QFont, QKeySequence, QBrush, QColor, QIcon

import fitz
# 压制 MuPDF 的 PDF 结构语法警告（不影响识别结果）
import os, sys
import contextlib

# 【修复】改用 PyMuPDF 官方开关一次性关闭 MuPDF 错误输出。
# 旧实现用 os.dup2 重定向整个 stderr：渲染是多线程的，并发时各线程会互相
# 覆盖/还原文件描述符，一旦交错，之后的错误输出就会被永久吞掉。
# 官方开关是进程级、线程安全、不碰文件描述符，真正的 Python/引擎错误照常显示。
try:
    if hasattr(fitz, 'TOOLS') and hasattr(fitz.TOOLS, 'mupdf_display_errors'):
        fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass


@contextlib.contextmanager
def _suppress_mupdf_warnings():
    """保留接口（所有调用点不变）。实际静音已由上方官方开关完成。"""
    yield

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


def engine_display_name(engine_id, fallback=None):
    """引擎内部 id（如 win7_v5）→ 界面统一显示名。

    软件里凡是给用户看的「引擎名字」一律走这里取名（引擎下拉框 / 处理日志 /
    提示弹窗 / 工具提示 / 简单模式配置摘要），保证同一个引擎在任何位置
    都叫同一个名字。曾出现「win7_v5」与「PP-OCRv5 (Paddle CPU)」两种叫法混用。
    """
    info = ENGINE_REGISTRY.get(engine_id)
    if info and info.get("name"):
        return info["name"]
    return fallback if fallback is not None else engine_id


register_engine(
    'umi_plugin_v6', 'PP-OCRv6 (ONNX CUDA)',
    'PP-OCRv6 主力推荐引擎\n支持NVIDIA GPU加速(CUDA)\n精度高速度快\n需NVIDIA显卡+安装CUDA\n配合下方语言选择可识别中/英/法/德/日',
    'umi_plugin_v6', 'PaddleOCR-json.bat', 'pipe',
    supports_gpu=True, supports_cpu=True,
    model_options=[('medium', '高精度 (Medium)'), ('small', '快速 (Small)')],
    supported_params=['vertical_text', 'cls', 'det', 'rec_batch_num',
                      'shrink_poly_ratio', 'blank_page_strategy'],
    priority=10
)


register_engine(
    'easyocr_universal',
    'EasyOCR (拉丁语系)',
    'EasyOCR 拉丁语系引擎\n仅支持英/法/意/西四种拉丁语系语言\n基于PyTorch，**仅 CPU 运行**\n（本包内置 CPU 版 PyTorch，无 CUDA，不提供 GPU 加速）',
    'easyocr_universal',   # plugin_rel_path
    'EasyOCR-Universal.bat',
    'pipe',                   # 启动方式: 管道
    supports_gpu=False,
    supports_cpu=True,
    model_options=[('universal', '自动 (按语言选择)')],
    supported_params=[],
    priority=6
)


register_engine(
    'win7_v5', 'PP-OCRv5 (Paddle CPU)',
    '备选引擎：其他引擎不可用时使用\n纯CPU运行（无GPU加速）\n基于Paddle Inference + MKL-DNN\n内置官方分语种识别模型\n覆盖全部支持语种\n（阿拉伯/天城文/泰/希腊/韩/俄/拉丁等）',
    'win7_x64_PaddleOCR-json_PP-OCRv5', 'PaddleOCR-json.exe', 'pipe',
    supports_gpu=False, supports_cpu=True,
    model_options=[],
    supported_params=['enable_mkldnn', 'cls', 'cpu_threads'],
    priority=9
)

register_engine(
    'win7_classic', 'PP-OCRv3 (Paddle CPU)',
    '经典PaddleOCR引擎\n纯CPU运行，兼容性最好\n基于Paddle Inference\nMKL-DNN加速\n适合无GPU的老电脑',
    'win7_x64_PaddleOCR-json', 'PaddleOCR-json.exe', 'pipe',
    supports_gpu=False, supports_cpu=True,
    model_options=[],
    supported_params=['enable_mkldnn', 'cls', 'cpu_threads'],
    priority=3
)

register_engine(
    'ncnn_vulkan', 'PP-OCR (ncnn Vulkan)',
    '\u2b50 速度最快，强力推荐！\n支持NVIDIA/AMD/Intel任意显卡\n也支持CPU模式（没显卡也能用）\n基于ncnn框架+Vulkan\nGPU模式需Vulkan驱动\n支持多版本模型（v3~v6）\n配合下方语言选择可识别中/英/法/德/日',
    'paddle-ocr-ncnn-cpp_plugin-master/PPOCR-ncnn-Vulkan',
    'ppocr_ocr_vulkan.exe', 'ncnn_vulkan',
    supports_gpu=True, supports_cpu=True,
    model_options=[],  # 运行时动态检测可用模型
    supported_params=['num_threads', 'enable_fp16', 'det_thres',
                      'unclip_ratio', 'enable_cls', 'gpu_device'],
    priority=20
)

register_engine(
    'ncnn_cpu', 'PP-OCR (ncnn CPU)',
    '纯CPU推理引擎\n不需要显卡，兼容性最好\n速度较GPU慢但精度不减\n适合没有NVIDIA显卡的用户',
    'paddle-ocr-ncnn-cpp_plugin-master/PPOCR-ncnn-CPU',
    'ppocr_ocr_cpu.exe', 'ncnn_cpu',
    supports_gpu=False, supports_cpu=True,
    model_options=[],  # 运行时动态检测可用模型
    supported_params=['num_threads', 'enable_fp16', 'det_thres',
                      'unclip_ratio', 'enable_cls'],
    priority=4
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
                "umi_plugin_v6": "ppocr_v6",
                "win7_x64_PaddleOCR-json": "ppocr_v3",
                "win7_x64_PaddleOCR-json_PP-OCRv5": "ppocr_v5",
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
            print(f"[Plugin] {engine_display_name(eid)}: 未找到入口程序，已跳过")
            continue

        found[eid] = {
            "plugin_dir": str(found_path),
            "entry_path": str(entry_path),
        }
        print(f"[Plugin] 已加载引擎: {engine_display_name(eid)}")

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
    print("[Plugin] 可用引擎: " + ", ".join(engine_display_name(k) for k in _PLUGIN_DIRS))

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


# ============================================================
# ncnn 字典 ↔ 模型 路由（唯一判定处；UI 与两版适配器共用，避免再次分叉）
# ============================================================
# rec 模型的输出维度（_rec.param 末尾 Gemm 8=）必须与所用字典的行数严格一致，
# 不一致时引擎会在 1~2 次请求后永久卡死 / 输出乱码：
#     PP_OCRv3_mobile_rec      6625  →  ppocr_keys_v1.txt        6625
#     PP_OCRv4_mobile_rec      6625  →  ppocr_keys_v1.txt        6625   ← 旧代码错走 v5 字典
#     PP_OCRv5_mobile_rec     18385  →  ppocr_keys_v5.txt       18385
#     PP_OCRv5_server_rec     18385  →  ppocr_keys_v5.txt       18385
#     PP_OCRv6_small_rec      18710  →  ppocr_keys_v6.txt       18710
#     PP_OCRv6_medium_rec     18710  →  ppocr_keys_v6.txt       18710
#     PP_OCRv6_tiny_rec        6906  →  ppocr_keys_v6_tiny.txt   6906   ← 旧代码错走 v6 字典
def _ncnn_keys_file_for_model(model_base):
    m = model_base or ""
    if "v6_tiny" in m:                     # 必须先于 "v6" 判断
        return "ppocr_keys_v6_tiny.txt"
    if "v3" in m or "v4" in m:             # v4 与 v3 同用 v1 字典（输出维度都是 6625）
        return "ppocr_keys_v1.txt"
    if "v6" in m:
        return "ppocr_keys_v6.txt"
    return "ppocr_keys_v5.txt"             # v5_mobile / v5_server 及兜底


# ============================================================
# ncnn「语言 × 当前字典」覆盖判据
# ============================================================
# 语言下拉必须反映「当前模型实际用的那份字典」的真实能力 —— 四份字典差别很大：
#   · 日文假名：v1 字典 4%、v6_tiny 字典 0%（v5/v6 是 100%）
#   · 希腊文　：v1 字典 21%（v5/v6/v6_tiny 是 100%）
#   · 越南文　：四份字典都只有 12%~41%（缺大量声调符号）
# 档位：ok（正常）/ partial（可选中，标注可能缺重音）/ unsupported（灰显，不可选）
_NCNN_PROBE_CJK    = "的一是不了在人有我他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生自"
_NCNN_PROBE_KANA   = ("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
                      "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン")
_NCNN_PROBE_HANGUL = ("가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허"
                      "고노도로모보소오조초코토포호")
_NCNN_PROBE_GREEK  = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
_NCNN_PROBE_CYR    = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯабвгдежзийклмнопрстуфхцчшщыэюя"
_NCNN_PROBE_ARAB   = "ابجدهوزحطيكلمنسعفصقرشتثخذضظغ"
_NCNN_PROBE_DEVA   = "अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह"
_NCNN_PROBE_THAI   = "กขคงจฉชซญฎฏฐณดตถทธนบปผฝพฟมยรลวศษสหฬอฮ"
_NCNN_PROBE_TELUGU = "అఆఇఈఉఊఎఏఐఒఓఔకఖగఘచఛజఝటఠడఢణతథదధనపఫబభమయరలవశషసహ"
_NCNN_PROBE_TAMIL  = "அஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஶஷஸஹ"

# 非拉丁语系：语言 → 其核心文字系探针（核心文字系缺失 = 该语言整个不支持）
_NCNN_SCRIPT_PROBE = {
    "ch": _NCNN_PROBE_CJK, "japan": _NCNN_PROBE_KANA, "korean": _NCNN_PROBE_HANGUL,
    "el": _NCNN_PROBE_GREEK, "th": _NCNN_PROBE_THAI,
    "te": _NCNN_PROBE_TELUGU, "ta": _NCNN_PROBE_TAMIL,
}
_NCNN_SCRIPT_PROBE.update({c: _NCNN_PROBE_CYR for c in (
    "ru", "uk", "be", "bg", "mk", "mn", "kk", "ky", "tg", "tt", "ba", "cv", "rs_cyrillic")})
_NCNN_SCRIPT_PROBE.update({c: _NCNN_PROBE_ARAB for c in (
    "ar", "fa", "ug", "ur", "ps", "sd", "ks", "bal")})
_NCNN_SCRIPT_PROBE.update({c: _NCNN_PROBE_DEVA for c in (
    "hi", "mr", "ne", "sa", "bh", "mai", "kok")})

# 拉丁语系：只记「该语言特有的带变音字母」（a-z 四份字典都有，不参与判定）
_NCNN_LATIN_EXTRA = {
    "fr": "àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ",
    "de": "äöüßÄÖÜ",
    "es": "áéíñóúü¿¡ÁÉÍÑÓÚÜ",
    "pt": "áâãàçéêíóôõúüÁÂÃÀÇÉÊÍÓÔÕÚÜ",
    "it": "àèéìòùÀÈÉÌÒÙ",
    "nl": "àèéëïöüÀÈÉËÏÖÜ",
    "ro": "ăâîșțşţĂÂÎȘȚŞŢ",
    "ca": "àçèéíïòóúüÀÇÈÉÍÏÒÓÚÜ",
    "gl": "áéíóúüñÁÉÍÓÚÜÑ",
    "da": "æøåÆØÅ",
    "sv": "åäöÅÄÖ",
    "no": "æøåÆØÅ",
    "fi": "äöåÄÖÅ",
    "is": "áðéíóúýþæöÁÐÉÍÓÚÝÞÆÖ",
    "pl": "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
    "cs": "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ",
    "sk": "áäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ",
    "hu": "áéíóöőúüűÁÉÍÓÖŐÚÜŰ",
    "hr": "čćđšžČĆĐŠŽ",
    "sl": "čšžČŠŽ",
    "bs": "čćđšžČĆĐŠŽ",
    "rs_latin": "čćđšžČĆĐŠŽ",
    "sq": "ëçËÇ",
    "ga": "áéíóúÁÉÍÓÚ",
    "cy": "ŵŷâêîôûŴŶÂÊÎÔÛ",
    "et": "õäöüšžÕÄÖÜŠŽ",
    "lt": "ąčęėįšųūžĄČĘĖĮŠŲŪŽ",
    "lv": "āčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ",
    "mt": "ċġħżĊĠĦŻ",
    "la": "æœÆŒ",
    "pi": "āīūṭḍṇṃĀĪŪṬḌṆṂ",
    "af": "áéíóúëÁÉÍÓÚË",
    "az": "çğıöşüÇĞİÖŞÜ",
    "uz": "ʻʼ",
    "ku": "çêîşûÇÊÎŞÛ",
    "eu": "ñÑ",
    "oc": "àçèéíòóúÀÇÈÉÍÒÓÚ",
    "vi": ("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
           "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵĂÂĐÊÔƠƯ"),
    "id": "àèéìòùÀÈÉÌÒÙ",
    "ms": "àèéìòùÀÈÉÌÒÙ",
    "tl": "ñáéíóúÑÁÉÍÓÚ",
    "mi": "āēīōūĀĒĪŌŪ",
    "tr": "çğıöşüÇĞİÖŞÜ",
}

_NCNN_DICT_POOL_CACHE = {}


def _load_ncnn_dict_pool(plugin_dir, keys_file):
    """读一份 ncnn 字典 → 字符集合（带缓存）。读不到返回空集，调用方据此不做判定。"""
    key = (plugin_dir, keys_file)
    if key not in _NCNN_DICT_POOL_CACHE:
        pool = set()
        try:
            with open(os.path.join(plugin_dir, "models", keys_file),
                      "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pool.update(line)
        except Exception as e:
            print(f"[ncnn] 读字典失败 {keys_file}: {e}")
        _NCNN_DICT_POOL_CACHE[key] = pool
    return _NCNN_DICT_POOL_CACHE[key]


def _ncnn_lang_tier(code, pool):
    """按「当前模型用的那份字典」判断语言档位：'ok' / 'partial' / 'unsupported'。

      · unsupported —— 核心文字系命中 < 60%（该文字系基本不在字典里，识别必乱码）
      · partial     —— 特有字符命中 < 60%（认得了基础字母，但会缺重音/变音符号）
      · ok          —— 覆盖良好
    """
    if not pool:
        return "ok"                        # 字典没读到 → 不判定，避免误杀
    script = _NCNN_SCRIPT_PROBE.get(code)
    if script is not None:
        u = set(script)
        hit = sum(1 for c in u if c in pool) / len(u)
        return "ok" if hit >= 0.6 else "unsupported"
    extra = _NCNN_LATIN_EXTRA.get(code, "")
    if extra:
        u = set(extra)
        hit = sum(1 for c in u if c in pool) / len(u)
        if hit < 0.6:
            return "partial"
    return "ok"


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
# 引擎适配器抽象基类
# ============================================================

class OCREngineAdapter(ABC):
    """所有OCR引擎的通用接口"""

    def __init__(self, engine_id, plugin_dir, entry_path):
        self.engine_id = engine_id
        self.plugin_dir = plugin_dir
        self.entry_path = entry_path
        self._ocr_times = collections.deque(maxlen=20)  # 最近OCR耗时(秒)，用于自适应超时（所有引擎通用）

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

    # ---------- 看门狗 / 自适应超时通用接口（所有引擎适配器均可用） ----------
    def get_dynamic_timeout(self, floor=180, ceiling=300):
        """根据所有活跃引擎实例的历史OCR耗时计算动态超时（通用实现）
        - 默认 180s，慢设备自动升到 300s
        - 样本不足 3 个时返回默认值（floor）
        """
        all_times = list(getattr(self, "_ocr_times", None) or [])
        try:
            client = globals().get("OCRClient")
            insts = getattr(client, "_instances", None) if client is not None else None
            for inst in (insts or []):
                if inst is not self:
                    all_times.extend(getattr(inst, "_ocr_times", None) or [])
        except Exception:
            pass
        if len(all_times) < 3:
            return floor
        sorted_times = sorted(all_times)
        p90_idx = int(len(sorted_times) * 0.9)
        if p90_idx >= len(sorted_times):
            p90_idx = len(sorted_times) - 1
        p90 = sorted_times[p90_idx]
        upgrade = p90 * 3
        if upgrade <= floor:
            return floor
        return int(min(upgrade, ceiling))

    def restart(self, params=None):
        """强制重启引擎（通用实现：stop + start）"""
        try:
            self.stop()
        except Exception:
            pass
        time.sleep(0.5)
        if params is None:
            params = getattr(self, "current_config", None)
        if params is None:
            return ""
        try:
            return self.start(params)
        except Exception as e:
            return f"[Error] restart failed: {e}"

class PaddlePipeAdapter(OCREngineAdapter):
    """基于 PaddleOCR-json 管道通信的引擎适配器"""

    def __init__(self, engine_id, plugin_dir, entry_path):
        super().__init__(engine_id, plugin_dir, entry_path)
        self.pipe = None
        self.startupinfo = None
        self._stderr_lines = []
        self._stderr_thread = None
        if "win32" in str(platform.system()).lower():
            self.startupinfo = subprocess.STARTUPINFO()
            self.startupinfo.dwFlags = (
                subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
            )
            self.startupinfo.wShowWindow = subprocess.SW_HIDE

    def start(self, params):
        self.stop()
        self.current_config = params
        try:
            exe_path = self.entry_path
            cwd = os.path.dirname(exe_path)
            cmds = [exe_path]
            if isinstance(params, dict):
                for key, value in params.items():
                    if isinstance(value, bool):
                        cmds += [f"--{key}={value}"]
                    elif isinstance(value, str):
                        cmds += [f"--{key}", value]
                    else:
                        cmds += [f"--{key}", str(value)]
            self.pipe = subprocess.Popen(
                cmds, cwd=cwd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, startupinfo=self.startupinfo,
            )
            self._stderr_lines = []
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, daemon=True
            )
            self._stderr_thread.start()
            while True:
                if self.pipe.poll() is not None:
                    self._stderr_thread.join(timeout=1.0)
                    err_msg = "".join(self._stderr_lines).strip()
                    return f"OCR init fail. stderr: {err_msg}"
                init_str = self.pipe.stdout.readline().decode("utf-8", errors="ignore")
                if "OCR init completed." in init_str:
                    break
            return ""
        except Exception as e:
            return f"[Error] Engine start failed: {str(e)}"

    def _drain_stderr(self):
        try:
            for line in iter(self.pipe.stderr.readline, b""):
                self._stderr_lines.append(line.decode("utf-8", errors="ignore"))
                if len(self._stderr_lines) > 50:
                    self._stderr_lines.pop(0)
        except Exception:
            pass

    def _run_dict(self, write_dict, timeout=180):
        if not self.pipe:
            return {"code": 901, "data": "引擎未启动"}
        if self.pipe.poll() is not None:
            err_msg = "".join(self._stderr_lines).strip()
            return {"code": 902, "data": f"子进程已崩溃。stderr: {err_msg}"}
        write_str = json.dumps(write_dict, ensure_ascii=True, indent=None) + "\n"
        t_ocr_start = time.time()
        try:
            self.pipe.stdin.write(write_str.encode("utf-8"))
            self.pipe.stdin.flush()
        except Exception as e:
            return {"code": 902, "data": f"向识别器进程传入指令失败。{e}"}
        result = {"data": None, "error": None}
        def read_thread():
            try:
                get_str = self.pipe.stdout.readline().decode("utf-8", errors="ignore")
                result["data"] = get_str
            except Exception as e:
                result["error"] = str(e)
        thread = threading.Thread(target=read_thread, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            return {"code": 905, "data": f"OCR处理超时({timeout}秒)"}
        if result["error"]:
            return {"code": 903, "data": "读取失败: " + str(result.get("error", ""))}
        if result["data"] is None:
            return {"code": 903, "data": "无返回数据"}
        try:
            _ret = json.loads(result["data"])
        except Exception as e:
            return {"code": 904, "data": f"JSON解析失败: {e}"}
        if isinstance(_ret, dict) and _ret.get("code") == 100:
            self._ocr_times.append(time.time() - t_ocr_start)
        return _ret

    def run_base64(self, image_base64, timeout=180):
        try:
            return self._run_dict({"image_base64": image_base64}, timeout)
        except Exception as e:
            return {"code": 900, "data": f"OCR error: {str(e)}"}

    def run_path(self, img_path, timeout=180):
        try:
            return self._run_dict({"image_path": os.path.abspath(img_path)}, timeout)
        except Exception as e:
            return {"code": 900, "data": f"OCR error: {str(e)}"}

    def stop(self):
        if self.pipe:
            try:
                self.pipe.kill()
            except Exception:
                pass
            self.pipe = None

    def close(self):
        self.stop()

class NcnnCPUAdapter(OCREngineAdapter):
    """基于 ncnn CPU 子进程的引擎适配器"""

    def __init__(self, engine_id, plugin_dir, entry_path):
        super().__init__(engine_id, plugin_dir, entry_path)
        self.config_path = os.path.join(plugin_dir, "config.json")
        self.current_config = {}

    def start(self, params):
        self.current_config = params
        try:
            config = self._build_config(params)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f)
            return ""
        except Exception as e:
            return f"[Error] Config failed: {str(e)}"

    def _build_config(self, params):
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
        available = _get_available_ncnn_models("ncnn_cpu")
        if not available:
            raise RuntimeError("没有找到可用的ncnn CPU模型文件")
        if model_version and model_version in available:
            selected = model_version
        else:
            selected = available[0]
        model_map = {}
        for base in available:
            model_map[base] = (base + "_det", base + "_rec")
        if not model_map:
            raise RuntimeError("没有找到可用的ncnn CPU模型")
        det_model, rec_model = model_map.get(selected, list(model_map.values())[0])
        lang = params.get("lang", "chinese")
        if lang != "chinese":
            print(f"[ncnn] Language: {lang} (model already supports all characters)")
        # 【修复】字典必须跟随「实际选中的模型」selected，而不是传入的字符串。
        # 传入名不在可用列表里时 selected 会回退为 available[0]；若仍按原字符串
        # 选字典，就会出现「v5 字典 + v6 模型」→ 引擎 1~2 次请求后永久卡死。
        # 路由统一交给 _ncnn_keys_file_for_model()（顺带修掉两处错配：
        # PP_OCRv4_mobile 输出维度 6625 应配 v1 字典、PP_OCRv6_tiny 输出 6906 应配 v6_tiny 字典）。
        keys_file = _ncnn_keys_file_for_model(selected)
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

    def _run_exe(self, img_path, timeout=180):
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
            t_ocr_start = time.time()
            stdout, stderr = proc.communicate(input=json_str.encode("utf-8"), timeout=timeout)
            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="ignore") if stderr else "Unknown"
                return {"code": 102, "data": f"Process error (exit {proc.returncode}): {err_msg}"}
            stdout_str = stdout.decode("utf-8", errors="ignore")
            _ret = self._parse_json(stdout_str)
            if isinstance(_ret, dict) and _ret.get("code") == 100:
                self._ocr_times.append(time.time() - t_ocr_start)
            return _ret
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
            return {"code": 102, "data": "No JSON in output"}
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
                return self._run_exe(tmp_path, timeout)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            return {"code": 900, "data": f"Base64 error: {str(e)}"}

    def run_path(self, img_path, timeout=180):
        return self._run_exe(img_path, timeout)

    def stop(self):
        pass

    def close(self):
        self.stop()

class NcnnVulkanAdapter(OCREngineAdapter):
    """基于 ncnn Vulkan TCP 服务器的引擎适配器"""

    def __init__(self, engine_id, plugin_dir, entry_path):
        super().__init__(engine_id, plugin_dir, entry_path)
        self.config_path = os.path.join(plugin_dir, "config.json")
        self.port = 18043
        self.port_offset = 0
        self.server_proc = None
        self.lock = threading.Lock()
        self.current_config = {}
        self._started = False
        self._ocr_times = collections.deque(maxlen=20)  # 最近OCR耗时(秒)，用于自适应超时

    def set_port_offset(self, offset):
        """设置端口偏移，支持多实例并行"""
        self.port_offset = offset

    @property
    def _server_port(self):
        return self.port + self.port_offset

    def start(self, params):
        self.stop()
        self.current_config = params
        use_gpu = params.get("use_gpu", True)
        try:
            config = self._build_config(params)
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
            if use_gpu:
                # 不传 --vulkan 命令行参数，完全依赖 config.json 的 use_vulkan + gpu_device_index
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
        except Exception as e:
            return f"[Error] Vulkan start failed: {str(e)}"

    def _build_config(self, params):
        base_dir = "models"
        model_version = params.get("model_version", "")
        num_threads = params.get("num_threads", -1)
        enable_fp16 = params.get("enable_fp16", False)
        det_thres = params.get("det_thres", 0.5)
        unclip_ratio = params.get("unclip_ratio", 1.58)
        enable_cls = params.get("enable_cls", True)
        max_side_len = params.get("max_side_len", 2000)
        gpu_device = params.get("gpu_device", -1)
        use_gpu = params.get("use_gpu", True)
        if num_threads <= 0:
            cpu_count = os.cpu_count() or 1
            num_threads = min(cpu_count, 6)
        # 获取实际可用模型
        available = _get_available_ncnn_models("ncnn_vulkan")
        if not available:
            raise RuntimeError("没有找到可用的ncnn Vulkan模型文件")
        # 选择用户指定的模型，或第一个可用模型
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
            print(f"[Vulkan] Language: {lang} (PP-OCRv6 dict 覆盖 中/英/日/拉丁/希腊，不含韩/西里尔/阿拉伯等)")
        # 【修复】字典必须跟随「实际选中的模型」selected，而不是传入的字符串。
        # 传入名不在可用列表里时 selected 会回退为 available[0]；若仍按原字符串
        # 选字典，就会出现「v5 字典 + v6 模型」→ 引擎 1~2 次请求后永久卡死。
        # 路由统一交给 _ncnn_keys_file_for_model()（顺带修掉两处错配：
        # PP_OCRv4_mobile 输出维度 6625 应配 v1 字典、PP_OCRv6_tiny 输出 6906 应配 v6_tiny 字典）。
        keys_file = _ncnn_keys_file_for_model(selected)
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
                "use_vulkan": use_gpu,
                "gpu_device_index": gpu_device
            },
            "cls": {
                "infer_threads": min(2, num_threads),
                "reco_threads": num_threads,
                "model_path": f"{base_dir}/PP_LCNet_x0_25_textline_ori",
                "enable": enable_cls,
                "most_angle": True,
                "fp16": enable_fp16,
                "use_vulkan": use_gpu,
                "gpu_device_index": gpu_device
            },
            "rec": {
                "infer_threads": min(4, num_threads),
                "reco_threads": num_threads,
                "model_path": f"{base_dir}/{rec_model}",
                "keys_path": f"{base_dir}/{keys_file}",
                "fp16": enable_fp16,
                "use_vulkan": use_gpu,
                "gpu_device_index": gpu_device
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
        self.start(self.current_config)
        return self._started

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
                t_ocr_start = time.time()
                request = {"img_path": tmp_path.replace("\\", "/")}
                result = self._tcp_request(request, timeout)
                if result.get("code") == 100:
                    self._ocr_times.append(time.time() - t_ocr_start)
                return result
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            return {"code": 900, "data": f"Base64 error: {str(e)}"}

    def run_path(self, img_path, timeout=180):
        request = {"img_path": img_path.replace("\\", "/")}
        return self._tcp_request(request, timeout)

    def stop(self):
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

    def get_dynamic_timeout(self, floor=180, ceiling=300):
        """根据所有活跃引擎实例的历史OCR耗时计算动态超时
        - 默认 180s，慢设备自动升到 300s
        """
        all_times = list(self._ocr_times)
        try:
            client = OCRClient._instance
            if client and hasattr(client, "_instances"):
                for inst in client._instances:
                    if inst is not self:
                        all_times.extend(getattr(inst, "_ocr_times", []))
        except Exception:
            pass
        if len(all_times) < 3:
            return floor
        sorted_times = sorted(all_times)
        p90_idx = int(len(sorted_times) * 0.9)
        p90 = sorted_times[p90_idx]
        upgrade = p90 * 3
        if upgrade <= floor:
            return floor
        return int(min(upgrade, ceiling))

    def restart(self, params=None):
        """强制重启引擎进程"""
        print(f"[NcnnAdapter] Restarting engine on port {self._server_port}...")
        self.stop()
        time.sleep(0.5)
        if params is None:
            params = self.current_config
        return self.start(params)

    def close(self):
        self.stop()

# ============================================================
# 引擎适配器工厂
# ============================================================

def create_engine_adapter(engine_id):
    """创建引擎适配器实例"""
    if engine_id not in _PLUGIN_DIRS:
        raise ValueError("引擎 " + engine_id + " 未找到插件目录")
    info = _PLUGIN_DIRS[engine_id]
    einfo = ENGINE_REGISTRY[engine_id]
    if einfo["entry_type"] == "pipe":
        return PaddlePipeAdapter(engine_id, info["plugin_dir"], info["entry_path"])
    elif einfo["entry_type"] == "ncnn_cpu":
        return NcnnCPUAdapter(engine_id, info["plugin_dir"], info["entry_path"])
    elif einfo["entry_type"] == "ncnn_vulkan":
        return NcnnVulkanAdapter(engine_id, info["plugin_dir"], info["entry_path"])
    else:
        raise ValueError("未知引擎类型: " + str(einfo.get("entry_type", "?")))

# ============================================================
# 引擎参数构建器
# ============================================================

def build_engine_params(engine_id, use_gpu, vertical_text, limit_side_len,
                        model_size_or_version, use_angle_cls, extra_params=None):
    """根据引擎类型和用户设置，构建引擎启动参数字典"""
    if extra_params is None:
        extra_params = {}
    params = {}
    einfo = ENGINE_REGISTRY[engine_id]

    if engine_id == "easyocr_universal":
        # EasyOCR passes language as command-line argument
        params["language"] = extra_params.get("easyocr_lang", "en")

    if einfo["entry_type"] == "pipe":
        params["use_gpu"] = use_gpu
        if engine_id == "easyocr_universal":
            # EasyOCR 仅 CPU：本包内置 torch 是 CPU 版
            # （实测 torch 2.13.0+cpu、torch.version.cuda=None、cuda.is_available()=False、device_count=0），
            # 即便传 --use_gpu=True 也会被 torch 静默忽略而回退 CPU。
            # 这里显式锁 False，杜绝「界面上像是用了GPU、实际在跑CPU」的假象。
            params["use_gpu"] = False
        params["limit_side_len"] = limit_side_len
        params["cls"] = use_angle_cls
        if engine_id.startswith("umi_plugin_v6"):
            lang = extra_params.get("lang", "chinese") or "chinese"
            config_path = f"models/config_{model_size_or_version}.txt"
            params["config_path"] = config_path
            # We pass lang via --lang CLI arg which PaddleOCR-json supports
            params["lang"] = lang

            params["model_size"] = model_size_or_version
            params["det"] = True
            params["blank_page_strategy"] = "skip"
            params["rec_batch_num"] = extra_params.get("rec_batch_num", 12 if use_gpu else 6)
            params["shrink_poly_ratio"] = extra_params.get("shrink_poly_ratio", 0.0)
            # CUDA 设备号（服务端 --gpu_device，默认 0；CPU 模式下服务端会忽略）
            params["gpu_device"] = extra_params.get("gpu_device", 0)
        elif engine_id == "win7_v5":
            params["enable_mkldnn"] = True
            params["cpu_threads"] = extra_params.get("cpu_threads", 4)
            # 按 lang 路由到对应文字系统的 config（各 config 指向官方分语种 rec 模型）
            # 阿拉伯/天城文/泰/希腊/泰卢固/泰米尔 + 韩文 + 西里尔(eslav/cyrillic) + 拉丁(latin)
            # 中/英/日及未列出语种回退 config_universal.txt（v5 server 模型）
            lang_v5 = extra_params.get("lang", "") or ""
            _v5_lang_config = {
                # 阿拉伯字母系（含新补：普什图/信德/克什米尔/俾路支）
                "ar": "arabic", "fa": "arabic", "ug": "arabic", "ur": "arabic",
                "ps": "arabic", "sd": "arabic", "ks": "arabic", "bal": "arabic",
                # 天城文系（含新补：博杰普尔/迈蒂利/孔卡尼）
                "hi": "devanagari", "mr": "devanagari", "ne": "devanagari", "sa": "devanagari",
                "bh": "devanagari", "mai": "devanagari", "kok": "devanagari",
                # 单文字系
                "th": "th", "el": "el", "te": "te", "ta": "ta",
                # 韩文（官方 v5 专用模型，准确率 88%）
                "korean": "korean",
                # 东斯拉夫（俄/白俄/乌克兰）
                "ru": "eslav", "uk": "eslav", "be": "eslav",
                # 其余西里尔语种
                "bg": "cyrillic", "mk": "cyrillic", "mn": "cyrillic", "kk": "cyrillic",
                "ky": "cyrillic", "tg": "cyrillic", "tt": "cyrillic", "ba": "cyrillic",
                "cv": "cyrillic", "rs_cyrillic": "cyrillic",
            }
            # 拉丁字母语种（40+ 种）统一走 latin 模型
            _LATIN_CODES = {
                "fr", "de", "es", "it", "pt", "nl", "ro", "ca", "gl", "da", "sv", "no",
                "fi", "is", "pl", "cs", "sk", "hu", "hr", "sl", "bs", "rs_latin", "sq",
                "ga", "cy", "et", "lt", "lv", "mt", "la", "pi", "af", "az", "uz", "ku",
                "eu", "oc", "vi", "id", "ms", "tl", "sw", "mi", "tr",
            }
            if lang_v5 in _LATIN_CODES:
                config_name = "latin"
            else:
                config_name = _v5_lang_config.get(lang_v5, "universal")
            params["config_path"] = f"models/config_{config_name}.txt"
        elif engine_id == "win7_classic":
            params["enable_mkldnn"] = True
            params["cpu_threads"] = extra_params.get("cpu_threads", 4)
            params["config_path"] = "models/config_chinese.txt"
        elif engine_id == "ppocr_full":
            lang = extra_params.get("lang", "ch") or "ch"
            params["language"] = lang
            params["ppocr_version"] = extra_params.get("ppocr_full_version", None)

    elif einfo["entry_type"] == "ncnn_cpu":
        params["model_version"] = model_size_or_version
        params["max_side_len"] = limit_side_len
        params["enable_cls"] = use_angle_cls
        params["num_threads"] = extra_params.get("num_threads", -1)
        params["enable_fp16"] = extra_params.get("enable_fp16", False)
        params["det_thres"] = extra_params.get("det_thres", 0.5)
        params["unclip_ratio"] = extra_params.get("unclip_ratio", 1.58)
        params["lang"] = extra_params.get("lang", "chinese") or "chinese"

    elif einfo["entry_type"] == "ncnn_vulkan":
        params["model_version"] = model_size_or_version
        params["max_side_len"] = limit_side_len
        params["enable_cls"] = use_angle_cls
        params["num_threads"] = extra_params.get("num_threads", -1)
        params["enable_fp16"] = extra_params.get("enable_fp16", False)
        params["det_thres"] = extra_params.get("det_thres", 0.5)
        params["unclip_ratio"] = extra_params.get("unclip_ratio", 1.58)
        params["use_gpu"] = extra_params.get("use_gpu", True)
        params["lang"] = extra_params.get("lang", "chinese") or "chinese"
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
    """多引擎OCR客户端 - 单例模式
    支持双实例并行OCR（dual_instance=True时启动两个独立引擎进程），
    通过轮询调度提升GPU利用率。
    """
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, engine_id="umi_plugin_v6", use_gpu=True,
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
            print(f"[OCRClient] 实例{i+1}就绪: {engine_display_name(engine_id)} | GPU={use_gpu} | 边长={limit_side_len} | 模型={model_size}")
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
                elif hasattr(a, 'pipe') and a.pipe:
                    a.pipe.kill()
            except Exception:
                pass
        self._instances = []
        self._initialized = False

    def restart_single(self, instance_idx):
        """重启单个引擎实例（Tier1恢复）"""
        instances = getattr(self, "_instances", [])
        if instance_idx < len(instances):
            try:
                instances[instance_idx].restart()
            except Exception as e:
                print(f"[OCRClient] restart_single({instance_idx}) failed: {e}")

    def restart_all(self):
        """重启全部引擎实例（Tier2恢复）"""
        instances = getattr(self, "_instances", [])
        for i, a in enumerate(instances):
            try:
                a.restart()
            except Exception as e:
                print(f"[OCRClient] restart_all[{i}] failed: {e}")

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
                    progress_callback=None, vertical_sort=False, overwrite_ocr=False):
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
            # 看门狗状态（per-consumer，复位周期=每页）
            _tier1_fired = False  # T1: 重启全部引擎（直接重启所有，不猜轮询映射）
            _tier2_fired = False  # T2: 触发全文件重启（清空结果、重建引擎、从头跑）
            while my_done < total_pages and not self._cancelled:
                try:
                    pn, b64_data = render_queue.get(timeout=0.3)
                except Empty:
                    if all_done.is_set():
                        break
                    continue
                if self._cancelled:
                    break
                # 每页开始时重置看门狗标志
                _tier1_fired = False
                _tier2_fired = False
                # 看门狗恢复循环：同一页最多重试到 T3（触发全文件重启）
                while True:
                    # 动态超时：取实例0的 get_dynamic_timeout（已聚合所有活跃引擎的耗时）
                    _inst0 = self.ocr._instances[0]
                    _gdt = getattr(_inst0, "get_dynamic_timeout", None)
                    timeout_sec = _gdt() if callable(_gdt) else 180
                    t_page = time.time()
                    try:
                        result = self.ocr.ocr_image_base64(b64_data, timeout_seconds=timeout_sec)
                    except Exception as e:
                        result = {"code": 900, "data": f"OCR error: {str(e)}"}
                    # 检测超时（引擎假死：code=102 + "timeout"）
                    if result.get("code") == 102 and "timeout" in str(result.get("data", "")).lower():
                        elapsed = time.time() - t_page
                        print(f"[Watchdog] Consumer-{consumer_id} Page {pn} timeout after {elapsed:.0f}s (limit={timeout_sec}s)")
                        if not _tier1_fired:
                            # Tier 1: 重启全部引擎实例（避免轮询映射错位）
                            _tier1_fired = True
                            print(f"[Watchdog] Tier1: restarting ALL engine instances")
                            self.ocr.restart_all()
                            continue
                        elif not _tier2_fired:
                            # Tier 2: 触发全文件重启（引擎重启无效，说明是页面级别问题）
                            _tier2_fired = True
                            print(f"[Watchdog] Tier2: all engine restarts failed → full file restart")
                            self._need_restart = True
                            self._cancelled = True
                            self.ocr.force_close()
                            return
                    # 非超时结果（成功 code=100 或 engine error）→ 退出看门狗循环
                    break
                # 正常/超时恢复成功：存结果
                if not self._need_restart and not self._cancelled:
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
                else:
                    break
        # 全文件重启外层循环（最多1次）
        self._need_restart = False
        for _full_retry in range(2):
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
            if self._need_restart and _full_retry == 0:
                print(f"[Watchdog] Full file restart #{_full_retry+1} triggered. Cleaning up...")
                self._cancelled = True
                self.ocr.force_close()
                time.sleep(2)
                print(f"[Watchdog] Rebuilding all engines for restart...")
                self.ocr.restart_all()
                self._cancelled = False
                self.results = {}
                self.completed_count = 0
                self._total_done = 0
                self._need_restart = False
                continue
            break
        if self._cancelled and not self._need_restart:
            return None, None
        if self._need_restart:
            print("[Watchdog] ERROR: Full file restart also failed. Aborting.")
            return None, None
        return self.write_results(input_path, output_dir, total_pages, scale,
                                vertical_sort=vertical_sort, overwrite_ocr=overwrite_ocr)

    def write_results(self, input_path, output_dir, total_pages, scale,
                     vertical_sort=False, overwrite_ocr=False):
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
            # 覆盖旧OCR：物理删除旧文字层（修订Redaction），完整保留图像与矢量图形
            if overwrite_ocr:
                for _p in output_pdf:
                    _p.add_redact_annot(_p.rect)   # 标记整页区域
                    _p.apply_redactions(
                        images=fitz.PDF_REDACT_IMAGE_NONE,      # 严禁删除图像
                        graphics=fitz.PDF_REDACT_LINE_ART_NONE  # 保留矢量图形
                    )
            output_page_count = 0
            for page_num in sorted(self.results.keys()):
                result = self.results[page_num]
                if result.get("code") != 100:
                    continue
                text_blocks = result.get("data", [])
                # 竖排排序：当勾选"竖排识别"时，按中心点X降序（右→左）再Y升序（上→下）
                if vertical_sort and text_blocks:
                    def _vertical_key(tb):
                        box = tb.get("box", [[0,0],[0,0],[0,0],[0,0]])
                        cx = (box[0][0] + box[2][0]) / 2
                        cy = (box[0][1] + box[2][1]) / 2
                        return (-cx, cy)  # X降序, Y升序
                    text_blocks = sorted(text_blocks, key=_vertical_key)
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
                    # OCR引擎返回的是渲染图像像素坐标（图像=页面xscale），
                    # 写入PDF前必须除以scale换算为PDF坐标，否则文字落在页面外无法提取/搜索
                    x0 /= scale
                    y0 /= scale
                    x2 /= scale
                    y2 /= scale
                    w = x2 - x0
                    h = y2 - y0
                    # 竖排判定：勾选"竖排识别"且OCR框高>宽（竖条）时文字竖排写入，
                    # 否则横排展开会超出页面（长句右侧文字落页外，查看器搜不到）。
                    # 未勾选"竖排识别"时保持旧版横排行为。
                    is_vertical_block = vertical_sort and h > w
                    fontsize = self.calculate_font_size(text, w, h)
                    if is_vertical_block:
                        # 竖排：从框顶向下排（rotate=270），整句不溢出页面
                        point = fitz.Point(x0, y0) * output_page.derotation_matrix
                        rotate = 270
                    else:
                        point = fitz.Point(x0, y2) * output_page.derotation_matrix
                        rotate = page_rotation
                    if not is_insert_font:
                        output_page.insert_font(fontname="cjk", fontbuffer=self.font.buffer)
                        is_insert_font = True
                    output_page.insert_text(
                        point, text, fontsize=fontsize, fontname="cjk",
                        rotate=rotate, stroke_opacity=0, fill_opacity=0
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
                "author": "CathayOCR Pro",
                "subject": f"OCR extracted on {et}",
                "creator": "CathayOCR Pro Processor",
            })
            try:
                if total_pages <= 2000:
                    output_pdf.save(output_pdf_path, deflate=True, garbage=3)
                else:
                    output_pdf.save(output_pdf_path, deflate=True, garbage=1)
            except Exception as e:
                print(f"[PDFProcessor] Save failed: {e}, retrying with no options...")
                try:
                    output_pdf.save(output_pdf_path)
                except Exception as e2:
                    print(f"[PDFProcessor] Retry also failed: {e2}")
            finally:
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
                 engine_id="umi_plugin_v6", use_gpu=True,
                 vertical_text=True, limit_side_len=2000,
                 model_size="medium", use_angle_cls=False,
                 scale=2.0, dual_instance=True, extra_params=None,
                 overwrite_ocr=False):
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
        self.overwrite_ocr = overwrite_ocr
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
                        vertical_sort=self.vertical_text,
                        overwrite_ocr=self.overwrite_ocr,
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
# 迷你窗口：处理中把两个窗口收成一张右下角的小进度卡
# ============================================================
class MiniWindow(QWidget):
    """超小卡片式进度窗 —— 只显示 当前文件 / 总文件数 / 已处理文件数 / 总进度。

    设计约定（2026-10-09 第 13 批）：
      * 无边框 + 置顶 + 圆角卡片，默认贴屏幕右下角（避开任务栏）；
      * 按住卡片任意位置可拖动；「恢复」按钮 / 双击卡片 / 任务栏图标都能回到完整窗口；
      * 数据全部由主窗口推过来（set_state），自己不碰任何业务逻辑 —— 纯 UI。
    """
    restore_requested = pyqtSignal()
    on_top_changed = pyqtSignal(bool)      # 置顶开关变化（主窗口负责记住这个选择）
    CARD_W, CARD_H = 420, 146

    def __init__(self):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowTitle("CathayOCR — 迷你进度窗")
        self.setFixedSize(self.CARD_W, self.CARD_H)
        self._drag = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(9, 9, 9, 9)
        card = QFrame(self)
        card.setObjectName("miniCard")
        card.setStyleSheet("QFrame#miniCard{background:#ffffff;border:1px solid #e2e8f0;"
                           "border-radius:12px;}")
        outer.addWidget(card)
        try:
            from PyQt5.QtWidgets import QGraphicsDropShadowEffect
            eff = QGraphicsDropShadowEffect(card)
            eff.setBlurRadius(16)
            eff.setOffset(0, 3)
            eff.setColor(QColor(15, 23, 42, 60))
            card.setGraphicsEffect(eff)
        except Exception:
            pass

        v = QVBoxLayout(card)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(7)

        top = QHBoxLayout()
        top.setSpacing(6)
        self.dot = QLabel("●")
        self.dot.setStyleSheet("color:#1a73e8;font-size:12px;")
        top.addWidget(self.dot)
        self.app_label = QLabel("CathayOCR")
        self.app_label.setStyleSheet("color:#1f2d3d;font-size:11px;font-weight:bold;")
        top.addWidget(self.app_label)
        self.state_label = QLabel("")
        self.state_label.setStyleSheet("color:#8a97a6;font-size:10px;")
        top.addWidget(self.state_label)
        top.addStretch()
        self.pin_btn = QPushButton("📌")
        self.pin_btn.setCheckable(True)
        self.pin_btn.setChecked(True)                 # 默认置顶
        self.pin_btn.setCursor(Qt.PointingHandCursor)
        self.pin_btn.setFixedWidth(28)
        self.pin_btn.setStyleSheet(
            "QPushButton{border:none;background:#f1f5f9;color:#94a3b8;"
            "border-radius:9px;padding:2px 4px;font-size:10px;}"
            "QPushButton:hover{background:#e2e8f0;color:#334155;}"
            "QPushButton:checked{background:#1a73e8;color:#ffffff;}")
        self.pin_btn.setToolTip("已置顶：点一下取消置顶")
        self.pin_btn.toggled.connect(self.set_on_top)
        top.addWidget(self.pin_btn)
        self.restore_btn = QPushButton("恢复")
        self.restore_btn.setCursor(Qt.PointingHandCursor)
        self.restore_btn.setToolTip("回到完整窗口（也可以双击本卡片，或点任务栏上的图标）")
        self.restore_btn.setStyleSheet(
            "QPushButton{border:none;background:#eaf1fb;color:#1a73e8;"
            "border-radius:9px;padding:2px 12px;font-size:10px;}"
            "QPushButton:hover{background:#1a73e8;color:#ffffff;}")
        self.restore_btn.clicked.connect(self.restore_requested.emit)
        top.addWidget(self.restore_btn)
        v.addLayout(top)

        self.file_label = QLabel("—")
        self.file_label.setStyleSheet("color:#334155;font-size:11px;")
        v.addWidget(self.file_label)

        # ── 当前文件（本 PDF）的页进度 ──
        #    数据与启动器日志窗口里的「[合计] x/y 页」同源（
        #    worker.file_progress 信号：已识别页数 / 该 PDF 总页数）。
        row_file = QHBoxLayout()
        row_file.setSpacing(9)
        self.page_tag = QLabel("本文件")
        self.page_tag.setStyleSheet("color:#8a97a6;font-size:10px;")
        self.page_tag.setFixedWidth(34)
        row_file.addWidget(self.page_tag)
        self.page_bar = QProgressBar()
        self.page_bar.setRange(0, 1000)
        self.page_bar.setTextVisible(False)
        self.page_bar.setFixedHeight(8)
        self.page_bar.setStyleSheet(
            "QProgressBar{background:#eef2f7;border:none;border-radius:4px;}"
            "QProgressBar::chunk{background:#34a853;border-radius:4px;}")
        row_file.addWidget(self.page_bar, 1)
        self.page_label = QLabel("— / — 页")
        self.page_label.setStyleSheet("color:#5b6b7c;font-size:10px;")
        self.page_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.page_label.setMinimumWidth(96)
        row_file.addWidget(self.page_label)
        v.addLayout(row_file)

        # ── 整个任务的进度（文件数） ──
        row = QHBoxLayout()
        row.setSpacing(9)
        self.total_tag = QLabel("总进度")
        self.total_tag.setStyleSheet("color:#8a97a6;font-size:10px;")
        self.total_tag.setFixedWidth(34)
        row.addWidget(self.total_tag)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        self.bar.setStyleSheet(
            "QProgressBar{background:#eef2f7;border:none;border-radius:4px;}"
            "QProgressBar::chunk{background:#1a73e8;border-radius:4px;}")
        row.addWidget(self.bar, 1)
        self.count_label = QLabel("— / —")
        self.count_label.setStyleSheet("color:#5b6b7c;font-size:10px;")
        self.count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.count_label.setMinimumWidth(96)
        row.addWidget(self.count_label)
        v.addLayout(row)

    # ---------- 对外接口 ----------
    def set_state(self, app_name, file_name, done, total, pct, processing,
                  page_done=0, page_total=0):
        """刷新卡片内容。processing=False 时显示「等待用户安排 OCR 任务」。

        page_done / page_total = **当前这个 PDF** 已识别页数 / 总页数，
        与启动器日志窗口里的「[合计] x/y 页」同源。
        """
        try:
            self.app_label.setText(str(app_name))
        except Exception:
            pass
        self.state_label.setText("正在识别…" if processing else "等待用户安排 OCR 任务")
        name = str(file_name or "—")
        budget = self.CARD_W - 2 * 9 - 2 * 14 - 10
        try:
            name = self.file_label.fontMetrics().elidedText(name, Qt.ElideMiddle, budget)
        except Exception:
            pass
        self.file_label.setText(("📄 " + name) if processing else name)
        self.file_label.setToolTip(str(file_name or ""))
        try:
            self.bar.setValue(max(0, min(1000, int(float(pct) * 1000))))
        except Exception:
            self.bar.setValue(0)
        if total > 0:
            self.count_label.setText("已处理 %d / %d" % (done, total))
        else:
            self.count_label.setText("— / —")
        # ── 当前文件的页进度 ──
        try:
            _pd = max(0, int(page_done or 0))
            _pt = max(0, int(page_total or 0))
        except Exception:
            _pd, _pt = 0, 0
        if _pt > 0:
            self.page_bar.setValue(max(0, min(1000, int(_pd * 1000.0 / _pt))))
            self.page_label.setText("第 %d / %d 页 (%.0f%%)"
                                    % (_pd, _pt, _pd * 100.0 / _pt))
        else:
            self.page_bar.setValue(0)
            self.page_label.setText("准备中…" if processing else "— / — 页")
        try:
            self.setToolTip("%s\n%s\n本文件：%s / %s 页\n已处理 %d / %d 个文件（%.0f%%）"
                            % (app_name, file_name,
                               str(_pd) if _pt > 0 else "—",
                               str(_pt) if _pt > 0 else "—",
                               done, total, float(pct) * 100))
        except Exception:
            pass

    def set_on_top(self, on, notify=True):
        """手动置顶 / 取消置顶。

        Qt 的 setWindowFlags() 会把窗口先隐藏，所以改完必须重新 show 一次；
        这里同时把几何位置补回去，免得卡片跳回屏幕左上角。
        """
        try:
            on = bool(on)
            flags = self.windowFlags()
            new_flags = (flags | Qt.WindowStaysOnTopHint) if on \
                else (flags & ~Qt.WindowStaysOnTopHint)
            if new_flags != flags:
                was_visible = self.isVisible()
                geo = self.geometry()
                self.setWindowFlags(new_flags)
                self.setGeometry(geo)
                if was_visible:
                    self.show()
                    self.raise_()
            self.pin_btn.blockSignals(True)
            self.pin_btn.setChecked(on)
            self.pin_btn.blockSignals(False)
            self.pin_btn.setToolTip("已置顶：点一下取消置顶" if on
                                    else "未置顶：点一下置顶")
        except Exception as e:
            print("[Mini] 切换置顶失败: %s" % e)
        if notify:
            try:
                self.on_top_changed.emit(on)
            except Exception:
                pass

    def show_at_bottom_right(self):
        """贴到屏幕右下角（用可用区域，自动避开任务栏）。"""
        try:
            g = QApplication.primaryScreen().availableGeometry()
            self.move(max(g.left(), g.right() - self.width() - 16),
                      max(g.top(), g.bottom() - self.height() - 16))
        except Exception:
            pass
        self.show()
        self.raise_()

    # ---------- 拖动 / 双击还原 ----------
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag is not None and (e.buttons() & Qt.LeftButton):
            self.move(e.globalPos() - self._drag)
            e.accept()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self.restore_requested.emit()


# ============================================================
# Main Window
# ============================================================

class MainWindow(QMainWindow):
    # ── 主语言列表（显示名 → 代码） ──
    _LANG_ITEMS = [
        # ── CJK + Korean ──
        ("中文 (Chinese)", "ch"),
        ("日本語 (日文)", "japan"),
        ("한국어 (韩文)", "korean"),

        # ── 西欧 拉丁文系 (V6 通用字典) ──
        ("English (英文)", "en"),
        ("Français (法文)", "fr"),
        ("Deutsch (德文)", "de"),
        ("Español (西班牙文)", "es"),
        ("Italiano (意大利文)", "it"),
        ("Português (葡萄牙文)", "pt"),
        ("Nederlands (荷兰文)", "nl"),
        ("Română (罗马尼亚文)", "ro"),
        ("Català (加泰罗尼亚文)", "ca"),
        ("Galego (加利西亚文／拉丁字母)", "gl"),

        # ── 北欧 拉丁文系 ──
        ("Dansk (丹麦文)", "da"),
        ("Svenska (瑞典文)", "sv"),
        ("Norsk (挪威文)", "no"),
        ("Suomi (芬兰文)", "fi"),
        ("Íslenska (冰岛文／拉丁字母)", "is"),

        # ── 中/东欧 拉丁文系 ──
        ("Polski (波兰文)", "pl"),
        ("Čeština (捷克文)", "cs"),
        ("Slovenčina (斯洛伐克文)", "sk"),
        ("Magyar (匈牙利文)", "hu"),
        ("Hrvatski (克罗地亚文／拉丁字母)", "hr"),
        ("Slovenščina (斯洛文尼亚文／拉丁字母)", "sl"),
        ("Bosanski (波斯尼亚文／拉丁字母)", "bs"),
        ("Srpski (拉丁) (塞尔维亚文/拉丁／拉丁字母)", "rs_latin"),
        ("Shqip (阿尔巴尼亚文／拉丁字母)", "sq"),

        # ── 凯尔特/不列颠 ──
        ("Gaeilge (爱尔兰文／拉丁字母)", "ga"),
        ("Cymraeg (威尔士文／拉丁字母)", "cy"),

        # ── 波罗的海 + 马耳他 ──
        ("Eesti (爱沙尼亚文／拉丁字母)", "et"),
        ("Lietuvių (立陶宛文／拉丁字母)", "lt"),
        ("Latviešu (拉脱维亚文／拉丁字母)", "lv"),
        ("Malti (马耳他文／拉丁字母)", "mt"),

        # ── 古典/其他 拉丁文系 ──
        ("Latina (拉丁文／拉丁字母)", "la"),
        ("Pāli (巴利文／拉丁字母)", "pi"),
        ("Afrikaans (南非荷兰文／拉丁字母)", "af"),
        ("Azərbaycan (阿塞拜疆文/拉丁／拉丁字母)", "az"),
        ("Oʻzbek (乌兹别克文/拉丁／拉丁字母)", "uz"),
        ("Kurdî (库尔德文/拉丁／拉丁字母)", "ku"),
        ("Euskara (巴斯克文／拉丁字母)", "eu"),
        ("Occitan (奥克文／拉丁字母)", "oc"),

        # ── 亚非 拉丁文系 ──
        ("Tiếng Việt (越南文)", "vi"),
        ("Bahasa Indonesia (印尼文)", "id"),
        ("Bahasa Melayu (马来文／拉丁字母)", "ms"),
        ("Tagalog (他加禄文/菲律宾／拉丁字母)", "tl"),
        ("Kiswahili (斯瓦希里文)", "sw"),
        ("Māori (毛利文／拉丁字母)", "mi"),

        # ── 土耳其文 ──
        ("Türkçe (土耳其文)", "tr"),

        # ── 西里尔文系 (需 PP-OCRv5 分语种模型: eslav/cyrillic；ncnn/V6 通用字典不含) ──
        ("Русский (俄文)", "ru"),
        ("українська (乌克兰文／西里尔字母)", "uk"),
        ("беларуская (白俄罗斯文／西里尔字母)", "be"),
        ("български (保加利亚文／西里尔字母)", "bg"),
        ("македонски (马其顿文／西里尔字母)", "mk"),
        ("монгол (蒙古文/西里尔／西里尔字母)", "mn"),
        ("қазақ (哈萨克文/西里尔／西里尔字母)", "kk"),
        ("кыргыз (吉尔吉斯文/西里尔／西里尔字母)", "ky"),
        ("тоҷикӣ (塔吉克文／西里尔字母)", "tg"),
        ("татар (鞑靼文／西里尔字母)", "tt"),
        ("башҡорт (巴什基尔文／西里尔字母)", "ba"),
        ("чӑваш (楚瓦什文／西里尔字母)", "cv"),
        ("Srpski (西里尔) (塞尔维亚文/西里尔／西里尔字母)", "rs_cyrillic"),

        # ── 希腊文 ──
        ("Ελληνικά (希腊文)", "el"),

        # ── 阿拉伯文系 (V5 阿拉伯 ONNX 模型) ──
        ("العربية (阿拉伯文)", "ar"),
        ("فارسی (波斯文／阿拉伯字母)", "fa"),
        ("ئۇيغۇرچە (维吾尔文／阿拉伯字母)", "ug"),
        ("اردو (乌尔都文／阿拉伯字母)", "ur"),
        ("پښتو (普什图文／阿拉伯字母)", "ps"),
        ("سنڌي (信德文／阿拉伯字母)", "sd"),
        ("کٲشُر (克什米尔文／阿拉伯字母)", "ks"),
        ("بلوچی (俾路支文／阿拉伯字母)", "bal"),

        # ── 天城文系 (V5 天城 ONNX 模型) ──
        ("हिन्दी (印地文)", "hi"),
        ("मराठी (马拉地文／天城文)", "mr"),
        ("नेपाली (尼泊尔文／天城文)", "ne"),
        ("संस्कृत (梵文／天城文)", "sa"),
        ("भोजपुरी (博杰普尔文／天城文)", "bh"),
        ("मैथिली (迈蒂利文／天城文)", "mai"),
        ("कोंकणी (孔卡尼文／天城文)", "kok"),

        # ── 东南亚文字 (V5 ONNX 模型) ──
        ("ภาษาไทย (泰文)", "th"),
        ("తెలుగు (泰卢固文)", "te"),
        ("தமிழ் (泰米尔文)", "ta"),

        # ── 多语言 (混合) ──
        ("多语言混排 (中·英·日 + 拉丁语系)", "multilang_v6"),
        ("多语言混排 (46 种拉丁语系)", "multilang_v5"),
    ]

    # ── 简单模式合并语言表：(显示名[带说明], 代码, 专业模式对应显示名) ──
    # 把 75 项按"引擎处理方式相同"的原则合并成 13 项：
    #  - 主力引擎（v6/ncnn）内置通用字典直接支持的语种归并为 大类
    #  - 需要 v5 分语种模型的单文字语种单列
    _SIMPLE_LANG_ITEMS = [
        # ⚠ 标记 = 勾选后切换到"专用分语种模型"（不在 v6/ncnn 通用字典里）：
        #   - NVIDIA 显卡（不论显存大小）→ PP-OCRv6 引擎，服务端自动回退 PP-OCRv5 分语种模型
        #   - 非 NVIDIA（AMD/Intel/纯CPU）→ win7_v5 (PP-OCRv5 Paddle CPU) 备选引擎
        #   （希腊文实测 ncnn 字典可识 → 不切换）
        ("中文", "ch", "中文 (Chinese)"),
        ("英文", "en", "English (英文)"),
        ("日文", "japan", "日本語 (日文)"),
        ("韩文⚠", "korean", "한국어 (韩文)"),
        ("西里尔文系⚠ (俄/乌/保等)", "ru", "Русский (俄文)"),
        ("拉丁语系 (法德西意等)", "fr", "Français (法文)"),
        ("阿拉伯文系⚠ (阿/波/维/乌等)", "ar", "العربية (阿拉伯文)"),
        ("天城文系⚠ (印地/马拉地等)", "hi", "हिन्दी (印地文)"),
        ("泰文⚠", "th", "ภาษาไทย (泰文)"),
        ("希腊文", "el", "Ελληνικά (希腊文)"),
        ("泰卢固文⚠", "te", "తెలుగు (泰卢固文)"),
        ("泰米尔文⚠", "ta", "தமிழ் (泰米尔文)"),
        ("多语言混排 (中·英·日 + 拉丁语系)", "multilang_v6", "多语言混排 (中·英·日 + 拉丁语系)"),
    ]

    # 需要"专用分语种模型"的语言组代码：这些文字不在 v6/ncnn 通用字典中
    # （实测字典 0 字符覆盖）。NVIDIA 机 → PP-OCRv6 引擎（自动回退 PP-OCRv5 分语种模型）；
    # 非 NVIDIA → win7_v5 (Paddle CPU) 备选。
    # 注意：希腊文(el) 已实测 ncnn Vulkan 字典含希腊字符且识别正确 → 不切换；
    #       多语言混排(multilang_v6) 以中英日为主不切换，混排韩俄请单独勾选对应组
    _V5_FORCE_CODES = {"ar", "hi", "th", "te", "ta",
                       "korean", "ru", "multilang_v5"}

    # ⚠ 语言在「简单模式语言表」里的先后顺序 —— 决定主语言是谁
    # （与 _SIMPLE_LANG_ITEMS 的排列一致；_simple_pick_engine 收到的可能是 set，
    #   用这张固定顺序表裁决才能和 _simple_primary_lang() 的结果对得上）
    _V5_SIMPLE_ORDER = ("korean", "ru", "ar", "hi", "th", "te", "ta", "multilang_v5")

    # 西里尔文系（简单模式唯一入口 = 俄/乌/保 等，代码 "ru"）——
    # 2026-10-09 真机实测（12 张俄/乌克兰图）：v6 引擎的 ONNX 版分语种模型
    # （eslav 与 cyrillic 都试过）会把 «Русский» 认成 «Russkiy»（11/12 对），
    # 而 win7_v5 的 Paddle Inference 版同款 eslav 模型 12/12 全对；
    # 且两边都跑 CPU（v6 引擎的 v5 分支强制 ORT CPU），整体耗时 7.9s vs 42.3s
    # → 西里尔文没有理由留在 v6 引擎，一律优先 Paddle CPU。
    _CYRILLIC_SIMPLE_CODES = {"ru"}

    # ncnn 引擎不支持的语言代码 —— **唯一事实来源**：
    #   · 专业模式「语言下拉」按它过滤（不列出必乱码的语言）；
    #   · 开跑前守卫 / 引擎切换提醒按它判定。
    # 依据：实测 ncnn 的 v6 字典 models/ppocr_keys_v6.txt（18709 字）字符覆盖为
    #   CJK 15565 · 假名 180 · ASCII 拉丁 52 · 希腊 76；
    #   而 韩文(Hangul)=0 · 西里尔=0 · 阿拉伯=0 · 天城文=0 · 泰/泰卢固/泰米尔=0
    #   （v5 字典同项亦分别仅 2 / 11 字，远不足以成词）→ 这些文字在 ncnn 上必乱码/空输出。
    # 注意：希腊文(el) 与「多语言混排(中·英·日+拉丁)」「多语言混排(46 拉丁)」
    #       均在 v6 字典覆盖范围内，**不**属于不支持集合。
    _NCNN_UNSUPPORTED = {
        # 韩文
        "korean",
        # 西里尔文系
        "ru", "uk", "be", "bg", "mk", "mn", "kk", "ky", "tg", "tt", "ba", "cv", "rs_cyrillic",
        # 阿拉伯字母系
        "ar", "fa", "ug", "ur", "ps", "sd", "ks", "bal",
        # 天城文系
        "hi", "mr", "ne", "sa", "bh", "mai", "kok",
        # 东南亚单文字系
        "th", "te", "ta",
    }

    # PP-OCRv6 引擎下「v6 通用模型不认识、会自动改走 PP-OCRv5 分语种模型」的语言代码。
    # 依据 ppocr_v6_server.py：_V6_LANGS（50 项，共用同一个 v6 模型 + 同一份 18705 字字典）
    # 之外的语言一律落入 v5 分支（_V5_LANGS + _resolve_lang 映射后的结果集），
    # 这些文字 v6 通用字典实测 0 覆盖，必须换模型 —— 此时语言才是真正生效的。
    # 共 33 项：韩文 1 + 西里尔 13 + 阿拉伯 8 + 天城 7 + 泰/希腊/泰卢固/泰米尔 4。
    _V6_V5_FALLBACK = {
        "korean",
        "ru", "uk", "be", "bg", "mk", "mn", "kk", "ky", "tg", "tt", "ba", "cv", "rs_cyrillic",
        "ar", "fa", "ug", "ur", "ps", "sd", "ks", "bal",
        "hi", "mr", "ne", "sa", "bh", "mai", "kok",
        "th", "el", "te", "ta",
    }

    # 「语言选择到底起什么作用」不常驻界面（太占地方），
    # 只放进「语言下拉」的悬停 tooltip（见 _update_lang_combo / lang_combo.setToolTip），
    # 完整规则见同目录《引擎规则说明.md》§3.4。
    # 依据 ppocr_v6_server.py 的实际分流：
    #   _V6_LANGS（ch / chinese_cht / japan + 46 种拉丁语系）→ 同一份 PP-OCRv6 模型 + 同一份通用字典
    #   其余（韩/西里尔/阿拉伯/天城/泰/泰卢固/泰米尔/希腊…）→ 自动改用 PP-OCRv5 官方分语种模型，
    #   且该分支强制 CPU 推理（ORT CUDA 下会 RUNTIME_EXCEPTION）。

    def _simple_langs_checked(self):
        """返回勾选的语言代码列表（按合并表顺序）"""
        if not hasattr(self, "simple_lang_checks"):
            return ["ch"]
        return [code for _disp, code, _full in self._SIMPLE_LANG_ITEMS
                if code in self.simple_lang_checks and self.simple_lang_checks[code].isChecked()]

    def _simple_primary_lang(self):
        """返回 (主语言代码, 专业模式显示名)：优先取第一个触发降级的勾选项，否则第一个勾选项"""
        checked = self._simple_langs_checked()
        if not checked:
            return "ch", "中文 (Chinese)"
        pick = next((c for c in checked if c in self._V5_FORCE_CODES), checked[0])
        for _disp, code, full in self._SIMPLE_LANG_ITEMS:
            if code == pick:
                return pick, full
        return "ch", "中文 (Chinese)"

    def _mixed_langs_problem(self):
        """简单模式下「多语言混排做不到」的说明文本；没问题返回 ''。

        实测结论：韩 / 西里尔 / 阿拉伯 / 天城 / 泰 / 泰卢固 / 泰米尔 这 7 类文字，
        三个引擎都只有「一个文字系一个模型」的专用模型，且 ncnn 通用字典对它们
        0 覆盖（识别必乱码）→ 与任何其它语言同时勾选时都不可能真正混合。
        """
        if not (hasattr(self, "ui_simple_btn") and self.ui_simple_btn.isChecked()):
            return ""
        checked = self._simple_langs_checked()
        if len(checked) <= 1:
            return ""
        hard = [f for _d, c, f in self._SIMPLE_LANG_ITEMS
                if c in checked and c in self._V5_FORCE_CODES]
        if not hard:
            return ""
        soft = [f for _d, c, f in self._SIMPLE_LANG_ITEMS
                if c in checked and c not in self._V5_FORCE_CODES]
        _pick_code, pick_full = self._simple_primary_lang()
        return ("你勾选了多个语言，但其中含「专用分语种文字」：\n\n"
                "    " + "、".join(hard) + "\n\n"
                "这类文字在三个引擎里都只有「一个文字系一个模型」的专用模型，\n"
                "而 ncnn 通用字典对它们 0 字符覆盖（识别必乱码）——\n"
                "所以它们无法与其它语言混合识别。\n\n"
                f"本次将只用「{pick_full}」的专用模型做整篇识别"
                + (f"，另外 {len(soft)} 项（{'、'.join(soft)}）会被忽略。" if soft else "。")
                + "\n\n要继续吗？\n"
                "（建议：拆成多次处理，每次只勾一个文字系；\n"
                "  或在专业模式里对同一批文件分两遍跑，再把结果合并。）")

    def _enforce_simple_lang_selection(self):
        """强制纠正「无法混排」的多语言勾选（2026-10-09 用户要求：不能只提示）。

        勾了 ⚠ 分语种文字（韩/西里尔/阿拉伯/天城文/泰/泰卢固/泰米尔/多语言v5）
        且还勾了别的语言时：分语种模型一次只认一个文字系、ncnn 通用字典对它们
        0 字符覆盖 —— 任何引擎都做不到混排。直接取消多余勾选、只保留主语言，
        并亮出红色警告条说明改了什么、为什么、正确做法。
        （希腊文不在 _V5_FORCE_CODES 里：希腊混排可由 ncnn v6 字典覆盖，不纠正。）
        返回警告文本；勾选组合没有问题返回 ""。
        """
        warn = getattr(self, "simple_lang_warn", None)
        if warn is None:
            return ""
        checked = self._simple_langs_checked()
        has_hard = any(c in self._V5_FORCE_CODES for c in checked)
        if len(checked) <= 1 or not has_hard:
            warn.setVisible(False)      # 幂等；不要用 isVisible() 判断（offscreen 恒 False）
            return ""
        pick_code, _pick_full = self._simple_primary_lang()
        dropped = [(disp, c) for disp, c, _f in self._SIMPLE_LANG_ITEMS
                   if c in checked and c != pick_code]
        for _disp, c in dropped:                      # 强制纠正：取消其余勾选
            cb = self.simple_lang_checks.get(c)
            if cb is not None:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
        kept = next(d for d, c, _f in self._SIMPLE_LANG_ITEMS if c == pick_code)
        text = (
            "⛔ 已自动纠正你的语言勾选 —— 这些语言无法混合识别\n\n"
            f"   保留：{kept}　　取消：{'、'.join(d for d, _c in dropped)}\n\n"
            "原因：⚠ 分语种文字（韩文/西里尔/阿拉伯/天城文/泰/泰卢固/泰米尔）每个引擎\n"
            "都只有「一个文字系一个模型」的专用模型，一次只能认一种；而 ncnn 通用字典\n"
            "对它们 0 字符覆盖 —— 无论换哪个引擎都做不到混排。\n\n"
            "如需同时识别多种文字：请分两次处理（先只勾 A 跑一遍，再只勾 B 跑一遍）。"
        )
        warn.setText(text)
        warn.setVisible(True)
        return text

    @classmethod
    def _simple_pick_engine(cls, checked_codes, gpu_idx,
                            available=("umi_plugin_v6", "ncnn_vulkan", "win7_v5")):
        """简单模式引擎决策（纯逻辑，便于单测）。

        gpu_idx 取值（与 simple_gpu 下拉一致）：
          0 = NVIDIA独显(≥12GB)        1 = NVIDIA独显(≤8GB)
          2 = AMD/Intel独显(≥12GB)     3 = AMD/Intel独显(≤8GB)
          4 = 仅有核显/纯CPU           5 = 自动检测

        规则：
        1. 勾选了 ⚠ 语言（_V5_FORCE_CODES：阿拉伯字母/天城文/泰/泰卢固/泰米尔/韩文/西里尔）：
           - **西里尔文（俄/乌/保）为主语言 → 一律 win7_v5 (Paddle CPU)**：
             实测 v6 引擎的 ONNX 版 eslav/cyrillic 模型会把 «Русский» 认成 «Russkiy»，
             Paddle 版 12/12 全对且更快（7.9s vs 42.3s，两边都跑 CPU）。
           - 其它 ⚠ 语言 + **NVIDIA 显卡（≥12GB 或 ≤8GB 都算）** → umi_plugin_v6：
             服务端自动回退官方 PP-OCRv5 分语种模型（实测韩/阿/泰逐字正确，~3 秒/页）。
             注：v5 分语种模型目前在该引擎内走 CPU 推理（ORT CUDA 内核在部分张量形状下
             会崩，与显存大小无关），版本门槛只用于区分"选哪个引擎"，不代表独占 CUDA。
           - 非 NVIDIA（AMD/Intel/核显/CPU）→ win7_v5 (PP-OCRv5 Paddle CPU) 备选（唯一实证可用）
           这些文字不在 v6/ncnn 通用字典（实测 0 字符覆盖），ncnn 引擎必乱码。
        2. 其余语言（普通路径）按显卡档位：
           NVIDIA≥12GB(0)                          → umi_plugin_v6 (CUDA，精度最高)
           其余档位(1/2/3/4/5)                      → ncnn_vulkan
             · 独显(1/2/3) 走 Vulkan/GPU；
             · 核显·纯CPU(4) 由 _apply_simple_settings 把模式设为 CPU(use_gpu=False) ——
               ncnn_cpu 与 ncnn_vulkan 是同一个二进制，故不再区分独立引擎。
           希腊文单勾时实测 ncnn 字典可识，归入普通路径（例外见规则 3a）。
           注：AMD/Intel 卡没有 CUDA，v6 引擎用不了 → 一律走 Vulkan。
        3. ⚠ 多语言混排裁决（依据实测字典覆盖，不是猜的）：
           a. 勾了希腊文且同时还勾了别的语言 → **强制 ncnn_vulkan**：
              CUDA 下希腊文会落到 PP-OCRv5 el 专用模型（单文字系）→ 汉字/英文全丢；
              而 ncnn 的 v6 字典同时含 15565 汉字 + 76 希腊字符 → 只有它能真正混合。
           b. 勾了硬文字（_V5_FORCE_CODES：韩/西里尔/阿拉伯/天城/泰/泰卢固/泰米尔）
              → 三个引擎都只能认一个文字系（ncnn 字典对这些文字 0 覆盖，必乱码）
              → 引擎仍取「能认该文字」的那个，由上层弹窗告知做不到混合。
        返回 (engine_id, downgraded)。downgraded=True 表示语言触发了专用引擎/模型切换。
        """
        nvidia = gpu_idx in (0, 1)  # 任意 NVIDIA 卡都具备 CUDA 能力（含 ≤8GB）
        if any(c in cls._V5_FORCE_CODES for c in checked_codes):
            # 主语言 = 按简单模式语言表顺序第一个命中的 ⚠ 语言（与 _simple_primary_lang 一致）
            primary = next((c for c in cls._V5_SIMPLE_ORDER if c in checked_codes), None)
            # 【西里尔特例】主语言是西里尔文 → 一律优先 Paddle CPU 引擎（实测更准且更快，
            #  依据见 _CYRILLIC_SIMPLE_CODES 注释）。v6 引擎的 v5 分支本就强制 CPU，
            #  这里换引擎不损失 GPU，只换掉那个会认错字的 ONNX 模型。
            if primary in cls._CYRILLIC_SIMPLE_CODES and "win7_v5" in available:
                return "win7_v5", True
            # 专用分语种模型路径：只要 N 卡就走 v6 引擎（不限显存），否则走 v5 CPU 备选
            if nvidia and "umi_plugin_v6" in available:
                return "umi_plugin_v6", True
            for cand in ("win7_v5", "ncnn_vulkan"):
                if cand in available:
                    return cand, True
            return "umi_plugin_v6", True
        # 【混排 3a】希腊文 + 其它语言：CUDA 会为一个希腊文降级成单文字系模型，
        # 而 ncnn 的 v6 字典里汉字与希腊字符共存 → 只有 ncnn 能真正混排。
        if "el" in checked_codes and len(checked_codes) > 1:
            if "ncnn_vulkan" in available:
                return "ncnn_vulkan", True
            if "umi_plugin_v6" in available:
                return "umi_plugin_v6", False
        # 普通路径：仅大显存 N 卡走 CUDA；其余走 ncnn Vulkan（核显/纯CPU 由模式=CPU 承担）
        if gpu_idx == 0 and "umi_plugin_v6" in available:
            return "umi_plugin_v6", False
        if "ncnn_vulkan" in available:
            return "ncnn_vulkan", False
        return "win7_v5", False

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CathayOCR Pro (专业版) - 多引擎PDF处理器")
        # ── 系统托盘：只作「显示 / 退出」入口，不接管最小化（详见 _setup_tray）──
        self._tray = None
        self._quitting = False
        # ── 迷你窗口：处理中把两个窗口收成一张右下角小进度卡（详见 MiniWindow）──
        self._mini = None
        self._mini_active = False
        self._mini_prev_state = Qt.WindowNoState
        self._current_file_name = "—"
        self._set_app_icon()
        # 界面就绪标志：__init__ 期间不弹「引擎/语言不匹配」提示
        self._ui_ready = False
        # 程序化切换引擎时抑制该提示（如简单模式同步）
        self._engine_guard_suppress = False
        # 处理中标志：为 True 时禁止一切会改变识别配置的操作
        # （含「简单模式 ↔ 专业模式」切换、语言/显卡/输出目录修改、往列表加文件）
        self._processing = False
        self.setGeometry(100, 100, 1100, 850)
        self.cfg = QSettings("QClaw", "PDFOCRProcessor")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        title = QLabel("PDF OCR 流水线处理工具 - 多引擎支持")
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
        # ── 「迷你窗口」：处理中把主窗口 + 日志窗口收成一张右下角小进度卡（纯 UI）──
        self.mini_btn = QPushButton("🗕 迷你窗口")
        self.mini_btn.setCursor(Qt.PointingHandCursor)
        self.mini_btn.setStyleSheet(
            "QPushButton{border:1px solid #cfd6dd;border-radius:4px;padding:3px 10px;"
            "background:#f5f7f9;color:#4a5a6a;}"
            "QPushButton:hover{background:#1a73e8;color:#ffffff;border-color:#1a73e8;}")
        self.mini_btn.setToolTip(
            "把主窗口和运行日志窗口收成右下角一张小进度卡，\n"
            "只显示当前文件 / 总文件数 / 已处理文件数和进度条。\n"
            "处理结束或点卡片上的「恢复」会自动还原（不影响识别功能）")
        self.mini_btn.clicked.connect(self.enter_mini_mode)
        ms_layout.addWidget(self.mini_btn)
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
            "NVIDIA独显 (显存≥12GB)",
            "NVIDIA独显 (显存≤8GB)",
            "AMD / Intel 独显 (显存≥12GB)",
            "AMD / Intel 独显 (显存≤8GB)",
            "仅有核显 / 纯CPU",
            "🤖 不知道有没有独显 / 显存多大 → 自动检测",
        ])
        self.simple_gpu.setMinimumWidth(300)
        def set_gpu_tip(idx):
            tips = [
                "NVIDIA 大显存 → ONNX CUDA 引擎，精度最高（也支持简单模式下的专用语种模型）；"
                "精度优先时边长 2560",
                "NVIDIA 小显存同样能用 CUDA（8GB 实测可用）；普通语言走 ncnn Vulkan+双实例更省显存，"
                "勾选⚠语种时自动切到 CUDA 引擎；精度优先时边长 2240（防爆显存）",
                "AMD / Intel 大显存独显 → ncnn Vulkan 引擎+双实例，速度不输 CUDA；精度优先时边长 2560",
                "AMD / Intel 小显存独显 → ncnn Vulkan 引擎+双实例；精度优先时边长 2240（防爆显存）",
                "纯CPU运行，兼容性最好但速度最慢；精度优先时边长 2560（无显存限制）",
                "不知道有没有独立显卡，或者不知道显存多大？选这项，软件自动检测后帮你落到上面某一档\n"
                "（检测顺序：nvidia-smi 读 NVIDIA 显存 → Vulkan 探测独显 → 都没有则按纯CPU）",
            ]
            self.simple_gpu.setToolTip(tips[idx] if idx < len(tips) else "")
        self.simple_gpu.currentIndexChanged.connect(set_gpu_tip)
        set_gpu_tip(0)
        r3.addWidget(self.simple_gpu)
        r3.addStretch()
        sl.addLayout(r3)

        # 问题4：语言（多选勾选，勾选文档中可能出现的语言；仅用于帮助选择引擎，
        #        不影响调用的识别字典/模型。⚠ 项会切到"专用分语种模型"）
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("🌐 文档语言"))
        lang_holder = QWidget()
        lang_grid = QGridLayout(lang_holder)
        lang_grid.setContentsMargins(0, 0, 0, 0)
        lang_grid.setHorizontalSpacing(10)
        lang_grid.setVerticalSpacing(2)
        self.simple_lang_checks = {}
        for _i, (disp, code, _full) in enumerate(self._SIMPLE_LANG_ITEMS):
            cb = QCheckBox(disp)
            if "⚠" in disp:
                cb.setToolTip("该文字不在 v6/ncnn 通用字典中，勾选后系统自动切到专用分语种模型：\n"
                              "  · NVIDIA 显卡（不论显存大小）→ PP-OCRv6 引擎，自动回退 PP-OCRv5 分语种模型\n"
                              "  · AMD/Intel/纯CPU        → PP-OCRv5 (Paddle CPU) 备选引擎\n"
                              "分语种模型目前为 CPU 推理，约 3 秒/页。\n"
                              "⚠ 分语种模型是「一个文字系一个模型」，一次只能认一个 ——\n"
                              "   同时勾选多个 ⚠ 组（如 韩文 + 泰文）做不到混合识别，\n"
                              "   系统只会按其中一个识别，另一种会输出乱码。需要混合请分次处理。")
            elif code == "multilang_v6":
                cb.setToolTip("中·英·日 + 拉丁语系混排：主力引擎的通用字典本就覆盖这些文字，\n"
                              "不需要切换引擎（中文字典内含全部拉丁字母，混排英文也能认）。\n"
                              "若混排中还含 韩文 / 俄文 等 ⚠ 文字，请另外把对应 ⚠ 组也勾上，\n"
                              "但注意分语种模型一次只能认一个文字系，无法与中英日真正混合。")
            else:
                cb.setToolTip("勾选文档中可能出现的语言（可多选）。\n"
                              "作用是让系统据此选对「引擎/模型」：\n"
                              "  · 只勾这类普通语种（中/英/日/拉丁/希腊）→ 按显卡档用主力引擎，\n"
                              "    其通用字典直接覆盖，不改引擎；\n"
                              "  · 一旦勾了带 ⚠ 的组 → 自动切到能认这些文字的引擎\n"
                              "    （NVIDIA：PP-OCRv6 回退 v5 分语种模型；非 NVIDIA：PP-OCRv5 Paddle 备选）。\n"
                              "💡 中文/拉丁/日文的通用字典互相包含（中文字典含整套拉丁字母），\n"
                              "   所以「中英混排」只勾中文即可，不必再勾英文。")
            self.simple_lang_checks[code] = cb
            lang_grid.addWidget(cb, _i // 4, _i % 4)
        self.simple_lang_checks["ch"].setChecked(True)
        r4.addWidget(lang_holder)
        r4.addStretch()
        sl.addLayout(r4)

        # 强制纠正警告条（红色醒目，勾了「无法混排」的语言组合时自动纠正并说明）
        self.simple_lang_warn = QLabel()
        self.simple_lang_warn.setStyleSheet(
            "background:#fdecea; color:#b71c1c; font-weight:bold; font-size:12px;"
            "padding:8px; border:1px solid #d32f2f; border-radius:4px;")
        self.simple_lang_warn.setWordWrap(True)
        self.simple_lang_warn.setVisible(False)
        sl.addWidget(self.simple_lang_warn)

        # 配置摘要（一行灰色小字）
        self.simple_preview = QLabel()
        self.simple_preview.setStyleSheet("color: #666; font-size: 11px; font-style: italic; padding: 0px; margin: 0px;")
        self.simple_preview.setWordWrap(True)
        sl.addWidget(self.simple_preview)

        # 监听变化自动更新预览
        self.simple_doc.currentIndexChanged.connect(self._apply_simple_settings)
        self.simple_speed.currentIndexChanged.connect(self._apply_simple_settings)
        self.simple_gpu.currentIndexChanged.connect(self._apply_simple_settings)
        for _cb in self.simple_lang_checks.values():
            _cb.stateChanged.connect(self._apply_simple_settings)
        sl.addStretch()
        layout.addWidget(self.simple_group)
        self.simple_group.setVisible(False)
        # 收集专业模式的所有参数分组，用于简单模式下隐藏
        self._expert_groups = []

        # === 引擎选择（专业模式）===
        eg = QGroupBox("OCR引擎")
        self._expert_groups.append(eg)
        el = QHBoxLayout(eg)
        el.addWidget(QLabel("选择引擎:"))
        self.engine_combo = QComboBox()
        # 隐藏引擎（不单独列出）：
        #   ncnn_cpu 与 ncnn_vulkan 实为同一个二进制（两个 exe 的 MD5 完全相同），
        #   CPU 运行统一由 ncnn Vulkan 的「CPU模式」(use_gpu=False) 承担，
        #   故不再作为独立引擎出现在下拉里；旧配置若指向它会在下方回退到 Vulkan。
        _HIDDEN_ENGINES = {"ncnn_cpu"}
        for eid, einfo in sorted(ENGINE_REGISTRY.items(), key=lambda x: x[1]["priority"], reverse=True):
            if eid in _PLUGIN_DIRS and eid not in _HIDDEN_ENGINES:
                label = einfo["name"]
                if einfo["gpu"] and einfo["cpu"]:
                    label += " (GPU/CPU)"
                elif einfo["gpu"]:
                    label += " (GPU)"
                idx = self.engine_combo.count()
                self.engine_combo.addItem(label, eid)
                desc = einfo.get("desc", "")
                if desc:
                    self.engine_combo.setItemData(idx, desc, Qt.ToolTipRole)
        # 兼容旧配置：若存的是已合并的 ncnn_cpu，在 _restore_engine_settings 里映射到 ncnn Vulkan
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.engine_combo.setToolTip(
            "选择OCR引擎（小白推荐：PP-OCRv6 (ONNX CUDA)）:\n"
            "  ⭐ PP-OCR (ncnn Vulkan) - 任意显卡均可用，速度最快；\n"
            "                            纯CPU机器请把右侧「模式」切到 CPU模式\n"
            "  PP-OCRv6 (ONNX CUDA)  - 精度最高，需NVIDIA独显\n"
            "  PP-OCRv5 (Paddle CPU) - 备选，覆盖全部语种（含韩/俄/阿等）\n"
            "  PP-OCRv3 (Paddle CPU) - 经典Paddle引擎，兼容性最好\n"
            "  EasyOCR               - 拉丁语系专用"
        )
        el.addWidget(self.engine_combo)
        el.addStretch()
        el.addWidget(QLabel("模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumWidth(100)
        self.mode_combo.setToolTip(
            "运行模式选择 (小白推荐: 自动):\n"
            "  自动(推荐) = 有独显自动用GPU，无独显用CPU\n"
            "  GPU模式     = 强制使用GPU加速\n"
            "  CPU模式     = 仅用CPU，省显存，适合老旧机器\n"
            "  注：本机显卡为独显时建议保持「自动」；ncnn Vulkan 下 CPU模式 = 同一个\n"
            "      引擎以 use_gpu=False 运行（纯 CPU 机器也能用），不会再切到别的引擎。"
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
            "  同一引擎下，改模型不依赖网络下载\n"
            "  ⚠ ncnn 引擎下，模型决定用哪份字典，语言列表会随之变化"
        )
        # 切模型 → 所用字典变了 → 语言列表必须跟着重建（ncnn 专有行为）
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        el.addWidget(self.model_combo)
        el.addStretch()
        el.addWidget(QLabel("语言:"))
        self.lang_combo = QComboBox()
        self.lang_combo.setMinimumWidth(260)
        self.lang_combo.setMaxVisibleItems(20)
        self.lang_combo.addItems(["中文", "English", "Français", "Deutsch", "日本語", "多语言"])
        self.lang_combo.setToolTip(
            "选择识别语言 —— 但「语言到底起不起作用」完全由当前引擎决定。\n"
            "\n"
            "⚠ 核心概念：PP-OCR 系引擎（PP-OCRv6 / PP-OCR ncnn / PP-OCRv5 Paddle）"
            "都是「一个模型配一份字典」，\n"
            "   很多语言共用同一份字典 → 在这些语种之间来回切，识别结果不会变。\n"
            "   语言在这里只是「声明文档里可能出现哪些文字」，不是选模型。\n"
            "\n"
            "【PP-OCRv6 (ONNX CUDA)】\n"
            "  · 中/英/日 + 46 种拉丁语系（共 47 项）→ 共用同一个 v6 模型、同一份\n"
            "    18705 字通用字典（内含 15565 汉字 + 180 假名 + 462 拉丁 + 76 希腊字母）。\n"
            "    ⚠ 这 47 项选哪个都一样，不用纠结。\n"
            "  · 韩/俄等西里尔/阿拉伯/天城/泰/泰卢固/泰米尔/希腊（共 33 项）→ 自动改用\n"
            "    PP-OCRv5 官方分语种模型，语言此时才真正生效（换模型；该分支走 CPU，约 3 秒/页）。\n"
            "\n"
            "【PP-OCR (ncnn Vulkan)】★ 字典由「模型」决定，多语言共用同一份：\n"
            "    v3/v4 模型→v1 字典 · v5 模型→v5 字典 · v6 small/medium→v6 字典 ·\n"
            "    v6 tiny→v6_tiny 字典。选中 ncnn 时本提示会按当前模型自动改写，\n"
            "    写明用哪份字典、覆盖哪些文字系、哪些语言被置灰。\n"
            "\n"
            "【PP-OCRv5 (Paddle CPU)】按语言切官方专用模型：\n"
            "    中/英/日→universal；40+ 拉丁语系→latin；韩/俄/阿/天城/泰/希腊等→各自专用。\n"
            "    即：只有跨文字系才会换模型，拉丁语系内部（法/德/西/越…）切换同样不改变结果。\n"
            "\n"
            "【EasyOCR】只有这个引擎的语言是逐项真正生效的：英/法/意/西 各加载不同模型（仅 CPU）。\n"
            "\n"
            "💡 「中文模式能不能认英文？」能，而且认出率不低 ——\n"
            "   通用（中文）字典本身就内置整套拉丁字母：\n"
            "     v6 通用字典 462 个拉丁字符（与专用拉丁模型的 462 个完全一致）\n"
            "     v5 通用字典 145 个 · ncnn v1 字典 86 个（a-z / A-Z / 0-9 三份字典都齐全）\n"
            "   所以中英混排文档直接选「中文」即可，不必切英文。\n"
            "   但专用模型专注度更高（英文模型字典仅 62 字符、拉丁模型 770 字符），\n"
            "   在纯英文或带重音/变音符号的拉丁文字上精度更好 —— 这类文档再选专用英/拉丁模型。\n"
            "\n"
            "⚠ 「PP-OCRv5 (Paddle CPU)」引擎按语言切分语种专用模型、语言真正生效；\n"
            "   ncnn 引擎里的「PP-OCRv5 模型」只是单份 v5 字典、语言不生效 —— 二者不同。"
        )
        self._lang_tip_base = self.lang_combo.toolTip()   # ncnn 分支会临时改写 tooltip
        self.lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        el.addWidget(self.lang_combo)
        el.addStretch()
        # GPU设备选择（ncnn Vulkan 与 PP-OCRv6 ONNX CUDA 下显示；其余引擎连标签一起隐藏）
        self.gpu_label = QLabel("GPU:")
        self.gpu_label.setVisible(False)
        el.addWidget(self.gpu_label)
        self.gpu_combo = QComboBox()
        self.gpu_combo.setMinimumWidth(200)
        self.gpu_combo.setToolTip("选择Vulkan GPU设备。自动=优先独立显卡。仅ncnn Vulkan生效")
        self.gpu_combo.setVisible(False)
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
        self.scale_combo.addItems(["1x(快速)", "2x（高清）", "3x(超清)"])
        self.scale_combo.setCurrentIndex(self.cfg.value("scale", 1, type=int))
        self.scale_combo.setToolTip(
            "PDF页面渲染倍率 (小白推荐: 2x):\n"
            "  1x = 最快，但过小/过密文字可能识别不全\n"
            "  2x（高清）(推荐) = 高清与速度的平衡，适用大部分文档\n"
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
        self.vertical_check = QCheckBox("竖排识别")
        self.vertical_check.setChecked(self.cfg.value("vertical", False, type=bool))
        self.vertical_check.setToolTip(
            "竖排识别 (v1.2.3 起真正生效):\n"
            "  勾选后识别结果按竖排阅读顺序排列（右→左、列内上→下），\n"
            "  且竖排文字以竖排方式写入PDF（可正常搜索）\n"
            "  古籍/碑帖/对联等竖排文档 → 建议勾选\n"
            "  普通横排文档 → 不勾选，保持横排行为"
        )
        ol.addWidget(self.vertical_check)
        self.overwrite_ocr_check = QCheckBox("覆盖旧OCR")
        self.overwrite_ocr_check.setChecked(self.cfg.value("overwrite_ocr", False, type=bool))
        self.overwrite_ocr_check.setToolTip(
            "覆盖旧OCR (替换双层文本):\n"
            "  勾选后导出双层PDF时物理删除原PDF旧文字层，\n"
            "  只保留本次新识别文字层（图像与矢量图形无损保留）\n"
            "  适用于PDF自带效果差的旧OCR层，希望整体替换"
        )
        ol.addWidget(self.overwrite_ocr_check)
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
        # 支持 Ctrl / Shift 多选，便于批量删除
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setToolTip("支持 Ctrl / Shift 多选；选中后按 Delete 键可删除")
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
        self.remove_sel_btn = QPushButton("删除选中")
        self.remove_sel_btn.setToolTip(
            "从待处理列表移除选中的文件（按住 Ctrl / Shift 可多选）\n"
            "只移除列表条目，不会删除磁盘上的任何文件")
        self.remove_sel_btn.clicked.connect(self.remove_selected_files)
        fb.addWidget(self.remove_sel_btn)
        self.prune_done_btn = QPushButton("剔除已完成")
        self.prune_done_btn.setToolTip(
            "一次性移除「目标目录里已经有 _result.txt + _layered.pdf」的文件\n"
            "（用于中断后重跑：只跑还没导出的那些）\n"
            "只移除列表条目，不会删除任何文件")
        self.prune_done_btn.clicked.connect(self.prune_completed_files)
        fb.addWidget(self.prune_done_btn)
        ml.addLayout(fb)
        # 选中列表项后按 Delete 键 = 删除选中（仅在列表获得焦点时生效）
        _del_sc = QShortcut(QKeySequence.Delete, self.file_list, self.remove_selected_files)
        _del_sc.setContext(Qt.WidgetWithChildrenShortcut)
        layout.addWidget(mg)

        # === 输出 ===
        og2 = QGroupBox("输出目录")
        ol2 = QHBoxLayout(og2)
        self.output_edit = QLineEdit()
        # 输出目录默认留空 = 输出到每个 PDF 自己所在的目录。
        # 不用上次记住的路径回填：否则「默认值」实际是个旧路径，容易误导出到别处。
        # 上次用过的目录仍记在 _last_output_dir，只作为「浏览」对话框的起始位置。
        self.output_edit.setPlaceholderText("留空 = 输出到每个 PDF 所在的目录（原目录）")
        ol2.addWidget(self.output_edit)
        self.browse_output_btn = QPushButton("浏览...")
        self.browse_output_btn.clicked.connect(self.browse_output)
        ol2.addWidget(self.browse_output_btn)
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
        self._job_total_pages = 0
        self._last_speed_pages = 0
        self._last_speed_time = 0
        self.setAcceptDrops(True)
        self.file_list.setAcceptDrops(True)
        self._update_model_combo()
        self._on_engine_changed()
        self._ui_ready = True  # 界面已就绪，之后才允许弹出「引擎/语言不匹配」提示

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        # 处理中不接受拖入：列表内容已被本次任务取走，再往里加会造成
        # 「任务跑的是旧列表、界面显示的是新列表」的错位
        if self._processing:
            self.log("⚠ 处理中：不接受拖入文件（等本批跑完或先停止）")
            return
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith('.pdf'):
                self._add_file(path)
            elif os.path.isdir(path):
                self._add_pdf_from_folder(path)

    def _add_file(self, path):
        if self._processing:
            return
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
        eid = self.engine_combo.currentData()
        if not eid or eid not in ENGINE_REGISTRY:
            self.model_combo.blockSignals(False)
            return
        # 动态检测ncnn引擎的所有可用模型(包含有.param但无.bin的)
        if eid in ('ncnn_vulkan', 'ncnn_cpu'):
            options = _get_ncnn_model_options(eid)
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
        else:
            models = ENGINE_REGISTRY[eid]["models"]
            if models:
                for value, label in models:
                    self.model_combo.addItem(label, value)
            elif eid == "win7_v5":
                # PP-OCRv5 (Paddle CPU)：模型不在这里选，而是由「语言」决定 ——
                # 服务端按语言路由到官方分语种模型（universal/latin/korean/eslav/cyrillic/
                # arabic/devanagari/th/el/te/ta），故不再显示"中文"这种会误导的标签。
                self.model_combo.addItem("按语言自动选择 (官方分语种模型)", "universal")
            else:
                self.model_combo.addItem("中文 (默认)", "chinese")
        # 单模型 / 语言驱动的引擎：模型下拉无意义，置灰
        self.model_combo.setEnabled(eid not in ("win7_v5", "win7_classic"))
        last_model = self.cfg.value("model_val", "")
        idx = self.model_combo.findData(last_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

    def _update_mode_combo(self):
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        eid = self.engine_combo.currentData()
        einfo = ENGINE_REGISTRY.get(eid, {})
        self.mode_combo.addItem("自动(推荐)", "auto")
        if einfo.get("gpu"):
            self.mode_combo.addItem("GPU模式", "gpu")
        if einfo.get("cpu"):
            self.mode_combo.addItem("CPU模式", "cpu")
        idx0 = self.mode_combo.findData("auto")
        if idx0 >= 0:
            if eid == "ncnn_vulkan":
                tip = ("自动:\n"
                       "检测到独显时使用独显（支持NVIDIA/AMD/Intel）\n"
                       "无独显时自动回退CPU\n"
                       "双实例可提升35%+速度")
            elif eid == "umi_plugin_v6":
                tip = ("自动:\n"
                       "查看CUDA是否可用，可用则GPU，否则CPU回退\n"
                       "仅支持NVIDIA显卡+安装CUDA 12.x\n"
                       "AMD/Intel显卡自动回退CPU")
            else:
                tip = "自动:\n优先使用GPU（如果支持），GPU无效时CPU回退"
            self.mode_combo.setItemData(idx0, tip, Qt.ToolTipRole)
        idx1 = self.mode_combo.findData("gpu")
        if idx1 >= 0:
            if eid == "ncnn_vulkan":
                tip = "GPU模式:\n强制使用已选Vulkan GPU设备"
            elif eid == "umi_plugin_v6":
                tip = ("GPU模式:\n"
                       "强制使用CUDA GPU加速\n"
                       "若CUDA不可用会自动回退CPU")
            else:
                tip = "GPU模式:\n强制使用GPU加速，若GPU不可用则报错"
            self.mode_combo.setItemData(idx1, tip, Qt.ToolTipRole)
        idx2 = self.mode_combo.findData("cpu")
        if idx2 >= 0:
            self.mode_combo.setItemData(idx2,
                "CPU模式:\n仅使用CPU推理，不加载GPU模块",
                Qt.ToolTipRole)
        last_mode = self.cfg.value("mode_val", "auto")
        idx = self.mode_combo.findData(last_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        self.mode_combo.blockSignals(False)

    def _on_engine_changed(self):
        eid = self.engine_combo.currentData()
        # 记录切换前的语言：用于「新引擎不支持原语言」的提示
        prev_lang = self._current_lang_name()
        prev_code = dict(self._LANG_ITEMS).get(prev_lang)
        self._update_model_combo()
        self._update_lang_combo()
        self._update_mode_combo()
        self.rec_batch_spin.setVisible(eid.startswith('umi_plugin_v6'))
        self.shrink_check.setVisible(eid.startswith('umi_plugin_v6'))
        self.tensorrt_check.setVisible(False)
        # GPU设备选择：ncnn Vulkan → Vulkan 设备；PP-OCRv6 ONNX CUDA → NVIDIA(CUDA) 设备。
        # 其余引擎既不支持 GPU，就「标签 + 下拉」一起隐藏，避免出现孤零零的「GPU:」。
        is_vulkan = (eid == 'ncnn_vulkan')
        is_cuda = bool(eid) and eid.startswith('umi_plugin_v6')
        self.gpu_label.setVisible(is_vulkan or is_cuda)
        self.gpu_combo.setVisible(is_vulkan or is_cuda)
        if is_vulkan:
            self._populate_gpu_combo("vulkan")
        elif is_cuda:
            self._populate_gpu_combo("cuda")
        # 注：ncnn 的 GPU/CPU 完全由「模式」下拉决定（CPU模式 = use_gpu=False，与 GPU 同一二进制），
        #     已无独立的 ncnn_cpu 引擎，故此处不再做「引擎 ↔ 模式」的联动切换。
        #     即：ncnn Vulkan + CPU模式 是一个合法且必要的状态（纯 CPU 机器就用它）。
        # 新引擎不支持原语言 → 提示可一键跳转到推荐引擎
        # （初始化 / 程序化切换 / 窗口尚未显示时不提示，避免干扰与无谓弹窗）
        if (getattr(self, "_ui_ready", False)
                and not getattr(self, "_engine_guard_suppress", False)
                and self.isVisible()):
            self._maybe_warn_lang_unsupported(eid, prev_lang, prev_code)

    def _on_mode_changed(self):
        """模式切换时处理逻辑"""
        mode = self.mode_combo.currentData()
        # 注：ncnn 的 CPU 运行已统一由 ncnn Vulkan 的「CPU模式」(use_gpu=False) 承担，
        #     与 Vulkan 是同一个二进制，故不再存在独立的 ncnn_cpu 引擎，
        #     模式切换也不需要再联动切换引擎。

        # CPU模式下禁用双实例
        if mode == "cpu":
            self.dual_check.setChecked(False)
            self.dual_check.setEnabled(False)
            self.dual_check.setToolTip(
                "❌ CPU模式下双实例无意义(反而多耗内存)，已自动关闭\n"
                "如需开启请切换回 自动 或 GPU 模式"
            )
        else:
            self.dual_check.setEnabled(True)
            self.dual_check.setToolTip(
                "双实例并行 (小白推荐: GPU开启):\n"
                "  启动两个OCR进程并行处理一页PDF\n"
                "  可提升GPU利用率30%~50%\n"
                "  双实例会增加约1GB显存占用"
            )

    def _nvidia_tier(self):
        """探测本机 NVIDIA 档位：返回 0(≥12GB) / 1(≤8GB) / None(无 NVIDIA 或探测失败)。

        只发一次 nvidia-smi，比跑整套 _auto_detect_gpu_idx 便宜
        （专业→简单同步时每次都要判一次）。"""
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8)
            if r.returncode != 0 or not r.stdout.strip():
                return None
            vram = max(int(x) for x in r.stdout.split() if x.strip().isdigit())
            return 0 if vram >= 11000 else 1  # ≈12GB 分界
        except Exception:
            return None

    # ── 显卡档位常量（simple_gpu 下拉索引）──
    #   0 NV≥12G | 1 NV≤8G | 2 A/I≥12G | 3 A/I≤8G | 4 核显·纯CPU | 5 自动检测
    #   区别只在「精度优先」模式的边长：大显存(或CPU) 2560，小显存 2240（防爆显存）
    _GPU_BIG = (0, 2, 4)
    _GPU_SMALL = (1, 3)
    # 简单模式统一使用的识别批处理数（不随精度档变化）
    _SIMPLE_REC_BATCH = 16

    def _auto_detect_gpu_idx(self):
        """自动检测显卡，返回 simple_gpu 的最佳索引

        索引含义：0=NVIDIA≥12GB  1=NVIDIA≤8GB  2=AMD/Intel≥12GB  3=AMD/Intel≤8GB  4=核显/纯CPU
        """
        # 方式1：nvidia-smi 检测 NVIDIA 独显 + 显存（返回值就是 0/1）
        nv = self._nvidia_tier()
        if nv is not None:
            return nv

        # 方式2：ncnn Vulkan 探测独显（nvidia-smi 不可用时）
        #   注意：Vulkan 探测拿不到显存大小，AMD/Intel 一律保守按"小显存档"(3)；
        #   用户若是 ≥12GB 的 A/I 卡，可在下拉里手动改选第 3 项。
        try:
            for dev in self._detect_vulkan_gpus():
                if not dev.get("dedicated"):
                    continue
                name = dev.get("name", "").lower()
                if any(k in name for k in ("nvidia", "geforce", "rtx", "gtx")):
                    return 1  # NVIDIA（显存未知，按小显存档）
                return 3      # AMD / Intel 独显
        except Exception:
            pass

        # 方式3：默认 → 核显 / 纯CPU
        return 4

    def _on_ui_mode_changed(self, simple_mode):
        """切换简单模式/专业模式"""
        # 【安全闸】处理中禁止切换界面模式。
        # 原因：切到简单模式会调用 _sync_simple_from_expert → _apply_simple_settings，
        # 直接改写引擎/模型/显卡/边长等参数；此时这批参数已被 BatchWorkerThread 取走用于
        # 正在跑的任务，界面显示与任务实际参数会就此分叉，用户再也看不出到底在用什么配置。
        # 这里直接拦掉并把单选按钮还原到切换前的状态。
        if self._processing:
            self.ui_simple_btn.blockSignals(True)
            self.ui_expert_btn.blockSignals(True)
            self.ui_simple_btn.setChecked(not simple_mode)
            self.ui_expert_btn.setChecked(simple_mode)
            self.ui_simple_btn.blockSignals(False)
            self.ui_expert_btn.blockSignals(False)
            self.log("⚠ 处理中：不能切换「简单模式 / 专业模式」")
            return
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
        # 语言勾选框在下方设置时逐个 blockSignals

        eid = self.engine_combo.currentData()
        mode = self.mode_combo.currentData()
        # 显卡：根据引擎推断（_nv: 0/1=NVIDIA 大小显存, None=非 NVIDIA）
        _nv = self._nvidia_tier()
        if eid in ("ncnn_cpu", "win7_v5", "win7_classic"):
            self.simple_gpu.setCurrentIndex(4)  # 核显/纯CPU
        elif eid == "ncnn_vulkan":
            if mode == "cpu":
                self.simple_gpu.setCurrentIndex(4)  # ncnn Vulkan 的 CPU 模式 ≡ 纯 CPU
            else:
                # 任意独显：是 NVIDIA 就按 NVIDIA 档；否则按 AMD/Intel
                # （Vulkan 探测读不到显存 → 保守落在小显存档 3）
                self.simple_gpu.setCurrentIndex(_nv if _nv is not None else 3)
        else:
            # v6 (CUDA) 引擎：按实测显存落 NVIDIA 档（小显存在精度优先下套用 2240 防爆显存规则），
            # 探测不到 NVIDIA 才回退到大显存档
            self.simple_gpu.setCurrentIndex(_nv if _nv is not None else 0)
        # 精度：根据模型判断
        model = self.model_combo.currentText()
        shrink = self.shrink_check.isChecked()
        if "small" in model or "tiny" in model:
            self.simple_speed.setCurrentIndex(0)  # 速度优先
        elif shrink:
            self.simple_speed.setCurrentIndex(2)  # 精度优先
        else:
            self.simple_speed.setCurrentIndex(1)  # 标准平衡
        # 文档：竖排是最可靠的标志（边长/渲染倍率已不再随文档类型变化）
        vertical = self.vertical_check.isChecked()
        angle = self.angle_cls_check.isChecked()
        if vertical:
            self.simple_doc.setCurrentIndex(1)  # 古籍竖排
        elif angle:
            self.simple_doc.setCurrentIndex(2)  # 扫描件
        else:
            self.simple_doc.setCurrentIndex(0)  # 普通文档
        # 语言：专业模式完整项 → 按代码归并到简单模式大类
        _pro_code = dict(self._LANG_ITEMS).get(self._current_lang_name(), "ch")
        # 拉丁/西里尔其他语种归并到对应大类
        _LATIN_SIMPLE = {"fr", "de", "es", "it", "pt", "nl", "ro", "ca", "gl", "da", "sv",
                         "no", "fi", "is", "pl", "cs", "sk", "hu", "hr", "sl", "bs",
                         "rs_latin", "sq", "ga", "cy", "et", "lt", "lv", "mt", "la", "pi",
                         "af", "az", "uz", "ku", "eu", "oc", "vi", "id", "ms", "tl", "sw",
                         "mi", "tr"}
        _CYR_SIMPLE = {"bg", "mk", "mn", "kk", "ky", "tg", "tt", "ba", "cv", "rs_cyrillic"}
        if _pro_code in _LATIN_SIMPLE:
            _pro_code = "fr"
        elif _pro_code in _CYR_SIMPLE:
            _pro_code = "ru"
        elif _pro_code in ("fa", "ug", "ur", "ps", "sd", "ks", "bal"):
            _pro_code = "ar"
        elif _pro_code in ("mr", "ne", "sa", "bh", "mai", "kok"):
            _pro_code = "hi"
        elif _pro_code == "multilang_v5":
            _pro_code = "multilang_v6"
        # 勾选对应语言组（仅勾该项，其余清空）
        for _cb in self.simple_lang_checks.values():
            _cb.blockSignals(True)
            _cb.setChecked(False)
            _cb.blockSignals(False)
        _cb_target = self.simple_lang_checks.get(_pro_code)
        if _cb_target is not None:
            _cb_target.blockSignals(True)
            _cb_target.setChecked(True)
            _cb_target.blockSignals(False)

        self.simple_gpu.blockSignals(False)
        self.simple_speed.blockSignals(False)
        self.simple_doc.blockSignals(False)
        # 语言勾选框已逐个解除 blockSignals

        # 刷新 GPU 悬停提示（信号被阻断后需要手动调用）
        gpu_tips = [
            "NVIDIA 大显存 → ONNX CUDA 引擎，精度最高；精度优先时边长 2560",
            "NVIDIA 小显存同样能用 CUDA（8GB 实测可用）；普通语言走 ncnn Vulkan 更省显存、"
            "可双实例加速，勾选⚠语种时自动切到 CUDA 引擎；精度优先时边长 2240",
            "AMD / Intel 大显存独显 → ncnn Vulkan 引擎+双实例；精度优先时边长 2560",
            "AMD / Intel 小显存独显 → ncnn Vulkan 引擎+双实例；精度优先时边长 2240",
            "纯CPU运行，无显存限制；精度优先时边长 2560，但速度最慢",
            "不知道有没有独立显卡，或不知道显存多大？选这项，自动检测后帮你落到上面某一档",
        ]
        self.simple_gpu.setToolTip(gpu_tips[self.simple_gpu.currentIndex()])

        self._apply_simple_settings()

    def _apply_simple_settings(self):
        """将简单模式的3个选择应用到实际参数"""
        # 【安全闸】处理中禁止改写识别参数（同 _on_ui_mode_changed 的理由）
        if self._processing:
            return
        # 【强制纠正】勾了分语种文字+其它语言 → 直接改掉勾选并亮红色警告条
        # （2026-10-09 用户要求：警告不能不显眼，必须强制纠正用户的选择）
        self._enforce_simple_lang_selection()
        doc_idx = self.simple_doc.currentIndex()
        speed_idx = self.simple_speed.currentIndex()
        gpu_idx = self.simple_gpu.currentIndex()

        # ── 0. 自动检测（我不知道选哪个）──
        if gpu_idx == 5:
            detected = getattr(self, "_auto_detecting", False)
            if detected:
                # 防止递归，直接降级为 CPU
                self._auto_detecting = True
                self.simple_gpu.blockSignals(True)
                self.simple_gpu.setCurrentIndex(4)
                self.simple_gpu.blockSignals(False)
                self._auto_detecting = False
                # 用 CPU 分支继续执行
                gpu_idx = 4
            else:
                # 首次：运行自动检测
                self._auto_detecting = True
                best = self._auto_detect_gpu_idx()
                self._auto_detecting = False
                # 切换到检测结果（会触发递归调用）
                self.simple_gpu.blockSignals(True)
                self.simple_gpu.setCurrentIndex(best)
                self.simple_gpu.blockSignals(False)
                # 通过blockSignals防止setCurrentIndex触发第二次apply
                # 但我们需要手动执行
                gpu_idx = best  # 继续走正确的分支

        # ── 1. 先设定引擎（开放信号，触发 _on_engine_changed 更新下拉选项） ──
        target_mode = None  # 稍后设置
        checked_langs = self._simple_langs_checked()
        target_engine, downgraded = self._simple_pick_engine(checked_langs, gpu_idx)

        idx = self.engine_combo.findData(target_engine)
        if idx >= 0:
            # 程序化切换引擎：抑制「引擎不支持原语言」提示，避免简单模式同步时弹窗
            self._engine_guard_suppress = True
            try:
                self.engine_combo.setCurrentIndex(idx)
            finally:
                self._engine_guard_suppress = False
            # _on_engine_changed 已触发，model/mode/lang 下拉已更新

        # ── 根据速度和文档类型计算参数 ──
        if speed_idx == 0:  # 速度优先
            target_model = "small"
            target_precision = "fp16"
            target_shrink = False
        elif speed_idx == 1:  # 标准平衡
            target_model = "medium"
            target_precision = "fp32"
            target_shrink = False
        else:  # 精度优先
            target_model = "medium"
            target_precision = "fp32"
            target_shrink = True

        # 识别批处理数：与精度档位无关，速度/标准/精度三档统一（_SIMPLE_REC_BATCH = 16）
        target_rec_batch = self._SIMPLE_REC_BATCH

        # 文档类型只决定「竖排 / 方向矫正」开关；边长与渲染倍率由「速度档 + 显存档」统一决定
        if doc_idx == 0:  # 普通文档
            target_vertical = False
            target_angle = False
        elif doc_idx == 1:  # 古籍竖排
            target_vertical = True
            target_angle = False
        else:  # 扫描件
            target_vertical = False
            target_angle = True

        # ── 边长：速度优先/标准一律 2240；精度优先 大显存(或CPU) 2560 / 小显存 2240 ──
        if speed_idx == 2:  # 精度优先
            target_side = 2560 if gpu_idx in self._GPU_BIG else 2240
        else:               # 速度优先(0) / 标准平衡(1)
            target_side = 2240

        # ── 渲染倍率：只有「古籍 + 精度优先 + 大显存(或CPU)」用 3x，其余一律 2x ──
        if doc_idx == 1 and speed_idx == 2 and gpu_idx in self._GPU_BIG:
            target_scale = 2  # 3x
        else:
            target_scale = 1  # 2x

        target_dual = True  # 占位，稍后在设 mode 后重新计算

        # ── 2. 批量设置其余参数（拦截信号避免级联触发） ──
        self.engine_combo.blockSignals(True)
        self.mode_combo.blockSignals(True)
        self.model_combo.blockSignals(True)
        self.side_len_spin.blockSignals(True)
        self.scale_combo.blockSignals(True)
        self.vertical_check.blockSignals(True)
        self.angle_cls_check.blockSignals(True)
        self.rec_batch_spin.blockSignals(True)
        self.shrink_check.blockSignals(True)
        self.dual_check.blockSignals(True)
        self.lang_combo.blockSignals(True)

        # ── 根据显卡选择引擎（已设置）和模式 ──
        if gpu_idx in (0, 1, 2, 3):  # 独显（任意品牌/显存）→ GPU 自动模式
            target_mode = "auto"
        else:  # 核显 / 纯CPU
            target_mode = "cpu"
        target_dual = (target_mode != "cpu")
        midx = self.mode_combo.findData(target_mode)
        if midx >= 0:
            self.mode_combo.setCurrentIndex(midx)
        # 设置模型
        midx2 = self.model_combo.findText(target_model, Qt.MatchContains)
        if midx2 >= 0:
            self.model_combo.setCurrentIndex(midx2)
        # 设置图像边长
        self.side_len_spin.setValue(target_side)
        # 设置渲染倍率
        self.scale_combo.setCurrentIndex(target_scale)
        # 设置选项
        self.vertical_check.setChecked(target_vertical)
        self.angle_cls_check.setChecked(target_angle)
        self.rec_batch_spin.setValue(target_rec_batch)
        self.shrink_check.setChecked(target_shrink)
        self.dual_check.setChecked(target_dual)
        # 设置语言（勾选项主语言 → 专业模式完整项）
        _simple_code, simple_full = self._simple_primary_lang()
        lang_idx = self._lang_index_for(simple_full)
        if lang_idx >= 0:
            self.lang_combo.setCurrentIndex(lang_idx)

        self.engine_combo.blockSignals(False)
        self.mode_combo.blockSignals(False)
        self.model_combo.blockSignals(False)
        self.side_len_spin.blockSignals(False)
        self.scale_combo.blockSignals(False)
        self.vertical_check.blockSignals(False)
        self.angle_cls_check.blockSignals(False)
        self.rec_batch_spin.blockSignals(False)
        self.shrink_check.blockSignals(False)
        self.dual_check.blockSignals(False)
        self.lang_combo.blockSignals(False)

        # ── 更新配置摘要 ──
        # 引擎显示名统一走注册表：原先是用内部 id 做字符串 replace 拼名字，
        # 漏了 win7_v5 → 简单模式摘要里会冒出「win7_v5」，与引擎下拉框叫法不一致。
        engine_name = engine_display_name(target_engine)
        if target_engine == "ncnn_cpu":          # 已隐藏的引擎，未注册，兜一个名字
            engine_name = "PP-OCR (ncnn CPU)"
        model_label = target_model
        gpu_text = "(GPU)" if target_mode != "cpu" else "(CPU)"
        scale_label = ["1x", "2x", "3x"][target_scale]
        vertical_label = "开" if target_vertical else "关"
        angle_label = "开" if target_angle else "关"
        shrink_label = "开" if target_shrink else "关"
        dual_label = "开" if target_dual else "关"
        precision_label = target_precision.upper()

        # GPU 设备信息（注意：变量名不要用 gpu_idx —— 那是上面的"显卡档位"，别被覆盖）
        gpu_dev_text = ""
        try:
            dev_idx = self.gpu_combo.currentData()
            gpu_name = self.gpu_combo.currentText()
            if dev_idx is not None and dev_idx >= 0:
                gpu_dev_text = gpu_name
            elif dev_idx is not None and dev_idx == -1:
                gpu_dev_text = "自动 (优先独显)"
        except Exception:
            gpu_dev_text = "—"

        # 语言
        lang_text = self._current_lang_name() if hasattr(self, 'lang_combo') else "—"

        # 一行灰色摘要
        preview_parts = [f"{engine_name} {gpu_text}", f"模型 {model_label}", f"{lang_text}",
                        f"边长{target_side}", f"渲染{scale_label}",
                        f"批数{target_rec_batch}", f"精度{precision_label}",
                        f"双实例{dual_label}", f"对齐{shrink_label}"]
        if target_vertical:
            preview_parts.append("竖排开")
        if target_angle:
            preview_parts.append("方向矫正")
        if downgraded:
            if target_engine == "umi_plugin_v6":
                preview_parts.append("⚠ 已切换：勾选⚠语言 → PP-OCRv6 引擎自动回退分语种模型")
            elif target_engine == "ncnn_vulkan":
                preview_parts.append(
                    "⚠ 已切换：语言含希腊文 → 改走 ncnn Vulkan"
                    "（CUDA 下希腊文会降级成单文字系专用模型、汉字/英文会丢；"
                    "ncnn 的 v6 字典同时含汉字与希腊字符）")
            elif _simple_code in self._CYRILLIC_SIMPLE_CODES:
                preview_parts.append(
                    "⚠ 已切换：西里尔文 → PP-OCRv5 (Paddle CPU) 引擎"
                    "（实测比 v6 引擎的 ONNX 版更准且更快：«Русский» 不会被认成 «Russkiy»）")
            else:
                preview_parts.append("⚠ 已切换：勾选⚠语言 → PP-OCRv5 (Paddle CPU) 备选引擎")
            # 硬文字（韩/俄/阿/天城/泰/泰卢固/泰米尔）都是「一个文字系一个模型」，
            # 与别的语言同时勾选时做不到混合 —— 必须说清，避免用户以为全都识别。
            if self._mixed_langs_problem():
                _others = [f for _d, _c, f in self._SIMPLE_LANG_ITEMS
                           if _c in checked_langs and _c != _simple_code]
                preview_parts.append(
                    "⚠ 分语种模型一次只能认一个文字系、做不到混合识别 —— "
                    f"本次实际只按「{lang_text}」识别"
                    + (f"，另勾的 {len(_others)} 项（{'、'.join(_others)}）会被忽略"
                       if _others else "")
                    + "；要混合请拆成多次处理")
        if gpu_dev_text and gpu_dev_text != "—":
            preview_parts.append(gpu_dev_text)
        preview_text = " · ".join(preview_parts)
        self.simple_preview.setText(f"当前配置：{preview_text}")

    def _current_lang_name(self):
        """当前选中语言的原名（不含「部分支持／此模型不支持」等显示后缀）。

        语言下拉的 itemData 一律存「语言原名」（_LANG_ITEMS 里的那个名字），
        显示文本才带后缀；引擎路由 / 持久化都必须用这个名字。
        """
        data = self.lang_combo.currentData()
        if isinstance(data, str) and data:
            return data
        if data is None:
            return ""            # 「说明行」（itemData=None）不是有效语言
        return self.lang_combo.currentText()

    def _lang_index_for(self, want):
        """按 lang_val 还原语言选择（新版存语言原名、旧版存带后缀文本，两种都认）。

        落在「灰显不可选」项上时返回 -1，由调用方回落到首项。
        """
        if not want:
            return -1
        i = self.lang_combo.findData(want)
        if i < 0:
            i = self.lang_combo.findText(want)
        if i < 0:                                  # 旧值可能是「名 + 后缀」
            for k in range(self.lang_combo.count()):
                if self.lang_combo.itemText(k).startswith(str(want)):
                    i = k
                    break
        if i < 0:                                  # 旧值可能是别的版本的显示写法
            for label, code in self._LANG_ITEMS:
                if label == want:
                    i = self.lang_combo.findData(code)
                    break
        try:
            if i >= 0 and not self.lang_combo.model().item(i).isEnabled():
                return -1                          # 灰显项 → 不选它
        except Exception:
            pass
        return i

    def _on_lang_changed(self):
        eid = self.engine_combo.currentData()
        if not eid:
            return
        # 【修复】存「语言原名」而不是下拉里显示的文本 ——
        # 显示文本带「（部分支持·可能缺重音）」这类后缀，存进去会跨次启动层层累积。
        self.cfg.setValue("lang_val", self._current_lang_name())

    def _update_lang_combo(self):
        self.lang_combo.blockSignals(True)
        self.lang_combo.clear()
        tip_base = getattr(self, "_lang_tip_base", "")
        eid = self.engine_combo.currentData()
        if not eid:
            self.lang_combo.blockSignals(False)
            return
        is_v6_engine = eid.startswith("umi_plugin_v6")
        is_ncnn = eid in ("ncnn_vulkan", "ncnn_cpu")
        if is_v6_engine:
            # umi_plugin_v6：48 项共用同一个 v6 模型（同一份字典）+ 34 项会自动切 v5 专用模型。
            # 两组之间插说明行，让「选哪个都一样」这件事在界面上直接可见（不必悬停）。
            # 注意 multilang_v5 不算「共用 v6 模型」—— 它会切到 PP-OCRv5 latin 模型
            # （2026-10-09 起服务端修好，此前选它会引擎初始化失败），归入第二组。
            native = [(n, c) for n, c in self._LANG_ITEMS
                      if c not in self._V6_V5_FALLBACK and c != "multilang_v5"]
            fallback = [(n, c) for n, c in self._LANG_ITEMS
                        if c in self._V6_V5_FALLBACK or c == "multilang_v5"]
            self._add_lang_note(
                f"ⓘ 多语言单模型：以下 {len(native)} 项共用同一个 PP-OCRv6 模型与同一份字典，"
                f"选哪个结果都一样")
            for name, _code in native:
                self.lang_combo.addItem(name, name)
            if fallback:
                self._add_lang_note(
                    f"── 以下 {len(fallback)} 项会自动切 PP-OCRv5 专用模型（语言真正生效 · 走 CPU）；"
                    "其中西里尔各项建议改选「PP-OCRv5 (Paddle CPU)」引擎（实测更准更快）──")
                for name, _code in fallback:
                    self.lang_combo.addItem(name, name)
            self.lang_combo.setToolTip(tip_base)
            self.lang_combo.setEnabled(True)
        elif is_ncnn:
            self._fill_ncnn_lang_combo(eid)
            self.lang_combo.setEnabled(True)
        elif eid == "easyocr_universal":
            self._add_lang_note("ⓘ 本引擎的语言是逐项真正生效的：英/法/意/西 各加载不同识别模型（仅 CPU）")
            for name in ("English (EasyOCR)", "Fran\u00e7ais (EasyOCR)",
                         "Italiano (EasyOCR)", "Espa\u00f1ol (EasyOCR)"):
                self.lang_combo.addItem(name, name)
            self.lang_combo.setToolTip(tip_base)
            self.lang_combo.setEnabled(True)
        elif eid == "win7_v5":
            # win7_v5 (PP-OCRv5 Paddle CPU)：按语言切官方专用模型。
            # 注意：只有跨文字系才换模型 —— 拉丁语系 40+ 种内部切换用同一份 latin 模型，结果不变。
            self._add_lang_note(
                "ⓘ 按语言切官方专用模型：中/英/日 共用 universal · 40+ 拉丁语系共用 latin · "
                "其余各自专用（同文字系内切换结果不变）")
            for name, _code in self._LANG_ITEMS:
                self.lang_combo.addItem(name, name)
            self.lang_combo.setToolTip(tip_base)
            self.lang_combo.setEnabled(True)
        else:
            self.lang_combo.addItem("中文", "chinese")
            self.lang_combo.setToolTip(tip_base)
            self.lang_combo.setEnabled(False)
        idx = self._lang_index_for(self.cfg.value("lang_val", "中文 (Chinese)"))
        if idx < 0:
            # 还原不到有效语言（空配置 / 旧值 / 落在说明行）→ 落到第一个可选语言项
            for _k in range(self.lang_combo.count()):
                if self.lang_combo.itemData(_k):
                    idx = _k
                    break
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        self.lang_combo.blockSignals(False)

    def _add_lang_note(self, text):
        """在语言下拉里插一条「说明行」：灰显、不可选。

        itemData 恒为 None —— 这样 findData() 永远不会命中它，
        _lang_index_for() / _current_lang_name() 也不会把它当成有效语言。
        """
        self.lang_combo.addItem(text, None)
        i = self.lang_combo.count() - 1
        try:
            item = self.lang_combo.model().item(i)
            if item is not None:
                item.setEnabled(False)
                item.setForeground(QBrush(QColor("#8b949e")))
        except Exception:
            pass
        return i

    def _fill_ncnn_lang_combo(self, eid):
        """按「当前模型实际用的那份字典」重建 ncnn 语言列表。

        四份字典的语言能力差别很大（v1 字典无假名/希腊文，v6_tiny 无假名……），
        所以语言列表必须随**模型**变化，不能一份写死的清单套到底：
          · 引擎级完全不支持（韩/西里尔/阿拉伯/天城/泰/泰卢固/泰米尔，四份字典皆 0 覆盖）
            → 不列出；
          · 本模型字典里没有该文字系 → 列出但**灰显、不可选**，标「此模型不支持」；
          · 本模型字典只覆盖基础字母 → 可选中，标「部分支持·可能缺重音」。
        语言框 tooltip 同步改写：写明「本模型 → 用哪份字典 → 覆盖了什么」。
        """
        model_base = self.model_combo.currentData() or ""
        keys_file = _ncnn_keys_file_for_model(model_base)
        info = _PLUGIN_DIRS.get(eid) or {}
        pool = _load_ncnn_dict_pool(info.get("plugin_dir", ""), keys_file)
        # 顶部说明行：把「语言在这里不参与识别」直接摆在列表最上面（不必悬停才看到）
        self._add_lang_note(
            "ⓘ ncnn 的语言只作声明、不参与识别 —— 字典由「模型」决定，下列语言共用同一份字典")
        brush_gray = QBrush(QColor("#8b949e"))
        brush_amber = QBrush(QColor("#b8860b"))
        hidden_langs, bad_langs, part_langs = [], [], []
        for name, code in self._LANG_ITEMS:
            if code in self._NCNN_UNSUPPORTED:
                hidden_langs.append(name)
                continue
            tier = _ncnn_lang_tier(code, pool)
            if tier == "unsupported":
                label = f"{name}（此模型不支持）"
            elif tier == "partial":
                label = f"{name}（部分支持·可能缺重音）"
            else:
                label = name
            self.lang_combo.addItem(label, name)     # itemData 始终存语言原名
            i = self.lang_combo.count() - 1
            if tier == "unsupported":
                try:
                    self.lang_combo.model().item(i).setEnabled(False)
                    self.lang_combo.model().item(i).setForeground(brush_gray)
                except Exception:
                    pass
                bad_langs.append(name)
            elif tier == "partial":
                try:
                    self.lang_combo.model().item(i).setForeground(brush_amber)
                except Exception:
                    pass
                part_langs.append(name)
        lines = [
            "ncnn 引擎：字典由「模型」决定，多个语言共用同一份字典 ——",
            "语言只表示「文档里可能出现哪些文字」，不改变识别结果。",
            "",
            f"当前模型：{model_base or '(未选)'}",
            f"所用字典：{keys_file}（{len(pool)} 个字符）",
            "",
            "💡 该字典内含整套 ASCII 拉丁字母与数字（a-z / A-Z / 0-9 齐全）——",
            "   所以「中文」模式下英文照样能识别，中英混排文档不必切到英文；",
            "   只是专门的拉丁/英文模型（字典更专注）精度更高，纯英文文档可选它们。",
        ]
        if bad_langs:
            lines += ["",
                      "✕ 下列文字系不在这份字典里，识别必乱码（已置灰、不可选）：",
                      "   " + "、".join(bad_langs)]
        if part_langs:
            lines += ["",
                      f"⚠ 另有 {len(part_langs)} 种语言这份字典只覆盖基础字母，",
                      "   选中后可能缺重音/变音符号（列表中已逐项标注「部分支持」）。"]
        if hidden_langs:
            lines += ["",
                      f"（另有 {len(hidden_langs)} 种文字系四份 ncnn 字典都不含，已整体隐藏：",
                      "   韩文、西里尔字母(俄/乌/白俄/保…)、阿拉伯字母(阿/波斯/维/乌尔都…)、",
                      "   天城文(印地/马拉地/尼泊尔…)、泰文、泰卢固文、泰米尔文）"]
        self.lang_combo.setToolTip("\n".join(lines))

    def _on_model_changed(self):
        """切「模型」→ 所用字典变了 → 重建语言列表（并提示被排掉的文字系）。"""
        prev = self._current_lang_name() if self.lang_combo.count() else ""
        self._update_lang_combo()
        now = self._current_lang_name() if self.lang_combo.count() else ""
        if prev and now and prev != now:
            self.log(f"[语言] 新模型所用字典不含「{prev}」，已自动回落到「{now}」")


    # ============================================================
    # 引擎 ↔ 语言 兼容性
    # ============================================================
    def _engine_lang_supported(self, eid, code):
        """判断某引擎是否支持某语言代码（据实测的字典/模型覆盖）。

          · umi_plugin_v6 —— 全部语言（普通语言走 v6 通用字典；⚠语言自动回退 PP-OCRv5 分语种模型）
          · win7_v5       —— 全部语言（语言 → 官方分语种模型路由）
          · ncnn_vulkan / ncnn_cpu —— 看「当前模型用的那份字典」：
                            引擎级 0 覆盖的文字系（韩/西里尔/阿拉伯/天城/泰/泰卢固/泰米尔）
                            一律不支持；其余按 _ncnn_lang_tier() 判（v1 字典无日文/希腊文，
                            v6_tiny 字典无日文）。与语言下拉的置灰口径完全一致。
          · easyocr_universal —— 仅 英/法/意/西
          · 其余（如 win7_classic）—— 宽松放行，不拦截
        """
        if not code:
            return True
        if eid in ("ncnn_vulkan", "ncnn_cpu"):
            if code in self._NCNN_UNSUPPORTED:
                return False
            try:
                model_base = self.model_combo.currentData() or ""
                info = _PLUGIN_DIRS.get(eid) or {}
                pool = _load_ncnn_dict_pool(info.get("plugin_dir", ""),
                                             _ncnn_keys_file_for_model(model_base))
                return _ncnn_lang_tier(code, pool) != "unsupported"
            except Exception:
                return True          # 判不了就放行，不误拦
        if eid == "easyocr_universal":
            return code in ("en", "fr", "it", "es")
        return True

    def _recommend_engine_for_lang(self, code):
        """给定语言代码，返回当前环境下「支持该语言」的推荐引擎 id（无则 None）。

        规则与简单模式一致：NVIDIA 机器优先 PP-OCRv6（⚠语言在其内部自动回退 v5 分语种模型），
        否则用 PP-OCRv5 (Paddle CPU) 备选。
        """
        avail = set(_PLUGIN_DIRS.keys())
        nv = self._nvidia_tier()  # 0/1 = NVIDIA，None = 非 NVIDIA / 探测失败
        if nv is not None and "umi_plugin_v6" in avail:
            return "umi_plugin_v6"
        if "win7_v5" in avail:
            return "win7_v5"
        if "umi_plugin_v6" in avail:
            return "umi_plugin_v6"
        return None

    def _switch_engine_preserving_lang(self, rec, lang_display):
        """切到推荐引擎，并尽量保持原语言仍被选中。返回新引擎 id（切换失败返回 None）。"""
        if not rec:
            return None
        idx = self.engine_combo.findData(rec)
        if idx < 0:
            return None
        self._engine_guard_suppress = True
        try:
            self.engine_combo.setCurrentIndex(idx)  # 触发 _on_engine_changed → 重建各下拉
            li = self._lang_index_for(lang_display)
            if li >= 0:
                self.lang_combo.setCurrentIndex(li)
        finally:
            self._engine_guard_suppress = False
        return rec

    def _resolve_engine_lang_mismatch(self, engine_id, lang_display, ocr_lang):
        """开跑前的「引擎-语言」守卫：当前引擎不支持所选语言时，

        弹窗提示并可**一键跳转到推荐引擎**。
        返回：最终应使用的 engine_id；用户选择「取消」时返回 None。
        """
        if self._engine_lang_supported(engine_id, ocr_lang):
            return engine_id
        cur_name = ENGINE_REGISTRY.get(engine_id, {}).get("name", engine_id)
        rec = self._recommend_engine_for_lang(ocr_lang)
        rec_name = ENGINE_REGISTRY.get(rec, {}).get("name", rec) if rec else "（无可用引擎）"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("引擎与语言不匹配")
        box.setText(f"当前引擎「{cur_name}」不支持「{lang_display}」的文字。")
        box.setInformativeText(
            "用当前引擎识别会输出乱码或空结果。\n\n"
            f"推荐改用：{rec_name}"
        )
        btn_switch = box.addButton(f"切换到 {rec_name}", QMessageBox.AcceptRole)
        btn_keep = box.addButton("仍用当前引擎", QMessageBox.DestructiveRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(btn_switch)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is btn_keep:
            return engine_id
        if clicked is btn_switch:
            return self._switch_engine_preserving_lang(rec, lang_display)
        return None  # 取消 / 关闭窗口

    def _maybe_warn_lang_unsupported(self, eid, prev_lang, prev_code):
        """用户切换引擎后，若新引擎不支持原语言 → 提示并可一键切回推荐引擎。"""
        # 处理中不弹此窗：一键切换会重建引擎参数，把正在跑的任务配置搅乱
        if self._processing:
            return
        if not prev_code or self._engine_lang_supported(eid, prev_code):
            return
        rec = self._recommend_engine_for_lang(prev_code)
        if not rec or rec == eid:
            return
        cur_name = ENGINE_REGISTRY.get(eid, {}).get("name", eid)
        rec_name = ENGINE_REGISTRY.get(rec, {}).get("name", rec)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("引擎不支持该语言")
        box.setText(f"「{cur_name}」不支持「{prev_lang}」的文字。")
        box.setInformativeText(
            f"是否切换到支持该语言的 {rec_name}？\n"
            "（若选择「保留」，语言会退回该引擎支持的第一项。）"
        )
        btn_switch = box.addButton(f"切换到 {rec_name}", QMessageBox.AcceptRole)
        box.addButton("保留当前引擎", QMessageBox.RejectRole)
        box.setDefaultButton(btn_switch)
        box.exec_()
        if box.clickedButton() is btn_switch:
            self._switch_engine_preserving_lang(rec, prev_lang)

    def _populate_gpu_combo(self, kind="vulkan"):
        """填充 GPU 设备下拉框。

        kind='vulkan' → ncnn Vulkan 用的设备（Vulkan 检测，含 AMD/Intel/NVIDIA）
        kind='cuda'   → PP-OCRv6 ONNX 用的设备（只列 NVIDIA —— 别的卡没有 CUDA 能力）
        两项选择分别存 gpu_device / gpu_device_cuda，互不覆盖。
        """
        self.gpu_combo.blockSignals(True)
        self.gpu_combo.clear()
        cfg_key = "gpu_device_cuda" if kind == "cuda" else "gpu_device"
        if kind == "cuda":
            self.gpu_combo.setToolTip(
                "选择 CUDA(NVIDIA) 设备（本机只有一块 NVIDIA 卡时保持「自动」即可）。\n"
                "仅 PP-OCRv6 (ONNX CUDA) 引擎生效")
        else:
            self.gpu_combo.setToolTip("选择Vulkan GPU设备。自动=优先独立显卡。仅ncnn Vulkan生效")
        devices = get_gpu_devices_for_ui()
        if kind == "cuda":
            devices = [d for d in devices if "nvidia" in str(d.get("name", "")).lower()]
        if not devices:
            self.gpu_combo.addItem("无检测到GPU" if kind != "cuda" else "无 NVIDIA(CUDA) 显卡", -1)
            self.gpu_combo.blockSignals(False)
            return
        # 自动选项
        if kind == "cuda":
            best_name = devices[0]["name"]
        else:
            auto_idx, auto_name = _select_best_gpu()
            best_name = auto_name if auto_idx >= 0 else "未知"
        self.gpu_combo.addItem(f"自动 (优先 {best_name})", -1)
        for d in devices:
            idx = self.gpu_combo.count()
            if kind == "cuda":
                self.gpu_combo.addItem(f"🎮 [CUDA:{d['index']}] {d['name']}", d['index'])
                self.gpu_combo.setItemData(idx, f"{d['name']}\n✓ 可 CUDA 加速", Qt.ToolTipRole)
                continue
            is_compat = d.get("supported", True)
            compat_flag = " ✅" if is_compat else " ❌"
            gpu_type = "🖥️" if d.get("dedicated") else "💻"
            label = f"{gpu_type} [{d['index']}] {d['name']} (score:{d['score']}){compat_flag}"
            self.gpu_combo.addItem(label, d['index'])
            tip = d['name'] + (" (独立显卡)" if d.get("dedicated") else " (集成显卡)")
            if not is_compat:
                tip += "\n低分GPU，走CPU可能更快"
            else:
                tip += "\n✓ 可GPU加速"
            self.gpu_combo.setItemData(idx, tip, Qt.ToolTipRole)
        # 还原上次选择（ncnn 与 v6 各用各的 key）
        try:
            want = int(self.cfg.value(cfg_key, -1))
        except Exception:
            want = -1
        idx = self.gpu_combo.findData(want)
        if idx < 0:                       # 存过的设备号不存在了（换卡/换机）→ 回落「自动」
            idx = self.gpu_combo.findData(-1)
        if idx >= 0:
            self.gpu_combo.setCurrentIndex(idx)
        self.gpu_combo.blockSignals(False)

    def get_selected_engine_id(self):
        return self.engine_combo.currentData()
    def get_use_gpu(self):
        mode = self.mode_combo.currentData()
        eid = self.engine_combo.currentData()
        einfo = ENGINE_REGISTRY.get(eid, {})
        if mode == "cpu":
            return False
        # auto/gpu mode: 检查引擎是否支持GPU
        engine_supports_gpu = einfo.get("gpu", False)
        if not engine_supports_gpu:
            return False
        # 对于 ONNX CUDA (umi_plugin_v6):
        # 直接信任 GPU 环境（服务器进程会自动回退 CPU），
        # 不再依赖 onnxruntime 的 get_available_providers() 检测——
        # 因为 UI 进程和子进程的 DLL 搜索路径可能不同。
        if eid == "umi_plugin_v6":
            if mode == "gpu":
                return True
            # auto 模式：尝试 CUDA，服务器会自动回退
            return True
        # 对于 ncnn Vulkan: 自动模式下，仅当有高分独显时才走 GPU
        # 低分GPU（特别是核显）走 GPU 效率反而不如 CPU
        if eid == "ncnn_vulkan":
            gpu_devices = _detect_vulkan_gpus()
            if mode == "auto":
                # 自动：找独立显卡 + supported（评分>=30）
                capable = [d for d in gpu_devices if d.get("supported") and d.get("dedicated")]
                if capable:
                    return True
                # 没有合适的独显 → CPU 模式
                return False
            # gpu 模式：强制 GPU（用户手动选的）
            return True
        # 其他引擎: 按注册的GPU能力返回
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
        if self._processing:
            QMessageBox.information(
                self, "处理中",
                "本批任务正在跑，暂时不能清空待处理列表。\n请先「取消」或等它跑完。")
            return
        self.file_list.clear()
        self._update_count()

    def _update_count(self):
        # 处理中不覆盖状态栏的「处理中...（参数已锁定）」
        if self._processing:
            return
        self.status_label.setText(f"已选择 {self.file_list.count()} 个文件")

    # ============================================================
    # 待处理列表管理：删除选中 / 一键剔除已完成
    # ============================================================
    def _item_output_dir(self, path):
        """某个待处理文件的输出目录。

        与 start_processing 的约定保持一致：输出目录留空 = 输出到该文件自己的源目录。
        """
        out = self.output_edit.text().strip()
        return out or os.path.dirname(path)

    def _outputs_done(self, path):
        """判断该文件是否「已处理完成、且已在目标目录导出」。

        完成时 write_results 会在同一目录里同时写下
        `{stem}_result.txt` 与 `{stem}_layered.pdf`。两个都在才算完成 ——
        TXT 是最先写的，任务中途崩掉会只留 TXT，所以不能只看 TXT。
        """
        out_dir = self._item_output_dir(path)
        stem = Path(path).stem
        txt = os.path.join(out_dir, f"{stem}_result.txt")
        pdf = os.path.join(out_dir, f"{stem}_layered.pdf")
        return os.path.isfile(txt) and os.path.isfile(pdf)

    def remove_selected_files(self):
        """从待处理列表移除选中的文件（支持多选）。只动列表，不动磁盘文件。"""
        if self._processing:
            QMessageBox.information(
                self, "处理中",
                "本批任务正在跑，暂时不能修改待处理列表。\n请先「取消」或等它跑完。")
            return
        items = self.file_list.selectedItems()
        if not items:
            QMessageBox.information(
                self, "未选中文件",
                "请先在列表里点选要删除的文件：\n\n"
                "· 按住 Ctrl 逐个多选\n"
                "· 按住 Shift 选中连续一段\n"
                "· 选中后直接按 Delete 键也可以\n\n"
                "（只从列表移除，不会删除磁盘上的文件）")
            return
        paths = [it.data(Qt.UserRole) for it in items]
        for it in items:
            self.file_list.takeItem(self.file_list.row(it))
        self._update_count()
        self.log(f"已从待处理列表移除 {len(items)} 个文件")
        for p in paths[:5]:
            self.log(f"   - {os.path.basename(p)}")
        if len(paths) > 5:
            self.log(f"   ... 其余 {len(paths) - 5} 个略")

    def prune_completed_files(self):
        """一键剔除「已完成、且已在目标目录导出结果」的待处理文件。

        判定：目标目录里同时存在 `{stem}_result.txt` 与 `{stem}_layered.pdf`。
        典型用途：一批任务中途停止/出错后，只把还没导出的那些留下重跑。
        ⚠ 只操作列表条目，绝不删除任何磁盘文件。
        """
        if self._processing:
            QMessageBox.information(
                self, "处理中",
                "本批任务正在跑，暂时不能修改待处理列表。\n请先「取消」或等它跑完。")
            return
        total = self.file_list.count()
        if total == 0:
            QMessageBox.information(self, "列表为空", "待处理列表里还没有文件。")
            return
        done_items, done_paths = [], []
        for i in range(total):
            it = self.file_list.item(i)
            p = it.data(Qt.UserRole)
            try:
                if self._outputs_done(p):
                    done_items.append(it)
                    done_paths.append(p)
            except Exception:
                continue  # 单个文件判定失败不影响其余
        if not done_items:
            out = self.output_edit.text().strip() or "（各文件自己的源目录）"
            QMessageBox.information(
                self, "没有已完成项",
                f"列表里 {total} 个文件，都没有在目标目录找到\n"
                f"「_result.txt + _layered.pdf」。\n\n当前输出目录：{out}")
            return
        out_desc = self.output_edit.text().strip() or "各文件自己的源目录"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("剔除已完成")
        box.setText(f"检测到 {len(done_items)} 个文件已完成导出（列表共 {total} 个）。")
        box.setInformativeText(
            f"输出目录：{out_desc}\n\n"
            f"将从列表移除这 {len(done_items)} 项，只留下还没导出的 "
            f"{total - len(done_items)} 项。\n"
            f"⚠ 只移除列表条目，不会删除任何文件。")
        btn_ok = box.addButton("剔除", QMessageBox.AcceptRole)
        box.addButton("算了", QMessageBox.RejectRole)
        box.setDefaultButton(btn_ok)
        box.exec_()
        if box.clickedButton() is not btn_ok:
            return
        for it in done_items:
            self.file_list.takeItem(self.file_list.row(it))
        self._update_count()
        self.log(f"已剔除 {len(done_items)} 个已完成文件，"
                 f"剩余 {self.file_list.count()} 个待处理")
        for p in done_paths[:10]:
            self.log(f"   ✔ {os.path.basename(p)}")
        if len(done_paths) > 10:
            self.log(f"   ... 其余 {len(done_paths) - 10} 个略")

    def _mark_item_done(self, filename):
        """把刚处理完的文件在列表里打上 ✔ 标记（按文件名匹配）。"""
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            base = os.path.basename(it.data(Qt.UserRole) or "")
            if base == filename and not it.text().startswith("✔"):
                it.setText(f"✔ {base}")
                it.setForeground(QBrush(QColor("#1a7f37")))
                return

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
        self.cfg.setValue("overwrite_ocr", self.overwrite_ocr_check.isChecked())
        self.cfg.setValue("angle_cls", self.angle_cls_check.isChecked())
        self.cfg.setValue("rec_batch", self.rec_batch_spin.value())
        self.cfg.setValue("shrink", self.shrink_check.isChecked())
        self.cfg.setValue("tensorrt", self.tensorrt_check.isChecked())
        self.cfg.setValue("dual", self.dual_check.isChecked())
        self.cfg.setValue("precision_idx", self.precision_combo.currentIndex())
        self.cfg.setValue("engine_id", self.engine_combo.currentData())
        self.cfg.setValue("model_val", self.model_combo.currentData() or "")
        self.cfg.setValue("mode_val", self.mode_combo.currentData() or "auto")
        # GPU 设备号：ncnn Vulkan 与 PP-OCRv6(CUDA) 分开存，避免互相覆盖
        if self.gpu_combo.isVisible():
            _eid_now = self.engine_combo.currentData() or ""
            _gpu_key = "gpu_device_cuda" if _eid_now.startswith("umi_plugin_v6") else "gpu_device"
            self.cfg.setValue(_gpu_key, self.gpu_combo.currentData())
        else:
            self.cfg.setValue("gpu_device", -2)
        out_dir = self.output_edit.text().strip()
        if out_dir:
            self.cfg.setValue("last_output_dir", out_dir)
        if self._last_input_dir:
            self.cfg.setValue("last_input_dir", self._last_input_dir)

    def _restore_engine_settings(self):
        # ── 一次性迁移：把遗留的 mode_val='cpu' 复位回 auto（只做一次） ──
        # 旧版本只在「开始处理」时落盘设置，用户改过「自动(推荐)」后直接关窗就丢了，
        # 注册表里长期残留 mode_val='cpu' → 每次打开都回到 CPU 模式，
        # 看起来就像「模式总是自己变成 CPU」。这里复位一次并打标记，
        # 之后完全按用户自己的选择记忆（v1.3.1 起 closeEvent 也会落盘）。
        if self.cfg.value("_mode_migrated_v131", "") != "yes":
            self.cfg.setValue("mode_val", "auto")
            self.cfg.setValue("_mode_migrated_v131", "yes")
            try:
                self._update_mode_combo()          # 立刻反映到界面上
            except Exception:
                pass
            try:
                self.log("[配置] 检测到旧版遗留的「CPU模式」设置，已复位为「自动(推荐)」"
                         "（本次为一次性迁移，之后按你的选择记忆）")
            except Exception:
                print("[配置] mode_val 已一次性复位为 auto")
        last_engine = self.cfg.value("engine_id", "")
        # 旧配置兼容：ncnn_cpu 已并入 ncnn Vulkan（同一二进制），映射后按其还原
        if last_engine == "ncnn_cpu":
            last_engine = "ncnn_vulkan"
        if last_engine and last_engine in _PLUGIN_DIRS:
            idx = self.engine_combo.findData(last_engine)
            if idx >= 0:
                # 启动还原配置：抑制「引擎不支持原语言」提示（语言会自动回退到该引擎首项）
                self._engine_guard_suppress = True
                try:
                    self.engine_combo.setCurrentIndex(idx)
                    self._on_engine_changed()
                finally:
                    self._engine_guard_suppress = False

    def start_processing(self):
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "警告", "请添加要处理的PDF文件")
            return
        # 【混排守卫】简单模式下若勾了多个语言且含「专用分语种文字」——
        # 三个引擎都只有单文字系模型（ncnn 字典 0 覆盖），做不到混合识别，必须弹窗说清。
        _mix = self._mixed_langs_problem()
        if _mix:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("多语言混排无法实现")
            box.setText(_mix)
            _go = box.addButton("仍要继续", QMessageBox.AcceptRole)
            _back = box.addButton("返回修改", QMessageBox.RejectRole)
            box.setDefaultButton(_back)
            box.exec_()
            if box.clickedButton() is not _go:
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
        overwrite_ocr = self.overwrite_ocr_check.isChecked()
        limit_side_len = self.side_len_spin.value()
        model_size = self.model_combo.currentData() or "medium"
        lang_map = dict(self._LANG_ITEMS)
        # EasyOCR 条目（简易模式不涉及，仅在专业模式选 EasyOCR 引擎时使用）
        lang_map["English (EasyOCR)"] = "en"
        lang_map["Fran\u00e7ais (EasyOCR)"] = "fr"
        lang_map["Italiano (EasyOCR)"] = "it"
        lang_map["Espa\u00f1ol (EasyOCR)"] = "es"
        lang_display = self._current_lang_name() or "中文 (Chinese)"
        ocr_lang = lang_map.get(lang_display, "chinese")
        # ── 引擎-语言兼容性守卫：当前引擎不支持所选语言时，提示并可一键跳转到推荐引擎 ──
        _resolved_eid = self._resolve_engine_lang_mismatch(engine_id, lang_display, ocr_lang)
        if _resolved_eid is None:
            return  # 用户选择取消
        if _resolved_eid != engine_id:
            engine_id = _resolved_eid
            use_gpu = self.get_use_gpu()
            lang_display = self._current_lang_name() or lang_display
            ocr_lang = lang_map.get(lang_display, ocr_lang)
        use_angle_cls = self.angle_cls_check.isChecked()
        scale = self.scale_combo.currentIndex() + 1
        extra_params = {}
        if engine_id.startswith('umi_plugin_v6'):
            extra_params["rec_batch_num"] = self.rec_batch_spin.value()
            extra_params["shrink_poly_ratio"] = 0.08 if self.shrink_check.isChecked() else 0.0
        elif engine_id in ('ncnn_cpu', 'ncnn_vulkan'):
            extra_params["enable_fp16"] = (self.precision_combo.currentData() == "fp16")
        if engine_id == 'ncnn_vulkan':
            extra_params["gpu_device"] = self.gpu_combo.currentData()
            extra_params["use_gpu"] = use_gpu
        if engine_id.startswith('umi_plugin_v6'):
            _gd = self.gpu_combo.currentData()
            extra_params["gpu_device"] = _gd if isinstance(_gd, int) and _gd >= 0 else 0
        extra_params["lang"] = ocr_lang
        if engine_id == "easyocr_universal":
            extra_params["easyocr_lang"] = lang_map.get(lang_display, "en")
        dual_instance = self.dual_check.isChecked()
        self.total_files = len(file_list)
        self.processed_files = 0
        self.overall_progress.setValue(0)
        einfo = ENGINE_REGISTRY.get(engine_id, {})
        mode_str = "GPU" if use_gpu else "CPU"
        self.log(f"开始处理 {self.total_files} 个文件")
        self.log("  引擎: " + einfo.get("name", engine_id) + " (" + mode_str + ")")
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
            overwrite_ocr=overwrite_ocr,
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
        self._job_total_pages = 0
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
        self._current_file_name = filename
        self.current_file_label.setText(f"当前文件: {filename}")
        pct = int((ocr_completed / total) * 100) if total > 0 else 0
        self.page_info_label.setText(f"OCR完成 {ocr_completed} / {total} 页 ({pct}%)")
        self._job_completed_pages = ocr_completed
        self._job_total_pages = int(total or 0)      # 当前 PDF 总页数（迷你卡片用）
        self._safety_timer.start(600000)
        if total > 0:
            self.page_progress.setValue(int((ocr_completed / total) * 10000))
            ov = int(((self.processed_files + ocr_completed / total) / self.total_files) * 10000)
            self.overall_progress.setValue(int(ov))
        self._mini_refresh()
    def _on_finished(self, filename, pdf_path, txt_path):
        self.processed_files += 1
        ov = int((self.processed_files / self.total_files) * 10000)
        self.overall_progress.setValue(ov)
        self.log(chr(10003) + f" 完成: {filename}")
        # 在待处理列表里给这一项打 ✔，一眼能看出哪些已导出
        self._mark_item_done(filename)
        self._mini_refresh()
    def _on_error(self, filename, err):
        self.processed_files += 1
        ov = int((self.processed_files / self.total_files) * 10000)
        self.overall_progress.setValue(ov)
        self.log(chr(10007) + f" 错误 [{filename}]: {err}")
        self._mini_refresh()
    def _on_cancelled(self, filename):
        self.log(f"已取消: {filename}")
    def _on_all_done(self, total, success, cancelled):
        if self._mini_active:
            self.exit_mini_mode()      # 处理完成 → 自动还原完整窗口
        self._safety_timer.stop()
        self._speed_timer.stop()
        self.log(f"\n处理完成! 成功: {success}/{total}")
        self._set_buttons_idle()
        self.status_label.setText(f"完成 - 成功 {success}/{total}")
        self.overall_progress.setValue(10000)
        self.speed_label.setText("处理速度: -- (已完成)")
        QMessageBox.information(self, "完成", f"批量处理完成!\n\n成功: {success}/{total} 个文件\n输出目录: {self.output_edit.text()}")
    def _set_buttons_processing(self):
        # 处理中：锁死一切会改变「本批任务配置」的入口。
        # 本批参数已在 start_processing 取走，中途改动会让界面显示与任务实际参数分叉。
        self._processing = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        # 文件列表：只能看，不能增删（列表保持可见，方便用户盯着进度）
        self.add_files_btn.setEnabled(False)
        self.add_folder_btn.setEnabled(False)
        self.add_folders_btn.setEnabled(False)
        self.clear_files_btn.setEnabled(False)
        self.remove_sel_btn.setEnabled(False)
        self.prune_done_btn.setEnabled(False)
        # 识别参数
        self.engine_combo.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.lang_combo.setEnabled(False)
        self.gpu_combo.setEnabled(False)
        self.side_len_spin.setEnabled(False)
        self.scale_combo.setEnabled(False)
        self.vertical_check.setEnabled(False)
        self.angle_cls_check.setEnabled(False)
        self.overwrite_ocr_check.setEnabled(False)
        self.rec_batch_spin.setEnabled(False)
        self.shrink_check.setEnabled(False)
        self.tensorrt_check.setEnabled(False)
        self.precision_combo.setEnabled(False)
        self.dual_check.setEnabled(False)
        # 输入模式 / 输出目录（输出目录是任务写入目标，中途改必然乱套）
        self.mode_files.setEnabled(False)
        self.mode_folder.setEnabled(False)
        self.output_edit.setEnabled(False)
        self.browse_output_btn.setEnabled(False)
        # 界面模式切换（简单 ↔ 专业）与其面板 —— 本次修复的核心
        self.ui_simple_btn.setEnabled(False)
        self.ui_expert_btn.setEnabled(False)
        self.simple_group.setEnabled(False)
        self.status_label.setText("处理中...（参数已锁定）")

    def _set_buttons_idle(self):
        self._safety_timer.stop()
        self._processing = False
        self._mini_refresh()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        self.cancel_btn.setEnabled(False)
        self.add_files_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        self.add_folders_btn.setEnabled(True)
        self.clear_files_btn.setEnabled(True)
        self.remove_sel_btn.setEnabled(True)
        self.prune_done_btn.setEnabled(True)
        self.engine_combo.setEnabled(True)
        self.mode_combo.setEnabled(True)
        # 模型下拉能否使用由引擎决定：win7_v5 / win7_classic 是语言驱动，无模型可选
        self.model_combo.setEnabled(
            self.engine_combo.currentData() not in ("win7_v5", "win7_classic"))
        self.lang_combo.setEnabled(True)
        self.gpu_combo.setEnabled(True)
        self.side_len_spin.setEnabled(True)
        self.scale_combo.setEnabled(True)
        self.vertical_check.setEnabled(True)
        self.angle_cls_check.setEnabled(True)
        self.overwrite_ocr_check.setEnabled(True)
        self.rec_batch_spin.setEnabled(True)
        self.shrink_check.setEnabled(True)
        self.tensorrt_check.setEnabled(True)
        self.precision_combo.setEnabled(True)
        self.mode_files.setEnabled(True)
        self.mode_folder.setEnabled(True)
        self.output_edit.setEnabled(True)
        self.browse_output_btn.setEnabled(True)
        self.ui_simple_btn.setEnabled(True)
        self.ui_expert_btn.setEnabled(True)
        self.simple_group.setEnabled(True)
        # 恢复引擎↔模式联动（例如 CPU 模式下双实例应保持禁用）
        self._on_mode_changed()
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

    # ════════════════════════════════════════════════════════════
    # 系统托盘（只作入口）+ 迷你窗口（真正的「收小」方案）
    # ------------------------------------------------------------
    # 设计意图（2026-10-09 修订）：
    #   * 最小化 = 普通最小化到任务栏。**不再收进托盘** —— Windows 11 会把新程序的
    #     托盘图标默认折进「隐藏的图标」里，用户最小化后容易连窗口带图标一起找不到；
    #   * 托盘图标只保留「显示主界面 / 显示运行日志窗口 / 退出程序」三个入口
    #     （日志窗口的调出是跨进程的，用 Win32 FindWindow 按标题找）；
    #   * 想「把两个窗口收小」请用迷你窗口模式：见 MiniWindow / enter_mini_mode。
    # ════════════════════════════════════════════════════════════
    APP_DISPLAY_NAME = "CathayOCR Pro"
    LOG_WINDOW_TITLE = "CathayOCR Pro — 运行日志"   # 必须与启动器 APP_TITLE 一致

    def _set_app_icon(self):
        """窗口图标 —— 与启动器的日志窗口用同一个 ico，看起来就是同一个程序。"""
        try:
            ico = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CathayOCR.ico")
            if os.path.isfile(ico):
                self.setWindowIcon(QIcon(ico))
        except Exception:
            pass

    def _setup_tray(self):
        """建立系统托盘图标。只作「显示 / 退出」的便捷入口，不接管最小化。

        托盘不可用时静默跳过，一切照旧。
        """
        try:
            if not QSystemTrayIcon.isSystemTrayAvailable():
                print("[Tray] 系统托盘不可用，跳过")
                return False
        except Exception:
            return False
        try:
            icon = self.windowIcon()
            if icon is None or icon.isNull():
                ico = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CathayOCR.ico")
                icon = QIcon(ico) if os.path.isfile(ico) else QIcon()
            if icon.isNull():
                icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
            self._tray = QSystemTrayIcon(icon, self)
            self._tray.setToolTip("%s —— 双击显示主界面" % self.APP_DISPLAY_NAME)

            menu = QMenu(self)
            act_show = QAction("显示主界面", self)
            act_show.triggered.connect(self._restore_from_tray)
            menu.addAction(act_show)
            act_log = QAction("显示运行日志窗口", self)
            act_log.triggered.connect(self._show_log_window)
            menu.addAction(act_log)
            menu.addSeparator()
            act_quit = QAction("退出程序", self)
            act_quit.triggered.connect(self._quit_from_tray)
            menu.addAction(act_quit)

            self._tray.setContextMenu(menu)
            self._tray.activated.connect(self._on_tray_activated)
            self._tray.show()
            print("[Tray] 托盘图标已就绪（仅作显示/退出入口；最小化 = 进任务栏）")
            return True
        except Exception as e:
            print("[Tray] 建立失败: %s" % e)
            self._tray = None
            return False

    def _on_tray_activated(self, reason):
        try:
            if reason in (QSystemTrayIcon.DoubleClick, QSystemTrayIcon.Trigger):
                self._restore_from_tray()
        except Exception:
            pass

    def _restore_from_tray(self):
        """显示主界面（日志窗口由启动器自动一起还原）。"""
        if self._mini_active:
            self.exit_mini_mode()          # 迷你模式中点托盘 → 直接还原完整窗口
            return
        try:
            self.showNormal()
            self.raise_()
            self.activateWindow()
        except Exception:
            pass

    def _show_log_window(self):
        """把启动器的日志窗口调出来 —— 跨进程，用 Win32 按窗口标题查找。

        （日志窗口是独立进程里的 tkinter 窗口，这里只做「显示 + 置顶」，不改它的状态。）
        """
        try:
            import ctypes
            u = ctypes.windll.user32
            hwnd = u.FindWindowW(None, self.LOG_WINDOW_TITLE)
            if not hwnd:
                hwnd = u.FindWindowW("TkTopLevel", self.LOG_WINDOW_TITLE)
            if not hwnd:
                self.log("（日志窗口不在运行 —— 直接双击 CathayOCR Pro.exe 即可打开）")
                return
            u.ShowWindow(hwnd, 5)          # SW_SHOW
            u.BringWindowToTop(hwnd)
            u.SetForegroundWindow(hwnd)
        except Exception as e:
            print("[Tray] 调出日志窗口失败: %s" % e)

    def _quit_from_tray(self):
        self._quitting = True
        self.close()                       # 走正常关闭流程（清理引擎 + 落盘设置）

    # ════════════════════════════════════════════════════════════
    # 迷你窗口模式（纯 UI：只读进度、只改窗口形态，不碰识别流程）
    # ------------------------------------------------------------
    #   * 进入：主窗口 hide() → 启动器的日志窗口检测到后自动一起收起；
    #     同时把右下角的小进度卡 show 出来（它自己是独立顶层窗口，不受影响）。
    #   * 退出：小卡片 hide()，主窗口按进入前的窗口状态还原 → 日志窗口自动回来。
    #   * 处理结束（_on_all_done）会自动调 exit_mini_mode()。
    # ════════════════════════════════════════════════════════════
    def _mini_snapshot(self):
        tot = int(getattr(self, "total_files", 0) or 0)
        done = int(getattr(self, "processed_files", 0) or 0)
        try:
            pct = self.overall_progress.value() / 10000.0
        except Exception:
            pct = 0.0
        # 页进度：仅在处理中上报，否则给 0（卡片显示「— / — 页」）
        if bool(self._processing):
            pg_done = int(getattr(self, "_job_completed_pages", 0) or 0)
            pg_total = int(getattr(self, "_job_total_pages", 0) or 0)
        else:
            pg_done = pg_total = 0
        return (self.APP_DISPLAY_NAME, self._current_file_name,
                done, tot, pct, bool(self._processing),
                pg_done, pg_total)

    def _mini_refresh(self):
        """把当前进度推给迷你卡片（没开迷你模式时什么都不做）。"""
        m = getattr(self, "_mini", None)
        if m is None or not m.isVisible():
            return
        try:
            m.set_state(*self._mini_snapshot())
        except Exception:
            pass

    def _ensure_mini_window(self):
        m = getattr(self, "_mini", None)
        if m is None:
            m = MiniWindow()
            m.restore_requested.connect(self.exit_mini_mode)
            m.on_top_changed.connect(self._save_mini_on_top)
            try:
                m.set_on_top(bool(self.cfg.value("mini_on_top", True, type=bool)),
                             notify=False)
            except Exception:
                pass
            self._mini = m
        return m

    def _save_mini_on_top(self, on):
        """记住「迷你窗口是否置顶」，下次进迷你模式沿用。"""
        try:
            self.cfg.setValue("mini_on_top", bool(on))
        except Exception:
            pass
        self.log("迷你窗口：%s" % ("已置顶" if on else "已取消置顶"))

    def enter_mini_mode(self):
        """把两个窗口收成一张右下角的小进度卡。"""
        if self._mini_active:
            return
        try:
            self._mini_prev_state = self.windowState()
        except Exception:
            self._mini_prev_state = Qt.WindowNoState
        m = self._ensure_mini_window()
        self._mini_active = True
        try:
            m.set_state(*self._mini_snapshot())
            m.show_at_bottom_right()
        except Exception as e:
            print("[Mini] 显示迷你窗口失败: %s" % e)
        try:
            self.hide()          # 日志窗口由启动器检测到主窗口不可见后自动一起收起
        except Exception:
            pass
        self.log("已切到迷你窗口模式 —— 处理结束后自动恢复；"
                 "双击右下角小卡片或点它的「恢复」可随时回来。")

    def exit_mini_mode(self):
        """从迷你卡片回到完整窗口（日志窗口由启动器自动一起还原）。"""
        if not self._mini_active:
            return
        self._mini_active = False
        try:
            if self._mini is not None:
                self._mini.hide()
        except Exception:
            pass
        try:
            self.show()
            if int(getattr(self, "_mini_prev_state", 0)) & int(Qt.WindowMaximized):
                self.showMaximized()
            else:
                self.showNormal()
            self.raise_()
            self.activateWindow()
        except Exception:
            pass
        self.log("已恢复完整窗口")

    def closeEvent(self, event):
        """窗口关闭时清理所有子进程"""
        print("[MainWindow] Cleaning up OCR instances...")
        # 迷你卡片是独立顶层窗口，关程序时一起收掉，免得留个孤儿窗
        try:
            if self._mini is not None:
                self._mini.hide()
                self._mini.deleteLater()
                self._mini = None
        except Exception:
            pass
        # 先把托盘图标摘掉，否则关窗后托盘里会留一个点不动的"幽灵图标"
        try:
            if self._tray is not None:
                self._tray.hide()
                self._tray.setVisible(False)
                self._tray = None
        except Exception:
            pass
        # 【修复】关窗时先落盘设置。
        # 旧实现只在「开始处理」时 _save_settings()，导致「改完设置直接关窗」的改动全部丢失
        # —— 用户把模式改成「自动(推荐)」后关窗，下次打开仍是上次处理时存的 CPU 模式，
        # 看起来就像「模式总是自己变成 CPU」。
        try:
            self._save_settings()
        except Exception as e:
            print(f"[MainWindow] save settings on close: {e}")
        try:
            # 【修复】只在引擎实例确实存在时才关闭。
            # 旧代码无条件 `OCRClient()`：实例不存在时会用默认参数真的启动
            # 一次引擎（默认双实例）再关掉 —— 关窗白等数秒，还会抢显存。
            if OCRClient._instance is not None:
                OCRClient._instance.close()
            OCRClient._instance = None
        except Exception as e:
            print(f"[MainWindow] OCR cleanup: {e}")
        # 清理属于本软件目录的残余子进程
        self._kill_orphans()
        event.accept()

    def _kill_orphans(self):
        """只结束『属于本软件目录』的残余引擎进程。
        【修复】旧实现是 os.system('taskkill /f /im ...') 按进程名全系统强杀，
        会误伤其它位置的同名引擎（例如另一份正在运行的 CathayOCR）。"""
        killed = _kill_engine_processes(CLEANUP_TARGETS)
        if killed:
            print(f"[MainWindow] Cleaned up local engine pids: {killed}")


CLEANUP_TARGETS = ["PaddleOCR-json.exe", "ppocr_ocr_vulkan.exe", "ppocr_ocr_cpu.exe"]


def _our_engine_dirs():
    """本软件的『版本根目录』——要清理的引擎都在它下面：
    ncnn/PPOCR-ncnn-Vulkan/、ncnn/PPOCR-ncnn-CPU/、ppocr_v3/、ppocr_v5/、ppocr_v6/。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.dirname(here)]


def _kill_engine_processes(names, dirs=None):
    """仅结束『exe 路径位于本软件目录内』且名字匹配的进程。
    返回被结束的 pid 列表。非 Windows 平台直接返回空列表。

    设计要点：绝不按进程名全局强杀 —— 只认完整路径，
    因此不会影响其它目录（含 C 盘那份）正在运行的同类程序。"""
    if platform.system() != "Windows":
        return []
    if dirs is None:
        dirs = _our_engine_dirs()
    names_l = {str(n).lower() for n in names}
    dirs_l = {os.path.normcase(os.path.abspath(d)) for d in dirs if d}
    if not dirs_l:
        return []
    killed = []
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return []
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except Exception:
        return []

    TH32CS_SNAPPROCESS = 0x00000002
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_TERMINATE = 0x0001

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    my_pid = os.getpid()
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == INVALID_HANDLE_VALUE:
        return []
    try:
        pe = PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(pe))
        while ok:
            pid = int(pe.th32ProcessID)
            exe_name = pe.szExeFile or ""
            if pid != my_pid and exe_name.lower() in names_l:
                h = kernel32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE, False, pid)
                if h:
                    try:
                        buf = ctypes.create_unicode_buffer(32768)
                        size = wintypes.DWORD(len(buf))
                        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                            full = os.path.normcase(os.path.abspath(buf.value))
                            parent = os.path.dirname(full)
                            in_our_dir = any(
                                parent == d or parent.startswith(d + os.sep) for d in dirs_l)
                            if in_our_dir:
                                if kernel32.TerminateProcess(h, 1):
                                    killed.append(pid)
                    finally:
                        kernel32.CloseHandle(h)
            ok = kernel32.Process32NextW(snap, ctypes.byref(pe))
    finally:
        kernel32.CloseHandle(snap)
    return killed


def _force_cleanup():
    """强制清理（只针对本软件目录内的引擎进程）"""
    _kill_engine_processes(CLEANUP_TARGETS)


if __name__ == '__main__':
    # 启动前清理『本软件目录内』的残余引擎进程
    # 【修复】不再按进程名全局强杀，避免误伤其它目录正在运行的同类程序
    _kill_engine_processes(CLEANUP_TARGETS)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window._restore_engine_settings()
    window._setup_tray()          # 系统托盘：最小化时与日志窗口一起收起
    window.show()
    sys.exit(app.exec_())
