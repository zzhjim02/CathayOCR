"""
CathayOCR Lite (轻量版) v1.3.0 - 纯 Vulkan 引擎PDF处理器
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
    QTextBrowser, QFrame, QShortcut,
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
    """引擎内部 id → 界面统一显示名（取注册表里的 name）。

    凡是给用户看的引擎名字一律走这里，保证下拉框 / 日志 / 提示口径一致，
    不再直接暴露 ncnn_vulkan 这类内部 id。
    """
    info = ENGINE_REGISTRY.get(engine_id)
    if info and info.get("name"):
        return info["name"]
    return fallback if fallback is not None else engine_id


register_engine(
    'ncnn_vulkan', 'PP-OCR (ncnn Vulkan)',
    '\u2b50 速度最快！\n支持NVIDIA/AMD/Intel任意显卡\nGPU模式+Vulkan加速\nCPU模式+ncnn原生\n支持多版本模型（v3~v6）\n支持中/英/法/德/日/意/西/葡/希腊/多语言\n（韩/俄/阿拉伯/天城文等请用专业版）',
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
# ncnn 字典 ↔ 模型 路由（唯一判定处；UI 与两处适配器共用，避免再次分叉）
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


# ── 显卡档位（轻量版 simple_gpu 下拉索引）──
#   0 大显存独显(≥12GB, 任意品牌) | 1 小显存独显(≤8GB) | 2 仅有核显 | 3 纯CPU
#   轻量版只有 ncnn Vulkan / CPU，显卡品牌不改变引擎，只影响「精度优先」时的边长
_GPU_BIG = (0, 3)    # 大显存独显 + 纯CPU（无显存限制）→ 精度优先边长 2560
_GPU_SMALL = (1, 2)  # 小显存独显 / 核显 → 精度优先边长 2240（防爆显存）

# ── OCR 结果码（2026-10-10 整理，两版一致）──────────────────────────────
#   100 引擎给出文字块（有结论）
#   101 引擎明确回复「本页无文字」（有结论，空白页——不算失败）
#   102 引擎层错误：真超时（卡满 timeout_sec）/ TCP 断开 / JSON 解析失败
#   103 引擎不可用：进程没了或端口不通，_ensure_running() 判定失败（约 1 秒返回）
#   900 调用侧异常（base64/PNG 解码、包装函数抛错）——与引擎无关
#
# 为什么要把 103 从 102 里拆出来（2026-10-10 修复·之一）：
#   旧版把「引擎不可用」硬写成 code=102 + data 含 "timeout"，只为蹭看门狗那条
#   「102 且 data 含 timeout」的判定。副作用是日志里出现自相矛盾的
#   「超时 1s (limit=180s)」——那 1 秒其实是 _server_running() 的探测超时，
#   跟 OCR 的 180 秒限时毫无关系，用户看到只会以为真超时了。
_RESULT_ENGINE_UNAVAILABLE = 103
# 引擎层故障（该重启引擎才对）：102 超时/引擎错 + 103 不可用。
# 600/900 之类调用侧异常**不在此列** —— 跟引擎无关，重启只会白等几秒。
_RESULT_ENGINE_ERROR_CODES = (102, _RESULT_ENGINE_UNAVAILABLE)

# 【2026-10-10 修复·之十】连接层瞬时抖动的「透明重发」次数。
# 线上现象（两份 715 页日志）：请求已发出、引擎一个字节都没回（已收=0B），
# 连接就在 0.6~0.7 秒被中止（WinError 10053 本机侧 / 10054 引擎侧），
# 而引擎进程全程健康 —— engine start 头未增加、exited=0、stderr 无任何报错。
# 语义上「响应 0 字节」= 引擎没有产出任何结果 = 这一页从未被处理过，
# 因此原样重发这一页是**幂等安全**的，不会产生重复识别等副作用。
# 在此处透明重发的好处：
#   · 不消耗页级重试预算（MAX_PAGE_RETRY 仍是 1，不擅改用户定稿的策略）
#   · 不重启任何引擎（不会牵连另一个本来健康的实例）
#   · 用户不必再看到「引擎层故障」——那其实只是连接抖了一下
_TCP_TX_RETRY = 2


def _simple_side_scale(doc_idx, speed_idx, gpu_idx):
    """简单模式「边长 / 渲染倍率」统一规则（与专业版一致）：
      · 速度优先(0) / 标准平衡(1)：边长一律 2240，渲染一律 2x
      · 精度优先(2)：大显存（或纯CPU）2560，小显存 2240
      · 渲染 3x 仅限「古籍 + 精度优先 + 大显存（或纯CPU）」，其余一律 2x
    返回 (target_side, target_scale_index)，scale index: 0=1x / 1=2x / 2=3x
    """
    if speed_idx == 2:  # 精度优先
        side = 2560 if gpu_idx in _GPU_BIG else 2240
    else:               # 速度优先 / 标准平衡
        side = 2240
    scale = 2 if (doc_idx == 1 and speed_idx == 2 and gpu_idx in _GPU_BIG) else 1
    return side, scale


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

    # 【2026-10-09 卡死修复】同一实例两次重启的最小间隔（秒）
    _RESTART_MIN_INTERVAL = 8.0

    def __init__(self, engine_id, plugin_dir, entry_path):
        super().__init__(engine_id, plugin_dir, entry_path)
        self.config_path = os.path.join(plugin_dir, "config.json")
        self.port = 18043
        self.port_offset = 0
        self.server_proc = None
        self.lock = threading.Lock()
        # 【2026-10-09 卡死修复】进程生命周期锁 —— 只保护 server_proc 的「读-改-写」。
        # 必须与 self.lock 严格分离：self.lock 会被卡在 sock.recv 的线程连续持有
        # 最多 180 秒，若看门狗重启去抢 self.lock，就会干等 180 秒才动手 ——
        # 那才是真正废掉看门狗。两把锁绝不嵌套获取。
        self._proc_lock = threading.Lock()
        self._last_restart = 0.0
        self.current_config = {}
        self._started = False
        self._use_gpu = True  # 适配器初值；实际由 params['use_gpu'] 覆盖（界面默认「自动(推荐)」）
        self._ocr_times = collections.deque(maxlen=20)  # 最近OCR耗时(秒)，用于自适应超时
        # 【2026-10-10 修复·之六】临时 PNG 文件改为「每个线程一份」，绝不再跨线程共用。
        # 旧实现所有线程共用同一个复用文件：两个 consumer 并发落到同一个 adapter 时，
        # A 的请求正被引擎读取、B 把同一个文件截断重写 → 引擎读到损坏的 PNG →
        # 原生崩溃（本机 WER 已禁用，崩溃在系统里不留任何痕迹）→ 连接被 RST →
        # 日志里成对出现的 TCP error 10053 / 10054。
        self._tls = threading.local()
        self._tmp_paths = []
        self._tmp_paths_lock = threading.Lock()

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

    def _open_stderr_log(self):
        """把引擎的 stderr 落到「引擎目录/engine_<端口>_stderr.log」（2026-10-10 修复·之五）。

        原版用的是 stderr=subprocess.PIPE，但**全程没有任何地方读这个管道**，两个后果：
          ① 引擎往 stderr 写得多了会把管道缓冲区（约 4KB）写满 → 引擎自己阻塞在 write
             上，看起来就是「引擎活着但一直不回复」，极难排查；
          ② 引擎临终前打印的报错被永久留在管道里 —— 它为什么死，程序自己丢掉了。
             2026-10-10 那份 715 页日志的死因因此查不出来。
        改成落盘后既不会阻塞，事后也能直接翻文件。父进程关闭自己的句柄不影响
        子进程（子进程持有的是继承过去的那份），所以可以先关再读。
        """
        self._close_stderr_log()
        try:
            path = os.path.join(self.plugin_dir,
                                "engine_%d_stderr.log" % self._server_port)
            # 【2026-10-10 修复·之七】改成**追加**：原来每次重启都用 "wb" 截断，
            # 等于把上一次崩溃前引擎留下的最后几行直接抹掉。本机 WER 已禁用，
            # 这个文件是唯一的死因线索。超过 512KB 轮转一次，避免无限增长。
            try:
                if os.path.exists(path) and os.path.getsize(path) > 512 * 1024:
                    os.replace(path, path + ".1")
            except Exception:
                pass
            self._stderr_path = path
            self._stderr_fh = open(path, "ab", buffering=0)
        except Exception:
            self._stderr_fh = None
            self._stderr_path = None

    def _close_stderr_log(self):
        """关掉 stderr 日志句柄（幂等；不影响正在写该文件的子进程）。"""
        fh = getattr(self, "_stderr_fh", None)
        self._stderr_fh = None
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass

    def _note_stderr(self, text):
        """往引擎日志里插一行「本程序视角」的记录（2026-10-10 修复·之七）。

        引擎若是原生崩溃，它自己不会留下任何遗言（本机 WER 已禁用，
        系统事件日志 / WER 归档里一条都查不到）。所以退出码、重启原因这些
        关键事实必须由我们写进同一个文件，否则事后无从判断「是崩了还是被杀」。
        """
        fh = getattr(self, "_stderr_fh", None)
        if fh is None:
            return
        try:
            fh.write(("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
                     ).encode("utf-8", errors="replace"))
        except Exception:
            pass

    def _read_stderr_tail(self, limit=600):
        """读引擎 stderr 日志的尾部（启动失败时把原因带回主日志）。"""
        path = getattr(self, "_stderr_path", None)
        if not path:
            return ""
        try:
            with open(path, "rb") as f:
                data = f.read()
            return data[-limit:].decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

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
        self._open_stderr_log()
        self.server_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            # 【2026-10-10 修复·之七】stdout 也一起落盘：引擎的部分报错走的是 stdout，
            # 原来 DEVNULL 直接丢掉了。两个流写同一个**文件**（不是管道），不会阻塞。
            stdout=(self._stderr_fh if self._stderr_fh is not None
                    else subprocess.DEVNULL),
            stderr=(self._stderr_fh if self._stderr_fh is not None
                    else subprocess.DEVNULL),
            cwd=self.plugin_dir,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )
        self._note_stderr("=== engine start pid=%s port=%s cmd=%s ==="
                          % (self.server_proc.pid, port, " ".join(cmd)))
        deadline = time.time() + 30
        while time.time() < deadline:
            if self._server_running():
                self._started = True
                print("[Vulkan] Server is ready on port " + str(port))
                return ""
            if self.server_proc.poll() is not None:
                _rc = self.server_proc.returncode
                self._note_stderr("engine exited during startup: returncode=%s" % _rc)
                self._close_stderr_log()
                stderr_text = self._read_stderr_tail()
                self.server_proc = None
                return ("[Error] Vulkan server exited during startup (returncode=%s)."
                        " stderr: %s" % (_rc, stderr_text))
            time.sleep(0.1)
        self._close_stderr_log()
        stderr_text = self._read_stderr_tail()
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
        # 字典必须按"实际选中的模型"(selected)选择，不能用传入的原始字符串：
        # 若传入短名(如 "medium")，selected 会回退到 available[0]（真实模型），
        # 此时仍按短名选字典会得到"v5字典 + v6模型"的不匹配 → 引擎 1~2 次请求后卡死。
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
        # 同 GPU 分支：字典必须跟随实际选中的模型(selected)，避免字典与模型不匹配
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

    def _server_running(self):
        try:
            with socket.create_connection(("127.0.0.1", self._server_port), timeout=1):
                return True
        except Exception:
            return False

    def _tcp_request(self, request, timeout=180):
        with self.lock:
            if not self._ensure_running():
                # 【2026-10-10 修复·之一】原版这里返回的是
                #     {"code": 102, "data": "OCR timeout (engine unavailable)"}
                # —— 塞进 "timeout" 关键字纯粹是为了蹭看门狗那条
                # 「code=102 且 data 含 timeout」的判定，好让它触发 Tier1。
                # 代价是「引擎不可用」和「真·180 秒超时」从此不可区分：
                # 日志上打出「超时 1s (limit=180s)」，用户根本没法判断到底发生了什么。
                # 现在改用独立结果码 103，语义明确，也不再需要蹭关键字。
                return {"code": _RESULT_ENGINE_UNAVAILABLE,
                        "data": "OCR engine unavailable (port %d not answering)"
                                % self._server_port}
            # 【2026-10-10 修复·之九】把 TCP 错误细分到「发生阶段」并记录耗时与异常类型。
            # 原来 connect / send / recv 三段共用一个 except，日志只剩
            #     "TCP error: [WinError 10053] ..."
            # —— 看不出断在哪一步、断了多久、什么异常，用户为此连着追问
            #    「引擎层故障到底是怎么回事」。补齐这三项后，下次再出现即可直接定性：
            #     阶段=connect  → 引擎监听 / accept 出问题（引擎侧）
            #     阶段=send     → 连接刚建成即被中止（本机栈 / 安全软件）
            #     阶段=recv，耗时很短   → 引擎处理中主动关掉了连接
            #     阶段=recv，耗时≈timeout → 超时边界
            # 注意辨别：WinError 10053「你的主机中的软件中止了一个已建立的连接」
            #   是**本机**一侧被中止（WSAECONNABORTED）；对端强行关闭是 10054
            #   「远程主机强迫关闭了一个现有的连接」（WSAECONNRESET）。两者别混。
            _err = "no response"
            # 【2026-10-10 修复·之十】透明重发循环 —— 判据与理由见 _TCP_TX_RETRY 的注释。
            for _tx in range(_TCP_TX_RETRY + 1):
                _t_stage = "connect"
                _t_beg = time.time()
                _t_recv = 0
                try:
                    with socket.create_connection(("127.0.0.1", self._server_port), timeout=timeout) as sock:
                        sock.settimeout(timeout)
                        _t_stage = "send"
                        json_str = json.dumps(request)
                        sock.sendall(json_str.encode("utf-8"))
                        _t_stage = "recv"
                        chunks = []
                        while True:
                            try:
                                chunk = sock.recv(4096)
                            except socket.timeout:
                                return {"code": 102, "data": f"OCR timeout ({timeout}s)"}
                            if not chunk:
                                break
                            chunks.append(chunk)
                            _t_recv += len(chunk)
                        _t_stage = "parse"
                        text = b"".join(chunks).decode("utf-8", errors="ignore")
                        return self._parse_json(text)
                except Exception as e:
                    _err = ("TCP error: %s [阶段=%s 耗时%.2fs 已收=%dB 类型=%s]"
                            % (e, _t_stage, time.time() - _t_beg,
                               _t_recv, type(e).__name__))
                    # 只对「连接层瞬时抖动」透明重发：异常在收发阶段 + 一个字节都没收到。
                    # 其余情况（解析失败 / 真超时 / 引擎报错）一律原样返回，绝不掩盖真问题。
                    if _tx >= _TCP_TX_RETRY or _t_stage != "recv" or _t_recv != 0:
                        return {"code": 102, "data": _err}
                    print("[NcnnAdapter] 端口 %d 连接在 %s 阶段被中止（响应 0 字节，"
                          "引擎未处理本页）→ 透明重发 %d/%d"
                          % (self._server_port, _t_stage, _tx + 1, _TCP_TX_RETRY))
                    time.sleep(0.03)
            return {"code": 102, "data": _err}

    def _engine_is_ours(self):
        """端口上有响应时，判断响应者是否确为「本对象启动、且仍存活」的那个引擎。

        只连得通是不够的：上一次会话被强杀后留下的孤儿引擎（或别的目录里的同类引擎）
        会把端口占住 —— 内核替它把连接 SYN 收下（connect 成功），请求却永远等不到
        回复（recv 一直挂到 180 秒超时，GPU 占用率掉到 0）。那种情况下
        self.server_proc 要么为空、要么早已退出，据此即可判定「响应者不是我们的」。

        【2026-10-10 修复·之二】判定依据从「self._started 且进程活着」改成
        「只看进程句柄」——因为 self._started 会被 _ensure_running() 第 3 步
        在端口探测失败时**无条件**打掉，而端口探测的超时只有 1 秒、引擎忙时必然
        失败。于是原先的保护逻辑自我拆台：
            一次探测失败 → _started=False → _engine_is_ours() 恒为 False
            → 该实例此后每次请求都在 1 秒内返回「引擎不可用」→ 且永不重启
            （_ensure_running 按设计不重启）→ 看门狗误当超时 → 整份文件报废。
        进程句柄才是「归属」的唯一可信依据：server_proc 只由本对象的 Popen 赋值。
        """
        p = self.server_proc
        try:
            return bool(p is not None and p.poll() is None)
        except Exception:
            return False

    def _ensure_running(self):
        """只做「引擎是否可用」的**检查**，绝不在本方法里自行重启（2026-10-09 修复·之三）。

        为什么改成「只检查、不重启」：
        本软件是双实例「轮询分页」——不同页发给不同引擎，共用一个递增计数器
        (_adapter_index)。如果这里只重启「自己这一个」，两个引擎就会进入
        「一新一旧 / 一好一坏」的不一致状态；用户明确要求：宁可两个一起重启，
        也不要留半个坏引擎在旁边。

        于是统一语义：
            引擎不可用 → 本页立即失败返回（code=103，见 _RESULT_ENGINE_UNAVAILABLE）
                      → 看门狗判定「引擎确实坏了」→ Tier1 触发 restart_all()
                      → **两个实例一起重启** → 同一页重发 → 恢复。
        注意（2026-10-10 修复·之六）：只有「103 不可用 / 引擎进程已退出 / 真超时」
        才走 Tier1。单纯一条连接被 RST（TCP error 10053 / 10054）而进程还活着时，
        那只是「这一页没拿到结果」，原地重发即可 —— 若也去 restart_all()，
        会把另一个本来健康的引擎一起杀掉，连带另一个 consumer 的在途请求也断。

        附带好处：恢复了「本方法永不 Popen 新进程」这一性质。上一版之所以会产出
        孤儿引擎，正是因为在旧进程还没杀掉时就在这里 _start_gpu()；现在这里一个
        进程都不建，孤儿来源从根上消失。

        另一处必须保留的判定：_server_running() 的探测超时只有 1 秒，引擎忙时
        会误判「已死」。所以**先看进程是否还活着**（_engine_is_ours），活着就一律
        视为可用 —— 否则大文件跑到一半会被误判、把两个引擎都重启掉。
        """
        # 1) 我们自己的引擎进程还活着 → 可用。探测超时只是「忙」，不能据此判死。
        if self._engine_is_ours():
            return True
        # 2) 进程不在，但端口有人响应 → 响应者是「不是我们启动的」占位引擎
        #    （上次被强杀留下的孤儿 / 别的目录里的同类程序）。清掉占位者，
        #    好让接下来「重启全部实例」能真正 bind 上端口。
        if self._server_running():
            with self._proc_lock:
                if self._engine_is_ours():          # 可能别的线程刚修好
                    return True
                print("[NcnnAdapter] 端口 %d 被非本进程启动的引擎占用 → 清理占位者"
                      % self._server_port)
                dropped = _kill_orphan_engines()
                if dropped:
                    print("[NcnnAdapter] 已清理孤儿引擎: %s" % dropped)
                old = self.server_proc
                self.server_proc = None
                self._started = False
                self._kill_proc(old)
            return False
        # 3) 进程已死 / 从未起来 → 清掉悬空引用（不在这里重启，交给看门狗）。
        # 【2026-10-10 修复·之二】到这里时进程**确实**已经不在（步骤 1 已经用
        #   进程句柄判定过：活着就直接 return True 了），所以置 _started = False
        #   是对的。但要注意 _started 从此不再是归属的判定依据
        #   —— 见 _engine_is_ours() 的注释（旧版无条件打掉它，把活着的引擎判死）。
        with self._proc_lock:
            old = self.server_proc
            if old is not None and old.poll() is not None:
                # 【2026-10-10 修复·之七】把退出码写进引擎日志：本机 WER 已禁用，
                # 引擎若是原生崩溃（0xC0000005 等），系统里查不到任何痕迹，
                # 只有这个 returncode 能证明「它是自己崩了，还是被我们杀掉的」。
                _rc = old.returncode
                print("[NcnnAdapter] 端口 %d 的引擎进程已退出，returncode=%s"
                      % (self._server_port, _rc))
                self._note_stderr("engine exited: returncode=%s"
                                  " (0xC0000005=访问违例崩溃, 15/1=被终止)" % _rc)
                self.server_proc = None
            self._started = False
        return False

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

    def _thread_tmp_path(self):
        """取「本线程独占」的临时 PNG 文件路径（懒创建 + 线程内复用）。

        【2026-10-10 修复·之六】每个线程一份文件，绝不再跨线程共用。
        旧实现所有线程共用同一个复用文件：两个 consumer 并发落到同一个 adapter 时，
        A 的请求正被引擎读取、B 把同一个文件截断重写 → 引擎读到损坏的 PNG →
        原生崩溃（本机 WER 已禁用，崩溃在系统里不留任何痕迹）→ 连接被 RST →
        日志里成对出现的 TCP error 10053 / 10054。
        「谁写的文件、引擎就读谁的文件」是这条请求路径能成立的前提。
        """
        tls = getattr(self, "_tls", None)
        if tls is None:
            tls = threading.local()
            self._tls = tls
        p = getattr(tls, "tmp_path", None)
        if p and os.path.exists(p):
            return p
        try:
            fd, p = tempfile.mkstemp(suffix=".png")
            os.close(fd)
        except Exception:
            return None
        tls.tmp_path = p
        try:
            with self._tmp_paths_lock:
                self._tmp_paths.append(p)
        except Exception:
            pass
        return p

    def _write_and_request(self, img_bytes, timeout=180):
        """把 PNG 字节写入「本线程独占」的临时文件后请求引擎（GPU走TCP / CPU走管道）。
        这是 run_base64 与 run_png_bytes 共用的核心路径 —— 引擎始终只收到文件路径，
        两种入口写出的文件字节完全一致，因此识别结果不可能有差异。"""
        tmp_path = self._thread_tmp_path()
        _tmp_owned = False
        if tmp_path is None:
            # 极少数情况（临时目录不可写）→ 退回「每次新建、用完即删」
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            _tmp_owned = True
        else:
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)
        try:
            t_ocr_start = time.time()
            if self._use_gpu:
                request = {"img_path": tmp_path.replace("\\", "/")}
                result = self._tcp_request(request, timeout)
            else:
                result = self._run_exe(tmp_path, timeout)
            elapsed = time.time() - t_ocr_start
            if result.get("code") == 100:
                self._ocr_times.append(elapsed)
            return result
        finally:
            if _tmp_owned:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def run_base64(self, image_base64, timeout=180):
        try:
            img_bytes = _b64.b64decode(image_base64)
            return self._write_and_request(img_bytes, timeout)
        except Exception as e:
            return {"code": 900, "data": f"Base64 error: {str(e)}"}

    def run_png_bytes(self, png_bytes, timeout=180):
        """直接接收 PNG 字节（Lite 内部队列优化：省去每页一次 base64 编解码）。
        写入临时文件后与 run_base64 走完全相同的请求路径，
        引擎读到的文件字节一致 → 识别结果不变。"""
        try:
            return self._write_and_request(png_bytes, timeout)
        except Exception as e:
            return {"code": 900, "data": f"PNG bytes error: {str(e)}"}

    def run_path(self, img_path, timeout=180):
        if self._use_gpu:
            request = {"img_path": img_path.replace("\\", "/")}
            return self._tcp_request(request, timeout)
        else:
            return self._run_exe(img_path, timeout)

    def _kill_proc(self, proc):
        """结束「传入的」进程对象（2026-10-09 卡死修复）。

        关键点：以参数为准，而不是读 self.server_proc。
        调用方一律先抢引用（old = self.server_proc; self.server_proc = None）
        再调用本方法 —— 这样即使多个线程并发重启，每个线程只杀自己抢到的那个，
        不会互相覆盖引用，更不会把别人刚建好的进程变成无人回收的孤儿。"""
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def kill_now(self):
        """立即结束引擎进程（供 force_close / 进程清理使用，2026-10-09 新增）。

        刻意 **不加** self._proc_lock：它的意义就是尽快打断正阻塞在 recv 上的调用，
        加锁反而可能被另一条重启路径挡住。「抢引用 + 杀对象」的写法本身已避免孤儿。"""
        old = self.server_proc
        self.server_proc = None
        self._started = False
        self._close_stderr_log()
        self._kill_proc(old)

    def stop(self):
        # 【2026-10-10 修复·之六】清理「所有线程」的复用临时文件
        # （旧版只有一个 self._tmp_path；改成线程私有后要按清单逐个删）
        for _p in list(getattr(self, "_tmp_paths", [])):
            try:
                os.unlink(_p)
            except Exception:
                pass
        try:
            self._tmp_paths = []
            self._tls = threading.local()
        except Exception:
            pass
        self._close_stderr_log()
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

    def get_dynamic_timeout(self, floor=180, ceiling=300):
        """根据所有活跃引擎实例的历史OCR耗时计算动态超时
        - 默认 180s（与原始版本一致，不额外多等）
        - 如果检测到本地处理速度慢（p90×3 > 180s），自动升到 min(p90×3, 300s)
        - 前3页无历史数据时用 180s
        - 查询时合并所有同类型适配器的 _ocr_times（多实例共享统计）
        """
        # 尝试从 OCRClient 聚合所有实例的计时
        all_times = list(self._ocr_times)
        # 查找 OCRClient 单例，聚合其他实例的计时
        from types import ModuleType
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
            return floor  # 设备正常，用默认 180s
        return int(min(upgrade, ceiling))  # 设备慢，自动升到 300s

    def restart(self, params=None):
        """强制重启引擎进程（杀旧进程→重建新进程）
        用于从Vulkan假死中恢复，不改变config。

        【2026-10-09 卡死修复】加 _proc_lock 串行化：原来每个 consumer 线程各跑一份
        看门狗、都可并发进 restart()，两个线程同时 stop()+start() 会让 server_proc
        引用互相覆盖，留下没人管的孤儿进程（实测同一端口被两个引擎同时 LISTEN）。
        另外加抑制窗口，避免两个 consumer 各触发一次 Tier1 把同一引擎重复重启两遍。
        看门狗的触发条件 / Tier 层级 / 重试次数一律未改，只改"重启时怎么记账"。
        """
        with self._proc_lock:
            if (self._server_running() and self._engine_is_ours()
                    and time.time() - self._last_restart < self._RESTART_MIN_INTERVAL):
                # 另一个线程刚重启过、且**我们自己的**服务确实已恢复 → 直接复用。
                # （必须带 _engine_is_ours：若端口被孤儿占着，则不能算「已恢复」，
                #   否则占位者会让恢复流程被抑制窗口挡掉 8 秒。）
                return ""
            print(f"[NcnnAdapter] Restarting engine on port {self._server_port}...")
            old = self.server_proc
            if old is not None and old.poll() is not None:
                # 【2026-10-10 修复·之七】引擎在我们重启它之前就已经死了 ——
                # 记下退出码（0xC0000005=访问违例崩溃 / 15、1=被终止 / 0=正常退出）。
                # 本机 WER 已禁用，这是唯一能区分「崩了」还是「被杀」的痕迹。
                print("[NcnnAdapter] 端口 %d 的引擎已先退出，returncode=%s"
                      % (self._server_port, old.returncode))
                self._note_stderr("engine already exited before restart:"
                                  " returncode=%s" % old.returncode)
            self.server_proc = None
            self._started = False
            self._close_stderr_log()
            self._kill_proc(old)
            time.sleep(0.5)  # 等操作系统释放端口
            # 【2026-10-09 修复·之三】端口仍被「不是我们启动的」引擎占着（孤儿/别目录引擎）
            # → 先清掉占位者。否则 _start_gpu() 里 _server_running() 会假性通过、
            #   _started 被置 True 看起来"重启成功"，实际请求全打进僵尸实例。
            if self._server_running() and not self._engine_is_ours():
                dropped = _kill_orphan_engines()
                if dropped:
                    print("[NcnnAdapter] 重启前清理孤儿引擎: %s" % dropped)
                time.sleep(0.2)
            if params is not None:
                self.current_config = params
            self._use_gpu = self.current_config.get("use_gpu", True)
            # 刻意不走 self.stop()/self.start()：stop() 会删掉复用的临时 PNG 文件，
            # 而此刻另一个 consumer 可能刚写完该文件正在等锁 —— 保留原语义即可。
            try:
                if self._use_gpu:
                    err = self._start_gpu(self.current_config)
                else:
                    err = self._start_cpu(self.current_config)
            except Exception as e:
                err = f"[Error] Engine restart failed: {str(e)}"
            self._last_restart = time.time()
            return err

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
                 model_size="PP_OCRv6_medium", use_angle_cls=False,
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
        # 【2026-10-10 修复·之六】派发计数器必须原子自增：两个 consumer 并发
        # 读-改-写 _adapter_index 会算出同一个 idx，把两页同时送进同一个引擎。
        self._dispatch_lock = threading.Lock()
        self._instances = []
        # 【2026-10-09】记住构造参数：Tier2 全文件重启时要用它把引擎**真正**重建起来
        # （见 rebuild()）。旧实现重建时直接 restart_all()，但那时 _instances 已被
        # force_close() 清空 —— 等于什么都没重建，恢复流程形同虚设。
        self._ctor = {
            "engine_id": engine_id, "use_gpu": use_gpu,
            "vertical_text": vertical_text, "limit_side_len": limit_side_len,
            "model_size": model_size, "use_angle_cls": use_angle_cls,
            "dual_instance": dual_instance, "extra_params": dict(extra_params),
        }
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
            # 【2026-10-10 修复·之六】原子派发，见 __init__ 里的 _dispatch_lock
            with self._dispatch_lock:
                idx = self._adapter_index % len(self._instances)
                self._adapter_index += 1
            return self._instances[idx].run_base64(image_base64, timeout_seconds)
        return self._instances[0].run_base64(image_base64, timeout_seconds)

    def ocr_image_png(self, png_bytes, timeout_seconds=180):
        """轮询调度多个实例（直接传 PNG 字节，与 ocr_image_base64 调度逻辑完全一致）"""
        if self._dual_instance and len(self._instances) > 0:
            # 【2026-10-10 修复·之六】原子派发，见 __init__ 里的 _dispatch_lock
            with self._dispatch_lock:
                idx = self._adapter_index % len(self._instances)
                self._adapter_index += 1
            return self._instances[idx].run_png_bytes(png_bytes, timeout_seconds)
        return self._instances[0].run_png_bytes(png_bytes, timeout_seconds)

    def engines_alive(self):
        """所有实例的引擎进程是否都还活着（2026-10-10 修复·之六）。

        看门狗靠它区分两种失败：
          · 引擎进程已退出（崩了 / 被外力杀了）→ 值得重建引擎；
          · 只是这一条 TCP 连接被 RST、进程还在 → 重发同一页即可。
        后者若也去 restart_all()，会把**另一个本来健康的引擎**一起杀掉，
        连带另一个 consumer 的在途请求也断 —— 日志里那对 10053/10054 就是这么来的。
        没有实例时返回 True（无可判定对象，不因此触发重建）。
        """
        insts = list(getattr(self, "_instances", None) or [])
        if not insts:
            return True
        for a in insts:
            try:
                chk = getattr(a, "_engine_is_ours", None)
                if callable(chk):
                    if not chk():
                        return False
                else:
                    p = getattr(a, "server_proc", None)
                    if p is not None and p.poll() is not None:
                        return False
            except Exception:
                return False
        return True

    def force_close(self):
        """强制关闭：直接 kill 子进程，立即中断阻塞的 OCR 调用"""
        # 【2026-10-09 修复】用 kill_now()（抢引用后杀），不再直接 kill 当前引用 ——
        # 避免与看门狗重启撞车时把引用覆盖、留下孤儿；先做快照防并发遍历出错。
        for a in list(getattr(self, "_instances", [])):
            try:
                a.kill_now()
            except Exception:
                pass
        self._instances = []
        self._initialized = False

    def restart_single(self, instance_idx):
        """【2026-10-09 语义统一】保留本方法只为兼容旧调用，**不再真的只重启一个**。

        双实例是轮询分页，只重启其中一个会让两个引擎处于「一新一旧」的不一致状态；
        用户明确要求「宁可两个一起重启」。所以这里直接转成 restart_all()。
        （说明：本方法在当前代码里没有任何调用方，看门狗走的本来就是 restart_all。）
        """
        print("[OCRClient] restart_single → 统一改为重启全部实例")
        self.restart_all()

    def restart_all(self):
        """重启全部引擎实例（看门狗 Tier1；也是本程序唯一的引擎恢复路径）"""
        # 【2026-10-09 修复】先做快照：force_close() 可能并发把 _instances 清空，
        # 直接遍历原列表会漏掉中途新建的实例。实例内部的并发由各自的 _proc_lock 兜住。
        instances = list(getattr(self, "_instances", []))
        for i, a in enumerate(instances):
            try:
                a.restart()
            except Exception as e:
                print(f"[OCRClient] restart_all[{i}] failed: {e}")

    def rebuild(self):
        """按原构造参数**真正重建**全部引擎实例（Tier2 全文件重启专用）。

        【2026-10-09 修复】原全文件重启流程是：force_close()（它会清空 _instances）
        → restart_all()。而 restart_all() 遍历的正是那个已被清空的列表 ——
        等于**一个引擎都没重建**；接着新线程一跑，ocr_image_png() 就会
        取 self._instances[0] 抛 IndexError，整份文件全是错误结果。
        也就是说「全文件重启」这条恢复路以前是**不可用**的，现在补上。
        """
        ctor = getattr(self, "_ctor", None)
        if not ctor:
            # 没有记录构造参数（极老的对象）→ 退化成重启现有实例
            return self.restart_all()
        # 先清掉可能占住端口的孤儿引擎，保证新引擎能 bind 上
        try:
            _dropped = _kill_orphan_engines()
            if _dropped:
                print("[OCRClient] rebuild 前清理孤儿引擎: %s" % _dropped)
        except Exception:
            pass
        for a in list(getattr(self, "_instances", [])):
            try:
                a.kill_now()
            except Exception:
                pass
        self._instances = []
        self._initialized = False
        try:
            params = build_engine_params(
                ctor["engine_id"], ctor["use_gpu"], ctor["vertical_text"],
                ctor["limit_side_len"], ctor["model_size"], ctor["use_angle_cls"],
                ctor["extra_params"])
        except Exception as e:
            print(f"[OCRClient] rebuild 参数构建失败: {e}")
            return "rebuild failed"
        num_instances = 2 if ctor["dual_instance"] else 1
        for i in range(num_instances):
            try:
                adapter = create_engine_adapter(ctor["engine_id"])
                if hasattr(adapter, "set_port_offset"):
                    adapter.set_port_offset(i * 10)   # 实例0: 18043, 实例1: 18053
                err = adapter.start(params)
                if err:
                    print(f"[OCRClient] rebuild 实例{i+1}启动失败: {err}")
                self._instances.append(adapter)
            except Exception as e:
                print(f"[OCRClient] rebuild 实例{i+1}异常: {e}")
        self.adapter = self._instances[0] if self._instances else None
        self._initialized = True
        print(f"[OCRClient] rebuild 完成：{len(self._instances)} 个实例")
        return ""

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
        # 【2026-10-10 修复·之十一】用户点「取消」的**粘性**标记。
        # _cancelled 会被下面的「整份文件重处理」逻辑复位成 False，
        # 所以它不能用来表达「用户明确要求停止」。
        # _user_cancelled 一旦置位，只有新建 PDFProcessor（= 新的一批任务）才回到 False。
        self._user_cancelled = False
        self._pipeline_alive = False    # 消费者线程是否还在跑（取消时决定要不要打断阻塞的 recv）
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
        # 【2026-10-10 修复·之十一】用户取消 = 最高优先级，且**必须与内部重跑区分开**
        self._user_cancelled = True
        self._cancelled = True
        self._paused = False
        # 旧实现在这里**同步**调用 ocr.force_close()：
        #   ① 它从 GUI 线程直接杀两个引擎进程 → 界面当场卡住（用户报的「卡住」）；
        #   ② 消费者线程随即拿到 code=900 + engines_alive()=False →
        #      走进「引擎坏了 → restart_all() → 重发本页」分支，
        #      重试预算耗尽后升级成「整份文件重处理」→ 取消被无视、整份重跑。
        # 现在改成：延迟 3 秒、且**只在流水线确实还没停下时**才去打断阻塞的 recv。
        # 正常取消（在途请求 1~2 秒就返回）根本不会碰到引擎。
        try:
            threading.Thread(target=self._delayed_force_close, daemon=True).start()
        except Exception:
            pass

    def _delayed_force_close(self, delay=3.0):
        # 取消 3 秒后若消费者线程还在跑，才强杀引擎打断阻塞的 recv（不再阻塞 GUI）
        try:
            time.sleep(delay)
        except Exception:
            return
        if self._cancelled and self._pipeline_alive:
            try:
                print("[OCRClient] 取消后流水线仍未停止 → 强制打断阻塞的 OCR 调用")
                self.ocr.force_close()
            except Exception:
                pass
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
        # 兼容旧接口：单页渲染（内部自行打开/关闭文档）
        with _suppress_mupdf_warnings():
            doc = fitz.open(pdf_path)
        try:
            return self.render_page_from_doc(doc, page_num, scale)
        finally:
            doc.close()

    def render_page_from_doc(self, doc, page_num, scale=2.0):
        """复用已打开的Document渲染单页（避免每页重复open大PDF）"""
        page = doc[page_num]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        return pix.tobytes("png")

    # 【2026-10-09 恢复策略（用户定稿，2026-10-09 深夜每层各减 1 次）】
    #   ① 某页出错 / 超时到 180 秒 → 重新处理该页，最多 MAX_PAGE_RETRY 次（=1）；
    #   ② 仍卡住                  → 重新处理整份文件，最多 MAX_FILE_RETRY 次（=1）；
    #   ③ 仍不行                  → 跳过该文件，并在源文件所在文件夹生成一份警告文件。
    # 计次含义：MAX_*_RETRY 是「**重试**次数」，不含首跑。
    #   → 页级最多尝试 1+1=2 次；文件级最多跑 1+1=2 遍。
    MAX_PAGE_RETRY = 1
    MAX_FILE_RETRY = 1

    # 【2026-10-09 完整性闸门】「有结论」的结果码：
    #   100 = 引擎给出了文字块
    #   101 = 引擎明确回复「本页无文字」（空白页 / 未检出文字）
    # 其它码（102 超时/引擎错误、103 引擎不可用、900 调用异常）都算「没有结论」——
    # 必须触发重处理，绝不能被当成空白页静默跳过。
    _RESULT_OK_CODES = (100, 101)

    def _incomplete_pages(self, total_pages):
        """返回「没有有效结果」的页码列表（0 基）。

        判定标准（用户要求：宁可整份文件重跑，也绝不交付错页/漏页/乱序结果）：
          1. 该页码在 results 里**根本不存在** —— 页面被彻底漏掉
             （渲染线程异常退出、线程在重启中提前返回等）；
          2. 结果码不在 _RESULT_OK_CODES 里 —— 引擎没给出结论
             （典型：Tier1 重启引擎时，另一个 consumer 正在等的请求被
               socket 断开打断，那一页以 code=102 落库）。
        写入阶段 write_results 对 `code != 100` 是直接 continue 的，
        也就是说这些页会被**静默丢弃**、不留任何痕迹 —— 这正是漏页的真正来源。
        """
        bad = []
        for pn in range(total_pages):
            r = self.results.get(pn)
            if r is None or r.get("code") not in self._RESULT_OK_CODES:
                bad.append(pn)
        return bad

    def _write_failure_notice(self, input_path, output_dir, total_pages, bad_pages):
        """在**源文件所在文件夹**写一份警告文件（写不进去则退到输出目录）。

        【2026-10-09 用户要求】单页重试 MAX_PAGE_RETRY 次 + 整份文件重处理 MAX_FILE_RETRY 次
        仍不成功时，跳过该文件，但必须让用户明确知道「这份文件没处理成功、也没有产出结果」——
        就在它自己所在的文件夹里留一份警告，文件名一眼可见。
        """
        name = Path(input_path).stem
        _fname = f"{name}_OCR未完成警告.txt"
        _show = "、".join(str(p + 1) for p in bad_pages[:30])
        if len(bad_pages) > 30:
            _show += " …"
        lines = [
            "=" * 64,
            "CathayOCR 处理未完成 —— 请重新处理本文件",
            "=" * 64,
            f"源文件   : {os.path.abspath(input_path)}",
            f"生成时间 : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"总页数   : {total_pages}",
            "",
            "【结果】本文件**没有**生成 OCR 结果文件（*_layered.pdf / *_result.txt）。",
            "       为避免交付一份「少了几页却看不出来」的结果，程序主动放弃了输出。",
            "",
            f"【问题页】共 {len(bad_pages)} 页没有拿到有效结果（页码从 1 计）：",
            f"   {_show}",
            "",
            "【已尝试的恢复步骤（全部用尽）】",
            f"   1) 重新处理单页   ：最多 {self.MAX_PAGE_RETRY} 次（超时前会重启全部 OCR 引擎）",
            f"   2) 重新处理整份文件：最多 {self.MAX_FILE_RETRY} 次",
            "   3) 仍不成功         → 跳过本文件并生成本警告",
            "",
            "【建议】",
            "   · 直接把本文件重新加入待处理列表再跑一次（多数情况是引擎临时假死，重跑即可）；",
            "   · 若每次都是同样的页码失败，请先检查该页扫描件是否损坏 / 空白 / 分辨率异常；",
            "   · 若多份文件接连出现同样问题，请关闭软件后重新打开（顺带清掉残留的引擎进程）。",
            "",
            "（本文件由程序自动生成，确认后可自行删除。）",
            "=" * 64,
            "",
        ]
        content = "\n".join(lines)
        dirs = []
        src_dir = os.path.dirname(os.path.abspath(input_path))
        if src_dir:
            dirs.append(src_dir)
        if output_dir:
            od = os.path.abspath(output_dir)
            if od not in dirs:
                dirs.append(od)
        for d in dirs:
            p = os.path.join(d, _fname)
            try:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(content)
                return p
            except Exception as e:
                print(f"[Watchdog] 写警告文件失败 {p}: {e}")
        return None

    def process_pdf(self, input_path, output_dir, total_pages, scale=2.0,
                    progress_callback=None, vertical_sort=False, overwrite_ocr=False):
        """处理单个PDF文件。output_dir=None时输出到源文件所在目录"""
        if output_dir is None:
            output_dir = os.path.dirname(input_path)
        if self._cancelled or self._user_cancelled:
            return None, None
        print(f"\n[PDFProcessor] Processing: {input_path} ({total_pages} pages, scale={scale}x)")
        self.reset()
        all_done = threading.Event()
        _render_done_count = [0]
        _render_done_lock = threading.Lock()
        def render_worker(start_page, end_page):
            # 每个渲染线程只open一次PDF文档，线程内复用渲染所有页
            # 避免每页重复解析整个PDF（大文件可省大量CPU）
            try:
                with _suppress_mupdf_warnings():
                    doc = fitz.open(input_path)
            except Exception as e:
                print(f"[Render] open failed: {e}")
                with _render_done_lock:
                    _render_done_count[0] += 1
                    if _render_done_count[0] >= self._num_workers:
                        all_done.set()
                return
            try:
                for pn in range(start_page, end_page):
                    if self._cancelled:
                        return
                    self.wait_if_paused()
                    if self._cancelled:
                        return
                    try:
                        png_bytes = self.render_page_from_doc(doc, pn, scale)
                        # 直接入队 PNG 字节（不再做 base64 编码：省一次编码 + 一次解码，
                        # 队列内存降约 1/4；引擎读到的文件字节完全一致）
                        while not self._cancelled:
                            try:
                                render_queue.put((pn, png_bytes), timeout=1)
                                break
                            except Full:
                                continue
                    except Exception as e:
                        print(f"[Render] Page {pn} error: {e}")
            finally:
                try:
                    doc.close()
                except Exception:
                    pass
            with _render_done_lock:
                _render_done_count[0] += 1
                if _render_done_count[0] >= self._num_workers:
                    all_done.set()
        def ocr_consumer(consumer_id):
            my_done = 0
            done_pages = [False] * total_pages
            next_to_store = 0
            t0 = time.time()
            # 【2026-10-09】重试计数已改为「每页复位」的局部变量 _page_retry（见下），
            # 不再需要 per-consumer 的 Tier1/Tier2 标志。
            while my_done < total_pages and not self._cancelled and not self._user_cancelled:
                try:
                    pn, png_bytes = render_queue.get(timeout=0.3)
                except Empty:
                    if all_done.is_set():
                        break
                    continue
                if self._cancelled:
                    break
                # 【2026-10-09 恢复策略（用户定稿）】单页重试计数，每页从 0 开始：
                #   ① 某页出错/超时 → 重新处理该页，最多 MAX_PAGE_RETRY 次；
                #   ② 仍卡住        → 触发整份文件重处理（外层循环）；
                #   ③ 文件也重试用尽 → 跳过该文件 + 写警告文件。
                _page_retry = 0
                # 本页「因引擎确实坏了而重建引擎」的次数。它**不消耗** _page_retry：
                # 引擎崩了/不可用是基础设施问题，不该把这一页的重试预算也吃掉
                # （否则崩一次就升级成整份文件重跑）。每页最多 1 次。
                _engine_recover = 0
                while True:
                    # 【2026-10-10 修复·之十一】用户取消优先级最高：立刻放弃本页，
                    # 绝不进入下面的「重建引擎 / 重发本页 / 升级整份重跑」分支。
                    if self._user_cancelled:
                        return
                    # 动态超时：取实例0的 get_dynamic_timeout（已聚合所有活跃引擎的耗时）
                    # 【2026-10-09】force_close 会并发清空 _instances，这里必须防空，
                    # 否则看门狗自己会抛 IndexError（那是"看门狗被自己弄崩"，不是被锁搞坏）
                    _insts = getattr(self.ocr, "_instances", None) or []
                    timeout_sec = _insts[0].get_dynamic_timeout() if _insts else 180
                    t_page = time.time()
                    try:
                        result = self.ocr.ocr_image_png(png_bytes, timeout_seconds=timeout_sec)
                    except Exception as e:
                        result = {"code": 900, "data": f"OCR error: {str(e)}"}
                    # 超时判定（引擎假死：code=102 且 data 含 "timeout"）—— 原判定式保留
                    _is_timeout = (result.get("code") == 102
                                   and "timeout" in str(result.get("data", "")).lower())
                    # 【2026-10-10 修复·之一】把「引擎不可用」单独识别出来。
                    # 它跟真超时是两回事：真超时是引擎卡满 180 秒；不可用是
                    # _ensure_running() 判定失败、约 1 秒就返回。旧版把两者混为一谈，
                    # 日志才会打出「超时 1s (limit=180s)」这种自相矛盾的记录。
                    _is_unavailable = (result.get("code")
                                       == _RESULT_ENGINE_UNAVAILABLE)
                    # 【2026-10-10 修复·之六】哪些失败「重建引擎才有救」：
                    #   · 103 引擎不可用 —— 进程已死 / 被占位者挡住；
                    #   · 任一实例的引擎进程已退出 —— 崩了（本机 WER 关着，系统里
                    #     查不到，只能靠 poll() 抓出来）；
                    #   · 真超时（下面单独判）。
                    # 只有这几类才值得重建。**光有一条连接被 RST（TCP error
                    # 10053/10054）而进程还活着时，绝不重建** —— 2026-10-10 那份
                    # 715 页日志证明：每次都 restart_all()，会把另一个健康引擎一起
                    # 杀掉，另一个 consumer 的在途请求也跟着断（成对的 10053+10054），
                    # 一次抖动付两次代价、日志全是重启刷屏。
                    _engine_fault = (result.get("code")
                                     in _RESULT_ENGINE_ERROR_CODES)
                    # 用 getattr 取：单测里注入的假 OCR 对象没有这个方法，
                    # 缺它就当作「无法判定 → 不因此重建」，绝不让看门狗自己抛异常。
                    _alive = getattr(self.ocr, "engines_alive", None)
                    _engines_dead = (not _alive()) if callable(_alive) else False
                    # 本页是否「没有拿到有效结论」：
                    #   100=引擎给出文字块 / 101=引擎明确回复本页无文字（空白页）→ 有结论
                    #   102/103 引擎层故障、900 调用异常 → 没有结论，必须重新处理该页
                    _page_failed = result.get("code") not in self._RESULT_OK_CODES
                    if not _page_failed:
                        break                    # 有结论 → 退出恢复循环，正常入库
                    # 【2026-10-10 修复·之十一】本页没有结论，但用户已经点了取消 →
                    # 立刻收手。否则这里会去 restart_all() 重建引擎并重发，
                    # 把取消彻底变成「程序自己重跑一遍」。
                    if self._user_cancelled:
                        return
                    if _is_unavailable:
                        print("[Watchdog] Consumer-%d Page %d 引擎不可用"
                              "（约 1 秒返回，不是 180 秒超时）: %s"
                              % (consumer_id, pn + 1,
                                 str(result.get("data", ""))[:120]))
                    elif _is_timeout:
                        elapsed = time.time() - t_page
                        print(f"[Watchdog] Consumer-{consumer_id} Page {pn+1} 超时 {elapsed:.0f}s (limit={timeout_sec}s)")
                    else:
                        # 【2026-10-10 修复·之八】把「连接被重置」和「引擎真的坏了」分开讲。
                        # code=102 在真超时之外只剩 TCP error（10053/10054）：那只说明
                        # 这一条连接断了，引擎进程往往完全健康、原地重发即可 ——
                        # 绝不该让用户以为「引擎坏了」。旧文案一律打「引擎层故障=True」，
                        # 用户为此连问三次「引擎层故障是怎么回事」。
                        _data_l = str(result.get("data", ""))
                        _kind = ("连接被重置，本页将原地重发（引擎进程正常）"
                                 if "tcp error" in _data_l.lower()
                                 else "引擎层错误")
                        print("[Watchdog] Consumer-%d Page %d 出错 code=%s（%s）: %s"
                              % (consumer_id, pn + 1, result.get("code"),
                                 _kind, _data_l[:120]))
                    if (_is_unavailable or _engines_dead) and _engine_recover < 1:
                        # 【2026-10-10 修复·之六】引擎确实坏了 → 重建两个引擎后
                        # **重发同一页**。这次重建不消耗页级重试预算（它是基础设施
                        # 恢复，不是"这一页又试了一次"），否则引擎崩一次就足以把
                        # 重试次数用光、直接升级成整份文件重跑。每页最多 1 次。
                        _engine_recover += 1
                        print("[Watchdog] Tier1: restarting ALL engine instances"
                              "（%s；重建后重发本页，不计入页重试）"
                              % ("引擎不可用" if _is_unavailable
                                 else "引擎进程已退出"))
                        self.ocr.restart_all()
                        continue
                    if _page_retry < self.MAX_PAGE_RETRY:
                        # ① 重新处理该页（最多 MAX_PAGE_RETRY 次）
                        _page_retry += 1
                        print("[Watchdog] → 重发第 %d 页（第 %d/%d 次；页号不连续是"
                              "并行渲染分段所致，不代表跳过了前面的页）"
                              % (pn + 1, _page_retry, self.MAX_PAGE_RETRY))
                        if _is_timeout:
                            # 【2026-10-10 修复·之六】真超时 = 引擎假死
                            # （进程活着但 180 秒不回复）→ 必须先重建两个引擎，
                            # 重试才有意义。而「连接被 RST」（10053/10054）不算，
                            # 那种情况引擎进程好好的，原地重发就够了。
                            print("[Watchdog] Tier1: restarting ALL engine instances"
                                  "（真超时 → 引擎假死）")
                            self.ocr.restart_all()
                        continue
                    # ② 单页重试用尽 → 触发全文件重处理（外层 for _full_retry 循环）
                    print("[Watchdog] Tier2: 第 %d 页重试 %d 次仍失败 → 全文件重处理"
                          % (pn + 1, self.MAX_PAGE_RETRY))
                    self._need_restart = True
                    self._cancelled = True
                    self.ocr.force_close()
                    return
                # 正常/超时恢复成功：存结果
                if (not self._need_restart and not self._cancelled
                        and not self._user_cancelled):
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
        # 全文件重处理外层循环：首跑 + MAX_FILE_RETRY 次重处理
        self._need_restart = False
        for _full_retry in range(self.MAX_FILE_RETRY + 1):
            num_consumers = 2 if self.dual_instance else 1
            render_queue = Queue(maxsize=20 if num_consumers > 1 else 12)
            # 渲染线程数：渲染能力(20~50页/秒)远大于OCR消化能力(1~4页/秒)，
            # 线程数永远不会成为瓶颈；过多线程反而与OCR进程抢CPU导致GPU空转。
            # 按文件规模动态调整：小文件少线程（省线程开销），大文件取平衡值。
            cpu_cores = os.cpu_count() or 8
            if total_pages <= 30:
                # 小文件：2线程渲染足够（全部渲染完也只需1~3秒，而OCR要几十秒）
                n_workers = min(2, total_pages)
            else:
                # 大文件：自适应渲染线程数（CPU核心数 - 消费实例数），最多32
                n_workers = min(max(4, cpu_cores - num_consumers), total_pages, 32)
                n_workers = min(n_workers, total_pages)
            n_workers = max(1, n_workers)
            # 队列反压：maxsize控制预渲染量，避免撑爆内存
            # 满队列时put()自动阻塞→渲染线程等待→自然调节投喂速度
            print(f"[PDFProcessor] {n_workers} render threads, queue={render_queue.maxsize} (CPU={cpu_cores}, dual={num_consumers>1})")
            print(f"[PDFProcessor] 提示：{n_workers} 个渲染线程并行分段渲染，"
                                  f"识别队列的页号不会连续（例如先出现第 470 页、再回到第 90 页），"
                                  f"这是并行的正常现象，不代表跳过或乱序处理。")
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
            self._pipeline_alive = True
            for c in consumers:
                c.join()
            for t in producers:
                t.join()
            self._pipeline_alive = False
            # 【2026-10-10 修复·之十一】用户取消 → 本文件就此作废，直接返回 None,None。
            # 绝不落到下面的「完整性闸门 / 整份文件重处理」——那正是取消被无视的元凶。
            if self._user_cancelled:
                print("[Watchdog] 用户已取消 → 放弃本文件（不重跑、不生成结果）")
                return None, None
            # 【2026-10-09 完整性闸门】用户要求：宁可整份文件重跑，也绝不交付
            # 错页 / 漏页 / 乱序的结果。这里是输出前的最后一道校验。
            # （页序本身是安全的：write_results 用 sorted(results.keys()) 按页码升序写，
            #   结果也是按页码 pn 存进 dict 的；乱序只可能来自「缺页」，而缺页会被这里拦住。）
            if not self._cancelled and not self._user_cancelled:
                _bad = self._incomplete_pages(total_pages)
                if _bad:
                    _show = "、".join(str(p + 1) for p in _bad[:10])
                    if len(_bad) > 10:
                        _show += " …"
                    print("[Watchdog] 完整性校验未通过：%d/%d 页没有有效结果（页码：%s）"
                          % (len(_bad), total_pages, _show))
                    if _full_retry < self.MAX_FILE_RETRY:
                        print("[Watchdog] → 触发全文件重处理（第 %d/%d 次）"
                              % (_full_retry + 1, self.MAX_FILE_RETRY))
                    else:
                        print("[Watchdog] → 全文件重处理已用尽 → 跳过本文件并生成警告文件")
                    self._need_restart = True
            # 【2026-10-10 修复·之十一】用户取消时绝不允许进入「整份文件重处理」——
            # 这一条以前无条件执行，会把 _cancelled 复位成 False，于是取消 = 从头重跑。
            if (self._need_restart and not self._user_cancelled
                    and _full_retry < self.MAX_FILE_RETRY):
                print(f"[Watchdog] Full file restart #{_full_retry+1} triggered. Cleaning up...")
                self._cancelled = True
                self.ocr.force_close()
                time.sleep(2)  # 等所有线程感知到 cancelled
                # 重建引擎
                # 【2026-10-09 修复】原来是 restart_all()：但上面 force_close() 已经把
                # _instances 清空，restart_all() 遍历空列表 = 什么都没重建（恢复失效）。
                # 改用 rebuild()：按记住的构造参数把两个引擎真正重新启动起来。
                print(f"[Watchdog] Rebuilding all engines for restart...")
                self.ocr.rebuild()
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
            # 【2026-10-09 策略】单页重试 MAX_PAGE_RETRY 次 + 整份文件重处理 MAX_FILE_RETRY 次
            # 全部用尽仍不行 → 跳过该文件，并在源文件所在文件夹生成一份警告文件。
            # 既不产出「少了几页却看不出来」的结果文件，也不让用户以为它成功了。
            # （这里抛异常而不是 return None,None：调用方把 None 显示成「已取消」，
            #   用户根本看不出是失败；抛异常会走 file_error 信号，日志里明确写出来。）
            _bad = self._incomplete_pages(total_pages)
            print("[Watchdog] ERROR: 全文件重处理已用尽（%d/%d 页无有效结果）→ 跳过本文件"
                  % (len(_bad), total_pages))
            _notice = self._write_failure_notice(input_path, output_dir, total_pages, _bad)
            if _notice:
                print(f"[Watchdog] 已生成警告文件: {_notice}")
            raise RuntimeError(
                "本文件处理未完成：%d/%d 页没有有效结果。单页重试 %d 次、整份文件重处理 %d 次"
                "均已用尽，已跳过本文件。%s"
                % (len(_bad), total_pages, self.MAX_PAGE_RETRY, self.MAX_FILE_RETRY,
                   ("已生成警告文件：%s。" % _notice) if _notice
                   else "（警告文件写入失败，请查看日志。）"))
        return self.write_results(input_path, output_dir, total_pages, scale, vertical_sort=vertical_sort,
                                  overwrite_ocr=overwrite_ocr)

    def write_results(self, input_path, output_dir, total_pages, scale, vertical_sort=False,
                     overwrite_ocr=False):
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
                    # OCR引擎返回的是渲染图像像素坐标（图像=页面xscale），
                    # 写入PDF前必须除以scale换算为PDF坐标，否则文字落在页面外无法提取/搜索
                    x0, y0 = box[0]
                    x2, y2 = box[2]
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
                "author": "CathayOCR Lite",
                "subject": f"OCR extracted on {et}",
                "creator": "CathayOCR Lite Processor",
            })
            try:
                if total_pages <= 2000:
                    # output_pdf.subset_fonts()  [REMOVED: causes CJK character loss]
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
                 engine_id="ncnn_vulkan", use_gpu=True,
                 vertical_text=True, limit_side_len=2000,
                 model_size="PP_OCRv6_medium", use_angle_cls=False,
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
    # ── 主语言列表（显示名 → 语言代码） ──
    # 顺序 = 下拉里的显示顺序：先放轻量版支持的，再放需专业版的（见 _lang_combo_codes）。
    # ⚠ 这里**故意不逐条罗列拉丁语系语言**：v6 通用字典实测含约 400 个带重音拉丁字母
    #   （拉丁-1补充 64 + 拉丁扩展A 128 + 拉丁扩展B 208），实际覆盖「所有拉丁字母语言」；
    #   逐条列反而会让人以为只支持列出的那几种。
    _LANG_ITEMS = [
        # ── 轻量版支持（ncnn v6 通用字典实测覆盖） ──
        ("中文 (Chinese)", "ch"),
        ("日本語 (日文)", "japan"),
        ("English (英文)", "en"),
        ("Français (法文)", "fr"),
        ("Deutsch (德文)", "de"),
        ("Español (西班牙文)", "es"),
        ("Italiano (意大利文)", "it"),
        ("Português (葡萄牙文)", "pt"),
        ("Nederlands (荷兰文)", "nl"),
        ("Polski (波兰文)", "pl"),
        ("Magyar (匈牙利文)", "hu"),
        ("Čeština (捷克文)", "cs"),
        ("Tiếng Việt (越南文)", "vi"),
        ("Ελληνικά (希腊文)", "el"),
        ("多语言混排 (中·英·日 + 拉丁语系)", "multilang_v6"),

        # ── 需专业版（ncnn v6 字典 0 覆盖，选了必乱码） ──
        ("한국어 (韩文)", "korean"),
        ("Pусский (俄文)", "ru"),
    ]

    # 语言支持范围不常驻界面（太占地方），写进下拉的悬停 tooltip。
    # 依据：实测 ncnn 的 models/ppocr_keys_v6.txt（18709 条）
    #   CJK 汉字 15565 + 扩展A 137 · 日文假名 180 · 希腊 76 · 带重音拉丁字母约 400 ✅
    #   韩文谚文 0 · 西里尔 0 · 阿拉伯 0 · 天城文 0 · 泰/泰卢固/泰米尔 0 ❌
    # 需专业版的项在显示名后加这个后缀，把「为什么灰」直接写在字面上，不用悬停猜
    _LITE_UNSUPPORTED_SUFFIX = "（需专业版）"
    # 「模型级不支持」后缀：该文字系不在「当前模型所用的那份字典」里 → 灰显、不可选
    _LITE_MODEL_UNSUPPORTED_SUFFIX = "（此模型不支持）"
    # 「部分支持」后缀：字典只覆盖基础字母，会缺重音/变音符号 → 仍可选中
    _LITE_PARTIAL_SUFFIX = "（部分支持·可能缺重音）"

    # ── 简单模式语言列表：(显示名, 语言代码) ──
    #   支持的排前面，用一条分隔行隔开，灰显项 = 轻量版识别会乱码
    _SIMPLE_LANG_SEP = "__NEED_PRO__"
    _SIMPLE_LANG_ITEMS = [
        # ── 轻量版支持 ──
        ("中文（简体 / 繁体）", "ch"),
        ("日文（含假名）", "japan"),
        ("英文 English", "en"),
        ("拉丁字母语言（法·德·西·意·葡·荷·波·捷·匈·土·越 等）", "fr"),
        ("希腊文", "el"),
        ("中·英·日·拉丁 混排", "multilang_v6"),
        # ── 需专业版 ──
        ("───────── 以下文字需专业版 ─────────", _SIMPLE_LANG_SEP),
        ("韩文（谚文）" + _LITE_UNSUPPORTED_SUFFIX, "korean"),
        ("西里尔文系（俄 / 乌 / 保 等）" + _LITE_UNSUPPORTED_SUFFIX, "ru"),
        ("阿拉伯文系（阿 / 波 / 维 / 乌 等）" + _LITE_UNSUPPORTED_SUFFIX, "ar"),
        ("天城文系（印地 / 马拉地 等）" + _LITE_UNSUPPORTED_SUFFIX, "hi"),
        ("泰文" + _LITE_UNSUPPORTED_SUFFIX, "th"),
        ("泰卢固文" + _LITE_UNSUPPORTED_SUFFIX, "te"),
        ("泰米尔文" + _LITE_UNSUPPORTED_SUFFIX, "ta"),
    ]
    _LITE_UNSUPPORTED_GROUPS = {"korean", "ru", "ar", "hi", "th", "te", "ta"}
    # 专业模式语言下拉的灰显黑名单：_LANG_ITEMS 里 v6 字典 0 覆盖的那两项。
    # （Lite 恒用 ncnn Vulkan，字典由「模型」决定：默认 v6 模型 → ppocr_keys_v6.txt）
    _LITE_UNSUPPORTED_CODES = {
        "korean", "ru",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CathayOCR Lite (轻量版) - PDF OCR处理器")
        # ── 系统托盘：只作「显示 / 退出」入口，不接管最小化（详见 _setup_tray）──
        self._tray = None
        self._quitting = False
        # ── 迷你窗口：处理中把两个窗口收成一张右下角小进度卡（详见 MiniWindow）──
        self._mini = None
        self._mini_active = False
        self._mini_prev_state = Qt.WindowNoState
        self._current_file_name = "—"
        self._set_app_icon()
        # 处理中标志：为 True 时禁止一切会改变识别配置的操作
        # （含「简单模式 ↔ 专业模式」切换、语言/显卡/输出目录修改、往列表加文件）
        self._processing = False
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
            "独显 (显存≥12GB, NVIDIA/AMD/Intel)",
            "独显 (显存≤8GB, NVIDIA/AMD/Intel)",
            "仅有核显",
            "纯CPU (兼容性最好, 速度最慢)",
            "🤖 不知道有没有独显 / 显存多大 → 自动检测",
        ])
        self.simple_gpu.setMinimumWidth(300)
        def set_gpu_tip(idx):
            tips = [
                "大显存独显 → 可用较大边长；精度优先时边长 2560",
                "≤8GB 显存独显 → 精度优先时边长 2240 避免爆显存",
                "集成显卡显存/算力有限 → 精度优先时边长 2240",
                "纯CPU运行，无显存限制；精度优先时边长 2560，但速度最慢",
                "不知道有没有独立显卡，或者不知道显存多大？选这项，软件自动检测后帮你落到上面某一档\n"
                "（检测顺序：nvidia-smi 读 NVIDIA 显存 → Vulkan 探测独显 → 都没有则按纯CPU）",
            ]
            self.simple_gpu.setToolTip(tips[idx] if idx < len(tips) else "")
        self.simple_gpu.currentIndexChanged.connect(set_gpu_tip)
        set_gpu_tip(0)
        r3.addWidget(self.simple_gpu)
        r3.addStretch()
        sl.addLayout(r3)

        # 问题4：语言（先"轻量版支持"，分隔行之后是"需专业版" → 灰显不可选）
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("🌐 文档语言"))
        self.simple_lang = QComboBox()
        for _disp, _code in self._SIMPLE_LANG_ITEMS:
            self.simple_lang.addItem(_disp, _code)
        _lm = self.simple_lang.model()
        for _i, (_disp, _code) in enumerate(self._SIMPLE_LANG_ITEMS):
            if _code == self._SIMPLE_LANG_SEP:
                _lm.item(_i).setEnabled(False)  # 分隔行：只作分界提示，不可选
                self.simple_lang.setItemData(_i, "分隔行：其下文字轻量版不支持", Qt.ToolTipRole)
            elif _code in self._LITE_UNSUPPORTED_GROUPS:
                _lm.item(_i).setEnabled(False)
                self.simple_lang.setItemData(
                    _i,
                    "轻量版只有 ncnn 引擎，其 PP-OCRv6 通用字典不含该文字（识别会乱码）。\n"
                    "请使用专业版：NVIDIA 显卡自动切换到专用分语种模型；\n"
                    "AMD/Intel/纯CPU 则使用 PP-OCRv5 (Paddle CPU) 备选引擎。",
                    Qt.ToolTipRole)
        self.simple_lang.setMinimumWidth(300)
        self.simple_lang.setMaxVisibleItems(20)
        self.simple_lang.setToolTip(
            "选择文档里会出现哪些文字。\n"
            "（轻量版只有 ncnn 引擎：认得出什么由「模型」决定，语言本身不改变识别结果）\n"
            "✅ 支持：汉字（简/繁）· 日文假名 · 所有拉丁字母语言 · 希腊文\n"
            "❌ 需专业版：韩文 · 西里尔文（俄等）· 阿拉伯文 · 天城文 · 泰文 · 泰卢固文 · 泰米尔文\n"
            "带「（需专业版）」的灰显项 = ncnn 字典 0 覆盖，选了也认不出。\n"
            "💡 中文/拉丁/日文字符都在同一份字典里（中文字典本身含整套拉丁字母），\n"
            "   所以中英混排直接选「中文」即可，不必切到英文。\n"
            "注：拉丁字母语言只列了常用的几种，实际所有用拉丁字母的语言都支持。")
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
        self.engine_combo = QLabel(engine_display_name("ncnn_vulkan") + " — CathayOCR Lite")
        self.engine_combo.setToolTip(
            "CathayOCR Lite (轻量版)\n"
            "GPU模式+Vulkan加速 | CPU模式+ncnn原生\n"
            "支持NVIDIA/AMD/Intel任意显卡\n"
            "✅ 支持：汉字（简/繁）· 日文假名 · 所有拉丁字母语言\n"
            "        （英/法/德/西/意/葡/荷/波/捷/匈/土/越 等，含带重音字母）· 希腊文\n"
            "❌ 需专业版：韩文 · 西里尔文（俄/乌/保）· 阿拉伯文 · 天城文（印地）\n"
            "          · 泰文 · 泰卢固文 · 泰米尔文\n"
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
            "  同一引擎下，改模型不依赖网络下载\n"
            "  ⚠ 模型决定用哪份字典，语言列表会随之变化"
        )
        # 切模型 → 所用字典变了 → 语言列表必须跟着重建
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        el.addWidget(self.model_combo)
        el.addStretch()
        el.addWidget(QLabel("语言:"))
        self.lang_combo = QComboBox()
        self.lang_combo.setMinimumWidth(260)
        self.lang_combo.setMaxVisibleItems(24)
        # 条目由 _update_lang_combo 在建窗末尾统一填充（支持项在前、需专业版在后并加后缀）
        self.lang_combo.setToolTip(
            "选择识别语言（轻量版 = ncnn 单一引擎）：\n"
            "  · 语言只用来声明「文档里会出现哪些文字」；真正决定认得出什么的是【模型】——\n"
            "    字典跟着模型走，多个语言**共用同一份字典**：\n"
            "      v6 模型 → v6 通用字典 / v5 模型 → v5 字典 / v3·v4 模型 → v1 字典。\n"
            "    所以同一次运行里，在这些语种之间来回切，识别结果不会变。\n"
            "  · 默认 PP-OCRv6 模型覆盖：汉字（简/繁）+ 日文假名 + 所有拉丁字母语言 + 希腊文。\n"
            "  · 带「（需专业版）」的灰显项 = 四份 ncnn 字典实测 0 覆盖（识别必乱码），需换专业版。\n"
            "  · 若把模型换成 v5 / v4 / v3，拉丁带重音字母的覆盖会明显变窄。\n"
            "\n"
            "💡 「中文模式能不能认英文？」能 —— 中文字典本身就内置整套拉丁字母\n"
            "   （v6 通用字典 462 个拉丁字符、v5 字典 145 个、v1 字典 86 个，a-z/A-Z/0-9 都齐全）。\n"
            "   所以中英混排文档直接选「中文」即可；只是纯英文时专门的拉丁/英文模型精度更好。\n"
            "  ⚠ 选中模型不同时，本提示下方会列出该模型实际用的字典与被排掉的文字系。"
        )
        self._lang_tip_base = self.lang_combo.toolTip()   # 语言框 tooltip 会随模型改写
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
            "竖排阅读顺序优化 (古籍/碑帖/对联等竖排文档建议开启):\n"
            "  开启后，OCR结果按竖排阅读顺序重新排列:\n"
            "  先按文字列位置 从右→左，列内再按 从上→下\n"
            "  普通横排文档 → 关闭保持原有顺序\n"
            "  纯CPU后处理排序，几乎不增加耗时"
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
        self.overwrite_ocr_check = QCheckBox("覆盖旧OCR")
        self.overwrite_ocr_check.setChecked(self.cfg.value("overwrite_ocr", False, type=bool))
        self.overwrite_ocr_check.setToolTip(
            "覆盖旧OCR层 (导出双层PDF时生效):\n"
            "  开启后，物理删除原PDF中的旧文字层，仅保留本次新识别文字\n"
            "  扫描图像层与矢量图形 100% 无损保留\n"
            "  适合对已有OCR层但效果差的PDF重新识别\n"
            "  未开启时，新旧文字层合并保留（原有行为）"
        )
        ol.addWidget(self.overwrite_ocr_check)
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
        self._update_lang_combo()
        self._update_mode_combo()
        self._populate_gpu_combo()

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
                        if vram is not None and vram >= 11000:  # ≈12GB 分界（8GB=8192MB 属小显存）
                            return 0  # 大显存独显
                        return 1  # 小显存独显（显存读不到时同样保守按小显存档）
                # 仅有核显
                return 2
        except Exception:
            pass
        # 无GPU
        return 3

    def _on_ui_mode_changed(self, simple_mode):
        """切换简单模式/专业模式"""
        # 【安全闸】处理中禁止切换界面模式。
        # 切到简单模式会走 _sync_simple_from_expert → _apply_simple_settings，
        # 直接改写模型/显卡/边长等参数；而这些参数已被 BatchWorkerThread 取走用于
        # 正在跑的任务，界面显示与任务实际参数会就此分叉。这里拦掉并还原单选按钮。
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
        self.simple_lang.blockSignals(True)

        mode = self.mode_combo.currentData()
        # 显卡：根据mode推断
        if mode == "cpu":
            self.simple_gpu.setCurrentIndex(3)  # 纯CPU
        else:
            self.simple_gpu.setCurrentIndex(0)  # 独显
        # 精度：根据模型 + 精对齐判断（与专业版保持一致）
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
        # 语言：专业模式的语言代码 → 归并到简单模式的组；不支持的文字回落到"中文"
        # 直接读 itemData（不再靠显示文本反查：文本带「（需专业版）」后缀会对不上）
        _pro_code = self.lang_combo.currentData() or "ch"
        _LATIN_SIMPLE = {"fr", "de", "es", "it", "pt", "nl", "ro", "ca", "gl", "da", "sv",
                         "no", "fi", "is", "pl", "cs", "sk", "hu", "hr", "sl", "bs",
                         "rs_latin", "sq", "ga", "cy", "et", "lt", "lv", "mt", "la", "pi",
                         "af", "az", "uz", "ku", "eu", "oc", "vi", "id", "ms", "tl", "sw",
                         "mi", "tr"}
        if _pro_code in _LATIN_SIMPLE:
            _pro_code = "fr"
        if _pro_code in self._LITE_UNSUPPORTED_GROUPS:
            _pro_code = "ch"  # 灰显项不可落在简单模式里，回落中文
        _si = self.simple_lang.findData(_pro_code)
        self.simple_lang.setCurrentIndex(_si if _si >= 0 else 0)

        self.simple_gpu.blockSignals(False)
        self.simple_speed.blockSignals(False)
        self.simple_doc.blockSignals(False)
        self.simple_lang.blockSignals(False)

        # 刷新 GPU 悬停提示（信号被阻断后需要手动调用）
        gpu_tips = [
            "大显存独显 → 可用较大边长；精度优先时边长 2560",
            "≤8GB 显存独显 → 精度优先时边长 2240 避免爆显存",
            "集成显卡显存/算力有限 → 精度优先时边长 2240",
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
        doc_idx = self.simple_doc.currentIndex()
        speed_idx = self.simple_speed.currentIndex()
        gpu_idx = self.simple_gpu.currentIndex()

        # ── 0. 「帮我选择」→ 自动检测显卡档位（0/1/2/3）──
        if gpu_idx == 4:
            if getattr(self, "_auto_detecting", False):
                # 防递归：探测过程中直接按"纯CPU"继续
                self._auto_detecting = True
                self.simple_gpu.blockSignals(True)
                self.simple_gpu.setCurrentIndex(3)
                self.simple_gpu.blockSignals(False)
                self._auto_detecting = False
                gpu_idx = 3
            else:
                self._auto_detecting = True
                best = self._auto_detect_gpu_idx()
                self._auto_detecting = False
                self.simple_gpu.blockSignals(True)
                self.simple_gpu.setCurrentIndex(best)
                self.simple_gpu.blockSignals(False)
                gpu_idx = best

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

        # 文档类型只决定「竖排 / 方向矫正」开关；边长与渲染倍率由统一规则决定
        if doc_idx == 0:  # 普通文档
            target_vertical = False
            target_angle = False
        elif doc_idx == 1:  # 古籍竖排
            target_vertical = True
            target_angle = False
        else:  # 扫描件
            target_vertical = False
            target_angle = True
        target_side, target_scale = _simple_side_scale(doc_idx, speed_idx, gpu_idx)

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
        # 语言同步：简单模式的语言代码 → 专业模式下拉（按代码定位，不再靠显示文本；
        # 旧代码用 findText 匹配显示文本，导致「希腊文」在专业模式语言表里找不到 → 悄悄按中文跑）
        _li = self.simple_lang.currentIndex()
        _entry = self._SIMPLE_LANG_ITEMS[_li] if 0 <= _li < len(self._SIMPLE_LANG_ITEMS) else None
        if (_entry is None or _entry[1] == self._SIMPLE_LANG_SEP
                or _entry[1] in self._LITE_UNSUPPORTED_GROUPS):
            _code = "ch"
            self.simple_lang.blockSignals(True)
            self.simple_lang.setCurrentIndex(0)
            self.simple_lang.blockSignals(False)
        else:
            _code = _entry[1]
        lang_idx = self.lang_combo.findData(_code)
        if lang_idx >= 0 and self.lang_combo.model().item(lang_idx).isEnabled():
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
        # 存「不带后缀的规范显示名」：lang_val 与专业版共用同一个配置键，
        # 存后缀会污染专业版那边的匹配（专业版按显示文本还原语言）。
        self.cfg.setValue("lang_val", self._lang_canonical_text())

    def _lang_canonical_text(self):
        """当前选择对应的规范显示名（_LANG_ITEMS 里那个，不带「（需专业版）」后缀）。"""
        code = self.lang_combo.currentData()
        for label, c in self._LANG_ITEMS:
            if c == code:
                return label
        return self.lang_combo.currentText()

    def _ncnn_model_lang_tiers(self):
        """当前模型所用字典的「语言代码 → 档位」表（ok / partial / unsupported）。"""
        try:
            model_base = self.model_combo.currentData() or ""
            info = _PLUGIN_DIRS.get("ncnn_vulkan") or {}
            pool = _load_ncnn_dict_pool(info.get("plugin_dir", ""),
                                        _ncnn_keys_file_for_model(model_base))
            return {code: _ncnn_lang_tier(code, pool) for _n, code in self._LANG_ITEMS}
        except Exception:
            return {}

    def _lang_combo_codes(self):
        """语言下拉的条目计划：[(显示文本, 代码, 档位)]。

        档位：'ok' / 'partial' / 'unsupported_model'（不可选）/ 'unsupported_engine'（不可选）/ 'sep'。
        支持的排前面，紧跟一条不可选的分隔行，需专业版的排在最后并加「（需专业版）」后缀。
        ⚠ 不逐条罗列拉丁语系语言（原因见 _LANG_ITEMS 上方注释）。
        ⚠ 档位随**当前模型**变 —— 模型决定用哪份字典，四份字典覆盖差别很大：
           日文假名 v1 字典只有 4%、v6_tiny 字典是 0%（v5/v6 是 100%）；
           希腊文 v1 字典只有 21%；越南文四份字典都只有 12~41%。
        """
        tiers = self._ncnn_model_lang_tiers()
        items, need_pro = [], []
        for label, code in self._LANG_ITEMS:
            if code in self._LITE_UNSUPPORTED_CODES:
                need_pro.append((label, code))
                continue
            t = tiers.get(code, "ok")
            if t == "unsupported":
                items.append((f"{label}{self._LITE_MODEL_UNSUPPORTED_SUFFIX}",
                              code, "unsupported_model"))
            elif t == "partial":
                items.append((f"{label}{self._LITE_PARTIAL_SUFFIX}", code, "partial"))
            else:
                items.append((label, code, "ok"))
        if need_pro:
            items.append((self._SIMPLE_LANG_SEP, self._SIMPLE_LANG_SEP, "sep"))
            items += [(f"{label}{self._LITE_UNSUPPORTED_SUFFIX}", code, "unsupported_engine")
                      for label, code in need_pro]
        return items

    def _lang_index_for(self, want):
        """按 lang_val 还原语言选择：新版存代码、旧版存显示文本，两种都认。"""
        if not want:
            return -1
        i = self.lang_combo.findData(want)      # 新版：语言代码
        if i >= 0:
            return i
        i = self.lang_combo.findText(want)      # 旧版：完整显示名
        if i >= 0:
            return i
        for k in range(self.lang_combo.count()):    # 旧值可能是「名 + 后缀」
            if self.lang_combo.itemText(k).startswith(str(want)):
                return k
        for label, code in self._LANG_ITEMS:    # 旧值可能被别的版本改过写法
            if label == want:
                i = self.lang_combo.findData(code)
                if i >= 0:
                    return i
        return -1

    def _lang_index_enabled(self, i):
        try:
            return bool(self.lang_combo.model().item(i).isEnabled())
        except Exception:
            return False

    def _first_enabled_lang_index(self):
        for i in range(self.lang_combo.count()):
            if self._lang_index_enabled(i):
                return i
        return 0

    def _update_lang_combo(self):
        self.lang_combo.blockSignals(True)
        self.lang_combo.clear()
        brush_gray = QBrush(QColor("#8b949e"))
        brush_amber = QBrush(QColor("#b8860b"))
        gray_engine, gray_model, partial = [], [], []
        raw_label = {c: l for l, c in self._LANG_ITEMS}   # 代码 → 不带后缀的原名
        # 顶部说明行：把「语言在这里不参与识别」直接摆在列表最上面（不必悬停才看到）
        self.lang_combo.addItem(
            "ⓘ 语言只作声明、不参与识别 —— 字典跟着「模型」走，下列语言共用同一份字典", None)
        _note = self.lang_combo.model().item(0)
        if _note is not None:
            _note.setEnabled(False)
            _note.setForeground(brush_gray)
        for text, code, state in self._lang_combo_codes():
            if state == "sep":
                self.lang_combo.addItem(text, code)
                _i = self.lang_combo.count() - 1
                self.lang_combo.model().item(_i).setEnabled(False)
                self.lang_combo.setItemData(_i, "分隔行：以下文字轻量版不支持", Qt.ToolTipRole)
                continue
            # 显示文本（含「部分支持／此模型不支持／需专业版」后缀）由 _lang_combo_codes 产出
            self.lang_combo.addItem(text, code)      # itemData 恒为语言代码，不带后缀
            _i = self.lang_combo.count() - 1
            item = self.lang_combo.model().item(_i)
            if state == "unsupported_engine":
                item.setEnabled(False)
                item.setForeground(brush_gray)
                self.lang_combo.setItemData(
                    _i,
                    "轻量版不支持该文字：四份 ncnn 字典实测 0 覆盖，识别必乱码。\n"
                    "请使用专业版：NVIDIA 显卡自动切换到专用分语种模型；\n"
                    "AMD / Intel / 纯 CPU 则使用 PP-OCRv5 (Paddle CPU) 备选引擎。",
                    Qt.ToolTipRole)
                gray_engine.append(raw_label.get(code, text))
            elif state == "unsupported_model":
                item.setEnabled(False)
                item.setForeground(brush_gray)
                self.lang_combo.setItemData(
                    _i,
                    "当前模型用的那份字典里没有这套文字（实测覆盖 < 60%），识别必乱码。\n"
                    "换成 PP-OCRv6 模型（small / medium）即可支持；或改用专业版。",
                    Qt.ToolTipRole)
                gray_model.append(raw_label.get(code, text))
            elif state == "partial":
                item.setForeground(brush_amber)
                self.lang_combo.setItemData(
                    _i,
                    "当前模型用的那份字典只覆盖该语言的基础字母，可能缺重音/变音符号。\n"
                    "换成 PP-OCRv6 模型覆盖最全。",
                    Qt.ToolTipRole)
                partial.append(raw_label.get(code, text))
        # tooltip = 基础说明 + 本次实测的「模型 → 字典 → 覆盖」
        model_base = self.model_combo.currentData() or "(未选)"
        keys_file = _ncnn_keys_file_for_model(model_base)
        info = _PLUGIN_DIRS.get("ncnn_vulkan") or {}
        pool = _load_ncnn_dict_pool(info.get("plugin_dir", ""), keys_file)
        lines = [getattr(self, "_lang_tip_base", ""), "",
                 f"当前模型：{model_base}",
                 f"所用字典：{keys_file}（{len(pool)} 个字符）",
                 "",
                 "💡 该字典内 a-z / A-Z / 0-9 齐全 —— 中文模式下英文照样能识别，",
                 "   中英混排文档不必切到英文；专门的拉丁/英文模型精度更高。"]
        if gray_model:
            lines += ["", "✕ 这份字典里没有（已置灰、不可选）：" + "、".join(gray_model)]
        if partial:
            lines += ["", f"⚠ 只覆盖基础字母、可能缺重音（{len(partial)} 项）："
                      + "、".join(partial)]
        if gray_engine:
            lines += ["", f"（另有 {len(gray_engine)} 种文字系四份 ncnn 字典都不含："
                      + "、".join(gray_engine) + "）"]
        self.lang_combo.setToolTip("\n".join(lines))
        idx = self._lang_index_for(self.cfg.value("lang_val", ""))
        if idx < 0 or not self._lang_index_enabled(idx):
            idx = self._first_enabled_lang_index()
        self.lang_combo.setCurrentIndex(idx)
        self.lang_combo.blockSignals(False)

    def _on_model_changed(self):
        """切「模型」→ 所用字典变了 → 重建语言列表（并提示被排掉的文字系）。"""
        prev = self._lang_canonical_text() if self.lang_combo.count() else ""
        self._update_lang_combo()
        now = self._lang_canonical_text() if self.lang_combo.count() else ""
        if prev and now and prev != now:
            try:
                self.log(f"[语言] 新模型所用字典不含「{prev}」，已自动回落到「{now}」")
            except Exception:
                print(f"[语言] 新模型所用字典不含「{prev}」，已回落到「{now}」")

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
        self.cfg.setValue("angle_cls", self.angle_cls_check.isChecked())
        self.cfg.setValue("rec_batch", self.rec_batch_spin.value())
        self.cfg.setValue("shrink", self.shrink_check.isChecked())
        self.cfg.setValue("tensorrt", self.tensorrt_check.isChecked())
        self.cfg.setValue("dual", self.dual_check.isChecked())
        self.cfg.setValue("overwrite_ocr", self.overwrite_ocr_check.isChecked())
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
        # 单一 ncnn_vulkan 引擎，无需 restore。
        # 但要做「一次性迁移」：旧版本只在「开始处理」时落盘设置，用户改过「自动(推荐)」
        # 后直接关窗就丢了，注册表里长期残留 mode_val='cpu' → 每次打开都回到 CPU 模式。
        # 这里复位一次并打标记，之后完全按用户自己的选择记忆（v1.3.1 起 closeEvent 也落盘）。
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
        model_size = self.model_combo.currentData() or _get_first_valid_ncnn_model(engine_id) or "PP_OCRv6_medium"
        lang_display = self.lang_combo.currentText() or "中文 (Chinese)"
        # 直接取 itemData：显示文本可能带「（需专业版）」后缀，靠文本反查会失配
        ocr_lang = self.lang_combo.currentData() or "chinese"
        self.log(f"  语言: {lang_display} ({ocr_lang})")
        use_angle_cls = self.angle_cls_check.isChecked()
        scale = self.scale_combo.currentIndex() + 1
        overwrite_ocr = self.overwrite_ocr_check.isChecked()
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
        self._set_buttons_idle()
        # 【2026-10-10 修复·之十一】用户中途取消 ≠ 处理完成。
        # 旧版无论怎么结束都弹「批量处理完成!」，取消后看着像跑完了。
        _was_cancelled = bool(self.worker and getattr(self.worker, "is_cancelled", False))
        if _was_cancelled:
            self.log(f"\n已取消。本批已完成 {success}/{total} 个文件（剩余未处理）")
            self.status_label.setText(f"已取消 - 已完成 {success}/{total}")
            self.speed_label.setText("处理速度: -- (已取消)")
            self.overall_progress.setValue(int((self.processed_files / self.total_files) * 10000)
                                           if self.total_files else 0)
            QMessageBox.information(
                self, "已取消",
                f"已取消本批任务。\n\n已完成: {success}/{total} 个文件"
                f"\n输出目录: {self.output_edit.text()}")
            return
        self.log(f"\n处理完成! 成功: {success}/{total}")
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
        self.model_combo.setEnabled(True)
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
    APP_DISPLAY_NAME = "CathayOCR Lite"
    LOG_WINDOW_TITLE = "CathayOCR Lite — 运行日志"   # 必须与启动器 APP_TITLE 一致

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

        日志窗口是独立进程里的 tkinter 窗口，这里只做「显示 + 置顶」，不改它的运行。

        【2026-10-09】日志窗口现在被登记成「本主窗口的从属窗口」，主窗口不可见时
        它自己也显示不出来；所以先把主界面叫回来（迷你模式下等价于点「恢复」），
        再调日志窗口，两个才会一起出现在最前面。
        """
        try:
            if getattr(self, "_mini_active", False):
                self.exit_mini_mode()
            else:
                self.show()
                if self.isMinimized():
                    self.showNormal()
                self.raise_()
                self.activateWindow()
        except Exception:
            pass
        try:
            import ctypes
            u = ctypes.windll.user32
            hwnd = u.FindWindowW(None, self.LOG_WINDOW_TITLE)
            if not hwnd:
                hwnd = u.FindWindowW("TkTopLevel", self.LOG_WINDOW_TITLE)
            if not hwnd:
                self.log("（日志窗口不在运行 —— 直接双击 CathayOCR Lite.exe 即可打开）")
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
        # 清理属于本目录的残余子进程
        self._kill_orphans()
        event.accept()

    def _kill_orphans(self):
        """只结束『属于本程序目录』的残余引擎进程。
        【修复】旧实现 taskkill /f /im 是按进程名全系统强杀，
        会误伤其它位置的同名引擎（例如另一份正在运行的 CathayOCR）。"""
        killed = _kill_engine_processes(CLEANUP_TARGETS)
        if killed:
            print(f"[MainWindow] Cleaned up local engine pids: {killed}")


CLEANUP_TARGETS = ["ppocr_ocr_vulkan.exe"]


def _kill_engine_processes(names, dirs=None):
    """仅结束『exe 路径位于本程序引擎目录内』且名字匹配的进程。
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


def _kill_orphan_engines(names=None):
    """结束「父进程已经消失」的孤儿引擎进程（2026-10-09 卡死修复·之二）。

    为什么需要它：主程序若被强制结束（任务管理器结束任务 / 崩溃 / 断电），
    它启动的引擎进程不会跟着退出 —— 这些孤儿引擎仍然 LISTEN 着 18043 / 18053。
    下次启动时新引擎 bind 不上（Windows 允许端口复用），内核会把连接随机分发，
    于是一半请求打进这个「父进程早已不存在」的僵尸实例：connect 成功、却永远
    等不到回复 → recv 卡满 180 秒超时 → GPU 占用率归零。

    _kill_engine_processes() 只清「本程序目录内」的引擎，覆盖不到这些孤儿
    （典型来源：开发目录、旧版本目录、别的安装位置留下的）。

    ★ 判定依据是「父进程是否还存在」，所以**绝不会**误伤另一份正在运行的
      CathayOCR —— 它的引擎父进程活着，会被直接跳过。C 盘那份不受影响。
    """
    if platform.system() != "Windows":
        return []
    names_l = {str(n).lower() for n in (names or CLEANUP_TARGETS)}
    try:
        import ctypes
        from ctypes import wintypes
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
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == INVALID_HANDLE_VALUE:
        return []

    alive, targets = set(), []
    try:
        pe = PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(pe))
        while ok:
            pid = int(pe.th32ProcessID)
            alive.add(pid)
            if pid != os.getpid() and (pe.szExeFile or "").lower() in names_l:
                targets.append((pid, int(pe.th32ParentProcessID)))
            ok = kernel32.Process32NextW(snap, ctypes.byref(pe))
    finally:
        kernel32.CloseHandle(snap)

    killed = []
    for pid, ppid in targets:
        # 父进程仍在 → 一定是另一份正在正常运行的软件，绝不碰（保守起见 PID 复用也算）
        if ppid in alive and ppid != 0:
            continue
        h = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE, False, pid)
        if not h:
            continue
        try:
            if kernel32.TerminateProcess(h, 1):
                killed.append(pid)
        finally:
            kernel32.CloseHandle(h)
    return killed


def _our_engine_dirs():
    """本程序所有引擎可执行文件所在目录（绝对路径）。"""
    dirs = []
    try:
        for info in _PLUGIN_DIRS.values():
            entry = info.get("entry_path")
            if entry:
                dirs.append(os.path.dirname(os.path.abspath(entry)))
    except Exception:
        pass
    return dirs


def _force_cleanup():
    """强制清理（只针对本目录内的引擎进程）"""
    _kill_engine_processes(CLEANUP_TARGETS)


# ============================================================
# 【2026-10-10 新增】运行日志落盘（Tee）
#   启动器只把主程序的 stdout/stderr 显示在日志窗口里，**不写文件**：
#   窗口一关（或强杀）日志就没了，事后无法复盘。
#   这里把 print 出去的内容**同时**追加进
#       <软件根目录>/logs/<TAG>-YYYY-MM-DD.log
#   纯旁路：不改变任何原有行为；目录不可写时静默跳过。
# ============================================================
class _LogTee:
    """把写入镜像到日志文件，同时原样透传给原 stdout/stderr。"""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle
        self._lock = threading.Lock()

    def write(self, data):
        if not isinstance(data, str):
            try:
                data = str(data)
            except Exception:
                return 0
        try:
            if self._stream is not None:
                self._stream.write(data)
        except Exception:
            pass
        if data:
            try:
                with self._lock:
                    self._handle.write(data)
                    self._handle.flush()
            except Exception:
                pass
        return len(data)

    def writelines(self, lines):
        for ln in lines:
            self.write(ln)

    def flush(self):
        try:
            if self._stream is not None:
                self._stream.flush()
        except Exception:
            pass
        try:
            with self._lock:
                self._handle.flush()
        except Exception:
            pass

    def isatty(self):
        return False


_LOG_TEE_FILE = None


def _install_file_log(tag):
    """把 stdout / stderr 复制一份到 <软件根目录>/logs/<tag>-YYYY-MM-DD.log"""
    global _LOG_TEE_FILE
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "%s-%s.log" % (tag, time.strftime("%Y-%m-%d")))
        _LOG_TEE_FILE = open(path, "a", encoding="utf-8", buffering=1,
                             errors="replace")
    except Exception:
        _LOG_TEE_FILE = None
        return None
    try:
        sys.stdout = _LogTee(sys.stdout, _LOG_TEE_FILE)
        sys.stderr = _LogTee(sys.stderr, _LOG_TEE_FILE)
    except Exception:
        pass
    try:
        _LOG_TEE_FILE.write(
            "\n" + "=" * 72 + "\n"
            + "[%s] ======== 程序启动（pid=%d）========" % (
                time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid()) + "\n"
            + "[log] 日志文件: %s\n" % path)
        _LOG_TEE_FILE.flush()
    except Exception:
        pass
    return path


if __name__ == '__main__':
    _install_file_log("CathayOCR-Lite")   # 2026-10-10：运行日志落盘（旁路，不影响启动）
    # 启动前清理『本程序目录内』的残余引擎进程
    # 【修复】不再全局按名强杀，避免误伤其它目录正在运行的同类程序
    _kill_engine_processes(CLEANUP_TARGETS)
    # 【2026-10-09 补】再清一遍「父进程已消失」的孤儿引擎 —— 它们占着 18043/18053
    # 会让新引擎 bind 不上，请求打进僵尸实例后永远等不到回复（表现为处理卡死、GPU 归零）。
    # 只按「父进程是否还存在」判定，绝不会误伤另一份正在运行的 CathayOCR。
    try:
        _orphans = _kill_orphan_engines()
        if _orphans:
            print("[Main] Cleaned up orphan engine pids: %s" % _orphans)
    except Exception as _e:
        print("[Main] orphan engine cleanup skipped: %s" % _e)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window._restore_engine_settings()
    window._setup_tray()          # 系统托盘：最小化时与日志窗口一起收起
    window.show()
    sys.exit(app.exec_())
