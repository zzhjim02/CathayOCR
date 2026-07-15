#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PP-OCRv6/v5 Multi-Language ONNX Server (修改版)
============================================
支持全部 PaddleOCR 官方语言:
  - PP-OCRv6: Latin(50) + 中日文 (通过本地 ONNX 缓存)
  - PP-OCRv5: Arabic(8) / Cyrillic(33) / Devanagari(13) / 
              Korean / Thai / Greek / Telugu / Tamil
              (通过 PaddleX 自动下载 ONNX 模型)

所有语言均通过 onnxruntime 推理，完全避开 Paddle Inference。
"""
import sys
import os
import json
import argparse
import base64
import gc

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _script_dir and p != '' and p != '.']
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(_script_dir, "models"))

_ocr = None
_det = True
_shrink_ratio = 0.0
_blank_page_strategy = 'skip'
_use_gpu = False
_current_lang = "ch"


def _setup_nvidia_dlls():
    """将 CUDA/cuDNN/cuBLAS DLL 路径加入搜索路径（onnxruntime CUDA 加速必需）"""
    try:
        import sysconfig
        site_dir = sysconfig.get_paths()["purelib"]
        # 1. nvidia pip 包的 bin 目录
        nvidia_base = os.path.join(site_dir, "nvidia")
        if os.path.isdir(nvidia_base):
            for sub in os.listdir(nvidia_base):
                dll_dir = os.path.join(nvidia_base, sub, "bin")
                if os.path.isdir(dll_dir):
                    os.add_dll_directory(dll_dir)
                    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        # 2. onnxruntime capi 目录（已复制 CUDA DLL 到此处）
        ort_dir = os.path.join(site_dir, "onnxruntime", "capi")
        if os.path.isdir(ort_dir):
            os.add_dll_directory(ort_dir)
            os.environ["PATH"] = ort_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _cleanup_gpu_memory():
    if not _use_gpu:
        return
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import paddle
        if hasattr(paddle, "device") and hasattr(paddle.device, "cuda"):
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


def _get_gpu_total_memory_gb():
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.total / (1024**3)
    except Exception:
        pass
    return 8.0


# ─── PP-OCRv6 语言范围 ───────────────────────────────────
# PP-OCRv6 内置字符集覆盖：中日韩 + Latin(46) + Cyrillic(33)
# 这些语言可以用 PP-OCRv6 ONNX 模型
_V6_LANGS = frozenset({
    # Latin
    "ch", "en", "french", "german", "es", "it", "pt", "nl", "pl", "ro", "sv",
    "cs", "da", "fi", "hu", "no", "sk", "sl", "hr", "lt", "lv", "et",
    "id", "ms", "tl", "vi", "tr", "az", "uz", "sw", "sq", "mt", "ga",
    "mi", "oc", "la", "cy", "is", "bs", "af", "ku", "rs_latin", "pi",
    "fr", "de", "ca", "eu", "gl",
    # CJK
    "japan", "chinese_cht",
    # Korean (PP-OCRv6 内嵌韩文字符)
    "korean",
    # Cyrillic (PP-OCRv6 内嵌)
    "ru", "be", "uk", "bg", "mn", "kk", "ky", "tg", "mk", "tt", "cv",
    "ba", "rs_cyrillic",
})

# PP-OCRv5 专用语言（PP-OCRv6 不支持，需走 v5 ONNX 模型）
_V5_LANGS = frozenset({
    "ar", "fa", "ug", "ur", "ps", "sd", "bal",      # Arabic
    "hi", "mr", "ne", "bh", "mai", "ang", "bho",     # Devanagari
    "mah", "sck", "new", "gom", "sa", "bgc",
    "th", "el", "te", "ta",                           # Thai/Greek/Telugu/Tamil
    "abq", "ady", "kbd", "ava", "dar", "inh", "che", # Cyrillic 扩展
    "lbe", "lez", "tab",
    "bo",                                              # Tibetan (尝试)
})


def _select_engine(use_gpu, cpu_threads=None, model_size="medium"):
    """自动选择推理引擎。优先 onnxruntime（轻量），未安装则回退 paddle。"""
    _setup_nvidia_dlls()
    try:
        import onnxruntime as ort
    except ImportError:
        return None, None
    providers = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" in providers:
        gpu_total_gb = _get_gpu_total_memory_gb()
        ratio = 0.80 if model_size == "medium" else 0.40
        gpu_mem_limit = int(gpu_total_gb * ratio * 1024 * 1024 * 1024)
        cfg = {
            "device_type": "gpu",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "provider_options": [
                {
                    "device_id": 0,
                    "gpu_mem_limit": gpu_mem_limit,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "0",
                },
                {},
            ],
            "graph_optimization_level": 99,
            "enable_mem_pattern": False,
        }
    else:
        cfg = {
            "device_type": "cpu",
            "providers": ["CPUExecutionProvider"],
            "graph_optimization_level": 99,
            "enable_mem_pattern": True,
            "enable_cpu_mem_arena": True,
        }
    if cpu_threads is not None and cpu_threads > 0:
        cfg["intra_op_num_threads"] = cpu_threads
        cfg["inter_op_num_threads"] = cpu_threads
    return "onnxruntime", cfg


def parse_config(args):
    """解析模型配置（model_size + lang）"""
    config = {"model_size": "medium", "lang": "ch"}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.config_path:
        config_file = args.config_path
        if not os.path.isabs(config_file):
            config_file = os.path.join(script_dir, config_file)
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        key, val = parts[0], parts[1]
                        if key in ("model_size", "lang"):
                            config[key] = val
    return config


def _resolve_lang(lang):
    """统一语言代码格式"""
    mapping = {
        "chinese": "ch", "ch_sim": "ch", "simplified": "ch",
        "english": "en", "french": "fr", "german": "de",
        "japanese": "japan", "korean": "korean",
        "arabic": "ar", "russian": "ru",
        "thai": "th", "hindi": "hi", "greek": "el",
        "italian": "it", "spanish": "es", "portuguese": "pt",
        "telugu": "te", "tamil": "ta",
        "chinese_cht": "chinese_cht",
        "tibetan": "bo",
    }
    if lang in mapping:
        return mapping[lang]
    # PP-OCRv6 language codes that pass through directly
    known_codes = {"ch", "en", "fr", "de", "japan", "korean",
                   "ru", "it", "es", "pt", "ar", "hi", "th",
                   "el", "te", "ta", "chinese_cht",
                   "multilang_v6", "multilang_v5"}
    if lang in known_codes:
        return lang
    return lang


def init_ocr(args):
    global _ocr, _det, _use_gpu, _shrink_ratio, _blank_page_strategy, _current_lang
    
    _setup_nvidia_dlls()
    from paddleocr import PaddleOCR

    config = parse_config(args)
    model_size = config["model_size"]
    # \u4f18\u5148\u7528 CLI --language \u53c2\u6570\uff0c\u5176\u6b21\u7528 --lang\uff0c\u6700\u540e\u7528\u914d\u7f6e\u6587\u4ef6\u7684 lang
    if args.language and args.language != "ch":
        raw_lang = args.language
    elif args.lang and args.lang != "ch":
        raw_lang = args.lang
    else:
        raw_lang = config.get("lang", "ch")
    lang = _resolve_lang(raw_lang)
    _current_lang = lang
    det = str(args.det).lower() == "true" if args.det is not None else True
    cls = str(args.cls).lower() == "true" if args.cls is not None else False
    rec_batch_num = int(args.rec_batch_num) if args.rec_batch_num and args.rec_batch_num.strip() else 6
    limit_side_len = int(args.limit_side_len) if args.limit_side_len and args.limit_side_len.strip() else 960
    use_gpu = str(args.use_gpu).lower() == "true"
    _use_gpu = use_gpu
    _shrink_ratio = max(0.0, float(getattr(args, "shrink_poly_ratio", 0.0) or 0.0))
    _blank_page_strategy = getattr(args, "blank_page_strategy", "skip") or "skip"
    _det = det

    # CLI参数均为字符串，需要转换为正确类型
    cpu_threads_val = None
    if args.cpu_threads and args.cpu_threads.strip():
        try:
            cpu_threads_val = int(args.cpu_threads)
        except ValueError:
            pass
    engine, engine_config = _select_engine(use_gpu, cpu_threads_val, model_size)
    engine_kwargs = {}
    if engine:
        engine_kwargs["engine"] = engine
        engine_kwargs["engine_config"] = engine_config

    # ─── 判断用 PP-OCRv6 还是 v5 ─────────────────────────
    if lang == "multilang_v6":
        # 多语言 (v6)：PP-OCRv6 多语言模式，覆盖中/英/法/德/日/韩/俄
        lang = "ch&en&fr&german&japan&korean&russian"
    elif lang == "multilang_v5":
        # 多语言 (v5)：PP-OCRv5 拉丁模型，覆盖 46 种拉丁语系语言
        lang = "latin"
    
    # ═══════════════════════════════════════════════════════
    # 模型初始化：所有语言统一使用 model_name 方式
    # 绝不传 model_dir，避免 PaddleX 读取 inference.yml 中的 Hpi 配置
    # 导致 use_hpip=True → 引擎自动检测为 HPI → 实际使用 CPU
    # ═══════════════════════════════════════════════════════
    gpu_info = "GPU" if use_gpu else "CPU"
    
    if lang in _V6_LANGS or "&" in lang:
        det_model = f"PP-OCRv6_{model_size}_det"
        rec_model = f"PP-OCRv6_{model_size}_rec"
        ocr_args = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": cls,
            "lang": lang,
            "text_det_limit_side_len": limit_side_len,
            "text_recognition_batch_size": rec_batch_num,
            "text_detection_model_name": det_model,
            "text_recognition_model_name": rec_model,
        }
        ocr_args.update(engine_kwargs)
        print(f"[Init] PP-OCRv6 ({lang}, {model_size}, {gpu_info})", flush=True)
        _ocr = PaddleOCR(**ocr_args)
    else:
        # ─── PP-OCRv5 多语言 ──────────────────────────────
        from paddleocr._utils.langs import DEVANAGARI_LANGS, ARABIC_LANGS, CYRILLIC_LANGS, ESLAV_LANGS
        _REC_LANG_MAP = [
            ("latin", lambda l: l == "latin"),
            ("eslav", lambda l: l in ESLAV_LANGS),
            ("arabic", lambda l: l in ARABIC_LANGS),
            ("cyrillic", lambda l: l in CYRILLIC_LANGS),
            ("devanagari", lambda l: l in DEVANAGARI_LANGS),
            ("korean", lambda l: l == "korean"),
            ("th", lambda l: l == "th"),
            ("el", lambda l: l == "el"),
            ("te", lambda l: l == "te"),
            ("ta", lambda l: l == "ta"),
        ]
        v5_rec_name = None
        for rec_lang, matcher in _REC_LANG_MAP:
            if matcher(lang):
                v5_rec_name = f"{rec_lang}_PP-OCRv5_mobile_rec"
                break
        ocr_args = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": cls,
            "lang": lang,
            "ocr_version": "PP-OCRv5",
            "text_det_limit_side_len": limit_side_len,
            "text_recognition_batch_size": rec_batch_num,
            "text_detection_model_name": "PP-OCRv5_server_det",
        }
        if v5_rec_name:
            ocr_args["text_recognition_model_name"] = v5_rec_name
        ocr_args.update(engine_kwargs)
        print(f"[Init] PP-OCRv5 ({lang}, {gpu_info})", flush=True)
        _ocr = PaddleOCR(**ocr_args)
    
    print(f"[Init] Ready (lang={lang}, det={det}, {gpu_info})", flush=True)
    print("OCR init completed.", flush=True)


def _shrink_poly(poly, ratio):
    """将 4 点检测框向重心收缩"""
    if ratio <= 0 or not poly or len(poly) < 4:
        return poly
    cx = sum(p[0] for p in poly) / 4
    cy = sum(p[1] for p in poly) / 4
    shrunk = []
    for x, y in poly:
        sx = x + (cx - x) * ratio
        sy = y + (cy - y) * ratio
        shrunk.append([sx, sy])
    return shrunk


def process_image(image_data: bytes) -> dict:
    if _ocr is None:
        return {"code": 901, "data": "引擎未初始化"}
    tmp_path = None
    try:
        import tempfile
        # 保存到临时文件调用 predict(file_path) 而非 predict(numpy)
        # 因为 PaddleX 对 numpy 数组的 CJK 字典映射有 bug
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(image_data)
        tmp.close()
        tmp_path = tmp.name
        results = list(_ocr.predict(tmp_path))
        data = []
        if results and len(results) > 0:
            r0 = results[0]
            texts = r0.rec_texts if hasattr(r0, 'rec_texts') else r0.get('rec_texts', [])
            scores = r0.rec_scores if hasattr(r0, 'rec_scores') else r0.get('rec_scores', [])
            boxes = r0.rec_polys if hasattr(r0, 'rec_polys') else r0.get('rec_polys', r0.get('dt_polys', []))
            for text, confidence, bbox in zip(texts, scores, boxes):
                if confidence < 0.3:
                    continue
                poly = [[int(bbox[0][0]), int(bbox[0][1])],
                        [int(bbox[1][0]), int(bbox[1][1])],
                        [int(bbox[2][0]), int(bbox[2][1])],
                        [int(bbox[3][0]), int(bbox[3][1])]]
                if _shrink_ratio > 0:
                    poly = _shrink_poly(poly, _shrink_ratio)
                data.append({
                    "box": poly,
                    "text": text,
                    "score": round(float(confidence), 4),
                })
        if not data and _blank_page_strategy == 'error':
            return {"code": 101, "data": "未识别到文字"}
        return {"code": 100, "data": data}
    except Exception as e:
        return {"code": 900, "data": f"OCR error: {str(e)}"}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="PP-OCR Multilingual ONNX Server")
    parser.add_argument("--use_gpu", type=str, default="False")
    parser.add_argument("--language", type=str, default="ch",
                        help="语言代码：ch/en/fr/de/japan/korean/ar/hi/th/ru 等")
    parser.add_argument("--config_path", type=str, default="",
                        help="配置文件路径（可选，会覆盖 --language）")
    for p in ["limit_side_len", "cls", "det", "rec_batch_num",
             "shrink_poly_ratio", "blank_page_strategy",
             "model_size", "lang", "cpu_threads"]:
        parser.add_argument(f"--{p}", type=str, default="")
    args = parser.parse_args()

    # 优先用 --language，其次是 --lang，最后是 config 文件
    if not args.language and not args.config_path:
        args.language = "ch"

    init_ocr(args)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        image_base64 = req.get("image_base64")
        if image_base64:
            try:
                image_bytes = base64.b64decode(image_base64)
            except Exception:
                result = {"code": 904, "data": "Base64 解码失败"}
                sys.stdout.write(json.dumps(result, ensure_ascii=True) + "\n")
                sys.stdout.flush()
                continue
            result = process_image(image_bytes)
        else:
            result = {"code": 903, "data": "缺少 image_base64 参数"}

        sys.stdout.write(json.dumps(result, ensure_ascii=True) + "\n")
        sys.stdout.flush()
        _cleanup_gpu_memory()


if __name__ == "__main__":
    main()
