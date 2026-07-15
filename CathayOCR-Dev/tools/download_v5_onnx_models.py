#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download all PP-OCRv5 ONNX recognition models"""
import sys, os, time, json, urllib.request, urllib.parse
import shutil

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

CACHE = r'C:\Users\zzhjim\.paddlex\official_models'

# PP-OCRv5 ONNX recognition models
MODELS = [
    'arabic_PP-OCRv5_mobile_rec_onnx',
    'devanagari_PP-OCRv5_mobile_rec_onnx', 
    'cyrillic_PP-OCRv5_mobile_rec_onnx',
    'korean_PP-OCRv5_mobile_rec_onnx',
    'th_PP-OCRv5_mobile_rec_onnx',
    'el_PP-OCRv5_mobile_rec_onnx',
    'te_PP-OCRv5_mobile_rec_onnx',
    'ta_PP-OCRv5_mobile_rec_onnx',
]

# Detection + classification models (language-independent, already cached)
DET_MODELS = [
    'PP-OCRv6_medium_det_onnx',
    'PP-LCNet_x0_25_textline_ori_infer',
]


def download_via_paddlex(model_name):
    """Use PaddleX internal download mechanism"""
    from paddlex.inference.models import resolve_model_name, official_models
    
    # First resolve the model name
    resolved_name = resolve_model_name(model_name)
    print(f'  Resolved: {resolved_name}')
    
    # Check official_models
    if hasattr(official_models, model_name):
        print(f'  Found in official_models')
    
    # Try direct download via paddlex hub
    try:
        from paddlex.hub import ModelHub
        hub = ModelHub()
        result = hub.download(model_name, save_dir=CACHE)
        return result
    except Exception as e:
        print(f'  Hub download failed: {e}')
    
    # Try via PaddleOCR model download
    try:
        from paddleocr._utils.model_download import download_model_with_progress
        result = download_model_with_progress(model_name, CACHE)
        return result
    except Exception as e:
        print(f'  PaddleOCR download failed: {e}')
    
    return None


def download_via_paddlex2(model_name):
    """PaddleX 3.x model download"""
    try:
        # The PaddleX download handler processes model configs
        from paddlex.inference.models import _resolve_model_dir, _resolve_local_model_dir
        # If we just resolve the model, it will download
        resolved = _resolve_model_dir(model_name)
        if resolved:
            print(f'  Resolved automatically: {resolved}')
            return resolved
    except Exception as e:
        print(f'  Auto-resolve failed: {e}')
    
    # Try using the PaddleX official model download endpoint directly
    try:
        from paddlex.utils import download as pd_download
        result = pd_download.download(model_name, CACHE)
        return result
    except Exception as e:
        print(f'  PaddleX download failed: {e}')
    
    return None


def download_direct(model_name):
    """Try direct download from model hub URL"""
    # The PaddleOCR model hub URLs follow a pattern
    # https://paddleocr.bj.bcebos.com/...
    # or https://paddle-model-ecology.bj.bcebos.com/...
    base_url = 'https://paddle-model-ecology.bj.bcebos.com'
    
    # Try different URL patterns
    urls = [
        f'{base_url}/paddlex/official_models/{model_name}.tar.gz',
        f'{base_url}/model/{model_name}.tar.gz',
    ]
    
    target_dir = os.path.join(CACHE, model_name)
    os.makedirs(target_dir, exist_ok=True)
    
    for url in urls:
        print(f'  Trying: {url}')
        try:
            req = urllib.request.Request(url, method='HEAD')
            resp = urllib.request.urlopen(req, timeout=10)
            if resp.status == 200:
                print(f'  Found! Downloading...')
                # Download with progress
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=300) as resp:
                    total = int(resp.headers.get('content-length', 0))
                    downloaded = 0
                    archive_path = os.path.join(target_dir, 'model.tar.gz')
                    with open(archive_path, 'wb') as f:
                        while True:
                            chunk = resp.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                pct = downloaded * 100 // total
                                print(f'\r    {pct}% ({downloaded//1024}/{total//1024} KB)', end='', flush=True)
                    print()
                    
                    # Extract
                    import tarfile
                    with tarfile.open(archive_path, 'r:gz') as tar:
                        tar.extractall(target_dir)
                    os.remove(archive_path)
                    
                    # Check if model.onnx exists
                    for root, _, files in os.walk(target_dir):
                        for f in files:
                            if f.endswith('.onnx'):
                                print(f'  Extracted: {os.path.join(root, f)}')
                                return target_dir
        except Exception as e:
            print(f'  Failed: {e}')
    
    return None


# Main download
print('=== 下载 PP-OCRv5 ONNX 识别模型 ===')
print()

for m in MODELS:
    target = os.path.join(CACHE, m)
    if os.path.isdir(target) and any(f.endswith('.onnx') for f in os.listdir(target)):
        sz = sum(os.path.getsize(os.path.join(target, f)) for f in os.listdir(target) 
                 if os.path.isfile(os.path.join(target, f)))
        print(f'  [CACHED] {m} ({sz//1024} KB)')
        continue
    
    print(f'  [DL] {m}...', flush=True)
    t0 = time.time()
    
    result = download_via_paddlex2(m)
    if not result:
        result = download_direct(m)
    
    if result:
        elapsed = time.time() - t0
        print(f'  [OK] {elapsed:.0f}s', flush=True)
    else:
        print(f'  [FAIL] Cannot download {m}', flush=True)

print()
print('=== 完成 ===')
