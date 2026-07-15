import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppocr_v6_server as srv
# 必须在 import paddleocr 之前注入 NVIDIA DLL 路径
srv._setup_nvidia_dlls()

from PIL import Image, ImageDraw, ImageFont

def make_test_image(path, lines=40):
    W, H = 1240, 1754
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("msyh.ttc", 28)
    except Exception:
        f = ImageFont.load_default()
    y = 60
    for i in range(lines):
        d.text((60, y), f"第 {i+1} 行 测试文本 Test Line ABCDEFG 1234567890 中文识别速度基准", fill="black", font=f)
        y += 38
    img.save(path)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--algo", default="HEURISTIC")
    ap.add_argument("--ws", default="0")
    ap.add_argument("--mem", type=float, default=0.80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--trt", action="store_true", help="在 CUDA 之前插入 TensorrtExecutionProvider (fp32)")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    args.config_path = f"models/config_{args.model}.txt"
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    img = make_test_image(os.path.join(plugin_dir, "_bench_page.png"))

    config = srv.parse_config(args)
    model_size = config["model_size"]
    lang = config["lang"]
    rec_batch_num = args.batch
    limit_side_len = 999999
    det_model = f"PP-OCRv6_{model_size}_det"
    rec_model = f"PP-OCRv6_{model_size}_rec"

    engine, engine_config = srv._select_engine(True, None, model_size)
    # 覆盖 cudnn 卷积算法搜索策略 / 工作区 / 显存上限
    engine_config["provider_options"][0]["cudnn_conv_algo_search"] = args.algo
    engine_config["provider_options"][0]["cudnn_conv_use_max_workspace"] = args.ws
    gpu_total = srv._get_gpu_total_memory_gb()
    engine_config["provider_options"][0]["gpu_mem_limit"] = int(gpu_total * args.mem * 1024**3)
    if args.trt:
        cache_dir = os.path.join(plugin_dir, "trt_cache")
        os.makedirs(cache_dir, exist_ok=True)
        trt_opts = {
            "device_id": 0,
            "trt_fp16_enable": "0",            # fp32 保精度
            "trt_engine_cache_enable": "1",
            "trt_engine_cache_path": cache_dir,
            "trt_max_workspace_size": 1 << 30,
        }
        engine_config["providers"] = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        engine_config["provider_options"] = [trt_opts, engine_config["provider_options"][0], {}]
        print(f"[TRT] providers={engine_config['providers']}")
    print(f"[model={args.model} algo={args.algo}] provider_options[0]={engine_config['provider_options'][0]}")

    from paddleocr import PaddleOCR
    ocr_args = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "lang": lang,
        "text_det_limit_side_len": limit_side_len,
        "text_recognition_batch_size": args.batch,
        "text_detection_model_name": det_model,
        "text_recognition_model_name": rec_model,
        "engine": engine,
        "engine_config": engine_config,
    }
    t0 = time.time()
    try:
        ocr = PaddleOCR(**ocr_args)
    except Exception as e:
        print(f"[model={args.model} algo={args.algo}] PaddleOCR 初始化失败: {e}")
        return
    print(f"[model={args.model} algo={args.algo}] init 耗时: {time.time()-t0:.2f}s")

    _ = ocr.ocr(img)  # warmup
    times = []
    for i in range(args.runs):
        t1 = time.time()
        res = ocr.ocr(img)
        dt = time.time() - t1
        times.append(dt)
        n = len(res[0]) if res and res[0] else 0
        print(f"  run {i+1}: {dt:.2f}s, 检测框数={n}")
    print(f"[model={args.model} algo={args.algo}] 平均单页(去warmup): {sum(times)/len(times):.2f}s")

if __name__ == "__main__":
    main()
