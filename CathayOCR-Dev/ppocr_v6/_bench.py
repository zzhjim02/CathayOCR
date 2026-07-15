import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppocr_v6_server as srv
from PIL import Image, ImageDraw, ImageFont

plugin_dir = os.path.dirname(os.path.abspath(__file__))

def make_test_image(path, lines=40, w=1240, h=1754):
    W, H = w, h
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
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--limit", type=int, default=999999)
    ap.add_argument("--wh", default=None, help="页面尺寸 WxH，如 2600x3600")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16, help="识别批大小 rec_batch_num")
    args = ap.parse_args()

    if args.wh:
        w, h = (int(x) for x in args.wh.lower().split("x"))
    else:
        w, h = 1240, 1754
    img = make_test_image(os.path.join(plugin_dir, "_bench_page.png"), w=w, h=h)

    class A: pass
    a = A()
    a.config_path = f"models/config_{args.model}.txt"
    a.det = True
    a.cls = False
    a.rec_batch_num = args.batch
    a.limit_side_len = args.limit
    a.use_gpu = args.use_gpu
    a.cpu_threads = None
    a.shrink_poly_ratio = 0.0

    t0 = time.time()
    srv.init_ocr(a)
    t_init = time.time() - t0
    print(f"[use_gpu={args.use_gpu} model={args.model}] init 耗时: {t_init:.2f}s")

    # warmup
    _ = srv._ocr.ocr(img)
    times = []
    for i in range(args.runs):
        t1 = time.time()
        res = srv._ocr.ocr(img)
        dt = time.time() - t1
        times.append(dt)
        n = len(res[0]) if res and res[0] else 0
        print(f"  run {i+1}: {dt:.2f}s, 检测框数={n}")
    avg = sum(times) / len(times)
    print(f"[use_gpu={args.use_gpu} model={args.model}] 平均单页(去warmup): {avg:.2f}s")

if __name__ == "__main__":
    main()
