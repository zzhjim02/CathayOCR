import sys, os, time, argparse
PLUGIN_DIR = r"D:\Program Files\Umi-OCR_Paddle_v2.1.4\UmiOCR-data\plugins\umi_plugin_v6"
sys.path.insert(0, PLUGIN_DIR)
import ppocr_v6_server as srv
import fitz

# 目标 PDF 路径来自 find_pdf 生成的文件（第一行即目标文档），脚本内不含中文
TP = r"C:\Users\zzhjim\.qclaw\workspace\_target_path.txt"
with open(TP, "r", encoding="utf-8") as fh:
    lines = [l.strip() for l in fh if l.strip()]
pdf = lines[0]
print("PDF =", pdf)

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, required=True)
ap.add_argument("--page", type=int, default=20)
ap.add_argument("--dpi", type=int, default=300)
args = ap.parse_args()

doc = fitz.open(pdf)
print("页数:", doc.page_count)
page = doc[args.page]
pix = page.get_pixmap(dpi=args.dpi)
png = os.path.join(PLUGIN_DIR, f"_acc_p{args.page}.png")
pix.save(png)
print(f"[limit={args.limit}] 源页渲染尺寸: {pix.width}x{pix.height} @ {args.dpi}dpi  (长边将被缩到 {args.limit})")

class A: pass
a = A()
a.config_path = "models/config_medium.txt"
a.det = True
a.cls = False
a.rec_batch_num = 24
a.limit_side_len = args.limit
a.use_gpu = True
a.cpu_threads = None
a.shrink_poly_ratio = 0.0

try:
    t0 = time.time()
    srv.init_ocr(a)
    t_init = time.time() - t0
    t1 = time.time()
    res = srv._ocr.ocr(png)
    dt = time.time() - t1
    boxes = res[0] if res and res[0] else []
    n = len(boxes)
    texts = [b[1][0] for b in boxes if b[1][0]]
    total_chars = sum(len(t) for t in texts)
    print(f"[limit={args.limit}] init={t_init:.2f}s run={dt:.2f}s 检测框数={n} 识别字数≈{total_chars}")
    print("   样本:", " | ".join(texts[:10])[:240])
except Exception as e:
    print(f"[limit={args.limit}] 错误/OOM: {type(e).__name__}: {str(e)[:200]}")
