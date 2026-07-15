"""rec_batch_num performance test for PPOCR V6 engine.

Tests different rec_batch_num values to find optimal Rec-stage throughput.
Uses a dense ancient-text PDF (many text boxes per page).

Run with: ppocr_v6_env\Scripts\python.exe _rec_batch_test.py
"""
import os
import sys
import time
import fitz  # PyMuPDF

# Add plugin dir to path so we can import ppocr_pipe
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PLUGIN_DIR, "..", "umi_plugin_v6"))
from ppocr_pipe import PPOCR_pipe

# Test PDF: dense ancient Chinese text (many boxes per page)
TEST_PDF = r"D:\OCR待AI处理\第二辑  清代扬州学记  顾亭林学记_11665482.pdf"
EXE_PATH = os.path.join(PLUGIN_DIR, "..", "umi_plugin_v6", "PaddleOCR-json.bat")

# Test config
N_PAGES = 30          # Pages to OCR per rec_batch_num value
SIDE_LEN = 2560       # Match production config
REC_BATCH_VALUES = [6, 16, 24, 32, 48, 64]  # 6 = default baseline

GRAY = fitz.csGRAY


def render_pages(pdf_path, n_pages, side_len):
    """Render first n_pages to grayscale bytes at given side length."""
    doc = fitz.open(pdf_path)
    total = min(n_pages, doc.page_count)
    pages = []
    for i in range(total):
        page = doc[i]
        # Scale to fit side_len on the longer edge
        rect = page.rect
        scale = min(side_len / rect.width, side_len / rect.height, 4.0)
        scale = max(scale, 0.1)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, colorspace=GRAY)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages


def run_test(rec_batch_num, pages):
    """Launch engine with given rec_batch_num, OCR all pages, return (secs, n_boxes)."""
    args = {
        "use_gpu": True,
        "gpu_mem_ratio": 0.45,
        "rec_batch_num": rec_batch_num,
        "limit_side_len": SIDE_LEN,
        "det": True,
        "cls": False,
        "blank_page_strategy": "skip",
    }
    pipe = PPOCR_pipe(EXE_PATH, args)
    t0 = time.time()
    total_boxes = 0
    for img_bytes in pages:
        import base64
        b64 = base64.b64encode(img_bytes).decode("ascii")
        res = pipe.runBase64(b64, timeout_seconds=120)
        if res.get("code") == 100:
            data = res.get("data", [])
            if isinstance(data, list):
                total_boxes += len(data)
    elapsed = time.time() - t0
    pipe.exit()
    return elapsed, total_boxes


def main():
    print(f"=== rec_batch_num Performance Test ===")
    print(f"Test PDF: {TEST_PDF}")
    print(f"Pages per test: {N_PAGES}, Side len: {SIDE_LEN}")
    print(f"rec_batch_num values: {REC_BATCH_VALUES}\n")

    print(f"[{time.strftime('%H:%M:%S')}] Rendering {N_PAGES} pages...")
    pages = render_pages(TEST_PDF, N_PAGES, SIDE_LEN)
    print(f"[{time.strftime('%H:%M:%S')}] Rendered {len(pages)} pages. Starting tests...\n")

    results = []
    for rb in REC_BATCH_VALUES:
        print(f"[{time.strftime('%H:%M:%S')}] Testing rec_batch_num={rb}...")
        t0 = time.time()
        elapsed, n_boxes = run_test(rb, pages)
        pps = len(pages) / elapsed if elapsed > 0 else 0
        results.append((rb, elapsed, n_boxes, pps))
        print(f"  -> {elapsed:.1f}s, {n_boxes} boxes, {pps:.2f} pages/sec\n")

    print("=== RESULTS ===")
    print(f"{'rec_batch_num':<15}{'time(s)':<12}{'boxes':<12}{'pages/sec':<12}")
    print("-" * 51)
    for rb, elapsed, n_boxes, pps in results:
        print(f"{rb:<15}{elapsed:<12.1f}{n_boxes:<12}{pps:<12.2f}")

    # Find best
    best = max(results, key=lambda x: x[3])
    print(f"\nBest: rec_batch_num={best[0]} at {best[3]:.2f} pages/sec")
    print(f"Default (6): {[r for r in results if r[0]==6][0][3]:.2f} pages/sec")
    improvement = (best[3] / [r for r in results if r[0]==6][0][3] - 1) * 100
    print(f"Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
