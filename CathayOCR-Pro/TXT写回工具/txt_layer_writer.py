# -*- coding: utf-8 -*-
"""
TXT文本层写回工具 - 核心处理模块
把 _result.txt 的 OCR 识别结果按页写回 _layered.pdf，替换原有错误文本层。

用法（作为模块被 GUI 调用，也可命令行单测）：
    from txt_layer_writer import process_pair
    process_pair(txt_path, pdf_path, out_suffix="_fixed")

竖排排布：每页从右上角起，每行文本一列，列从右到左，透明文字。

旧文字层清除方式：删除内容流中全部 BT..ET 文字对象（含 redaction 残留的
TD 块），保留图像与矢量图形。比 apply_redactions 更彻底（redaction 会把
被删文字重新编码为 TD 残留在内容流中，get_text 仍可能提取）。
"""
import fitz
import re
import os


# ---------------------------------------------------------------
# 1. 解析 _result.txt：按"第 N 页"分页，返回 {page_num: [lines]}
# ---------------------------------------------------------------
def parse_result_txt(txt_path):
    """解析 OCR 结果 TXT，返回 {页码(int): [文本行列表]}。过滤分隔线和空行。"""
    pages = {}
    cur = None
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"第\s*(\d+)\s*页", line.strip())
            if m:
                cur = int(m.group(1))
                pages.setdefault(cur, [])
                continue
            if cur is None:
                continue
            s = line.strip()
            if not s:
                continue
            # 过滤页眉分隔线（=====）和仅等号/横线行
            if re.match(r"^[=\-*]{5,}$", s):
                continue
            pages.setdefault(cur, []).append(s)
    return pages


# ---------------------------------------------------------------
# 2. 清除页面旧文字层：删除内容流中所有 BT..ET 文字块，保留图像/矢量
# ---------------------------------------------------------------
def strip_text_layer(doc, page):
    """删除 page 内容流中全部 BT..ET 文字对象（含 TD 残留），保留图像与矢量图形。
    返回删除的文字块数。"""
    removed = 0
    for x in page.get_contents():
        stream = doc.xref_stream(x)
        new, n = re.subn(rb"BT.*?ET", b"", stream, flags=re.S)
        if n:
            removed += n
            doc.update_stream(x, new)
    page.clean_contents()
    return removed


# ---------------------------------------------------------------
# 3. 单页竖排写入：从右上角起，每行一列，列右→左
# ---------------------------------------------------------------
def write_page_vertical(page, lines, font):
    """把 lines 竖排写入 page。从右上角起，每行文本一列，列从右到左。
    返回 (列数, 字号)。文字透明（fill_opacity=0）。"""
    pr = page.rect
    W, H = pr.width, pr.height
    margin = 40.0          # 页边距
    n = len(lines)
    if n == 0:
        return 0, 0
    # 列宽：可用宽度均分
    avail_w = W - 2 * margin
    col_w = avail_w / n
    # 字号：不超过列宽和最大字号，且保证最长行能放下
    max_len = max(len(s) for s in lines)
    avail_h = H - 2 * margin
    fontsize = min(col_w, 28.0)
    # 若最长行放不下（纵向），缩字号
    if max_len * fontsize > avail_h:
        fontsize = avail_h / max_len
    # 列起点：最右列中心在 W-margin-col_w/2，逐列左移 col_w
    page.insert_font(fontname="cjk", fontbuffer=font.buffer)
    for i, text in enumerate(lines):
        if not text:
            continue
        # 列中心 x：从右到左
        x = W - margin - col_w * (i + 1) + col_w / 2
        # rotate=270：文字从 (x, y0) 向下排；y0 用页顶
        y0 = margin
        point = fitz.Point(x, y0) * page.derotation_matrix
        page.insert_text(
            point, text, fontsize=fontsize, fontname="cjk",
            rotate=270, stroke_opacity=0, fill_opacity=0,
        )
    return n, fontsize


# ---------------------------------------------------------------
# 4. 处理一对文件：TXT 写回 PDF
# ---------------------------------------------------------------
def process_pair(txt_path, pdf_path, out_suffix="_fixed", progress_callback=None):
    """把 txt_path 的内容按页写回 pdf_path，输出 pdf_path 替换后缀为 out_suffix。
    返回 (输出路径, 处理的页数)。"""
    pages = parse_result_txt(txt_path)
    if not pages:
        raise ValueError(f"TXT 无有效页内容: {txt_path}")

    font = fitz.Font("cjk")
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    processed = 0
    for pno in range(1, total_pages + 1):
        if pno not in pages:
            if progress_callback:
                progress_callback(pno, total_pages, f"第{pno}页无TXT内容，跳过")
            continue
        page = doc[pno - 1]
        # 清除旧文字层（内容流级，保留图像/矢量）
        removed = strip_text_layer(doc, page)
        # 竖排写入 TXT 文本
        lines = pages[pno]
        n, fs = write_page_vertical(page, lines, font)
        processed += 1
        if progress_callback:
            progress_callback(pno, total_pages,
                              f"第{pno}页清理{removed}块 写入{n}行 (字号{fs:.0f})")

    # 输出路径：同目录，替换 _layered.pdf 或 .pdf 后缀
    dirn = os.path.dirname(pdf_path)
    base = os.path.basename(pdf_path)
    stem = base[:-4] if base.lower().endswith(".pdf") else base
    out_path = os.path.join(dirn, stem + out_suffix + ".pdf")
    doc.save(out_path, deflate=True, garbage=3)
    doc.close()
    return out_path, processed


# ---------------------------------------------------------------
# 命令行单测
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        txt, pdf = sys.argv[1], sys.argv[2]
        suffix = sys.argv[3] if len(sys.argv) > 3 else "_fixed"
        out, n = process_pair(txt, pdf, suffix)
        print(f"完成: {out} (处理 {n} 页)")
    else:
        print("用法: python txt_layer_writer.py <result.txt> <layered.pdf> [后缀]")
