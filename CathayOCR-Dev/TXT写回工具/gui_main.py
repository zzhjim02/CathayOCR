# -*- coding: utf-8 -*-
"""
TXT文本层写回工具 - GUI 主程序
批量把 _result.txt 的 OCR 识别结果按页写回 _layered.pdf，替换错误文本层。
竖排古籍风格，透明文字层，另存 _fixed.pdf 保留原件。

运行：portapython\python.exe gui_main.py
"""
import os
import re
import sys
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk, filedialog, messagebox

# 确保能导入同目录的 txt_layer_writer 模块（双击运行时当前目录可能不是脚本目录）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from txt_layer_writer import process_pair


class App:
    def __init__(self, root):
        self.root = root
        root.title("TXT文本层写回工具 v1.0")
        root.geometry("760x560")

        # 目录选择
        frm_dir = ttk.Frame(root, padding=8)
        frm_dir.pack(fill="x")
        ttk.Label(frm_dir, text="目录:").pack(side="left")
        self.dir_var = tk.StringVar()
        ent = ttk.Entry(frm_dir, textvariable=self.dir_var, width=60)
        ent.pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(frm_dir, text="选择目录", command=self.pick_dir).pack(side="left", padx=4)
        ttk.Button(frm_dir, text="扫描配对", command=self.scan).pack(side="left")

        # 选项
        frm_opt = ttk.Frame(root, padding=8)
        frm_opt.pack(fill="x")
        self.opt_suffix = tk.StringVar(value="_fixed")
        ttk.Label(frm_opt, text="输出后缀:").pack(side="left")
        ttk.Entry(frm_opt, textvariable=self.opt_suffix, width=10).pack(side="left", padx=4)
        # 并行数（多核加速）
        ttk.Label(frm_opt, text="并行数:").pack(side="left", padx=(12, 2))
        self.workers_var = tk.IntVar(value=min(4, (os.cpu_count() or 4)))
        ttk.Spinbox(frm_opt, from_=1, to=16, textvariable=self.workers_var, width=4).pack(side="left")
        ttk.Button(frm_opt, text="全选", command=lambda: self.set_all(True)).pack(side="left", padx=4)
        ttk.Button(frm_opt, text="全不选", command=lambda: self.set_all(False)).pack(side="left")

        # 文件列表
        frm_list = ttk.Frame(root, padding=8)
        frm_list.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(frm_list, columns=("pdf", "status"), show="tree", height=18)
        self.tree.heading("#0", text="TXT 文件")
        self.tree.heading("pdf", text="PDF 文件")
        self.tree.heading("status", text="状态")
        self.tree.column("pdf", width=320, anchor="w")
        self.tree.column("status", width=160, anchor="center")
        vsb = ttk.Scrollbar(frm_list, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # 进度与日志
        frm_prog = ttk.Frame(root, padding=8)
        frm_prog.pack(fill="x")
        self.prog = ttk.Progressbar(frm_prog, mode="determinate")
        self.prog.pack(fill="x")
        self.log_var = tk.StringVar(value="就绪")
        ttk.Label(frm_prog, textvariable=self.log_var).pack(anchor="w", pady=2)

        frm_btn = ttk.Frame(root, padding=8)
        frm_btn.pack(fill="x")
        self.btn_run = ttk.Button(frm_btn, text="开始处理选中项", command=self.start)
        self.btn_run.pack(side="left")
        self.btn_open = ttk.Button(frm_btn, text="打开输出目录", command=self.open_outdir, state="disabled")
        self.btn_open.pack(side="left", padx=4)

        self.pairs = []      # [(txt, pdf, basename)]
        self.running = False

    # ---------- 目录与扫描 ----------
    def pick_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.dir_var.set(d)
            self.scan()

    def scan(self):
        d = self.dir_var.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showerror("错误", "请选择有效目录")
            return
        self.tree.delete(*self.tree.get_children())
        self.pairs = []
        # 明确的后缀配对规则（TXT后缀 -> PDF后缀）
        #   规则A: XXX_result.txt      <-> XXX_layered.pdf        （CathayOCR Lite 输出）
        #   规则B: XXX_PD6AIFOCR.txt   <-> XXX_PD6AIFOCR_opt.pdf  （FOCR 工具输出）
        rules = [
            ("_result.txt", "_layered.pdf"),
            ("_PD6AIFOCR.txt", "_PD6AIFOCR_opt.pdf"),
            ("_PD6AIFOCR.txt", "_PD6AIFOCR.pdf"),   # 无 _opt 变体（如北洋政府公报）
        ]
        # 递归扫描当前目录及所有子目录（os.walk），支持整棵目录树
        for root, dirs, fns in os.walk(d):
            files = set(fns)
            for txt_suf, pdf_suf in rules:
                for fn in sorted(files):
                    if not fn.endswith(txt_suf):
                        continue
                    stem = fn[: -len(txt_suf)]
                    pdf_fn = stem + pdf_suf
                    if pdf_fn in files:
                        self.pairs.append((os.path.join(root, fn),
                                           os.path.join(root, pdf_fn), stem))
                        # 列表显示相对路径（含子目录）：#0=TXT名，pdf列=PDF名
                        rel_txt = os.path.relpath(os.path.join(root, fn), d)
                        rel_pdf = os.path.relpath(os.path.join(root, pdf_fn), d)
                        iid = f"p{len(self.pairs)-1}"
                        self.tree.insert("", "end", iid=iid, text=rel_txt,
                                         values=(rel_pdf, "待处理"), open=False)
        self.log_var.set(f"扫描到 {len(self.pairs)} 对文件（含子目录）")
        if not self.pairs:
            messagebox.showinfo("提示", "未找到可配对的 TXT 与 PDF 文件")

    def set_all(self, on):
        for iid in self.tree.get_children():
            self.tree.item(iid, open=on)

    # ---------- 处理 ----------
    def start(self):
        if self.running:
            return
        sel = list(self.tree.get_children())
        if not sel:
            messagebox.showinfo("提示", "没有选中文件")
            return
        self.running = True
        self.btn_run.config(state="disabled")
        self.btn_open.config(state="disabled")
        self.prog.config(maximum=len(sel), value=0)
        self.outdir = self.dir_var.get().strip()
        # 主线程预取任务参数（避免 worker 线程读 UI）
        tasks = []
        for iid in sel:
            try:
                pair_idx = int(iid[1:])
            except (ValueError, IndexError):
                pair_idx = self.tree.index(iid)
            txt, pdf, stem = self.pairs[pair_idx]
            pdf_disp = self.tree.item(iid, "values")[0]  # PDF列显示值（主线程读）
            tasks.append((iid, txt, pdf, stem, pdf_disp))
        n = max(1, self.workers_var.get())
        th = threading.Thread(target=self._worker, args=(tasks, n), daemon=True)
        th.start()

    def _worker(self, tasks, n_workers):
        suffix = self.opt_suffix.get().strip() or "_fixed"
        total = len(tasks)
        # 线程安全计数器
        lock = threading.Lock()
        state = {"ok": 0, "fail": 0, "done": 0}

        def run_one(item):
            iid, txt, pdf, stem, pdf_disp = item
            self.root.after(0, lambda i=iid, t=f"处理中: {os.path.basename(txt)}", v=pdf_disp: self.tree.item(
                i, values=(v, t)))
            try:
                out, n = process_pair(txt, pdf, suffix)
                with lock:
                    state["ok"] += 1
                self.root.after(0, lambda i=iid, o=out, v=pdf_disp: self.tree.item(
                    i, values=(v, f"完成 → {os.path.basename(o)}")))
            except Exception as e:
                with lock:
                    state["fail"] += 1
                self.root.after(0, lambda i=iid, e=str(e), v=pdf_disp: self.tree.item(
                    i, values=(v, f"失败: {e[:50]}")))
            finally:
                with lock:
                    state["done"] += 1
                    d = state["done"]
                self.root.after(0, lambda v=d: self.prog.config(value=v))
                # 闭包捕获 d（局部变量，每个任务独立），避免默认参数提前求值问题
                self.root.after(0, lambda dd=d, tt=total: self.log_var.set(f"[{dd}/{tt}]"))

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for _ in ex.map(run_one, tasks):
                pass
        ok, fail = state["ok"], state["fail"]
        self.running = False
        self.root.after(0, lambda: self.btn_run.config(state="normal"))
        self.root.after(0, lambda: self.btn_open.config(state="normal"))
        self.root.after(0, lambda: self.log_var.set(f"完成: 成功 {ok}，失败 {fail}"))

    def open_outdir(self):
        d = self.outdir
        if d and os.path.isdir(d):
            os.startfile(d)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
