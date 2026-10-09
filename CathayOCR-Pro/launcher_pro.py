# -*- coding: utf-8 -*-
"""
CathayOCR Pro — 标准程序启动器 + 运行日志窗口
================================================
作用
----
1. 双击即可启动 CathayOCR Pro。内部用软件自带的便携 Python 运行**原来的主程序源码**，
   运行环境与原来的 .bat 启动方式完全一致（不打包主程序，杜绝"换运行环境引入新问题"）。
2. 把原本显示在 DOS 黑框里的**全部输出**（进度 / 引擎日志 / 错误信息）搬进本窗口：
   错误红色高亮、可暂停滚动、可另存日志。
3. 日志窗口与任务**解耦**：
   * 点 X 关闭只是把窗口藏起来，任务继续在后台运行；
   * 任务运行中再次双击 exe，不会重复启动程序，而是把日志窗口重新调出来（单实例互斥）；
   * 默认就吸附在主程序右侧、自动缩成窄侧边栏并平滑跟随主窗口移动；
   * 主程序正常退出后，日志窗口停留数秒展示退出结论再自动关闭；异常退出则保持打开。
4. 与主程序「看起来是一个程序」：
   * 日志窗口带 WS_EX_TOOLWINDOW —— 任务栏**只有主程序一个图标**（不设 CATHAYOCR_LAUNCHER_TASKBAR=1 时）；
   * 主程序最小化/收进托盘时，日志窗口同步收起；还原时一起还原；
   * 配色与主程序一致（浅色/白色主题）。
5. 启动时先做一次**硬件自检**（CPU / 内存 / 显卡 / Vulkan / 引擎插件）并给出推荐引擎。

原则
----
* 不结束任何"不是自己启动的"进程；只管理自己那一个子进程。
* 如遇主程序异常退出，仅清理**本软件目录内**的残留引擎进程（绝不按进程名全局强杀）。
* 原 `启动.bat` / `CathayOCR Lite.bat` 完整保留，作为后备入口。
"""

import os
import sys
import time
import queue
import shutil
import hashlib
import threading
import subprocess

import tkinter as tk
from tkinter import filedialog, messagebox

APP_TITLE = "CathayOCR Pro — 运行日志"

# ── 配色（浅色 / 白色，与主程序的 Fusion 浅色主题一致）──
BG        = "#ffffff"
BG_BAR    = "#f4f6f8"
BG_BTN    = "#eaeef2"
BG_BTN_HV = "#dbe2ea"
FG        = "#1f2328"
FG_DIM    = "#8a939e"
C_ERR     = "#c62828"
C_WARN    = "#a86a00"
C_OK      = "#1a7f37"
C_INFO    = "#1565c0"
C_ACCENT  = "#c0392b"
BORDER    = "#e3e7ec"

ERR_KEYS  = ("traceback", "error", "错误", "失败", "exception", "failed", "超时",
             "timeout", "cannot", "无法", "崩溃", "crash", "not found", "找不到")
WARN_KEYS = ("warn", "警告", "skip", "跳过", "重试", "retry", "restart", "重启",
             "watchdog", "看门狗", "不规范", "缺失")
OK_KEYS   = ("成功", "完成", "finished", "done", "saved", "已保存", "ready")

# ── 布局常量 ──
GEOM_NORMAL = "980x620"
MIN_NORMAL  = (680, 400)
DOCK_WIDTH  = 330          # 吸附时的侧边栏宽度
MIN_DOCK    = (280, 260)
POLL_MS      = 200         # 空闲时的巡检周期（外部唤起 / 主窗口状态同步）
DOCK_POLL_MS = 20          # 吸附跟随时的高频巡检（=50fps，消除拖动卡顿）
DOCK_GRACE_S = 20.0        # 启动后多久还没吸附上主窗口 → 自动退回独立窗口
EXIT_STAY_MS = 5000        # 主程序退出后日志窗口停留时长（毫秒）

# ── 本版本（专业版）的引擎清单，供硬件自检判断组件是否齐全 ──
CLI_VERSION_NAME  = "CathayOCR Pro"
REQUIRED_ENGINE_DIRS = ("ncnn", "ppocr_v6", "portapython")
HAVE_CUDA_ENGINE  = True   # 专业版含 PP-OCRv6（可用 NVIDIA CUDA）

# Win32：扩展窗口样式（让日志窗口不出现在任务栏 / Alt+Tab）
GWL_EXSTYLE       = -20
WS_EX_TOOLWINDOW  = 0x00000080
WS_EX_APPWINDOW   = 0x00040000

# 两种布局下的按钮文字（顺序与创建顺序一致：滚动/保存/清空/结束/吸附）
BTN_LABELS_NORMAL = ("暂停滚动", "保存日志", "清空", "结束任务并退出", "⇥ 吸附侧边")
BTN_LABELS_DOCK   = ("滚动", "保存", "清空", "结束任务", "⇤ 独立")


# ============================================================
# Win32 小工具（ctypes，不引入额外依赖）
# ============================================================
def _get_error():
    import ctypes
    return ctypes.get_last_error()


def _acquire_instance_mutex():
    """单实例互斥。成功返回句柄；已有实例在跑时返回 None。

    互斥体名字按安装目录散列派生 —— Lite / Pro / 不同安装位置互不干扰。
    """
    if not sys.platform.startswith("win"):
        return object()                       # 非 Windows：不做单实例
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    name = "Local\\CathayOCR-Log-%s" % hashlib.md5(
        os.path.normcase(os.path.abspath(ROOT_DIR)).encode("utf-8")).hexdigest()[:12]
    k32.CreateMutexW.restype = ctypes.c_void_p
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    h = k32.CreateMutexW(None, 0, name)
    if not h or _get_error() == 183:          # ERROR_ALREADY_EXISTS
        return None
    return h


def _show_existing_log_window():
    """把已在运行的实例的日志窗口调出来（任务运行中被关掉后重开用）。"""
    try:
        u = _user32()
        hwnd = u.FindWindowW("TkTopLevel", APP_TITLE)      # tkinter 顶层窗口类
        if not hwnd:
            hwnd = u.FindWindowW(None, APP_TITLE)
        if not hwnd:
            return False
        u.ShowWindow(hwnd, 5)                              # SW_SHOW
        u.BringWindowToTop(hwnd)
        u.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


def _find_window_of_pid(pid, min_w=380, min_h=280):
    """找某 PID 的可见主窗口（按面积取最大者）。用于吸附时定位主程序窗口。"""
    if not sys.platform.startswith("win"):
        return 0
    import ctypes
    from ctypes import wintypes
    u = _user32()
    best, best_area = [0], [0]

    class RECT(ctypes.Structure):
        _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                    ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

    CB = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lp):
        wpid = wintypes.DWORD(0)
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
        if wpid.value != pid or not u.IsWindowVisible(hwnd):
            return True
        if u.GetWindowTextLengthW(hwnd) <= 0:
            return True
        r = RECT()
        if not u.GetWindowRect(hwnd, ctypes.byref(r)):
            return True
        w, h = r.right - r.left, r.bottom - r.top
        if w < min_w or h < min_h:
            return True
        area = w * h
        if area > best_area[0]:
            best[0], best_area[0] = hwnd, area
        return True

    cb = CB(_cb)
    u.EnumWindows(cb, 0)
    return best[0]


def _set_tool_window(hwnd, enable=True):
    """把 hwnd 设为「工具窗口」——不出现在任务栏，也不进 Alt+Tab。

    这样任务栏里只留主程序一个图标，两个窗口看起来就是同一个程序。
    必须在窗口首次显示（map）之前调用，改完不会闪一下。
    设 CATHAYOCR_LAUNCHER_TASKBAR=1 可保留日志窗口自己的任务栏图标。
    """
    if not sys.platform.startswith("win") or not hwnd:
        return False
    if enable and os.environ.get("CATHAYOCR_LAUNCHER_TASKBAR") == "1":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        u = _user32()
        u.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        u.GetWindowLongW.restype = wintypes.LONG
        u.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
        u.SetWindowLongW.restype = wintypes.LONG
        style = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if enable:
            new = (style | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
        else:
            new = (style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
        if new != style:
            u.SetWindowLongW(hwnd, GWL_EXSTYLE, new)
        return True
    except Exception:
        return False


# ============================================================
# 启动期硬件自检（CPU / 内存 / 显卡 / Vulkan / 引擎插件）
# ============================================================
def _hw_cpu_name():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        import platform as _pf
        return (_pf.processor() or "").strip() or "未知"


def _hw_ram_gb():
    """返回 (总内存GB, 可用内存GB)；失败返回 (0, 0)。"""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return 0.0, 0.0
        return ms.ullTotalPhys / (1024.0 ** 3), ms.ullAvailPhys / (1024.0 ** 3)
    except Exception:
        return 0.0, 0.0


def _hw_nvidia():
    """nvidia-smi 查询 NVIDIA 显卡 → [(名字, 显存GB), ...]；没有/失败返回 []。"""
    out = []
    try:
        txt = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            timeout=4, encoding="utf-8", errors="ignore",
            creationflags=CREATE_NO_WINDOW)
        for line in txt.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0]:
                vram = 0.0
                digits = "".join(ch for ch in parts[1] if ch.isdigit())
                if digits:
                    vram = int(digits) / 1024.0          # MiB → GiB
                out.append((parts[0], vram))
    except Exception:
        return []
    return out


def _hw_vulkan_ready():
    """系统是否有 Vulkan 运行时（ncnn Vulkan 引擎的前提）。"""
    if not sys.platform.startswith("win"):
        return False
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    for d in (os.path.join(sysroot, "System32"),
              os.path.join(sysroot, "SysWOW64")):
        if os.path.isfile(os.path.join(d, "vulkan-1.dll")):
            return True
    return False


def hardware_report():
    """返回 [(tag, text), ...]：启动期硬件自检结果（纯读取，不改任何东西）。"""
    rows = []

    def add(tag, text):
        rows.append((tag, "  " + text))

    add("dim", "CPU      : %s（%s 逻辑核心）"
        % (_hw_cpu_name(), os.cpu_count() or "?"))
    total, avail = _hw_ram_gb()
    if total:
        add("dim", "内存     : %.1f GB（可用 %.1f GB）" % (total, avail))

    nv = _hw_nvidia()
    vk = _hw_vulkan_ready()
    if nv:
        for i, (name, vram) in enumerate(nv):
            add("dim", "NVIDIA   : [%d] %s（显存约 %.1f GB）" % (i, name, vram))
    else:
        add("dim", "NVIDIA   : 未检测到（nvidia-smi 不可用或无 N 卡）")
    add("dim", "Vulkan   : %s" % ("运行时可用（ncnn Vulkan 引擎可走 GPU）" if vk
                                  else "未找到 vulkan-1.dll（ncnn 只能走 CPU 模式）"))

    # ── 引擎插件是否齐全 ──
    missing = []
    for d in REQUIRED_ENGINE_DIRS:
        if not os.path.exists(os.path.join(ROOT_DIR, d)):
            missing.append(d)
    if missing:
        add("warn", "引擎组件 : 缺少 %s —— 相关引擎不可用" % "、".join(missing))
    else:
        add("dim", "引擎组件 : %s 均在位" % " / ".join(REQUIRED_ENGINE_DIRS))

    # ── 推荐（与主程序「简单模式」的显卡档位规则一致）──
    if HAVE_CUDA_ENGINE and nv:
        big = max(v for _n, v in nv)
        if big >= 11:
            add("ok", "检测结论 : 有 ≥12GB 的 NVIDIA 显卡 → 可用 CUDA 加速（PP-OCRv6）；"
                      "也可选 ncnn Vulkan。默认「自动」会落到此处。")
        else:
            add("ok", "检测结论 : 检测到 NVIDIA 显卡（显存 %.1f GB）→ 建议选 ncnn Vulkan，"
                      "避免大图爆显存。默认「自动」会落到此处。" % big)
    elif vk:
        add("ok", "检测结论 : %s → 主流引擎走 ncnn Vulkan（任意显卡可用）。"
            % ("检测到 NVIDIA 显卡" if nv else "无 N 卡但有 Vulkan"))
    else:
        add("warn", "检测结论 : 未发现可用 GPU 加速 → 将走 CPU 推理，速度较慢但结果一致。")
    return rows


# ============================================================
# 路径定位（兼容源码运行 与 PyInstaller onefile 打包运行）
# ============================================================
def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _find_project_root(start):
    """从 start 起向上查找，定位包含 portapython/python.exe 的项目根目录。"""
    cur = os.path.abspath(start)
    for _ in range(6):
        if os.path.isfile(os.path.join(cur, "portapython", "python.exe")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _find_main_py(app_dir):
    """定位主程序：exe 可能放在版本根目录，也可能放在 CathayOCR-Lite 子目录。"""
    cands = [
        os.path.join(app_dir, "umi_ocr_pdf_processor_ui.py"),
        os.path.join(app_dir, "CathayOCR-Pro", "umi_ocr_pdf_processor_ui.py"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]


APP_DIR  = _app_dir()
ROOT_DIR = _find_project_root(APP_DIR) or os.path.dirname(APP_DIR)
PYTHON   = os.path.join(ROOT_DIR, "portapython", "python.exe")
MAIN_PY  = _find_main_py(APP_DIR)
MAIN_DIR = os.path.dirname(MAIN_PY)
ICON     = os.path.join(MAIN_DIR, "CathayOCR.ico")

CREATE_NO_WINDOW = 0x08000000


def _user32():
    import ctypes
    return ctypes.windll.user32


# ============================================================
# 仅清理"本软件目录内"的残留引擎进程（不按进程名全局强杀）
# ============================================================
def kill_local_engines(names=("ppocr_ocr_vulkan.exe", "PaddleOCR-json.exe", "ppocr_ocr_cpu.exe"), root=None):
    """结束 exe 完整路径位于本项目目录内的同名进程。返回 pid 列表。"""
    if not sys.platform.startswith("win"):
        return []
    root = os.path.normcase(os.path.abspath(root or ROOT_DIR))
    names_l = {str(n).lower() for n in names}
    killed = []
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return []

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]

    snapshot = k32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        if not k32.Process32FirstW(snapshot, ctypes.byref(entry)):
            return []
        while True:
            exe = entry.szExeFile or ""
            if exe.lower() in names_l:
                h = k32.OpenProcess(0x1000 | 0x0001, False, entry.th32ProcessID)
                if h:
                    try:
                        buf = ctypes.create_unicode_buffer(32768)
                        size = wintypes.DWORD(len(buf))
                        if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                            path = os.path.normcase(os.path.abspath(buf.value))
                            if path.startswith(root + os.sep):
                                if k32.TerminateProcess(h, 1):
                                    killed.append(entry.th32ProcessID)
                    finally:
                        k32.CloseHandle(h)
            if not k32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    finally:
        k32.CloseHandle(snapshot)
    return killed


# ============================================================
# 解包残留自清
# ------------------------------------------------------------
# 本程序是 PyInstaller「单文件」模式：每次启动都会把内嵌的运行环境
# （约 20MB）解包到系统 TEMP 下的 _MEIxxxxxx 目录，正常退出时由引导程序
# 自动删除。但如果本程序被【强制结束】（任务管理器 / 崩溃 / 关机时仍在运行），
# 引导程序来不及清理，这个目录就会永久留在 C 盘，越积越多。
# 因此每次启动时顺手把【上一次遗留的】清掉。
#
# 安全边界（很重要）：
#   * 只删「目录名以 _MEI 开头」且「目录里存在本启动器专属标记文件」的目录
#     —— 机器上其他 PyInstaller 程序的解包目录绝不会被误删；
#   * 跳过本次实例正在使用的那个；
#   * 删不掉（说明正被某个实例占用）就安静跳过，绝不强来。
# ============================================================
CLEAN_MARKER = "_cathayocr_launcher_bundle.marker"
TRASH_SUFFIX = ".cathayocr_trash"
CLEAN_DELAY = 6.0         # 后台删除前稍等一下，避开与新实例解包的磁盘竞争
_STALE_CLEANED = []       # 本次启动已处理的残留目录，供日志窗口显示
_PENDING_TRASH = []       # 已改名、等待删除的目录


def _spawn_detached_cleanup(paths):
    """交给一个独立进程去删 —— 本程序退出后它仍会把活干完。

    删除 20MB 解包目录在这台机器上要好几秒（杀毒软件实时扫描拖的），
    放在进程内做的话，一旦程序退出、或用户很快再点一次，
    没删完的就会留下来。用独立进程就彻底没这个问题。
    路径只可能是我们自己刚改名出来的那个目录，不接受任何外部输入。
    """
    for p in paths:
        try:
            subprocess.Popen(
                ["cmd", "/c", "rmdir", "/s", "/q", p],
                creationflags=CREATE_NO_WINDOW | 0x00000008,   # | DETACHED_PROCESS
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def _delete_later(paths):
    """进程内先把删除推迟几秒做一遍（大多数情况下这一步就够了）。

    没删完的会留在 _PENDING_TRASH 里，程序退出时再交给独立进程收尾。
    """
    _PENDING_TRASH.extend(paths)

    def _work():
        try:
            time.sleep(CLEAN_DELAY)
        except Exception:
            pass
        for p in list(_PENDING_TRASH):
            shutil.rmtree(p, ignore_errors=True)
            if not os.path.exists(p):
                try:
                    _PENDING_TRASH.remove(p)
                except ValueError:
                    pass

    t = threading.Thread(target=_work, name="stale-unpack-cleanup", daemon=True)
    t.start()
    return t


def _flush_pending_cleanup():
    """程序退出前的兜底：还没删掉的交给独立进程继续删。"""
    left = [p for p in list(_PENDING_TRASH) if os.path.isdir(p)]
    if left:
        _spawn_detached_cleanup(left)
    return left


def cleanup_stale_unpack_dirs():
    """处理本启动器上次遗留的 TEMP 解包目录。返回已改名的路径列表。

    安全边界：
      * 只碰「目录名以 _MEI 开头」且「目录里存在本启动器专属标记文件」的目录
        —— 机器上其他 PyInstaller 程序的解包目录绝不会被误删；
      * 跳过本次实例正在使用的那个；
      * 只做改名（瞬时、可逆），删除交给后台线程；被占用就安静跳过，绝不强来。
    """
    if not sys.platform.startswith("win"):
        return []
    if os.environ.get("CATHAYOCR_NO_PACK_CLEANUP"):
        return []                                     # 诊断用开关
    temp = os.environ.get("TEMP") or os.environ.get("TMP")
    if not temp or not os.path.isdir(temp):
        return []
    cur = getattr(sys, "_MEIPASS", None)
    cur = os.path.normcase(os.path.abspath(cur)) if cur else None
    pending, handled = [], []
    try:
        names = os.listdir(temp)
    except OSError:
        return []
    for name in names:
        p = os.path.join(temp, name)
        if name.endswith(TRASH_SUFFIX) or TRASH_SUFFIX in name:
            if os.path.isdir(p) and not os.path.islink(p):
                pending.append(p)                     # 上次没删完的，继续删
            continue
        if not name.upper().startswith("_MEI"):
            continue
        if cur and os.path.normcase(os.path.abspath(p)) == cur:
            continue                                  # 自己在用
        if not os.path.isdir(p) or os.path.islink(p):
            continue
        if not os.path.isfile(os.path.join(p, CLEAN_MARKER)):
            continue                                  # 不是本程序的包，绝不碰
        newp = "%s%s%d" % (p, TRASH_SUFFIX, os.getpid())
        try:
            os.rename(p, newp)                        # 瞬时
        except Exception:
            continue                                  # 被占用 → 跳过
        pending.append(newp)
        handled.append(newp)
    if pending:
        _delete_later(pending)
    return handled


# ============================================================
# 启动器 / 日志窗口
# ============================================================
class LauncherApp:
    def __init__(self, root, extra_args=None):
        self.root = root
        self.extra_args = list(extra_args or [])
        self.proc = None
        self.q = queue.Queue()
        self.t0 = 0.0
        self.autoscroll = True
        self.finished = False

        # 窗口状态
        self._hidden = False        # 窗口不可见（点 X 隐藏 / 跟随主程序收起）
        self._hidden_by_user = False  # 其中「用户主动点 X」的那种
        self._closing = False       # 「结束任务」路径
        # 默认吸附到主程序右侧（CATHAYOCR_DOCK=0 可关闭）
        self._dock = os.environ.get("CATHAYOCR_DOCK", "1") != "0"
        self._dock_user_toggled = False   # 用户手动切过就不再自动回退
        self._dock_since = time.time() if self._dock else 0.0
        self._saved_geom = None     # 吸附前的窗口几何，取消吸附时恢复
        self._applied_rect = None   # 上次应用的吸附矩形 (x, y, w, h)，避免重复搬运
        self._main_hwnd_cache = 0
        self._main_seen_visible = False   # 主窗口曾经真正显示过（避免启动期误判为"已收起"）
        self._main_hidden_sync = False     # 当前是否因为主窗口收起而跟着收起
        self._hwnd_cache = None
        self._tool_win_applied = False
        self._exit_left = 0.0       # 退出倒计时剩余秒数

        root.title(APP_TITLE)
        root.configure(bg=BG)
        root.geometry(GEOM_NORMAL)
        root.minsize(*MIN_NORMAL)
        self._set_icon()

        # ── 头部：标题 + 状态 + 计时 ──
        bar = tk.Frame(root, bg=BG_BAR, height=40)
        bar.pack(side="top", fill="x")
        tk.Label(bar, text="●", bg=BG_BAR, fg=FG_DIM,
                 font=("Microsoft YaHei UI", 11), anchor="w").pack(side="left", padx=(12, 2))
        self.dot = bar.winfo_children()[-1]
        self.status = tk.Label(bar, text="正在启动…", bg=BG_BAR, fg=FG,
                               font=("Microsoft YaHei UI", 10, "bold"), anchor="w")
        self.status.pack(side="left", padx=4)
        self.clock = tk.Label(bar, text="", bg=BG_BAR, fg=FG_DIM,
                              font=("Consolas", 10), anchor="e")
        self.clock.pack(side="right", padx=12)
        tk.Label(bar, text=APP_TITLE.split(" — ")[0], bg=BG_BAR, fg=FG_DIM,
                 font=("Microsoft YaHei UI", 9), anchor="e").pack(side="right", padx=2)
        tk.Frame(root, bg=BORDER, height=1).pack(side="top", fill="x")   # 细分隔线

        # ── 日志区 ──
        mid = tk.Frame(root, bg=BG)
        mid.pack(side="top", fill="both", expand=True)
        self.text = tk.Text(mid, bg=BG, fg=FG, insertbackground=FG, wrap="word",
                            relief="flat", borderwidth=0, padx=10, pady=8,
                            selectbackground="#cfe3ff", selectforeground=FG,
                            font=("Consolas", 10), state="disabled")
        sb = tk.Scrollbar(mid, command=self.text.yview, width=14,
                          bg=BG_BAR, troughcolor=BG_BAR, activebackground=BG_BTN_HV,
                          relief="flat", borderwidth=0)
        self.text.configure(yscrollcommand=self._on_scroll)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        for tag, color in (("err", C_ERR), ("warn", C_WARN), ("ok", C_OK),
                           ("info", C_INFO), ("dim", FG_DIM)):
            self.text.tag_configure(tag, foreground=color)

        # ── 按钮条 ──
        bot = tk.Frame(root, bg=BG_BAR, height=44)
        bot.pack(side="bottom", fill="x")
        self.btn_pause = self._mk_btn(bot, BTN_LABELS_NORMAL[0], self.toggle_scroll)
        self.btn_save  = self._mk_btn(bot, BTN_LABELS_NORMAL[1], self.save_log)
        self.btn_clear = self._mk_btn(bot, BTN_LABELS_NORMAL[2], self.clear_log)
        self.btn_stop  = self._mk_btn(bot, BTN_LABELS_NORMAL[3], self.stop_and_quit)
        self.btn_dock  = self._mk_btn(bot, BTN_LABELS_NORMAL[4], self.toggle_dock)
        self.hint = tk.Label(bot, text="", bg=BG_BAR, fg=FG_DIM,
                             font=("Microsoft YaHei UI", 9))
        self.hint.pack(side="right", padx=12)
        tk.Frame(root, bg=BORDER, height=1).pack(side="bottom", fill="x")   # 按钮条上方的分隔线
        if self._dock:
            self.root.minsize(*MIN_DOCK)
        self._apply_dock_ui()

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Unmap>", self._on_unmap)

        self._log("info", "CathayOCR Pro 启动器 v1.3.0")
        self._log("dim", "项目目录 : %s" % ROOT_DIR)
        self._log("dim", "解释器   : %s" % PYTHON)
        self._log("dim", "主程序   : %s" % MAIN_PY)
        if _STALE_CLEANED:
            self._log("dim", "已接手清理上次遗留的解包目录 %d 个（约 %d MB，后台删除中）"
                      % (len(_STALE_CLEANED), len(_STALE_CLEANED) * 20))
        self._log("info", "【硬件自检】正在检查本机配置…")

        # ── 让日志窗口不占任务栏：必须在窗口首次显示前设置 ──
        self._apply_tool_window()
        try:
            self.root.deiconify()
        except Exception:
            pass

        self.root.after(60, self._drain)
        self.root.after(500, self._tick)
        self.root.after(DOCK_POLL_MS if self._dock else POLL_MS, self._ui_poll)
        self.start()
        # 硬件自检放到后台线程：nvidia-smi 偶尔要 1~4 秒，不能卡住窗口首屏
        threading.Thread(target=self._hw_probe, name="hw-probe", daemon=True).start()

    # ---------- 界面小工具 ----------
    def _apply_tool_window(self):
        hwnd = self._own_hwnd()
        if hwnd and _set_tool_window(hwnd, True):
            self._tool_win_applied = True

    def _hw_probe(self):
        """后台跑硬件自检，把结果投进日志队列（tkinter 只能在主线程写窗口）。"""
        try:
            rows = hardware_report()
        except Exception as e:
            rows = [("warn", "  硬件自检失败：%s" % e)]
        for tag, text in rows:
            self.q.put(("__tag__", tag, text))
        self.q.put(("__tag__", "dim", "-" * 78))

    def _set_icon(self):
        try:
            if os.path.isfile(ICON):
                self.root.iconbitmap(ICON)
        except Exception:
            pass

    def _mk_btn(self, parent, text, cmd):
        b = tk.Button(parent, text=text, command=cmd, relief="flat",
                      bg=BG_BTN, fg=FG, activebackground=BG_BTN_HV,
                      activeforeground=FG, font=("Microsoft YaHei UI", 9),
                      padx=12, pady=4, cursor="hand2", borderwidth=0)
        b.bind("<Enter>", lambda e, w=b: w.configure(bg=BG_BTN_HV))
        b.bind("<Leave>", lambda e, w=b: w.configure(bg=BG_BTN))
        b.pack(side="left", padx=(8, 2), pady=7)
        return b

    def _apply_dock_ui(self):
        labels = BTN_LABELS_DOCK if self._dock else BTN_LABELS_NORMAL
        for b, t in zip((self.btn_pause, self.btn_save, self.btn_clear,
                         self.btn_stop, self.btn_dock), labels):
            b.configure(text=t)
        self.hint.configure(
            text="侧边栏（任务栏只有主程序一个图标）" if self._dock
            else "独立窗口 · 关闭=隐藏 · 双击 exe 可重新打开")

    def _own_hwnd(self):
        if self._hwnd_cache:
            return self._hwnd_cache
        try:
            import ctypes
            hwnd = _user32().GetParent(self.root.winfo_id())
            if hwnd:
                self._hwnd_cache = hwnd
        except Exception:
            pass
        return self._hwnd_cache

    def _on_scroll(self, first, last):
        self.text.yview_moveto(first)
        try:
            self.at_bottom = float(last) >= 0.999
        except Exception:
            self.at_bottom = True

    # ---------- 日志写入 ----------
    def _log(self, tag, msg):
        self.text.configure(state="normal")
        self.text.insert("end", msg.rstrip("\n") + "\n", tag if tag else None)
        self.text.configure(state="disabled")
        if self.autoscroll and getattr(self, "at_bottom", True):
            self.text.see("end")

    @staticmethod
    def _classify(line):
        low = line.lower()
        stripped = line.strip()
        if stripped.startswith("[W]") or stripped.startswith("[V]"):
            return "dim"
        for k in ERR_KEYS:
            if k in low:
                return "err"
        for k in WARN_KEYS:
            if k in low:
                return "warn"
        for k in OK_KEYS:
            if k in low:
                return "ok"
        return None

    def _drain(self):
        """定时把子进程输出刷进窗口（tkinter 只能在主线程操作）。"""
        n = 0
        while n < 400:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
            n += 1
            if item is None:            # 结束哨兵
                self._on_proc_end()
                continue
            if isinstance(item, tuple):          # 带标签的消息（硬件自检等）
                self._log(item[1], item[2])
                continue
            self._log(self._classify(item), item)
        self.root.after(60, self._drain)

    def _tick(self):
        if self.proc is not None and self.proc.poll() is None:
            el = time.time() - self.t0
            self.clock.configure(text="运行中 %02d:%02d" % (el // 60, el % 60))
        self.root.after(500, self._tick)

    # ---------- 周期巡检：外部唤起 + 吸附跟随 + 与主窗口同步收起/还原 ----------
    def _ui_poll(self):
        interval = POLL_MS
        try:
            hwnd = self._own_hwnd()
            if hwnd:
                # 1) 窗口被第二个实例从外部唤起 → 同步 Tk 状态
                if self._hidden and _user32().IsWindowVisible(hwnd):
                    self._hidden = False
                    self._hidden_by_user = False
                    self._main_hidden_sync = False
                    try:
                        self.root.deiconify()
                        self.root.lift()
                    except Exception:
                        pass
                # 2) 吸附跟随（高频，跟手）
                if self._dock and not self._hidden:
                    self._dock_follow()
                    self._maybe_give_up_dock()
                    if not self._hidden:
                        interval = DOCK_POLL_MS
                # 3) 与主窗口同步：主程序收进托盘/最小化 → 日志窗口一起收
                if not self._closing:
                    self._sync_with_main_window()
        finally:
            self.root.after(interval, self._ui_poll)

    def _maybe_give_up_dock(self):
        """启动后一段时间仍找不到主窗口 → 自动退回独立窗口（避免永远卡在窄条）。"""
        if self._dock_user_toggled or self._main_hwnd_cache:
            return
        if not self._dock_since or (time.time() - self._dock_since) < DOCK_GRACE_S:
            return
        self._dock_user_toggled = True          # 只自动回退一次
        self.toggle_dock()
        self._log("dim", "[布局] 未找到主程序窗口，已自动切回独立窗口")

    def _sync_with_main_window(self):
        """主窗口一旦被最小化/收进托盘，日志窗口跟着收起；还原时一起还原。

        只在「主窗口确实显示过至少一次」之后才生效 —— 否则程序启动初期
        主窗口还没建好，会被误判成"已收起"，把日志窗口一起藏掉。
        """
        if self.finished:
            return
        u = _user32()
        mh = self._main_hwnd_cache
        if not mh or not u.IsWindow(mh):
            self._main_hwnd_cache = 0
            self._main_seen_visible = False
            return
        shown = bool(u.IsWindowVisible(mh)) and not u.IsIconic(mh)
        if shown:
            self._main_seen_visible = True
            if self._main_hidden_sync:
                self._main_hidden_sync = False
                self._hidden = False
                self._hidden_by_user = False
                try:
                    self.root.deiconify()
                    self.root.lift()
                except Exception:
                    pass
            return
        if self._main_seen_visible and not self._main_hidden_sync:
            self._main_hidden_sync = True
            self._hidden = True
            try:
                self.root.withdraw()
            except Exception:
                pass

    def _dock_follow(self):
        u = _user32()
        # 主窗口句柄校验 / 重找
        if self._main_hwnd_cache and not u.IsWindow(self._main_hwnd_cache):
            self._main_hwnd_cache = 0
            self._main_seen_visible = False
        if not self._main_hwnd_cache:
            if self.proc is None or self.proc.poll() is not None:
                return
            self._main_hwnd_cache = _find_window_of_pid(self.proc.pid)
        mh = self._main_hwnd_cache
        if not mh:
            return
        import ctypes
        from ctypes import wintypes

        class RECT(ctypes.Structure):
            _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                        ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

        r = RECT()
        if not u.GetWindowRect(mh, ctypes.byref(r)):
            return
        if r.left < -9000 or u.IsIconic(mh):     # 主窗口最小化 / 已收起
            return
        w, h = r.right - r.left, r.bottom - r.top
        if w < 380 or h < 280:                   # 不像主窗口（对话框等）
            return
        side = DOCK_WIDTH
        x = r.right
        try:
            sw = self.root.winfo_screenwidth()
            if x + side > sw:
                x = max(0, min(sw - side, r.right - side))   # 屏幕不够 → 贴内侧
        except Exception:
            pass
        rect = (x, r.top, side, h)
        if rect == self._applied_rect:
            return                               # 主窗口没动 → 什么都不做
        if self._saved_geom is None:
            # 首次吸附前先记住"独立窗口"时的几何，取消吸附时好还原
            try:
                self._saved_geom = self.root.geometry()
            except Exception:
                self._saved_geom = GEOM_NORMAL
        self._applied_rect = rect
        # 【流畅度关键】直接用 Win32 搬窗口，不走 Tk 的 geometry()。
        # 旧实现每 250ms 调一次 root.geometry()：那会触发 Tk 整套布局重算 +
        # 重绘，拖动主窗口时日志窗明显"一顿一顿"、落后一大截。
        # SWP_NOZORDER 保持 Z 序不变（不抢焦点、不闪），SWP_NOACTIVATE 不激活。
        hwnd = self._own_hwnd()
        SWP_NOZORDER, SWP_NOACTIVATE = 0x0004, 0x0010
        if hwnd:
            u.SetWindowPos(hwnd, 0, x, r.top, side, h,
                           SWP_NOZORDER | SWP_NOACTIVATE)
        else:
            self.root.geometry("%dx%d+%d+%d" % (side, h, x, r.top))

    def toggle_dock(self):
        self._dock_user_toggled = True
        if not self._dock:
            self._saved_geom = self.root.geometry()
            self._dock = True
            self._dock_since = time.time()
            self._applied_rect = None
            self.root.minsize(*MIN_DOCK)
            self._apply_dock_ui()
            self._log("dim", "[布局] 已吸附到主程序右侧（侧边栏模式）")
        else:
            self._dock = False
            self._applied_rect = None
            self.root.minsize(*MIN_NORMAL)
            if self._saved_geom:
                self.root.geometry(self._saved_geom)
            self._keep_on_screen()
            self._apply_dock_ui()
            self._log("dim", "[布局] 已恢复独立窗口")

    def _keep_on_screen(self):
        """恢复独立窗口后确保窗口在屏幕内（否则可能停在屏幕外的侧边栏位置）。"""
        try:
            self.root.update_idletasks()
            x, y = self.root.winfo_x(), self.root.winfo_y()
            w = max(self.root.winfo_width(), MIN_NORMAL[0])
            h = max(self.root.winfo_height(), MIN_NORMAL[1])
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            nx = max(0, min(x, max(0, sw - w)))
            ny = max(0, min(y, max(0, sh - h)))
            if (nx, ny, w, h) != (x, y, self.root.winfo_width(), self.root.winfo_height()):
                self.root.geometry("%dx%d+%d+%d" % (w, h, nx, ny))
        except Exception:
            pass

    def toggle_scroll(self):
        self.autoscroll = not self.autoscroll
        text = "恢复滚动" if not self.autoscroll else ("滚动" if self._dock else "暂停滚动")
        self.btn_pause.configure(text=text)
        if self.autoscroll:
            self.text.see("end")

    def clear_log(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def save_log(self):
        path = filedialog.asksaveasfilename(
            title="保存运行日志", defaultextension=".log",
            initialfile=time.strftime("CathayOCR-Pro-%Y%m%d-%H%M%S.log"),
            filetypes=[("日志文件", "*.log"), ("文本文件", "*.txt"), ("全部文件", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text.get("1.0", "end"))
            messagebox.showinfo(APP_TITLE, "日志已保存：\n%s" % path)
        except Exception as e:
            messagebox.showerror(APP_TITLE, "保存失败：%s" % e)

    # ---------- 子进程管理 ----------
    def start(self):
        if not os.path.isfile(PYTHON):
            self._log("err", "[启动失败] 找不到便携解释器：%s" % PYTHON)
            self.status.configure(text="启动失败", fg=C_ERR)
            messagebox.showerror(APP_TITLE, "找不到便携 Python：\n%s\n\n请确认启动器与 portapython 目录的相对位置未被移动。" % PYTHON)
            return
        if not os.path.isfile(MAIN_PY):
            self._log("err", "[启动失败] 找不到主程序：%s" % MAIN_PY)
            self.status.configure(text="启动失败", fg=C_ERR)
            messagebox.showerror(APP_TITLE, "找不到主程序：\n%s" % MAIN_PY)
            return

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"     # 保证日志窗口不乱码
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"  # 不在软件目录留下 __pycache__

        # ⚠️ 这里必须用 SW_SHOWNORMAL(=1)，**不能用 SW_HIDE**：
        # STARTF_USESHOWWINDOW 会决定子进程"首个顶层窗口"的显示状态，
        # 用 SW_HIDE 会把主程序的 Qt 主窗口一起藏掉（日志窗口正常、主界面不出来）。
        # 注意：Python 的 subprocess 只定义了 SW_HIDE，没有 SW_SHOWNORMAL，必须写数值 1。
        # 控制台不会冒出来：CREATE_NO_WINDOW 已经保证子进程没有控制台。
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 1  # SW_SHOWNORMAL

        cmd = [PYTHON, "-u", MAIN_PY] + self.extra_args
        self._log("dim", "> " + " ".join('"%s"' % c if " " in c else c for c in cmd))
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=MAIN_DIR, env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                startupinfo=si, creationflags=CREATE_NO_WINDOW, bufsize=0)
        except Exception as e:
            self._log("err", "[启动失败] %s" % e)
            self.status.configure(text="启动失败", fg=C_ERR)
            return

        self.t0 = time.time()
        self.status.configure(text="运行中", fg=C_OK)
        self.dot.configure(fg=C_OK)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            for raw in iter(self.proc.stdout.readline, b""):
                self.q.put(raw.decode("utf-8", "replace"))
        except Exception:
            pass
        finally:
            try:
                self.proc.stdout.close()
            except Exception:
                pass
            self.q.put(None)

    def _on_proc_end(self):
        if self.finished:
            return
        self.finished = True
        # 【修复】本函数是「子进程 stdout 已 EOF」触发的，不代表进程已经结束：
        # 子进程在解释器收尾时会先关掉 stdout，之后还要几十~几百毫秒才真正退出，
        # 此刻 poll() 返回 None → 被误报成「异常退出（退出码 None）」。
        # 实测主程序关窗后 app.exec_() 返回 0、进程退出码 0，完全正常。
        # 故这里等它真正结束再取退出码（超时兜底退回 poll）。
        code = None
        if self.proc is not None:
            try:
                code = self.proc.wait(timeout=8)
            except Exception:
                code = self.proc.poll()
        el = time.time() - self.t0 if self.t0 else 0
        self._log("dim", "-" * 78)
        if code == 0:
            self._log("ok", "[结束] 主程序已正常退出（退出码 0，用时 %d 秒）" % el)
            self._log("ok", "[结束] 引擎子进程已随主程序一并收尾，没有残留。")
        else:
            self._log("err", "[结束] 主程序异常退出（退出码 %s，用时 %d 秒）—— 请查看上方红色日志" % (code, el))
        self.dot.configure(fg=C_ACCENT if code == 0 else C_ERR)
        self._main_hwnd_cache = 0
        self._main_seen_visible = False
        self._main_hidden_sync = False
        if self._dock:
            self.toggle_dock()
        try:
            self.btn_stop.configure(
                text="关闭" if self._dock else "关闭窗口", command=self.on_close)
        except Exception:
            pass

        if self._closing:
            self.root.destroy()
            return
        if self._hidden_by_user:
            self.root.destroy()               # 用户自己关掉的窗口 → 安静退出
            return

        # ── 走到这里说明窗口还在（含"跟着主程序收进托盘"的情况）──
        # 无论正常还是异常退出，都把窗口显示出来停一会儿，让用户看清结论。
        was_auto_hidden = self._hidden and not self._hidden_by_user
        self._hidden = False
        self._main_hidden_sync = False
        try:
            if was_auto_hidden:
                self.root.deiconify()
            self.root.lift()
        except Exception:
            pass

        if code == 0:
            # 正常退出：停留 EXIT_STAY_MS 毫秒（带秒级倒计时）后自动关闭
            self._exit_left = EXIT_STAY_MS / 1000.0
            self._tick_exit()
        else:
            # 异常退出：保持打开，便于查看错误日志
            self.status.configure(text="异常结束（退出码 %s）——窗口保持打开" % code, fg=C_ERR)

    def _tick_exit(self):
        """正常退出后的停留倒计时（每秒刷新一次状态栏）。"""
        if self._closing:
            return
        if self._exit_left <= 0:
            self._auto_close()
            return
        self.status.configure(
            text="已正常结束 · %.0f 秒后自动关闭窗口" % self._exit_left, fg=C_ACCENT)
        self._exit_left -= 1
        self.root.after(1000, self._tick_exit)

    def _auto_close(self):
        if not self._hidden_by_user and not self._closing:
            self.root.destroy()

    def stop_and_quit(self):
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno(
                    APP_TITLE,
                    "确定要结束正在运行的 OCR 任务吗？\n\n"
                    "· 已完成的页面结果已按文件保存，不会损坏；\n"
                    "· 当前正在识别的文件会被中断，下次可重新处理。\n\n"
                    "（更稳妥的做法：先在主界面中正常关闭程序）"):
                return
            self._closing = True
            self._terminate()
        self.root.destroy()

    def _terminate(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self.proc.terminate()
            for _ in range(30):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.1)
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
        # 主程序被强杀时不会走自身清理流程 → 仅清理本目录内的残留引擎
        try:
            killed = kill_local_engines()
            if killed:
                self._log("warn", "[清理] 已结束本目录残留引擎进程: %s" % killed)
        except Exception:
            pass

    def _on_unmap(self, _evt=None):
        """日志窗口被最小化（点标题栏的最小化）→ 跟着主程序一起收起。

        日志窗口是"工具窗口"、任务栏里本来就没有它，一旦最小化就再也点不回来，
        所以这里把最小化改造成"把主程序一起最小化，然后自己也藏起来"。
        主程序究竟是进任务栏还是收进托盘，由主界面上的「最小化到托盘」开关决定。
        """
        try:
            if self.root.state() != "iconic":
                return
        except Exception:
            return
        if self.finished or self._closing or self._hidden:
            return
        self._minimize_both()

    def _minimize_both(self):
        """最小化主程序（去任务栏还是去托盘由主程序自己的开关决定），自己同时藏起来。"""
        mh = self._main_hwnd_cache
        if mh and _user32().IsWindow(mh):
            try:
                u = _user32()
                SW_MINIMIZE = 6
                u.ShowWindow(mh, SW_MINIMIZE)
            except Exception:
                pass
        self._hidden = True
        try:
            self.root.withdraw()
        except Exception:
            pass
        self._log("dim", "[%s] 已收起 —— 还原时本窗口会一起回来。"
                  % time.strftime("%H:%M:%S"))

    def on_close(self):
        """点 X：任务运行中只是隐藏窗口（任务继续跑），否则关闭退出。"""
        if self.proc is not None and self.proc.poll() is None and not self.finished:
            self._hidden = True
            self._hidden_by_user = True
            self.root.withdraw()
            self._log("dim", "[%s] 日志窗口已隐藏 —— 任务仍在后台运行；"
                      "再次双击 exe 可重新打开本窗口。" % time.strftime("%H:%M:%S"))
            return
        self.root.destroy()


def _crash_log(exc: str):
    try:
        p = os.path.join(os.environ.get("TEMP", APP_DIR), "CathayOCR_Pro_launcher_crash.log")
        with open(p, "a", encoding="utf-8") as f:
            f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), exc))
    except Exception:
        pass


def main():
    # ── 单实例：已有实例在跑 → 只是把它的日志窗口调出来，不重复启动程序 ──
    try:
        mutex = _acquire_instance_mutex()
    except Exception:
        mutex = object()
    if mutex is None:
        _show_existing_log_window()
        return
    # ── 先清掉上次被强杀时遗留的解包目录（内部已做安全边界）──
    try:
        _STALE_CLEANED.extend(cleanup_stale_unpack_dirs())
    except Exception:
        pass
    try:
        root = tk.Tk()
    except Exception as e:
        _crash_log(repr(e))
        raise
    # 先藏起来：等窗口部件建好、扩展样式（不占任务栏）设好之后再显示，
    # 这样任务栏里不会先闪出一下日志窗口的图标。
    try:
        root.withdraw()
    except Exception:
        pass
    LauncherApp(root, extra_args=sys.argv[1:])
    root.mainloop()
    try:
        _flush_pending_cleanup()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        _crash_log(traceback.format_exc())
        raise
