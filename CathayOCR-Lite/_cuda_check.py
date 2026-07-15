"""CUDA detection for CathayOCR Lite (Vulkan-only edition)
Checks if NVIDIA GPU is available. If not, just prints info.
This version doesn't use CUDA - all GPU goes through Vulkan.
"""
import sys
import subprocess

try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0 and result.stdout.strip():
        print("[GPU] NVIDIA GPU detected (will use Vulkan backend)")
    else:
        print("[GPU] No NVIDIA GPU found. Vulkan will use available GPUs or CPU.")
except FileNotFoundError:
    print("[GPU] nvidia-smi not found. Vulkan will use available GPUs or CPU.")
except Exception as e:
    print(f"[GPU] Detection skipped: {e}")

print("[OK] Vulkan-only edition ready")
