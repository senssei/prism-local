"""
foundry_ng.telemetry: Hardware Telemetry and CUDA Environment Bootstrapper.
Directly probes NVIDIA NVML under WSL2/Linux and ensures dynamic linker resolution.
"""

import ctypes
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_BOOTSTRAPPED = False

def bootstrap_cuda_env() -> None:
    """Configures LD_LIBRARY_PATH with user-space CUDA and cuDNN libraries before ORT loads."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED or os.environ.get("_FOUNDRY_NG_CUDA_BOOTSTRAPPED") == "1":
        return

    home = str(Path.home())
    candidate_dirs = [
        "/usr/lib/wsl/lib",
        f"{home}/.local/lib/python3.12/site-packages/nvidia/cublas/lib",
        f"{home}/.local/lib/python3.12/site-packages/nvidia/cudnn/lib",
        f"{home}/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
        f"{home}/.local/lib/python3.12/site-packages/nvidia/curand/lib",
        f"{home}/.local/lib/python3.12/site-packages/nvidia/cufft/lib",
        f"{home}/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib",
        f"{home}/02-ollama-loadtest/.venv/lib/python3.12/site-packages/nvidia/cublas/lib",
        f"{home}/02-ollama-loadtest/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
    ]

    existing_dirs = [d for d in candidate_dirs if os.path.isdir(d)]
    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")

    dirs_to_add = [d for d in existing_dirs if d not in cur_ld.split(":")]
    if dirs_to_add:
        new_ld = ":".join(dirs_to_add) + (":" + cur_ld if cur_ld else "")
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["_FOUNDRY_NG_CUDA_BOOTSTRAPPED"] = "1"
        _BOOTSTRAPPED = True

class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]

def get_gpu_info() -> Dict[str, Any]:
    """Queries NVIDIA GPU telemetry via libnvidia-ml.so.1."""
    nvml_paths = [
        "/usr/lib/wsl/lib/libnvidia-ml.so.1",
        "libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    ]
    nvml = None
    loaded_path = None
    for p in nvml_paths:
        try:
            nvml = ctypes.CDLL(p)
            loaded_path = p
            break
        except OSError:
            continue

    if not nvml:
        return {"available": False, "error": "libnvidia-ml.so.1 not found"}

    try:
        if nvml.nvmlInit_v2() != 0:
            return {"available": False, "error": "nvmlInit_v2 failed"}

        count = ctypes.c_uint()
        nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))

        devices: List[Dict[str, Any]] = []
        for i in range(count.value):
            handle = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)) != 0:
                continue

            name_buf = ctypes.create_string_buffer(64)
            nvml.nvmlDeviceGetName(handle, name_buf, 64)
            dev_name = name_buf.value.decode("utf-8")

            major = ctypes.c_int()
            minor = ctypes.c_int()
            nvml.nvmlDeviceGetCudaComputeCapability(handle, ctypes.byref(major), ctypes.byref(minor))

            mem = NvmlMemory()
            nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))

            devices.append({
                "index": i,
                "name": dev_name,
                "compute_capability": f"{major.value}.{minor.value}",
                "vram_total_mb": round(mem.total / (1024 * 1024), 1),
                "vram_used_mb": round(mem.used / (1024 * 1024), 1),
                "vram_free_mb": round(mem.free / (1024 * 1024), 1),
                "qualifying_cuda": major.value >= 5,
            })

        nvml.nvmlShutdown()
        return {
            "available": len(devices) > 0,
            "driver_path": loaded_path,
            "device_count": count.value,
            "devices": devices,
        }
    except Exception as ex:
        return {"available": False, "error": str(ex)}
