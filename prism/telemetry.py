"""
prism.telemetry: Hardware Telemetry and CUDA Environment Bootstrapper.
Directly probes NVIDIA NVML under WSL2/Linux and ensures dynamic linker resolution.
"""

import ctypes
import glob
import os
import site
import sys
from pathlib import Path
from typing import Any, Dict, List

_BOOTSTRAPPED = False

WSL_LIB_DIR = "/usr/lib/wsl/lib"

# Load order matters a little: cuDNN and cuBLAS depend on the CUDA runtime.
_NVIDIA_LIB_PREFIXES = (
    "libcudart", "libcublasLt", "libcublas", "libcurand", "libcufft", "libnvrtc", "libcudnn",
)


def _site_package_dirs() -> List[str]:
    """Absolute site-packages style directories for the running interpreter (venv, user, system)."""
    candidates: List[str] = []
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:  # some old virtualenv builds lack getsitepackages
        pass
    try:
        candidates.append(site.getusersitepackages())
    except AttributeError:
        pass
    candidates.extend(p for p in sys.path if p)
    seen = set()
    result: List[str] = []
    for path in candidates:
        if os.path.isabs(path) and path not in seen and os.path.isdir(path):
            seen.add(path)
            result.append(path)
    return result


def find_nvidia_lib_dirs() -> List[str]:
    """Finds `nvidia/<pkg>/lib` directories shipped by the pip `nvidia-*-cu12` wheels."""
    dirs: List[str] = []
    for sp in _site_package_dirs():
        for lib_dir in sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib"))):
            if lib_dir not in dirs:
                dirs.append(lib_dir)
    return dirs


def _preload_cuda_libs(lib_dirs: List[str]) -> List[str]:
    """
    dlopen()s the pip-installed CUDA libraries with RTLD_GLOBAL so ONNX Runtime's CUDA provider
    can resolve them by soname. Changing LD_LIBRARY_PATH after the interpreter starts does not
    affect this process's own dlopen search path, so preloading is what actually makes them visible.
    Returns the paths that were loaded.
    """
    loaded: List[str] = []
    for prefix in _NVIDIA_LIB_PREFIXES:
        for lib_dir in lib_dirs:
            matches = sorted(glob.glob(os.path.join(lib_dir, f"{prefix}.so*")))
            if not matches:
                continue
            try:
                ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                loaded.append(matches[0])
                break
            except OSError:
                continue
    return loaded


def bootstrap_cuda_env() -> None:
    """
    Makes user-space CUDA/cuDNN wheels visible before ORT loads: preloads them into this process and
    prepends their directories to LD_LIBRARY_PATH (which only helps child processes such as `prism mcp`).
    A no-op when no `nvidia/*/lib` directories exist.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True

    lib_dirs = find_nvidia_lib_dirs()
    if os.path.isdir(WSL_LIB_DIR):
        lib_dirs.insert(0, WSL_LIB_DIR)

    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    dirs_to_add = [d for d in lib_dirs if d not in cur_ld.split(":")]
    if dirs_to_add:
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs_to_add + ([cur_ld] if cur_ld else []))

    _preload_cuda_libs([d for d in lib_dirs if d != WSL_LIB_DIR])

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
