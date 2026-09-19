"""
foundry_wsl.doctor: Diagnostic health checker for Microsoft Foundry Local on WSL2.
"""

import ctypes
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

def check_nvml_detection() -> Dict[str, Any]:
    """Inspects whether NVIDIA NVML is exposed in WSL2."""
    nvml_paths = [
        "/usr/lib/wsl/lib/libnvidia-ml.so.1",
        "libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    ]
    loaded_path = None
    nvml_lib = None
    for p in nvml_paths:
        try:
            nvml_lib = ctypes.CDLL(p)
            loaded_path = p
            break
        except OSError:
            continue

    if not nvml_lib:
        return {"status": "error", "message": "libnvidia-ml.so.1 not found"}

    try:
        nvml_init = nvml_lib.nvmlInit_v2
        if nvml_init() != 0:
            return {"status": "error", "message": "nvmlInit_v2 failed"}

        count = ctypes.c_uint()
        nvml_lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
        
        devices = []
        for i in range(count.value):
            handle = ctypes.c_void_p()
            nvml_lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle))
            name_buf = ctypes.create_string_buffer(64)
            nvml_lib.nvmlDeviceGetName(handle, name_buf, 64)
            major = ctypes.c_int()
            minor = ctypes.c_int()
            nvml_lib.nvmlDeviceGetCudaComputeCapability(handle, ctypes.byref(major), ctypes.byref(minor))
            devices.append({
                "index": i,
                "name": name_buf.value.decode("utf-8"),
                "compute_capability": f"{major.value}.{minor.value}",
                "qualifying": major.value >= 5
            })

        nvml_lib.nvmlShutdown()
        return {
            "status": "ok",
            "library_path": loaded_path,
            "device_count": count.value,
            "devices": devices
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def check_cuda_libraries() -> Dict[str, Any]:
    """Verifies whether libonnxruntime_providers_cuda.so has all dependencies resolved."""
    foundry_cli_dir = Path.home() / ".local/lib/foundry-cli"
    cuda_ep_lib = foundry_cli_dir / "libonnxruntime_providers_cuda.so"
    
    if not cuda_ep_lib.exists():
        return {"status": "warning", "message": f"{cuda_ep_lib} does not exist"}

    try:
        ldd_res = subprocess.run(["ldd", str(cuda_ep_lib)], capture_output=True, text=True, check=False)
        missing = [line.strip() for line in ldd_res.stdout.splitlines() if "not found" in line]
        resolved = [line.strip() for line in ldd_res.stdout.splitlines() if "=>" in line]
        return {
            "status": "ok" if not missing else "error",
            "missing_libraries": missing,
            "resolved_count": len(resolved)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_daemon_info() -> Dict[str, Any]:
    """Reads active listening URLs and PID from ~/.foundry/daemon.json."""
    daemon_json = Path.home() / ".foundry/daemon.json"
    if not daemon_json.exists():
        return {"status": "stopped", "message": "daemon.json not found"}
    try:
        data = json.loads(daemon_json.read_text())
        return {
            "status": "running" if data.get("pid") else "unknown",
            "pid": data.get("pid"),
            "urls": data.get("urls", [])
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_doctor_report() -> Dict[str, Any]:
    """Executes a complete diagnostic assessment."""
    return {
        "nvml": check_nvml_detection(),
        "cuda_libraries": check_cuda_libraries(),
        "daemon": get_daemon_info()
    }
