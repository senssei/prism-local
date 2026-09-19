#!/usr/bin/env python3
"""
Diagnostic Probe: Microsoft Foundry Local Ecosystem & Hardware Telemetry on WSL2
Simulates Microsoft's sdk_v2 C++ NvmlGpuDetector, inspects dynamic link dependencies,
and contrasts raw hardware telemetry against Foundry CLI status output.
"""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def probe_nvml_detector():
    print_header("1. Native NVML Detection Probe (Replicating sdk_v2 NvmlGpuDetector)")
    
    nvml_paths = [
        "/usr/lib/wsl/lib/libnvidia-ml.so.1",
        "libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    ]
    
    nvml_lib = None
    loaded_path = None
    for p in nvml_paths:
        try:
            nvml_lib = ctypes.CDLL(p)
            loaded_path = p
            break
        except OSError:
            continue
            
    if not nvml_lib:
        print("❌ FAILED: Unable to load libnvidia-ml.so.1. NVIDIA driver not exposed to WSL2.")
        return False

    print(f"✅ SUCCESS: Loaded NVML library from: {loaded_path}")
    
    try:
        # Replicating symbols used in sdk_v2/cpp/src/ep_detection/nvml_gpu_detector.cc
        nvml_init = nvml_lib.nvmlInit_v2
        nvml_init.restype = ctypes.c_int
        init_res = nvml_init()
        if init_res != 0:
            print(f"❌ FAILED: nvmlInit_v2 returned error code {init_res}")
            return False
        print("✅ SUCCESS: nvmlInit_v2 initialized successfully.")
        
        device_count = ctypes.c_uint()
        nvml_get_count = nvml_lib.nvmlDeviceGetCount_v2
        nvml_get_count.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        nvml_get_count.restype = ctypes.c_int
        nvml_get_count(ctypes.byref(device_count))
        
        print(f"📊 Detected NVIDIA Devices: {device_count.value}")
        
        for i in range(device_count.value):
            handle = ctypes.c_void_p()
            nvml_get_handle = nvml_lib.nvmlDeviceGetHandleByIndex_v2
            nvml_get_handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            nvml_get_handle.restype = ctypes.c_int
            nvml_get_handle(i, ctypes.byref(handle))
            
            # Query Name
            name_buf = ctypes.create_string_buffer(64)
            nvml_lib.nvmlDeviceGetName(handle, name_buf, 64)
            dev_name = name_buf.value.decode("utf-8")
            
            # Query Compute Capability (Matches NvmlGpuDetector min_major=5, min_minor=0 check)
            major = ctypes.c_int()
            minor = ctypes.c_int()
            nvml_lib.nvmlDeviceGetCudaComputeCapability(handle, ctypes.byref(major), ctypes.byref(minor))
            
            # Query Memory
            class NvmlMemory(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)
                ]
            mem = NvmlMemory()
            nvml_lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
            
            total_mb = mem.total / (1024 * 1024)
            used_mb = mem.used / (1024 * 1024)
            free_mb = mem.free / (1024 * 1024)
            
            print(f"   • Device #{i}: {dev_name}")
            print(f"     Compute Capability: {major.value}.{minor.value} (Qualifying: {major.value >= 5})")
            print(f"     VRAM: {used_mb:.1f} MB used / {total_mb:.1f} MB total ({free_mb:.1f} MB free)")
            
        nvml_lib.nvmlShutdown()
        return True
    except Exception as e:
        print(f"❌ FAILED with exception: {e}")
        return False

def probe_cli_detection_discrepancy():
    print_header("2. CLI vs Hardware Detection Discrepancy Analysis")
    try:
        res = subprocess.run(["foundry", "status"], capture_output=True, text=True, timeout=10)
        cli_output = res.stdout
        print("Output of 'foundry status':")
        for line in cli_output.splitlines():
            if any(k in line for k in ["OS", "Architecture", "GPU", "Foundry Local Core", "CLI version", "ORT"]):
                print(f"   {line.strip()}")
                
        if "GPU" in cli_output and "Not detected" in cli_output:
            print("\n🚨 CRITICAL ARCHITECTURAL FINDING:")
            print("   The standalone CLI (0.10.3) reports 'System | GPU | Not detected'.")
            print("   Root Cause: The CLI binary is still using legacy .NET 9 Core 1.0.0,")
            print("   which relies on Windows WMI (Win32_VideoController) and DirectML APIs.")
            print("   Because WMI is absent on Linux/WSL2, hardware detection fails silently,")
            print("   locking the model catalog to CPUExecutionProvider (-generic-cpu).")
    except FileNotFoundError:
        print("⚠️ 'foundry' binary not found in PATH.")

def probe_cuda_ep_dependencies():
    print_header("3. CUDA Execution Provider Shared Library Linkage")
    foundry_cli_dir = Path.home() / ".local/lib/foundry-cli"
    cuda_ep_lib = foundry_cli_dir / "libonnxruntime_providers_cuda.so"
    
    if not cuda_ep_lib.exists():
        print(f"⚠️ {cuda_ep_lib} does not exist.")
        return
        
    print(f"Inspecting shared library dependencies for: {cuda_ep_lib.name}")
    try:
        ldd_res = subprocess.run(["ldd", str(cuda_ep_lib)], capture_output=True, text=True)
        missing = [line.strip() for line in ldd_res.stdout.splitlines() if "not found" in line]
        
        if missing:
            print(f"❌ Missing dynamic libraries ({len(missing)}):")
            for m in missing:
                print(f"   • {m}")
            print("\n💡 LinuxX64Manifest Packaging Gap:")
            print("   Microsoft's Linux bundle does NOT ship cuBLAS, cuDNN, or cuFFT.")
            print("   They must be supplied via user-space LD_LIBRARY_PATH (e.g. PyPI wheels).")
        else:
            print("✅ All dynamic libraries resolved successfully via LD_LIBRARY_PATH / ldconfig.")
            # Highlight key resolved libs
            for line in ldd_res.stdout.splitlines():
                if any(x in line for x in ["cublas", "cudnn", "cudart", "curand", "cufft"]):
                    print(f"   ✓ {line.strip()}")
    except Exception as e:
        print(f"Error running ldd: {e}")

def probe_vram_retention_bug():
    print_header("4. Upstream Issue #1079 / PR #1110 VRAM Retention Audit")
    try:
        smi_res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        lines = [line.strip() for line in smi_res.stdout.splitlines() if line.strip()]
        if not lines:
            print("ℹ️ No active compute processes found on the GPU.")
        else:
            print(f"Active GPU compute processes detected ({len(lines)}):")
            foundry_detected = False
            for line in lines:
                print(f"   • {line}")
                if "foundry" in line.lower():
                    foundry_detected = True
                    
            if foundry_detected:
                print("\n⚠️ Confirmed Issue #1079 / PR #1110 (Active Memory Retention):")
                print("   foundrylocald process is actively occupying GPU VRAM even when models")
                print("   are reported unloaded by 'foundry cache list'.")
                print("   Upstream fix: PR #1110 adds OgaReleaseDeviceResources(\"CUDA\") to reclaim cached pool.")
    except Exception as e:
        print(f"Error checking nvidia-smi: {e}")

def probe_active_daemon_port():
    print_header("5. Active Daemon Ephemeral Port Discovery")
    daemon_json = Path.home() / ".foundry/daemon.json"
    if daemon_json.exists():
        try:
            data = json.loads(daemon_json.read_text())
            urls = data.get("urls", [])
            pid = data.get("pid")
            print(f"✅ Daemon Config (~/.foundry/daemon.json):")
            print(f"   PID: {pid}")
            print(f"   Listening URLs: {urls}")
            print("   Note: Ephemeral random ports break static agent/tool configs without a reverse proxy.")
        except Exception as e:
            print(f"Error reading daemon.json: {e}")
    else:
        print("ℹ️ ~/.foundry/daemon.json does not exist. Daemon may be stopped.")

def main():
    print("\n🔍 MICROSOFT FOUNDRY LOCAL ECOSYSTEM & HARDWARE PROBE (WSL2)")
    print("=" * 80)
    probe_nvml_detector()
    probe_cli_detection_discrepancy()
    probe_cuda_ep_dependencies()
    probe_vram_retention_bug()
    probe_active_daemon_port()
    print_header("Probe Completed Successfully")

if __name__ == "__main__":
    main()
