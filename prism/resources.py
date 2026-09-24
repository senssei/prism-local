"""
prism.resources: RAM and VRAM Resource Budget and Capacity Guard.
Enforces memory limits prior to loading models to prevent system thrashing or host crashes.
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from prism.engine import ModelLoadError
from prism.telemetry import get_gpu_info

DEFAULT_VRAM_RESERVE_MB = 1536.0
DEFAULT_RAM_RESERVE_MB = 2048.0
DEFAULT_PREFILL_HEADROOM_MB_PER_TOKEN = 1.4


class InsufficientResourcesError(ModelLoadError):
    """Raised when available memory (RAM or VRAM) is insufficient to load the model safely."""


def dir_size_mb(path: Union[str, Path]) -> float:
    """Calculates total size of files directly within a directory or file in megabytes."""
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return round(p.stat().st_size / (1024 * 1024), 1)
    size_bytes = sum(f.stat().st_size for f in p.glob("*") if f.is_file())
    return round(size_bytes / (1024 * 1024), 1)


def system_memory_info() -> Dict[str, float]:
    """Returns total, available, and used system RAM and swap in megabytes by reading /proc/meminfo."""
    stats = {
        "ram_total_mb": 0.0,
        "ram_available_mb": 0.0,
        "ram_used_mb": 0.0,
        "swap_total_mb": 0.0,
        "swap_free_mb": 0.0,
        "swap_used_mb": 0.0,
    }
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    try:
                        val_kb = float(parts[1])
                    except ValueError:
                        continue
                    if key == "MemTotal":
                        stats["ram_total_mb"] = round(val_kb / 1024, 1)
                    elif key == "MemAvailable":
                        stats["ram_available_mb"] = round(val_kb / 1024, 1)
                    elif key == "SwapTotal":
                        stats["swap_total_mb"] = round(val_kb / 1024, 1)
                    elif key == "SwapFree":
                        stats["swap_free_mb"] = round(val_kb / 1024, 1)
        stats["ram_used_mb"] = round(max(0.0, stats["ram_total_mb"] - stats["ram_available_mb"]), 1)
        stats["swap_used_mb"] = round(max(0.0, stats["swap_total_mb"] - stats["swap_free_mb"]), 1)
    except (OSError, ValueError):
        pass
    return stats


def ram_available_mb() -> float:
    """Returns available system RAM in megabytes by reading /proc/meminfo."""
    return system_memory_info()["ram_available_mb"]


def is_wsl_system() -> bool:
    """Returns True if the current environment is running inside WSL."""
    forced = os.environ.get("PRISM_FORCE_WSL")
    if forced == "1":
        return True
    if forced == "0":
        return False
    if os.environ.get("PRISM_WSLCONFIG_PATH"):
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            content = f.read().lower()
            return "microsoft" in content or "wsl" in content
    except OSError:
        return False


def find_wslconfig_path() -> Optional[Path]:
    """Finds the path to .wslconfig on the Windows host when running in WSL."""
    explicit = os.environ.get("PRISM_WSLCONFIG_PATH")
    if explicit:
        return Path(explicit)
    if not is_wsl_system():
        return None

    users_dir = Path("/mnt/c/Users")
    if users_dir.is_dir():
        ignored = {"public", "default", "default user", "all users"}
        try:
            for user_dir in users_dir.iterdir():
                if user_dir.is_dir() and user_dir.name.lower() not in ignored:
                    candidate = user_dir / ".wslconfig"
                    if candidate.is_file():
                        return candidate
        except OSError:
            pass
    return None


def inspect_wslconfig(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Inspects a Windows .wslconfig file for memory limits and autoMemoryReclaim.
    Read-only: Prism NEVER modifies .wslconfig.
    """
    wsl = is_wsl_system() or path is not None
    p = Path(path) if path is not None else find_wslconfig_path()
    if p is None or not p.is_file():
        return {
            "is_wsl": wsl,
            "found": False,
            "path": str(p) if p else None,
            "has_memory": False,
            "has_reclaim": False,
        }
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {
            "is_wsl": wsl,
            "found": True,
            "path": str(p),
            "has_memory": False,
            "has_reclaim": False,
        }
    has_memory = False
    has_reclaim = False
    current_section = ""
    for line in content.splitlines():
        line = line.strip().lower()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            continue
        if current_section == "wsl2":
            compact = line.replace(" ", "")
            if compact.startswith("memory="):
                has_memory = True
            if "automemoryreclaim=gradual" in compact or "automemoryreclaim=dropcache" in compact:
                has_reclaim = True
    return {
        "is_wsl": wsl,
        "found": True,
        "path": str(p),
        "has_memory": has_memory,
        "has_reclaim": has_reclaim,
    }


def active_holder_summary() -> Optional[str]:
    """Returns a summary string of the process currently holding the model load lock, if any."""
    try:
        from prism.paths import state_dir
        from prism.machine_lock import _read_holder_info
        lock_file = state_dir() / "load.lock"
        info = _read_holder_info(lock_file)
        if info.get("pid"):
            return f"PID {info['pid']} (loading '{info.get('model', 'unknown')}')"
    except Exception:
        pass
    return None


def vram_free_mb(device_index: int = 0) -> Optional[float]:
    """Returns free VRAM in megabytes for the specified GPU device using live telemetry."""
    gpu = get_gpu_info(max_age=0)
    if not gpu.get("available"):
        return None
    devices = gpu.get("devices", [])
    if 0 <= device_index < len(devices):
        return float(devices[device_index].get("vram_free_mb", 0.0))
    return None


def low_vram_threshold_mb() -> float:
    """Returns the per-device free-VRAM threshold in MiB below which a diagnostic warning is warranted.

    Reads `$PRISM_VRAM_RESERVE_MB` (the same reserve `check_can_load` uses to refuse a load).
    Default `DEFAULT_VRAM_RESERVE_MB` (1536.0). Empty, non-numeric, or non-finite values
    (`inf`, `-inf`, `nan`) fall back to the default — matching `check_can_load`'s tolerant
    handling at the call site below AND ensuring the diagnostic site in `prism doctor`
    can never crash or silently disable the warning (spec P17: "Failure modes: none.
    `prism doctor` exits 0 regardless of VRAM state").
    """
    raw = os.environ.get("PRISM_VRAM_RESERVE_MB", "").strip()
    if not raw:
        return DEFAULT_VRAM_RESERVE_MB
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_VRAM_RESERVE_MB
    if not math.isfinite(value):
        return DEFAULT_VRAM_RESERVE_MB
    return value


def estimate_load_mb(model_path: Union[str, Path], prefill_chunk: Optional[int] = None) -> float:
    """
    Estimates memory required to host and run inference on the model.
    Accounts for weights footprint plus prefill activation headroom.
    """
    weights_mb = dir_size_mb(model_path)
    if prefill_chunk is None:
        raw_chunk = os.environ.get("PRISM_PREFILL_CHUNK", "1024").strip().lower()
        if raw_chunk in ("0", "off", "none", "false"):
            chunk_tokens = 0
        else:
            try:
                chunk_tokens = int(raw_chunk)
            except ValueError:
                chunk_tokens = 1024
    else:
        chunk_tokens = prefill_chunk

    headroom_mb = chunk_tokens * DEFAULT_PREFILL_HEADROOM_MB_PER_TOKEN
    return round(weights_mb + headroom_mb, 1)


def check_can_load(model_path: Union[str, Path], device: str = "auto") -> None:
    """
    Verifies that system RAM (and VRAM if running on CUDA) is sufficient to load
    and run the model without exceeding hardware limits.
    """
    if os.environ.get("PRISM_RESOURCE_CHECK", "").strip().lower() in ("off", "0", "false", "no"):
        return

    try:
        ram_reserve = float(os.environ.get("PRISM_RAM_RESERVE_MB", DEFAULT_RAM_RESERVE_MB))
    except ValueError:
        ram_reserve = DEFAULT_RAM_RESERVE_MB

    try:
        vram_reserve = float(os.environ.get("PRISM_VRAM_RESERVE_MB", DEFAULT_VRAM_RESERVE_MB))
    except ValueError:
        vram_reserve = DEFAULT_VRAM_RESERVE_MB

    needed_mb = estimate_load_mb(model_path)
    holder = active_holder_summary()
    holder_str = f" Held by {holder}." if holder else ""

    # Check host RAM
    avail_ram = ram_available_mb()
    required_ram = needed_mb + ram_reserve
    if avail_ram < required_ram:
        raise InsufficientResourcesError(
            f"Insufficient RAM to load model at '{model_path}': "
            f"{avail_ram:.1f} MB available < {required_ram:.1f} MB required "
            f"({needed_mb:.1f} MB model footprint + {ram_reserve:.1f} MB reserve).{holder_str} "
            f"Override with PRISM_RESOURCE_CHECK=off or adjust PRISM_RAM_RESERVE_MB."
        )

    # Resolve device if 'auto'
    target_device = device.lower()
    if target_device == "auto":
        gpu = get_gpu_info(max_age=0)
        target_device = "cuda" if gpu.get("available") else "cpu"

    # Check GPU VRAM if running on CUDA
    if target_device == "cuda":
        free_vram = vram_free_mb()
        if free_vram is not None:
            required_vram = needed_mb + vram_reserve
            if free_vram < required_vram:
                raise InsufficientResourcesError(
                    f"Insufficient VRAM to load model at '{model_path}' on CUDA: "
                    f"{free_vram:.1f} MB free < {required_vram:.1f} MB required "
                    f"({needed_mb:.1f} MB model footprint + {vram_reserve:.1f} MB reserve).{holder_str} "
                    f"Override with PRISM_RESOURCE_CHECK=off or adjust PRISM_VRAM_RESERVE_MB."
                )
