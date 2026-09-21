"""
prism.machine_lock: Cross-Process Model Load Lock.
Serializes model loading across processes on the local machine using flock on ~/.prism/load.lock.
"""

import errno
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Optional

from prism.engine import ModelLoadError
from prism.paths import state_dir

DEFAULT_LOAD_TIMEOUT_S = 120.0


class MachineLockTimeoutError(ModelLoadError):
    """Raised when waiting for the cross-process load lock times out."""


def _read_holder_info(lock_path: Path) -> dict:
    try:
        if lock_path.exists():
            content = lock_path.read_text(encoding="utf-8").strip()
            if content:
                return json.loads(content)
    except Exception:
        pass
    return {}


class MachineLoadLock:
    def __init__(self, model_name: str = "", timeout: Optional[float] = None):
        self.model_name = model_name
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.environ.get("PRISM_LOAD_TIMEOUT", DEFAULT_LOAD_TIMEOUT_S))
        )
        self.fd: Optional[int] = None
        self.lock_path: Optional[Path] = None

    def __enter__(self):
        if os.environ.get("PRISM_LOAD_LOCK", "").strip().lower() in ("off", "0", "false", "no"):
            return self

        s_dir = state_dir()
        s_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = s_dir / "load.lock"

        self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        start_time = time.monotonic()
        acquired = False

        try:
            while not acquired:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    pass
                except OSError as ex:
                    if ex.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                        raise
                if not acquired:
                    if time.monotonic() - start_time >= self.timeout:
                        holder = _read_holder_info(self.lock_path)
                        holder_pid = holder.get("pid", "unknown")
                        holder_model = holder.get("model", "unknown")
                        raise MachineLockTimeoutError(
                            f"Timed out waiting for model load lock ({self.lock_path}) after {self.timeout:.1f}s. "
                            f"Held by PID {holder_pid} (loading '{holder_model}')."
                        )
                    time.sleep(0.05)
        except BaseException:
            if not acquired and self.fd is not None:
                try:
                    os.close(self.fd)
                except OSError:
                    pass
                self.fd = None
            raise

        # Write metadata about current holder
        try:
            os.ftruncate(self.fd, 0)
            os.lseek(self.fd, 0, os.SEEK_SET)
            meta = json.dumps({
                "pid": os.getpid(),
                "model": self.model_name,
                "time": time.time(),
            })
            os.write(self.fd, meta.encode("utf-8"))
            os.fsync(self.fd)
        except OSError:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            try:
                os.ftruncate(self.fd, 0)
                os.fsync(self.fd)
            except OSError:
                pass
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None


def machine_load_lock(model_name: str = "", timeout: Optional[float] = None) -> MachineLoadLock:
    return MachineLoadLock(model_name, timeout)
