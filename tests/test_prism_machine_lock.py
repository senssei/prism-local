import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from prism.engine import ModelLoadError
from prism.machine_lock import (
    MachineLockTimeoutError,
    machine_load_lock,
)
from prism.paths import state_dir


class TestMachineLock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state_dir_path = Path(self.tmp.name)
        self.env_patcher = patch.dict(os.environ, {"PRISM_STATE_DIR": str(self.state_dir_path)})
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        self.tmp.cleanup()

    def test_state_dir_override(self):
        self.assertEqual(state_dir(), self.state_dir_path)

    def test_lock_acquire_and_release(self):
        with machine_load_lock("model-a", timeout=1.0):
            lock_file = self.state_dir_path / "load.lock"
            self.assertTrue(lock_file.exists())

    def test_lock_conflict_and_timeout(self):
        with machine_load_lock("model-first", timeout=1.0):
            # Attempt to acquire the lock for a second model with a short timeout
            with self.assertRaises(MachineLockTimeoutError) as ctx:
                with machine_load_lock("model-second", timeout=0.1):
                    pass
            msg = str(ctx.exception)
            self.assertIn("model-first", msg)
            self.assertIn(str(os.getpid()), msg)
            self.assertTrue(issubclass(MachineLockTimeoutError, ModelLoadError))

    def test_lock_release_allows_subsequent_acquire(self):
        with machine_load_lock("model-a", timeout=1.0):
            pass
        # Should acquire without error
        with machine_load_lock("model-b", timeout=1.0):
            pass

    def test_subprocess_holder_releases_on_exit(self):
        # Launch a python subprocess that signals when lock is acquired, then holds it for 0.3s
        sub_code = (
            "import os\n"
            "import sys\n"
            "import time\n"
            "from prism.machine_lock import machine_load_lock\n"
            "with machine_load_lock('sub-model', timeout=2.0):\n"
            "    sys.stdout.write('LOCKED\\n')\n"
            "    sys.stdout.flush()\n"
            "    time.sleep(0.3)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", sub_code],
            stdout=subprocess.PIPE,
            env={**os.environ, "PRISM_STATE_DIR": str(self.state_dir_path)},
        )
        # Deterministically wait until subprocess has acquired the lock
        line = proc.stdout.readline()
        self.assertEqual(line.strip(), b"LOCKED")

        # Now parent tries to acquire; blocks until subprocess exits (~0.3s)
        t0 = time.monotonic()
        with machine_load_lock("parent-model", timeout=2.0):
            elapsed = time.monotonic() - t0
            self.assertGreaterEqual(elapsed, 0.15)
        proc.stdout.close()
        proc.wait()
        self.assertEqual(proc.returncode, 0)

    def test_unexpected_oserror_raises_immediately_without_waiting_timeout(self):
        import errno
        with patch("fcntl.flock", side_effect=OSError(errno.EBADF, "Bad file descriptor")):
            t0 = time.monotonic()
            with self.assertRaises(OSError) as ctx:
                with machine_load_lock("err-model", timeout=10.0):
                    pass
            elapsed = time.monotonic() - t0
            self.assertEqual(ctx.exception.errno, errno.EBADF)
            self.assertLess(elapsed, 1.0)

    def test_keyboard_interrupt_cleans_up_fd(self):
        with patch("fcntl.flock", side_effect=BlockingIOError), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            lock = machine_load_lock("sig-model", timeout=5.0)
            with self.assertRaises(KeyboardInterrupt):
                lock.__enter__()
            self.assertIsNone(lock.fd)


if __name__ == "__main__":
    unittest.main()
