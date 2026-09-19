import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prism import paths
from prism.catalog import ModelCatalog


class TestPrismPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.a = os.path.join(self.tmp, "a")
        self.b = os.path.join(self.tmp, "b")
        os.makedirs(self.a)
        os.makedirs(self.b)

    def test_default_model_dir_without_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRISM_MODEL_DIRS", None)
            self.assertEqual(paths.default_model_dir(), Path.home() / ".prism" / "models")

    def test_env_dirs_take_priority_and_first_is_pull_target(self):
        env = {"PRISM_MODEL_DIRS": os.pathsep.join([self.a, self.b, "/does/not/exist"])}
        with patch.dict(os.environ, env):
            self.assertEqual(paths.default_model_dir(), Path(self.a))
            found = paths.model_search_paths()
            self.assertEqual(found[:2], [self.a, self.b])
            self.assertNotIn("/does/not/exist", found)

    def test_search_paths_are_deduplicated(self):
        env = {"PRISM_MODEL_DIRS": os.pathsep.join([self.a, self.a])}
        with patch.dict(os.environ, env):
            self.assertEqual(paths.model_search_paths().count(self.a), 1)

    def test_catalog_uses_env_dirs_and_no_personal_paths(self):
        with patch.dict(os.environ, {"PRISM_MODEL_DIRS": self.a}):
            self.assertIn(self.a, ModelCatalog().search_paths)
        with patch.dict(os.environ):
            os.environ.pop("PRISM_MODEL_DIRS", None)  # do not inherit the developer's real setting
            self.assertFalse(any("02-ollama-loadtest" in p for p in ModelCatalog().search_paths))

    def test_pull_defaults_to_default_model_dir(self):
        catalog = ModelCatalog(search_paths=[])
        calls = {}

        def fake_download(**kwargs):
            calls.update(kwargs)

        with patch.dict(os.environ, {"PRISM_MODEL_DIRS": self.a}), \
                patch.dict("sys.modules", {"huggingface_hub": type("M", (), {"snapshot_download": staticmethod(fake_download)})}):
            catalog.pull_model("phi-4-mini", ep="cpu")
        self.assertTrue(calls["local_dir"].startswith(self.a))


if __name__ == "__main__":
    unittest.main()
