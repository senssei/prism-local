import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "set_slug.py"
spec = importlib.util.spec_from_file_location("set_slug", SCRIPT)
set_slug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(set_slug)


class TestRepoSlug(unittest.TestCase):
    def test_project_slug_is_valid_and_used_consistently(self):
        self.assertRegex(set_slug.read_slug(), set_slug.SLUG_RE)
        self.assertEqual(set_slug.check(), [], "run: python3 scripts/set_slug.py --check")


class TestRename(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "project.slug").write_text("Acme/widget\n")
        (self.tmp / "README.md").write_text(
            "[ci](https://github.com/Acme/widget/actions) docs https://acme.github.io/widget/ "
            "other https://github.com/Acme/widget-extra and https://github.com/microsoft/foundry-local\n")
        (self.tmp / "blob.bin").write_bytes(b"\xff\xfe Acme/widget \x00")
        self.orig = set_slug.project_files
        set_slug.project_files = lambda root=self.tmp: [self.tmp / "README.md", self.tmp / "blob.bin"]
        self.addCleanup(setattr, set_slug, "project_files", self.orig)

    def test_rewrites_repo_and_pages_urls_only(self):
        changed = set_slug.rename("Acme/widget", "NewOrg/gadget", self.tmp)
        text = (self.tmp / "README.md").read_text()
        self.assertEqual([p.name for p in changed], ["README.md"])
        self.assertIn("https://github.com/NewOrg/gadget/actions", text)
        self.assertIn("https://neworg.github.io/gadget/", text)      # owner lowercased for Pages
        self.assertIn("https://github.com/Acme/widget-extra", text)  # a longer name is not touched
        self.assertIn("microsoft/foundry-local", text)
        self.assertEqual((self.tmp / "project.slug").read_text().strip(), "NewOrg/gadget")
        self.assertEqual((self.tmp / "blob.bin").read_bytes(), b"\xff\xfe Acme/widget \x00")  # binary untouched

    def test_rename_is_reversible(self):
        original = (self.tmp / "README.md").read_text()
        set_slug.rename("Acme/widget", "NewOrg/gadget", self.tmp)
        set_slug.rename("NewOrg/gadget", "Acme/widget", self.tmp)
        self.assertEqual((self.tmp / "README.md").read_text(), original)


if __name__ == "__main__":
    unittest.main()
