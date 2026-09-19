#!/usr/bin/env python3
"""
Single place to change the GitHub repository slug (OWNER/NAME).

`project.slug` holds the current slug. This script rewrites every reference to it (repo URLs, badges, Pages URL,
pyproject URLs, mkdocs settings) in tracked and untracked-but-not-ignored text files.

    python3 scripts/set_slug.py OWNER/NAME    # rename everywhere and update project.slug
    python3 scripts/set_slug.py --check       # exit 1 if project.slug is not used consistently
    python3 scripts/set_slug.py --list        # show which files reference the slug

The GitHub Pages URL is derived as https://<owner lowercased>.github.io/<name>/.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
SLUG_FILE = ROOT / "project.slug"
SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
# Files that must reference the slug; --check fails if a partial rename left any without it.
MUST_REFERENCE = ["README.md", "pyproject.toml", "mkdocs.yml", "SECURITY.md"]


def read_slug() -> str:
    slug = SLUG_FILE.read_text().strip()
    if not SLUG_RE.match(slug):
        raise SystemExit(f"project.slug must look like OWNER/NAME, got {slug!r}")
    return slug


def pages_form(slug: str) -> str:
    owner, name = slug.split("/")
    return f"{owner.lower()}.github.io/{name}"


def _patterns(slug: str) -> List[re.Pattern]:
    # Not followed by a name character, so OWNER/NAME never matches inside OWNER/NAME-extra.
    forms = {slug, pages_form(slug)}
    return [re.compile(re.escape(f) + r"(?![\w.-])") for f in sorted(forms, key=len, reverse=True)]


def project_files(root: Path = ROOT) -> List[Path]:
    """Tracked plus untracked-not-ignored files (falls back to a filesystem walk outside git)."""
    try:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root, check=True,
                             capture_output=True).stdout.decode()
        names = [n for n in out.split("\0") if n]
    except (OSError, subprocess.CalledProcessError):
        names = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    return [root / n for n in names if (root / n).is_file() and n != "project.slug"]


def _read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None  # binary or unreadable


def references(slug: str, root: Path = ROOT) -> List[Path]:
    pats = _patterns(slug)
    hits = []
    for path in project_files(root):
        text = _read_text(path)
        if text is not None and any(p.search(text) for p in pats):
            hits.append(path)
    return hits


def rename(old: str, new: str, root: Path = ROOT) -> List[Path]:
    """Rewrites `old` (and its Pages form) to `new`. Returns the files changed."""
    replacements: List[Tuple[re.Pattern, str]] = [
        (re.compile(re.escape(pages_form(old)) + r"(?![\w.-])"), pages_form(new)),
        (re.compile(re.escape(old) + r"(?![\w.-])"), new),
    ]
    changed = []
    for path in project_files(root):
        text = _read_text(path)
        if text is None:
            continue
        updated = text
        for pat, repl in replacements:
            updated = pat.sub(repl, updated)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(path)
    (root / "project.slug").write_text(new + "\n", encoding="utf-8")
    return changed


def check(root: Path = ROOT) -> List[str]:
    slug = read_slug()
    problems = [f"{name} does not reference {slug}" for name in MUST_REFERENCE
                if not (root / name).is_file() or not any(p.search((root / name).read_text()) for p in _patterns(slug))]
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug", nargs="?", help="new OWNER/NAME")
    ap.add_argument("--check", action="store_true", help="verify project.slug is used consistently")
    ap.add_argument("--list", action="store_true", help="list files that reference the current slug")
    args = ap.parse_args()

    current = read_slug()
    if args.check:
        problems = check()
        for p in problems:
            print("FAIL", p)
        print("ok" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    if args.list or not args.slug:
        for path in references(current):
            print(path.relative_to(ROOT))
        return 0
    if not SLUG_RE.match(args.slug):
        print(f"error: slug must look like OWNER/NAME, got {args.slug!r}", file=sys.stderr)
        return 2
    if args.slug == current:
        print(f"already {current}")
        return 0
    changed = rename(current, args.slug)
    print(f"{current} -> {args.slug}: updated {len(changed)} file(s)")
    for path in changed:
        print("  ", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
