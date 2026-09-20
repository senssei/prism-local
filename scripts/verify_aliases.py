#!/usr/bin/env python3
"""
Checks every curated alias in prism.catalog.KNOWN_HF_MODELS against Hugging Face: the repo must exist and each variant's
pattern must match a complete model folder: genai_config.json, an .onnx file and a tokenizer. Needs network access; exits 1 if any alias is broken.

    PYTHONPATH=. python3 scripts/verify_aliases.py
"""

import fnmatch
import json
import sys
import urllib.error
import urllib.request

from prism.catalog import KNOWN_HF_MODELS


def repo_files(repo_id: str):
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=1"
    req = urllib.request.Request(url, headers={"User-Agent": "prism-verify-aliases"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return [e["path"] for e in json.load(resp) if e["type"] == "file"]


def main() -> int:
    bad = 0
    for alias, info in KNOWN_HF_MODELS.items():
        repo = info["repo_id"]
        try:
            files = repo_files(repo)
        except (urllib.error.URLError, OSError) as ex:
            print(f"FAIL {alias:20} {repo}: {ex}")
            bad += 1
            continue
        for ep, variant in info["variants"].items():
            matched = [f for f in files if fnmatch.fnmatch(f, variant["pattern"])]
            names = [f.rsplit("/", 1)[-1] for f in matched]
            ok = ("genai_config.json" in names and any(n.endswith(".onnx") for n in names)
                  and ("tokenizer.json" in names or "tokenizer.model" in names))
            print(f"{'ok  ' if ok else 'FAIL'} {alias:20} {ep:5} {len(matched):3} files  {repo}")
            bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
