"""
prism.catalog: Model Discovery, Hugging Face Downloader & Metadata Resolution.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from prism.engine import default_device
from prism.paths import default_model_dir, model_search_paths
from prism.ollama_bridge import list_ollama_models, pull_ollama_model
from prism.telemetry import get_gpu_info
from prism.templates import resolve_template

CACHE_TTL_SEC = 5.0


def planned_device(model: Dict[str, Any]) -> str:
    """The device a model will actually run on, for listings.

    The catalog's own `device` only says what the model files were exported for (a `generic-cpu` or a `cuda` variant). Prism
    ignores that: it picks the execution provider from --device / $PRISM_DEVICE, and `auto` uses CUDA whenever a GPU is present,
    so a `generic-cpu` model runs on the GPU. Ollama models keep the catalog's label.
    """
    if model.get("backend") != "onnx":
        return model.get("device", "GPU")
    try:
        requested = default_device()
    except ValueError:
        return model.get("device", "CPU")
    if requested == "cpu":
        return "CPU"
    if requested == "cuda":
        return "CUDA (GPU)"
    return "CUDA (GPU)" if get_gpu_info().get("available") else "CPU"


class AmbiguousModelError(ValueError):
    """Raised when a model name matches more than one local model."""

    def __init__(self, query: str, candidates: List[str]):
        self.query = query
        self.candidates = candidates
        super().__init__(f"Model '{query}' is ambiguous; matches: {', '.join(candidates)}")

def _onnx_variants(cuda_name: str, cuda_dir: str, cpu_name: str, cpu_dir: str) -> Dict[str, Dict[str, Any]]:
    """The cuda and cpu variants of a repo that holds one complete model per folder: local names and the folders to download."""
    return {
        "cuda": {"pattern": f"{cuda_dir}/*", "subfolder": cuda_dir, "name": cuda_name},
        "cpu": {"pattern": f"{cpu_dir}/*", "subfolder": cpu_dir, "name": cpu_name},
    }


def _microsoft_onnx_variants(name: str, gpu_quant: str, cpu_quant: str,
                             gpu_dir: str = "gpu") -> Dict[str, Dict[str, Any]]:
    """Variants for Microsoft's official ONNX Runtime GenAI repos (gpu/ and cpu_and_mobile/ subfolders)."""
    return _onnx_variants(f"{name}-cuda-gpu", f"{gpu_dir}/{gpu_quant}", f"{name}-generic-cpu", f"cpu_and_mobile/{cpu_quant}")


# Curated shortcuts. Only repos whose layout has been checked against Hugging Face belong here
# (run scripts/verify_aliases.py); any other repo can be pulled as `prism pull owner/repo`.
KNOWN_HF_MODELS = {
    "phi-4-mini": {
        "repo_id": "microsoft/Phi-4-mini-instruct-onnx",
        "variants": _microsoft_onnx_variants(
            "Phi-4-mini-instruct", "gpu-int4-rtn-block-32", "cpu-int4-rtn-block-32-acc-level-4"),
        "family": "Phi-4",
    },
    "phi-4": {
        "repo_id": "microsoft/phi-4-onnx",
        "variants": _microsoft_onnx_variants(
            "Phi-4-instruct", "gpu-int4-rtn-block-32", "cpu-int4-rtn-block-32-acc-level-4"),
        "family": "Phi-4",
    },
    "phi-4-mini-reasoning": {
        "repo_id": "microsoft/Phi-4-mini-reasoning-onnx",
        "variants": _microsoft_onnx_variants(
            "Phi-4-mini-reasoning", "gpu-int4-rtn-block-32", "cpu-int4-rtn-block-32-acc-level-4"),
        "family": "Phi-4",
    },
    "phi-4-reasoning": {
        "repo_id": "microsoft/Phi-4-reasoning-onnx",
        "variants": _microsoft_onnx_variants(
            "Phi-4-reasoning", "gpu-int4-rtn-block-32", "cpu-int4-rtn-block-32-acc-level-4"),
        "family": "Phi-4",
    },
    "phi-3-mini-4k": {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-onnx",
        "variants": _microsoft_onnx_variants(
            "Phi-3-mini-4k-instruct", "cuda-int4-rtn-block-32", "cpu-int4-rtn-block-32-acc-level-4", gpu_dir="cuda"),
        "family": "Phi-3",
    },
    "mistral-7b-instruct-v0.2": {
        "repo_id": "microsoft/mistral-7b-instruct-v0.2-ONNX",
        "variants": _onnx_variants(
            "mistral-7b-instruct-v0.2-cuda-int4-rtn-block-32",
            "onnx/cuda/mistral-7b-instruct-v0.2-cuda-int4-rtn-block-32",
            "mistral-7b-instruct-v0.2-cpu-int4-rtn-block-32-acc-level-4",
            "onnx/cpu_and_mobile/mistral-7b-instruct-v0.2-cpu-int4-rtn-block-32-acc-level-4"),
        "family": "Mistral",
    },
    "phi-3.5-mini": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct-onnx",
        "variants": _microsoft_onnx_variants(
            "Phi-3.5-mini-instruct", "gpu-int4-awq-block-128", "cpu-int4-awq-block-128-acc-level-4"),
        "family": "Phi-3.5",
    },
}


# Variants Prism cannot run, or that are for other hardware; only chosen when nothing else fits.
_FOREIGN_VARIANT_MARKERS = {"directml", "npu", "qnn", "web"}


def _path_tokens(path: str) -> set:
    """Lower-case words of a repo path (`onnx/cuda/model-cuda-int4` -> onnx, cuda, model, int4), so `npu` does not match `input`."""
    return set(re.split(r"[/_.\-]", path.lower()))


def choose_variant_dir(files: List[str], ep: str, quant: str = "int4",
                       hint: Optional[str] = None) -> "tuple[Optional[str], Optional[str]]":
    """Picks which model folder of a Hugging Face repo to download, from the repo's file list.

    Microsoft's ONNX repos hold several complete models side by side (`cuda/…`, `cpu_and_mobile/…`, `directml/…`), each folder with its own
    genai_config.json. Returns (folder, None); (None, None) when the repo is a single model at its root; or (None, error) when the choice
    is ambiguous or `hint` (a substring of the folder path, `--variant`) matches nothing. Without a hint the folder is chosen by execution
    provider, then quantization, then Microsoft's `acc-level-4` builds, and it is an error if that still leaves several.
    """
    dirs = sorted({f.rsplit("/", 1)[0] if "/" in f else "" for f in files if f.rsplit("/", 1)[-1] == "genai_config.json"})
    if "" in dirs or not dirs:
        return None, None
    candidates = dirs
    if hint:
        candidates = [d for d in dirs if hint.lower() in d.lower()]
        if not candidates:
            return None, f"no model folder matches '{hint}'; the repo has: {', '.join(dirs)}"
    elif len(dirs) > 1:
        usable = [d for d in candidates if not _path_tokens(d) & _FOREIGN_VARIANT_MARKERS] or candidates
        wanted = {"cuda", "gpu"} if ep == "cuda" else {"cpu"}
        candidates = [d for d in usable if _path_tokens(d) & wanted] or usable
        candidates = [d for d in candidates if quant in d.lower()] or candidates
        if len(candidates) > 1:
            candidates = [d for d in candidates if "acc-level-4" in d] or candidates
    if len(candidates) > 1:
        return None, f"several model folders fit: {', '.join(candidates)}; pass --variant to pick one"
    return candidates[0], None


def variant_dest_name(repo_id: str, variant_dir: str) -> str:
    """Local folder name for a downloaded variant: its own folder name, prefixed with the repo's when that would not say which model it is."""
    leaf = variant_dir.rsplit("/", 1)[-1]
    base = repo_id.split("/")[-1]
    for suffix in ("-onnx", "-ONNX"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return leaf if base.lower() in leaf.lower() else f"{base}-{leaf}"


def verify_model_dir(path: Path, ep: str, title: str = "MODEL DOWNLOAD VERIFICATION", failure: str = "Download is incomplete or is") -> bool:
    """Prints a summary of `path` and returns whether it is an ONNX Runtime GenAI model folder Prism can load
    (genai_config.json plus *.onnx weights). Shared by `prism pull` and `prism convert`."""
    has_config = (path / "genai_config.json").exists()
    has_weights = any(path.glob("*.onnx")) or any(path.glob("*.onnx.data"))
    size_bytes = sum(f.stat().st_size for f in path.glob("*") if f.is_file())
    size_mb = round(size_bytes / (1024 * 1024), 1)

    print("\n" + "=" * 60)
    print(f" 📦 {title}")
    print("=" * 60)
    print(f"  • Model Directory: {path}")
    print(f"  • Total Size:      {size_mb} MB")
    print(f"  • genai_config:    {'✅ Present' if has_config else '⚠️ Missing'}")
    print(f"  • Model Weights:   {'✅ Present' if has_weights else '⚠️ Missing'}")
    print(f"  • Target Hardware: {ep.upper()}")
    print("=" * 60)
    if not has_config or not has_weights:
        print(f"❌ {failure} not an ONNX Runtime GenAI model folder "
              "(needs genai_config.json and *.onnx weights); it cannot be loaded by Prism.\n")
        return False
    return True


class ModelCatalog:
    def __init__(self, search_paths: Optional[List[str]] = None):
        self.search_paths = search_paths if search_paths is not None else model_search_paths()
        self._onnx_cache: Optional[tuple] = None
        self._ollama_cache: Optional[tuple] = None

    def invalidate_cache(self) -> None:
        self._onnx_cache = None
        self._ollama_cache = None

    def _ollama_models(self) -> List[Dict[str, Any]]:
        now = time.monotonic()
        if self._ollama_cache and now - self._ollama_cache[0] < CACHE_TTL_SEC:
            return list(self._ollama_cache[1])
        models = list_ollama_models()
        self._ollama_cache = (now, models)
        return list(models)

    def discover_onnx_models(self) -> List[Dict[str, Any]]:
        """Finds all ONNX model folders containing genai_config.json or model.onnx (cached briefly)."""
        now = time.monotonic()
        if self._onnx_cache and now - self._onnx_cache[0] < CACHE_TTL_SEC:
            return list(self._onnx_cache[1])
        models = self._scan_onnx_models()
        self._onnx_cache = (now, models)
        return list(models)

    def _scan_onnx_models(self) -> List[Dict[str, Any]]:
        found: Dict[str, Dict[str, Any]] = {}

        for base_dir in self.search_paths:
            if not os.path.isdir(base_dir):
                continue
            for root, dirs, files in os.walk(base_dir):
                if "genai_config.json" in files or "model.onnx" in files:
                    dirs[:] = []  # a model folder holds no other models: do not walk its subfolders
                    model_dir = Path(root)
                    name = model_dir.name
                    # If folder is v1/v2/v5, use parent name
                    if name.startswith("v") and name[1:].isdigit():
                        name = f"{model_dir.parent.name}:{name}"

                    size_bytes = sum(f.stat().st_size for f in model_dir.glob("*") if f.is_file())
                    cfg_path = model_dir / "genai_config.json"
                    device = "CPU"
                    model_type = ""
                    context_length = None
                    if cfg_path.exists():
                        try:
                            cfg = json.loads(cfg_path.read_text())
                            model_type = str(cfg.get("model", {}).get("type", ""))
                            ctx = cfg.get("model", {}).get("context_length")
                            if isinstance(ctx, int) and ctx > 0:
                                context_length = ctx
                            opts = cfg.get("session_options", {}).get("provider_options", [])
                            if any("cuda" in opt for opt in opts):
                                device = "CUDA (GPU)"
                        except Exception:
                            pass
                    if "cuda" in name.lower() or "gpu" in name.lower():
                        device = "CUDA (GPU)"

                    if name not in found:
                        template, template_source = resolve_template(name, model_type, str(model_dir))
                        found[name] = {
                            "id": name,
                            "name": name,
                            "engine": "ONNX Runtime GenAI",
                            "type": "ONNX Graph",
                            "device": device,
                            "size_mb": round(size_bytes / (1024 * 1024), 1),
                            "path": str(model_dir),
                            "backend": "onnx",
                            "template": template,
                            "template_source": template_source,
                            "context_length": context_length,
                        }
        return list(found.values())

    def list_all_models(self, include_ollama: bool = True) -> List[Dict[str, Any]]:
        """Returns unified list of ONNX and Ollama models."""
        models = self.discover_onnx_models()
        if include_ollama:
            models.extend(self._ollama_models())
        return models

    def resolve_model(self, model_id_or_alias: str) -> Optional[Dict[str, Any]]:
        """
        Resolves a model id, name, path, or unique substring to model metadata.
        Exact matches win, then a curated alias resolves to its best installed variant, then a unique
        substring; a substring matching several models raises AmbiguousModelError.
        """
        query = (model_id_or_alias or "").strip()
        if not query:
            return None
        if query.startswith("ollama:"):
            return {
                "id": query,
                "name": query.replace("ollama:", ""),
                "backend": "ollama",
            }

        q = query.lower()
        onnx_models = self.discover_onnx_models()
        abs_query = os.path.abspath(query) if os.path.isdir(query) else None

        # 1. Exact id / name / path
        for m in onnx_models:
            if m["id"].lower() == q or m["name"].lower() == q or m["path"] == abs_query:
                return m
        ollama_models = self._ollama_models()
        for o in ollama_models:
            if o["name"].lower() == q:
                return o

        # 1b. A curated alias ("phi-4-mini") means the variant this machine should run, if it is installed;
        # otherwise it would be ambiguous whenever both variants (or a Foundry-cache copy) exist.
        alias = KNOWN_HF_MODELS.get(q)
        if alias:
            prefer = ("cuda", "cpu") if get_gpu_info().get("available") else ("cpu", "cuda")
            for ep in prefer:
                variant_name = alias["variants"][ep]["name"].lower()
                for m in onnx_models:
                    if m["name"].lower() == variant_name:
                        return m

        # 2. Unique substring match across ONNX names
        partial = [m for m in onnx_models if q in m["name"].lower()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            raise AmbiguousModelError(query, [m["id"] for m in partial])
        return None

    def pull_model(
        self,
        model_id_or_alias: str,
        output_dir: Optional[str] = None,
        ep: Optional[str] = None,
        quant: str = "int4",
        backend: str = "auto",
        variant: Optional[str] = None,
    ) -> Optional[str]:
        """
        Pulls models across either Hugging Face (ONNX) or Ollama (GGUF).
        """
        # 1. Route to Ollama if explicitly requested or prefixed
        if backend == "ollama" or model_id_or_alias.startswith("ollama:"):
            success = pull_ollama_model(model_id_or_alias)
            self.invalidate_cache()
            return model_id_or_alias if success else None

        # 2. Determine execution provider (CUDA vs CPU)
        if ep is None:
            gpu_info = get_gpu_info()
            ep = "cuda" if gpu_info.get("available") else "cpu"
        ep = ep.lower()

        # 3. Resolve Hugging Face repository and pattern
        alias_key = model_id_or_alias.lower()
        target_info = KNOWN_HF_MODELS.get(alias_key)

        repo_id = target_info["repo_id"] if target_info else model_id_or_alias
        subfolder = None
        pattern = None

        if target_info:
            variants = target_info.get("variants", {})
            chosen_variant = variants.get(ep) or variants.get("cuda") or variants.get("cpu")
            if chosen_variant:
                dest_name = chosen_variant.get("name")
                pattern = chosen_variant.get("pattern")
                subfolder = chosen_variant.get("subfolder")
            else:
                dest_name = f"{alias_key}-{ep}"
        else:
            dest_name = repo_id.split("/")[-1]

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("❌ 'huggingface_hub' is required to pull models from Hugging Face.")
            print("   Install via: pip install huggingface_hub")
            return None

        if not target_info:
            # A repo that keeps several models in subfolders: download the one that fits this machine, not all of them.
            try:
                from huggingface_hub import list_repo_files
                repo_files = list(list_repo_files(repo_id))
            except Exception:  # no listing (offline, old huggingface_hub, unknown repo): download the repo as it is
                repo_files = None
            if repo_files is not None:
                chosen, problem = choose_variant_dir(repo_files, ep, quant, variant)
                if problem:
                    print(f"❌ {repo_id}: {problem}")
                    return None
                if chosen:
                    subfolder, pattern = chosen, f"{chosen}/*"
                    dest_name = variant_dest_name(repo_id, chosen)

        dest_path = (Path(output_dir) if output_dir else default_model_dir()) / dest_name
        dest_path.mkdir(parents=True, exist_ok=True)

        print(f"📥 Pulling ONNX model '{repo_id}' [{ep.upper()} | {quant.upper()}] to: {dest_path}")

        kwargs: Dict[str, Any] = {
            "repo_id": repo_id,
            "local_dir": str(dest_path),
            "local_dir_use_symlinks": False,
        }
        if pattern:
            kwargs["allow_patterns"] = [pattern]
            kwargs["ignore_patterns"] = ["*.azDownload*"]  # leftovers of Microsoft's upload tooling in some repos

        try:
            snapshot_download(**kwargs)
        except Exception as ex:
            print(f"❌ Failed to download repository '{repo_id}': {ex}")
            return None

        # Move subfolder contents to root of dest_path if nested
        if subfolder:
            sub_dir = dest_path / subfolder
            if sub_dir.exists():
                for item in sub_dir.iterdir():
                    target = dest_path / item.name
                    if not target.exists():
                        item.rename(target)
                # Drop the now-empty nested folders (e.g. gpu/gpu-int4-rtn-block-32).
                cur = sub_dir
                while cur != dest_path:
                    try:
                        cur.rmdir()
                    except OSError:
                        break
                    cur = cur.parent

        # 4. Post-pull verification
        if not verify_model_dir(dest_path, ep, title="MODEL DOWNLOAD VERIFICATION", failure="Download is incomplete or is"):
            self.invalidate_cache()
            return None

        print(f"\n💡 Test your model with:")
        print(f"   prism run {dest_name} \"Write a hello world program in Python.\"\n")

        self.invalidate_cache()
        return str(dest_path)
