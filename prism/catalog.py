"""
prism.catalog: Model Discovery, Hugging Face Downloader & Metadata Resolution.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from prism.ollama_bridge import list_ollama_models, pull_ollama_model
from prism.telemetry import get_gpu_info

KNOWN_HF_MODELS = {
    "phi-4-mini": {
        "repo_id": "microsoft/Phi-4-mini-instruct-onnx",
        "variants": {
            "cuda": {
                "pattern": "gpu/gpu-int4-rtn-block-32/*",
                "subfolder": "gpu/gpu-int4-rtn-block-32",
                "name": "Phi-4-mini-instruct-cuda-gpu",
            },
            "cpu": {
                "pattern": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*",
                "subfolder": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
                "name": "Phi-4-mini-instruct-generic-cpu",
            },
        },
        "family": "Phi-4",
    },
    "phi-4": {
        "repo_id": "microsoft/phi-4-onnx",
        "variants": {
            "cuda": {
                "pattern": "gpu/gpu-int4-rtn-block-32/*",
                "subfolder": "gpu/gpu-int4-rtn-block-32",
                "name": "Phi-4-instruct-cuda-gpu",
            },
            "cpu": {
                "pattern": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*",
                "subfolder": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
                "name": "Phi-4-instruct-generic-cpu",
            },
        },
        "family": "Phi-4",
    },
    "phi-3.5-mini": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct-onnx",
        "variants": {
            "cuda": {
                "pattern": "gpu/gpu-int4-rtn-block-32/*",
                "subfolder": "gpu/gpu-int4-rtn-block-32",
                "name": "Phi-3.5-mini-instruct-cuda-gpu",
            },
            "cpu": {
                "pattern": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*",
                "subfolder": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
                "name": "Phi-3.5-mini-instruct-generic-cpu",
            },
        },
        "family": "Phi-3.5",
    },
    "qwen2.5-coder-7b": {
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-ONNX",
        "variants": {
            "cuda": {
                "pattern": "*",
                "subfolder": None,
                "name": "qwen2.5-coder-7b-onnx",
            },
            "cpu": {
                "pattern": "*",
                "subfolder": None,
                "name": "qwen2.5-coder-7b-onnx",
            },
        },
        "family": "Qwen2.5",
    },
    "qwen2.5-coder-1.5b": {
        "repo_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct-ONNX",
        "variants": {
            "cuda": {
                "pattern": "*",
                "subfolder": None,
                "name": "qwen2.5-coder-1.5b-onnx",
            },
            "cpu": {
                "pattern": "*",
                "subfolder": None,
                "name": "qwen2.5-coder-1.5b-onnx",
            },
        },
        "family": "Qwen2.5",
    },
    "deepseek-r1-distill-qwen-7b": {
        "repo_id": "onnx-community/DeepSeek-R1-Distill-Qwen-7B-ONNX",
        "variants": {
            "cuda": {
                "pattern": "*",
                "subfolder": None,
                "name": "deepseek-r1-distill-qwen-7b-onnx",
            },
            "cpu": {
                "pattern": "*",
                "subfolder": None,
                "name": "deepseek-r1-distill-qwen-7b-onnx",
            },
        },
        "family": "DeepSeek-R1",
    },
    "llama-3.2-3b-instruct": {
        "repo_id": "microsoft/Llama-3.2-3B-Instruct-ONNX",
        "variants": {
            "cuda": {
                "pattern": "gpu/gpu-int4-rtn-block-32/*",
                "subfolder": "gpu/gpu-int4-rtn-block-32",
                "name": "llama-3.2-3b-instruct-cuda-gpu",
            },
            "cpu": {
                "pattern": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*",
                "subfolder": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
                "name": "llama-3.2-3b-instruct-generic-cpu",
            },
        },
        "family": "Llama-3.2",
    },
}


class ModelCatalog:
    def __init__(self, search_paths: Optional[List[str]] = None):
        home = str(Path.home())
        default_paths = [
            os.path.abspath("models"),
            os.path.abspath("../02-ollama-loadtest/models"),
            f"{home}/.foundry/cache/models/Microsoft",
            f"{home}/.foundry/cache/models",
        ]
        self.search_paths = search_paths or [p for p in default_paths if os.path.isdir(p)]

    def discover_onnx_models(self) -> List[Dict[str, Any]]:
        """Finds all ONNX model folders containing genai_config.json or model.onnx."""
        found: Dict[str, Dict[str, Any]] = {}

        for base_dir in self.search_paths:
            if not os.path.isdir(base_dir):
                continue
            for root, dirs, files in os.walk(base_dir):
                if "genai_config.json" in files or "model.onnx" in files:
                    model_dir = Path(root)
                    name = model_dir.name
                    # If folder is v1/v2/v5, use parent name
                    if name.startswith("v") and name[1:].isdigit():
                        name = f"{model_dir.parent.name}:{name}"

                    size_bytes = sum(f.stat().st_size for f in model_dir.glob("*") if f.is_file())
                    cfg_path = model_dir / "genai_config.json"
                    device = "CPU"
                    if cfg_path.exists():
                        try:
                            cfg = json.loads(cfg_path.read_text())
                            opts = cfg.get("session_options", {}).get("provider_options", [])
                            if any("cuda" in opt for opt in opts):
                                device = "CUDA (GPU)"
                        except Exception:
                            pass
                    if "cuda" in name.lower() or "gpu" in name.lower():
                        device = "CUDA (GPU)"

                    if name not in found:
                        found[name] = {
                            "id": name,
                            "name": name,
                            "engine": "ONNX Runtime GenAI",
                            "type": "ONNX Graph",
                            "device": device,
                            "size_mb": round(size_bytes / (1024 * 1024), 1),
                            "path": str(model_dir),
                            "backend": "onnx",
                        }
        return list(found.values())

    def list_all_models(self, include_ollama: bool = True) -> List[Dict[str, Any]]:
        """Returns unified list of ONNX and Ollama models."""
        models = self.discover_onnx_models()
        if include_ollama:
            models.extend(list_ollama_models())
        return models

    def resolve_model(self, model_id_or_alias: str) -> Optional[Dict[str, Any]]:
        """Resolves model alias or path to model metadata."""
        if model_id_or_alias.startswith("ollama:"):
            return {
                "id": model_id_or_alias,
                "name": model_id_or_alias.replace("ollama:", ""),
                "backend": "ollama",
            }

        # Check local discovered models
        for m in self.discover_onnx_models():
            if (
                m["id"].lower() == model_id_or_alias.lower()
                or model_id_or_alias.lower() in m["name"].lower()
                or model_id_or_alias.lower() in m["path"].lower()
            ):
                return m

        # Check if it's an Ollama model without prefix
        for o in list_ollama_models():
            if o["name"].lower() == model_id_or_alias.lower():
                return o

        return None

    def pull_model(
        self,
        model_id_or_alias: str,
        output_dir: str = "models",
        ep: Optional[str] = None,
        quant: str = "int4",
        backend: str = "auto",
    ) -> Optional[str]:
        """
        Pulls models across either Hugging Face (ONNX) or Ollama (GGUF).
        """
        # 1. Route to Ollama if explicitly requested or prefixed
        if backend == "ollama" or model_id_or_alias.startswith("ollama:"):
            success = pull_ollama_model(model_id_or_alias)
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
            variant = variants.get(ep) or variants.get("cuda") or variants.get("cpu")
            if variant:
                dest_name = variant.get("name")
                pattern = variant.get("pattern")
                subfolder = variant.get("subfolder")
            else:
                dest_name = f"{alias_key}-{ep}"
        else:
            dest_name = repo_id.split("/")[-1]

        dest_path = Path(output_dir) / dest_name
        dest_path.mkdir(parents=True, exist_ok=True)

        print(f"📥 Pulling ONNX model '{repo_id}' [{ep.upper()} | {quant.upper()}] to: {dest_path}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("❌ 'huggingface_hub' is required to pull models from Hugging Face.")
            print("   Install via: pip install huggingface_hub")
            return None

        kwargs: Dict[str, Any] = {
            "repo_id": repo_id,
            "local_dir": str(dest_path),
            "local_dir_use_symlinks": False,
        }
        if pattern:
            kwargs["allow_patterns"] = [pattern]

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

        # 4. Post-pull verification
        has_config = (dest_path / "genai_config.json").exists()
        has_weights = any(dest_path.glob("*.onnx")) or any(dest_path.glob("*.onnx.data"))
        size_bytes = sum(f.stat().st_size for f in dest_path.glob("*") if f.is_file())
        size_mb = round(size_bytes / (1024 * 1024), 1)

        print("\n" + "=" * 60)
        print(" 📦 MODEL DOWNLOAD VERIFICATION")
        print("=" * 60)
        print(f"  • Model Directory: {dest_path}")
        print(f"  • Total Size:      {size_mb} MB")
        print(f"  • genai_config:    {'✅ Present' if has_config else '⚠️ Missing'}")
        print(f"  • Model Weights:   {'✅ Present' if has_weights else '⚠️ Missing'}")
        print(f"  • Target Hardware: {ep.upper()}")
        print("=" * 60)
        print(f"\n💡 Test your model with:")
        print(f"   prism run {dest_name} \"Write a hello world program in Python.\"\n")

        return str(dest_path)
