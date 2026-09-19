"""
foundry_ng.catalog: Model Discovery, Hugging Face Downloader & Metadata Resolution.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from foundry_ng.ollama_bridge import list_ollama_models

KNOWN_HF_MODELS = {
    "phi-4-mini": {
        "repo_id": "microsoft/Phi-4-mini-instruct-onnx",
        "pattern": "gpu/gpu-int4-rtn-block-32/*",
        "subfolder": "gpu/gpu-int4-rtn-block-32",
        "name": "Phi-4-mini-instruct-cuda-gpu",
        "family": "Phi-4",
    },
    "phi-4": {
        "repo_id": "microsoft/phi-4-onnx",
        "pattern": "gpu/gpu-int4-rtn-block-32/*",
        "subfolder": "gpu/gpu-int4-rtn-block-32",
        "name": "Phi-4-instruct-cuda-gpu",
        "family": "Phi-4",
    },
    "phi-3.5-mini": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct-onnx",
        "pattern": "gpu/gpu-int4-rtn-block-32/*",
        "subfolder": "gpu/gpu-int4-rtn-block-32",
        "name": "Phi-3.5-mini-instruct-cuda-gpu",
        "family": "Phi-3.5",
    },
    "qwen2.5-coder-7b": {
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-ONNX",
        "pattern": "*",
        "subfolder": None,
        "name": "qwen2.5-coder-7b-onnx",
        "family": "Qwen2.5",
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

    def pull_model(self, model_id_or_alias: str, output_dir: str = "models") -> str:
        """Pulls genuine ONNX models directly from Hugging Face."""
        from huggingface_hub import snapshot_download

        target_info = KNOWN_HF_MODELS.get(model_id_or_alias.lower())
        repo_id = target_info["repo_id"] if target_info else model_id_or_alias
        dest_name = target_info["name"] if target_info else repo_id.split("/")[-1]
        pattern = target_info["pattern"] if target_info else None

        dest_path = Path(output_dir) / dest_name
        dest_path.mkdir(parents=True, exist_ok=True)

        print(f"📥 Pulling model '{repo_id}' to: {dest_path}")
        kwargs: Dict[str, Any] = {
            "repo_id": repo_id,
            "local_dir": str(dest_path),
            "local_dir_use_symlinks": False,
        }
        if pattern:
            kwargs["allow_patterns"] = [pattern]

        snapshot_download(**kwargs)

        # Move subfolder contents if nested
        if target_info and target_info.get("subfolder"):
            sub_dir = dest_path / target_info["subfolder"]
            if sub_dir.exists():
                for item in sub_dir.iterdir():
                    target = dest_path / item.name
                    if not target.exists():
                        item.rename(target)

        return str(dest_path)
