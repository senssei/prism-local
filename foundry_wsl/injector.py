"""
foundry_wsl.injector: Automated Model Cache Injector for Microsoft Foundry Local
Configures genuine CUDA weights, genai_config.json session options, and cache metadata.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

def configure_cuda_genai_config(genai_config_path: Path, device_id: int = 0) -> bool:
    """Updates genai_config.json to inject CUDA provider options."""
    if not genai_config_path.exists():
        return False
    try:
        data = json.loads(genai_config_path.read_text())
        if "session_options" not in data:
            data["session_options"] = {}
        data["session_options"]["log_id"] = "onnxruntime-genai"
        data["session_options"]["provider_options"] = [
            {"cuda": {"device_id": str(device_id)}}
        ]
        genai_config_path.write_text(json.dumps(data, indent=4))
        return True
    except Exception:
        return False

def create_inference_model_json(target_path: Path, model_name: str) -> bool:
    """Creates a basic inference_model.json template."""
    content = {
        "Name": model_name,
        "PromptTemplate": {
            "system": "<|system|>\n{Content}<|end|>",
            "user": "<|user|>\n{Content}<|end|>",
            "assistant": "<|assistant|>\n{Content}<|end|>",
            "prompt": "<|user|>\n{Content}<|end|>\n<|assistant|>"
        }
    }
    try:
        target_path.write_text(json.dumps(content, indent=2))
        return True
    except Exception:
        return False

def mark_model_cached_in_modelinfo(cache_dir: Path, model_id: str) -> bool:
    """Marks model as cached in foundry.modelinfo.json."""
    info_file = cache_dir / "models/foundry.modelinfo.json"
    if not info_file.exists():
        return False
    try:
        data = json.loads(info_file.read_text())
        modified = False
        for item in data.get("models", []):
            if item.get("id") == model_id or item.get("alias") == model_id:
                item["cached"] = True
                modified = True
        if modified:
            info_file.write_text(json.dumps(data, indent=2))
            return True
    except Exception:
        pass
    return False
