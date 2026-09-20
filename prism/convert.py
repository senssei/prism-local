"""
prism.convert: Convert and quantize a Hugging Face model to an ONNX Runtime GenAI folder with its model builder.

The builder (`python -m onnxruntime_genai.models.builder`) is what Microsoft Olive's `optimize` runs for text models too, so Prism calls it
directly and needs no Olive. It is optional and heavy (it imports torch and transformers), so it runs as a subprocess of the current
Python. The result is installed next to pulled models, so `list`, `run` and `serve` treat it like any other model.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from prism.catalog import verify_model_dir
from prism.paths import default_model_dir
from prism.telemetry import get_gpu_info

EPS = ("cuda", "cpu")
QUANTS = ("int4", "fp16")
# Modules the builder imports at start-up: the wheel of onnxruntime-genai declares none of them.
REQUIRED_MODULES = {
    "onnxruntime_genai": "onnxruntime-genai",
    "torch": "torch",
    "transformers": "transformers",
    "onnx_ir": "onnx-ir",
    "safetensors": "safetensors",
}
INSTALL_HINT = 'pip install "prism-local[convert]"'


def missing_dependencies() -> List[str]:
    """pip names of the builder's requirements that are not installed in this Python (none of them is imported here: torch is slow to load)."""
    return [pip_name for module, pip_name in REQUIRED_MODULES.items() if importlib.util.find_spec(module) is None]


def default_name(source: str, ep: str, quant: str) -> str:
    """Folder name for a converted model; it contains the EP so that `prism list` reports the right device."""
    base = Path(source.rstrip("/\\")).name or "model"
    return f"{base}-{ep}-{quant}"


def hf_cache_dir() -> str:
    """Where Hugging Face keeps downloaded weights, so that they are reused between runs and never left in the working directory."""
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return hub
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def has_hf_token() -> bool:
    """Whether a Hugging Face token is configured. The builder asks for one by default, which fails even for public models without it."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.isfile(os.path.join(home, "token"))


def build_command(source: str, output_path: str, ep: str, quant: str, trust_remote_code: bool = False) -> List[str]:
    """The model builder command line (run with this Python, so it uses the environment Prism is installed in)."""
    command = [sys.executable, "-m", "onnxruntime_genai.models.builder"]
    command += ["-i", os.path.abspath(source)] if os.path.isdir(source) else ["-m", source]
    command += ["-o", output_path, "-p", quant, "-e", ep, "-c", hf_cache_dir()]
    options = []
    if trust_remote_code:
        options.append("hf_remote=true")
    if not has_hf_token():
        options.append("hf_token=false")
    if options:
        command += ["--extra_options", *options]
    return command


def find_model_dir(root: Path) -> Optional[Path]:
    """The folder under `root` that holds genai_config.json (the builder writes it to the output folder itself, but be lenient), or None."""
    for current, _dirs, files in os.walk(root):
        if "genai_config.json" in files:
            return Path(current)
    return None


def convert_model(
    source: str,
    output_dir: Optional[str] = None,
    ep: Optional[str] = None,
    quant: str = "int4",
    name: Optional[str] = None,
    force: bool = False,
    trust_remote_code: bool = False,
) -> Optional[str]:
    """Runs the model builder on `source` (a Hugging Face repo id or a local model folder) and installs the result.

    Returns the installed model folder, or None after printing why it failed.
    """
    if ep is None:
        ep = "cuda" if get_gpu_info().get("available") else "cpu"
    ep = ep.lower()
    if ep not in EPS:
        print(f"❌ --ep must be one of {', '.join(EPS)} (got '{ep}').")
        return None
    if quant not in QUANTS:
        print(f"❌ --quant must be one of {', '.join(QUANTS)} (got '{quant}').")
        return None
    if ep == "cpu" and quant == "fp16":
        print("❌ fp16 is not supported on CPU. Use --quant int4, or --ep cuda.")
        return None
    missing = missing_dependencies()
    if missing:
        print(f"❌ The ONNX Runtime GenAI model builder needs packages that are not installed in this Python environment: {', '.join(missing)}.")
        print(f"   Install them with: {INSTALL_HINT}   (onnxruntime-genai comes with the 'cuda' extra, or install it for CPU)")
        return None

    dest_name = name or default_name(source, ep, quant)
    dest_parent = Path(output_dir) if output_dir else default_model_dir()
    dest_path = dest_parent / dest_name
    if dest_path.exists() and not force:
        print(f"❌ {dest_path} already exists. Use --force to replace it, or --name to pick another folder.")
        return None

    staging = dest_parent / f".convert-{dest_name}-tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    print(f"🛠️  Converting '{source}' [{ep.upper()} | {quant.upper()}] with the ONNX Runtime GenAI model builder. This can take several minutes.")
    try:
        try:
            result = subprocess.run(build_command(source, str(staging), ep, quant, trust_remote_code))
        except OSError as ex:
            print(f"❌ Could not run the model builder: {ex}")
            return None
        except KeyboardInterrupt:
            print("\n❌ Conversion interrupted.")
            return None
        if result.returncode != 0:
            print(f"❌ The model builder failed (exit code {result.returncode}). See its output above.")
            return None

        model_dir = find_model_dir(staging)
        if model_dir is None:
            print("❌ The model builder finished but produced no genai_config.json.")
            return None
        if not verify_model_dir(model_dir, ep, title="MODEL CONVERSION VERIFICATION", failure="The builder produced a folder that is"):
            return None

        if dest_path.exists():
            shutil.rmtree(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(model_dir), str(dest_path))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print("\n💡 Test your model with:")
    print(f"   prism run {dest_name} \"Write a hello world program in Python.\"\n")
    return str(dest_path)
