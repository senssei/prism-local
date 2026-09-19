"""
foundry_ng.engine: Native ONNX Runtime GenAI Execution Engine on NVIDIA CUDA.
Delivers direct GPU acceleration, streaming token generation, and clean memory lifecycle.
"""

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from foundry_ng.telemetry import bootstrap_cuda_env

# Ensure CUDA paths are set before ONNX Runtime loads
bootstrap_cuda_env()

try:
    import onnxruntime_genai as og
    OG_AVAILABLE = True
except ImportError:
    og = None
    OG_AVAILABLE = False


def format_prompt(messages: List[Dict[str, str]], template: str = "phi4") -> str:
    """Formats a list of OpenAI-style messages into the appropriate prompt format."""
    system_msg = ""
    turns: List[Tuple[str, str]] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_msg = content
        else:
            turns.append((role, content))

    if template in ["phi4", "phi3"]:
        out = ""
        if system_msg:
            out += f"<|system|>\n{system_msg}<|end|>\n"
        for role, content in turns:
            out += f"<|{role}|>\n{content}<|end|>\n"
        out += "<|assistant|>\n"
        return out
    elif template == "chatml":
        out = ""
        if system_msg:
            out += f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        for role, content in turns:
            out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        out += "<|im_start|>assistant\n"
        return out
    else:
        out = f"System: {system_msg}\n\n" if system_msg else ""
        for role, content in turns:
            out += f"{role.capitalize()}: {content}\n"
        out += "Assistant: "
        return out


class OnnxGenAiEngine:
    """Wraps onnxruntime_genai for direct CUDA execution."""

    def __init__(self, model_path: str):
        if not OG_AVAILABLE:
            raise RuntimeError("onnxruntime_genai is not installed in the current environment.")

        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        """Initializes og.Model and og.Tokenizer."""
        try:
            self._model = og.Model(self.model_path)
            self._tokenizer = og.Tokenizer(self._model)
        except Exception as ex:
            raise RuntimeError(f"Failed to load ONNX model at {self.model_path}: {ex}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Runs non-streaming generation and returns text with performance metrics."""
        tokens: List[str] = []
        ttft = 0.0
        t0 = time.perf_counter()

        for chunk, is_first, speed in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            if is_first:
                ttft = time.perf_counter() - t0
            tokens.append(chunk)

        total_time = max(time.perf_counter() - t0, 1e-6)
        token_count = len(tokens)
        decode_speed = token_count / total_time

        return {
            "text": "".join(tokens),
            "tokens_generated": token_count,
            "ttft_sec": round(ttft, 4),
            "elapsed_sec": round(total_time, 3),
            "decode_tok_per_sec": round(decode_speed, 1),
        }

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> Iterator[Tuple[str, bool, float]]:
        """
        Yields (token_text, is_first_token, current_tokens_per_sec).
        """
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model is not loaded.")

        input_tokens = self._tokenizer.encode(prompt)
        prompt_len = len(input_tokens)

        params = og.GeneratorParams(self._model)
        search_kwargs: Dict[str, Any] = {"max_length": prompt_len + max_tokens}
        if temperature > 0.0:
            search_kwargs["temperature"] = temperature
            search_kwargs["top_p"] = top_p
            search_kwargs["do_sample"] = True
        else:
            search_kwargs["do_sample"] = False

        params.set_search_options(**search_kwargs)

        generator = og.Generator(self._model, params)
        generator.append_tokens(input_tokens)
        tokenizer_stream = self._tokenizer.create_stream()

        first = True
        count = 0
        t_start = time.perf_counter()

        try:
            while not generator.is_done():
                generator.generate_next_token()
                next_tokens = generator.get_next_tokens()
                if next_tokens:
                    text = tokenizer_stream.decode(next_tokens[0])
                    count += 1
                    elapsed = max(time.perf_counter() - t_start, 1e-6)
                    speed = count / elapsed
                    yield text, first, speed
                    first = False
        finally:
            del generator
            del params
            del tokenizer_stream

    def unload(self) -> None:
        """Deterministically releases model and GPU allocations."""
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if self._model is not None:
            del self._model
            self._model = None

        gc.collect()
