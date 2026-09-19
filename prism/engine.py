"""
prism.engine: Native ONNX Runtime GenAI Execution Engine on NVIDIA CUDA.
Delivers direct GPU acceleration, streaming token generation, and clean memory lifecycle.
"""

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple

from prism.telemetry import bootstrap_cuda_env
from prism.templates import format_prompt  # noqa: F401  (re-exported for callers)

# Ensure CUDA paths are set before ONNX Runtime loads
bootstrap_cuda_env()

try:
    import onnxruntime_genai as og
    OG_AVAILABLE = True
except ImportError:
    og = None
    OG_AVAILABLE = False


class Engine(Protocol):
    """What the server, chat and MCP layers need from an inference engine."""

    last_finish_reason: str

    def count_tokens(self, text: str) -> int: ...

    def generate(
        self, prompt: str, max_tokens: int = ..., temperature: float = ..., top_p: float = ...
    ) -> Dict[str, Any]: ...

    def stream_generate(
        self, prompt: str, max_tokens: int = ..., temperature: float = ..., top_p: float = ...
    ) -> Iterator[Tuple[str, bool, float]]: ...

    def unload(self) -> None: ...


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
        # "stop" (EOS) or "length" (hit max_tokens) for the most recent completed generation.
        self.last_finish_reason = "stop"
        self._load_model()

    def _load_model(self):
        """Initializes og.Model and og.Tokenizer."""
        try:
            self._model = og.Model(self.model_path)
            self._tokenizer = og.Tokenizer(self._model)
        except Exception as ex:
            raise RuntimeError(f"Failed to load ONNX model at {self.model_path}: {ex}")

    def count_tokens(self, text: str) -> int:
        """Returns the number of tokens `text` encodes to."""
        if not self._tokenizer:
            raise RuntimeError("Model is not loaded.")
        return len(self._tokenizer.encode(text))

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
            "finish_reason": self.last_finish_reason,
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

        self.last_finish_reason = "stop"
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
            if count >= max_tokens:
                self.last_finish_reason = "length"
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
