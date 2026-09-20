"""
prism.engine: Native ONNX Runtime GenAI Execution Engine on NVIDIA CUDA.
Delivers direct GPU acceleration, streaming token generation, and clean memory lifecycle.
"""

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple

from prism.telemetry import bootstrap_cuda_env, get_gpu_info
from prism.templates import format_prompt  # noqa: F401  (re-exported for callers)
from prism.tools import TOOL_MARKERS

logger = logging.getLogger("prism.engine")

DEVICES = ("auto", "cuda", "cpu")

# Ensure CUDA paths are set before ONNX Runtime loads
bootstrap_cuda_env()

try:
    import onnxruntime_genai as og
    OG_AVAILABLE = True
except ImportError:
    og = None
    OG_AVAILABLE = False


class ModelLoadError(RuntimeError):
    """The model could not be loaded (bad files, or the requested execution provider is unavailable)."""


def default_device() -> str:
    """Requested execution provider: $PRISM_DEVICE (auto|cuda|cpu), default auto."""
    value = os.environ.get("PRISM_DEVICE", "auto").strip().lower() or "auto"
    if value not in DEVICES:
        raise ValueError(f"PRISM_DEVICE must be one of {', '.join(DEVICES)} (got '{value}')")
    return value


DEFAULT_PREFILL_CHUNK = 1024


def prefill_chunk_tokens() -> Optional[int]:
    """Prompt tokens processed per step: $PRISM_PREFILL_CHUNK (a positive integer), default 1024; 0 or "off" processes the whole prompt at once.

    ONNX Runtime GenAI's GPU memory otherwise grows with the prompt (about 1.4 MB per token for Phi-4-mini) and a large part of it is
    still held after the model is unloaded, which starves the next model that is loaded. Chunking bounds both.
    """
    raw = os.environ.get("PRISM_PREFILL_CHUNK", "").strip()
    if not raw:
        return DEFAULT_PREFILL_CHUNK
    if raw.lower() == "off":
        return None
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(f"PRISM_PREFILL_CHUNK must be a positive integer, 0 or 'off' (got '{raw}')")
    return value or None


def read_eos_token_ids(model_path: str) -> frozenset:
    """End-of-sequence token ids from the model's genai_config.json (`model.eos_token_id`: an int or a list); empty when unknown."""
    try:
        with open(os.path.join(model_path, "genai_config.json"), encoding="utf-8") as f:
            value = json.load(f).get("model", {}).get("eos_token_id")
    except (OSError, ValueError, AttributeError):
        return frozenset()
    values = value if isinstance(value, list) else [value]
    return frozenset(v for v in values if isinstance(v, int) and not isinstance(v, bool))


def read_tool_marker_tokens(model_path: str) -> Dict[int, str]:
    """{token id: text} for the tool-call marker tokens (`<tool_call>`, `[TOOL_CALLS]`, ...) that the model's tokenizer.json defines."""
    try:
        with open(os.path.join(model_path, "tokenizer.json"), encoding="utf-8") as f:
            added = json.load(f).get("added_tokens") or []
    except (OSError, ValueError, AttributeError):
        return {}
    return {t["id"]: t["content"] for t in added
            if isinstance(t, dict) and t.get("content") in TOOL_MARKERS and isinstance(t.get("id"), int)}


DEFAULT_TOP_K = 40

# A model that is stuck repeats one short cycle of tokens until max_tokens. A cycle is reported when the tail of the output repeats
# with a period of at most LOOP_MAX_PERIOD tokens over at least LOOP_MIN_SPAN tokens and LOOP_MIN_REPEATS repetitions, which ordinary
# text and code (long lists, tables, repeated lines) do not reach.
LOOP_MAX_PERIOD = 64
LOOP_MIN_SPAN = 200
LOOP_MIN_REPEATS = 12
LOOP_CHECK_EVERY = 16


def read_config_top_k(model_path: str) -> int:
    """`search.top_k` from the model's genai_config.json when it is above 1, else DEFAULT_TOP_K.

    Most genai_config files ship `top_k: 1`, and with it ONNX Runtime GenAI ignores `temperature` and `top_p` altogether (every request is
    greedy), so a top_k of 1 cannot be the default for sampling.
    """
    try:
        with open(os.path.join(model_path, "genai_config.json"), encoding="utf-8") as f:
            value = json.load(f).get("search", {}).get("top_k")
    except (OSError, ValueError, AttributeError):
        return DEFAULT_TOP_K
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 1 else DEFAULT_TOP_K


def loop_guard_enabled() -> bool:
    """$PRISM_LOOP_GUARD: `off` or `0` lets a repeating model run to max_tokens; anything else (default) stops it once it is clearly stuck."""
    return os.environ.get("PRISM_LOOP_GUARD", "").strip().lower() not in ("off", "0")


def find_token_cycle(tokens: List[int]) -> Optional[int]:
    """The period of the token cycle that `tokens` ends in (see LOOP_* above), or None."""
    n = len(tokens)
    for period in range(1, LOOP_MAX_PERIOD + 1):
        span = max(LOOP_MIN_SPAN, LOOP_MIN_REPEATS * period)
        if n < span:
            break  # a longer period needs a longer span
        if all(tokens[i] == tokens[i - period] for i in range(n - 1, n - span + period - 1, -1)):
            return period
    return None


class Engine(Protocol):
    """What the server, chat and MCP layers need from an inference engine."""

    last_finish_reason: str
    device: str

    def count_tokens(self, text: str) -> int: ...

    def generate(
        self, prompt: str, max_tokens: int = ..., temperature: float = ..., top_p: float = ...,
        top_k: Optional[int] = ..., repetition_penalty: Optional[float] = ...
    ) -> Dict[str, Any]: ...

    def stream_generate(
        self, prompt: str, max_tokens: int = ..., temperature: float = ..., top_p: float = ...,
        top_k: Optional[int] = ..., repetition_penalty: Optional[float] = ...
    ) -> Iterator[Tuple[str, bool, float]]: ...

    def unload(self) -> None: ...


class OnnxGenAiEngine:
    """Wraps onnxruntime_genai for direct CUDA execution."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        if not OG_AVAILABLE:
            raise RuntimeError("onnxruntime_genai is not installed in the current environment.")

        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        self.requested_device = (device or default_device()).lower()
        if self.requested_device not in DEVICES:
            raise ValueError(f"device must be one of {', '.join(DEVICES)} (got '{device}')")
        # The execution provider actually in use: "cuda", "cpu", or "default" when the installed
        # onnxruntime_genai cannot choose and the model's own genai_config decides.
        self.device = "default"
        self.fallback_reason: Optional[str] = None
        self._eos_ids = read_eos_token_ids(self.model_path)
        self._top_k = read_config_top_k(self.model_path)
        self._loop_guard = loop_guard_enabled()
        # Marker tokens the tokenizer decodes to nothing; without them a tool call cannot be told from prose. See _find_stripped_markers.
        self._literal_tokens: Dict[int, str] = {}
        self._model = None
        self._tokenizer = None
        # "stop" (EOS) or "length" (hit max_tokens) for the most recent completed generation.
        self.last_finish_reason = "stop"
        self._load_model()

    def _model_for(self, provider: str):
        """Builds a model that runs on `provider` ("cuda" or "cpu"), ignoring genai_config's provider list."""
        config = og.Config(self.model_path)
        config.clear_providers()  # an empty list means CPU
        if provider == "cuda":
            config.append_provider("cuda")
        chunk = prefill_chunk_tokens()
        if chunk and hasattr(config, "overlay"):
            config.overlay(json.dumps({"search": {"chunk_size": chunk}}))
        return og.Model(config)

    def _load_model(self):
        """Initializes og.Model (on the requested execution provider) and og.Tokenizer."""
        want = self.requested_device
        try:
            if not hasattr(og, "Config"):  # older onnxruntime_genai cannot pick a provider
                self._model = og.Model(self.model_path)
            elif want == "cpu":
                self._model, self.device = self._model_for("cpu"), "cpu"
            elif want == "cuda":
                try:
                    self._model, self.device = self._model_for("cuda"), "cuda"
                except Exception as ex:
                    raise RuntimeError(
                        f"CUDA execution provider unavailable: {ex}. Run 'prism doctor' to diagnose, "
                        f"or use --device cpu / --device auto."
                    )
            else:  # auto
                if get_gpu_info().get("available"):
                    try:
                        self._model, self.device = self._model_for("cuda"), "cuda"
                    except Exception as ex:
                        self.fallback_reason = f"CUDA execution provider failed to load: {ex}"
                        logger.warning("%s; falling back to CPU (run 'prism doctor' to diagnose)", self.fallback_reason)
                else:
                    self.fallback_reason = "no NVIDIA GPU detected"
                if self._model is None:
                    self._model, self.device = self._model_for("cpu"), "cpu"
            self._tokenizer = og.Tokenizer(self._model)
            self._literal_tokens = self._find_stripped_markers()
        except Exception as ex:
            self._model = self._tokenizer = None
            raise ModelLoadError(f"Failed to load ONNX model at {self.model_path}: {ex}")

    def _find_stripped_markers(self) -> Dict[int, str]:
        """Of the tool-call marker tokens, those that ONNX Runtime GenAI's streaming decoder turns into empty text (it drops special
        tokens, and Qwen, Phi-4-mini, Mistral and Llama 3.1 mark tool calls with them). Their literal text is put back into the output."""
        stripped = {}
        for token_id, content in read_tool_marker_tokens(self.model_path).items():
            try:
                if self._tokenizer.create_stream().decode(token_id) == "":
                    stripped[token_id] = content
            except Exception:  # a token this tokenizer does not know
                continue
        return stripped

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
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
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
            top_k=top_k,
            repetition_penalty=repetition_penalty,
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
            "device": self.device,
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
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Iterator[Tuple[str, bool, float]]:
        """
        Yields (token_text, is_first_token, current_tokens_per_sec).

        `top_k` applies when sampling (temperature > 0); by default the model's own genai_config value, or DEFAULT_TOP_K when that is 1.
        A run that falls into a repeating token cycle is stopped and reported as finish_reason "length" (see find_token_cycle).
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
            search_kwargs["top_k"] = top_k or self._top_k
            search_kwargs["do_sample"] = True
        else:
            search_kwargs["do_sample"] = False
        if repetition_penalty is not None:
            search_kwargs["repetition_penalty"] = repetition_penalty

        params.set_search_options(**search_kwargs)

        generator = og.Generator(self._model, params)
        generator.append_tokens(input_tokens)
        tokenizer_stream = self._tokenizer.create_stream()

        self.last_finish_reason = "stop"
        first = True
        count = 0
        last_token = None
        generated: List[int] = []
        looped = False
        t_start = time.perf_counter()

        try:
            while not generator.is_done():
                generator.generate_next_token()
                next_tokens = generator.get_next_tokens()
                # A numpy array of one element is falsy when that element is 0, and token id 0 is a real token ("!" in Phi-4).
                if len(next_tokens):
                    last_token = int(next_tokens[0])
                    text = tokenizer_stream.decode(next_tokens[0])
                    if not text and last_token in self._literal_tokens:
                        text = self._literal_tokens[last_token]
                    count += 1
                    elapsed = max(time.perf_counter() - t_start, 1e-6)
                    speed = count / elapsed
                    yield text, first, speed
                    first = False
                    if self._loop_guard:
                        generated.append(last_token)
                        if count % LOOP_CHECK_EVERY == 0:
                            period = find_token_cycle(generated)
                            if period:
                                logger.warning("stopped after %d tokens: the model is repeating a %d-token cycle "
                                               "(PRISM_LOOP_GUARD=off disables this check)", count, period)
                                looped = True
                                break
            # The library reports "done" for both an EOS token and the length cap. Ending on EOS exactly at the cap is still a stop.
            if looped or (count >= max_tokens and last_token not in self._eos_ids):
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
