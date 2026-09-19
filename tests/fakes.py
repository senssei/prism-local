"""Shared test doubles: a fake Engine for server tests and a fake onnxruntime_genai module."""

import json
import os
import threading
import time
from typing import List


def make_model(root: str, name: str, model_type: str = "phi3") -> str:
    """Creates a minimal ONNX GenAI model folder (genai_config.json + stub weights)."""
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "genai_config.json"), "w") as f:
        json.dump({"model": {"type": model_type}}, f)
    with open(os.path.join(d, "model.onnx"), "wb") as f:
        f.write(b"\0" * 1024)
    return d


class FakeEngine:
    """
    Implements prism.engine.Engine. Records prompts, and detects unsafe concurrent use
    and use-after-unload. Class attributes are shared state; call reset() in setUp.
    """
    prompts: List[str] = []
    active = 0
    max_active = 0
    produced = 0
    fail_load = False
    pieces: List[str] = ["Hel", "lo", " world"]
    delay = 0.02
    _guard = threading.Lock()

    @classmethod
    def reset(cls):
        cls.prompts = []
        cls.active = cls.max_active = cls.produced = 0
        cls.fail_load = False
        cls.pieces = ["Hel", "lo", " world"]
        cls.delay = 0.02

    def __init__(self, path):
        if FakeEngine.fail_load:
            raise RuntimeError("boom")
        self.path = path
        self.unloaded = False
        self.last_finish_reason = "stop"

    def count_tokens(self, text):
        return len(text.split())

    def stream_generate(self, prompt, max_tokens=512, temperature=0.1, top_p=0.9):
        FakeEngine.prompts.append(prompt)
        with FakeEngine._guard:
            FakeEngine.active += 1
            FakeEngine.max_active = max(FakeEngine.max_active, FakeEngine.active)
        try:
            self.last_finish_reason = "stop"
            pieces = FakeEngine.pieces
            for i, piece in enumerate(pieces):
                if self.unloaded:
                    raise RuntimeError("use after unload")
                time.sleep(FakeEngine.delay)
                FakeEngine.produced += 1
                yield piece, i == 0, 1.0
                if i + 1 >= max_tokens:
                    break
            if len(pieces) >= max_tokens:
                self.last_finish_reason = "length"
        finally:
            with FakeEngine._guard:
                FakeEngine.active -= 1

    def generate(self, prompt, max_tokens=512, temperature=0.1, top_p=0.9):
        text = "".join(p for p, _, _ in self.stream_generate(prompt, max_tokens, temperature, top_p))
        return {"text": text, "finish_reason": self.last_finish_reason, "tokens_generated": len(text.split()),
                "ttft_sec": 0.01, "decode_tok_per_sec": 1.0}

    def unload(self):
        self.unloaded = True


class FakeOg:
    """
    Stand-in for the `onnxruntime_genai` module. Each generated token is an incrementing int that
    decodes to "t<N> ". Generation ends at `eos_after` tokens (if set) or at search option max_length.
    Inspect `calls` for what the engine asked of the library.
    """

    def __init__(self, eos_after=None, load_error=None):
        self.eos_after = eos_after
        self.load_error = load_error
        self.calls = {"search_options": [], "models": [], "appended": []}
        fake = self

        class Model:
            def __init__(self, path):
                if fake.load_error:
                    raise fake.load_error
                fake.calls["models"].append(path)

        class _Stream:
            def decode(self, token):
                return f"t{token} "

        class Tokenizer:
            def __init__(self, model):
                pass

            def encode(self, text):
                return list(range(len(text.split())))

            def create_stream(self):
                return _Stream()

        class GeneratorParams:
            def __init__(self, model):
                self.options = {}

            def set_search_options(self, **kwargs):
                self.options = kwargs
                fake.calls["search_options"].append(kwargs)

        class Generator:
            def __init__(self, model, params):
                self.params = params
                self.length = 0
                self.generated = 0

            def append_tokens(self, tokens):
                self.length += len(tokens)
                fake.calls["appended"].append(list(tokens))

            def is_done(self):
                if fake.eos_after is not None and self.generated >= fake.eos_after:
                    return True
                return self.length >= self.params.options["max_length"]

            def generate_next_token(self):
                self.generated += 1
                self.length += 1

            def get_next_tokens(self):
                return [self.generated]

        self.Model, self.Tokenizer, self.GeneratorParams, self.Generator = Model, Tokenizer, GeneratorParams, Generator
