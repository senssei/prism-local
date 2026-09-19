import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import prism.engine as engine_mod
from prism.engine import OnnxGenAiEngine, format_prompt
from tests.fakes import FakeOg


class TestPrismFormatPrompt(unittest.TestCase):
    def test_format_prompt_phi4(self):
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        formatted = format_prompt(messages, template="phi4")
        self.assertIn("<|system|>\nYou are a helpful coding assistant.<|end|>", formatted)
        self.assertIn("<|user|>\nHello!<|end|>", formatted)
        self.assertTrue(formatted.endswith("<|assistant|>\n"))

    def test_format_prompt_chatml(self):
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User prompt."}
        ]
        formatted = format_prompt(messages, template="chatml")
        self.assertIn("<|im_start|>system\nSystem prompt.<|im_end|>", formatted)
        self.assertIn("<|im_start|>user\nUser prompt.<|im_end|>", formatted)
        self.assertTrue(formatted.endswith("<|im_start|>assistant\n"))


class TestOnnxGenAiEngine(unittest.TestCase):
    """Exercises engine logic against a fake onnxruntime_genai module (no GPU needed)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make(self, **og_kwargs):
        fake = FakeOg(**og_kwargs)
        for target, value in (("og", fake), ("OG_AVAILABLE", True)):
            p = patch.object(engine_mod, target, value)
            p.start()
            self.addCleanup(p.stop)
        return OnnxGenAiEngine(self.tmp), fake

    def test_stream_yields_tokens_and_first_flag(self):
        engine, _ = self.make()
        out = list(engine.stream_generate("one two three", max_tokens=3))
        self.assertEqual([t for t, _, _ in out], ["t1 ", "t2 ", "t3 "])
        self.assertEqual([first for _, first, _ in out], [True, False, False])

    def test_finish_reason_length_when_token_cap_hit(self):
        engine, _ = self.make()
        result = engine.generate("one two three", max_tokens=4)
        self.assertEqual(result["tokens_generated"], 4)
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(engine.last_finish_reason, "length")

    def test_finish_reason_stop_on_eos_before_cap(self):
        engine, _ = self.make(eos_after=2)
        result = engine.generate("one two three", max_tokens=50)
        self.assertEqual(result["tokens_generated"], 2)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["text"], "t1 t2 ")

    def test_search_options_sampling(self):
        engine, fake = self.make()
        list(engine.stream_generate("a b c", max_tokens=5, temperature=0.7, top_p=0.8))
        self.assertEqual(fake.calls["search_options"][-1],
                         {"max_length": 3 + 5, "temperature": 0.7, "top_p": 0.8, "do_sample": True})
        list(engine.stream_generate("a b c", max_tokens=5, temperature=0.0))
        self.assertEqual(fake.calls["search_options"][-1], {"max_length": 8, "do_sample": False})

    def test_count_tokens(self):
        engine, _ = self.make()
        self.assertEqual(engine.count_tokens("a b c d"), 4)

    def test_closing_the_stream_early_does_not_claim_length(self):
        engine, _ = self.make()
        gen = engine.stream_generate("a b", max_tokens=10)
        next(gen)
        gen.close()
        self.assertEqual(engine.last_finish_reason, "stop")

    def test_unload_is_idempotent_and_blocks_further_use(self):
        engine, _ = self.make()
        engine.unload()
        engine.unload()
        with self.assertRaises(RuntimeError):
            list(engine.stream_generate("a"))
        with self.assertRaises(RuntimeError):
            engine.count_tokens("a")

    def test_missing_model_path(self):
        with patch.object(engine_mod, "og", FakeOg()), patch.object(engine_mod, "OG_AVAILABLE", True):
            with self.assertRaises(FileNotFoundError):
                OnnxGenAiEngine(os.path.join(self.tmp, "nope"))

    def test_load_failure_is_wrapped(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.make(load_error=ValueError("bad graph"))
        self.assertIn("Failed to load ONNX model", str(ctx.exception))
        self.assertIn("bad graph", str(ctx.exception))

    def test_requires_onnxruntime_genai(self):
        with patch.object(engine_mod, "OG_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                OnnxGenAiEngine(self.tmp)


if __name__ == "__main__":
    unittest.main()
