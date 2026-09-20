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

    def make(self, gpu=True, device=None, **og_kwargs):
        fake = FakeOg(**og_kwargs)
        for target, value in (("og", fake), ("OG_AVAILABLE", True),
                              ("get_gpu_info", lambda: {"available": gpu})):
            p = patch.object(engine_mod, target, value)
            p.start()
            self.addCleanup(p.stop)
        return OnnxGenAiEngine(self.tmp, device=device), fake

    def test_stream_yields_tokens_and_first_flag(self):
        engine, _ = self.make()
        out = list(engine.stream_generate("one two three", max_tokens=3))
        self.assertEqual([t for t, _, _ in out], ["t1 ", "t2 ", "t3 "])
        self.assertEqual([first for _, first, _ in out], [True, False, False])

    def write_config(self, **model):
        import json
        with open(os.path.join(self.tmp, "genai_config.json"), "w") as f:
            json.dump({"model": model}, f)

    def write_tokenizer(self, *added):
        import json
        with open(os.path.join(self.tmp, "tokenizer.json"), "w") as f:
            json.dump({"added_tokens": [{"id": i, "content": c, "special": True} for i, c in added]}, f)

    def test_tool_call_markers_the_tokenizer_drops_are_put_back(self):
        # tokens 2 and 4 are `<tool_call>` / `</tool_call>` and decode to nothing; 3 is ordinary text
        self.write_tokenizer((2, "<tool_call>"), (4, "</tool_call>"), (7, "<|im_end|>"))
        engine, _ = self.make(silent_tokens={2, 4, 7})
        text = "".join(t for t, _, _ in engine.stream_generate("one two three four", max_tokens=4))
        self.assertEqual(text, "t1 <tool_call>t3 </tool_call>")

    def test_markers_the_tokenizer_already_decodes_are_not_doubled(self):
        self.write_tokenizer((2, "<tool_call>"))
        engine, _ = self.make()  # nothing is silent: token 2 decodes to "t2 " by itself
        text = "".join(t for t, _, _ in engine.stream_generate("one two three", max_tokens=3))
        self.assertEqual(text, "t1 t2 t3 ")

    def test_other_special_tokens_stay_silent(self):
        self.write_tokenizer((2, "<|im_start|>"))  # not a tool marker
        engine, _ = self.make(silent_tokens={2})
        text = "".join(t for t, _, _ in engine.stream_generate("one two three", max_tokens=3))
        self.assertEqual(text, "t1 t3 ")

    def test_eos_exactly_at_the_cap_is_a_stop(self):
        self.write_config(eos_token_id=[3, 9])  # FakeOg's third token is 3
        engine, _ = self.make()
        result = engine.generate("one two three", max_tokens=3)
        self.assertEqual((result["tokens_generated"], result["finish_reason"]), (3, "stop"))

    def test_a_single_int_eos_id_works_too(self):
        self.write_config(eos_token_id=3)
        engine, _ = self.make()
        self.assertEqual(engine.generate("one two", max_tokens=3)["finish_reason"], "stop")

    def test_cap_without_eos_is_still_length(self):
        self.write_config(eos_token_id=99)
        engine, _ = self.make()
        self.assertEqual(engine.generate("one two", max_tokens=3)["finish_reason"], "length")

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
                         {"max_length": 3 + 5, "temperature": 0.7, "top_p": 0.8, "top_k": 40, "do_sample": True})
        list(engine.stream_generate("a b c", max_tokens=5, temperature=0.0))
        self.assertEqual(fake.calls["search_options"][-1], {"max_length": 8, "do_sample": False})

    def test_prefill_is_chunked_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRISM_PREFILL_CHUNK", None)
            _, fake = self.make()
        self.assertEqual(fake.calls["overlays"], [{"search": {"chunk_size": 1024}}])

    def test_prefill_chunking_can_be_switched_off(self):
        for off in ("0", "off", "OFF"):
            with self.subTest(value=off), patch.dict(os.environ, {"PRISM_PREFILL_CHUNK": off}):
                _, fake = self.make()
            self.assertEqual(fake.calls["overlays"], [])

    def test_prefill_chunk_size_is_passed_to_the_library(self):
        with patch.dict(os.environ, {"PRISM_PREFILL_CHUNK": "256"}):
            _, fake = self.make()
        self.assertEqual(fake.calls["overlays"], [{"search": {"chunk_size": 256}}])

    def test_prefill_chunk_applies_on_the_cpu_too(self):
        with patch.dict(os.environ, {"PRISM_PREFILL_CHUNK": "512"}):
            _, fake = self.make(gpu=False)
        self.assertEqual(fake.calls["overlays"], [{"search": {"chunk_size": 512}}])

    def test_invalid_prefill_chunk_is_rejected_with_a_clear_message(self):
        for bad in ("abc", "-5", "2.5"):
            with self.subTest(value=bad), patch.dict(os.environ, {"PRISM_PREFILL_CHUNK": bad}):
                with self.assertRaises(Exception) as ctx:
                    self.make()
                self.assertIn("PRISM_PREFILL_CHUNK", str(ctx.exception))

    def test_blank_prefill_chunk_means_unset(self):
        with patch.dict(os.environ, {"PRISM_PREFILL_CHUNK": "  "}):
            _, fake = self.make()
        self.assertEqual(fake.calls["overlays"], [{"search": {"chunk_size": 1024}}])

    def test_sampling_never_inherits_a_top_k_of_one(self):
        # genai_config.json usually says top_k 1, which makes the library ignore temperature and top_p (every request greedy).
        import json
        with open(os.path.join(self.tmp, "genai_config.json"), "w") as f:
            json.dump({"model": {}, "search": {"top_k": 1}}, f)
        engine, fake = self.make()
        list(engine.stream_generate("a b", max_tokens=2, temperature=0.7))
        self.assertEqual(fake.calls["search_options"][-1]["top_k"], engine_mod.DEFAULT_TOP_K)

    def test_a_larger_top_k_from_genai_config_is_kept(self):
        import json
        with open(os.path.join(self.tmp, "genai_config.json"), "w") as f:
            json.dump({"model": {}, "search": {"top_k": 20}}, f)
        engine, fake = self.make()
        list(engine.stream_generate("a b", max_tokens=2, temperature=0.7))
        self.assertEqual(fake.calls["search_options"][-1]["top_k"], 20)

    def test_top_k_and_repetition_penalty_from_the_caller(self):
        engine, fake = self.make()
        list(engine.stream_generate("a b", max_tokens=2, temperature=0.7, top_k=5, repetition_penalty=1.05))
        opts = fake.calls["search_options"][-1]
        self.assertEqual((opts["top_k"], opts["repetition_penalty"]), (5, 1.05))

    def test_greedy_runs_take_no_top_k_and_no_penalty_unless_asked(self):
        engine, fake = self.make()
        list(engine.stream_generate("a b", max_tokens=2, temperature=0.0, top_k=5))
        self.assertEqual(fake.calls["search_options"][-1], {"max_length": 4, "do_sample": False})

    def test_generate_passes_the_sampling_arguments_on(self):
        engine, fake = self.make()
        engine.generate("a b", max_tokens=2, temperature=0.5, top_k=7, repetition_penalty=1.02)
        opts = fake.calls["search_options"][-1]
        self.assertEqual((opts["top_k"], opts["repetition_penalty"]), (7, 1.02))

    def test_token_id_zero_is_a_token_like_any_other(self):
        # Phi-4's "!" is id 0; a truthiness check on the returned array used to drop it from the output and from the count.
        engine, _ = self.make(cycle=[0, 5], eos_after=6)
        pieces = [t for t, _, _ in engine.stream_generate("a b", max_tokens=100)]
        self.assertEqual(pieces, ["t0 ", "t5 "] * 3)
        self.assertEqual(engine.last_finish_reason, "stop")

    def test_the_cap_is_reported_as_length_when_id_zero_is_generated(self):
        engine, _ = self.make(cycle=[0, 5])
        self.assertEqual(len(list(engine.stream_generate("a b", max_tokens=10))), 10)
        self.assertEqual(engine.last_finish_reason, "length")

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

    # ---- execution provider selection (regression: engine used to silently run on CPU)

    def test_auto_uses_cuda_when_gpu_present(self):
        engine, fake = self.make(gpu=True)
        self.assertEqual((engine.device, engine.fallback_reason), ("cuda", None))
        self.assertEqual(fake.calls["models"], [["cuda"]])  # genai_config's own provider list is overridden

    def test_auto_falls_back_to_cpu_and_says_why_when_cuda_fails_to_load(self):
        engine, fake = self.make(gpu=True, cuda_error=RuntimeError("libcublasLt.so.13 not found"))
        self.assertEqual(engine.device, "cpu")
        self.assertIn("libcublasLt.so.13", engine.fallback_reason)
        self.assertEqual(fake.calls["models"], [[]])  # cleared providers == CPU
        self.assertEqual(engine.generate("a b", max_tokens=2)["device"], "cpu")

    def test_auto_without_gpu_never_tries_cuda(self):
        engine, fake = self.make(gpu=False, cuda_error=RuntimeError("should not be attempted"))
        self.assertEqual(engine.device, "cpu")
        self.assertEqual(engine.fallback_reason, "no NVIDIA GPU detected")

    def test_cpu_forces_cpu_even_with_a_gpu(self):
        engine, fake = self.make(gpu=True, device="cpu")
        self.assertEqual(engine.device, "cpu")
        self.assertEqual(fake.calls["models"], [[]])

    def test_cuda_is_strict_and_error_is_actionable(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.make(gpu=True, device="cuda", cuda_error=RuntimeError("libcublasLt.so.13 not found"))
        message = str(ctx.exception)
        self.assertIn("CUDA execution provider unavailable", message)
        self.assertIn("libcublasLt.so.13", message)
        self.assertIn("prism doctor", message)

    def test_old_onnxruntime_genai_without_config_reports_default(self):
        fake = FakeOg()
        del fake.Config
        for target, value in (("og", fake), ("OG_AVAILABLE", True), ("get_gpu_info", lambda: {"available": True})):
            p = patch.object(engine_mod, target, value)
            p.start()
            self.addCleanup(p.stop)
        engine = OnnxGenAiEngine(self.tmp)
        self.assertEqual(engine.device, "default")

    def test_device_from_environment_and_validation(self):
        with patch.dict(os.environ, {"PRISM_DEVICE": "cpu"}):
            engine, _ = self.make(gpu=True)
            self.assertEqual(engine.device, "cpu")
        with patch.dict(os.environ, {"PRISM_DEVICE": "tpu"}):
            with self.assertRaises(ValueError):
                engine_mod.default_device()
        with self.assertRaises(ValueError):
            self.make(device="tpu")

    def test_requires_onnxruntime_genai(self):
        with patch.object(engine_mod, "OG_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                OnnxGenAiEngine(self.tmp)


if __name__ == "__main__":
    unittest.main()


class TestLoopGuard(unittest.TestCase):
    CYCLE = [11, 12, 13, 14, 15, 16, 17, 18]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make(self, **og_kwargs):
        fake = FakeOg(**og_kwargs)
        for target, value in (("og", fake), ("OG_AVAILABLE", True), ("get_gpu_info", lambda: {"available": False})):
            p = patch.object(engine_mod, target, value)
            p.start()
            self.addCleanup(p.stop)
        return OnnxGenAiEngine(self.tmp, device="cpu")

    def test_find_token_cycle(self):
        find = engine_mod.find_token_cycle
        self.assertEqual(find(list(range(50)) + self.CYCLE * 30), len(self.CYCLE))
        self.assertEqual(find([7] * 250), 1)
        self.assertIsNone(find(list(range(400))))
        self.assertIsNone(find(self.CYCLE * 5))  # a cycle that has not run long enough
        self.assertIsNone(find(list(range(300)) + self.CYCLE * 10))  # 80 tokens of repeat: below the span
        self.assertIsNone(find(self.CYCLE * 30 + [99]))  # broken at the very end

    def test_a_stuck_model_is_stopped_early_and_reported_as_length(self):
        engine = self.make(cycle=self.CYCLE)
        with self.assertLogs("prism.engine", "WARNING") as logs:
            pieces = list(engine.stream_generate("a b c", max_tokens=5000))
        self.assertLess(len(pieces), 400)
        self.assertEqual(engine.last_finish_reason, "length")
        self.assertIn("8-token cycle", logs.output[0])

    def test_the_guard_can_be_switched_off(self):
        for off in ("off", "0"):
            with self.subTest(value=off), patch.dict(os.environ, {"PRISM_LOOP_GUARD": off}):
                engine = self.make(cycle=self.CYCLE)
                self.assertEqual(len(list(engine.stream_generate("a b c", max_tokens=600))), 600)
                self.assertEqual(engine.last_finish_reason, "length")

    def test_ordinary_output_is_not_flagged(self):
        engine = self.make(eos_after=800)
        pieces = list(engine.stream_generate("a b c", max_tokens=5000))
        self.assertEqual(len(pieces), 800)
        self.assertEqual(engine.last_finish_reason, "stop")
