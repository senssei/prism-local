import io
import json
import unittest
from unittest.mock import patch

from prism.ollama_bridge import ollama_base_url, stream_ollama_chat


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def ndjson(*objs):
    return FakeResponse(b"".join(json.dumps(o).encode() + b"\n" for o in objs))


class TestBaseUrl(unittest.TestCase):
    def url(self, value):
        with patch.dict("os.environ", {"OLLAMA_HOST": value} if value is not None else {}, clear=False) as env:
            if value is None:
                env.pop("OLLAMA_HOST", None)
            return ollama_base_url()

    def test_default(self):
        self.assertEqual(self.url(None), "http://localhost:11434")
        self.assertEqual(self.url("  "), "http://localhost:11434")

    def test_forms_of_ollama_host(self):
        self.assertEqual(self.url("gpu-box"), "http://gpu-box:11434")
        self.assertEqual(self.url("gpu-box:8080"), "http://gpu-box:8080")
        self.assertEqual(self.url("https://ollama.example/"), "https://ollama.example")
        self.assertEqual(self.url("http://10.0.0.5:9999"), "http://10.0.0.5:9999")


class TestStreamStats(unittest.TestCase):
    def stream(self, *lines):
        stats = {}
        with patch("urllib.request.urlopen", return_value=ndjson(*lines)):
            text = "".join(stream_ollama_chat("ollama:m", [], stats=stats))
        return text, stats

    def test_final_chunk_fills_stats(self):
        text, stats = self.stream(
            {"message": {"content": "Hi"}, "done": False},
            {"message": {"content": " there"}, "done": False},
            {"message": {"content": ""}, "done": True, "done_reason": "length", "prompt_eval_count": 9, "eval_count": 4})
        self.assertEqual(text, "Hi there")
        self.assertEqual(stats, {"prompt_tokens": 9, "completion_tokens": 4, "finish_reason": "length"})

    def test_missing_counts_are_left_out(self):
        _, stats = self.stream({"message": {"content": "x"}, "done": True})
        self.assertEqual(stats, {"finish_reason": "stop"})

    def test_works_without_stats(self):
        with patch("urllib.request.urlopen", return_value=ndjson({"message": {"content": "x"}, "done": True})):
            self.assertEqual("".join(stream_ollama_chat("m", [])), "x")


if __name__ == "__main__":
    unittest.main()
