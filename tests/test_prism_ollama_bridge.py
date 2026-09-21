import io
import json
import unittest
from unittest.mock import patch

from prism.ollama_bridge import embed_ollama, ollama_base_url, stream_ollama_chat


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


class TestEmbed(unittest.TestCase):
    def call(self, response, inputs=("a", "b")):
        sent = {}

        def fake_urlopen(req, timeout=None):
            sent.update(url=req.full_url, body=json.loads(req.data))
            return FakeResponse(json.dumps(response).encode())

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            return embed_ollama("ollama:nomic-embed-text", list(inputs)), sent

    def test_request_and_response(self):
        (vectors, tokens), sent = self.call({"embeddings": [[1, 2], [3, 4]], "prompt_eval_count": 5})
        self.assertEqual((vectors, tokens), ([[1, 2], [3, 4]], 5))
        self.assertTrue(sent["url"].endswith("/api/embed"))
        self.assertEqual(sent["body"], {"model": "nomic-embed-text", "input": ["a", "b"]})

    def test_a_reply_without_the_right_number_of_vectors_is_an_error(self):
        for bad in ({}, {"embeddings": []}, {"embeddings": [[1]]}, {"error": "x"}):
            with self.assertRaises(ValueError):
                self.call(bad)

    def test_missing_token_count_is_zero(self):
        (_, tokens), _ = self.call({"embeddings": [[1]]}, inputs=("a",))
        self.assertEqual(tokens, 0)


class TestTools(unittest.TestCase):
    def test_tools_are_sent_and_calls_collected(self):
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        sent = {}

        def fake_urlopen(req, timeout=None):
            sent.update(json.loads(req.data))
            return ndjson({"message": {"content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                {"function": {"name": "other", "arguments": "not a dict"}}, {"function": {}}]}, "done": True})

        stats = {}
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertEqual("".join(stream_ollama_chat("m", [], stats=stats, tools=tools)), "")
        self.assertEqual(sent["tools"], tools)
        self.assertEqual(stats["tool_calls"], [{"name": "get_weather", "arguments": {"city": "Paris"}},
                                               {"name": "other", "arguments": {}}])

    def test_no_tools_key_when_there_are_none(self):
        sent = {}
        with patch("urllib.request.urlopen", side_effect=lambda req, timeout=None: (sent.update(json.loads(req.data)), ndjson())[1]):
            list(stream_ollama_chat("m", []))
        self.assertNotIn("tools", sent)


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


class TestThinking(unittest.TestCase):
    """`stream_ollama_chat` wraps Ollama's `message.thinking` deltas with `<think>`/`</think>` tags so the
    downstream reasoning extraction in `prism.reasoning` has the same wire shape it already handles. These
    tests exercise that wrapping with ndjson fixtures at the bridge level (no server, no FakeEngine)."""

    def stream(self, *lines):
        stats = {}
        with patch("urllib.request.urlopen", return_value=ndjson(*lines)):
            text = "".join(stream_ollama_chat("ollama:m", [], stats=stats))
        return text, stats

    def test_thinking_then_content_is_wrapped(self):
        text, _ = self.stream(
            {"message": {"thinking": "step one", "content": ""}, "done": False},
            {"message": {"thinking": " step two", "content": ""}, "done": False},
            {"message": {"thinking": "", "content": "answer"}, "done": False},
            {"message": {"content": ""}, "done": True},
        )
        self.assertEqual(text, "<think>step one step two</think>answer")

    def test_thinking_only_flushes_close_tag_on_done(self):
        # The daemon never sends a content chunk before `done`. The bridge must still emit the closing
        # </think> so the downstream extractor sees a complete block.
        text, _ = self.stream(
            {"message": {"thinking": "only thinking", "content": ""}, "done": False},
            {"message": {"thinking": " more", "content": ""}, "done": True},
        )
        self.assertEqual(text, "<think>only thinking more</think>")

    def test_empty_thinking_is_skipped(self):
        text, _ = self.stream(
            {"message": {"thinking": "", "content": "no reasoning"}, "done": False},
            {"message": {"content": ""}, "done": True},
        )
        self.assertEqual(text, "no reasoning")
        self.assertNotIn("<think>", text)

    def test_thinking_chunks_with_no_content_keep_state(self):
        text, _ = self.stream(
            {"message": {"thinking": "alpha", "content": ""}, "done": False},
            {"message": {"thinking": "beta", "content": ""}, "done": False},
            {"message": {"thinking": "", "content": "hi"}, "done": False},
            {"message": {"content": ""}, "done": True},
        )
        # Exactly one open tag and one close tag, regardless of how many `thinking` chunks arrived in between.
        self.assertEqual(text.count("<think>"), 1)
        self.assertEqual(text.count("</think>"), 1)
        self.assertEqual(text, "<think>alphabeta</think>hi")


if __name__ == "__main__":
    unittest.main()
