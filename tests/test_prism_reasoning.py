"""Hermetic unit tests for reasoning extraction (extract_reasoning and stream_reasoning)."""

import unittest

from prism.reasoning import extract_reasoning, stream_reasoning


class TestExtractReasoning(unittest.TestCase):
    def test_no_think_tags(self):
        reasoning, content = extract_reasoning("Hello world! Just plain text.")
        self.assertIsNone(reasoning)
        self.assertEqual(content, "Hello world! Just plain text.")

    def test_simple_think_block(self):
        raw = "<think>\nLet us calculate 6 * 7 = 42.\n</think>\nThe answer is 42."
        reasoning, content = extract_reasoning(raw)
        self.assertEqual(reasoning, "Let us calculate 6 * 7 = 42.")
        self.assertEqual(content, "The answer is 42.")

    def test_unclosed_think_tag(self):
        raw = "<think>\nThinking in progress and cut off by max_tokens"
        reasoning, content = extract_reasoning(raw)
        self.assertEqual(reasoning, "Thinking in progress and cut off by max_tokens")
        self.assertEqual(content, "")

    def test_unclosed_think_with_leading_whitespace(self):
        raw = "\n  <think>\nThinking in progress with leading whitespace"
        reasoning, content = extract_reasoning(raw)
        self.assertEqual(reasoning, "Thinking in progress with leading whitespace")
        self.assertEqual(content, "")

    def test_leading_whitespace_before_think_in_extract(self):
        raw = "\n  <think>thought</think>\n\nAnswer"
        reasoning, content = extract_reasoning(raw)
        self.assertEqual(reasoning, "thought")
        self.assertEqual(content, "Answer")

    def test_preamble_before_think(self):
        raw = "Notes: <think>internal calculation</think>Final result"
        reasoning, content = extract_reasoning(raw)
        self.assertEqual(reasoning, "internal calculation")
        self.assertEqual(content, "Notes: Final result")

    def test_empty_string(self):
        reasoning, content = extract_reasoning("")
        self.assertIsNone(reasoning)
        self.assertEqual(content, "")

    def test_empty_think_block(self):
        # An empty `<think></think>` is a think block with no content; we surface it as an empty reasoning
        # string so the caller can decide whether to emit `reasoning_content` or omit it.
        reasoning, content = extract_reasoning("<think></think>")
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "")


class TestStreamReasoning(unittest.TestCase):
    def test_no_think_tags(self):
        # Coalescing is allowed (and expected) for chunk-boundary-spanning emissions; the spec only requires
        # that no <think> fragment ever leaks into a `content` chunk and that the pieces concatenate correctly.
        pieces = ["Hello", " world", "!"]
        result = list(stream_reasoning(pieces))
        self.assertTrue(all(kind == "content" for kind, _ in result))
        self.assertEqual("".join(text for _, text in result), "Hello world!")
        for _, text in result:
            self.assertNotIn("<think>", text)
            self.assertNotIn("</think>", text)

    def test_simple_think_stream(self):
        pieces = ["<think>\n", "thinking step 1\n", "</think>\n", "Final answer"]
        result = list(stream_reasoning(pieces))
        # Ensure reasoning pieces are labeled "reasoning" and content is labeled "content"
        reasoning = "".join(text for kind, text in result if kind == "reasoning")
        content = "".join(text for kind, text in result if kind == "content")
        self.assertIn("thinking step 1", reasoning)
        self.assertNotIn("<think>", reasoning)
        self.assertNotIn("</think>", reasoning)
        self.assertEqual(content, "Final answer")

    def test_split_opening_tag(self):
        pieces = ["<", "th", "ink>", "thought", "</think>", "ans"]
        result = list(stream_reasoning(pieces))
        reasoning = "".join(text for kind, text in result if kind == "reasoning")
        content = "".join(text for kind, text in result if kind == "content")
        self.assertEqual(reasoning, "thought")
        self.assertEqual(content, "ans")
        self.assertNotIn("<think>", reasoning)

    def test_split_closing_tag(self):
        pieces = ["<think>", "thought", "</", "th", "ink>", "ans"]
        result = list(stream_reasoning(pieces))
        reasoning = "".join(text for kind, text in result if kind == "reasoning")
        content = "".join(text for kind, text in result if kind == "content")
        self.assertEqual(reasoning, "thought")
        self.assertEqual(content, "ans")
        self.assertNotIn("</think>", reasoning)

    def test_lookalike_tag_not_think(self):
        pieces = ["<", "div>", "hello"]
        result = list(stream_reasoning(pieces))
        self.assertTrue(all(kind == "content" for kind, _ in result))
        full = "".join(text for _, text in result)
        self.assertEqual(full, "<div>hello")

    def test_unclosed_think_in_stream(self):
        pieces = ["<think>", "thinking...", " incomplete"]
        result = list(stream_reasoning(pieces))
        self.assertTrue(all(kind == "reasoning" for kind, _ in result))
        reasoning = "".join(text for _, text in result)
        self.assertEqual(reasoning, "thinking... incomplete")

    def test_leading_whitespace_before_think(self):
        pieces = ["\n  ", "<think>", "thought", "</think>", "ans"]
        result = list(stream_reasoning(pieces))
        reasoning = "".join(text for kind, text in result if kind == "reasoning")
        content = "".join(text for kind, text in result if kind == "content")
        self.assertEqual(reasoning, "thought")
        self.assertEqual(content, "ans")

    def test_single_piece_both_tags(self):
        pieces = ["<think>fast thought</think>answer here"]
        result = list(stream_reasoning(pieces))
        reasoning = "".join(text for kind, text in result if kind == "reasoning")
        content = "".join(text for kind, text in result if kind == "content")
        self.assertEqual(reasoning, "fast thought")
        self.assertEqual(content, "answer here")

    def test_html_tags_in_content_after_think(self):
        pieces = ["<think>thought</think>", "Use <div> tag and <span> tag"]
        result = list(stream_reasoning(pieces))
        reasoning = "".join(text for kind, text in result if kind == "reasoning")
        content = "".join(text for kind, text in result if kind == "content")
        self.assertEqual(reasoning, "thought")
        self.assertEqual(content, "Use <div> tag and <span> tag")

    def test_multiple_newlines_after_think_in_stream(self):
        pieces = ["<think>thought</think>\n\nAnswer"]
        result = list(stream_reasoning(pieces))
        reasoning = "".join(text for kind, text in result if kind == "reasoning")
        content = "".join(text for kind, text in result if kind == "content")
        self.assertEqual(reasoning, "thought")
        self.assertEqual(content, "Answer")

    def test_empty_stream(self):
        result = list(stream_reasoning([]))
        self.assertEqual(result, [])

    def test_preamble_before_think_in_stream(self):
        # A reasoning model that emits a non-whitespace preamble before <think> must still split cleanly:
        # preamble becomes `content`, the inside of the think block becomes `reasoning`, and no `<think>` substring
        # leaks into any `content` chunk.
        result = list(stream_reasoning(["Notes:<think>thought</think>Answer"]))
        reasoning = "".join(text for kind, text in result if kind == "reasoning")
        content = "".join(text for kind, text in result if kind == "content")
        self.assertEqual(reasoning, "thought")
        self.assertEqual(content, "Notes:Answer")
        self.assertNotIn("<think>", content)
        self.assertNotIn("</think>", content)

    def test_preamble_split_across_pieces(self):
        # Same shape as above, but the preamble, the <think> tag, the inner reasoning, the </think> tag and the
        # trailing content each arrive in their own piece. Mirrors the chunk boundaries a streaming ONNX engine
        # produces token by token.
        pieces = ["Note", "s", ":", "<", "th", "ink", ">", "th", "ought", "</", "th", "ink", ">", "A", "nswer"]
        result = list(stream_reasoning(pieces))
        reasoning = "".join(text for kind, text in result if kind == "reasoning")
        content = "".join(text for kind, text in result if kind == "content")
        self.assertEqual(reasoning, "thought")
        self.assertEqual(content, "Notes:Answer")
        self.assertNotIn("<think>", content)
        self.assertNotIn("</think>", content)


if __name__ == "__main__":
    unittest.main()
