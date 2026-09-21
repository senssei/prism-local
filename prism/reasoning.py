"""Reasoning extraction from <think>...</think> blocks for chat completions."""

import re
from typing import Iterable, Iterator, Optional, Tuple

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"
# Per spec P7, the streaming lookahead buffer is at most 8 characters so that a
# `<think>` tag split across chunk boundaries is never partially emitted as
# `content` and never leaks tag fragments.
LOOKAHEAD = 8


def extract_reasoning(text: str) -> Tuple[Optional[str], str]:
    """Splits (reasoning_content, content) from text containing optional <think>...</think>.

    If no <think> tag is found, returns (None, text).
    If an unclosed <think> tag is found, returns (reasoning, "").
    """
    if not text:
        return None, ""

    match = re.search(r"<think>(.*?)(?:</think>|$)", text, flags=re.DOTALL)
    if not match:
        return None, text

    start, end = match.span()
    inner = match.group(1)
    reasoning = inner.strip("\r\n")

    if not match.group(0).endswith(CLOSE_TAG):
        return reasoning, ""

    preamble = text[:start].lstrip()
    after = text[end:].lstrip("\r\n")
    content = preamble + after
    return reasoning, content


def _prefix_suffix_len(buf: str, tag: str) -> int:
    """Length of `buf`'s longest non-empty suffix that is a prefix of `tag` (0 if none)."""
    n = min(len(buf), len(tag))
    for k in range(n, 0, -1):
        if tag.startswith(buf[-k:]):
            return k
    return 0


def stream_reasoning(pieces: Iterable[str]) -> Iterator[Tuple[str, str]]:
    """Streams (kind, text) chunks where kind is 'reasoning' or 'content'.

    Extracts tokens inside <think>...</think> as reasoning, cleanly handling
    opening and closing tags even if split across chunk boundaries, and a
    non-whitespace preamble before <think>.
    """
    state = "start"
    buf = ""

    for piece in pieces:
        if not piece:
            continue
        buf += piece

        while buf:
            if state == "start":
                idx = buf.find(OPEN_TAG)
                if idx >= 0:
                    pre = buf[:idx]
                    # Mirror the original behaviour: a leading whitespace-only
                    # prefix before <think> is silently dropped, anything else
                    # becomes the first content chunk.
                    if pre and pre.strip():
                        yield "content", pre
                    state = "thinking"
                    buf = buf[idx + len(OPEN_TAG):]
                    continue
                if len(buf) < LOOKAHEAD:
                    break
                window = buf[-LOOKAHEAD:]
                if OPEN_TAG in window:
                    tag_idx = len(buf) - LOOKAHEAD + window.find(OPEN_TAG)
                    pre = buf[:tag_idx]
                    if pre and pre.strip():
                        yield "content", pre
                    state = "thinking"
                    buf = buf[tag_idx + len(OPEN_TAG):]
                    continue
                k = _prefix_suffix_len(window, OPEN_TAG)
                if k > 0:
                    pre = buf[:len(buf) - k]
                    if pre and pre.strip():
                        yield "content", pre
                    buf = buf[len(buf) - k:]
                    break
                if buf.strip():
                    yield ("content", buf)
                buf = ""
                break

            elif state == "thinking":
                idx = buf.find(CLOSE_TAG)
                if idx >= 0:
                    inner = buf[:idx]
                    if inner:
                        yield "reasoning", inner
                    state = "content"
                    buf = buf[idx + len(CLOSE_TAG):].lstrip("\r\n")
                    continue
                if len(buf) < LOOKAHEAD:
                    break
                window = buf[-LOOKAHEAD:]
                if CLOSE_TAG in window:
                    tag_idx = len(buf) - LOOKAHEAD + window.find(CLOSE_TAG)
                    inner = buf[:tag_idx]
                    if inner:
                        yield "reasoning", inner
                    state = "content"
                    buf = buf[tag_idx + len(CLOSE_TAG):].lstrip("\r\n")
                    continue
                k = _prefix_suffix_len(window, CLOSE_TAG)
                if k > 0:
                    inner = buf[:len(buf) - k]
                    if inner:
                        yield "reasoning", inner
                    buf = buf[len(buf) - k:]
                    break
                if buf:
                    yield "reasoning", buf
                buf = ""
                break

            elif state == "content":
                if buf:
                    yield "content", buf
                buf = ""
                break

    # Flush remainder
    if buf:
        if state == "thinking":
            yield "reasoning", buf
        else:
            yield "content", buf