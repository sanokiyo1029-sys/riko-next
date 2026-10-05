"""Turn an LLM token stream into speakable chunks as early as possible.

The original Riko waited for the full LLM reply, then synthesised it in one
request. Here, the first sentence goes to TTS while the LLM is still writing
the rest, which is the single biggest perceived-latency win.
"""
from __future__ import annotations

import re
import unicodedata

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "approx",
    "no", "vol", "fig", "inc", "ltd", "co", "mt", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec",
}
# Terminal punctuation (+ optional closing quotes/brackets) followed by whitespace.
_LATIN_END = re.compile(r"([.!?…]+)([\"'”’)\]]*)(\s+)")
# CJK terminal punctuation needs no following whitespace.
_CJK_END = re.compile(r"([。！？]+)([」』”’）]*)")
_SOFT_BREAK = re.compile(r"([,;:—–]|\s-\s)\s*")


def _visual_len(text: str) -> int:
    """CJK characters carry roughly a word each; count them double."""
    return len(text) + sum(1 for c in text if unicodedata.east_asian_width(c) in "WF")


class SentenceChunker:
    """Incrementally split streamed text into sentence-sized chunks.

    * first chunk is emitted eagerly (at a comma) once it grows past
      ``first_chunk_soft_chars`` so audio can start sooner;
    * chunks never exceed ``max_chars`` (split at whitespace);
    * abbreviations ("Dr.") and decimals ("3.14") do not end a sentence.
    """

    def __init__(self, min_chars: int = 12, first_chunk_soft_chars: int = 48, max_chars: int = 240) -> None:
        self.min_chars = min_chars
        self.first_soft = first_chunk_soft_chars
        self.max_chars = max_chars
        self._buf = ""
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            chunk = self._next_chunk()
            if chunk is None:
                break
            out.append(chunk)
        return out

    def flush(self) -> list[str]:
        rest = self._buf.strip()
        self._buf = ""
        if not rest:
            return []
        self._emitted += 1
        return [rest]

    # ------------------------------------------------------------------ internals
    def _emit(self, end: int) -> str:
        chunk, self._buf = self._buf[:end].strip(), self._buf[end:]
        if chunk:
            self._emitted += 1
        return chunk

    def _next_chunk(self) -> str | None:
        buf = self._buf
        min_chars = 1 if self._emitted == 0 else self.min_chars
        for m in self._sentence_ends(buf):
            end = m
            candidate = buf[:end].strip()
            if _visual_len(candidate) >= min_chars:
                chunk = self._emit(end)
                if chunk:
                    return chunk
        if self._emitted == 0 and len(buf) >= self.first_soft:
            soft = [m.end() for m in _SOFT_BREAK.finditer(buf) if m.start() >= 8]
            if soft:
                return self._emit(soft[-1]) or None
        if len(buf) > self.max_chars:
            cut = buf.rfind(" ", 0, self.max_chars)
            return self._emit(cut if cut > 0 else self.max_chars) or None
        return None

    def _sentence_ends(self, buf: str) -> list[int]:
        ends: list[int] = []
        for m in _LATIN_END.finditer(buf):
            if m.group(1) == "." and self._is_false_period(buf, m.start(1)):
                continue
            ends.append(m.end(2))
        for m in _CJK_END.finditer(buf):
            ends.append(m.end())
        return sorted(ends)

    @staticmethod
    def _is_false_period(buf: str, dot_idx: int) -> bool:
        word_match = re.search(r"([A-Za-z.]+)$", buf[:dot_idx])
        if word_match:
            word = word_match.group(1).lower().strip(".")
            if word in _ABBREVIATIONS:
                return True
            if len(word) == 1 and word.isalpha() and buf[dot_idx - 1].isupper():
                return True  # initials: "J. R. R."
        return False


class ThinkFilter:
    """Remove ``<think>...</think>`` reasoning blocks from a token stream.

    Reasoning models (DeepSeek-R1, Qwen3, ...) served via Ollama / LM Studio
    emit these; they must never be spoken. Tags may be split across tokens.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._inside = False
        self._pending = ""

    def feed(self, text: str) -> str:
        data = self._pending + text
        self._pending = ""
        out = []
        while data:
            tag = self.CLOSE if self._inside else self.OPEN
            idx = data.find(tag)
            if idx >= 0:
                if not self._inside:
                    out.append(data[:idx])
                data = data[idx + len(tag):]
                self._inside = not self._inside
                continue
            # keep a possible partial tag at the end for the next call
            keep = 0
            for k in range(min(len(tag) - 1, len(data)), 0, -1):
                if tag.startswith(data[-k:]):
                    keep = k
                    break
            body = data[: len(data) - keep]
            self._pending = data[len(data) - keep:]
            if not self._inside:
                out.append(body)
            data = ""
        return "".join(out)

    def flush(self) -> str:
        rest, self._pending = self._pending, ""
        return "" if self._inside else rest


_ACTION = re.compile(r"\*[^*\n]{1,80}\*")
_PAREN_ACTION = re.compile(r"\((?:[a-z][^()\n]{0,60})\)")
_MARKDOWN = re.compile(r"[#`_~>|]+")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0xFE00 <= cp <= 0xFE0F
        or cp == 0x200D or 0x1F900 <= cp <= 0x1F9FF
    )


def clean_for_tts(text: str, strip_actions: bool = True) -> str:
    """Make text speakable: drop *actions*, markdown, emojis and URLs."""
    text = text.replace("**", "").replace("__", "")
    if strip_actions:
        text = _ACTION.sub(" ", text)
        text = _PAREN_ACTION.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _MARKDOWN.sub(" ", text)
    text = "".join(ch for ch in text if not _is_emoji(ch))
    text = _WS.sub(" ", text).strip()
    # Nothing worth speaking if there are no letters/digits left.
    if not any(unicodedata.category(c)[0] in "LN" for c in text):
        return ""
    return text
