"""Sentence buffer for streaming LLM token accumulation.

Accumulates text tokens from streaming LLM responses and flushes
at sentence boundaries or clause boundaries (speculative mode).
Filters out code blocks so TTS only speaks prose.
"""

from __future__ import annotations

import re


class SentenceBuffer:
    """Buffers LLM text tokens and flushes speakable segments.

    Two modes:
      - "sentence": flush only on sentence-ending punctuation (. ! ? newline)
      - "speculative": flush on clause boundaries too (comma, semicolon, colon,
        dash, or after ~5 words) for lower latency at the cost of slightly
        less natural prosody.

    Code blocks (triple backticks) are tracked and excluded from speech output.
    Tool usage markers and other non-prose content is also filtered.
    """

    SENTENCE_ENDS = re.compile(r"[.!?]\s|[.!?]$|\n")
    CLAUSE_ENDS = re.compile(r"[,;:—–]\s")
    CODE_FENCE = re.compile(r"```")

    def __init__(self, mode: str = "speculative", min_words_speculative: int = 5):
        self._mode = mode
        self._min_words = min_words_speculative
        self._buffer = ""
        self._in_code_block = False
        self._pending_segments: list[str] = []

    def add(self, text: str):
        """Add a text token/chunk from the LLM stream."""
        for char in text:
            self._buffer += char
            fences = self.CODE_FENCE.findall(self._buffer)
            if len(fences) % 2 == 1:
                self._in_code_block = True
            else:
                self._in_code_block = False

        if self._in_code_block:
            return

        self._try_flush()

    def _try_flush(self):
        text = self._buffer

        if self._mode == "speculative":
            self._try_flush_speculative(text)
        else:
            self._try_flush_sentence(text)

    def _try_flush_sentence(self, text: str):
        match = self.SENTENCE_ENDS.search(text)
        if match:
            end_pos = match.end()
            segment = text[:end_pos].strip()
            remainder = text[end_pos:]
            if segment:
                clean = self._clean_for_speech(segment)
                if clean:
                    self._pending_segments.append(clean)
            self._buffer = remainder

    def _try_flush_speculative(self, text: str):
        sent_match = self.SENTENCE_ENDS.search(text)
        if sent_match:
            end_pos = sent_match.end()
            segment = text[:end_pos].strip()
            remainder = text[end_pos:]
            if segment:
                clean = self._clean_for_speech(segment)
                if clean:
                    self._pending_segments.append(clean)
            self._buffer = remainder
            return

        clause_match = self.CLAUSE_ENDS.search(text)
        word_count = len(text.split())
        if clause_match and word_count >= self._min_words:
            end_pos = clause_match.end()
            segment = text[:end_pos].strip()
            remainder = text[end_pos:]
            if segment:
                clean = self._clean_for_speech(segment)
                if clean:
                    self._pending_segments.append(clean)
            self._buffer = remainder

    def _clean_for_speech(self, text: str) -> str:
        """Remove code fences, inline code, and other non-speakable content."""
        cleaned = self.CODE_FENCE.sub("", text)
        cleaned = re.sub(r"`[^`]+`", "", cleaned)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def has_segment(self) -> bool:
        return len(self._pending_segments) > 0

    def flush(self) -> str:
        """Return the next ready segment for TTS."""
        if self._pending_segments:
            return self._pending_segments.pop(0)
        return ""

    def flush_all(self) -> str:
        """Flush everything including the current buffer (for end-of-response)."""
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining and not self._in_code_block:
            clean = self._clean_for_speech(remaining)
            if clean:
                self._pending_segments.append(clean)
        segments = self._pending_segments[:]
        self._pending_segments.clear()
        return " ".join(segments)

    def reset(self):
        """Clear all state for a new conversation turn."""
        self._buffer = ""
        self._in_code_block = False
        self._pending_segments.clear()

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in ("sentence", "speculative"):
            raise ValueError(f"Invalid mode: {value}")
        self._mode = value
