"""SSML parser that converts SSML documents into a list of SSMLSegments."""

import logging
import re
import xml.etree.ElementTree as ET
from copy import deepcopy

from pocket_tts.ssml.text_normalizer import normalize_say_as
from pocket_tts.ssml.types import MarkEvent, ProsodyParams, SSMLSegment

logger = logging.getLogger(__name__)

# Default break durations for paragraph and sentence boundaries (ms)
_PARAGRAPH_BREAK_MS = 500
_SENTENCE_BREAK_MS = 300

# Emphasis level to prosody mapping
_EMPHASIS_MAP = {
    "strong": ProsodyParams(rate=0.9, volume=1.3),
    "moderate": ProsodyParams(rate=0.95, volume=1.15),
    "none": ProsodyParams(rate=1.0, volume=0.7),
    "reduced": ProsodyParams(rate=1.1, volume=0.8),
}

# Named break strengths to durations in ms
_BREAK_STRENGTH_MAP = {
    "none": 0,
    "x-weak": 100,
    "weak": 200,
    "medium": 400,
    "strong": 600,
    "x-strong": 1000,
}


def _strip_ns(tag: str) -> str:
    """Strip XML namespace from tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_time_ms(time_str: str) -> int:
    """Parse a time string like '500ms' or '1.5s' into milliseconds."""
    time_str = time_str.strip()
    if time_str.endswith("ms"):
        return int(float(time_str[:-2]))
    elif time_str.endswith("s"):
        return int(float(time_str[:-1]) * 1000)
    else:
        try:
            return int(float(time_str) * 1000)
        except ValueError:
            logger.warning("Could not parse time string: %s, defaulting to 0", time_str)
            return 0


def _parse_rate(rate_str: str) -> float:
    """Parse prosody rate string to a float multiplier."""
    rate_str = rate_str.strip().lower()
    rate_map = {
        "x-slow": 0.5,
        "slow": 0.75,
        "medium": 1.0,
        "fast": 1.25,
        "x-fast": 1.75,
        "default": 1.0,
    }
    if rate_str in rate_map:
        return rate_map[rate_str]
    if rate_str.endswith("%"):
        return float(rate_str[:-1]) / 100.0
    try:
        return float(rate_str)
    except ValueError:
        logger.warning("Could not parse rate: %s, defaulting to 1.0", rate_str)
        return 1.0


def _parse_pitch(pitch_str: str) -> float:
    """Parse prosody pitch string to a float multiplier."""
    pitch_str = pitch_str.strip().lower()
    pitch_map = {
        "x-low": 0.5,
        "low": 0.75,
        "medium": 1.0,
        "high": 1.25,
        "x-high": 1.5,
        "default": 1.0,
    }
    if pitch_str in pitch_map:
        return pitch_map[pitch_str]
    # Handle percentage: +10%, -20%
    if pitch_str.endswith("%"):
        val = pitch_str[:-1]
        if val.startswith("+"):
            return 1.0 + float(val[1:]) / 100.0
        elif val.startswith("-"):
            return 1.0 - float(val[1:]) / 100.0
        else:
            return float(val) / 100.0
    # Handle semitones: +2st, -3st
    if pitch_str.endswith("st"):
        st = float(pitch_str[:-2])
        return 2.0 ** (st / 12.0)
    # Handle Hz values (approximate)
    if pitch_str.endswith("hz"):
        hz = float(pitch_str[:-2].lstrip("+"))
        return hz / 200.0  # Rough approximation relative to ~200Hz base
    try:
        return float(pitch_str)
    except ValueError:
        logger.warning("Could not parse pitch: %s, defaulting to 1.0", pitch_str)
        return 1.0


def _parse_volume(volume_str: str) -> float:
    """Parse prosody volume string to a float multiplier."""
    volume_str = volume_str.strip().lower()
    volume_map = {
        "silent": 0.0,
        "x-soft": 0.25,
        "soft": 0.5,
        "medium": 1.0,
        "loud": 1.5,
        "x-loud": 2.0,
        "default": 1.0,
    }
    if volume_str in volume_map:
        return volume_map[volume_str]
    if volume_str.endswith("db"):
        db = float(volume_str[:-2])
        return 10.0 ** (db / 20.0)
    if volume_str.endswith("%"):
        return float(volume_str[:-1]) / 100.0
    try:
        return float(volume_str)
    except ValueError:
        logger.warning("Could not parse volume: %s, defaulting to 1.0", volume_str)
        return 1.0


def _load_lexicon(uri: str) -> dict[str, str]:
    """Load a pronunciation lexicon from a PLS XML URI or local file.

    Supports the W3C Pronunciation Lexicon Specification (PLS) format:
    https://www.w3.org/TR/pronunciation-lexicon/

    Returns word->replacement mapping based on <grapheme>/<alias> pairs.
    Falls back to <grapheme>/<phoneme> with text pass-through if no alias.
    """
    try:
        import os

        if uri.startswith("http://") or uri.startswith("https://"):
            import requests

            response = requests.get(uri, timeout=10)
            response.raise_for_status()
            content = response.text
        elif os.path.isfile(uri):
            with open(uri) as f:
                content = f.read()
        else:
            logger.warning("Lexicon file not found: %s", uri)
            return {}

        root = ET.fromstring(content)
        lexicon = {}

        for lexeme in root.iter():
            tag = _strip_ns(lexeme.tag)
            if tag != "lexeme":
                continue

            grapheme = None
            alias = None

            for child in lexeme:
                child_tag = _strip_ns(child.tag)
                if child_tag == "grapheme":
                    grapheme = (child.text or "").strip()
                elif child_tag == "alias":
                    alias = (child.text or "").strip()
                elif child_tag == "phoneme":
                    # Use phoneme text as fallback if no alias
                    if alias is None:
                        alias = (child.text or "").strip()

            if grapheme and alias:
                lexicon[grapheme] = alias

        logger.info("Loaded %d entries from lexicon: %s", len(lexicon), uri)
        return lexicon

    except Exception as e:
        logger.warning("Failed to load lexicon from %s: %s", uri, e)
        return {}


class _SSMLWalker:
    """Walks an SSML element tree and produces SSMLSegments."""

    def __init__(self):
        self.segments: list[SSMLSegment] = []
        self.current_voice: str | None = None
        self.current_prosody: ProsodyParams = ProsodyParams()
        self.lexicon: dict[str, str] = {}
        self._pending_text: str = ""
        self._pending_break_before: int = 0

    def _flush_text(self, break_after_ms: int = 0):
        """Flush accumulated text into a segment."""
        text = self._pending_text.strip()
        if text:
            seg = SSMLSegment(
                text=text,
                voice=self.current_voice,
                prosody=deepcopy(self.current_prosody),
                break_before_ms=self._pending_break_before,
                break_after_ms=break_after_ms,
            )
            self.segments.append(seg)
            self._pending_text = ""
            self._pending_break_before = 0

    def _add_text(self, text: str):
        if text:
            # Apply lexicon substitutions
            for word, replacement in self.lexicon.items():
                text = re.sub(r"\b" + re.escape(word) + r"\b", replacement, text)
            self._pending_text += text

    def walk(self, element: ET.Element):
        """Recursively walk an SSML element tree."""
        tag = _strip_ns(element.tag)

        if tag == "speak":
            self._walk_children(element)
            self._flush_text()

        elif tag == "p":
            self._flush_text(break_after_ms=_PARAGRAPH_BREAK_MS)
            self._walk_children(element)
            self._flush_text(break_after_ms=_PARAGRAPH_BREAK_MS)

        elif tag == "s":
            self._flush_text(break_after_ms=_SENTENCE_BREAK_MS)
            self._walk_children(element)
            self._flush_text(break_after_ms=_SENTENCE_BREAK_MS)

        elif tag == "break":
            time_attr = element.get("time")
            strength = element.get("strength", "medium")
            if time_attr:
                ms = _parse_time_ms(time_attr)
            else:
                ms = _BREAK_STRENGTH_MAP.get(strength, 400)
            self._flush_text()
            self._pending_break_before = ms

        elif tag == "say-as":
            interpret_as = element.get("interpret-as", "")
            fmt = element.get("format", "")
            detail = element.get("detail", "")
            raw_text = element.text or ""
            normalized = normalize_say_as(raw_text, interpret_as, fmt, detail)
            self._add_text(normalized)

        elif tag == "sub":
            alias = element.get("alias", "")
            self._add_text(alias if alias else (element.text or ""))

        elif tag == "phoneme":
            alphabet = element.get("alphabet", "ipa")
            ph = element.get("ph", "")
            logger.info(
                "Phoneme tag encountered (alphabet=%s, ph=%s). "
                "Using text content as best-effort (model does not support phonemes directly).",
                alphabet,
                ph,
            )
            self._add_text(element.text or "")

        elif tag == "prosody":
            saved_prosody = deepcopy(self.current_prosody)
            adjustments = ProsodyParams()
            if element.get("rate"):
                adjustments.rate = _parse_rate(element.get("rate"))
            if element.get("pitch"):
                adjustments.pitch = _parse_pitch(element.get("pitch"))
            if element.get("volume"):
                adjustments.volume = _parse_volume(element.get("volume"))
            self.current_prosody = self.current_prosody.merge(adjustments)
            self._flush_text()
            self._walk_children(element)
            self._flush_text()
            self.current_prosody = saved_prosody

        elif tag == "emphasis":
            level = element.get("level", "moderate")
            saved_prosody = deepcopy(self.current_prosody)
            emphasis_params = _EMPHASIS_MAP.get(level, ProsodyParams())
            self.current_prosody = self.current_prosody.merge(emphasis_params)
            self._flush_text()
            self._walk_children(element)
            self._flush_text()
            self.current_prosody = saved_prosody

        elif tag == "voice":
            name = element.get("name", "")
            saved_voice = self.current_voice
            self.current_voice = name if name else self.current_voice
            self._flush_text()
            self._walk_children(element)
            self._flush_text()
            self.current_voice = saved_voice

        elif tag == "audio":
            src = element.get("src", "")
            if src:
                self._flush_text()
                self.segments.append(
                    SSMLSegment(
                        text=src,
                        is_audio=True,
                        voice=self.current_voice,
                        prosody=deepcopy(self.current_prosody),
                    )
                )
            # Fallback content (desc or text children) if audio can't be played
            self._walk_children(element)

        elif tag == "mark":
            name = element.get("name", "")
            if name:
                self._flush_text()
                if self.segments:
                    self.segments[-1].marks.append(MarkEvent(name=name))
                else:
                    seg = SSMLSegment(marks=[MarkEvent(name=name)])
                    self.segments.append(seg)

        elif tag == "lexicon":
            uri = element.get("uri", "")
            if uri:
                self.lexicon.update(_load_lexicon(uri))

        elif tag == "desc":
            # Description element - ignored for audio generation
            pass

        else:
            logger.warning("Unknown SSML tag: <%s>, treating as passthrough", tag)
            self._walk_children(element)

    def _walk_children(self, element: ET.Element):
        """Walk child elements, handling interleaved text.

        After walking each child, adds the child's tail text (text that
        follows the child's closing tag but belongs to the parent).
        """
        if element.text:
            self._add_text(element.text)
        for child in element:
            self.walk(child)
            if child.tail:
                self._add_text(child.tail)


def parse_ssml(ssml_text: str) -> list[SSMLSegment]:
    """Parse an SSML document into a list of SSMLSegments.

    Args:
        ssml_text: SSML document string. Must have <speak> as root element.

    Returns:
        List of SSMLSegment objects ready for synthesis.

    Raises:
        ValueError: If the SSML is malformed or missing <speak> root.
    """
    ssml_text = ssml_text.strip()

    # Handle missing XML declaration
    if not ssml_text.startswith("<?xml"):
        if not ssml_text.startswith("<speak"):
            raise ValueError("SSML must start with <speak> or <?xml")

    try:
        root = ET.fromstring(ssml_text)
    except ET.ParseError as e:
        raise ValueError(f"Malformed SSML: {e}") from e

    root_tag = _strip_ns(root.tag)
    if root_tag != "speak":
        raise ValueError(f"SSML root element must be <speak>, got <{root_tag}>")

    walker = _SSMLWalker()
    walker.walk(root)

    # Filter out empty segments (unless they have breaks or marks)
    result = []
    for seg in walker.segments:
        if seg.has_text or seg.is_audio or seg.break_before_ms > 0 or seg.marks:
            result.append(seg)

    return result


def is_ssml(text: str) -> bool:
    """Check if text appears to be SSML (starts with <speak> tag)."""
    text = text.strip()
    return text.startswith("<speak") or text.startswith("<?xml")
