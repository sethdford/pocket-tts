"""SSML (Speech Synthesis Markup Language) parsing and processing for pocket-tts."""

from pocket_tts.ssml.parser import is_ssml, parse_ssml
from pocket_tts.ssml.prosody import apply_prosody, crossfade_segments, generate_silence
from pocket_tts.ssml.text_normalizer import normalize_say_as
from pocket_tts.ssml.types import MarkEvent, ProsodyParams, SSMLSegment

__all__ = [
    "parse_ssml",
    "is_ssml",
    "SSMLSegment",
    "ProsodyParams",
    "MarkEvent",
    "apply_prosody",
    "crossfade_segments",
    "generate_silence",
    "normalize_say_as",
]
