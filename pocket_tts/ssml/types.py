"""Data classes for SSML elements and segments."""

from dataclasses import dataclass, field


@dataclass
class ProsodyParams:
    """Parameters for prosody adjustments."""

    rate: float = 1.0  # 1.0 = normal speed, 0.5 = half speed, 2.0 = double speed
    pitch: float = 1.0  # 1.0 = normal pitch, >1 higher, <1 lower
    volume: float = 1.0  # 1.0 = normal volume, 0.0 = silent, 2.0 = double

    def merge(self, other: "ProsodyParams") -> "ProsodyParams":
        """Merge another ProsodyParams on top of this one (multiplicative)."""
        return ProsodyParams(
            rate=self.rate * other.rate,
            pitch=self.pitch * other.pitch,
            volume=self.volume * other.volume,
        )


@dataclass
class MarkEvent:
    """A timing marker event."""

    name: str
    offset_samples: int = 0


@dataclass
class SSMLSegment:
    """A segment of parsed SSML ready for synthesis.

    Each segment represents a unit of text to synthesize with associated
    control parameters (voice, prosody, breaks, etc.).
    """

    text: str = ""
    voice: str | None = None  # Voice name/URL or None for current voice
    prosody: ProsodyParams = field(default_factory=ProsodyParams)
    break_before_ms: int = 0  # Silence to insert before this segment (milliseconds)
    break_after_ms: int = 0  # Silence to insert after this segment (milliseconds)
    is_audio: bool = False  # If True, `text` is a URL/path to an audio file
    marks: list[MarkEvent] = field(default_factory=list)
    phoneme_alphabet: str | None = None  # IPA or x-sampa, best-effort
    phoneme_ph: str | None = None  # Phoneme string, best-effort

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())
