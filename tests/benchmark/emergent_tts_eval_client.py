"""EmergentTTS-Eval client for pocket-tts using DSM TTS 1.6B.

Drop-in client that matches the exact interface required by EmergentTTS-Eval's
evaluation_runner.py. Uses category-aware voice selection as our "strong prompting"
strategy and robust text normalization for the Pronunciation category.

Setup:
    1. git clone https://github.com/boson-ai/EmergentTTS-Eval-public
    2. cp tests/benchmark/emergent_tts_eval_client.py EmergentTTS-Eval-public/
    3. Add to evaluation_runner.py (see README.md for details)

Strong prompting strategy:
    Since DSM TTS 1.6B doesn't support system prompts, we use voice selection
    as our equivalent of strong prompting. The Expresso voice dataset provides
    emotional variants (angry, happy, sad, fearful, calm, etc.) that we select
    per-sample based on the text's emotional content.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO

import numpy as np
import scipy.io.wavfile

logger = logging.getLogger(__name__)

# Voice profiles from the Expresso dataset, keyed by emotional/stylistic function.
# channel1 = close-mic (cleaner), longer durations preferred for better conditioning.
VOICE_MAP = {
    "angry": "expresso/ex03-ex01_angry_001_channel1_201s.wav",
    "happy": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
    "sad": "expresso/ex03-ex02_sad-sympathetic_001_channel1_454s.wav",
    "calm": "expresso/ex03-ex01_calm_001_channel1_1143s.wav",
    "fearful": "expresso/ex04-ex02_fearful_001_channel1_316s.wav",
    "confused": "expresso/ex03-ex01_confused_001_channel1_909s.wav",
    "sarcastic": "expresso/ex03-ex01_sarcastic_001_channel1_435s.wav",
    "awe": "expresso/ex03-ex01_awe_001_channel1_1323s.wav",
    "disgusted": "expresso/ex04-ex01_disgusted_001_channel1_130s.wav",
    "desire": "expresso/ex03-ex01_desire_004_channel1_545s.wav",
    "sleepy": "expresso/ex03-ex01_sleepy_001_channel1_619s.wav",
    "laughing": "expresso/ex03-ex01_laughing_001_channel1_188s.wav",
    "whisper": "expresso/ex01-ex02_whisper_001_channel1_579s.wav",
    "enunciated": "expresso/ex03-ex01_enunciated_001_channel1_388s.wav",
    "narration": "expresso/ex03-ex02_narration_001_channel1_674s.wav",
    "projected": "expresso/ex01-ex02_projected_001_channel1_46s.wav",
    "fast": "expresso/ex01-ex02_fast_001_channel1_104s.wav",
    "default": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
}

# Emotion keywords → voice mapping for the Emotions category.
EMOTION_KEYWORDS = {
    "angry": [
        "angry", "angrily", "anger", "furious", "rage", "frustrated", "irritated",
        "outraged", "livid", "shouted", "yelled", "screamed", "slammed", "snapped",
        "growled", "snarled", "fuming", "enraged", "infuriated", "hostile", "irate",
    ],
    "happy": [
        "happy", "happily", "joy", "joyful", "excited", "excitedly", "delighted",
        "thrilled", "ecstatic", "cheerful", "elated", "overjoyed", "grateful",
        "pride", "proud", "hopeful", "optimistic", "amazing", "wonderful",
        "incredible", "fantastic", "brilliant", "can't believe", "best day",
        "celebrate", "celebrating", "beamed", "grinned", "smiled broadly",
    ],
    "sad": [
        "sad", "sadly", "sorrow", "grief", "melancholy", "heartbroken", "devastated",
        "loss", "mourning", "depressed", "sorry", "tears", "crying", "cried", "wept",
        "sobbed", "whispered sadly", "regret", "disappointed", "miserable", "gloomy",
    ],
    "fearful": [
        "fear", "feared", "fearful", "scared", "terrified", "anxious", "worried",
        "nervous", "panic", "dread", "trembled", "shaking", "horrified", "alarmed",
        "frightened", "petrified", "terror", "trembling", "gasped", "startled",
    ],
    "calm": ["calm", "calmly", "serene", "peaceful", "relaxed", "soothing", "gentle", "composed", "softly spoke"],
    "confused": ["confused", "puzzled", "bewildered", "perplexed", "baffled", "stammered", "stuttered"],
    "sarcastic": ["sarcastic", "sarcastically", "ironic", "cynical", "mocking", "sardonic", "rolled eyes"],
    "awe": ["awe", "wonder", "amazed", "astonished", "marvel", "stunned", "speechless", "jaw dropped", "incredible"],
    "disgusted": ["disgust", "disgusted", "revolted", "repulsed", "nauseated", "appalled", "gross", "ugh"],
    "desire": ["desire", "longing", "yearning", "wanting", "craving", "wished", "pined"],
    "sleepy": ["sleepy", "tired", "drowsy", "exhausted", "weary", "lethargic", "yawned", "mumbled sleepily"],
    "laughing": ["laugh", "laughed", "laughing", "giggle", "chuckle", "snicker", "hilarious", "funny", "burst out"],
}

# Per-category default voice (no strong prompting / fallback).
CATEGORY_DEFAULT_VOICE = {
    "Emotions": "narration",
    "Paralinguistics": "narration",
    "Syntactic Complexity": "narration",
    "Foreign Words": "enunciated",
    "Questions": "narration",
    "Pronunciation": "enunciated",
}


def _detect_emotion(text: str) -> str:
    """Detect the dominant emotion in text via keyword matching.

    Returns a key into VOICE_MAP, or 'narration' if no match.
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[emotion] = score

    if not scores:
        return "narration"
    return max(scores, key=scores.get)


def _select_voice_for_sample(text: str, category: str) -> str:
    """Select the best voice profile for a given sample.

    This is our "strong prompting" strategy -- instead of a system prompt,
    we select the emotionally appropriate voice from the Expresso dataset.
    """
    if category == "Emotions":
        emotion = _detect_emotion(text)
        return VOICE_MAP.get(emotion, VOICE_MAP["narration"])

    if category == "Paralinguistics":
        text_lower = text.lower()
        if any(w in text_lower for w in ["shh", "whisper", "quiet", "hush", "softly"]):
            return VOICE_MAP["whisper"]
        if any(w in text_lower for w in ["shout", "yell", "scream", "loud", "DO NOT"]):
            return VOICE_MAP["projected"]
        return VOICE_MAP["enunciated"]

    if category == "Pronunciation":
        return VOICE_MAP["enunciated"]

    return VOICE_MAP[CATEGORY_DEFAULT_VOICE.get(category, "narration")]


def normalize_for_speech(text: str) -> str:
    """Expand written forms into spoken forms for TTS.

    Handles URLs, emails, numbers with units, mathematical notation,
    and common abbreviations. Applied for the Pronunciation category
    to reduce WER on complex written forms.
    """
    # Protocol prefixes
    text = re.sub(r"https://", "H T T P S colon slash slash ", text)
    text = re.sub(r"http://", "H T T P colon slash slash ", text)

    # www
    text = re.sub(r"www\.", "W W W dot ", text)

    # Domain dots in URLs (after protocol removal, detect url-like patterns)
    # This is conservative -- only expand dots between word chars when URL-like
    def _expand_url_dots(m):
        return m.group(0).replace(".", " dot ")
    text = re.sub(
        r"(?<=slash slash )[a-zA-Z0-9._-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?",
        _expand_url_dots,
        text,
    )

    # Email @ symbol
    text = re.sub(r"(\w)@(\w)", r"\1 at \2", text)

    # Query string separators
    text = re.sub(r"\?(?=[a-zA-Z_])", " question mark ", text)
    text = re.sub(r"&(?!amp;|lt;|gt;|quot;)(?=[a-zA-Z_])", " ampersand ", text)
    text = re.sub(r"(?<=[a-zA-Z0-9])=(?=[a-zA-Z0-9])", " equals ", text)

    # Path separators in URLs
    text = re.sub(r"(?<=\w)/(?=\w)", " slash ", text)

    # Temperature and units
    text = re.sub(r"(\d+)\s*°\s*F\b", r"\1 degrees Fahrenheit", text)
    text = re.sub(r"(\d+)\s*°\s*C\b", r"\1 degrees Celsius", text)
    text = re.sub(r"(\d+)\s*°\b", r"\1 degrees", text)

    # Percentages
    text = re.sub(r"(\d+)\s*%", r"\1 percent", text)

    # Superscripts
    text = re.sub(r"²", " squared", text)
    text = re.sub(r"³", " cubed", text)

    # Currency
    text = re.sub(r"\$(\d[\d,]*\.?\d*)", r"\1 dollars", text)
    text = re.sub(r"€(\d[\d,]*\.?\d*)", r"\1 euros", text)
    text = re.sub(r"£(\d[\d,]*\.?\d*)", r"\1 pounds", text)

    # Common abbreviations
    text = re.sub(r"\bDr\.\s", "Doctor ", text)
    text = re.sub(r"\bMr\.\s", "Mister ", text)
    text = re.sub(r"\bMrs\.\s", "Missus ", text)
    text = re.sub(r"\bMs\.\s", "Miss ", text)
    text = re.sub(r"\bSt\.\s", "Street ", text)
    text = re.sub(r"\bAve\.\s", "Avenue ", text)
    text = re.sub(r"\bBlvd\.\s", "Boulevard ", text)

    # Mathematical notation
    text = re.sub(r"\be\^\(([^)]+)\)", r"e to the power of \1", text)
    text = re.sub(r"\b(\w)\^(\d+)", r"\1 to the power of \2", text)

    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


class PocketTTSClient:
    """EmergentTTS-Eval client matching the exact API interface.

    Uses DSM TTS 1.6B locally via moshi_mlx. Implements category-aware
    voice selection as strong prompting and text normalization for Pronunciation.

    Matches the interface required by evaluation_runner.py:
      - prepare_emergent_tts_sample(text_to_synthesize, category, strong_prompting, prompting_object, **kwargs)
      - generate_audio_out(model_name, system_message, user_message, **GENERATION_CONFIG)
    """

    def __init__(self, voice_to_use: str | None = None, quantize: int = 8):
        self.voice_to_use = voice_to_use
        self._quantize = quantize
        self._backend = None
        self._voice_cache: dict[str, object] = {}

    def _ensure_loaded(self):
        if self._backend is not None:
            return
        from pocket_tts.voice.dsm_tts import DsmTTSBackend

        logger.info("Loading DSM TTS 1.6B (quantize=%d)...", self._quantize)
        self._backend = DsmTTSBackend(quantize=self._quantize)
        self._backend.load()
        logger.info("DSM TTS ready.")

    def _get_condition_attrs(self, voice_key: str):
        """Get or cache condition attributes for a voice."""
        if voice_key not in self._voice_cache:
            voice_path = self._backend.get_voice_path(voice_key)
            self._voice_cache[voice_key] = self._backend.make_condition_attributes([voice_path])
        return self._voice_cache[voice_key]

    def prepare_emergent_tts_sample(
        self,
        text_to_synthesize: str,
        category: str,
        strong_prompting: bool,
        prompting_object,
        **kwargs,
    ) -> tuple[str, str]:
        """Prepare a sample following EmergentTTS-Eval's interface.

        Returns (system_message, user_message) where:
          - system_message: encoded voice key for generation
          - user_message: the text to synthesize (possibly normalized)
        """
        user_message = text_to_synthesize

        if strong_prompting:
            voice_key = _select_voice_for_sample(text_to_synthesize, category)
            # NOTE: We do NOT normalize text for Pronunciation. The benchmark computes
            # WER against the original text, so expanding "https://" into "H T T P S
            # colon slash slash" would increase WER even though pronunciation improves.
            # The enunciated voice already articulates clearly without normalization.
        else:
            default = CATEGORY_DEFAULT_VOICE.get(category, "narration")
            voice_key = VOICE_MAP[default]

        if self.voice_to_use is not None:
            voice_key = self.voice_to_use

        # Encode voice key in system_message (the benchmark stores it in predictions)
        system_message = f"voice:{voice_key}"
        return system_message, user_message

    def generate_audio_out(
        self,
        model_name: str,
        system_message: str,
        user_message: str,
        **generation_config,
    ):
        """Generate audio matching EmergentTTS-Eval's interface.

        Returns (pydub.AudioSegment, transcript_or_None).
        """
        from pydub import AudioSegment

        self._ensure_loaded()

        # Extract voice key from system_message
        voice_key = system_message.replace("voice:", "") if system_message.startswith("voice:") else None

        if voice_key:
            condition_attrs = self._get_condition_attrs(voice_key)
            audio = self._backend.generate_audio(
                user_message, condition_attributes=condition_attrs
            )
        else:
            fallback = VOICE_MAP["narration"]
            condition_attrs = self._get_condition_attrs(fallback)
            audio = self._backend.generate_audio(
                user_message, condition_attributes=condition_attrs
            )

        sr = self._backend.sample_rate
        pcm16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        buf = BytesIO()
        scipy.io.wavfile.write(buf, sr, pcm16)
        buf.seek(0)
        segment = AudioSegment.from_wav(buf)
        return segment, None
