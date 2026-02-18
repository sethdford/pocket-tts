"""Tests for SSML parsing, text normalization, and prosody processing."""

import numpy as np
import pytest

from pocket_tts.ssml.parser import is_ssml, parse_ssml
from pocket_tts.ssml.prosody import (
    apply_prosody,
    crossfade_segments,
    generate_silence,
)
from pocket_tts.ssml.text_normalizer import (
    _cardinal,
    _characters,
    _currency,
    _date,
    _fraction,
    _ordinal,
    _telephone,
    _time,
    _unit,
    normalize_say_as,
)
from pocket_tts.ssml.types import ProsodyParams


# ============================================================
# SSML Parser Tests
# ============================================================


class TestIsSSML:
    def test_speak_tag(self):
        assert is_ssml("<speak>Hello</speak>")

    def test_speak_with_attributes(self):
        assert is_ssml('<speak version="1.0">Hello</speak>')

    def test_xml_declaration(self):
        assert is_ssml('<?xml version="1.0"?><speak>Hello</speak>')

    def test_plain_text(self):
        assert not is_ssml("Hello world")

    def test_html(self):
        assert not is_ssml("<html><body>Hello</body></html>")

    def test_whitespace(self):
        assert is_ssml("  <speak>Hello</speak>  ")


class TestParseSSMLBasic:
    def test_plain_text_in_speak(self):
        segments = parse_ssml("<speak>Hello world</speak>")
        assert len(segments) == 1
        assert segments[0].text == "Hello world"

    def test_empty_speak(self):
        segments = parse_ssml("<speak></speak>")
        assert len(segments) == 0

    def test_missing_speak_root(self):
        with pytest.raises(ValueError, match="must be <speak>|must start with"):
            parse_ssml("<div>Hello</div>")

    def test_malformed_xml(self):
        with pytest.raises(ValueError, match="Malformed SSML"):
            parse_ssml("<speak>unclosed")

    def test_xml_declaration_with_speak(self):
        segments = parse_ssml('<?xml version="1.0"?><speak>Test</speak>')
        assert len(segments) == 1
        assert segments[0].text == "Test"


class TestParseSSMLBreak:
    def test_break_with_time(self):
        segments = parse_ssml("<speak>Hello<break time='500ms'/>world</speak>")
        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[1].text == "world"
        assert segments[1].break_before_ms == 500

    def test_break_with_strength(self):
        segments = parse_ssml("<speak>Hello<break strength='strong'/>world</speak>")
        assert len(segments) >= 1
        found_break = False
        for seg in segments:
            if seg.break_before_ms == 600:
                found_break = True
        assert found_break

    def test_break_seconds(self):
        segments = parse_ssml("<speak>Hello<break time='1.5s'/>world</speak>")
        assert any(seg.break_before_ms == 1500 for seg in segments)

    def test_break_after_paragraph(self):
        segments = parse_ssml("<speak><p>First.</p><p>Second.</p></speak>")
        # Paragraphs should have break_after_ms
        assert any(seg.break_after_ms > 0 for seg in segments)


class TestParseSSMLParagraphSentence:
    def test_paragraph(self):
        segments = parse_ssml("<speak><p>First paragraph.</p><p>Second paragraph.</p></speak>")
        assert len(segments) >= 2
        texts = [s.text for s in segments if s.has_text]
        assert "First paragraph." in texts
        assert "Second paragraph." in texts

    def test_sentence(self):
        segments = parse_ssml("<speak><s>First sentence.</s><s>Second sentence.</s></speak>")
        assert len(segments) >= 2


class TestParseSSMLSub:
    def test_sub_alias(self):
        segments = parse_ssml('<speak>I live on <sub alias="Fifth Avenue">5th Ave</sub>.</speak>')
        assert len(segments) == 1
        assert "Fifth Avenue" in segments[0].text

    def test_sub_empty_alias(self):
        segments = parse_ssml("<speak>Test <sub>fallback</sub> text.</speak>")
        assert len(segments) == 1
        assert "fallback" in segments[0].text


class TestParseSSMLSayAs:
    def test_cardinal(self):
        segments = parse_ssml(
            '<speak>There are <say-as interpret-as="cardinal">42</say-as> items.</speak>'
        )
        assert len(segments) == 1
        assert "forty-two" in segments[0].text

    def test_ordinal(self):
        segments = parse_ssml(
            '<speak>It is the <say-as interpret-as="ordinal">3</say-as> time.</speak>'
        )
        assert len(segments) == 1
        assert "third" in segments[0].text

    def test_characters(self):
        segments = parse_ssml(
            '<speak>Spell: <say-as interpret-as="characters">ABC</say-as></speak>'
        )
        assert len(segments) == 1
        assert "A B C" in segments[0].text

    def test_fraction(self):
        segments = parse_ssml(
            '<speak><say-as interpret-as="fraction">1/2</say-as></speak>'
        )
        assert len(segments) == 1
        assert "half" in segments[0].text

    def test_verbatim(self):
        segments = parse_ssml(
            '<speak><say-as interpret-as="verbatim">AB</say-as></speak>'
        )
        assert len(segments) == 1
        assert "A" in segments[0].text and "B" in segments[0].text

    def test_unit(self):
        segments = parse_ssml(
            '<speak><say-as interpret-as="unit">5km</say-as></speak>'
        )
        assert len(segments) == 1
        assert "kilometers" in segments[0].text


class TestParseSSMLProsody:
    def test_rate(self):
        segments = parse_ssml('<speak><prosody rate="fast">Quick speech</prosody></speak>')
        assert len(segments) == 1
        assert segments[0].prosody.rate == 1.25

    def test_volume(self):
        segments = parse_ssml('<speak><prosody volume="loud">Loud speech</prosody></speak>')
        assert len(segments) == 1
        assert segments[0].prosody.volume == 1.5

    def test_pitch(self):
        segments = parse_ssml('<speak><prosody pitch="high">High pitch</prosody></speak>')
        assert len(segments) == 1
        assert segments[0].prosody.pitch == 1.25

    def test_rate_percentage(self):
        segments = parse_ssml('<speak><prosody rate="150%">Fast</prosody></speak>')
        assert len(segments) == 1
        assert abs(segments[0].prosody.rate - 1.5) < 0.01

    def test_pitch_semitones(self):
        segments = parse_ssml('<speak><prosody pitch="+2st">Higher</prosody></speak>')
        assert len(segments) == 1
        assert segments[0].prosody.pitch > 1.0

    def test_volume_db(self):
        segments = parse_ssml('<speak><prosody volume="+6db">Louder</prosody></speak>')
        assert len(segments) == 1
        assert segments[0].prosody.volume > 1.0


class TestParseSSMLEmphasis:
    def test_strong_emphasis(self):
        segments = parse_ssml(
            '<speak><emphasis level="strong">Important</emphasis> text.</speak>'
        )
        assert any(s.prosody.volume > 1.0 for s in segments if "Important" in s.text)

    def test_reduced_emphasis(self):
        segments = parse_ssml(
            '<speak><emphasis level="reduced">Quiet</emphasis> text.</speak>'
        )
        assert any(s.prosody.volume < 1.0 for s in segments if "Quiet" in s.text)


class TestParseSSMLVoice:
    def test_voice_name(self):
        segments = parse_ssml('<speak><voice name="alba">Hello</voice></speak>')
        assert len(segments) == 1
        assert segments[0].voice == "alba"

    def test_voice_switch_back(self):
        segments = parse_ssml(
            '<speak>Default voice. <voice name="alba">Alba voice.</voice> Back to default.</speak>'
        )
        assert len(segments) >= 2
        alba_segs = [s for s in segments if s.voice == "alba"]
        default_segs = [s for s in segments if s.voice is None]
        assert len(alba_segs) >= 1
        assert len(default_segs) >= 1


class TestParseSSMLAudio:
    def test_audio_src(self):
        segments = parse_ssml(
            '<speak>Before. <audio src="sound.wav"/> After.</speak>'
        )
        audio_segs = [s for s in segments if s.is_audio]
        assert len(audio_segs) == 1
        assert audio_segs[0].text == "sound.wav"


class TestParseSSMLMark:
    def test_mark_name(self):
        segments = parse_ssml(
            '<speak>Hello <mark name="mid"/>world.</speak>'
        )
        all_marks = []
        for seg in segments:
            all_marks.extend(seg.marks)
        assert any(m.name == "mid" for m in all_marks)


class TestParseSSMLPhoneme:
    def test_phoneme_passthrough(self):
        segments = parse_ssml(
            '<speak><phoneme alphabet="ipa" ph="həˈloʊ">hello</phoneme></speak>'
        )
        assert len(segments) == 1
        assert "hello" in segments[0].text


class TestParseSSMLLexicon:
    def test_lexicon_unknown_uri_warns(self):
        """Lexicon with non-existent URI should not crash, just warn."""
        segments = parse_ssml(
            '<speak><lexicon uri="nonexistent.pls"/><p>Hello world</p></speak>'
        )
        assert len(segments) >= 1

    def test_unknown_tag_passthrough(self):
        """Unknown tags should pass through their text content."""
        segments = parse_ssml("<speak><custom>Hello</custom></speak>")
        assert len(segments) == 1
        assert "Hello" in segments[0].text


# ============================================================
# Text Normalizer Tests
# ============================================================


class TestCardinal:
    def test_basic(self):
        assert _cardinal("42") == "forty-two"

    def test_zero(self):
        assert _cardinal("0") == "zero"

    def test_large(self):
        assert "million" in _cardinal("1000000")

    def test_negative(self):
        assert _cardinal("-5") == "negative five"

    def test_float(self):
        result = _cardinal("3.14")
        assert "three" in result and "point" in result

    def test_with_commas(self):
        assert "thousand" in _cardinal("1,000")


class TestOrdinal:
    def test_basic(self):
        assert _ordinal("1") == "first"
        assert _ordinal("2") == "second"
        assert _ordinal("3") == "third"

    def test_with_suffix(self):
        assert _ordinal("1st") == "first"
        assert _ordinal("2nd") == "second"
        assert _ordinal("3rd") == "third"

    def test_teens(self):
        assert _ordinal("11") == "eleventh"
        assert _ordinal("12") == "twelfth"

    def test_tens(self):
        assert _ordinal("20") == "twentieth"
        assert _ordinal("21") == "twenty-first"


class TestCharacters:
    def test_letters(self):
        assert _characters("ABC") == "A B C"

    def test_digits(self):
        result = _characters("123")
        assert "one" in result and "two" in result and "three" in result

    def test_mixed(self):
        result = _characters("A1")
        assert "A" in result and "one" in result


class TestFraction:
    def test_half(self):
        assert _fraction("1/2") == "one half"

    def test_quarter(self):
        assert _fraction("1/4") == "one quarter"

    def test_three_quarters(self):
        assert _fraction("3/4") == "three quarters"

    def test_two_thirds(self):
        assert _fraction("2/3") == "two thirds"

    def test_generic_fraction(self):
        result = _fraction("3/7")
        assert "three" in result and "seventh" in result

    def test_invalid(self):
        assert _fraction("not a fraction") == "not a fraction"


class TestDate:
    def test_us_format(self):
        result = _date("2/14/2026")
        assert "February" in result
        assert "fourteenth" in result

    def test_iso_format(self):
        result = _date("2026-02-14")
        assert "February" in result
        assert "fourteenth" in result

    def test_dmy_format(self):
        result = _date("14/2/2026", fmt="dmy")
        assert "February" in result
        assert "fourteenth" in result

    def test_month_day_only(self):
        result = _date("3/15/2026", fmt="md")
        assert "March" in result
        assert "fifteenth" in result
        assert "twenty" not in result  # no year in md format

    def test_year_only(self):
        result = _date("1/1/2026", fmt="y")
        assert "twenty" in result


class TestTime:
    def test_basic(self):
        result = _time("3:30 PM")
        assert "three" in result
        assert "thirty" in result
        assert "PM" in result

    def test_noon(self):
        result = _time("12:00 PM")
        assert result == "noon"

    def test_midnight_12am(self):
        result = _time("12:00 AM")
        assert result == "midnight"

    def test_midnight_0(self):
        result = _time("0:00")
        assert result == "midnight"

    def test_oh_minutes(self):
        result = _time("3:05")
        assert "oh" in result and "five" in result

    def test_oclock(self):
        result = _time("3:00")
        assert "o'clock" in result


class TestTelephone:
    def test_basic(self):
        result = _telephone("555-1234")
        assert "five" in result
        assert "one" in result


class TestCurrency:
    def test_dollars(self):
        result = _currency("$42.50")
        assert "forty-two" in result
        assert "dollars" in result
        assert "fifty" in result
        assert "cents" in result

    def test_euros(self):
        result = _currency("€100")
        assert "one hundred" in result
        assert "euro" in result

    def test_postfix_symbol(self):
        result = _currency("100$")
        assert "one hundred" in result
        assert "dollars" in result

    def test_pounds(self):
        result = _currency("£5.99")
        assert "five" in result
        assert "pounds" in result

    def test_one_dollar(self):
        result = _currency("$1")
        assert "one dollar" in result


class TestUnit:
    def test_kilometers(self):
        result = _unit("5km")
        assert "five" in result
        assert "kilometers" in result

    def test_celsius(self):
        result = _unit("37°C")
        assert "degrees Celsius" in result

    def test_pounds(self):
        result = _unit("10lbs")
        assert "ten" in result
        assert "pounds" in result

    def test_unknown_unit(self):
        result = _unit("5xyz")
        assert result == "5xyz"


class TestNormalizeSayAs:
    def test_unknown_type(self):
        result = normalize_say_as("hello", "unknown_type")
        assert result == "hello"

    def test_cardinal(self):
        result = normalize_say_as("42", "cardinal")
        assert result == "forty-two"

    def test_ordinal(self):
        result = normalize_say_as("3", "ordinal")
        assert result == "third"

    def test_fraction(self):
        result = normalize_say_as("1/2", "fraction")
        assert "half" in result

    def test_verbatim(self):
        result = normalize_say_as("AB", "verbatim")
        assert "A" in result and "B" in result

    def test_unit(self):
        result = normalize_say_as("5km", "unit")
        assert "kilometers" in result

    def test_number_alias(self):
        result = normalize_say_as("42", "number")
        assert result == "forty-two"

    def test_phone_alias(self):
        result = normalize_say_as("555-1234", "phone")
        assert "five" in result

    def test_spell_out_alias(self):
        result = normalize_say_as("ABC", "spell-out")
        assert "A B C" == result


# ============================================================
# Prosody Processing Tests
# ============================================================


class TestGenerateSilence:
    def test_basic(self):
        silence = generate_silence(24000, 1000)
        assert silence.shape == (24000,)
        assert np.all(silence == 0)

    def test_short(self):
        silence = generate_silence(24000, 100)
        assert silence.shape == (2400,)


class TestApplyProsody:
    def test_identity(self):
        audio = np.random.randn(24000).astype(np.float32)
        result = apply_prosody(audio, 24000, ProsodyParams())
        np.testing.assert_array_almost_equal(result, audio, decimal=5)

    def test_volume_increase(self):
        audio = np.ones(1000, dtype=np.float32) * 0.5
        result = apply_prosody(audio, 24000, ProsodyParams(volume=2.0))
        assert np.abs(result).mean() > np.abs(audio).mean()

    def test_volume_decrease(self):
        audio = np.ones(1000, dtype=np.float32) * 0.5
        result = apply_prosody(audio, 24000, ProsodyParams(volume=0.5))
        assert np.abs(result).mean() < np.abs(audio).mean()

    def test_volume_soft_clipping(self):
        """Soft-knee limiter should not distort quiet samples."""
        audio = np.array([0.1, 0.5, 0.9, 1.2], dtype=np.float32)
        result = apply_prosody(audio, 24000, ProsodyParams(volume=1.5))
        # Quiet samples should scale linearly (below knee threshold)
        assert abs(result[0] - 0.15) < 0.01
        # Loud samples should be compressed but not above 1.0
        assert np.all(np.abs(result) <= 1.0 + 0.01)

    def test_rate_faster(self):
        audio = np.random.randn(24000).astype(np.float32)
        result = apply_prosody(audio, 24000, ProsodyParams(rate=2.0))
        assert len(result) < len(audio)

    def test_rate_slower(self):
        audio = np.random.randn(24000).astype(np.float32)
        result = apply_prosody(audio, 24000, ProsodyParams(rate=0.5))
        assert len(result) > len(audio)

    def test_empty_audio(self):
        audio = np.array([], dtype=np.float32)
        result = apply_prosody(audio, 24000, ProsodyParams(rate=2.0))
        assert result.size == 0


class TestCrossfadeSegments:
    def test_basic_crossfade(self):
        a = np.ones(1000, dtype=np.float32)
        b = np.zeros(1000, dtype=np.float32)
        result = crossfade_segments(a, b, 24000)
        # Result should be shorter than simple concatenation
        assert len(result) < len(a) + len(b)
        # Should start at ~1.0 and end at ~0.0
        assert abs(result[0] - 1.0) < 0.01
        assert abs(result[-1] - 0.0) < 0.01

    def test_empty_prev(self):
        a = np.array([], dtype=np.float32)
        b = np.ones(1000, dtype=np.float32)
        result = crossfade_segments(a, b, 24000)
        np.testing.assert_array_equal(result, b)

    def test_empty_next(self):
        a = np.ones(1000, dtype=np.float32)
        b = np.array([], dtype=np.float32)
        result = crossfade_segments(a, b, 24000)
        np.testing.assert_array_equal(result, a)

    def test_short_segments(self):
        a = np.array([1.0], dtype=np.float32)
        b = np.array([0.0], dtype=np.float32)
        result = crossfade_segments(a, b, 24000)
        assert len(result) == 2  # Too short to crossfade, just concat


class TestProsodyParams:
    def test_merge_multiplicative(self):
        p1 = ProsodyParams(rate=1.5, volume=0.8)
        p2 = ProsodyParams(rate=2.0, volume=1.25)
        merged = p1.merge(p2)
        assert abs(merged.rate - 3.0) < 0.01
        assert abs(merged.volume - 1.0) < 0.01

    def test_merge_identity(self):
        p = ProsodyParams(rate=1.5, pitch=0.8, volume=1.2)
        identity = ProsodyParams()
        merged = p.merge(identity)
        assert abs(merged.rate - 1.5) < 0.01
        assert abs(merged.pitch - 0.8) < 0.01
        assert abs(merged.volume - 1.2) < 0.01


# ============================================================
# Integration: Full SSML Document Parsing
# ============================================================


class TestFullSSMLDocument:
    def test_complex_document(self):
        ssml = """
        <speak>
            <p>
                <s>Welcome to our service.</s>
                <s>Your order number is <say-as interpret-as="cardinal">42</say-as>.</s>
            </p>
            <break time="1s"/>
            <p>
                <prosody rate="slow" volume="loud">
                    Please listen carefully.
                </prosody>
            </p>
        </speak>
        """
        segments = parse_ssml(ssml)
        assert len(segments) >= 2
        all_text = " ".join(s.text for s in segments if s.has_text)
        assert "forty-two" in all_text
        slow_segs = [s for s in segments if s.prosody.rate < 1.0]
        assert len(slow_segs) >= 1

    def test_voice_switching_document(self):
        ssml = """
        <speak>
            Hello from the default voice.
            <voice name="alba">
                Now I am Alba.
            </voice>
            Back to default.
        </speak>
        """
        segments = parse_ssml(ssml)
        assert len(segments) >= 2
        voices = set(s.voice for s in segments)
        assert None in voices or "alba" in voices

    def test_nested_prosody(self):
        ssml = """
        <speak>
            <prosody rate="fast">
                Fast speech.
                <prosody volume="loud">Fast and loud.</prosody>
            </prosody>
        </speak>
        """
        segments = parse_ssml(ssml)
        fast_loud = [s for s in segments if s.prosody.rate > 1.0 and s.prosody.volume > 1.0]
        assert len(fast_loud) >= 1

    def test_say_as_with_date_format(self):
        ssml = """
        <speak>
            The date is <say-as interpret-as="date" format="dmy">14/2/2026</say-as>.
        </speak>
        """
        segments = parse_ssml(ssml)
        all_text = " ".join(s.text for s in segments if s.has_text)
        assert "February" in all_text
        assert "fourteenth" in all_text

    def test_emphasis_levels(self):
        ssml = """
        <speak>
            <emphasis level="strong">Very important!</emphasis>
            <emphasis level="none">Not important.</emphasis>
        </speak>
        """
        segments = parse_ssml(ssml)
        assert any(
            s.prosody.volume > 1.0 for s in segments if "Very important" in s.text
        )
        assert any(
            s.prosody.volume < 1.0 for s in segments if "Not important" in s.text
        )

    def test_mixed_content(self):
        ssml = """
        <speak>
            Call us at <say-as interpret-as="telephone">555-1234</say-as>.
            Your total is <say-as interpret-as="currency">$42.50</say-as>.
            That's <say-as interpret-as="fraction">3/4</say-as> done.
        </speak>
        """
        segments = parse_ssml(ssml)
        all_text = " ".join(s.text for s in segments if s.has_text)
        assert "five" in all_text  # telephone
        assert "dollars" in all_text  # currency
        assert "quarter" in all_text or "three" in all_text  # fraction
