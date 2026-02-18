"""EmergentTTS-Eval-inspired benchmark challenge texts.

6 categories from the NeurIPS'25 EmergentTTS-Eval benchmark
(https://github.com/boson-ai/EmergentTTS-Eval-public), targeting the exact
failure modes that separate top TTS models from the rest.

Each entry has:
  - text: plain text for baseline generation
  - ssml: SSML-enhanced version leveraging pocket-tts prosody (where applicable)
  - category: one of the 6 EmergentTTS-Eval categories
  - difficulty: 1-3 (maps roughly to EmergentTTS-Eval "depth")
  - notes: what this tests and why models fail

The leaderboard baseline KyutAI-TTS scores 12.72% WER / 21.94% win-rate.
Our targets with SSML prosody: <10% WER, >40% win-rate.
"""

BENCHMARK_TEXTS = [
    # -------------------------------------------------------------------------
    # Category 1: EMOTIONS
    # Models fail at conveying emotional nuance. SSML prosody (pitch, rate,
    # volume) can dramatically improve this.
    # -------------------------------------------------------------------------
    {
        "id": "emo-01",
        "category": "emotions",
        "difficulty": 1,
        "text": "I can't believe you actually did it! This is the best day of my life!",
        "ssml": (
            '<speak><prosody rate="fast" pitch="+15%" volume="loud">'
            "I can't believe you actually did it!"
            "</prosody> "
            '<prosody pitch="+20%" volume="x-loud">'
            "This is the best day of my life!"
            "</prosody></speak>"
        ),
        "notes": "Joy/excitement — requires raised pitch, faster rate, higher energy",
    },
    {
        "id": "emo-02",
        "category": "emotions",
        "difficulty": 1,
        "text": "I'm so sorry for your loss. She was a wonderful person.",
        "ssml": (
            '<speak><prosody rate="slow" pitch="-10%" volume="soft">'
            "I'm so sorry for your loss."
            "</prosody> "
            '<prosody rate="0.85" pitch="-5%" volume="soft">'
            "She was a wonderful person."
            "</prosody></speak>"
        ),
        "notes": "Sadness — requires lower pitch, slower rate, softer volume",
    },
    {
        "id": "emo-03",
        "category": "emotions",
        "difficulty": 2,
        "text": "You promised you would be here. You always do this. I'm done.",
        "ssml": (
            "<speak>"
            '<prosody pitch="+10%" rate="1.1">'
            "You promised you would be here."
            "</prosody> "
            '<prosody pitch="+20%" volume="loud" rate="1.15">'
            "You always do this."
            "</prosody> "
            '<break time="400ms"/>'
            '<prosody pitch="-15%" rate="0.8" volume="medium">'
            "I'm done."
            "</prosody></speak>"
        ),
        "notes": "Anger → resignation — emotional arc within one utterance",
    },
    {
        "id": "emo-04",
        "category": "emotions",
        "difficulty": 2,
        "text": "Wait, what? Are you serious? That's actually hilarious!",
        "ssml": (
            "<speak>"
            '<prosody pitch="+25%" rate="fast">'
            "Wait, what?"
            "</prosody> "
            '<prosody pitch="+15%">'
            "Are you serious?"
            "</prosody> "
            '<prosody pitch="+10%" rate="1.1" volume="loud">'
            "That's actually hilarious!"
            "</prosody></speak>"
        ),
        "notes": "Surprise → amusement — rapid emotional transition",
    },
    {
        "id": "emo-05",
        "category": "emotions",
        "difficulty": 3,
        "text": (
            "The results are in. I regret to inform you that the project has been cancelled. "
            "However, your team's work was exceptional, and we'd like to offer you "
            "a position on the new initiative."
        ),
        "ssml": (
            "<speak>"
            '<prosody rate="0.9" pitch="-5%">'
            "The results are in."
            "</prosody> "
            '<break time="300ms"/>'
            '<prosody rate="0.85" pitch="-10%" volume="soft">'
            "I regret to inform you that the project has been cancelled."
            "</prosody> "
            '<break time="500ms"/>'
            '<prosody rate="1.0" pitch="+5%" volume="medium">'
            "However, your team's work was exceptional,"
            "</prosody> "
            '<prosody rate="1.05" pitch="+10%" volume="loud">'
            "and we'd like to offer you a position on the new initiative."
            "</prosody></speak>"
        ),
        "notes": "Formal bad-news-then-good-news — requires tonal pivot mid-utterance",
    },
    # -------------------------------------------------------------------------
    # Category 2: PARALINGUISTICS
    # Non-verbal cues: hesitation, emphasis, whispering, sighing.
    # This is where SSML <emphasis> and <break> shine.
    # -------------------------------------------------------------------------
    {
        "id": "para-01",
        "category": "paralinguistics",
        "difficulty": 1,
        "text": "Well, I mean, it's not exactly what I expected, but it'll do.",
        "ssml": (
            "<speak>"
            '<prosody rate="0.9">'
            "Well,"
            "</prosody>"
            '<break time="200ms"/>'
            "I mean,"
            '<break time="150ms"/>'
            "it's not <emphasis>exactly</emphasis> what I expected,"
            '<break time="100ms"/>'
            "but it'll do."
            "</speak>"
        ),
        "notes": "Hedging/hesitation — natural pause patterns, de-emphasized delivery",
    },
    {
        "id": "para-02",
        "category": "paralinguistics",
        "difficulty": 2,
        "text": "Do NOT touch that. I said, do NOT touch that.",
        "ssml": (
            "<speak>"
            '<emphasis level="strong">Do NOT touch that.</emphasis>'
            '<break time="400ms"/>'
            '<prosody rate="0.8" volume="loud">'
            "I said,"
            "</prosody> "
            '<emphasis level="strong">'
            '<prosody volume="x-loud">do NOT touch that.</prosody>'
            "</emphasis></speak>"
        ),
        "notes": "Emphatic repetition — volume escalation, stressed words",
    },
    {
        "id": "para-03",
        "category": "paralinguistics",
        "difficulty": 2,
        "text": "Shh, be very quiet. I think there's someone in the house.",
        "ssml": (
            "<speak>"
            '<prosody volume="x-soft" rate="0.85">'
            "Shh, be very quiet."
            "</prosody> "
            '<break time="300ms"/>'
            '<prosody volume="soft" rate="0.8" pitch="-5%">'
            "I think there's someone in the house."
            "</prosody></speak>"
        ),
        "notes": "Whispering/quiet urgency — very low volume, slow deliberate rate",
    },
    {
        "id": "para-04",
        "category": "paralinguistics",
        "difficulty": 3,
        "text": (
            "So I told him, I said, look buddy, if you think you can just waltz in here "
            "and tell me how to do my job, you've got another thing coming."
        ),
        "ssml": (
            "<speak>"
            "So I told him, I said,"
            '<break time="200ms"/>'
            '<prosody rate="1.1" pitch="+5%">'
            "look buddy,"
            "</prosody> "
            "if you think you can just "
            '<emphasis level="moderate">waltz</emphasis> in here '
            "and tell <emphasis>me</emphasis> how to do <emphasis>my</emphasis> job,"
            '<break time="200ms"/>'
            '<prosody pitch="+10%" rate="1.05">'
            "you've got another thing coming."
            "</prosody></speak>"
        ),
        "notes": "Reported speech with attitude — conversational emphasis patterns",
    },
    # -------------------------------------------------------------------------
    # Category 3: FOREIGN WORDS
    # Proper pronunciation of non-English words within English sentences.
    # This tests the model's phonetic capability. SSML <sub> can help.
    # -------------------------------------------------------------------------
    {
        "id": "forn-01",
        "category": "foreign_words",
        "difficulty": 1,
        "text": "Let's meet at the café for some crème brûlée after the soirée.",
        "ssml": None,
        "notes": "Common French loanwords — café, crème brûlée, soirée",
    },
    {
        "id": "forn-02",
        "category": "foreign_words",
        "difficulty": 2,
        "text": "The zeitgeist of the era was defined by a certain schadenfreude and wanderlust.",
        "ssml": None,
        "notes": "German loanwords — zeitgeist, schadenfreude, wanderlust",
    },
    {
        "id": "forn-03",
        "category": "foreign_words",
        "difficulty": 2,
        "text": "She ordered the prosciutto and mozzarella panini with a cappuccino.",
        "ssml": None,
        "notes": "Italian food words — prosciutto, mozzarella, panini, cappuccino",
    },
    {
        "id": "forn-04",
        "category": "foreign_words",
        "difficulty": 3,
        "text": "The coup d'état was orchestrated by the former attaché at the Élysée Palace.",
        "ssml": None,
        "notes": "French political terms with accents — coup d'état, attaché, Élysée",
    },
    # -------------------------------------------------------------------------
    # Category 4: COMPLEX PRONUNCIATION
    # URLs, email addresses, formulas, abbreviations, numbers.
    # This is where the leaderboard WER diverges most. SSML <say-as> helps.
    # -------------------------------------------------------------------------
    {
        "id": "pron-01",
        "category": "complex_pronunciation",
        "difficulty": 1,
        "text": "Visit our website at w w w dot example dot com slash products.",
        "ssml": (
            "<speak>"
            "Visit our website at "
            '<say-as interpret-as="characters">www</say-as>'
            " dot example dot com slash products."
            "</speak>"
        ),
        "notes": "URL spelling — www must be spoken as letters",
    },
    {
        "id": "pron-02",
        "category": "complex_pronunciation",
        "difficulty": 2,
        "text": "The equation is E equals m c squared, or E = mc².",
        "ssml": (
            "<speak>"
            "The equation is E equals m c squared, or "
            '<sub alias="E equals m c squared">E = mc²</sub>.'
            "</speak>"
        ),
        "notes": "Mathematical formula — proper reading of E=mc²",
    },
    {
        "id": "pron-03",
        "category": "complex_pronunciation",
        "difficulty": 2,
        "text": "Please call us at 1-800-555-0199 or email support at info@example.com.",
        "ssml": (
            "<speak>"
            "Please call us at "
            '<say-as interpret-as="telephone">1-800-555-0199</say-as>'
            " or email support at "
            '<sub alias="info at example dot com">info@example.com</sub>.'
            "</speak>"
        ),
        "notes": "Phone number + email — digit grouping and @ symbol",
    },
    {
        "id": "pron-04",
        "category": "complex_pronunciation",
        "difficulty": 3,
        "text": (
            "The API endpoint is https://api.example.com/v2/users?limit=100&offset=0 "
            "and returns JSON with a 200 OK status."
        ),
        "ssml": (
            "<speak>"
            "The API endpoint is "
            '<sub alias="H T T P S colon slash slash api dot example dot com slash v 2 slash users '
            'question mark limit equals 100 ampersand offset equals 0">'
            "https://api.example.com/v2/users?limit=100&amp;offset=0"
            "</sub>"
            " and returns JSON with a "
            '<say-as interpret-as="cardinal">200</say-as>'
            " OK status."
            "</speak>"
        ),
        "notes": "Full URL with query params — the hardest pronunciation challenge",
    },
    {
        "id": "pron-05",
        "category": "complex_pronunciation",
        "difficulty": 2,
        "text": "The temperature is 72°F, that's about 22°C, with humidity at 65%.",
        "ssml": (
            "<speak>"
            "The temperature is "
            '<say-as interpret-as="unit">72°F</say-as>'
            ", that's about "
            '<say-as interpret-as="unit">22°C</say-as>'
            ", with humidity at "
            '<say-as interpret-as="cardinal">65</say-as> percent.'
            "</speak>"
        ),
        "notes": "Units and symbols — degrees, Fahrenheit/Celsius, percent",
    },
    # -------------------------------------------------------------------------
    # Category 5: QUESTIONS
    # Rising intonation at end, proper stress on wh-words. Models often
    # deliver questions with statement intonation.
    # -------------------------------------------------------------------------
    {
        "id": "ques-01",
        "category": "questions",
        "difficulty": 1,
        "text": "Are you coming to the party tonight?",
        "ssml": None,
        "notes": "Simple yes/no question — requires rising terminal intonation",
    },
    {
        "id": "ques-02",
        "category": "questions",
        "difficulty": 1,
        "text": "What time does the meeting start, and who else will be there?",
        "ssml": None,
        "notes": "Compound wh-question — stress on 'what' and 'who'",
    },
    {
        "id": "ques-03",
        "category": "questions",
        "difficulty": 2,
        "text": "You actually thought that was a good idea?",
        "ssml": (
            "<speak>"
            "You <emphasis>actually</emphasis> thought that was a "
            '<prosody pitch="+10%">good idea?</prosody>'
            "</speak>"
        ),
        "notes": "Rhetorical question with incredulity — emphasis + rising pitch",
    },
    {
        "id": "ques-04",
        "category": "questions",
        "difficulty": 3,
        "text": (
            "If we assume the hypothesis is correct, then wouldn't the data suggest "
            "that the correlation is spurious rather than causal?"
        ),
        "ssml": None,
        "notes": "Complex embedded question — conditional + negated question + technical vocab",
    },
    # -------------------------------------------------------------------------
    # Category 6: SYNTACTIC COMPLEXITY
    # Garden path sentences, nested clauses, comma splices, ambiguous parsing.
    # These test whether the model maintains coherent prosody across long,
    # structurally complex sentences.
    # -------------------------------------------------------------------------
    {
        "id": "syn-01",
        "category": "syntactic_complexity",
        "difficulty": 1,
        "text": "The old man the boats while the young fish the streams.",
        "ssml": None,
        "notes": "Garden path sentence — 'man' and 'fish' are verbs, not nouns",
    },
    {
        "id": "syn-02",
        "category": "syntactic_complexity",
        "difficulty": 2,
        "text": (
            "The horse raced past the barn fell, which surprised everyone who had "
            "been watching the race from the hillside."
        ),
        "ssml": None,
        "notes": "Classic garden path + relative clause nesting",
    },
    {
        "id": "syn-03",
        "category": "syntactic_complexity",
        "difficulty": 2,
        "text": (
            "The report that the committee which the board appointed submitted was "
            "rejected by the shareholders."
        ),
        "ssml": None,
        "notes": "Triple center-embedding — the/that/which nested relative clauses",
    },
    {
        "id": "syn-04",
        "category": "syntactic_complexity",
        "difficulty": 3,
        "text": (
            "That that is, is; that that is not, is not; is that it? It is."
        ),
        "ssml": (
            "<speak>"
            "That that is, <break time='200ms'/> is; "
            '<break time="150ms"/>'
            "that that is not, <break time='200ms'/> is not; "
            '<break time="300ms"/>'
            '<prosody pitch="+10%">is that it?</prosody> '
            '<break time="200ms"/>'
            "It is."
            "</speak>"
        ),
        "notes": "Punctuation-dependent meaning — same words, meaning from pauses alone",
    },
    {
        "id": "syn-05",
        "category": "syntactic_complexity",
        "difficulty": 3,
        "text": (
            "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo, "
            "which is a grammatically correct sentence."
        ),
        "ssml": None,
        "notes": "The famous Buffalo sentence — proper noun vs noun vs verb disambiguation",
    },
]

CATEGORIES = {
    "emotions": "Emotional expression and tonal variation",
    "paralinguistics": "Non-verbal cues: hesitation, emphasis, whispering",
    "foreign_words": "Non-English words within English sentences",
    "complex_pronunciation": "URLs, formulas, numbers, abbreviations",
    "questions": "Interrogative intonation and stress patterns",
    "syntactic_complexity": "Garden paths, nested clauses, ambiguous parsing",
}
