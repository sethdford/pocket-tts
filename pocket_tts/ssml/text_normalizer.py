"""Text normalization for SSML <say-as> tag.

Converts numbers, dates, times, currencies, fractions, etc. to their spoken word forms.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Number words
_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
]
_ORDINAL_ONES = [
    "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
]
_ORDINAL_TENS = [
    "", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
    "sixtieth", "seventieth", "eightieth", "ninetieth",
]

_MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_CHAR_NAMES = {
    " ": "space",
    ".": "dot",
    ",": "comma",
    "!": "exclamation mark",
    "?": "question mark",
    "@": "at",
    "#": "hash",
    "$": "dollar sign",
    "%": "percent",
    "&": "ampersand",
    "*": "asterisk",
    "+": "plus",
    "-": "dash",
    "/": "slash",
    "\\": "backslash",
    "=": "equals",
    "(": "open parenthesis",
    ")": "close parenthesis",
    "[": "open bracket",
    "]": "close bracket",
    "{": "open brace",
    "}": "close brace",
    "<": "less than",
    ">": "greater than",
    ":": "colon",
    ";": "semicolon",
    "'": "apostrophe",
    '"': "quote",
    "_": "underscore",
    "~": "tilde",
    "|": "pipe",
    "^": "caret",
    "`": "backtick",
}

_CURRENCY_SYMBOLS = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "¥": ("yen", "yen", "sen", "sen"),
    "₹": ("rupee", "rupees", "paisa", "paise"),
    "₩": ("won", "won", "", ""),
    "₿": ("bitcoin", "bitcoins", "satoshi", "satoshis"),
    "CHF": ("Swiss franc", "Swiss francs", "centime", "centimes"),
    "kr": ("krone", "kroner", "øre", "øre"),
    "R$": ("real", "reais", "centavo", "centavos"),
}

# Common fraction names
_FRACTIONS = {
    (1, 2): "one half",
    (1, 3): "one third",
    (2, 3): "two thirds",
    (1, 4): "one quarter",
    (3, 4): "three quarters",
    (1, 5): "one fifth",
    (2, 5): "two fifths",
    (3, 5): "three fifths",
    (4, 5): "four fifths",
    (1, 6): "one sixth",
    (1, 8): "one eighth",
    (3, 8): "three eighths",
    (5, 8): "five eighths",
    (7, 8): "seven eighths",
    (1, 10): "one tenth",
}


def _number_to_words(n: int) -> str:
    """Convert an integer to English words."""
    if n < 0:
        return "negative " + _number_to_words(-n)
    if n == 0:
        return "zero"

    parts = []

    if n >= 1_000_000_000_000:
        trillions = n // 1_000_000_000_000
        parts.append(_number_to_words(trillions) + " trillion")
        n %= 1_000_000_000_000

    if n >= 1_000_000_000:
        billions = n // 1_000_000_000
        parts.append(_number_to_words(billions) + " billion")
        n %= 1_000_000_000

    if n >= 1_000_000:
        millions = n // 1_000_000
        parts.append(_number_to_words(millions) + " million")
        n %= 1_000_000

    if n >= 1000:
        thousands = n // 1000
        parts.append(_number_to_words(thousands) + " thousand")
        n %= 1000

    if n >= 100:
        hundreds = n // 100
        parts.append(_ONES[hundreds] + " hundred")
        n %= 100

    if n >= 20:
        tens_idx = n // 10
        ones_idx = n % 10
        if ones_idx:
            parts.append(_TENS[tens_idx] + "-" + _ONES[ones_idx])
        else:
            parts.append(_TENS[tens_idx])
    elif n > 0:
        parts.append(_ONES[n])

    return " ".join(parts)


def _ordinal_to_words(n: int) -> str:
    """Convert an integer to English ordinal words."""
    if n < 0:
        return "negative " + _ordinal_to_words(-n)
    if n == 0:
        return "zeroth"

    # Handle the last two digits as ordinal, rest as cardinal
    if n >= 100:
        prefix = _number_to_words((n // 100) * 100)
        remainder = n % 100
        if remainder == 0:
            return prefix.rsplit(" ", 1)[0] + " " + prefix.rsplit(" ", 1)[1] + "th"
        return prefix + " " + _ordinal_to_words(remainder)

    if n >= 20:
        tens_idx = n // 10
        ones_idx = n % 10
        if ones_idx:
            return _TENS[tens_idx] + "-" + _ORDINAL_ONES[ones_idx]
        else:
            return _ORDINAL_TENS[tens_idx]
    else:
        return _ORDINAL_ONES[n]


def _float_to_words(f: float) -> str:
    """Convert a float to English words."""
    if f == int(f):
        return _number_to_words(int(f))

    text = str(f)
    if "." in text:
        integer_part, decimal_part = text.split(".", 1)
        int_words = _number_to_words(int(integer_part))
        dec_words = " ".join(_ONES[int(d)] if d != "0" else "zero" for d in decimal_part)
        return f"{int_words} point {dec_words}"
    return _number_to_words(int(f))


def _cardinal(text: str) -> str:
    """Convert cardinal number text to words."""
    text = text.strip().replace(",", "")
    try:
        if "." in text:
            return _float_to_words(float(text))
        return _number_to_words(int(text))
    except ValueError:
        return text


def _ordinal(text: str) -> str:
    """Convert ordinal number text to words."""
    text = text.strip().lower()
    # Remove ordinal suffixes
    text = re.sub(r"(st|nd|rd|th)$", "", text)
    text = text.replace(",", "")
    try:
        return _ordinal_to_words(int(text))
    except ValueError:
        return text


def _characters(text: str) -> str:
    """Spell out each character."""
    parts = []
    for ch in text:
        if ch.isalpha():
            parts.append(ch.upper())
        elif ch.isdigit():
            parts.append(_ONES[int(ch)] if int(ch) > 0 else "zero")
        elif ch in _CHAR_NAMES:
            parts.append(_CHAR_NAMES[ch])
        else:
            parts.append(ch)
    return " ".join(parts)


def _fraction(text: str) -> str:
    """Convert fraction text to spoken form (e.g. '1/2' -> 'one half')."""
    text = text.strip()
    m = re.match(r"(\d+)\s*/\s*(\d+)", text)
    if m:
        numerator = int(m.group(1))
        denominator = int(m.group(2))
        if denominator == 0:
            return text

        # Check for common named fractions
        key = (numerator, denominator)
        if key in _FRACTIONS:
            return _FRACTIONS[key]

        # Build from ordinal denominator
        num_words = _number_to_words(numerator)
        if denominator == 2:
            denom_word = "half" if numerator == 1 else "halves"
        else:
            denom_word = _ordinal_to_words(denominator)
            if numerator != 1:
                denom_word += "s"
        return f"{num_words} {denom_word}"

    # Mixed number: "1 1/2"
    m = re.match(r"(\d+)\s+(\d+)\s*/\s*(\d+)", text)
    if m:
        whole = int(m.group(1))
        num = int(m.group(2))
        den = int(m.group(3))
        whole_words = _number_to_words(whole)
        frac_words = _fraction(f"{num}/{den}")
        return f"{whole_words} and {frac_words}"

    return text


def _date(text: str, fmt: str = "") -> str:
    """Convert date string to spoken form."""
    text = text.strip()
    fmt = fmt.strip().lower()

    # Try common date formats: MM/DD/YYYY or M/D/YYYY
    m = re.match(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})", text)
    if m:
        first, second, year = int(m.group(1)), int(m.group(2)), int(m.group(3))

        # Parse based on format hint
        if fmt in ("dmy", "d/m/y", "dm", "d/m"):
            day, month = first, second
        elif fmt in ("ymd", "y/m/d"):
            year, month, day = first, second, int(m.group(3))
        elif fmt in ("mdy", "m/d/y", "md", "m/d", ""):
            month, day = first, second
        else:
            month, day = first, second

        if 1 <= month <= 12:
            month_name = _MONTHS[month]
        else:
            month_name = _ordinal_to_words(month)

        day_ord = _ordinal_to_words(day)

        if year < 100:
            year += 2000 if year < 50 else 1900
        year_words = _year_to_words(year)

        # Handle format-only modes
        if fmt in ("d",):
            return day_ord
        elif fmt in ("m",):
            return month_name
        elif fmt in ("y",):
            return year_words
        elif fmt in ("md", "m/d"):
            return f"{month_name} {day_ord}"
        elif fmt in ("dm", "d/m"):
            return f"{day_ord} of {month_name}"
        elif fmt in ("my", "m/y", "ym", "y/m"):
            return f"{month_name} {year_words}"

        return f"{month_name} {day_ord}, {year_words}"

    # YYYY-MM-DD (ISO format)
    m = re.match(r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})", text)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12:
            month_name = _MONTHS[month]
        else:
            month_name = _ordinal_to_words(month)
        day_ord = _ordinal_to_words(day)
        year_words = _year_to_words(year)
        return f"{month_name} {day_ord}, {year_words}"

    return text


def _year_to_words(year: int) -> str:
    """Convert a year to spoken form."""
    if year >= 2000 and year < 2010:
        return "two thousand" + (" " + _ONES[year - 2000] if year > 2000 else "")
    elif year >= 2010 and year < 2100:
        return "twenty " + _number_to_words(year - 2000)
    elif year >= 1000:
        hi = year // 100
        lo = year % 100
        if lo == 0:
            return _number_to_words(hi) + " hundred"
        return _number_to_words(hi) + " " + _number_to_words(lo)
    return _number_to_words(year)


def _time(text: str, fmt: str = "") -> str:
    """Convert time string to spoken form."""
    text = text.strip()

    # HH:MM AM/PM or HH:MM:SS
    m = re.match(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)?", text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        second = int(m.group(3)) if m.group(3) else None
        ampm = m.group(4)

        # Handle midnight and noon
        if minute == 0 and second in (None, 0):
            if ampm:
                ampm_upper = ampm.upper()
                if hour == 12 and ampm_upper == "AM":
                    return "midnight"
                if hour == 12 and ampm_upper == "PM":
                    return "noon"
            elif hour == 0:
                return "midnight"
            elif hour == 12:
                return "noon"

        # Validate hours
        if hour > 23:
            return text

        parts = []
        parts.append(_number_to_words(hour))

        if minute == 0:
            if not ampm:
                parts.append("o'clock")
        elif minute < 10:
            parts.append("oh " + _number_to_words(minute))
        else:
            parts.append(_number_to_words(minute))

        if second is not None and second > 0:
            parts.append("and " + _number_to_words(second) + " seconds")

        if ampm:
            parts.append(ampm.upper())

        return " ".join(parts)

    return text


def _telephone(text: str) -> str:
    """Convert telephone number to spoken form (digit by digit with grouping)."""
    digits = re.sub(r"[^\d]", " ", text).strip()
    parts = []
    for ch in digits:
        if ch == " ":
            parts.append(",")
        elif ch.isdigit():
            n = int(ch)
            parts.append("zero" if n == 0 else _ONES[n])
    return " ".join(parts)


def _currency(text: str) -> str:
    """Convert currency string to spoken form.

    Handles both prefix ($100) and postfix (100$) symbol placement.
    """
    text = text.strip()

    for symbol, (singular, plural, cent_singular, cent_plural) in _CURRENCY_SYMBOLS.items():
        # Check if symbol is present (handle both prefix and postfix)
        if symbol not in text:
            continue

        # Extract amount by removing the symbol
        amount_str = text.replace(symbol, "").strip().replace(",", "")
        if not amount_str:
            continue
        try:
            amount = float(amount_str)
        except ValueError:
            return text

        if "." in amount_str:
            integer_part = int(amount)
            decimal_part = round((amount - integer_part) * 100)
        else:
            integer_part = int(amount)
            decimal_part = 0

        parts = []
        dollar_word = singular if integer_part == 1 else plural
        parts.append(f"{_number_to_words(integer_part)} {dollar_word}")

        if decimal_part > 0 and cent_singular:
            cent_word = cent_singular if decimal_part == 1 else cent_plural
            parts.append(f"and {_number_to_words(decimal_part)} {cent_word}")

        return " ".join(parts)

    return text


def _unit(text: str) -> str:
    """Convert measurement units to spoken form."""
    text = text.strip()

    unit_words = {
        "km": "kilometers", "m": "meters", "cm": "centimeters", "mm": "millimeters",
        "mi": "miles", "ft": "feet", "in": "inches", "yd": "yards",
        "kg": "kilograms", "g": "grams", "mg": "milligrams",
        "lb": "pounds", "lbs": "pounds", "oz": "ounces",
        "l": "liters", "ml": "milliliters", "gal": "gallons",
        "°C": "degrees Celsius", "°F": "degrees Fahrenheit",
        "mph": "miles per hour", "km/h": "kilometers per hour",
        "kph": "kilometers per hour",
    }

    for abbrev, full in sorted(unit_words.items(), key=lambda x: -len(x[0])):
        if text.endswith(abbrev):
            number_part = text[: -len(abbrev)].strip()
            try:
                num = float(number_part.replace(",", ""))
                num_words = _float_to_words(num) if "." in number_part else _number_to_words(
                    int(num)
                )
                return f"{num_words} {full}"
            except ValueError:
                pass

    return text


def normalize_say_as(text: str, interpret_as: str, fmt: str = "", detail: str = "") -> str:
    """Normalize text based on SSML <say-as> interpret-as attribute.

    Args:
        text: The raw text content.
        interpret_as: The interpret-as attribute value.
        fmt: Optional format hint.
        detail: Optional detail hint.

    Returns:
        Normalized text suitable for TTS.
    """
    interpret_as = interpret_as.strip().lower()

    handlers = {
        "cardinal": lambda: _cardinal(text),
        "number": lambda: _cardinal(text),
        "ordinal": lambda: _ordinal(text),
        "characters": lambda: _characters(text),
        "spell-out": lambda: _characters(text),
        "verbatim": lambda: _characters(text),
        "date": lambda: _date(text, fmt),
        "time": lambda: _time(text, fmt),
        "telephone": lambda: _telephone(text),
        "phone": lambda: _telephone(text),
        "currency": lambda: _currency(text),
        "fraction": lambda: _fraction(text),
        "unit": lambda: _unit(text),
    }

    handler = handlers.get(interpret_as)
    if handler:
        try:
            return handler()
        except Exception:
            logger.warning(
                "Failed to normalize '%s' as %s, returning as-is", text, interpret_as
            )
            return text
    else:
        logger.warning("Unknown interpret-as value: '%s', returning text as-is", interpret_as)
        return text
