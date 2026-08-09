"""Keyword parser - the fallback when no LLM is available.

Three reasons it exists:

  1. The app is never dead. No key, no quota, no network - Style Me still works,
     and the UI says it's degraded.
  2. It's the control condition. Having a baseline is what lets the evaluation
     say what the LLM actually buys.
  3. It makes the LLM's failure mode safe: any response that fails validation
     falls back here.

Intentionally literal. Where it can't tell, it leaves the default rather than
guessing.
"""

from __future__ import annotations

import re

from src.features import (
    COLOR_KEYWORD_TO_FAMILY,
    DEFAULT_OCCASION,
    EXPLICIT_COLOUR_MATCHES,
    GARMENT_SYNONYMS,
)
from src.schemas import ItemRequest, StylePreferences

# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------
# Handles "under Rs.3000", "₹3,000", "3000 rupees", "3k", "below 2.5k".
_BUDGET_PATTERNS = (
    re.compile(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)\s*(k\b)?", re.I),
    re.compile(r"\b(?:under|below|within|upto|up to|less than|max(?:imum)?|"
               r"nothing over|no more than|not more than|"
               r"budget(?:\s+(?:of|is))?|around|about|roughly|approx(?:imately)?)\s*"
               r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(k\b)?", re.I),
    re.compile(r"\b([\d,]+(?:\.\d+)?)\s*(k\b)?\s*(?:rupees|rs\b|bucks|budget)", re.I),
    # Bare "6k" / "2.5k" with no currency word at all.
    re.compile(r"\b(\d+(?:\.\d+)?)\s*(k)\b", re.I),
    # A trailing amount after a comma - "...for a wedding, 6000." This phrasing
    # is common and unambiguous enough to be worth catching.
    re.compile(r",\s*(?:rs\.?|inr|₹)?\s*(\d{3,6})\s*(k\b)?\s*[.!?]?\s*$", re.I),
)

_SPELLED_BUDGETS = {
    "five hundred": 500, "thousand": 1000, "fifteen hundred": 1500,
    "two thousand": 2000, "twenty five hundred": 2500, "three thousand": 3000,
    "four thousand": 4000, "five thousand": 5000, "six thousand": 6000,
    "eight thousand": 8000, "ten thousand": 10000,
}


def extract_budget(text: str) -> int | None:
    lowered = text.lower()
    for phrase, value in _SPELLED_BUDGETS.items():
        if phrase in lowered:
            return value

    for pattern in _BUDGET_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        raw, k_suffix = match.group(1), match.group(2)
        try:
            amount = float(raw.replace(",", ""))
        except ValueError:
            continue
        if k_suffix:
            amount *= 1000
        if amount >= 100:  # ignore "size 40", "2 shirts" and similar noise
            return int(amount)
    return None


# --------------------------------------------------------------------------
# Gender
# --------------------------------------------------------------------------
# Gender signals come in two strengths, and mixing them causes real mistakes.
# "a saree for men" contains one women's-garment hint and one explicit "men";
# treating them as equal votes produced a tie that silently resolved to Women -
# the exact opposite of what the user asked for.
_EXPLICIT_MEN = re.compile(
    r"\b(men|man|mens|men's|male|guy|guys|boyfriend|husband|him|his|he|"
    r"gentleman|gents)\b", re.I
)
_EXPLICIT_WOMEN = re.compile(
    r"\b(women|woman|womens|women's|female|girl|girls|lady|ladies|"
    r"girlfriend|wife|her|she)\b", re.I
)
# Garments that merely *suggest* a gender. Only consulted when nothing explicit
# was said.
_IMPLIED_WOMEN = re.compile(r"\b(saree|sari|kurti|kurtis|dress|dresses|skirt|heels|blouse)\b", re.I)


def extract_gender(text: str, default: str = "Women") -> str:
    men = len(_EXPLICIT_MEN.findall(text))
    women = len(_EXPLICIT_WOMEN.findall(text))
    if men > women:
        return "Men"
    if women > men:
        return "Women"

    # Nothing explicit (or a genuine tie): fall back to garment hints.
    if _IMPLIED_WOMEN.search(text):
        return "Women"
    return default


# --------------------------------------------------------------------------
# Occasion and style
# --------------------------------------------------------------------------
# First match wins, so more specific phrases are listed first.
_OCCASION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("interview", ("interview", "presentation", "viva", "seminar", "conference",
                   "job talk", "placement")),
    ("work_office", ("office", "work", "meeting", "client", "corporate",
                     "business", "internship")),
    ("wedding_festive", ("wedding", "shaadi", "sangeet", "mehendi", "diwali",
                         "festive", "festival", "puja", "pooja", "engagement",
                         "reception", "haldi", "traditional", "ethnic")),
    ("party", ("party", "night out", "club", "clubbing", "birthday",
               "celebration", "new year")),
    ("date_night", ("date", "dinner", "brunch", "anniversary", "romantic")),
    ("workout", ("gym", "workout", "running", "jogging", "yoga", "sports",
                 "training", "exercise", "cricket", "football")),
    ("travel", ("travel", "trip", "vacation", "holiday", "flight", "airport",
                "road trip")),
    ("college", ("college", "campus", "class", "lecture", "university",
                 "school", "fresher")),
)

_STYLE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("smart_casual", ("smart casual", "smart-casual", "semi formal",
                      "semi-formal", "business casual")),
    ("formal", ("formal", "professional", "corporate", "suited", "office wear")),
    ("ethnic", ("ethnic", "traditional", "indian wear", "kurta", "saree",
                "sari", "kurti", "salwar", "churidar")),
    ("sporty", ("sporty", "athletic", "activewear", "gym wear", "sportswear")),
    ("party", ("party wear", "partywear", "glam", "dressy")),
    ("relaxed", ("relaxed", "comfy", "comfortable", "loungewear", "chill")),
    ("casual", ("casual", "everyday", "day to day", "laid back")),
)


def _first_keyword_match(
    text: str, table: tuple[tuple[str, tuple[str, ...]], ...]
) -> str | None:
    lowered = text.lower()
    for key, keywords in table:
        if any(keyword in lowered for keyword in keywords):
            return key
    return None


def extract_occasion(text: str) -> str:
    return _first_keyword_match(text, _OCCASION_KEYWORDS) or DEFAULT_OCCASION


def extract_style(text: str) -> str | None:
    return _first_keyword_match(text, _STYLE_KEYWORDS)


# --------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------
# Sorted longest-first so "navy blue" is matched before "blue".
_COLOR_TERMS = sorted(COLOR_KEYWORD_TO_FAMILY, key=len, reverse=True)


def _colour_words(text: str) -> list[str]:
    """Colour words present in the text, longest match first, each used once."""
    remaining = text.lower()
    found: list[str] = []
    for term in _COLOR_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", remaining):
            found.append(term)
            remaining = re.sub(rf"\b{re.escape(term)}\b", " ", remaining)
    return found


def extract_colors(text: str) -> list[str]:
    """Colour families mentioned anywhere in the text (soft preference)."""
    families: list[str] = []
    for term in _colour_words(text):
        family = COLOR_KEYWORD_TO_FAMILY[term]
        if family not in families:
            families.append(family)
    return families[:3]


# --------------------------------------------------------------------------
# Explicit item requests  ("a black shirt")
# --------------------------------------------------------------------------
# The rule that separates a hard constraint from a soft preference is
# ADJACENCY: a colour word immediately before a garment word names that
# garment's colour. A colour word floating on its own is a palette preference.
#
#   "black shirt"                       -> hard: shirt, colour black
#   "shirt"                             -> hard: shirt, any colour
#   "black and white, smart casual"     -> soft: neutral palette
#
# Garment phrases are matched longest-first so "formal shoes" wins over "shoes"
# and "track pants" wins over "pants".

_GARMENT_PHRASES = sorted(GARMENT_SYNONYMS, key=len, reverse=True)
_GARMENT_ALTERNATION = "|".join(re.escape(phrase) for phrase in _GARMENT_PHRASES)
_COLOUR_ALTERNATION = "|".join(
    re.escape(word) for word in sorted(EXPLICIT_COLOUR_MATCHES, key=len, reverse=True)
)

# An optional colour, optionally separated by one filler word ("dark", "solid",
# "plain"), immediately followed by a garment.
_ITEM_PATTERN = re.compile(
    rf"\b(?:(?P<colour>{_COLOUR_ALTERNATION})\s+(?:dark\s+|light\s+|solid\s+|plain\s+)?)?"
    rf"(?P<garment>{_GARMENT_ALTERNATION})\b",
    re.I,
)

# "dress code" is about formality, not about a dress.
_FALSE_GARMENT_CONTEXT = re.compile(r"\bdress\s+code\b", re.I)


def extract_item_requests(text: str) -> tuple[list[ItemRequest], set[str]]:
    """Find explicitly named garments.

    Returns the hard constraints plus the set of colour words that were consumed
    by them, so the caller can avoid double-counting those colours as a soft
    palette preference.
    """
    cleaned = _FALSE_GARMENT_CONTEXT.sub(" ", text)
    requests: list[ItemRequest] = []
    consumed_colours: set[str] = set()
    seen: set[tuple[str, str | None]] = set()

    for match in _ITEM_PATTERN.finditer(cleaned):
        garment_key = GARMENT_SYNONYMS.get(match.group("garment").lower())
        if garment_key is None:
            continue
        colour = match.group("colour")
        colour = colour.lower() if colour else None

        key = (garment_key, colour)
        if key in seen:
            continue
        seen.add(key)

        try:
            requests.append(ItemRequest(garment=garment_key, colour=colour))
        except Exception:
            continue
        if colour:
            consumed_colours.add(colour)

    return requests[:4], consumed_colours


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
_NO_ACCESSORY = re.compile(
    r"\b(no|without|skip|don'?t want|do not want)\s+"
    r"(accessor\w*|jewel\w*|watch\w*|bag|belt)", re.I
)


def parse_request(text: str, default_gender: str = "Women") -> StylePreferences:
    """Best-effort structured preferences from free text, using rules only."""
    required_items, consumed_colours = extract_item_requests(text)

    # A colour already attached to a garment is a hard constraint on that
    # garment, not a wish for the whole outfit. Counting it twice would push
    # every other slot towards the same colour - "black shirt" would quietly
    # become "everything black".
    soft_colours = [
        family for word, family in
        ((w, COLOR_KEYWORD_TO_FAMILY.get(w)) for w in _colour_words(text))
        if word not in consumed_colours and family
    ]
    deduped: list[str] = []
    for family in soft_colours:
        if family not in deduped:
            deduped.append(family)

    return StylePreferences(
        gender=extract_gender(text, default=default_gender),
        occasion=extract_occasion(text),
        style=extract_style(text),
        budget=extract_budget(text),
        preferred_colors=deduped[:3],
        required_items=required_items,
        include_accessory=not bool(_NO_ACCESSORY.search(text)),
        notes=None,
    )


def looks_unparseable(text: str, prefs: StylePreferences) -> bool:
    """True when we extracted essentially nothing and should say so.

    Used to show an honest "I did not understand that" message rather than
    silently returning a default everyday-casual outfit for gibberish input.
    """
    if len(text.strip()) < 3:
        return True
    signals = [
        prefs.occasion != DEFAULT_OCCASION,
        prefs.style is not None,
        prefs.budget is not None,
        bool(prefs.preferred_colors),
    ]
    has_fashion_word = bool(
        re.search(
            r"\b(wear|outfit|look|style|dress|shirt|jeans|shoes|clothes|"
            r"clothing|fashion|top|bottom|kurta|saree|casual|formal)\b",
            text, re.I,
        )
    )
    return not any(signals) and not has_fashion_word
