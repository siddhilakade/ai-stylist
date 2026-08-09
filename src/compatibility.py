"""Rule-based compatibility scoring between catalog items.

Compatibility of a pair is a weighted sum of four signals, each in [0, 1]:

    colour     0.30   most visible to a user
    formality  0.30   the most common real styling error is a formality mismatch
    occasion   0.25   items should suit the same context
    ethnic     0.15   ethnic wear pairs by its own logic; smaller weight because
                      it only discriminates on part of the catalog

No learned model here: there are no compatibility labels to train on, and the
app has to be able to tell a user *why* two items go together.

Weights are round numbers with stated reasons, not tuned against the evaluation
set - fitting four weights to 34 scenarios would measure nothing.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

from src.features import FORMALITY_SCALE_MAX

Item = Mapping[str, Any]

# --------------------------------------------------------------------------
# Signal weights
# --------------------------------------------------------------------------
WEIGHT_COLOR = 0.30
WEIGHT_FORMALITY = 0.30
WEIGHT_OCCASION = 0.25
WEIGHT_ETHNIC = 0.15

SIGNAL_WEIGHTS = {
    "color": WEIGHT_COLOR,
    "formality": WEIGHT_FORMALITY,
    "occasion": WEIGHT_OCCASION,
    "ethnic": WEIGHT_ETHNIC,
}


# --------------------------------------------------------------------------
# 1. Colour compatibility
# --------------------------------------------------------------------------
# Rules in priority order, straight out of everyday styling advice:
#   both neutral              -> 1.00  (black + white always works)
#   one neutral               -> 0.90  (a neutral anchors any colour)
#   same family               -> 0.72  (tonal; safe but visually flat)
#   known harmonious families -> 0.75  (colour-wheel neighbours / classics)
#   anything with a print     -> 0.55  ("Multi" is busy, pair with care)
#   two unrelated strong hues -> 0.35  (red top + green trousers)

SCORE_BOTH_NEUTRAL = 1.00
SCORE_ONE_NEUTRAL = 0.90
SCORE_HARMONIOUS = 0.75
SCORE_SAME_FAMILY = 0.72
SCORE_WITH_MULTI = 0.55
SCORE_CLASHING = 0.35

# Family pairs that are conventionally flattering: colour-wheel neighbours
# (blue/green, red/pink) plus long-standing menswear classics (blue/brown).
HARMONIOUS_COLOR_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in (
        ("blue", "brown"),     # navy + tan: the classic smart-casual pairing
        ("green", "brown"),    # earth tones
        ("brown", "orange"),
        ("brown", "yellow"),
        ("blue", "green"),     # analogous
        ("blue", "purple"),    # analogous
        ("purple", "pink"),    # analogous
        ("red", "pink"),       # analogous
        ("red", "orange"),     # analogous
        ("yellow", "orange"),  # analogous
        ("blue", "orange"),    # complementary
        ("blue", "yellow"),    # complementary
        ("purple", "yellow"),  # complementary
    )
)


def color_pair_score(item_a: Item, item_b: Item) -> float:
    fam_a, fam_b = item_a["color_family"], item_b["color_family"]
    neutral_a, neutral_b = bool(item_a["is_neutral"]), bool(item_b["is_neutral"])

    if neutral_a and neutral_b:
        return SCORE_BOTH_NEUTRAL
    if neutral_a or neutral_b:
        return SCORE_ONE_NEUTRAL
    if "multi" in (fam_a, fam_b):
        return SCORE_WITH_MULTI
    if fam_a == fam_b:
        return SCORE_SAME_FAMILY
    if frozenset((fam_a, fam_b)) in HARMONIOUS_COLOR_PAIRS:
        return SCORE_HARMONIOUS
    return SCORE_CLASHING


# --------------------------------------------------------------------------
# 2. Formality compatibility
# --------------------------------------------------------------------------
# Both items sit on the same 0..4 scale, so compatibility is simply how close
# they are. Identical formality -> 1.0; opposite ends (flip-flops + blazer) -> 0.

def formality_pair_score(item_a: Item, item_b: Item) -> float:
    distance = abs(float(item_a["formality"]) - float(item_b["formality"]))
    return round(max(0.0, 1.0 - distance / FORMALITY_SCALE_MAX), 4)


# --------------------------------------------------------------------------
# 3. Occasion compatibility
# --------------------------------------------------------------------------
# The dataset's `usage` label is a coarse occasion tag. Rather than write a full
# 8x8 matrix of magic numbers, we list only the pairs that are clearly good or
# clearly bad and default everything else to "neutral" (0.5).

SCORE_SAME_USAGE = 1.0
SCORE_ADJACENT_USAGE = 0.75
SCORE_DEFAULT_USAGE = 0.5
SCORE_CONFLICTING_USAGE = 0.25

ADJACENT_USAGE_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in (
        ("Casual", "Smart Casual"),
        ("Smart Casual", "Formal"),
        ("Smart Casual", "Party"),
        ("Formal", "Party"),
        ("Casual", "Travel"),
        ("Casual", "Sports"),
        ("Ethnic", "Party"),
        ("Travel", "Sports"),
        ("Ethnic", "Casual"),  # kurta + jeans is a normal Indian outfit
    )
)

CONFLICTING_USAGE_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in (
        ("Formal", "Sports"),
        ("Formal", "Casual"),
        ("Formal", "Travel"),
        ("Ethnic", "Sports"),
        ("Ethnic", "Formal"),
        ("Party", "Sports"),
    )
)


def occasion_pair_score(item_a: Item, item_b: Item) -> float:
    usage_a, usage_b = item_a["usage"], item_b["usage"]
    if usage_a == usage_b:
        return SCORE_SAME_USAGE
    pair = frozenset((usage_a, usage_b))
    if pair in ADJACENT_USAGE_PAIRS:
        return SCORE_ADJACENT_USAGE
    if pair in CONFLICTING_USAGE_PAIRS:
        return SCORE_CONFLICTING_USAGE
    return SCORE_DEFAULT_USAGE


# --------------------------------------------------------------------------
# 4. Ethnic coherence
# --------------------------------------------------------------------------
# Ethnic wear mixes well with itself and with plain casual staples (jeans,
# leggings, flats), but not with western formalwear or sportswear.

WESTERN_FORMAL_TYPES = {"Blazers", "Ties", "Formal Shoes", "Waistcoat", "Trousers"}
SPORTS_TYPES = {"Sports Shoes", "Track Pants", "Sports Sandals", "Sweatshirts"}


def ethnic_pair_score(item_a: Item, item_b: Item) -> float:
    ethnic_a, ethnic_b = bool(item_a["is_ethnic"]), bool(item_b["is_ethnic"])
    if ethnic_a == ethnic_b:
        return 1.0  # both ethnic, or neither - no conflict either way

    western = item_b if ethnic_a else item_a
    if western["articleType"] in WESTERN_FORMAL_TYPES:
        return 0.25
    if western["articleType"] in SPORTS_TYPES:
        return 0.2
    return 0.65  # e.g. kurta + jeans: acceptable, not ideal


# --------------------------------------------------------------------------
# 5. Hard clashes
# --------------------------------------------------------------------------
# A short blocklist of combinations that no score should be able to rescue.
# These are hard constraints, applied by the outfit builder before scoring.

HARD_CLASHES: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in (
        ("Formal Shoes", "Shorts"),
        ("Formal Shoes", "Track Pants"),
        ("Formal Shoes", "Sarees"),
        ("Heels", "Track Pants"),
        ("Blazers", "Shorts"),
        ("Blazers", "Track Pants"),
        ("Ties", "Tshirts"),
        ("Flip Flops", "Blazers"),
        ("Flip Flops", "Trousers"),
        ("Sports Shoes", "Sarees"),
        ("Sports Shoes", "Blazers"),
    )
)


def is_gender_conflict(item_a: Item, item_b: Item) -> bool:
    """Menswear and womenswear must not be mixed inside one outfit.

    This only bites when the request did not pin a gender: a "Unisex" request
    draws from the whole catalog, and without this rule the builder is free to
    pair a men's shirt with a women's skirt. Unisex items (bags, caps, eyewear)
    are compatible with everything.
    """
    gender_a, gender_b = item_a["gender"], item_b["gender"]
    if "Unisex" in (gender_a, gender_b):
        return False
    return gender_a != gender_b


def is_hard_clash(item_a: Item, item_b: Item) -> bool:
    if is_gender_conflict(item_a, item_b):
        return True
    return frozenset((item_a["articleType"], item_b["articleType"])) in HARD_CLASHES


# --------------------------------------------------------------------------
# 6. Pair and outfit level compatibility
# --------------------------------------------------------------------------

def pair_signals(item_a: Item, item_b: Item) -> dict[str, float]:
    """The four raw signals for one pair of items."""
    return {
        "color": color_pair_score(item_a, item_b),
        "formality": formality_pair_score(item_a, item_b),
        "occasion": occasion_pair_score(item_a, item_b),
        "ethnic": ethnic_pair_score(item_a, item_b),
    }


def pair_compatibility(item_a: Item, item_b: Item) -> float:
    """Weighted compatibility of one pair, in [0, 1]. 0.0 for hard clashes."""
    if is_hard_clash(item_a, item_b):
        return 0.0
    signals = pair_signals(item_a, item_b)
    return round(sum(signals[name] * w for name, w in SIGNAL_WEIGHTS.items()), 4)


# More than two strong (non-neutral) colour families in one outfit reads as
# "busy". A flat 0.85 multiplier nudges the ranker towards calmer outfits
# without ever hard-blocking a colourful look.
BUSY_OUTFIT_PENALTY = 0.85
MAX_STRONG_COLOR_FAMILIES = 2


def outfit_signals(items: Sequence[Item]) -> dict[str, float]:
    """Average each signal over every pair in the outfit.

    This dict is the *grounding payload*: it is what the UI renders as bullet
    points and what is handed to Gemini to narrate. Nothing else is passed.
    """
    if len(items) < 2:
        return {"color": 1.0, "formality": 1.0, "occasion": 1.0, "ethnic": 1.0}

    pairs = list(combinations(items, 2))
    totals = {"color": 0.0, "formality": 0.0, "occasion": 0.0, "ethnic": 0.0}
    for a, b in pairs:
        for name, value in pair_signals(a, b).items():
            totals[name] += value

    averaged = {name: round(total / len(pairs), 4) for name, total in totals.items()}

    strong_families = {
        item["color_family"]
        for item in items
        if not item["is_neutral"] and item["color_family"] not in ("unknown", "other")
    }
    if len(strong_families) > MAX_STRONG_COLOR_FAMILIES:
        averaged["color"] = round(averaged["color"] * BUSY_OUTFIT_PENALTY, 4)

    return averaged


def outfit_compatibility(items: Sequence[Item]) -> float:
    """Single compatibility number for a whole outfit, in [0, 1]."""
    signals = outfit_signals(items)
    return round(sum(signals[name] * w for name, w in SIGNAL_WEIGHTS.items()), 4)


def has_hard_clash(items: Sequence[Item]) -> bool:
    return any(is_hard_clash(a, b) for a, b in combinations(items, 2))
