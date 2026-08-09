"""Deterministic feature engineering for the fashion catalog.

The source dataset ships raw metadata only (gender, masterCategory, subCategory,
articleType, baseColour, season, usage, productDisplayName). None of that is
directly usable by an outfit builder, so this module derives five explainable
features plus a synthetic price:

    outfit_slot      which part of an outfit the item occupies
    formality        0.0 (relaxed) .. 4.0 (very formal), on one interpretable scale
    formality_tier   human-readable bucket of `formality`
    color_family     ~10 coarse colour groups collapsed from ~46 baseColour values
    is_neutral       whether the colour pairs with (almost) anything
    is_ethnic        whether the item belongs to Indian ethnic wear
    price            SYNTHETIC — see `synthetic_price` for the honest caveat

Every table below is small, hand-curated and documented. There are no learned
weights and no unexplained magic numbers: an interviewer can read any single
mapping and see why it is what it is.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import pandas as pd

# --------------------------------------------------------------------------
# 1. OUTFIT SLOTS
# --------------------------------------------------------------------------
# An outfit is assembled from slots, not from raw article types. Mapping
# articleType -> slot is what turns a flat product catalog into something an
# outfit builder can reason about.
#
# ONEPIECE is a first-class slot: a dress or saree replaces TOP + BOTTOM, so the
# builder needs to know it satisfies both at once.

SLOT_TOP = "top"
SLOT_BOTTOM = "bottom"
SLOT_FOOTWEAR = "footwear"
SLOT_OUTERWEAR = "outerwear"
SLOT_ACCESSORY = "accessory"
SLOT_ONEPIECE = "onepiece"

ALL_SLOTS = (
    SLOT_TOP,
    SLOT_BOTTOM,
    SLOT_FOOTWEAR,
    SLOT_OUTERWEAR,
    SLOT_ACCESSORY,
    SLOT_ONEPIECE,
)

SLOT_LABELS = {
    SLOT_TOP: "Top",
    SLOT_BOTTOM: "Bottom",
    SLOT_FOOTWEAR: "Footwear",
    SLOT_OUTERWEAR: "Layer",
    SLOT_ACCESSORY: "Accessory",
    SLOT_ONEPIECE: "One-piece",
}

# articleType -> outfit slot. Anything not listed here (personal care, innerwear,
# home, sporting goods, free gifts) is deliberately excluded from the catalog:
# we cannot style with it, so it would only add noise.
ARTICLE_TYPE_TO_SLOT: dict[str, str] = {
    # --- tops ---
    "Shirts": SLOT_TOP,
    "Tshirts": SLOT_TOP,
    "Tops": SLOT_TOP,
    "Kurtas": SLOT_TOP,
    "Kurtis": SLOT_TOP,
    "Tunics": SLOT_TOP,
    "Sweatshirts": SLOT_TOP,
    "Sweaters": SLOT_TOP,
    "Shrug": SLOT_TOP,
    # --- bottoms ---
    "Jeans": SLOT_BOTTOM,
    "Trousers": SLOT_BOTTOM,
    "Shorts": SLOT_BOTTOM,
    "Track Pants": SLOT_BOTTOM,
    "Skirts": SLOT_BOTTOM,
    "Capris": SLOT_BOTTOM,
    "Leggings": SLOT_BOTTOM,
    "Churidar": SLOT_BOTTOM,
    "Patiala": SLOT_BOTTOM,
    "Salwar": SLOT_BOTTOM,
    # --- footwear ---
    "Casual Shoes": SLOT_FOOTWEAR,
    "Formal Shoes": SLOT_FOOTWEAR,
    "Sports Shoes": SLOT_FOOTWEAR,
    "Heels": SLOT_FOOTWEAR,
    "Flats": SLOT_FOOTWEAR,
    "Sandals": SLOT_FOOTWEAR,
    "Flip Flops": SLOT_FOOTWEAR,
    "Sports Sandals": SLOT_FOOTWEAR,
    # --- outerwear / layers ---
    "Jackets": SLOT_OUTERWEAR,
    "Blazers": SLOT_OUTERWEAR,
    "Rain Jacket": SLOT_OUTERWEAR,
    "Waistcoat": SLOT_OUTERWEAR,
    "Nehru Jackets": SLOT_OUTERWEAR,
    # --- accessories ---
    "Watches": SLOT_ACCESSORY,
    "Belts": SLOT_ACCESSORY,
    "Sunglasses": SLOT_ACCESSORY,
    "Wallets": SLOT_ACCESSORY,
    "Handbags": SLOT_ACCESSORY,
    "Clutches": SLOT_ACCESSORY,
    "Backpacks": SLOT_ACCESSORY,
    "Ties": SLOT_ACCESSORY,
    "Caps": SLOT_ACCESSORY,
    "Scarves": SLOT_ACCESSORY,
    "Stoles": SLOT_ACCESSORY,
    "Dupatta": SLOT_ACCESSORY,
    "Necklace and Chains": SLOT_ACCESSORY,
    "Earrings": SLOT_ACCESSORY,
    "Bracelet": SLOT_ACCESSORY,
    "Bangle": SLOT_ACCESSORY,
    "Ring": SLOT_ACCESSORY,
    "Jewellery Set": SLOT_ACCESSORY,
    "Pendant": SLOT_ACCESSORY,
    "Duffel Bag": SLOT_ACCESSORY,
    "Laptop Bag": SLOT_ACCESSORY,
    "Messenger Bag": SLOT_ACCESSORY,
    # --- one-piece looks ---
    "Dresses": SLOT_ONEPIECE,
    "Sarees": SLOT_ONEPIECE,
    "Jumpsuit": SLOT_ONEPIECE,
    "Rompers": SLOT_ONEPIECE,
}

# subCategory allowlist. A second, coarser gate so that a stray articleType that
# happens to share a name with something unwearable cannot slip through.
ALLOWED_SUBCATEGORIES = {
    "Topwear",
    "Bottomwear",
    "Shoes",
    "Sandal",
    "Flip Flops",
    "Dress",
    "Saree",
    "Watches",
    "Belts",
    "Eyewear",
    "Bags",
    "Wallets",
    "Ties",
    "Jewellery",
    "Headwear",
    "Scarves",
    "Stoles",
    "Accessories",
}


# --------------------------------------------------------------------------
# 2. FORMALITY  (single 0..4 scale, five named tiers)
# --------------------------------------------------------------------------
# Formality is the strongest signal in outfit compatibility: the single most
# common styling mistake is mixing items from very different formality levels
# (formal brogues with track pants). We therefore put every item on ONE numeric
# axis so "how compatible are these two?" reduces to a distance.
#
# The score is built from two small, defensible inputs:
#   base   = the dataset's own `usage` label (the strongest available signal)
#   delta  = a per-articleType nudge, only where the article type genuinely
#            carries formality information the `usage` label misses.
# The result is clipped to [0, 4].

USAGE_BASE_FORMALITY: dict[str, float] = {
    "Formal": 3.0,
    "Party": 2.5,
    "Smart Casual": 2.0,
    "Ethnic": 2.0,  # ethnic wear sits mid-scale; "ethnic-ness" is a separate axis
    "Casual": 1.0,
    "Travel": 0.75,
    "Sports": 0.75,
    "Home": 0.0,
}
DEFAULT_USAGE = "Casual"

# Only article types whose formality is not already captured by `usage`.
# This table does real work: the dataset labels 77% of items simply "Casual",
# so without it a casual shirt and a pair of flip-flops would score identically.
ARTICLE_TYPE_FORMALITY_DELTA: dict[str, float] = {
    # dress-code anchors
    "Formal Shoes": +1.0,
    "Ties": +1.0,
    "Blazers": +1.0,
    "Waistcoat": +1.0,
    "Sarees": +0.75,
    "Shirts": +0.5,
    "Trousers": +0.5,
    "Heels": +0.5,
    # explicitly informal items
    "Sports Shoes": -0.25,
    "Tshirts": -0.25,
    # A sandal is a step below a closed casual shoe at any usage level. Without
    # this, casual sandals pass the filter for smart-casual requests.
    "Sandals": -0.5,
    "Sweatshirts": -0.5,
    "Caps": -0.5,
    "Sports Sandals": -0.5,
    "Shorts": -0.75,
    "Track Pants": -0.75,
    "Flip Flops": -1.0,
}

# Bucket boundaries for the human-readable tier. Lower bound is inclusive.
# Calibrated against the table above so that the obvious cases land right:
#   Formal Shoes (Formal)    4.00 -> Very Formal
#   Shirt (Formal)           3.50 -> Formal
#   Shirt (Casual)           1.50 -> Smart Casual
#   Jeans / Tshirt (Casual)  1.00 / 0.75 -> Casual
#   Flip Flops (Casual)      0.00 -> Relaxed
FORMALITY_TIERS: tuple[tuple[float, str], ...] = (
    (3.75, "Very Formal"),
    (2.75, "Formal"),
    (1.50, "Smart Casual"),
    (0.75, "Casual"),
    (0.0, "Relaxed"),
)

FORMALITY_SCALE_MAX = 4.0


# Guardrail against label noise. The dataset contains items like a novelty
# t-shirt tagged usage="Formal"; taken at face value that would put a printed
# tee in an interview outfit. Some garments have a hard ceiling on how formal
# they can be, no matter what the label says.
FORMALITY_CEILING: dict[str, float] = {
    "Tshirts": 2.0,
    "Sweatshirts": 2.0,
    "Sports Shoes": 1.75,
    "Track Pants": 1.5,
    "Shorts": 1.5,
    "Flip Flops": 1.0,
}


def formality_score(usage: str | float | None, article_type: str) -> float:
    """Place an item on the 0..4 formality scale."""
    key = usage if isinstance(usage, str) and usage.strip() else DEFAULT_USAGE
    base = USAGE_BASE_FORMALITY.get(key, USAGE_BASE_FORMALITY[DEFAULT_USAGE])
    delta = ARTICLE_TYPE_FORMALITY_DELTA.get(article_type, 0.0)
    ceiling = FORMALITY_CEILING.get(article_type, FORMALITY_SCALE_MAX)
    return round(min(ceiling, max(0.0, base + delta)), 2)


def formality_tier(score: float) -> str:
    """Human-readable bucket, used in explanations and in the UI."""
    for lower_bound, label in FORMALITY_TIERS:
        if score >= lower_bound:
            return label
    return FORMALITY_TIERS[-1][1]


# --------------------------------------------------------------------------
# 2b. OCCASION AND STYLE VOCABULARY
# --------------------------------------------------------------------------
# This is the *closed vocabulary* the LLM is allowed to emit. Constraining the
# model to these enums is what makes its output safe to feed into a filter: an
# unknown occasion can never silently become an empty result set.
#
# Note we do NOT filter on the dataset's `usage` column directly. It is far too
# skewed for that (34,392 "Casual" vs 67 "Smart Casual"), so filtering
# `usage == "Smart Casual"` would return almost nothing. Instead each occasion
# maps to a target position on the formality scale, and we retrieve items whose
# derived formality falls inside a tolerance band. `usage` is then used as a
# softer preference signal.

OCCASION_PROFILES: dict[str, dict[str, Any]] = {
    "everyday_casual": {
        "label": "Everyday / Casual",
        "target_formality": 1.0,
        "tolerance": 1.0,
        "preferred_usage": {"Casual"},
    },
    "college": {
        "label": "College / Campus",
        "target_formality": 1.5,
        "tolerance": 1.0,
        "preferred_usage": {"Casual", "Smart Casual"},
    },
    "work_office": {
        "label": "Work / Office",
        "target_formality": 3.0,
        "tolerance": 1.25,
        "preferred_usage": {"Formal", "Smart Casual"},
    },
    "interview": {
        "label": "Interview / Presentation",
        "target_formality": 3.25,
        "tolerance": 1.25,
        "preferred_usage": {"Formal", "Smart Casual"},
    },
    "date_night": {
        "label": "Date Night",
        "target_formality": 2.25,
        "tolerance": 1.25,
        "preferred_usage": {"Smart Casual", "Party", "Casual"},
    },
    "party": {
        "label": "Party / Night Out",
        "target_formality": 2.5,
        "tolerance": 1.25,
        "preferred_usage": {"Party", "Smart Casual", "Formal"},
    },
    "wedding_festive": {
        "label": "Wedding / Festive",
        "target_formality": 2.75,
        "tolerance": 1.25,
        "preferred_usage": {"Ethnic", "Party"},
    },
    "travel": {
        "label": "Travel",
        "target_formality": 1.0,
        "tolerance": 1.0,
        "preferred_usage": {"Casual", "Travel", "Sports"},
    },
    "workout": {
        "label": "Workout / Sports",
        "target_formality": 0.5,
        "tolerance": 0.85,
        "preferred_usage": {"Sports"},
    },
}
DEFAULT_OCCASION = "everyday_casual"

# An explicit style request overrides the occasion's formality target, because
# the user stating "smart casual" is stronger evidence than us inferring it.
STYLE_TARGET_FORMALITY: dict[str, float] = {
    "relaxed": 0.5,
    "casual": 1.0,
    "sporty": 0.75,
    "smart_casual": 2.0,
    "party": 2.5,
    "ethnic": 2.0,
    "formal": 3.25,
}

# Styles that imply the outfit should be Indian ethnic wear.
ETHNIC_STYLES = {"ethnic"}
ETHNIC_OCCASIONS = {"wedding_festive"}


# When the user states a style outright ("smart casual"), that is a much
# stronger signal than a style we inferred from the occasion, so we retrieve
# from a narrower band around it. Without this, a request for "smart casual"
# admits plain casual items, and because a t-shirt/jeans/sneakers combination is
# extremely self-consistent it can out-score the shirt-and-trousers outfit the
# user actually asked for.
# Calibrated so that a "smart casual" request (target 2.0) admits shirts,
# chinos, jeans and casual shoes but excludes t-shirts and flip-flops, while a
# "formal" request (target 3.25) admits only genuine formalwear.
EXPLICIT_STYLE_TOLERANCE = 1.0


def resolve_formality_target(occasion: str, style: str | None) -> tuple[float, float]:
    """Return (target_formality, tolerance) for a request."""
    profile = OCCASION_PROFILES.get(occasion, OCCASION_PROFILES[DEFAULT_OCCASION])
    target = float(profile["target_formality"])
    tolerance = float(profile["tolerance"])
    if style in STYLE_TARGET_FORMALITY:
        target = STYLE_TARGET_FORMALITY[style]
        tolerance = EXPLICIT_STYLE_TOLERANCE
    return target, tolerance


# --------------------------------------------------------------------------
# 3. COLOUR
# --------------------------------------------------------------------------
# The dataset has ~46 baseColour values, which is far too granular for pairing
# rules. We collapse them into ~10 families, because colour-matching advice is
# given at family level ("blue goes with brown"), not at "Turquoise Blue" level.

COLOR_FAMILIES: dict[str, tuple[str, ...]] = {
    "neutral": (
        "Black", "White", "Off White", "Cream", "Grey", "Charcoal",
        "Grey Melange", "Beige", "Taupe", "Nude", "Skin", "Silver",
        "Steel", "Metallic", "Mushroom Brown",
    ),
    "blue": ("Blue", "Navy Blue", "Turquoise Blue", "Teal"),
    "green": ("Green", "Sea Green", "Olive", "Lime Green", "Fluorescent Green"),
    "red": ("Red", "Maroon", "Burgundy", "Rust"),
    "pink": ("Pink", "Peach", "Rose", "Magenta", "Mauve"),
    "purple": ("Purple", "Lavender"),
    "yellow": ("Yellow", "Mustard", "Gold", "Bronze", "Copper"),
    "orange": ("Orange",),
    "brown": ("Brown", "Coffee Brown", "Tan", "Khaki"),
    "multi": ("Multi",),
}

_COLOR_TO_FAMILY: dict[str, str] = {
    colour.lower(): family
    for family, colours in COLOR_FAMILIES.items()
    for colour in colours
}

# Families that behave as neutrals when styling. "brown" is included because tan
# and brown leather (shoes, belts, bags) pair with essentially any outfit, which
# is exactly the property `is_neutral` is meant to capture. Navy is handled
# explicitly for the same reason.
NEUTRAL_FAMILIES = {"neutral", "brown"}
NEUTRAL_EXCEPTIONS = {"navy blue"}


def color_family(base_colour: str | float | None) -> str:
    """Collapse a raw baseColour into one of ~10 families ('unknown' if absent)."""
    if not isinstance(base_colour, str) or not base_colour.strip():
        return "unknown"
    return _COLOR_TO_FAMILY.get(base_colour.strip().lower(), "other")


def is_neutral_color(base_colour: str | float | None) -> bool:
    """True when the colour coordinates with almost anything."""
    if isinstance(base_colour, str) and base_colour.strip().lower() in NEUTRAL_EXCEPTIONS:
        return True
    return color_family(base_colour) in NEUTRAL_FAMILIES


# Colour words a user might type, mapped onto our families. Used to check whether
# a recommended item honours a stated colour preference.
COLOR_KEYWORD_TO_FAMILY: dict[str, str] = {
    **{c.lower(): f for f, cs in COLOR_FAMILIES.items() for c in cs},
    "navy": "blue",
    "denim": "blue",
    "neutrals": "neutral",
    "neutral": "neutral",
    "monochrome": "neutral",
    "pastel": "pink",
    "earthy": "brown",
    "earth tones": "brown",
    "khaki": "brown",
    "beige": "neutral",
    "maroon": "red",
    "wine": "red",
}


def normalise_color_preference(colours: Iterable[str]) -> list[str]:
    """Map free-text colour words onto colour families, dropping unknowns."""
    families: list[str] = []
    for raw in colours:
        if not isinstance(raw, str):
            continue
        family = COLOR_KEYWORD_TO_FAMILY.get(raw.strip().lower())
        if family is None:
            family = color_family(raw)
        if family not in ("unknown", "other") and family not in families:
            families.append(family)
    return families


# --------------------------------------------------------------------------
# 3b. EXPLICIT ITEM REQUESTS
# --------------------------------------------------------------------------
# There is a critical difference between two things a user can say:
#
#   "a black shirt"                -> a HARD constraint. The top slot must be a
#                                     shirt, and it must be black. Returning a
#                                     white t-shirt is simply wrong.
#   "I like neutral colours"       -> a SOFT preference. It should influence
#                                     ranking, never empty the result set.
#
# The tables below support the first case. Note they do NOT use `color_family`:
# families exist for compatibility scoring, where "black goes with everything"
# is the useful abstraction. But black and white are BOTH in the `neutral`
# family, so a family-level match cannot tell "black shirt" from "white shirt".
# An explicit request therefore matches concrete `baseColour` values.

# Everyday word -> canonical garment key.
GARMENT_SYNONYMS: dict[str, str] = {
    # tops
    "shirt": "shirt", "shirts": "shirt", "blouse": "shirt",
    "t-shirt": "tshirt", "tshirt": "tshirt", "t shirt": "tshirt",
    "tee": "tshirt", "tees": "tshirt", "tshirts": "tshirt",
    "top": "top", "tops": "top",
    "kurta": "kurta", "kurtas": "kurta",
    "kurti": "kurti", "kurtis": "kurti",
    "tunic": "tunic", "tunics": "tunic",
    "sweater": "sweater", "sweaters": "sweater", "pullover": "sweater",
    "sweatshirt": "sweatshirt", "sweatshirts": "sweatshirt", "hoodie": "sweatshirt",
    # bottoms
    "jeans": "jeans", "denims": "jeans",
    "trousers": "trousers", "trouser": "trousers", "chinos": "trousers",
    "pants": "trousers", "slacks": "trousers",
    "shorts": "shorts",
    "track pants": "track_pants", "joggers": "track_pants", "trackpants": "track_pants",
    "skirt": "skirt", "skirts": "skirt",
    "leggings": "leggings", "jeggings": "leggings",
    "capri": "capri", "capris": "capri",
    "churidar": "churidar", "salwar": "churidar",
    # one-piece
    "dress": "dress", "dresses": "dress", "gown": "dress",
    "saree": "saree", "sarees": "saree", "sari": "saree",
    "jumpsuit": "jumpsuit", "romper": "jumpsuit",
    # layers
    "jacket": "jacket", "jackets": "jacket",
    "blazer": "blazer", "blazers": "blazer",
    "waistcoat": "waistcoat",
    # footwear
    "shoes": "shoes", "footwear": "shoes",
    "sneakers": "sneakers", "trainers": "sneakers", "casual shoes": "sneakers",
    "formal shoes": "formal_shoes", "oxfords": "formal_shoes",
    "brogues": "formal_shoes", "loafers": "formal_shoes", "derbies": "formal_shoes",
    "sports shoes": "sports_shoes", "running shoes": "sports_shoes",
    "heels": "heels", "stilettos": "heels", "pumps": "heels",
    "flats": "flats", "ballerinas": "flats",
    "sandals": "sandals", "sandal": "sandals",
    "flip flops": "flip_flops", "slippers": "flip_flops",
    # accessories
    "watch": "watch", "watches": "watch",
    "belt": "belt", "belts": "belt",
    "handbag": "handbag", "handbags": "handbag", "purse": "handbag",
    "bag": "bag", "bags": "bag",
    "backpack": "backpack", "backpacks": "backpack", "rucksack": "backpack",
    "clutch": "clutch",
    "sunglasses": "sunglasses", "shades": "sunglasses",
    "tie": "tie", "ties": "tie",
    "wallet": "wallet", "wallets": "wallet",
    "cap": "cap", "caps": "cap",
    "scarf": "scarf", "scarves": "scarf",
    "stole": "stole", "dupatta": "dupatta",
    "earrings": "earrings", "necklace": "necklace",
    "bracelet": "bracelet", "bangles": "bangle", "bangle": "bangle",
    "ring": "ring",
}

# Canonical garment key -> the catalog article types that satisfy it.
GARMENT_ARTICLE_TYPES: dict[str, tuple[str, ...]] = {
    "shirt": ("Shirts",),
    "tshirt": ("Tshirts",),
    "top": ("Tops",),
    "kurta": ("Kurtas",),
    "kurti": ("Kurtis",),
    "tunic": ("Tunics",),
    "sweater": ("Sweaters",),
    "sweatshirt": ("Sweatshirts",),
    "jeans": ("Jeans",),
    "trousers": ("Trousers",),
    "shorts": ("Shorts",),
    "track_pants": ("Track Pants",),
    "skirt": ("Skirts",),
    "leggings": ("Leggings",),
    "capri": ("Capris",),
    "churidar": ("Churidar", "Salwar", "Patiala"),
    "dress": ("Dresses",),
    "saree": ("Sarees",),
    "jumpsuit": ("Jumpsuit", "Rompers"),
    "jacket": ("Jackets", "Rain Jacket"),
    "blazer": ("Blazers",),
    "waistcoat": ("Waistcoat", "Nehru Jackets"),
    "shoes": ("Casual Shoes", "Formal Shoes", "Sports Shoes", "Flats", "Heels"),
    "sneakers": ("Casual Shoes", "Sports Shoes"),
    "formal_shoes": ("Formal Shoes",),
    "sports_shoes": ("Sports Shoes",),
    "heels": ("Heels",),
    "flats": ("Flats",),
    "sandals": ("Sandals", "Sports Sandals"),
    "flip_flops": ("Flip Flops",),
    "watch": ("Watches",),
    "belt": ("Belts",),
    "handbag": ("Handbags", "Clutches"),
    "bag": ("Handbags", "Backpacks", "Clutches", "Duffel Bag", "Laptop Bag",
            "Messenger Bag"),
    "backpack": ("Backpacks",),
    "clutch": ("Clutches",),
    "sunglasses": ("Sunglasses",),
    "tie": ("Ties",),
    "wallet": ("Wallets",),
    "cap": ("Caps",),
    "scarf": ("Scarves",),
    "stole": ("Stoles",),
    "dupatta": ("Dupatta",),
    "earrings": ("Earrings",),
    "necklace": ("Necklace and Chains", "Pendant"),
    "bracelet": ("Bracelet",),
    "bangle": ("Bangle",),
    "ring": ("Ring",),
}

CANONICAL_GARMENTS = tuple(sorted(GARMENT_ARTICLE_TYPES))

# Human labels for messages ("no black shirt matched your budget").
GARMENT_LABELS: dict[str, str] = {
    key: key.replace("_", " ") for key in GARMENT_ARTICLE_TYPES
}

# Colour word -> the concrete baseColour values that satisfy it. Deliberately
# tight: "black" must not quietly match charcoal, or the constraint stops
# meaning what the user said.
EXPLICIT_COLOUR_MATCHES: dict[str, tuple[str, ...]] = {
    "black": ("Black",),
    "white": ("White", "Off White"),
    "grey": ("Grey", "Grey Melange", "Charcoal"),
    "gray": ("Grey", "Grey Melange", "Charcoal"),
    "charcoal": ("Charcoal",),
    "blue": ("Blue", "Navy Blue", "Turquoise Blue"),
    "navy": ("Navy Blue",),
    "teal": ("Teal",),
    "turquoise": ("Turquoise Blue",),
    "red": ("Red",),
    "maroon": ("Maroon", "Burgundy"),
    "burgundy": ("Burgundy",),
    "rust": ("Rust",),
    "green": ("Green", "Sea Green", "Olive", "Lime Green", "Fluorescent Green"),
    "olive": ("Olive",),
    "yellow": ("Yellow", "Mustard"),
    "mustard": ("Mustard",),
    "orange": ("Orange",),
    "pink": ("Pink", "Rose", "Magenta"),
    "peach": ("Peach",),
    "magenta": ("Magenta",),
    "purple": ("Purple", "Lavender"),
    "lavender": ("Lavender",),
    "brown": ("Brown", "Coffee Brown", "Tan"),
    "tan": ("Tan",),
    "khaki": ("Khaki",),
    "beige": ("Beige", "Cream"),
    "cream": ("Cream",),
    "gold": ("Gold",),
    "silver": ("Silver",),
    "multi": ("Multi",),
    "printed": ("Multi",),
}

EXPLICIT_COLOUR_WORDS = tuple(sorted(EXPLICIT_COLOUR_MATCHES))


def garment_slot(garment: str) -> str | None:
    """Which outfit slot a canonical garment key occupies."""
    types = GARMENT_ARTICLE_TYPES.get(garment)
    if not types:
        return None
    for article_type in types:
        slot = ARTICLE_TYPE_TO_SLOT.get(article_type)
        if slot:
            return slot
    return None


# --------------------------------------------------------------------------
# 4. ETHNIC WEAR
# --------------------------------------------------------------------------
# Indian ethnic wear follows its own pairing logic (a kurta does not go with
# formal trousers), so it gets a dedicated boolean rather than being squeezed
# onto the formality axis.

ETHNIC_ARTICLE_TYPES = {
    "Kurtas", "Kurtis", "Sarees", "Churidar", "Patiala", "Salwar",
    "Dupatta", "Nehru Jackets", "Tunics", "Stoles",
}


def is_ethnic_item(usage: str | float | None, article_type: str) -> bool:
    if article_type in ETHNIC_ARTICLE_TYPES:
        return True
    return isinstance(usage, str) and usage.strip().lower() == "ethnic"


# --------------------------------------------------------------------------
# 5. SYNTHETIC PRICE
# --------------------------------------------------------------------------
# HONEST CAVEAT, repeated in the README and surfaced in the UI:
# `ashraq/fashion-product-images-small` contains NO price column. Budget is
# central to the product ("under Rs.3000"), so prices are generated here rather
# than pulled from a second dataset (which would not join reliably anyway).
#
# The generator is deterministic and defensible:
#   price = base(articleType) * usage_multiplier * jitter(product_id)
# `jitter` is derived from a SHA-256 hash of the product id, so the same product
# always gets the same price on every machine and every run - the catalog is
# reproducible, and tests can assert exact values.

BASE_PRICE_INR: dict[str, int] = {
    # tops
    "Tshirts": 599, "Shirts": 1299, "Tops": 799, "Kurtas": 1099,
    "Kurtis": 899, "Tunics": 899, "Sweatshirts": 1299, "Sweaters": 1599,
    "Shrug": 999,
    # bottoms
    "Jeans": 1799, "Trousers": 1599, "Shorts": 799, "Track Pants": 999,
    "Skirts": 999, "Capris": 899, "Leggings": 499, "Churidar": 799,
    "Patiala": 899, "Salwar": 899,
    # footwear
    "Casual Shoes": 1999, "Formal Shoes": 2499, "Sports Shoes": 2799,
    "Heels": 1499, "Flats": 999, "Sandals": 1199, "Flip Flops": 399,
    "Sports Sandals": 1299,
    # outerwear
    "Jackets": 2999, "Blazers": 3999, "Rain Jacket": 1999,
    "Waistcoat": 1799, "Nehru Jackets": 2299,
    # accessories
    "Watches": 2999, "Belts": 799, "Sunglasses": 1499, "Wallets": 899,
    "Handbags": 1799, "Clutches": 1099, "Backpacks": 1499, "Ties": 599,
    "Caps": 499, "Scarves": 599, "Stoles": 699, "Dupatta": 599,
    "Necklace and Chains": 899, "Earrings": 499, "Bracelet": 599,
    "Bangle": 499, "Ring": 699, "Jewellery Set": 1299, "Pendant": 699,
    "Duffel Bag": 1999, "Laptop Bag": 1999, "Messenger Bag": 1499,
    # one-piece
    "Dresses": 1999, "Sarees": 2499, "Jumpsuit": 1999, "Rompers": 1499,
}
DEFAULT_BASE_PRICE = 999

# Formal and party pieces retail higher than their casual equivalents.
USAGE_PRICE_MULTIPLIER: dict[str, float] = {
    "Formal": 1.25,
    "Party": 1.20,
    "Ethnic": 1.10,
    "Smart Casual": 1.05,
    "Casual": 1.00,
    "Travel": 0.95,
    "Sports": 1.00,
    "Home": 0.80,
}

# Spread around the base price. 0.45x..1.55x reproduces the long tail of a real
# marketplace listing page without producing absurd values.
_JITTER_LOW, _JITTER_HIGH = 0.45, 1.55


def synthetic_price(product_id: int | str, article_type: str, usage: str | float | None) -> int:
    """Deterministic, reproducible synthetic price in INR (charm-priced)."""
    base = BASE_PRICE_INR.get(article_type, DEFAULT_BASE_PRICE)
    usage_key = usage if isinstance(usage, str) and usage.strip() else DEFAULT_USAGE
    multiplier = USAGE_PRICE_MULTIPLIER.get(usage_key, 1.0)

    digest = hashlib.sha256(f"ai-stylist:{product_id}".encode()).digest()
    # Two bytes give 1/65536 resolution: plenty, and stable across platforms.
    unit = int.from_bytes(digest[:2], "big") / 65535
    jitter = _JITTER_LOW + unit * (_JITTER_HIGH - _JITTER_LOW)

    raw = base * multiplier * jitter
    # Charm pricing: round up to the next multiple of 50, then subtract 1
    # (Rs.1249, Rs.799, ...), exactly like a real marketplace.
    return max(99, int(math.ceil(raw / 50.0) * 50) - 1)


# Discount tiers, drawn from the same deterministic hash. About a third of the
# catalog carries no discount, which is what a real listing page looks like.
# Part of the SAME synthetic pricing layer as `synthetic_price` and disclosed
# together with it - the dataset has no MRP field either.
DISCOUNT_TIERS = (0, 0, 0, 20, 30, 30, 40, 40, 50, 60)


def synthetic_discount(product_id: int | str) -> int:
    """Deterministic discount percentage (0 means 'not on offer')."""
    digest = hashlib.sha256(f"ai-stylist-discount:{product_id}".encode()).digest()
    return DISCOUNT_TIERS[digest[0] % len(DISCOUNT_TIERS)]


def synthetic_mrp(price: int, discount_pct: int) -> int:
    """The struck-through 'original' price implied by a discount."""
    if discount_pct <= 0:
        return price
    mrp = price / (1 - discount_pct / 100)
    return int(math.ceil(mrp / 50.0) * 50) - 1


# --------------------------------------------------------------------------
# 6. APPLYING EVERYTHING TO A DATAFRAME
# --------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all derived columns. Pure function: returns a new DataFrame."""
    out = df.copy()
    out["usage"] = out["usage"].where(out["usage"].notna(), DEFAULT_USAGE)
    out["outfit_slot"] = out["articleType"].map(ARTICLE_TYPE_TO_SLOT)
    out["formality"] = [
        formality_score(u, a) for u, a in zip(out["usage"], out["articleType"])
    ]
    out["formality_tier"] = out["formality"].map(formality_tier)
    out["color_family"] = out["baseColour"].map(color_family)
    out["is_neutral"] = out["baseColour"].map(is_neutral_color)
    out["is_ethnic"] = [
        is_ethnic_item(u, a) for u, a in zip(out["usage"], out["articleType"])
    ]
    out["price"] = [
        synthetic_price(i, a, u)
        for i, a, u in zip(out["id"], out["articleType"], out["usage"])
    ]
    out["discount_pct"] = [synthetic_discount(i) for i in out["id"]]
    out["mrp"] = [
        synthetic_mrp(p, d) for p, d in zip(out["price"], out["discount_pct"])
    ]
    out["brand"] = out["productDisplayName"].map(extract_brand)
    return out


# The dataset packs brand and description into one `productDisplayName` string
# ("Lee Men Essential Black Shirt"). Marketplace cards lead with the brand, so we
# split it out. The first token is the brand for single-word brands; two tokens
# when the second is not a gender/descriptor word - which covers "Lee Cooper",
# "Indigo Nation", "Carlton London", "Red Tape".
_NON_BRAND_SECOND_WORDS = {
    "men", "men's", "mens", "women", "women's", "womens", "unisex", "boys",
    "girls", "kids", "man", "woman",
}


def extract_brand(display_name: str) -> str:
    words = str(display_name).split()
    if not words:
        return ""
    if len(words) > 2 and words[1].lower() not in _NON_BRAND_SECOND_WORDS:
        return " ".join(words[:2])
    return words[0]


def extract_short_name(display_name: str, brand: str) -> str:
    """The description line on a card: the display name minus the brand."""
    remainder = str(display_name)[len(brand):].strip()
    return remainder or str(display_name)


def wearable_mask(df: pd.DataFrame) -> pd.Series:
    """Rows we can actually style with (drops personal care, innerwear, home...)."""
    return (
        df["articleType"].isin(ARTICLE_TYPE_TO_SLOT)
        & df["subCategory"].isin(ALLOWED_SUBCATEGORIES)
        & df["masterCategory"].isin({"Apparel", "Footwear", "Accessories"})
    )
