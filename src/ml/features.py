"""Feature extraction for the pairwise compatibility model.

Features are RAW PRODUCT ATTRIBUTES ONLY - never the rule engine's scores, nor
anything derived from them.

That restriction is the point. The training labels come from the rule engine
(there is no real compatibility ground truth), so feeding the rule's own
sub-scores back in as features would let the model learn
`label = threshold(weighted_sum(inputs))` and report a meaningless ~0.99. Using
raw attributes forces it to induce the structure instead, which makes
"does it generalise to unseen garment types?" a real question.

Pairs are canonicalised into a fixed slot order first, since compatibility has
no direction.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

Product = Mapping[str, Any]

# Canonical ordering, so (trousers, shirt) and (shirt, trousers) featurise
# identically. Ordering is by outfit slot, then by product id as a tiebreak.
SLOT_ORDER = {
    "top": 0, "onepiece": 1, "bottom": 2, "outerwear": 3,
    "footwear": 4, "accessory": 5,
}

# --- the feature contract -------------------------------------------------
# Training and inference both import these lists, so the two can never drift.
CATEGORICAL_FEATURES = [
    "articleType_a", "articleType_b",
    "baseColour_a", "baseColour_b",
    "usage_a", "usage_b",
    "season_a", "season_b",
    "slot_a", "slot_b",
    "gender_a", "gender_b",
]

NUMERIC_FEATURES = [
    "formality_a", "formality_b",
    "price_a", "price_b",
    "price_ratio",
]

BOOLEAN_FEATURES = [
    "is_neutral_a", "is_neutral_b",
    "is_ethnic_a", "is_ethnic_b",
    "same_colour",
    "same_usage",
    "same_season",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOLEAN_FEATURES


def canonical_order(item_a: Product, item_b: Product) -> tuple[Product, Product]:
    """Put a pair into a deterministic (a, b) order."""
    key_a = (SLOT_ORDER.get(item_a["outfit_slot"], 9), int(item_a["id"]))
    key_b = (SLOT_ORDER.get(item_b["outfit_slot"], 9), int(item_b["id"]))
    return (item_a, item_b) if key_a <= key_b else (item_b, item_a)


def pair_features(item_a: Product, item_b: Product) -> dict[str, Any]:
    """Raw-attribute features for one pair. No rule scores anywhere."""
    a, b = canonical_order(item_a, item_b)

    price_a, price_b = float(a["price"]), float(b["price"])
    # Ratio of the cheaper to the dearer item. Nothing in the rule engine uses
    # price for compatibility at all, so this is genuinely new information: it
    # captures whether two items sit at the same market tier.
    price_ratio = min(price_a, price_b) / max(price_a, price_b, 1.0)

    return {
        "articleType_a": str(a["articleType"]), "articleType_b": str(b["articleType"]),
        "baseColour_a": str(a["baseColour"]), "baseColour_b": str(b["baseColour"]),
        "usage_a": str(a["usage"]), "usage_b": str(b["usage"]),
        "season_a": str(a.get("season", "Unknown")),
        "season_b": str(b.get("season", "Unknown")),
        "slot_a": str(a["outfit_slot"]), "slot_b": str(b["outfit_slot"]),
        "gender_a": str(a["gender"]), "gender_b": str(b["gender"]),

        "formality_a": float(a["formality"]), "formality_b": float(b["formality"]),
        "price_a": price_a, "price_b": price_b,
        "price_ratio": round(price_ratio, 4),

        "is_neutral_a": bool(a["is_neutral"]), "is_neutral_b": bool(b["is_neutral"]),
        "is_ethnic_a": bool(a["is_ethnic"]), "is_ethnic_b": bool(b["is_ethnic"]),

        # Raw equality checks on the source labels - not colour *families*, which
        # is what the rule engine reasons over.
        "same_colour": str(a["baseColour"]) == str(b["baseColour"]),
        "same_usage": str(a["usage"]) == str(b["usage"]),
        "same_season": str(a.get("season")) == str(b.get("season")),
    }


def features_frame(pairs: list[tuple[Product, Product]]) -> pd.DataFrame:
    """Featurise many pairs into a DataFrame with exactly FEATURE_COLUMNS."""
    rows = [pair_features(a, b) for a, b in pairs]
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return frame
