"""Shared fixtures.

`make_item` builds a minimal product dict with exactly the derived fields the
scoring code reads. Using synthetic items rather than real catalog rows keeps the
compatibility tests readable and independent of whatever happens to be in the
catalog on a given day.
"""

from __future__ import annotations

import pytest

from src.features import (
    color_family,
    formality_score,
    formality_tier,
    is_ethnic_item,
    is_neutral_color,
)

_ID_COUNTER = iter(range(900_000, 999_999))


@pytest.fixture
def make_item():
    def _make(
        article_type: str,
        colour: str = "Black",
        usage: str = "Casual",
        slot: str | None = None,
        price: int = 999,
        gender: str = "Men",
        product_id: int | None = None,
    ) -> dict:
        from src.features import ARTICLE_TYPE_TO_SLOT

        formality = formality_score(usage, article_type)
        return {
            "id": product_id if product_id is not None else next(_ID_COUNTER),
            "productDisplayName": f"Test {colour} {article_type}",
            "gender": gender,
            "articleType": article_type,
            "baseColour": colour,
            "usage": usage,
            "outfit_slot": slot or ARTICLE_TYPE_TO_SLOT[article_type],
            "formality": formality,
            "formality_tier": formality_tier(formality),
            "color_family": color_family(colour),
            "is_neutral": is_neutral_color(colour),
            "is_ethnic": is_ethnic_item(usage, article_type),
            "price": price,
            "image_file": "missing.jpg",
        }

    return _make
