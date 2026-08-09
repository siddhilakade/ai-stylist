"""Feature engineering: the derived columns everything else depends on."""

from __future__ import annotations

import pytest

from src.features import (
    ARTICLE_TYPE_TO_SLOT,
    FORMALITY_CEILING,
    SLOT_BOTTOM,
    SLOT_FOOTWEAR,
    SLOT_ONEPIECE,
    SLOT_TOP,
    color_family,
    formality_score,
    formality_tier,
    is_ethnic_item,
    is_neutral_color,
    normalise_color_preference,
    resolve_formality_target,
    synthetic_price,
)


class TestOutfitSlots:
    @pytest.mark.parametrize(
        "article_type,expected",
        [
            ("Shirts", SLOT_TOP),
            ("Kurtas", SLOT_TOP),
            ("Jeans", SLOT_BOTTOM),
            ("Formal Shoes", SLOT_FOOTWEAR),
            ("Sarees", SLOT_ONEPIECE),
            ("Dresses", SLOT_ONEPIECE),
        ],
    )
    def test_known_article_types_map_to_expected_slot(self, article_type, expected):
        assert ARTICLE_TYPE_TO_SLOT[article_type] == expected

    def test_unwearable_categories_are_absent(self):
        # These exist in the source dataset and must never become outfit items.
        for article_type in ("Lipstick", "Deodorant", "Bra", "Briefs", "Nail Polish"):
            assert article_type not in ARTICLE_TYPE_TO_SLOT


class TestFormality:
    def test_scale_is_ordered_the_way_a_human_would_order_it(self):
        casual_tee = formality_score("Casual", "Tshirts")
        casual_shirt = formality_score("Casual", "Shirts")
        formal_shirt = formality_score("Formal", "Shirts")
        formal_shoes = formality_score("Formal", "Formal Shoes")
        flip_flops = formality_score("Casual", "Flip Flops")

        assert flip_flops < casual_tee < casual_shirt < formal_shirt <= formal_shoes

    def test_tiers_match_the_documented_calibration(self):
        assert formality_tier(formality_score("Formal", "Formal Shoes")) == "Very Formal"
        assert formality_tier(formality_score("Formal", "Shirts")) == "Formal"
        assert formality_tier(formality_score("Casual", "Shirts")) == "Smart Casual"
        assert formality_tier(formality_score("Casual", "Jeans")) == "Casual"
        assert formality_tier(formality_score("Casual", "Flip Flops")) == "Relaxed"

    def test_ceiling_defends_against_mislabelled_data(self):
        # The dataset really does contain t-shirts tagged usage="Formal".
        assert formality_score("Formal", "Tshirts") == FORMALITY_CEILING["Tshirts"]
        assert formality_score("Formal", "Tshirts") < formality_score("Formal", "Shirts")

    def test_missing_usage_falls_back_to_casual(self):
        assert formality_score(None, "Jeans") == formality_score("Casual", "Jeans")
        assert formality_score(float("nan"), "Jeans") == formality_score("Casual", "Jeans")

    def test_score_is_always_inside_the_scale(self):
        for usage in ("Formal", "Casual", "Sports", "Home", None):
            for article_type in ARTICLE_TYPE_TO_SLOT:
                assert 0.0 <= formality_score(usage, article_type) <= 4.0

    def test_explicit_style_narrows_the_retrieval_band(self):
        _, inferred = resolve_formality_target("interview", None)
        target, explicit = resolve_formality_target("interview", "smart_casual")
        assert explicit < inferred
        assert target == 2.0  # the style's target, not the occasion's


class TestColour:
    def test_families_collapse_shades(self):
        assert color_family("Navy Blue") == "blue"
        assert color_family("Turquoise Blue") == "blue"
        assert color_family("Coffee Brown") == "brown"
        assert color_family("Off White") == "neutral"

    def test_unknown_and_missing_colours_are_handled(self):
        assert color_family(None) == "unknown"
        assert color_family("") == "unknown"
        assert color_family("Chartreuse") == "other"

    def test_neutrals(self):
        for colour in ("Black", "White", "Grey", "Beige", "Navy Blue", "Tan"):
            assert is_neutral_color(colour), colour
        for colour in ("Red", "Lime Green", "Magenta"):
            assert not is_neutral_color(colour), colour

    def test_user_colour_words_map_onto_families(self):
        assert normalise_color_preference(["navy", "white"]) == ["blue", "neutral"]
        assert normalise_color_preference(["neutrals"]) == ["neutral"]
        assert normalise_color_preference(["banana"]) == []


class TestEthnic:
    def test_article_type_and_usage_both_count(self):
        assert is_ethnic_item("Casual", "Kurtas")
        assert is_ethnic_item("Ethnic", "Shirts")
        assert not is_ethnic_item("Casual", "Jeans")


class TestSyntheticPrice:
    def test_is_deterministic(self):
        first = synthetic_price(15970, "Shirts", "Formal")
        second = synthetic_price(15970, "Shirts", "Formal")
        assert first == second

    def test_spreads_across_a_realistic_range(self):
        prices = [synthetic_price(i, "Tshirts", "Casual") for i in range(200)]
        # Charm pricing quantises to multiples of 50, so the number of distinct
        # values is bounded by the range - what matters is that the spread is
        # wide and the distribution does not collapse onto one price point.
        assert len(set(prices)) >= 10
        assert max(prices) >= 2 * min(prices)

    def test_respects_category_ordering_on_average(self):
        tees = [synthetic_price(i, "Tshirts", "Casual") for i in range(200)]
        blazers = [synthetic_price(i, "Blazers", "Formal") for i in range(200)]
        assert sum(blazers) / len(blazers) > sum(tees) / len(tees)

    def test_is_always_a_plausible_positive_price(self):
        for i in range(300):
            price = synthetic_price(i, "Shirts", "Casual")
            assert 99 <= price <= 20000
            assert price % 50 == 49, "charm pricing: should end in 49 or 99"
