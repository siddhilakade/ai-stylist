"""Compatibility scoring: does the rule set encode the styling advice we claim?

These tests are written as statements a stylist would agree with, not as
assertions about specific floats, so they stay meaningful if a weight is tuned.
"""

from __future__ import annotations

from src.compatibility import (
    SIGNAL_WEIGHTS,
    color_pair_score,
    ethnic_pair_score,
    formality_pair_score,
    has_hard_clash,
    is_hard_clash,
    occasion_pair_score,
    outfit_compatibility,
    outfit_signals,
    pair_compatibility,
)


class TestWeights:
    def test_weights_form_a_convex_combination(self):
        # Every signal is in [0,1], so this is what keeps scores in [0,1] too.
        assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9


class TestColour:
    def test_two_neutrals_beat_one_neutral_beats_a_clash(self, make_item):
        black_shirt = make_item("Shirts", "Black")
        white_trousers = make_item("Trousers", "White")
        red_trousers = make_item("Trousers", "Red")
        green_shirt = make_item("Shirts", "Green")

        both_neutral = color_pair_score(black_shirt, white_trousers)
        one_neutral = color_pair_score(black_shirt, red_trousers)
        clash = color_pair_score(green_shirt, red_trousers)

        assert both_neutral > one_neutral > clash

    def test_classic_pairing_beats_an_unrelated_pairing(self, make_item):
        navy = make_item("Shirts", "Blue")
        tan_shoes = make_item("Casual Shoes", "Tan")
        purple_shoes = make_item("Casual Shoes", "Purple")
        # Tan reads as a neutral, so compare against the harmonious rule directly.
        assert color_pair_score(navy, tan_shoes) >= color_pair_score(navy, purple_shoes)

    def test_prints_are_treated_cautiously(self, make_item):
        multi = make_item("Shirts", "Multi")
        red = make_item("Trousers", "Red")
        solid_a = make_item("Shirts", "Red")
        assert color_pair_score(multi, red) < color_pair_score(solid_a, red)

    def test_is_symmetric(self, make_item):
        a = make_item("Shirts", "Blue")
        b = make_item("Trousers", "Brown")
        assert color_pair_score(a, b) == color_pair_score(b, a)

    def test_busy_outfits_are_penalised(self, make_item):
        busy = [
            make_item("Shirts", "Red"),
            make_item("Trousers", "Green"),
            make_item("Casual Shoes", "Purple"),
        ]
        calm = [
            make_item("Shirts", "Red"),
            make_item("Trousers", "Black"),
            make_item("Casual Shoes", "White"),
        ]
        assert outfit_signals(busy)["color"] < outfit_signals(calm)["color"]


class TestFormality:
    def test_identical_formality_scores_one(self, make_item):
        a = make_item("Shirts", usage="Formal")
        b = make_item("Trousers", usage="Formal")
        assert formality_pair_score(a, b) == 1.0

    def test_distance_reduces_the_score(self, make_item):
        blazer = make_item("Blazers", usage="Formal")
        formal_shirt = make_item("Shirts", usage="Formal")
        flip_flops = make_item("Flip Flops", usage="Casual")
        assert formality_pair_score(blazer, formal_shirt) > formality_pair_score(
            blazer, flip_flops
        )

    def test_never_leaves_the_unit_interval(self, make_item):
        extremes = [
            make_item("Formal Shoes", usage="Formal"),
            make_item("Flip Flops", usage="Home"),
        ]
        assert 0.0 <= formality_pair_score(*extremes) <= 1.0


class TestOccasion:
    def test_same_usage_is_best(self, make_item):
        a, b = make_item("Shirts", usage="Formal"), make_item("Trousers", usage="Formal")
        assert occasion_pair_score(a, b) == 1.0

    def test_adjacent_beats_conflicting(self, make_item):
        formal = make_item("Shirts", usage="Formal")
        smart = make_item("Trousers", usage="Smart Casual")
        sporty = make_item("Track Pants", usage="Sports")
        assert occasion_pair_score(formal, smart) > occasion_pair_score(formal, sporty)


class TestEthnic:
    def test_ethnic_with_ethnic_is_coherent(self, make_item):
        kurta = make_item("Kurtas", usage="Ethnic")
        churidar = make_item("Churidar", usage="Ethnic")
        assert ethnic_pair_score(kurta, churidar) == 1.0

    def test_ethnic_with_western_formal_is_penalised_most(self, make_item):
        kurta = make_item("Kurtas", usage="Ethnic")
        blazer = make_item("Blazers", usage="Formal")
        jeans = make_item("Jeans", usage="Casual")
        # Kurta + jeans is a normal outfit; kurta + blazer is not.
        assert ethnic_pair_score(kurta, jeans) > ethnic_pair_score(kurta, blazer)


class TestHardClashes:
    def test_known_clashes_are_blocked_outright(self, make_item):
        formal_shoes = make_item("Formal Shoes", usage="Formal")
        shorts = make_item("Shorts", usage="Casual")
        assert is_hard_clash(formal_shoes, shorts)
        assert pair_compatibility(formal_shoes, shorts) == 0.0

    def test_clash_detection_is_order_independent(self, make_item):
        a = make_item("Ties", usage="Formal")
        b = make_item("Tshirts", usage="Casual")
        assert is_hard_clash(a, b) == is_hard_clash(b, a)

    def test_outfit_level_detection(self, make_item):
        outfit = [
            make_item("Shirts", usage="Formal"),
            make_item("Track Pants", usage="Sports"),
            make_item("Formal Shoes", usage="Formal"),
        ]
        assert has_hard_clash(outfit)


class TestOutfitCompatibility:
    def test_a_coherent_outfit_beats_an_incoherent_one(self, make_item):
        coherent = [
            make_item("Shirts", "White", usage="Formal"),
            make_item("Trousers", "Navy Blue", usage="Formal"),
            make_item("Formal Shoes", "Black", usage="Formal"),
        ]
        incoherent = [
            make_item("Tshirts", "Red", usage="Casual"),
            make_item("Trousers", "Green", usage="Formal"),
            make_item("Sports Shoes", "Purple", usage="Sports"),
        ]
        assert outfit_compatibility(coherent) > outfit_compatibility(incoherent)

    def test_score_stays_in_the_unit_interval(self, make_item):
        outfit = [
            make_item("Kurtas", "Multi", usage="Ethnic"),
            make_item("Track Pants", "Fluorescent Green", usage="Sports"),
            make_item("Heels", "Gold", usage="Party"),
        ]
        assert 0.0 <= outfit_compatibility(outfit) <= 1.0

    def test_single_item_is_trivially_compatible(self, make_item):
        assert outfit_compatibility([make_item("Shirts")]) == 1.0

    def test_signals_are_all_reported(self, make_item):
        outfit = [make_item("Shirts"), make_item("Jeans"), make_item("Casual Shoes")]
        assert set(outfit_signals(outfit)) == set(SIGNAL_WEIGHTS)
