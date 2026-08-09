"""End-to-end recommendation behaviour against the real catalog.

The most important tests in the suite are in `TestNoHallucination`: they assert
the property the whole architecture exists to guarantee - that every product
shown to a user came out of the catalog.
"""

from __future__ import annotations

import pytest

from src.data import catalog_records, get_product, load_catalog, product_ids
from src.features import OCCASION_PROFILES, SLOT_FOOTWEAR, resolve_formality_target
from src.outfit_builder import TEMPLATE_ONEPIECE, TEMPLATE_STANDARD
from src.recommender import (
    ACCESSORY_TOLERANCE_BONUS,
    complete_the_look,
    grounding_payload,
    hard_filter,
    infer_preferences_from_product,
    outfit_reasons,
    preference_match,
    recommend_outfits,
    score_outfit,
)
from src.schemas import StylePreferences


@pytest.fixture(scope="module")
def catalog():
    return list(catalog_records())


class TestCatalog:
    def test_loads_with_all_required_columns(self):
        df = load_catalog()
        assert len(df) > 300
        for column in ("id", "outfit_slot", "formality", "price", "image_file"):
            assert column in df.columns

    def test_every_product_has_an_image_on_disk(self):
        from src.data import IMAGE_DIR

        missing = [
            row["image_file"]
            for row in catalog_records()
            if not (IMAGE_DIR / str(row["image_file"])).exists()
        ]
        assert not missing, f"{len(missing)} catalog images are missing"

    def test_prices_are_positive_integers(self):
        for product in catalog_records():
            assert isinstance(product["price"], int)
            assert product["price"] > 0

    def test_every_slot_has_candidates_for_both_genders(self):
        df = load_catalog()
        for gender in ("Men", "Women"):
            available = df[df["gender"].isin([gender, "Unisex"])]
            for slot in ("top", "bottom", "footwear", "accessory"):
                assert (available["outfit_slot"] == slot).sum() >= 5, (gender, slot)


class TestHardFilters:
    def test_gender_is_enforced(self, catalog):
        prefs = StylePreferences(gender="Men", occasion="everyday_casual")
        for product in hard_filter(catalog, prefs):
            assert product["gender"] in ("Men", "Unisex")

    def test_formality_band_is_enforced(self, catalog):
        prefs = StylePreferences(gender="Men", occasion="interview", style="formal")
        target, tolerance = resolve_formality_target(prefs.occasion, prefs.style)
        for product in hard_filter(catalog, prefs):
            allowed = tolerance + (
                ACCESSORY_TOLERANCE_BONUS if product["outfit_slot"] == "accessory" else 0.0
            )
            assert abs(product["formality"] - target) <= allowed

    def test_budget_removes_unaffordable_items(self, catalog):
        prefs = StylePreferences(gender="Women", occasion="everyday_casual", budget=1000)
        for product in hard_filter(catalog, prefs):
            assert product["price"] <= 1000

    def test_ethnic_request_returns_ethnic_garments(self, catalog):
        prefs = StylePreferences(gender="Women", occasion="wedding_festive", style="ethnic")
        garments = [
            p for p in hard_filter(catalog, prefs)
            if p["outfit_slot"] in ("top", "bottom", "onepiece")
        ]
        assert garments
        assert all(p["is_ethnic"] for p in garments)

    def test_colour_is_never_a_hard_filter(self, catalog):
        """Colour must stay a soft signal, or result sets collapse to empty."""
        with_colour = hard_filter(
            catalog,
            StylePreferences(gender="Women", occasion="party", preferred_colors=["purple"]),
        )
        without_colour = hard_filter(
            catalog, StylePreferences(gender="Women", occasion="party")
        )
        assert len(with_colour) == len(without_colour)


class TestRecommendation:
    def test_returns_complete_ranked_outfits(self):
        prefs = StylePreferences(gender="Men", occasion="work_office", style="formal", budget=8000)
        result = recommend_outfits(prefs)
        assert result.ok
        scores = [o.final_score for o in result.outfits]
        assert scores == sorted(scores, reverse=True)
        for outfit in result.outfits:
            assert set(TEMPLATE_STANDARD.required_slots) <= set(outfit.items)

    def test_respects_the_budget(self):
        for budget in (2000, 3000, 5000, 12000):
            result = recommend_outfits(
                StylePreferences(gender="Men", occasion="college", budget=budget)
            )
            for outfit in result.outfits:
                assert outfit.total_price <= budget

    def test_outfits_are_diverse(self):
        result = recommend_outfits(
            StylePreferences(gender="Women", occasion="everyday_casual", budget=9000)
        )
        seen: set[int] = set()
        for outfit in result.outfits:
            ids = set(outfit.product_ids)
            assert not (ids & seen), "outfits must not share products"
            seen |= ids

    def test_is_deterministic(self):
        prefs = StylePreferences(gender="Women", occasion="party", budget=6000)
        first = recommend_outfits(prefs)
        second = recommend_outfits(prefs)
        assert [o.product_ids for o in first.outfits] == [o.product_ids for o in second.outfits]

    def test_is_fast_enough_for_an_interactive_ui(self):
        result = recommend_outfits(
            StylePreferences(gender="Women", occasion="college", budget=5000)
        )
        assert result.latency_ms < 1000

    def test_scores_stay_in_the_unit_interval(self):
        result = recommend_outfits(
            StylePreferences(gender="Men", occasion="date_night", budget=7000)
        )
        for outfit in result.outfits:
            assert 0.0 <= score_outfit(outfit) <= 1.0
            assert 0.0 <= outfit.compatibility <= 1.0
            assert 0.0 <= outfit.preference_match <= 1.0

    @pytest.mark.parametrize("occasion", list(OCCASION_PROFILES))
    def test_every_occasion_produces_something_or_explains_itself(self, occasion):
        for gender in ("Men", "Women"):
            result = recommend_outfits(
                StylePreferences(gender=gender, occasion=occasion, budget=10000)
            )
            assert result.ok or result.failure is not None


class TestUnisexRequests:
    """Regression test for evaluation scenario S24."""

    def test_unisex_draws_from_the_whole_catalog(self, catalog):
        # The source tags essentially no Unisex bottomwear, so a literal reading
        # of the label makes every Unisex request impossible.
        kept = hard_filter(catalog, StylePreferences(gender="Unisex", occasion="travel"))
        assert any(p["outfit_slot"] == "bottom" for p in kept)

    def test_unisex_request_produces_a_complete_outfit(self):
        result = recommend_outfits(
            StylePreferences(gender="Unisex", occasion="travel", budget=6000)
        )
        assert result.ok
        for outfit in result.outfits:
            template = TEMPLATE_ONEPIECE if outfit.template == "onepiece" else TEMPLATE_STANDARD
            assert set(template.required_slots) <= set(outfit.items)

    def test_unisex_outfits_never_mix_menswear_and_womenswear(self):
        result = recommend_outfits(
            StylePreferences(gender="Unisex", occasion="travel", budget=6000)
        )
        for outfit in result.outfits:
            genders = {i["gender"] for i in outfit.items.values()} - {"Unisex"}
            assert len(genders) <= 1, f"outfit mixes {genders}"


class TestConflictDetection:
    """Regression test for evaluation scenario S21."""

    def test_opposing_occasion_and_style_is_flagged(self):
        from src.recommender import detect_conflicts

        conflicts = detect_conflicts(
            StylePreferences(gender="Men", occasion="workout", style="formal")
        )
        assert conflicts, "a formal gym outfit is a contradiction and must be flagged"

    def test_coherent_request_is_not_flagged(self):
        from src.recommender import detect_conflicts

        assert not detect_conflicts(
            StylePreferences(gender="Men", occasion="work_office", style="formal")
        )

    def test_two_strong_colours_are_flagged(self):
        from src.recommender import detect_conflicts

        conflicts = detect_conflicts(
            StylePreferences(gender="Women", occasion="party",
                             preferred_colors=["purple", "green"])
        )
        assert any("colour" in c for c in conflicts)

    def test_conflicting_request_still_returns_a_best_effort_answer(self):
        result = recommend_outfits(
            StylePreferences(gender="Men", occasion="workout", style="formal", budget=5000)
        )
        assert result.diagnostics["conflicts"]
        assert result.ok, "we warn, but we still answer"


class TestPreferenceMatch:
    def test_requested_colour_scores_above_a_miss(self, catalog):
        prefs = StylePreferences(gender="Men", occasion="everyday_casual",
                                 preferred_colors=["blue"])
        blue = next(p for p in catalog if p["color_family"] == "blue"
                    and p["gender"] == "Men")
        loud = next(p for p in catalog if p["color_family"] in ("red", "pink", "green")
                    and not p["is_neutral"] and p["gender"] == "Men")
        assert preference_match(blue, prefs) > preference_match(loud, prefs)

    def test_no_colour_preference_does_not_penalise_anything(self, catalog):
        prefs = StylePreferences(gender="Men", occasion="everyday_casual")
        item = next(p for p in catalog if p["gender"] == "Men")
        assert preference_match(item, prefs) <= 1.0


class TestFailureHandling:
    def test_impossible_budget_fails_clearly_instead_of_crashing(self):
        result = recommend_outfits(
            StylePreferences(gender="Men", occasion="work_office", style="formal", budget=400)
        )
        assert not result.ok
        assert result.failure is not None
        assert result.failure.reason_code in ("empty_slot", "budget_infeasible")
        assert result.failure.suggestion

    def test_failure_never_returns_partial_or_unrelated_products(self):
        result = recommend_outfits(
            StylePreferences(gender="Men", occasion="work_office", style="formal", budget=400)
        )
        assert result.outfits == []

    def test_diagnostics_explain_where_the_filter_chain_emptied(self):
        result = recommend_outfits(
            StylePreferences(gender="Men", occasion="interview", style="formal", budget=600)
        )
        assert "candidates_after_filter" in result.diagnostics

    def test_tight_but_possible_budget_still_succeeds(self):
        result = recommend_outfits(
            StylePreferences(gender="Men", occasion="everyday_casual", budget=1500)
        )
        assert result.ok, "the catalog holds items cheap enough for this"


class TestCompleteTheLook:
    def test_anchor_is_always_present(self):
        for product in list(catalog_records())[:25]:
            result = complete_the_look(int(product["id"]), budget=12000)
            for outfit in result.outfits:
                assert int(product["id"]) in outfit.product_ids

    def test_unknown_product_fails_gracefully(self):
        result = complete_the_look(-1)
        assert not result.ok
        assert result.failure.reason_code == "unknown_product"

    def test_anchor_slot_is_not_duplicated(self):
        product = next(p for p in catalog_records() if p["outfit_slot"] == "top")
        result = complete_the_look(int(product["id"]), budget=10000)
        for outfit in result.outfits:
            assert outfit.items["top"]["id"] == product["id"]

    def test_inferred_preferences_track_the_anchor(self):
        ethnic = next(p for p in catalog_records() if p["is_ethnic"])
        prefs = infer_preferences_from_product(ethnic)
        assert prefs.style == "ethnic"
        assert prefs.occasion == "wedding_festive"


class TestNoHallucination:
    """The core safety property of the whole design."""

    def test_every_recommended_product_exists_in_the_catalog(self):
        valid = product_ids()
        scenarios = [
            StylePreferences(gender="Men", occasion="work_office", style="formal", budget=9000),
            StylePreferences(gender="Women", occasion="wedding_festive", style="ethnic", budget=8000),
            StylePreferences(gender="Women", occasion="party", budget=6000),
            StylePreferences(gender="Men", occasion="workout", budget=5000),
            StylePreferences(gender="Unisex", occasion="travel", budget=7000),
        ]
        for prefs in scenarios:
            for outfit in recommend_outfits(prefs).outfits:
                for product_id in outfit.product_ids:
                    assert product_id in valid
                    assert get_product(product_id) is not None

    def test_displayed_prices_match_the_catalog_exactly(self):
        result = recommend_outfits(
            StylePreferences(gender="Women", occasion="date_night", budget=8000)
        )
        for outfit in result.outfits:
            for item in outfit.items.values():
                assert item["price"] == get_product(item["id"])["price"]

    def test_grounding_payload_contains_no_ids_and_only_real_data(self):
        prefs = StylePreferences(gender="Men", occasion="college", budget=5000)
        outfit = recommend_outfits(prefs).outfits[0]
        payload = grounding_payload(outfit, prefs)

        # No ids leave the system, so the model cannot echo one back as a choice.
        assert "id" not in str(payload).lower().split('"slot"')[0]
        names = {item["name"] for item in payload["selected_items"]}
        assert names == {i["productDisplayName"] for i in outfit.items.values()}
        assert payload["total_price_inr"] == outfit.total_price

    def test_reasons_are_generated_from_real_numbers(self):
        prefs = StylePreferences(gender="Men", occasion="college", budget=5000)
        outfit = recommend_outfits(prefs).outfits[0]
        reasons = outfit_reasons(outfit, prefs)
        assert reasons
        assert any(str(f"{outfit.total_price:,}") in reason for reason in reasons)
