"""Outfit assembly: completeness, budget enforcement and useful failures."""

from __future__ import annotations

from src.features import SLOT_ACCESSORY, SLOT_BOTTOM, SLOT_FOOTWEAR, SLOT_TOP
from src.outfit_builder import (
    TEMPLATE_ONEPIECE,
    TEMPLATE_STANDARD,
    BuildFailure,
    Outfit,
    budget_fit,
    build_outfit,
    minimum_outfit_cost,
)


def flat_score(_item) -> float:
    """Neutral solo score, so tests isolate assembly from preference matching."""
    return 0.8


def pools(make_item, **overrides):
    base = {
        SLOT_TOP: [
            make_item("Shirts", "White", usage="Formal", price=1200),
            make_item("Shirts", "Blue", usage="Formal", price=900),
            make_item("Tshirts", "Black", usage="Casual", price=400),
        ],
        SLOT_BOTTOM: [
            make_item("Trousers", "Navy Blue", usage="Formal", price=1500),
            make_item("Jeans", "Blue", usage="Casual", price=1100),
        ],
        SLOT_FOOTWEAR: [
            make_item("Formal Shoes", "Black", usage="Formal", price=2200),
            make_item("Casual Shoes", "White", usage="Casual", price=1300),
        ],
        SLOT_ACCESSORY: [
            make_item("Belts", "Black", usage="Formal", price=500),
        ],
    }
    base.update(overrides)
    return base


class TestCompleteness:
    def test_fills_every_required_slot(self, make_item):
        outfit = build_outfit(pools(make_item), TEMPLATE_STANDARD, None, flat_score)
        assert isinstance(outfit, Outfit)
        for slot in TEMPLATE_STANDARD.required_slots:
            assert slot in outfit.items

    def test_total_price_equals_the_sum_of_its_parts(self, make_item):
        outfit = build_outfit(pools(make_item), TEMPLATE_STANDARD, None, flat_score)
        assert outfit.total_price == sum(i["price"] for i in outfit.items.values())

    def test_never_repeats_a_product(self, make_item):
        outfit = build_outfit(pools(make_item), TEMPLATE_STANDARD, None, flat_score)
        ids = outfit.product_ids
        assert len(ids) == len(set(ids))

    def test_onepiece_template_needs_no_separates(self, make_item):
        catalog = {
            "onepiece": [make_item("Dresses", "Black", usage="Party", price=2500)],
            SLOT_FOOTWEAR: [make_item("Heels", "Black", usage="Party", price=1500)],
        }
        outfit = build_outfit(catalog, TEMPLATE_ONEPIECE, 6000, flat_score)
        assert isinstance(outfit, Outfit)
        assert SLOT_TOP not in outfit.items and SLOT_BOTTOM not in outfit.items


class TestBudget:
    def test_never_exceeds_the_budget(self, make_item):
        for budget in range(2500, 9000, 250):
            result = build_outfit(pools(make_item), TEMPLATE_STANDARD, budget, flat_score)
            if isinstance(result, Outfit):
                assert result.total_price <= budget, budget

    def test_lookahead_prevents_spending_the_budget_on_one_slot(self, make_item):
        # Cheapest complete outfit is 400 + 1100 + 1300 = 2800. With a budget of
        # 3000 a naive greedy picks the 2200 shoes first and then cannot afford
        # a top and a bottom.
        result = build_outfit(pools(make_item), TEMPLATE_STANDARD, 3000, flat_score)
        assert isinstance(result, Outfit)
        assert result.total_price <= 3000
        assert len(result.items) >= 3

    def test_reports_infeasible_budget_with_the_real_minimum(self, make_item):
        catalog = pools(make_item)
        floor = minimum_outfit_cost(catalog, TEMPLATE_STANDARD)
        result = build_outfit(catalog, TEMPLATE_STANDARD, floor - 100, flat_score)
        assert isinstance(result, BuildFailure)
        assert result.reason_code in ("budget_infeasible", "empty_slot")

    def test_optional_slots_only_use_leftover_money(self, make_item):
        tight = build_outfit(pools(make_item), TEMPLATE_STANDARD, 2900, flat_score)
        assert isinstance(tight, Outfit)
        assert SLOT_ACCESSORY not in tight.items

    def test_no_budget_means_no_constraint(self, make_item):
        outfit = build_outfit(pools(make_item), TEMPLATE_STANDARD, None, flat_score)
        assert isinstance(outfit, Outfit)
        assert outfit.budget_fit == 1.0


class TestBudgetFit:
    def test_full_marks_between_sixty_and_one_hundred_percent(self):
        assert budget_fit(3000, 3000) == 1.0
        assert budget_fit(1800, 3000) == 1.0

    def test_tapers_when_far_under_budget(self):
        assert budget_fit(600, 3000) < budget_fit(1800, 3000)

    def test_zero_when_over_budget(self):
        assert budget_fit(3100, 3000) == 0.0

    def test_no_budget_is_neutral(self):
        assert budget_fit(5000, None) == 1.0


class TestFailures:
    def test_empty_slot_is_reported_by_name(self, make_item):
        catalog = pools(make_item)
        catalog[SLOT_FOOTWEAR] = []
        result = build_outfit(catalog, TEMPLATE_STANDARD, 10000, flat_score)
        assert isinstance(result, BuildFailure)
        assert result.reason_code == "empty_slot"
        assert SLOT_FOOTWEAR in result.detail["empty_slots"]
        assert result.suggestion, "a failure must tell the user what to change"

    def test_failure_messages_are_actionable(self, make_item):
        catalog = pools(make_item)
        catalog[SLOT_TOP] = []
        result = build_outfit(catalog, TEMPLATE_STANDARD, 10000, flat_score)
        assert isinstance(result, BuildFailure)
        assert len(result.message) > 20 and len(result.suggestion) > 20

    def test_hard_clashes_are_avoided_during_assembly(self, make_item):
        # Only shorts available at the bottom, only formal shoes as footwear:
        # that pairing is on the hard-clash list, so assembly must fail rather
        # than emit it.
        catalog = {
            SLOT_TOP: [make_item("Shirts", "White", usage="Formal", price=900)],
            SLOT_BOTTOM: [make_item("Shorts", "Beige", usage="Casual", price=700)],
            SLOT_FOOTWEAR: [make_item("Formal Shoes", "Black", usage="Formal", price=2000)],
        }
        result = build_outfit(catalog, TEMPLATE_STANDARD, 10000, flat_score)
        if isinstance(result, Outfit):
            from src.compatibility import has_hard_clash

            assert not has_hard_clash(list(result.items.values()))
        else:
            assert result.reason_code == "slot_unfillable"


class TestDeterminism:
    def test_same_input_gives_the_same_outfit(self, make_item):
        catalog = pools(make_item)
        first = build_outfit(catalog, TEMPLATE_STANDARD, 8000, flat_score)
        second = build_outfit(catalog, TEMPLATE_STANDARD, 8000, flat_score)
        assert first.product_ids == second.product_ids

    def test_exclusions_produce_a_different_outfit(self, make_item):
        catalog = pools(make_item)
        first = build_outfit(catalog, TEMPLATE_STANDARD, 8000, flat_score)
        second = build_outfit(
            catalog, TEMPLATE_STANDARD, 8000, flat_score,
            exclude_ids=frozenset(first.product_ids),
        )
        if isinstance(second, Outfit):
            assert not set(first.product_ids) & set(second.product_ids)
