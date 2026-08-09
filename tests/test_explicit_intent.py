"""Explicit user intent must never be silently violated.

This is the highest-value test module in the suite. The bug it guards against is
the one a user notices immediately: asking for "a black shirt" and being handed a
white top.

The distinction under test:

    "black shirt"        HARD  - the top must be a shirt, and it must be black
    "neutral colours"    SOFT  - influences ranking, never empties the result

Every assertion below checks the *returned products*, not the explanation text.
A system that describes the outfit correctly while returning the wrong garment
still fails these tests.
"""

from __future__ import annotations

import pytest

from src.data import catalog_records
from src.features import EXPLICIT_COLOUR_MATCHES, GARMENT_ARTICLE_TYPES
from src.nlu import extract_item_requests, parse_request
from src.recommender import complete_the_look, recommend_outfits
from src.schemas import ItemRequest, StylePreferences


def outfits_for(text: str, gender: str = "Men", budget: int | None = None):
    prefs = parse_request(text, default_gender=gender)
    if budget is not None:
        prefs.budget = budget
    return prefs, recommend_outfits(prefs)


def assert_satisfies(result, prefs, garment: str, colour: str | None):
    """Either every outfit honours the request, or nothing was returned."""
    article_types = set(GARMENT_ARTICLE_TYPES[garment])
    colours = set(EXPLICIT_COLOUR_MATCHES[colour]) if colour else None

    if not result.ok:
        # A clean, explained refusal is acceptable. Silent substitution is not.
        assert result.outfits == []
        assert result.failure is not None
        assert result.failure.suggestion
        return

    for outfit in result.outfits:
        matching = [
            item for item in outfit.items.values()
            if item["articleType"] in article_types
        ]
        assert matching, (
            f"no {garment} in outfit: "
            f"{[i['articleType'] for i in outfit.items.values()]}"
        )
        if colours is not None:
            assert any(item["baseColour"] in colours for item in matching), (
                f"{garment} present but wrong colour: "
                f"{[i['baseColour'] for i in matching]} (wanted {sorted(colours)})"
            )


# --------------------------------------------------------------------------
# The five cases named in the requirements
# --------------------------------------------------------------------------

EXPLICIT_CASES = [
    ("black shirt", "Men", "shirt", "black"),
    ("white shirt", "Men", "shirt", "white"),
    ("blue jeans", "Men", "jeans", "blue"),
    ("black trousers", "Men", "trousers", "black"),
    ("red dress", "Women", "dress", "red"),
]


@pytest.mark.parametrize("text,gender,garment,colour", EXPLICIT_CASES)
def test_explicit_request_is_honoured(text, gender, garment, colour):
    prefs, result = outfits_for(text, gender=gender)
    assert prefs.required_items, f"{text!r} produced no hard constraint"
    assert_satisfies(result, prefs, garment, colour)


@pytest.mark.parametrize("text,gender,garment,colour", EXPLICIT_CASES)
def test_explicit_request_parses_into_a_hard_constraint(text, gender, garment, colour):
    prefs = parse_request(text, default_gender=gender)
    assert [(i.garment, i.colour) for i in prefs.required_items] == [(garment, colour)]
    # The colour belongs to the garment, not to the whole outfit.
    assert prefs.preferred_colors == []


@pytest.mark.parametrize("text,gender,garment,colour", EXPLICIT_CASES)
def test_explicit_request_survives_a_budget(text, gender, garment, colour):
    prefs, result = outfits_for(text, gender=gender, budget=6000)
    assert_satisfies(result, prefs, garment, colour)
    for outfit in result.outfits:
        assert outfit.total_price <= 6000


def test_black_and_white_shirts_are_not_the_same_request(self=None):
    """Both colours live in the `neutral` family - the original bug.

    Family-level matching cannot tell them apart, which is why explicit requests
    match concrete baseColour values instead.
    """
    _, black = outfits_for("black shirt", gender="Men")
    _, white = outfits_for("white shirt", gender="Men")
    if black.ok and white.ok:
        assert black.outfits[0].product_ids != white.outfits[0].product_ids


# --------------------------------------------------------------------------
# Soft preferences must stay soft
# --------------------------------------------------------------------------

class TestSoftPreferences:
    def test_general_colour_wish_is_not_a_hard_filter(self):
        prefs = parse_request("I want something in neutral colours")
        assert prefs.required_items == []
        assert prefs.preferred_colors == ["neutral"]

    def test_general_colour_wish_still_returns_results(self):
        prefs = parse_request("something in neutral colours for a woman")
        result = recommend_outfits(prefs)
        assert result.ok, "a vague palette wish must never empty the result set"

    def test_a_loud_colour_preference_does_not_fail(self):
        prefs = parse_request("something purple and green for a party, women")
        result = recommend_outfits(prefs)
        assert result.ok
        assert prefs.required_items == []

    def test_colour_without_a_garment_stays_soft(self):
        prefs = parse_request("smart casual, black and white, under 3000")
        assert prefs.required_items == []
        assert "neutral" in prefs.preferred_colors


# --------------------------------------------------------------------------
# Parser behaviour
# --------------------------------------------------------------------------

class TestItemExtraction:
    def test_garment_without_colour_is_still_a_hard_constraint(self):
        requests, consumed = extract_item_requests("I need jeans")
        assert [r.garment for r in requests] == ["jeans"]
        assert requests[0].colour is None
        assert consumed == set()

    def test_longest_garment_phrase_wins(self):
        requests, _ = extract_item_requests("formal shoes please")
        assert requests[0].garment == "formal_shoes"

        requests, _ = extract_item_requests("track pants")
        assert requests[0].garment == "track_pants"

    def test_multiple_items_are_captured(self):
        requests, _ = extract_item_requests("a white shirt and blue jeans")
        assert {(r.garment, r.colour) for r in requests} == {
            ("shirt", "white"), ("jeans", "blue")
        }

    def test_dress_code_is_not_a_dress(self):
        requests, _ = extract_item_requests("what is the office dress code")
        assert all(r.garment != "dress" for r in requests)

    def test_unknown_garment_is_ignored_not_guessed(self):
        requests, _ = extract_item_requests("I need a spacesuit")
        assert requests == []

    def test_colour_word_far_from_garment_is_not_attached(self):
        requests, consumed = extract_item_requests(
            "a shirt, and I generally like black"
        )
        assert requests[0].garment == "shirt"
        assert requests[0].colour is None
        assert consumed == set()


# --------------------------------------------------------------------------
# Constraint resolution and failure behaviour
# --------------------------------------------------------------------------

class TestConstraintResolution:
    def test_same_slot_requests_are_unioned(self):
        prefs = StylePreferences(
            gender="Women",
            required_items=[{"garment": "saree", "colour": None},
                            {"garment": "dress", "colour": None}],
        )
        constraints = prefs.constraints_by_slot()
        assert set(constraints) == {"onepiece"}
        assert {"Sarees", "Dresses"} <= constraints["onepiece"]["article_types"]

    def test_a_colourless_sibling_drops_the_colour_constraint(self):
        # "a saree or a red dress" must not silently require every saree be red.
        prefs = StylePreferences(
            gender="Women",
            required_items=[{"garment": "saree", "colour": None},
                            {"garment": "dress", "colour": "red"}],
        )
        assert prefs.constraints_by_slot()["onepiece"]["base_colours"] == set()

    def test_invalid_garment_is_dropped_by_validation(self):
        prefs = StylePreferences(
            required_items=[{"garment": "spaceship", "colour": "black"},
                            {"garment": "shirt", "colour": "black"}]
        )
        assert [i.garment for i in prefs.required_items] == ["shirt"]

    def test_unknown_colour_is_dropped_but_garment_kept(self):
        request = ItemRequest(garment="shirt", colour="chartreuse")
        assert request.colour is None
        assert request.article_types == ("Shirts",)

    def test_impossible_request_fails_with_a_specific_message(self):
        prefs = StylePreferences(
            gender="Men",
            required_items=[{"garment": "saree", "colour": "gold"}],
        )
        result = recommend_outfits(prefs)
        assert not result.ok
        assert result.failure.reason_code == "unsatisfiable_item_request"
        assert "saree" in result.failure.message
        assert result.outfits == []

    def test_a_named_garment_priced_above_the_whole_budget_fails_by_name(self):
        """The named slot is exempt from the formality band, not from the budget.

        Every saree in the catalog costs more than 1,000, so this request cannot
        be satisfied. It must say so in the user's own words. Previously the
        over-budget sarees survived `hard_filter`, `unsatisfied_constraints`
        concluded the request was satisfiable, and the user got the generic
        "no one-piece matches all of your constraints" instead.
        """
        prefs = StylePreferences(
            gender="Women", occasion="wedding_festive", budget=1000,
            required_items=[{"garment": "saree", "colour": None}],
        )
        result = recommend_outfits(prefs)
        assert not result.ok
        assert result.failure.reason_code == "unsatisfiable_item_request"
        assert "saree" in result.failure.message
        assert "1,000" in result.failure.message

    def test_failure_messages_never_expose_internal_slot_keys(self):
        """`outfit_slot` values are identifiers, not English. "onepiece" and
        "outerwear" must never reach a shopper."""
        scenarios = [
            StylePreferences(gender="Women", occasion="wedding_festive",
                             budget=1000,
                             required_items=[{"garment": "saree", "colour": None}]),
            StylePreferences(gender="Men", occasion="work_office", budget=400),
            StylePreferences(gender="Women", occasion="party", budget=350),
        ]
        for prefs in scenarios:
            result = recommend_outfits(prefs)
            if result.ok:
                continue
            text = f"{result.failure.message} {result.failure.suggestion}"
            for raw in ("onepiece", "outerwear", "outfit_slot"):
                assert raw not in text.lower(), f"{raw!r} leaked into: {text}"

    def test_template_follows_the_requested_garment(self):
        prefs = parse_request("red dress", default_gender="Women")
        result = recommend_outfits(prefs)
        if result.ok:
            assert all(o.template == "onepiece" for o in result.outfits)

        prefs = parse_request("blue jeans", default_gender="Men")
        result = recommend_outfits(prefs)
        if result.ok:
            assert all(o.template == "standard" for o in result.outfits)


# --------------------------------------------------------------------------
# Catalog coverage that these requests depend on
# --------------------------------------------------------------------------

class TestStatedColoursSurviveIntoTheOutfit:
    """A soft colour preference is not a filter, but it must still be visible in
    the result - especially in the OPTIONAL slots, which are a bonus and should
    never be the thing that breaks the brief."""

    def test_a_neutral_request_does_not_gain_a_coloured_accessory(self):
        """The reported bug: "smart casual, black and white, under 3000"
        returned a white dress, grey shoes and a GREEN bangle - because the
        dress and shoes left only 252 rupees and the bangle was the one
        accessory that fit. An optional extra that fights the request should be
        dropped, not squeezed in."""
        prefs = parse_request(
            "College presentation tomorrow - smart casual, black and white, under 3000",
            default_gender="Women",
        )
        assert prefs.preferred_colors == ["neutral"]
        result = recommend_outfits(prefs)
        assert result.ok

        best = result.outfits[0]
        offenders = [
            f"{i['baseColour']} {i['articleType']}"
            for slot, i in best.items.items()
            if slot in ("accessory", "outerwear") and not i["is_neutral"]
        ]
        assert not offenders, f"off-brief optional pieces: {offenders}"

    def test_the_best_look_for_a_neutral_request_is_actually_neutral(self):
        prefs = parse_request("Party outfit for a woman, black, 6000",
                              default_gender="Women")
        best = recommend_outfits(prefs).outfits[0]
        neutral_share = sum(1 for i in best.items.values() if i["is_neutral"])
        assert neutral_share == len(best.items)


class TestStapleCoverage:
    """Guaranteed by scripts/build_catalog.py's STAPLES table."""

    @pytest.mark.parametrize("gender,article_type,colour", [
        ("Men", "Shirts", "Black"),
        ("Men", "Shirts", "White"),
        ("Men", "Trousers", "Black"),
        ("Men", "Jeans", "Blue"),
        ("Women", "Dresses", "Red"),
        ("Women", "Tops", "Black"),
    ])
    def test_staple_combination_exists(self, gender, article_type, colour):
        assert any(
            p["gender"] == gender
            and p["articleType"] == article_type
            and p["baseColour"] == colour
            for p in catalog_records()
        ), f"missing staple: {gender} {colour} {article_type}"


# --------------------------------------------------------------------------
# Complete the Look must respect the anchor the same way
# --------------------------------------------------------------------------

class TestGenderInference:
    """Regression: "a saree for men" resolved to Women."""

    def test_explicit_gender_beats_a_garment_hint(self):
        assert parse_request("a purple saree for men, 5000").gender == "Men"
        assert parse_request("a dress for my husband").gender == "Men"

    def test_garment_hint_still_used_when_nothing_explicit(self):
        assert parse_request("a red saree, 5000").gender == "Women"
        assert parse_request("a summer dress").gender == "Women"

    def test_explicit_women_still_wins(self):
        assert parse_request("shirt for women").gender == "Women"


class TestLookDiversityWithRequiredItems:
    def test_a_required_garment_may_repeat_across_looks(self):
        """Excluding it would cap the result at one look per matching product."""
        prefs = parse_request("black shirt for men", default_gender="Men")
        prefs.budget = 8000
        result = recommend_outfits(prefs)
        assert result.ok
        if len(result.outfits) > 1:
            tops = [o.items["top"]["id"] for o in result.outfits]
            others = [
                {i["id"] for s, i in o.items.items() if s != "top"}
                for o in result.outfits
            ]
            assert others[0] != others[1], "everything but the shirt should vary"
            assert len(set(tops)) >= 1


class TestOptionalItemsDoNotDamageTheLook:
    def test_optional_extra_never_lowers_compatibility_materially(self):
        from src.compatibility import outfit_compatibility
        from src.outfit_builder import (
            OPTIONAL_SLOT_MAX_COMPATIBILITY_DROP,
            TEMPLATE_ONEPIECE,
            TEMPLATE_STANDARD,
        )

        for text, gender in [("red dress", "Women"), ("black shirt", "Men"),
                             ("a saree", "Women")]:
            prefs = parse_request(text, default_gender=gender)
            prefs.budget = 12000
            for outfit in recommend_outfits(prefs).outfits:
                template = (TEMPLATE_ONEPIECE if outfit.template == "onepiece"
                            else TEMPLATE_STANDARD)
                required = [
                    item for slot, item in outfit.items.items()
                    if slot in template.required_slots
                ]
                full = outfit_compatibility(list(outfit.items.values()))
                core = outfit_compatibility(required)
                assert full >= core - (
                    OPTIONAL_SLOT_MAX_COMPATIBILITY_DROP * len(outfit.items)
                ), f"optional items damaged the look for {text!r}"


class TestCompleteTheLookRespectsTheAnchor:
    def test_anchor_is_never_substituted(self):
        black_shirt = next(
            p for p in catalog_records()
            if p["articleType"] == "Shirts" and p["baseColour"] == "Black"
        )
        result = complete_the_look(int(black_shirt["id"]), budget=9000)
        assert result.ok
        for outfit in result.outfits:
            assert int(black_shirt["id"]) in outfit.product_ids
