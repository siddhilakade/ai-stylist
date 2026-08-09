"""Greedy, budget-aware outfit assembly.

Choosing one item per slot so everything is mutually compatible and the total
fits the budget is a maximum-weight clique on a slot-partite graph. Rather than
search it exhaustively:

  1. Fill the most constrained slot first (fewest surviving candidates) - the
     standard CSP heuristic. If a slot is going to fail, fail before the budget
     is committed elsewhere.
  2. Pick the candidate maximising compatibility-with-what's-chosen plus fit to
     the request.
  3. Budget lookahead: before accepting anything, check the money left still
     covers the cheapest item in every unfilled slot. Without this, greedy buys
     good shoes and then can't afford trousers.
  4. Optional slots (accessory, layer) come last, from what's left.

Runs in milliseconds and is easy to explain, which matters more here than
closing the optimality gap.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from src.compatibility import is_hard_clash, outfit_compatibility, outfit_signals
from src.features import (
    SLOT_ACCESSORY,
    SLOT_BOTTOM,
    SLOT_FOOTWEAR,
    SLOT_LABELS,
    SLOT_ONEPIECE,
    SLOT_OUTERWEAR,
    SLOT_TOP,
)

Product = Mapping[str, Any]


# --------------------------------------------------------------------------
# Outfit templates
# --------------------------------------------------------------------------
# A template says which slots make an outfit. Two templates cover the catalog:
# the western top/bottom/shoes combination, and the one-piece look (a dress or
# saree) where a single garment satisfies top and bottom together.

@dataclass(frozen=True)
class OutfitTemplate:
    key: str
    label: str
    required_slots: tuple[str, ...]
    optional_slots: tuple[str, ...]

    @property
    def all_slots(self) -> tuple[str, ...]:
        return self.required_slots + self.optional_slots


TEMPLATE_STANDARD = OutfitTemplate(
    key="standard",
    label="Top + Bottom + Footwear",
    required_slots=(SLOT_TOP, SLOT_BOTTOM, SLOT_FOOTWEAR),
    optional_slots=(SLOT_ACCESSORY, SLOT_OUTERWEAR),
)

TEMPLATE_ONEPIECE = OutfitTemplate(
    key="onepiece",
    label="One-piece + Footwear",
    required_slots=(SLOT_ONEPIECE, SLOT_FOOTWEAR),
    optional_slots=(SLOT_ACCESSORY, SLOT_OUTERWEAR),
)

TEMPLATES = {t.key: t for t in (TEMPLATE_STANDARD, TEMPLATE_ONEPIECE)}


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class Outfit:
    """A complete, budget-valid outfit plus every number behind it."""

    template: str
    items: dict[str, Product]
    total_price: int
    compatibility: float
    signals: dict[str, float]
    preference_match: float
    budget_fit: float
    final_score: float
    budget: int | None = None
    anchor_slot: str | None = None

    @property
    def ordered_items(self) -> list[tuple[str, Product]]:
        """Items in a stable, human-sensible display order."""
        order = (SLOT_TOP, SLOT_ONEPIECE, SLOT_BOTTOM, SLOT_OUTERWEAR,
                 SLOT_FOOTWEAR, SLOT_ACCESSORY)
        return [(s, self.items[s]) for s in order if s in self.items]

    @property
    def product_ids(self) -> list[int]:
        return [int(item["id"]) for item in self.items.values()]

    def budget_headroom(self) -> int | None:
        return None if self.budget is None else self.budget - self.total_price


@dataclass
class BuildFailure:
    """Why assembly failed, in terms a user can act on."""

    reason_code: str
    message: str
    suggestion: str
    detail: dict[str, Any] = field(default_factory=dict)


def _join_naturally(words: Sequence[str]) -> str:
    """['top', 'bottom', 'footwear'] -> 'top, bottom or footwear'."""
    if len(words) == 1:
        return words[0]
    return f"{', '.join(words[:-1])} or {words[-1]}"


def slot_label(slot: str) -> str:
    """Internal slot key -> the word a shopper would use.

    `outfit_slot` values are identifiers, not English: "onepiece" and "outerwear"
    are fine in a dict key and wrong in a sentence shown to a user. Failure
    messages are the one place these strings escape the engine, so they are
    translated here rather than rendered raw.
    """
    return SLOT_LABELS.get(slot, slot).lower()


def _cheapest(candidates: Sequence[Product]) -> int:
    return min((int(c["price"]) for c in candidates), default=0)


def minimum_outfit_cost(
    candidates_by_slot: Mapping[str, Sequence[Product]],
    template: OutfitTemplate,
) -> int:
    """Floor price of any complete outfit from this template's required slots."""
    return sum(
        _cheapest(candidates_by_slot.get(slot, ()))
        for slot in template.required_slots
    )


def build_outfit(
    candidates_by_slot: Mapping[str, Sequence[Product]],
    template: OutfitTemplate,
    budget: int | None,
    solo_score: Callable[[Product], float],
    exclude_ids: frozenset[int] = frozenset(),
    tiebreak_salt: str = "",
    penalise_monotony: bool = True,
) -> Outfit | BuildFailure:
    """Assemble one outfit greedily. Returns an Outfit or a BuildFailure.

    `tiebreak_salt` decides which candidate wins when several score identically -
    see `_tiebreak`. Callers pass a string derived from the request; the default
    reproduces the old catalog-order behaviour.
    """

    salt_key = _salt_key(tiebreak_salt)

    # ---- 0. prune ------------------------------------------------------
    pool: dict[str, list[Product]] = {}
    for slot in template.all_slots:
        items = [
            item for item in candidates_by_slot.get(slot, ())
            if int(item["id"]) not in exclude_ids
        ]
        if budget is not None:
            items = [item for item in items if int(item["price"]) <= budget]
        pool[slot] = items

    # ---- 1. feasibility: every required slot must be non-empty ---------
    empty = [slot for slot in template.required_slots if not pool[slot]]
    if empty:
        return BuildFailure(
            reason_code="empty_slot",
            message=(
                "No "
                + _join_naturally([slot_label(s) for s in empty])
                + " in the catalog matches all of your constraints."
            ),
            suggestion=(
                "Try raising the budget, widening the occasion, "
                "or removing the colour preference."
            ),
            detail={"empty_slots": empty},
        )

    # ---- 2. feasibility: the cheapest complete outfit must fit ---------
    floor_price = minimum_outfit_cost(pool, template)
    if budget is not None and floor_price > budget:
        return BuildFailure(
            reason_code="budget_infeasible",
            message=(
                f"The cheapest complete outfit matching your request costs "
                f"₹{floor_price:,}, which is over your ₹{budget:,} budget."
            ),
            suggestion=f"Raise the budget to about ₹{floor_price:,} or relax the occasion.",
            detail={"minimum_cost": floor_price, "budget": budget},
        )

    # ---- 3. fill required slots, most constrained first ----------------
    fill_order = sorted(template.required_slots, key=lambda s: len(pool[s]))
    chosen: dict[str, Product] = {}
    spent = 0

    for position, slot in enumerate(fill_order):
        remaining_slots = fill_order[position + 1:]
        best_item: Product | None = None
        best_value = float("-inf")

        for candidate in pool[slot]:
            price = int(candidate["price"])

            # Budget lookahead: can we still afford everything left to fill?
            if budget is not None:
                reserve = sum(_cheapest(pool[s]) for s in remaining_slots)
                if spent + price + reserve > budget:
                    continue

            # Hard styling clashes are constraints, not penalties.
            if any(is_hard_clash(candidate, picked) for picked in chosen.values()):
                continue

            value = (_candidate_value(candidate, chosen, solo_score,
                                      penalise_monotony)
                     + _tiebreak(candidate, salt_key))
            if value > best_value:
                best_value, best_item = value, candidate

        if best_item is None:
            # Everything in this slot was priced out or clashed with earlier picks.
            return BuildFailure(
                reason_code="slot_unfillable",
                message=(
                    f"Could not fill the {slot} slot without breaking the budget "
                    "or clashing with the rest of the outfit."
                ),
                suggestion="Increase the budget or loosen the style constraint.",
                detail={"slot": slot, "spent_so_far": spent, "budget": budget},
            )

        chosen[slot] = best_item
        spent += int(best_item["price"])

    # ---- 4. optional slots, only from leftover budget ------------------
    for slot in template.optional_slots:
        if not pool.get(slot):
            continue
        affordable = [
            c for c in pool[slot]
            if budget is None or spent + int(c["price"]) <= budget
        ]
        affordable = [
            c for c in affordable
            if not any(is_hard_clash(c, picked) for picked in chosen.values())
        ]
        if not affordable:
            continue

        best = max(
            affordable,
            key=lambda c: (_candidate_value(c, chosen, solo_score,
                                            penalise_monotony)
                           + _tiebreak(c, salt_key)),
        )
        # Only add an optional item if it genuinely fits the look. A weak
        # accessory makes the outfit worse, not better.
        if (_candidate_value(best, chosen, solo_score, penalise_monotony)
                < OPTIONAL_SLOT_THRESHOLD):
            continue
        # And it must not drag the outfit down. Without this an optional layer
        # can clear the bar on its own merits while still being wrong for the
        # look - a denim waistcoat over a saree scores acceptably in isolation
        # and ruins the outfit in context.
        before = outfit_compatibility(list(chosen.values()), penalise_monotony)
        after = outfit_compatibility([best, *chosen.values()], penalise_monotony)
        if after < before - OPTIONAL_SLOT_MAX_COMPATIBILITY_DROP:
            continue
        # ...and it must not drag down how well the outfit answers the REQUEST.
        # Compatibility alone was not enough: for "black and white, under 3000"
        # the dress and shoes left 252 rupees, and the only accessory that fit
        # was a green bangle. It sat happily beside two neutrals on the
        # compatibility signals while missing the one thing the user actually
        # asked for. An optional extra is a bonus - if the only affordable one
        # fights the brief, the right outfit is the one without it.
        before_fit = sum(solo_score(i) for i in chosen.values()) / len(chosen)
        after_fit = (before_fit * len(chosen) + solo_score(best)) / (len(chosen) + 1)
        if after_fit < before_fit - OPTIONAL_SLOT_MAX_PREFERENCE_DROP:
            continue
        chosen[slot] = best
        spent += int(best["price"])

    items = list(chosen.values())
    return Outfit(
        template=template.key,
        items=chosen,
        total_price=spent,
        compatibility=outfit_compatibility(items, penalise_monotony),
        signals=outfit_signals(items, penalise_monotony),
        preference_match=round(sum(solo_score(i) for i in items) / len(items), 4),
        budget_fit=budget_fit(spent, budget),
        final_score=0.0,  # filled in by the ranker, which owns the weights
        budget=budget,
        anchor_slot=fill_order[0] if fill_order else None,
    )


# An optional item must clear this combined (compatibility + fit) bar to be
# added. Set just above the midpoint: we would rather return a clean three-piece
# outfit than pad it with a marginal accessory.
OPTIONAL_SLOT_THRESHOLD = 0.60

# How much overall compatibility an optional extra may cost. Small on purpose:
# an accessory or layer should be neutral-to-positive for the look, never a
# meaningful downgrade.
OPTIONAL_SLOT_MAX_COMPATIBILITY_DROP = 0.02

# The same idea applied to request fit rather than internal coherence. An
# accessory a little below the outfit's average is fine; one far below it is
# answering a different question than the one asked.
OPTIONAL_SLOT_MAX_PREFERENCE_DROP = 0.05


# --------------------------------------------------------------------------
# Tie-breaking
# --------------------------------------------------------------------------
# Candidate scores are coarse: `preference_match` takes only 3-12 distinct values
# across a whole filtered pool, so on average 10 of 29 candidates tie at the exact
# maximum, and for a formal request it is routinely ALL of them. Something has to
# break those ties, and until now it was `>` keeping the first item in pool order
# - i.e. the lowest catalog id. Measured over 31 slot fills, the winner was the
# lowest tied id 31 times out of 31. The same handful of low-id products therefore
# won almost every request, which is what made results feel repetitive.
#
# Catalog id is an arbitrary basis for that choice. A hash of (request, id) is
# equally arbitrary but varies BETWEEN requests, which is the property we want:
#
#   same request  -> same salt -> same nudge -> identical output (determinism,
#                    reproducibility and every existing test are preserved)
#   different request -> different salt -> a different member of the tied group
#
# The magnitude is the safety argument. Every real score is rounded to 4 decimals
# before it reaches here, so the smallest genuine difference between two
# candidates is 5e-5. The nudge is capped at 1e-6, fifty times smaller - it can
# only ever reorder candidates that are already exactly equal, and can never
# promote a genuinely worse item over a better one.
TIEBREAK_EPSILON = 1e-6


def _salt_key(salt: str) -> bytes:
    """Compress the request signature to 8 bytes, once per build.

    The signature is the whole serialised request, so hashing it per candidate
    would rebuild a ~200 character string a few hundred times per recommendation.
    """
    return hashlib.blake2b(salt.encode(), digest_size=8).digest()


def _tiebreak(product: Product, salt_key: bytes) -> float:
    """A deterministic, request-dependent nudge in [0, TIEBREAK_EPSILON)."""
    digest = hashlib.blake2b(
        salt_key + int(product["id"]).to_bytes(8, "big", signed=True),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") / 2**64 * TIEBREAK_EPSILON


def _candidate_value(
    candidate: Product,
    chosen: Mapping[str, Product],
    solo_score: Callable[[Product], float],
    penalise_monotony: bool = True,
) -> float:
    """How good is this candidate given what we have already picked?

    Half compatibility with the partial outfit, half fit to the user's request.
    For the very first (anchor) item there is nothing to be compatible with, so
    it is scored purely on request fit.
    """
    fit = solo_score(candidate)
    if not chosen:
        return fit
    compat = outfit_compatibility([candidate, *chosen.values()], penalise_monotony)
    return 0.5 * compat + 0.5 * fit


def budget_fit(total_price: int, budget: int | None) -> float:
    """How well an outfit uses the stated budget, in [0, 1].

    Full marks for anything between 60% and 100% of budget. Below 60% the score
    tapers off: an outfit costing ₹400 against a ₹3,000 budget is technically
    valid but usually means we under-served the user. Above 100% is impossible -
    the builder never produces it.
    """
    if budget is None or budget <= 0:
        return 1.0
    ratio = total_price / budget
    if ratio > 1.0:
        return 0.0
    if ratio >= 0.6:
        return 1.0
    return round(ratio / 0.6, 4)
