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

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from src.compatibility import is_hard_clash, outfit_compatibility, outfit_signals
from src.features import (
    SLOT_ACCESSORY,
    SLOT_BOTTOM,
    SLOT_FOOTWEAR,
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
) -> Outfit | BuildFailure:
    """Assemble one outfit greedily. Returns an Outfit or a BuildFailure."""

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
                + _join_naturally(empty)
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

            value = _candidate_value(candidate, chosen, solo_score)
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

        best = max(affordable, key=lambda c: _candidate_value(c, chosen, solo_score))
        # Only add an optional item if it genuinely fits the look. A weak
        # accessory makes the outfit worse, not better.
        if _candidate_value(best, chosen, solo_score) < OPTIONAL_SLOT_THRESHOLD:
            continue
        # And it must not drag the outfit down. Without this an optional layer
        # can clear the bar on its own merits while still being wrong for the
        # look - a denim waistcoat over a saree scores acceptably in isolation
        # and ruins the outfit in context.
        before = outfit_compatibility(list(chosen.values()))
        after = outfit_compatibility([best, *chosen.values()])
        if after < before - OPTIONAL_SLOT_MAX_COMPATIBILITY_DROP:
            continue
        chosen[slot] = best
        spent += int(best["price"])

    items = list(chosen.values())
    return Outfit(
        template=template.key,
        items=chosen,
        total_price=spent,
        compatibility=outfit_compatibility(items),
        signals=outfit_signals(items),
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


def _candidate_value(
    candidate: Product,
    chosen: Mapping[str, Product],
    solo_score: Callable[[Product], float],
) -> float:
    """How good is this candidate given what we have already picked?

    Half compatibility with the partial outfit, half fit to the user's request.
    For the very first (anchor) item there is nothing to be compatible with, so
    it is scored purely on request fit.
    """
    fit = solo_score(candidate)
    if not chosen:
        return fit
    compat = outfit_compatibility([candidate, *chosen.values()])
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
