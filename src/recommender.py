"""The recommendation pipeline.

    StylePreferences
        -> hard filters      (deterministic constraints; wrong items removed)
        -> candidate pools   (per outfit slot)
        -> greedy assembly   (outfit_builder, budget-aware)
        -> ranking           (transparent weighted sum)
        -> diversity pass    (disjoint outfits)
        -> RecommendationResult

No LLM is involved anywhere in this file. Gemini turns free text into
`StylePreferences` upstream, and narrates the result downstream; the selection
itself is entirely deterministic, reproducible and inspectable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src import semantic_retriever
from src.data import Product, catalog_records, get_product
from src.features import (
    DEFAULT_OCCASION,
    ETHNIC_OCCASIONS,
    ETHNIC_STYLES,
    FORMALITY_SCALE_MAX,
    OCCASION_PROFILES,
    SLOT_ACCESSORY,
    SLOT_BOTTOM,
    SLOT_FOOTWEAR,
    SLOT_ONEPIECE,
    SLOT_OUTERWEAR,
    SLOT_TOP,
    STYLE_TARGET_FORMALITY,
    resolve_formality_target,
)
from src.outfit_builder import (
    TEMPLATE_ONEPIECE,
    TEMPLATE_STANDARD,
    BuildFailure,
    Outfit,
    OutfitTemplate,
    build_outfit,
    slot_label,
)
from src.schemas import StylePreferences

# --------------------------------------------------------------------------
# Ranking weights
# --------------------------------------------------------------------------
# Three top-level terms, each in [0, 1]:
#
#   compatibility     do these items actually go together?  (the core problem)
#   preference_match  does each item honour what the user asked for?
#   budget_fit        does the outfit use the stated budget sensibly?
#
# Compatibility carries the most weight because outfit coherence is the product's
# entire reason to exist; budget is a tiebreaker because the builder has already
# guaranteed the outfit is affordable. These are round numbers with stated
# reasons, and they are NOT tuned against the evaluation set - fitting three
# weights to 25 scenarios would measure nothing.
WEIGHT_COMPATIBILITY = 0.45
WEIGHT_PREFERENCE = 0.35
WEIGHT_BUDGET = 0.20

# Sub-weights inside preference_match, for a single item.
PREF_WEIGHT_COLOR = 0.35
PREF_WEIGHT_FORMALITY = 0.30
PREF_WEIGHT_OCCASION = 0.25
PREF_WEIGHT_ETHNIC = 0.10

# Colour preference is a soft signal, never a hard filter: hard-filtering colour
# is the fastest way to produce an empty result set. A neutral still scores
# decently because neutrals coordinate with any requested palette.
COLOR_MATCH_EXACT = 1.0
COLOR_MATCH_NEUTRAL = 0.55
COLOR_MATCH_MISS = 0.15

# Occasions where Indian ethnic apparel would be out of place.
NON_ETHNIC_OCCASIONS = {"work_office", "interview", "workout"}

# Slots where an explicit "ethnic" request is a hard requirement.
ETHNIC_ENFORCED_SLOTS = {SLOT_TOP, SLOT_BOTTOM, SLOT_ONEPIECE}

# Extra formality tolerance for accessories - see hard_filter for why.
ACCESSORY_TOLERANCE_BONUS = 1.0

# Quality floor, relative to the best outfit found. An outfit scoring far below
# the leader is not a useful "alternative" - it is a worse answer to the same
# question, and showing it makes the whole result set look unconsidered.
RELATIVE_QUALITY_FLOOR = 0.80

# If a required slot empties out, widen the formality band once by this much
# before giving up. Reported to the user rather than applied silently.
FORMALITY_RELAXATION_STEP = 0.75

DEFAULT_NUM_OUTFITS = 3


def request_salt(prefs: StylePreferences, extra: str = "") -> str:
    """A stable string identifying this request, used to break scoring ties.

    Ties are frequent (see `outfit_builder._tiebreak`) and something arbitrary
    has to settle them. Deriving that arbitrariness from the request rather than
    from catalog order means two different requests land on different members of
    a tied group, while the same request always lands on the same one - so the
    engine stays fully deterministic and reproducible.
    """
    return f"{prefs.model_dump_json()}|{extra}"


# --------------------------------------------------------------------------
# Conflicting requests
# --------------------------------------------------------------------------
# A user can state an occasion and a style that pull in opposite directions
# ("a formal gym outfit"). The engine still returns its best attempt - refusing
# would be worse - but it must say out loud that the request was contradictory,
# rather than quietly honouring one half and dropping the other.

# Two full tiers apart on the 0-4 scale, i.e. a genuine contradiction rather
# than a mild stretch.
CONFLICT_FORMALITY_GAP = 2.0


def detect_conflicts(prefs: StylePreferences) -> list[str]:
    """Human-readable warnings about self-contradictory requests."""
    conflicts: list[str] = []
    profile = OCCASION_PROFILES.get(prefs.occasion, OCCASION_PROFILES[DEFAULT_OCCASION])

    if prefs.style in STYLE_TARGET_FORMALITY:
        gap = abs(STYLE_TARGET_FORMALITY[prefs.style] - float(profile["target_formality"]))
        if gap >= CONFLICT_FORMALITY_GAP:
            conflicts.append(
                f"“{prefs.style.replace('_', ' ')}” and “{profile['label']}” pull in "
                "opposite directions on formality. The stated style was used, so "
                "these looks may be dressier or more relaxed than the occasion needs."
            )

    if prefs.style in ETHNIC_STYLES and prefs.occasion in NON_ETHNIC_OCCASIONS:
        conflicts.append(
            f"Ethnic wear is unusual for “{profile['label']}”. The ethnic "
            "requirement was honoured; judge the result accordingly."
        )

    if len(prefs.preferred_colors) >= 2:
        strong = [c for c in prefs.preferred_colors if c != "neutral"]
        if len(strong) >= 2:
            conflicts.append(
                f"Two strong colour families were requested ({', '.join(strong)}). "
                "Outfits combining both will score lower on colour coordination."
            )

    return conflicts


@dataclass
class RecommendationResult:
    """Everything the UI needs, including why things did or did not work."""

    outfits: list[Outfit] = field(default_factory=list)
    failure: BuildFailure | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.outfits)


# --------------------------------------------------------------------------
# 1. Hard filters
# --------------------------------------------------------------------------

def hard_filter(
    products: Sequence[Product],
    prefs: StylePreferences,
    tolerance_bonus: float = 0.0,
) -> list[Product]:
    """Remove items that cannot satisfy the request, whatever their score.

    Five constraints, each a genuine yes/no:
      gender      a men's kurta is not a candidate for a women's outfit
      explicit    a garment the user named outright ("a black shirt") pins both
                  the article type and, if stated, the colour of its slot
      formality   the item must sit inside the occasion's formality band
      ethnic      ethnic apparel is excluded from office/interview/gym requests,
                  and required when the user asked for ethnic wear
      budget      no single item may cost more than the entire outfit budget

    Note the asymmetry in how colour is treated, which is the whole point:
    a colour attached to a named garment is enforced here, while a general
    palette preference ("neutral tones") is only scored. Hard-filtering a vague
    preference is the fastest way to an empty result set; NOT enforcing a
    specific request is how you end up returning a white top for "black shirt".
    """
    target, tolerance = resolve_formality_target(prefs.occasion, prefs.style)
    item_constraints = prefs.constraints_by_slot()
    tolerance += tolerance_bonus
    wants_ethnic = (prefs.style in ETHNIC_STYLES) or (prefs.occasion in ETHNIC_OCCASIONS)

    # "Unisex" means "show me anything", not "show me only items the dataset
    # happened to tag Unisex". The source tags almost no topwear or bottomwear
    # as Unisex (mostly bags, caps and eyewear), so restricting to that label
    # makes a complete outfit impossible - a bug found by evaluation scenario S24.
    allowed_genders = (
        {"Men", "Women", "Unisex"} if prefs.gender == "Unisex"
        else {prefs.gender, "Unisex"}
    )

    kept: list[Product] = []
    for product in products:
        if product["gender"] not in allowed_genders:
            continue

        slot = product["outfit_slot"]

        # Explicit request: this slot was named by the user, so only the named
        # garment (and colour, if given) can fill it.
        constraint = item_constraints.get(slot)
        if constraint:
            if product["articleType"] not in constraint["article_types"]:
                continue
            colours = constraint["base_colours"]
            if colours and product["baseColour"] not in colours:
                continue
            # This slot is exempt from the FORMALITY band, and only that. The
            # band comes from an occasion we usually INFERRED rather than were
            # told, so applying it here could only discard items the user named
            # outright: "white shirt" would fail, because most white shirts in
            # the catalog are tagged Formal while the default occasion is
            # casual. Formality still shapes every other slot and still SCORES
            # this one, so the outfit built around the requested garment stays
            # coherent - we just never veto the thing that was asked for.
            #
            # Budget is NOT exempt, because it is stated rather than inferred: an
            # item costing more than the entire outfit cannot appear in any valid
            # outfit, whoever asked for it. Keeping those items here used to make
            # `unsatisfied_constraints` believe a request was satisfiable when it
            # was not, so "a saree, budget 1000" failed later with the generic
            # "no one-piece matches your constraints" instead of the specific
            # "the catalog has no saree for Women under 1,000".
            if prefs.budget is not None and int(product["price"]) > prefs.budget:
                continue
            kept.append(product)
            continue

        # Accessories get a wider formality band. In this dataset almost every
        # belt, watch and bag is tagged "Casual" regardless of how formal it
        # actually looks, so a tight band would silently remove accessories from
        # every formal outfit. Compatibility scoring still penalises a genuinely
        # mismatched accessory - we just stop pretending the label is reliable.
        slot_tolerance = tolerance + (ACCESSORY_TOLERANCE_BONUS if slot == SLOT_ACCESSORY else 0.0)
        if abs(float(product["formality"]) - target) > slot_tolerance:
            continue

        # The "must be ethnic" rule applies only to the main garments. Ethnic
        # footwear and outerwear barely exist in the source data (and in real
        # life people wear ordinary flats with a kurta), so enforcing it there
        # would starve those slots for no styling benefit.
        if slot in ETHNIC_ENFORCED_SLOTS:
            if wants_ethnic and prefs.style in ETHNIC_STYLES and not product["is_ethnic"]:
                continue
        if slot != SLOT_ACCESSORY:
            if not wants_ethnic and product["is_ethnic"] and prefs.occasion in NON_ETHNIC_OCCASIONS:
                continue

        if prefs.budget is not None and int(product["price"]) > prefs.budget:
            continue

        kept.append(product)
    return kept


def unsatisfied_constraints(
    prefs: StylePreferences, pools: Mapping[str, Sequence[Product]]
) -> list[str]:
    """Explicit requests for which no candidate survived filtering."""
    return [
        ", ".join(sorted(constraint["labels"]))
        for slot, constraint in prefs.constraints_by_slot().items()
        if not pools.get(slot)
    ]


def describe_unmet_constraints(unmet: list[str], prefs: StylePreferences) -> BuildFailure:
    """Say which requested item is unavailable, and what would help."""
    wanted = " and ".join(f"“{label}”" for label in unmet)
    qualifiers = [f"for {prefs.gender}"]
    if prefs.budget is not None:
        qualifiers.append(f"under ₹{prefs.budget:,}")

    return BuildFailure(
        reason_code="unsatisfiable_item_request",
        message=(
            f"The catalog has no {wanted} {' '.join(qualifiers)} that also suits "
            "this occasion."
        ),
        suggestion=(
            "Try a different colour or garment, raise the budget, or drop the "
            "specific item and let the stylist choose."
        ),
        detail={"unmet": unmet},
    )


def candidates_by_slot(products: Sequence[Product]) -> dict[str, list[Product]]:
    pools: dict[str, list[Product]] = {
        SLOT_TOP: [], SLOT_BOTTOM: [], SLOT_FOOTWEAR: [],
        SLOT_OUTERWEAR: [], SLOT_ACCESSORY: [], SLOT_ONEPIECE: [],
    }
    for product in products:
        slot = product["outfit_slot"]
        if slot in pools:
            pools[slot].append(product)
    return pools


# --------------------------------------------------------------------------
# 2. Per-item preference match
# --------------------------------------------------------------------------

def preference_match(product: Product, prefs: StylePreferences) -> float:
    """How well one item honours the stated request, in [0, 1]."""
    target, _ = resolve_formality_target(prefs.occasion, prefs.style)
    profile = OCCASION_PROFILES.get(prefs.occasion, OCCASION_PROFILES[DEFAULT_OCCASION])

    # colour
    if not prefs.preferred_colors:
        color = 1.0
    elif product["color_family"] in prefs.preferred_colors:
        color = COLOR_MATCH_EXACT
    elif bool(product["is_neutral"]):
        color = COLOR_MATCH_NEUTRAL
    else:
        color = COLOR_MATCH_MISS

    # formality proximity to the request's target
    formality = max(
        0.0, 1.0 - abs(float(product["formality"]) - target) / FORMALITY_SCALE_MAX
    )

    # the dataset's own occasion tag, used softly
    occasion = 1.0 if product["usage"] in profile["preferred_usage"] else 0.5

    # ethnic alignment
    wants_ethnic = (prefs.style in ETHNIC_STYLES) or (prefs.occasion in ETHNIC_OCCASIONS)
    if product["outfit_slot"] == SLOT_ACCESSORY:
        ethnic = 1.0  # accessories are largely occasion-agnostic
    elif wants_ethnic:
        ethnic = 1.0 if product["is_ethnic"] else 0.3
    else:
        ethnic = 0.7 if product["is_ethnic"] else 1.0

    return round(
        PREF_WEIGHT_COLOR * color
        + PREF_WEIGHT_FORMALITY * formality
        + PREF_WEIGHT_OCCASION * occasion
        + PREF_WEIGHT_ETHNIC * ethnic,
        4,
    )


# --------------------------------------------------------------------------
# 3. Ranking
# --------------------------------------------------------------------------

def score_outfit(outfit: Outfit) -> float:
    """Transparent weighted sum. Every term is already on a [0, 1] scale."""
    return round(
        WEIGHT_COMPATIBILITY * outfit.compatibility
        + WEIGHT_PREFERENCE * outfit.preference_match
        + WEIGHT_BUDGET * outfit.budget_fit,
        4,
    )


def score_breakdown(outfit: Outfit) -> list[dict[str, Any]]:
    """Row-per-term breakdown, rendered directly in the UI's 'Why' panel."""
    return [
        {"term": "Compatibility", "value": outfit.compatibility,
         "weight": WEIGHT_COMPATIBILITY,
         "contribution": round(WEIGHT_COMPATIBILITY * outfit.compatibility, 4)},
        {"term": "Preference match", "value": outfit.preference_match,
         "weight": WEIGHT_PREFERENCE,
         "contribution": round(WEIGHT_PREFERENCE * outfit.preference_match, 4)},
        {"term": "Budget fit", "value": outfit.budget_fit,
         "weight": WEIGHT_BUDGET,
         "contribution": round(WEIGHT_BUDGET * outfit.budget_fit, 4)},
    ]


# --------------------------------------------------------------------------
# 4. Templates to try
# --------------------------------------------------------------------------

def applicable_templates(prefs: StylePreferences, pools: Mapping[str, Sequence[Product]]) -> list[OutfitTemplate]:
    """Which outfit shapes make sense for this request?

    An explicitly named garment settles it: asking for a dress means the
    one-piece shape, and asking for jeans means separates. Offering the other
    shape would produce an outfit that cannot contain what the user asked for.
    """
    constrained_slots = set(prefs.constraints_by_slot())
    if SLOT_ONEPIECE in constrained_slots:
        return [TEMPLATE_ONEPIECE]
    if constrained_slots & {SLOT_TOP, SLOT_BOTTOM}:
        return [TEMPLATE_STANDARD]

    templates = [TEMPLATE_STANDARD]
    if pools.get(SLOT_ONEPIECE):
        templates.append(TEMPLATE_ONEPIECE)
    return templates


# --------------------------------------------------------------------------
# 5. Main entry point
# --------------------------------------------------------------------------

def recommend_outfits(
    prefs: StylePreferences,
    num_outfits: int = DEFAULT_NUM_OUTFITS,
    products: Sequence[Product] | None = None,
) -> RecommendationResult:
    """Generate up to `num_outfits` distinct, ranked, budget-valid outfits."""
    started = time.perf_counter()
    catalog = list(products) if products is not None else list(catalog_records())

    diagnostics: dict[str, Any] = {
        "catalog_size": len(catalog),
        "relaxed": [],
        "conflicts": detect_conflicts(prefs),
    }

    filtered = hard_filter(catalog, prefs)
    pools = candidates_by_slot(filtered)

    # Single, reported relaxation step: if a required slot came back empty, widen
    # the formality band rather than failing outright. The user is told this
    # happened. Explicitly requested slots are included, because "the only red
    # dresses are dressier than this occasion implies" is exactly the case where
    # a wider band is the right answer - and the garment itself is never relaxed.
    core_slots = tuple(
        {SLOT_TOP, SLOT_BOTTOM, SLOT_FOOTWEAR} | set(prefs.constraints_by_slot())
    )
    if any(not pools[s] for s in core_slots):
        widened = hard_filter(catalog, prefs, tolerance_bonus=FORMALITY_RELAXATION_STEP)
        widened_pools = candidates_by_slot(widened)
        if all(widened_pools[s] for s in core_slots):
            filtered, pools = widened, widened_pools
            diagnostics["relaxed"].append(
                "Widened the formality range to find enough matching items."
            )

    # SEMANTIC RETRIEVAL. Runs strictly AFTER hard filtering, so it only ever
    # sees products that already satisfy gender, explicit garment and colour,
    # formality band, ethnic rules and budget. It can reorder and narrow that
    # pool; it can never add a product back or overrule a constraint.
    #
    # Its job is the part the rule engine structurally cannot do: act on the
    # free-text remainder of a request ("nothing too flashy") that has no
    # structured field to live in. Retrieval is per slot, never global, so it
    # cannot empty a required slot and manufacture a failure.
    filtered, semantic_diagnostics = semantic_retriever.rerank_candidates(filtered, prefs)
    diagnostics["semantic"] = semantic_diagnostics
    if semantic_diagnostics.get("applied"):
        pools = candidates_by_slot(filtered)

    diagnostics["candidates_after_filter"] = {s: len(v) for s, v in pools.items()}

    # An explicit request that nothing satisfies is reported in the user's own
    # words. This is the single most useful failure message the system produces:
    # "no black shirt" is actionable, "no top" is not.
    unmet = unsatisfied_constraints(prefs, pools)
    if unmet:
        return RecommendationResult(
            failure=describe_unmet_constraints(unmet, prefs),
            diagnostics=diagnostics,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def solo(product: Product) -> float:
        return preference_match(product, prefs)

    outfits: list[Outfit] = []
    used_ids: set[int] = set()
    last_failure: BuildFailure | None = None
    constrained_slots = set(prefs.constraints_by_slot())
    salt = request_salt(prefs)
    # "black and white" resolves to the neutral family, so an all-neutral outfit
    # is exactly what was asked for. Leaving the monotony penalty on would mark
    # compliance down - and, worse, make adding a coloured accessory *raise*
    # compatibility. See MONOTONY_PENALTY in compatibility.py.
    penalise_monotony = "neutral" not in prefs.preferred_colors

    templates = applicable_templates(prefs, pools)
    if not prefs.include_accessory:
        templates = [
            OutfitTemplate(
                key=t.key, label=t.label,
                required_slots=t.required_slots,
                optional_slots=tuple(s for s in t.optional_slots if s != SLOT_ACCESSORY),
            )
            for t in templates
        ]

    # Round-robin over templates so a women's request returns a mix of
    # separates and one-piece looks rather than three near-identical dresses.
    attempt = 0
    max_attempts = num_outfits * len(templates) + 2
    while len(outfits) < num_outfits and attempt < max_attempts:
        template = templates[attempt % len(templates)]
        attempt += 1

        result = build_outfit(
            candidates_by_slot=pools,
            template=template,
            budget=prefs.budget,
            solo_score=solo,
            # Diversity: later outfits may not reuse any item from earlier ones.
            # Simple, guarantees visibly different results, and needs no
            # similarity metric between outfits.
            exclude_ids=frozenset(used_ids),
            tiebreak_salt=salt,
            penalise_monotony=penalise_monotony,
        )
        if isinstance(result, BuildFailure):
            last_failure = result
            continue

        result.final_score = score_outfit(result)
        outfits.append(result)
        # Diversity excludes reuse - except in slots the user pinned by name.
        # If someone asks for a black shirt and the catalog holds one, excluding
        # it after the first look means we can only ever show one look. The
        # requested garment repeats; everything styled around it varies.
        used_ids.update(
            int(item["id"])
            for slot, item in result.items.items()
            if slot not in constrained_slots
        )

    outfits.sort(key=lambda o: o.final_score, reverse=True)
    if outfits:
        floor = outfits[0].final_score * RELATIVE_QUALITY_FLOOR
        kept = [o for o in outfits if o.final_score >= floor]
        if len(kept) < len(outfits):
            diagnostics["dropped_below_quality_floor"] = len(outfits) - len(kept)
        outfits = kept

    latency = (time.perf_counter() - started) * 1000

    if not outfits:
        return RecommendationResult(
            outfits=[],
            failure=last_failure or _generic_failure(prefs, pools),
            diagnostics=diagnostics,
            latency_ms=round(latency, 2),
        )

    return RecommendationResult(
        outfits=outfits, diagnostics=diagnostics, latency_ms=round(latency, 2)
    )


def _generic_failure(prefs: StylePreferences, pools: Mapping[str, Sequence[Product]]) -> BuildFailure:
    empty = [s for s in (SLOT_TOP, SLOT_BOTTOM, SLOT_FOOTWEAR) if not pools.get(s)]
    if empty:
        return BuildFailure(
            reason_code="empty_slot",
            message=(
                "The catalog has no "
                f"{', '.join(slot_label(s) for s in empty)} matching this request."
            ),
            suggestion="Try a different occasion, gender or budget.",
            detail={"empty_slots": empty},
        )
    return BuildFailure(
        reason_code="no_outfit",
        message="Could not assemble a complete outfit from the matching items.",
        suggestion="Try relaxing the budget or the style constraint.",
        detail={},
    )


# --------------------------------------------------------------------------
# 6. Complete the Look
# --------------------------------------------------------------------------

def infer_preferences_from_product(
    product: Product,
    budget: int | None = None,
    gender: str | None = None,
) -> StylePreferences:
    """Derive a request from an anchor product, for 'Complete the Look'.

    The anchor's own formality tier and occasion tag tell us what kind of outfit
    the user is building, so we pick the occasion whose formality target sits
    closest to the anchor's.
    """
    anchor_formality = float(product["formality"])
    occasion = min(
        OCCASION_PROFILES,
        key=lambda key: abs(OCCASION_PROFILES[key]["target_formality"] - anchor_formality),
    )
    if product["is_ethnic"]:
        occasion = "wedding_festive"

    resolved_gender = gender or product["gender"]
    if resolved_gender == "Unisex":
        resolved_gender = "Women"

    return StylePreferences(
        gender=resolved_gender,
        occasion=occasion,
        style="ethnic" if product["is_ethnic"] else None,
        budget=budget,
        preferred_colors=[],
        include_accessory=True,
    )


def complete_the_look(
    anchor_id: int,
    budget: int | None = None,
    num_outfits: int = DEFAULT_NUM_OUTFITS,
    prefs: StylePreferences | None = None,
) -> RecommendationResult:
    """Build outfits around a product the user already chose.

    The anchor is pinned: it is the only candidate in its slot, so every returned
    outfit necessarily contains it. Everything else runs through the exact same
    filter/score/assemble path as Style Me - there is no second code path to keep
    in sync, and no way for the anchor to be quietly swapped out.
    """
    started = time.perf_counter()
    anchor = get_product(anchor_id)
    if anchor is None:
        return RecommendationResult(
            failure=BuildFailure(
                reason_code="unknown_product",
                message="That product is not in the catalog.",
                suggestion="Pick a product from the catalog grid.",
            ),
            latency_ms=0.0,
        )

    prefs = prefs or infer_preferences_from_product(anchor, budget=budget)
    anchor_slot = anchor["outfit_slot"]

    catalog = [p for p in catalog_records() if int(p["id"]) != int(anchor["id"])]
    filtered = hard_filter(catalog, prefs)
    pools = candidates_by_slot(filtered)

    diagnostics: dict[str, Any] = {"anchor_id": int(anchor["id"]), "relaxed": []}
    # Relax once if pinning the anchor left one of the other required slots empty.
    template = TEMPLATE_ONEPIECE if anchor_slot == SLOT_ONEPIECE else TEMPLATE_STANDARD
    needed = [s for s in template.required_slots if s != anchor_slot]
    if any(not pools[s] for s in needed):
        widened = hard_filter(catalog, prefs, tolerance_bonus=FORMALITY_RELAXATION_STEP)
        widened_pools = candidates_by_slot(widened)
        if all(widened_pools[s] for s in needed):
            pools = widened_pools
            diagnostics["relaxed"].append(
                "Widened the formality range to find items that match this product."
            )

    # Pin the anchor: it becomes the ONLY candidate in its slot, so the builder
    # cannot drop it. If the anchor is an accessory or a layer, that slot is
    # optional in the template - so we also promote it to required, otherwise a
    # tight budget could legitimately skip the very product the user picked.
    pools = dict(pools)
    pools[anchor_slot] = [anchor]
    if anchor_slot in template.optional_slots:
        template = OutfitTemplate(
            key=template.key,
            label=template.label,
            required_slots=template.required_slots + (anchor_slot,),
            optional_slots=tuple(s for s in template.optional_slots if s != anchor_slot),
        )

    def solo(product: Product) -> float:
        return preference_match(product, prefs)

    outfits: list[Outfit] = []
    used_ids: set[int] = set()
    last_failure: BuildFailure | None = None
    # The anchor is part of the request here, so two different anchors that
    # happen to infer the same preferences still get different tie-breaks.
    salt = request_salt(prefs, extra=str(anchor["id"]))
    penalise_monotony = "neutral" not in prefs.preferred_colors

    for _ in range(num_outfits):
        result = build_outfit(
            candidates_by_slot=pools,
            template=template,
            budget=prefs.budget,
            solo_score=solo,
            exclude_ids=frozenset(used_ids),
            tiebreak_salt=salt,
            penalise_monotony=penalise_monotony,
        )
        if isinstance(result, BuildFailure):
            last_failure = result
            break
        result.final_score = score_outfit(result)
        outfits.append(result)
        # Keep the anchor available for the next outfit; vary everything else.
        used_ids.update(pid for pid in result.product_ids if pid != int(anchor["id"]))

    diagnostics["candidates_after_filter"] = {s: len(v) for s, v in pools.items()}
    latency = (time.perf_counter() - started) * 1000

    if not outfits:
        return RecommendationResult(
            failure=last_failure
            or BuildFailure(
                reason_code="no_outfit",
                message="Could not build a complete look around this product.",
                suggestion="Try increasing the budget.",
            ),
            diagnostics=diagnostics,
            latency_ms=round(latency, 2),
        )

    outfits.sort(key=lambda o: o.final_score, reverse=True)
    return RecommendationResult(
        outfits=outfits, diagnostics=diagnostics, latency_ms=round(latency, 2)
    )


# --------------------------------------------------------------------------
# 7. Human-readable reasons (the grounding payload)
# --------------------------------------------------------------------------
# These strings are generated from the numbers above - never written by an LLM.
# They are what the UI shows as bullet points, and they are exactly what gets
# handed to Gemini to paraphrase.

SIGNAL_LABELS = {
    "color": "Colour coordination",
    "formality": "Formality consistency",
    "occasion": "Occasion suitability",
    "ethnic": "Style coherence",
}


def outfit_reasons(outfit: Outfit, prefs: StylePreferences) -> list[str]:
    """Factual bullet points backing one outfit."""
    reasons: list[str] = []
    tiers = {item["formality_tier"] for item in outfit.items.values()}
    if len(tiers) == 1:
        reasons.append(f"Every piece sits at the same {next(iter(tiers))} level")
    else:
        reasons.append(
            "Formality consistency "
            f"{outfit.signals['formality']:.0%} across {', '.join(sorted(tiers))}"
        )

    neutrals = sum(1 for i in outfit.items.values() if i["is_neutral"])
    reasons.append(
        f"Colour coordination {outfit.signals['color']:.0%} "
        f"({neutrals} of {len(outfit.items)} pieces are neutral)"
    )

    profile = OCCASION_PROFILES.get(prefs.occasion, OCCASION_PROFILES[DEFAULT_OCCASION])
    reasons.append(
        f"Occasion suitability {outfit.signals['occasion']:.0%} for {profile['label']}"
    )

    if prefs.preferred_colors:
        hits = [
            i["baseColour"] for i in outfit.items.values()
            if i["color_family"] in prefs.preferred_colors
        ]
        if hits:
            reasons.append(
                f"Matches your colour preference: {', '.join(sorted(set(hits)))}"
            )
        else:
            reasons.append(
                "No exact colour match available - selected neutrals that "
                "coordinate with your preference"
            )

    if prefs.budget is not None:
        headroom = outfit.budget_headroom() or 0
        reasons.append(
            f"Total ₹{outfit.total_price:,} of your ₹{prefs.budget:,} budget "
            f"(₹{headroom:,} left)"
        )
    else:
        reasons.append(f"Total ₹{outfit.total_price:,}")

    return reasons


def grounding_payload(outfit: Outfit, prefs: StylePreferences) -> dict[str, Any]:
    """The ONLY thing handed to Gemini when it writes an explanation.

    It contains the real products the engine chose and the real numbers behind
    that choice. Gemini never sees the catalog, so it has nothing to invent from.
    """
    return {
        "request": {
            "gender": prefs.gender,
            "occasion": OCCASION_PROFILES.get(
                prefs.occasion, OCCASION_PROFILES[DEFAULT_OCCASION]
            )["label"],
            "style": prefs.style,
            "budget_inr": prefs.budget,
            "preferred_colors": prefs.preferred_colors,
            "notes": prefs.notes,
        },
        "selected_items": [
            {
                "slot": slot,
                "name": item["productDisplayName"],
                "article_type": item["articleType"],
                "colour": item["baseColour"],
                "formality_tier": item["formality_tier"],
                "price_inr": int(item["price"]),
            }
            for slot, item in outfit.ordered_items
        ],
        "total_price_inr": outfit.total_price,
        "signals": {
            SIGNAL_LABELS[k]: round(v, 3) for k, v in outfit.signals.items()
        },
        "scores": {
            "compatibility": outfit.compatibility,
            "preference_match": outfit.preference_match,
            "budget_fit": outfit.budget_fit,
            "final_score": outfit.final_score,
        },
    }
