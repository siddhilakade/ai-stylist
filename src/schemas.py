"""Typed contracts between the LLM, the recommender and the UI.

`StylePreferences` is the single most important type in the project: it is the
*only* thing the LLM is allowed to produce in the request path. The LLM never
sees the catalog, never names a product and never sets a price - it fills in this
small, closed-vocabulary struct, and the deterministic engine takes it from there.

Using Pydantic here is not decoration. It gives us:
  * a JSON Schema to hand to Gemini as a response schema (structured output),
  * validation of whatever actually comes back, and
  * a hard failure boundary: an invalid extraction is rejected and we fall back
    to the rule-based parser rather than feeding garbage into the filters.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

from src.features import (
    CANONICAL_GARMENTS,
    COLOR_FAMILIES,
    DEFAULT_OCCASION,
    EXPLICIT_COLOUR_MATCHES,
    GARMENT_ARTICLE_TYPES,
    OCCASION_PROFILES,
    STYLE_TARGET_FORMALITY,
    garment_slot,
)

# Closed vocabularies. Generated from the domain tables in features.py so the
# LLM's vocabulary and the engine's vocabulary can never drift apart.
Gender = Literal["Men", "Women", "Unisex"]
OCCASION_VALUES = tuple(OCCASION_PROFILES.keys())
STYLE_VALUES = tuple(STYLE_TARGET_FORMALITY.keys())
COLOR_FAMILY_VALUES = tuple(COLOR_FAMILIES.keys())

# Guard rails on budget. A budget below this cannot buy a complete outfit from
# this catalog (cheapest top + bottom + footwear is around Rs.850), and one above
# it is meaningless for a 536-item demo catalog.
MIN_SENSIBLE_BUDGET = 300
MAX_SENSIBLE_BUDGET = 100_000


class ExtractionSource(str, Enum):
    """Where a set of preferences came from - surfaced in the UI for honesty."""

    GEMINI = "gemini"
    RULE_BASED = "rule_based"
    FORM = "form"


class ItemRequest(BaseModel):
    """One explicitly requested garment, e.g. "a black shirt".

    This is a HARD constraint. If the user names a garment, the outfit must
    contain that garment; if they name its colour, it must be that colour.
    Unsatisfiable requests fail loudly rather than being quietly substituted.
    """

    model_config = {"extra": "forbid"}

    garment: str
    colour: Optional[str] = None

    @field_validator("garment", mode="before")
    @classmethod
    def _known_garment(cls, value: Any) -> str:
        if not isinstance(value, str) or value not in GARMENT_ARTICLE_TYPES:
            raise ValueError(f"unknown garment: {value!r}")
        return value

    @field_validator("colour", mode="before")
    @classmethod
    def _known_colour(cls, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        lowered = value.strip().lower()
        # Anything outside the colour vocabulary is dropped rather than guessed:
        # a constraint we cannot evaluate must not silently become a filter.
        return lowered if lowered in EXPLICIT_COLOUR_MATCHES else None

    @property
    def slot(self) -> str | None:
        return garment_slot(self.garment)

    @property
    def article_types(self) -> tuple[str, ...]:
        return GARMENT_ARTICLE_TYPES.get(self.garment, ())

    @property
    def base_colours(self) -> tuple[str, ...]:
        if not self.colour:
            return ()
        return EXPLICIT_COLOUR_MATCHES.get(self.colour, ())

    def describe(self) -> str:
        label = self.garment.replace("_", " ")
        return f"{self.colour} {label}" if self.colour else label


class StylePreferences(BaseModel):
    """Structured user intent. The LLM fills this in; the engine consumes it."""

    model_config = {"extra": "forbid"}

    gender: Gender = Field(
        default="Women",
        description="Who the outfit is for. Use Unisex only if genuinely unclear.",
    )
    occasion: str = Field(
        default=DEFAULT_OCCASION,
        description=f"One of: {', '.join(OCCASION_VALUES)}",
    )
    style: Optional[str] = Field(
        default=None,
        description=f"Explicit style if the user stated one. One of: {', '.join(STYLE_VALUES)}",
    )
    budget: Optional[int] = Field(
        default=None,
        description="Total budget for the whole outfit in INR. Null if not stated.",
    )
    preferred_colors: list[str] = Field(
        default_factory=list,
        description=(
            "SOFT preference only - general palette wishes like 'neutral tones'. "
            f"Each one of: {', '.join(COLOR_FAMILY_VALUES)}"
        ),
    )
    required_items: list[ItemRequest] = Field(
        default_factory=list,
        description=(
            "HARD constraints - garments the user named outright, e.g. "
            "'a black shirt'. The outfit must contain each of these."
        ),
    )
    include_accessory: bool = Field(
        default=True,
        description="Whether to add an accessory to the outfit.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any other stated preference, verbatim and short. Null if none.",
    )

    # --- validation ------------------------------------------------------
    # Everything below defends against a plausible-looking but wrong LLM
    # response. We coerce where coercion is unambiguous and reject otherwise.

    @field_validator("occasion", mode="before")
    @classmethod
    def _valid_occasion(cls, value: Any) -> str:
        if not isinstance(value, str) or value not in OCCASION_PROFILES:
            return DEFAULT_OCCASION
        return value

    @field_validator("style", mode="before")
    @classmethod
    def _valid_style(cls, value: Any) -> Optional[str]:
        if isinstance(value, str) and value in STYLE_TARGET_FORMALITY:
            return value
        return None

    @field_validator("budget", mode="before")
    @classmethod
    def _valid_budget(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            budget = int(float(value))
        except (TypeError, ValueError):
            return None
        if budget <= 0:
            return None
        return min(budget, MAX_SENSIBLE_BUDGET)

    @field_validator("preferred_colors", mode="before")
    @classmethod
    def _valid_colors(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return []
        # Drop anything outside our colour vocabulary rather than guessing.
        seen: list[str] = []
        for item in value:
            if isinstance(item, str) and item in COLOR_FAMILY_VALUES and item not in seen:
                seen.append(item)
        return seen[:3]  # more than three colour constraints is unsatisfiable

    @field_validator("notes", mode="before")
    @classmethod
    def _clip_notes(cls, value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip()[:200]

    @field_validator("required_items", mode="before")
    @classmethod
    def _valid_required_items(cls, value: Any) -> list[Any]:
        if not isinstance(value, (list, tuple)):
            return []
        kept: list[Any] = []
        for entry in value:
            try:
                kept.append(
                    entry if isinstance(entry, ItemRequest)
                    else ItemRequest.model_validate(entry)
                )
            except Exception:
                continue  # an unrecognised garment is dropped, never guessed
        return kept[:4]

    def constraints_by_slot(self) -> dict[str, dict[str, set[str]]]:
        """Merge the explicit item requests into one constraint per slot.

        Two requests can land on the same slot ("a saree or a dress"), in which
        case the article types and colours are unioned: the item must be one of
        the named garments, in one of the named colours.
        """
        merged: dict[str, dict[str, set[str]]] = {}
        colourless: set[str] = set()
        for request in self.required_items:
            slot = request.slot
            if slot is None:
                continue
            bucket = merged.setdefault(
                slot, {"article_types": set(), "base_colours": set(), "labels": set()}
            )
            bucket["article_types"].update(request.article_types)
            bucket["base_colours"].update(request.base_colours)
            bucket["labels"].add(request.describe())
            if not request.colour:
                colourless.add(slot)

        # If any request for a slot named no colour ("a saree or a red dress"),
        # drop the colour constraint for that slot entirely. Applying one
        # request's colour to another request's garment would invent a
        # constraint the user never stated.
        for slot in colourless:
            merged[slot]["base_colours"] = set()
        return merged

    @model_validator(mode="after")
    def _no_impossible_combinations(self) -> "StylePreferences":
        # Unisex has almost no topwear/bottomwear in the catalog, so treating a
        # Unisex request as a Women's request would silently mislead. We keep it,
        # but the recommender widens gender matching (see recommender.py).
        return self

    def budget_is_stated(self) -> bool:
        return self.budget is not None


class ExtractionResult(BaseModel):
    """A `StylePreferences` plus provenance, so the UI can be transparent."""

    preferences: StylePreferences
    source: ExtractionSource
    # Which LLM actually answered ("gemini" / "groq"), when one did. The UI shows
    # this because "an LLM understood you" is less useful than knowing which one,
    # especially when the primary provider has silently fallen over.
    provider: Optional[str] = None
    model: Optional[str] = None
    raw_model_output: Optional[str] = None
    warning: Optional[str] = None
    # True when `warning` describes the SYSTEM (no key, provider down, invalid
    # response) rather than the user's request. A shopper cannot act on
    # infrastructure status and should not be shown it as an alert; the
    # provenance badge already states which path answered. Warnings about the
    # request itself are actionable and stay visible.
    degraded: bool = False


# --------------------------------------------------------------------------
# The LLM-facing schema
# --------------------------------------------------------------------------
# This is deliberately a SEPARATE, flatter model from `StylePreferences`:
#
#   * every field is required and non-nullable. Structured-output APIs handle a
#     flat enum/int/bool/string schema far more reliably than one full of
#     `anyOf: [T, null]` unions, so we use sentinels ("unspecified", 0, "")
#     instead of nulls and translate them at the boundary.
#   * every free-choice field is an enum. The model cannot invent an occasion,
#     a style or a colour that the engine does not understand.
#   * there is no product id, product name or price field anywhere. The model is
#     structurally incapable of naming a product, because it has nowhere to put
#     one.
#
# The literals are spelled out rather than generated so the contract is readable
# at a glance; the assertions below fail loudly if they ever drift from the
# domain tables in features.py.

OccasionLiteral = Literal[
    "everyday_casual", "college", "work_office", "interview",
    "date_night", "party", "wedding_festive", "travel", "workout",
]
StyleLiteral = Literal[
    "relaxed", "casual", "sporty", "smart_casual", "party", "ethnic", "formal",
    "unspecified",
]
ColorLiteral = Literal[
    "neutral", "blue", "green", "red", "pink", "purple", "yellow", "orange",
    "brown", "multi",
]

assert set(get_args(OccasionLiteral)) == set(OCCASION_VALUES), "occasion vocabulary drift"
assert set(get_args(StyleLiteral)) == set(STYLE_VALUES) | {"unspecified"}, "style vocabulary drift"
assert set(get_args(ColorLiteral)) == set(COLOR_FAMILY_VALUES), "colour vocabulary drift"

NOT_SPECIFIED = "unspecified"
NO_BUDGET = 0


class GeminiItemRequest(BaseModel):
    """One explicitly named garment, as Gemini reports it."""

    model_config = {"extra": "ignore"}

    garment: str
    colour: str  # "" when the user named no colour for this garment


class GeminiExtraction(BaseModel):
    """Exactly what Gemini is asked to return. Nothing else is accepted."""

    model_config = {"extra": "ignore"}

    gender: Gender
    occasion: OccasionLiteral
    style: StyleLiteral
    budget_inr: int
    preferred_colors: list[ColorLiteral]
    # Requested in the response schema, but tolerated if absent: strict about
    # what we ask for, lenient about what we accept. An otherwise perfect
    # extraction should not be thrown away over one missing empty list.
    required_items: list[GeminiItemRequest] = Field(default_factory=list)
    include_accessory: bool
    notes: str

    def to_preferences(self) -> StylePreferences:
        """Translate sentinels into a validated `StylePreferences`."""
        return StylePreferences(
            gender=self.gender,
            occasion=self.occasion,
            style=None if self.style == NOT_SPECIFIED else self.style,
            budget=None if self.budget_inr <= NO_BUDGET else self.budget_inr,
            preferred_colors=list(self.preferred_colors),
            # Unrecognised garments are dropped by StylePreferences' validator
            # rather than rejected, so one bad entry cannot lose the whole
            # extraction.
            required_items=[
                {"garment": item.garment, "colour": item.colour or None}
                for item in self.required_items
            ],
            include_accessory=self.include_accessory,
            notes=self.notes or None,
        )
