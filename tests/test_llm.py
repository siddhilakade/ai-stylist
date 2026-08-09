"""The LLM boundary: schema validation, malformed responses, and grounding.

No test here calls the network. They exercise the code that decides whether to
*trust* a model response - which is the part that has to be right.
"""

from __future__ import annotations

import json

import pytest

from src import llm_client
from src.llm_client import (
    EXPLANATION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    _safe_error,
    _template_explanation,
    explain_outfit,
    extract_preferences,
    validate_explanation,
)
from src.nlu import parse_request
from src.recommender import grounding_payload, recommend_outfits
from src.schemas import (
    COLOR_FAMILY_VALUES,
    OCCASION_VALUES,
    STYLE_VALUES,
    GeminiExtraction,
    StylePreferences,
)


@pytest.fixture(scope="module")
def payload():
    prefs = StylePreferences(gender="Men", occasion="college", budget=5000)
    outfit = recommend_outfits(prefs).outfits[0]
    return grounding_payload(outfit, prefs)


class TestStructuredOutputSchema:
    def test_valid_response_parses(self):
        raw = json.dumps({
            "gender": "Men",
            "occasion": "interview",
            "style": "smart_casual",
            "budget_inr": 3000,
            "preferred_colors": ["neutral"],
            "include_accessory": True,
            "notes": "nothing flashy",
        })
        prefs = GeminiExtraction.model_validate_json(raw).to_preferences()
        assert prefs.gender == "Men"
        assert prefs.occasion == "interview"
        assert prefs.style == "smart_casual"
        assert prefs.budget == 3000
        assert prefs.preferred_colors == ["neutral"]

    def test_sentinels_become_none(self):
        raw = json.dumps({
            "gender": "Women", "occasion": "college", "style": "unspecified",
            "budget_inr": 0, "preferred_colors": [], "include_accessory": True,
            "notes": "",
        })
        prefs = GeminiExtraction.model_validate_json(raw).to_preferences()
        assert prefs.style is None
        assert prefs.budget is None
        assert prefs.notes is None

    @pytest.mark.parametrize("bad", [
        "not json at all",
        "",
        "{}",
        '{"gender": "Martian", "occasion": "interview", "style": "formal", '
        '"budget_inr": 1, "preferred_colors": [], "include_accessory": true, "notes": ""}',
        '{"gender": "Men", "occasion": "space_walk", "style": "formal", '
        '"budget_inr": 1, "preferred_colors": [], "include_accessory": true, "notes": ""}',
        '{"gender": "Men", "occasion": "college", "style": "formal", '
        '"budget_inr": 1, "preferred_colors": ["chartreuse"], '
        '"include_accessory": true, "notes": ""}',
    ])
    def test_malformed_or_out_of_vocabulary_responses_are_rejected(self, bad):
        with pytest.raises(Exception):
            GeminiExtraction.model_validate_json(bad)

    def test_schema_has_no_field_that_could_carry_a_product(self):
        fields = set(GeminiExtraction.model_fields)
        for forbidden in ("product", "product_id", "id", "price", "name", "items"):
            assert forbidden not in fields

    def test_prompt_states_the_vocabulary_the_code_enforces(self):
        assert "budget_inr" in EXTRACTION_SYSTEM_PROMPT
        assert "unspecified" in EXTRACTION_SYSTEM_PROMPT
        assert "NOT choosing products" in EXTRACTION_SYSTEM_PROMPT


class TestPreferenceValidation:
    def test_out_of_range_values_are_coerced_or_dropped(self):
        prefs = StylePreferences(
            occasion="nonsense", style="nonsense",
            budget="not a number", preferred_colors=["chartreuse", "blue"],
        )
        assert prefs.occasion in OCCASION_VALUES
        assert prefs.style is None
        assert prefs.budget is None
        assert prefs.preferred_colors == ["blue"]

    def test_negative_budget_becomes_none(self):
        assert StylePreferences(budget=-500).budget is None

    def test_colour_list_is_capped(self):
        prefs = StylePreferences(preferred_colors=list(COLOR_FAMILY_VALUES))
        assert len(prefs.preferred_colors) <= 3

    def test_vocabularies_do_not_drift(self):
        # The literals in schemas.py assert this at import time; re-check here so
        # the failure is a named test rather than a collection error.
        assert set(STYLE_VALUES)
        assert set(OCCASION_VALUES)


class TestFallbackPath:
    def test_extraction_without_a_key_still_returns_valid_preferences(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        result = extract_preferences(
            "smart casual for a college presentation under 3000", allow_llm=False
        )
        assert result.source.value == "rule_based"
        assert result.preferences.budget == 3000
        assert result.preferences.style == "smart_casual"

    def test_empty_request_is_handled(self):
        result = extract_preferences("", allow_llm=False)
        assert result.preferences is not None
        assert result.warning

    def test_rule_parser_covers_the_documented_examples(self):
        prefs = parse_request("Need a formal outfit for an office meeting, navy, budget 8000")
        assert prefs.occasion == "work_office"
        assert prefs.style == "formal"
        assert prefs.budget == 8000
        assert prefs.preferred_colors == ["blue"]

    def test_explanation_falls_back_to_a_grounded_template(self, payload):
        result = explain_outfit(payload, allow_llm=False)
        assert result.source == "template"
        assert str(payload["total_price_inr"]) in result.text.replace(",", "")


class TestExplanationGrounding:
    def test_accepts_an_explanation_that_only_uses_given_figures(self, payload):
        total = payload["total_price_inr"]
        text = f"This smart-casual look works well together and comes to Rs.{total:,} in total."
        ok, problem = validate_explanation(text, payload)
        assert ok, problem

    def test_rejects_an_invented_price(self, payload):
        ok, problem = validate_explanation(
            "A great outfit, and the shirt is only Rs.12345.", payload
        )
        assert not ok
        assert "12345" in problem

    def test_rejects_links(self, payload):
        ok, _ = validate_explanation("Buy it at https://example.com", payload)
        assert not ok

    def test_rejects_empty_and_overlong_responses(self, payload):
        assert not validate_explanation("", payload)[0]
        assert not validate_explanation("word " * 400, payload)[0]

    def test_prompt_forbids_invention(self):
        for phrase in ("Do not invent", "ONLY the items", "do not add"):
            assert phrase.lower() in EXPLANATION_SYSTEM_PROMPT.lower()

    def test_payload_carries_no_catalog_and_no_ids(self, payload):
        assert "selected_items" in payload
        for item in payload["selected_items"]:
            assert "id" not in item
        # The model receives at most a handful of items, never the catalog.
        assert len(payload["selected_items"]) <= 6

    def test_template_explanation_is_itself_grounded(self, payload):
        text = _template_explanation(payload)
        ok, problem = validate_explanation(text, payload)
        assert ok, problem


def trip_all_providers(error: str = "boom") -> None:
    """Open the breaker on every configured provider.

    Breakers are per provider, so tripping one is no longer enough to disable
    the LLM path - a blocked Gemini project must still fall through to Groq.
    """
    for provider in llm_client.PROVIDER_ORDER:
        for _ in range(llm_client.FAILURE_THRESHOLD):
            llm_client._record_failure(RuntimeError(error), provider)


class TestCircuitBreaker:
    """A dead dependency must cost one failed call, not one per outfit."""

    def setup_method(self):
        llm_client.reset_circuit()

    def teardown_method(self):
        llm_client.reset_circuit()

    def test_starts_closed(self):
        assert not llm_client.circuit_is_open()

    def test_opens_after_every_provider_fails(self):
        trip_all_providers("403 PERMISSION_DENIED")
        assert llm_client.circuit_is_open()
        assert "PERMISSION_DENIED" in llm_client.circuit_error()

    def test_one_dead_provider_does_not_disable_the_others(self, monkeypatch):
        """The situation this project actually hit: Gemini blocked, Groq fine."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
        monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
        for _ in range(llm_client.FAILURE_THRESHOLD):
            llm_client._record_failure(
                RuntimeError("403"), llm_client.PROVIDER_GEMINI
            )
        assert llm_client._circuit_open_for(llm_client.PROVIDER_GEMINI)
        assert not llm_client._circuit_open_for(llm_client.PROVIDER_GROQ)
        assert not llm_client.circuit_is_open(), "Groq is still usable"
        assert llm_client.active_provider() == llm_client.PROVIDER_GROQ

    def test_open_circuit_suppresses_further_calls(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
        monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
        llm_client._client.cache_clear()
        trip_all_providers()
        assert not llm_client.is_available()

        def explode(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("an LLM was called while the circuit was open")

        monkeypatch.setattr(llm_client, "_call_gemini_extraction", explode)
        monkeypatch.setattr(llm_client, "_call_groq_extraction", explode)
        result = extract_preferences("formal outfit for office, 8000", allow_llm=True)
        assert result.source.value == "rule_based"
        assert result.preferences.budget == 8000
        # The user-facing message must be calm and non-technical: no status
        # codes, no provider names, no stack text.
        warning = result.warning or ""
        assert warning == llm_client.FRIENDLY_UNAVAILABLE
        for leak in ("403", "429", "PERMISSION_DENIED", "Traceback", "ClientError"):
            assert leak not in warning

    def test_success_closes_it_again(self):
        llm_client._record_failure(RuntimeError("boom"))
        llm_client._record_success()
        assert not llm_client.circuit_is_open()

    def test_a_configured_key_is_still_reported_separately(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
        monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
        trip_all_providers()
        # The UI needs to distinguish "no key" from "key present but broken".
        assert llm_client.has_api_key()
        assert not llm_client.is_available()
        assert llm_client.active_provider() is None


class TestSecrets:
    def test_error_text_never_leaks_a_key(self):
        leaked = Exception(
            "request failed: https://api.example.com/v1?key=AIzaSyFAKEKEY1234567890abc"
        )
        safe = _safe_error(leaked)
        assert "AIzaSyFAKEKEY1234567890abc" not in safe
        assert "***" in safe

    def test_no_key_is_hardcoded_anywhere_in_the_source(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        pattern = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
        for path in list(root.glob("*.py")) + list((root / "src").glob("*.py")):
            assert not pattern.search(path.read_text(encoding="utf-8")), path
