"""Semantic retrieval: index integrity, and — critically — that it cannot
bypass a single hard constraint.

The whole safety argument for putting a vector search inside a
constraint-satisfaction pipeline is that retrieval runs strictly AFTER hard
filtering and can only ever narrow an already-legal pool. That argument is worth
nothing unless it is enforced by tests, so most of this file is about what
retrieval is *not* allowed to do.
"""

from __future__ import annotations

import pytest

from src import semantic_retriever
from src.data import catalog_records, get_product, product_ids
from src.recommender import hard_filter, recommend_outfits
from src.schemas import StylePreferences

pytestmark = pytest.mark.skipif(
    not semantic_retriever.is_available(),
    reason="semantic index absent; run scripts/build_vector_index.py",
)


@pytest.fixture(scope="module")
def index():
    return semantic_retriever.load_index()


class TestIndexIntegrity:
    def test_index_loads(self, index):
        assert index.size == len(catalog_records())
        assert index.dimension == semantic_retriever.EMBEDDING_DIM

    def test_load_is_cached(self):
        assert semantic_retriever.load_index() is semantic_retriever.load_index()

    def test_every_indexed_id_is_a_real_product(self, index):
        valid = product_ids()
        assert set(index.product_ids) <= valid

    def test_index_covers_the_whole_catalog(self, index):
        assert set(index.product_ids) == product_ids()

    def test_meta_records_the_encoder(self, index):
        assert index.model_name == semantic_retriever.MODEL_NAME

    def test_missing_index_raises_a_useful_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(semantic_retriever, "INDEX_PATH", tmp_path / "nope.faiss")
        semantic_retriever.load_index.cache_clear()
        with pytest.raises(semantic_retriever.IndexUnavailable) as info:
            semantic_retriever.load_index()
        assert "build_vector_index" in str(info.value)
        semantic_retriever.load_index.cache_clear()


class TestSearch:
    def test_returns_scored_product_ids(self, index):
        results = index.search("a formal white shirt for the office", top_k=5)
        assert 0 < len(results) <= 5
        for product_id, score in results:
            assert get_product(product_id) is not None
            assert -1.0 <= score <= 1.0001

    def test_results_are_ordered_by_score(self, index):
        results = index.search("bright festive ethnic wear", top_k=10)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_is_deterministic(self, index):
        first = index.search("smart casual outfit", top_k=8)
        second = index.search("smart casual outfit", top_k=8)
        assert [pid for pid, _ in first] == [pid for pid, _ in second]

    def test_allowed_ids_is_respected(self, index):
        allowed = [int(p["id"]) for p in list(catalog_records())[:25]]
        results = index.search("anything at all", allowed_ids=allowed, top_k=10)
        assert {pid for pid, _ in results} <= set(allowed)

    def test_empty_allowed_set_returns_nothing(self, index):
        assert index.search("shirt", allowed_ids=[], top_k=5) == []

    def test_batched_vector_search_matches_single(self, index):
        query = "a navy blazer for a meeting"
        single = index.search(query, top_k=6)
        vector = semantic_retriever.encode([query])
        batched = index.search_with_vector(vector, top_k=6)
        assert [pid for pid, _ in single] == [pid for pid, _ in batched]


class TestQueryConstruction:
    def test_free_text_notes_reach_the_query(self):
        """`notes` is the reason this layer exists - it must not be dropped."""
        prefs = StylePreferences(
            gender="Women", occasion="date_night", notes="nothing too flashy"
        )
        assert "nothing too flashy" in semantic_retriever.preference_query(prefs)

    def test_query_includes_occasion_and_explicit_items(self):
        prefs = StylePreferences(
            gender="Men", occasion="work_office", style="formal",
            required_items=[{"garment": "shirt", "colour": "white"}],
        )
        query = semantic_retriever.preference_query(prefs)
        assert "work" in query.lower()
        assert "white shirt" in query.lower()

    def test_product_document_is_grammatical(self):
        product = next(
            p for p in catalog_records() if p["articleType"] == "Backpacks"
        )
        document = semantic_retriever.product_document(product)
        assert "backpack for" in document.lower()
        assert "backpacks for" not in document.lower()


class TestCannotBypassHardConstraints:
    """The safety property. Retrieval narrows; constraints decide."""

    def test_reranking_only_ever_returns_a_subset(self):
        prefs = StylePreferences(gender="Men", occasion="work_office", budget=8000)
        allowed = hard_filter(list(catalog_records()), prefs)
        kept, _ = semantic_retriever.rerank_candidates(allowed, prefs)
        allowed_ids = {int(p["id"]) for p in allowed}
        assert {int(p["id"]) for p in kept} <= allowed_ids

    def test_reranking_never_grows_the_pool(self):
        prefs = StylePreferences(gender="Women", occasion="party", budget=6000)
        allowed = hard_filter(list(catalog_records()), prefs)
        kept, _ = semantic_retriever.rerank_candidates(allowed, prefs)
        assert len(kept) <= len(allowed)

    def test_empty_pool_stays_empty(self):
        prefs = StylePreferences(gender="Men", occasion="work_office")
        kept, diagnostics = semantic_retriever.rerank_candidates([], prefs)
        assert kept == []

    def test_gender_constraint_survives_retrieval(self):
        prefs = StylePreferences(gender="Men", occasion="everyday_casual", budget=9000)
        for outfit in recommend_outfits(prefs).outfits:
            for item in outfit.items.values():
                assert item["gender"] in ("Men", "Unisex")

    def test_explicit_colour_and_garment_survive_retrieval(self):
        from src.nlu import parse_request

        prefs = parse_request("black shirt for men", default_gender="Men")
        result = recommend_outfits(prefs)
        for outfit in result.outfits:
            top = outfit.items.get("top")
            assert top is not None
            assert top["articleType"] == "Shirts"
            assert top["baseColour"] == "Black"

    def test_budget_survives_retrieval(self):
        for budget in (2000, 4000, 9000):
            prefs = StylePreferences(gender="Women", occasion="college", budget=budget)
            for outfit in recommend_outfits(prefs).outfits:
                assert outfit.total_price <= budget

    def test_retrieval_never_raises_the_outfit_price_floor(self):
        """Narrowing must not make a satisfiable budget unsatisfiable.

        If the cheapest item in a slot were dropped for being semantically
        unremarkable, retrieval would be overruling the budget constraint - a
        heuristic silently defeating a guarantee.
        """
        from src.recommender import candidates_by_slot, hard_filter

        core = ("top", "bottom", "footwear")
        for gender, occasion in [("Men", "college"), ("Men", "everyday_casual"),
                                 ("Women", "college"), ("Women", "party")]:
            prefs = StylePreferences(gender=gender, occasion=occasion, budget=9000)
            filtered = hard_filter(list(catalog_records()), prefs)
            before = candidates_by_slot(filtered)
            kept, _ = semantic_retriever.rerank_candidates(filtered, prefs)
            after = candidates_by_slot(kept)

            for slot in core:
                if not before[slot]:
                    continue
                assert after[slot], f"{gender}/{occasion}: {slot} was emptied"
                assert min(int(i["price"]) for i in after[slot]) == \
                       min(int(i["price"]) for i in before[slot]), (
                    f"{gender}/{occasion}: cheapest {slot} was dropped"
                )

    def test_impossible_request_still_fails_gracefully(self):
        prefs = StylePreferences(
            gender="Men", occasion="wedding_festive", style="ethnic",
            required_items=[{"garment": "saree", "colour": "purple"}],
        )
        result = recommend_outfits(prefs)
        assert not result.ok
        assert result.outfits == []
        assert result.failure.suggestion

    def test_every_recommended_product_is_still_real(self):
        valid = product_ids()
        prefs = StylePreferences(gender="Women", occasion="wedding_festive",
                                 style="ethnic", budget=8000)
        for outfit in recommend_outfits(prefs).outfits:
            for product_id in outfit.product_ids:
                assert product_id in valid


class TestDegradesGracefully:
    def test_missing_index_leaves_recommendations_working(self, monkeypatch, tmp_path):
        """A missing index must degrade to the previous behaviour, not break."""
        monkeypatch.setattr(semantic_retriever, "INDEX_PATH", tmp_path / "gone.faiss")
        semantic_retriever.load_index.cache_clear()
        try:
            prefs = StylePreferences(gender="Men", occasion="work_office", budget=8000)
            result = recommend_outfits(prefs)
            assert result.ok
            assert result.diagnostics["semantic"]["applied"] is False
            assert "reason" in result.diagnostics["semantic"]
        finally:
            semantic_retriever.load_index.cache_clear()

    def test_diagnostics_report_what_retrieval_did(self):
        prefs = StylePreferences(gender="Women", occasion="everyday_casual")
        result = recommend_outfits(prefs)
        semantic = result.diagnostics["semantic"]
        assert semantic["applied"] is True
        assert semantic["pool_after"] <= semantic["pool_before"]
        assert semantic["model"] == semantic_retriever.MODEL_NAME
        assert semantic["per_slot"]
