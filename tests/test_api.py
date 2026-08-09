"""FastAPI layer.

The API must be a thin adapter: same engine, same guarantees. These tests check
the routes work AND that the constraints enforced in the Python API survive the
HTTP boundary.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import app
from src.data import catalog_records

client = TestClient(app)


class TestHealth:
    def test_reports_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["catalog"]["products"] > 0

    def test_reports_subsystem_state_honestly(self):
        body = client.get("/health").json()
        assert "semantic_index" in body
        assert "llm" in body
        # The ML experiment must never claim to be in the production path.
        assert body["ml_experiment"]["in_production_path"] is False


class TestRecommend:
    def test_form_style_request(self):
        response = client.post("/recommend", json={
            "gender": "Men", "occasion": "work_office", "style": "formal",
            "budget": 8000, "use_llm": False,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["outfits"]
        for outfit in body["outfits"]:
            assert outfit["total_price"] <= 8000
            assert outfit["items"]

    def test_natural_language_request(self):
        response = client.post("/recommend", json={
            "query": "black shirt for men under 4000", "use_llm": False,
        })
        body = response.json()
        assert response.status_code == 200
        assert body["understood"]["source"] in ("rule_based", "gemini")
        if body["ok"]:
            for outfit in body["outfits"]:
                tops = [i for i in outfit["items"] if i["slot"] == "top"]
                assert tops and tops[0]["article_type"] == "Shirts"
                assert tops[0]["colour"] == "Black"

    def test_budget_is_never_exceeded(self):
        for budget in (2000, 5000):
            body = client.post("/recommend", json={
                "gender": "Women", "occasion": "college",
                "budget": budget, "use_llm": False,
            }).json()
            for outfit in body["outfits"]:
                assert outfit["total_price"] <= budget

    def test_impossible_request_fails_gracefully(self):
        body = client.post("/recommend", json={
            "gender": "Men", "occasion": "work_office", "style": "formal",
            "budget": 400, "use_llm": False,
        }).json()
        assert body["ok"] is False
        assert body["outfits"] == []
        assert body["failure"]["suggestion"]

    def test_every_returned_product_is_real(self):
        from src.data import product_ids

        valid = product_ids()
        body = client.post("/recommend", json={
            "gender": "Women", "occasion": "party", "budget": 7000,
            "use_llm": False,
        }).json()
        for outfit in body["outfits"]:
            for item in outfit["items"]:
                assert item["id"] in valid

    def test_response_carries_reasons_and_scores(self):
        body = client.post("/recommend", json={
            "gender": "Men", "occasion": "college", "budget": 5000,
            "use_llm": False,
        }).json()
        outfit = body["outfits"][0]
        assert outfit["reasons"]
        assert outfit["score_breakdown"]
        assert 0.0 <= outfit["compatibility"] <= 1.0

    def test_validates_input(self):
        assert client.post("/recommend", json={"num_outfits": 99}).status_code == 422
        assert client.post("/recommend", json={"budget": -5}).status_code == 422


class TestCompleteTheLook:
    def test_anchor_is_pinned(self):
        product = next(p for p in catalog_records() if p["outfit_slot"] == "top")
        body = client.get(
            f"/products/{product['id']}/complete-the-look?budget=9000"
        ).json()
        assert body["ok"] is True
        for outfit in body["outfits"]:
            assert any(i["id"] == int(product["id"]) for i in outfit["items"])

    def test_unknown_product_is_404(self):
        assert client.get("/products/999999999/complete-the-look").status_code == 404


class TestCatalogAndSearch:
    def test_product_search(self):
        body = client.get("/products?q=shirt&gender=Men&limit=5").json()
        assert body["count"] <= 5

    def test_semantic_search(self):
        from src import semantic_retriever

        response = client.get("/search/semantic?q=something%20formal&limit=5")
        if not semantic_retriever.is_available():
            assert response.status_code == 503
            return
        body = response.json()
        assert response.status_code == 200
        assert 0 < len(body["results"]) <= 5
        scores = [r["score"] for r in body["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_catalog_summary(self):
        body = client.get("/catalog/summary").json()
        assert body["products"] > 0


class TestFeedback:
    def test_records_a_signal(self):
        product = next(iter(catalog_records()))
        body = client.post("/feedback", json={
            "outfit_product_ids": [int(product["id"])], "signal": "like",
        }).json()
        assert body["recorded"] is True
        # It must state plainly that it is not personalisation.
        assert "not used for ranking" in body["note"].lower()
        assert "no personalisation" in body["note"].lower()

    def test_rejects_unknown_products(self):
        response = client.post("/feedback", json={
            "outfit_product_ids": [999999999], "signal": "like",
        })
        assert response.status_code == 400

    def test_rejects_invalid_signal(self):
        response = client.post("/feedback", json={
            "outfit_product_ids": [], "signal": "sideways",
        })
        assert response.status_code == 422
