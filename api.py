"""FastAPI service layer over the existing recommender.

This is a THIN adapter, deliberately. It contains no business logic: every route
translates JSON into the same `StylePreferences` the Streamlit UI builds, calls
the same `recommend_outfits` / `complete_the_look`, and serialises the result.
If a rule changes, it changes in `src/` and both interfaces get it for free.

    FastAPI  ─┐
              ├─► src/recommender.py ─► semantic retrieval ─► compatibility
    Streamlit ┘                      ─► outfit builder (budget lookahead)

Run:  uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs   (OpenAPI, usable from Postman)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent / ".env", override=False)

from src import llm_client, semantic_retriever  # noqa: E402
from src.data import catalog_summary, get_product, load_catalog, search_products  # noqa: E402
from src.ml import model as ml_model  # noqa: E402
from src.recommender import (  # noqa: E402
    RecommendationResult,
    complete_the_look,
    grounding_payload,
    outfit_reasons,
    recommend_outfits,
    score_breakdown,
)
from src.schemas import StylePreferences  # noqa: E402

app = FastAPI(
    title="AI Stylist API",
    version="1.0.0",
    description=(
        "Outfit recommendation over a fashion catalog. An LLM turns free text "
        "into structured preferences; a Sentence Transformer + FAISS narrows "
        "candidates; deterministic rules enforce constraints, budget and "
        "assembly. The LLM never selects products."
    ),
)


# --------------------------------------------------------------------------
# Wire models
# --------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Either free text, or explicit preferences, or both."""

    query: Optional[str] = Field(
        default=None,
        description="Natural-language request, e.g. 'black shirt for a college "
                    "presentation under 3000'. Parsed by the LLM when available.",
        examples=["smart casual outfit for an interview, under 5000"],
    )
    gender: Optional[Literal["Men", "Women", "Unisex"]] = None
    occasion: Optional[str] = None
    style: Optional[str] = None
    budget: Optional[int] = Field(default=None, ge=0)
    preferred_colors: list[str] = Field(default_factory=list)
    include_accessory: bool = True
    num_outfits: int = Field(default=3, ge=1, le=5)
    use_llm: bool = True


class ItemOut(BaseModel):
    id: int
    slot: str
    brand: str
    name: str
    article_type: str
    colour: str
    formality_tier: str
    price: int
    image_url: str


class OutfitOut(BaseModel):
    template: str
    items: list[ItemOut]
    total_price: int
    compatibility: float
    preference_match: float
    budget_fit: float
    final_score: float
    signals: dict[str, float]
    reasons: list[str]
    score_breakdown: list[dict[str, Any]]


class RecommendResponse(BaseModel):
    ok: bool
    outfits: list[OutfitOut]
    understood: dict[str, Any]
    failure: Optional[dict[str, Any]] = None
    diagnostics: dict[str, Any]
    latency_ms: float


class FeedbackRequest(BaseModel):
    """Feedback capture. Stored only in memory, and never used for ranking."""

    outfit_product_ids: list[int]
    signal: Literal["like", "dislike", "try_another"]
    query: Optional[str] = None


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def _item_out(slot: str, item: dict) -> ItemOut:
    return ItemOut(
        id=int(item["id"]), slot=slot,
        brand=str(item.get("brand", "")),
        name=str(item["productDisplayName"]),
        article_type=str(item["articleType"]),
        colour=str(item["baseColour"]),
        formality_tier=str(item["formality_tier"]),
        price=int(item["price"]),
        image_url=f"/static/products/{item['image_file']}",
    )


def _outfit_out(outfit, prefs: StylePreferences) -> OutfitOut:
    return OutfitOut(
        template=outfit.template,
        items=[_item_out(slot, item) for slot, item in outfit.ordered_items],
        total_price=outfit.total_price,
        compatibility=outfit.compatibility,
        preference_match=outfit.preference_match,
        budget_fit=outfit.budget_fit,
        final_score=outfit.final_score,
        signals=outfit.signals,
        reasons=outfit_reasons(outfit, prefs),
        score_breakdown=score_breakdown(outfit),
    )


def _response(result: RecommendationResult, prefs: StylePreferences,
              understood: dict[str, Any]) -> RecommendResponse:
    return RecommendResponse(
        ok=result.ok,
        outfits=[_outfit_out(o, prefs) for o in result.outfits],
        understood=understood,
        failure=None if result.ok else {
            "reason_code": result.failure.reason_code,
            "message": result.failure.message,
            "suggestion": result.failure.suggestion,
            "detail": result.failure.detail,
        },
        diagnostics=result.diagnostics,
        latency_ms=result.latency_ms,
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    """Liveness plus the state of every optional subsystem.

    Deliberately reports degraded components rather than failing: the app is
    designed to work without an LLM and without the vector index, and the health
    check should make that visible instead of hiding it.
    """
    try:
        catalog_rows = len(load_catalog())
        catalog_ok = catalog_rows > 0
    except Exception as exc:
        return {"status": "unhealthy", "catalog": {"ok": False, "error": str(exc)}}

    return {
        "status": "ok" if catalog_ok else "unhealthy",
        "catalog": {"ok": catalog_ok, "products": catalog_rows},
        "semantic_index": {
            "available": semantic_retriever.is_available(),
            "model": semantic_retriever.MODEL_NAME,
        },
        "llm": {
            "configured_providers": llm_client.configured_providers(),
            "active_provider": llm_client.active_provider(),
            "degraded": llm_client.circuit_is_open(),
        },
        "ml_experiment": {
            "model_present": ml_model.model_exists(),
            "in_production_path": ml_model.IS_IN_PRODUCTION_PATH,
        },
    }


@app.post("/recommend", response_model=RecommendResponse, tags=["recommendation"])
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Free text and/or explicit preferences -> complete, budget-valid outfits."""
    started = time.perf_counter()

    if request.query:
        extraction = llm_client.extract_preferences(
            request.query,
            default_gender=request.gender or "Women",
            allow_llm=request.use_llm,
        )
        prefs = extraction.preferences
        understood = {
            "source": extraction.source.value,
            "provider": extraction.provider,
            "model": extraction.model,
            "warning": extraction.warning,
        }
        # Explicit fields override anything the LLM inferred.
        if request.gender:
            prefs.gender = request.gender
        if request.budget is not None:
            prefs.budget = request.budget
    else:
        prefs = StylePreferences(
            gender=request.gender or "Women",
            occasion=request.occasion or "everyday_casual",
            style=request.style,
            budget=request.budget,
            preferred_colors=request.preferred_colors,
            include_accessory=request.include_accessory,
        )
        understood = {"source": "form", "provider": None, "model": None,
                      "warning": None}

    understood["preferences"] = prefs.model_dump()
    result = recommend_outfits(prefs, num_outfits=request.num_outfits)
    response = _response(result, prefs, understood)
    response.latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return response


@app.get("/products/{product_id}/complete-the-look", response_model=RecommendResponse,
         tags=["recommendation"])
def complete_look(product_id: int, budget: int = Query(default=7000, ge=500)) -> RecommendResponse:
    """Build outfits around a product, which is pinned and cannot be swapped."""
    product = get_product(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=f"No product {product_id}")

    result = complete_the_look(product_id, budget=budget)
    prefs = StylePreferences(
        gender=product["gender"] if product["gender"] != "Unisex" else "Women",
        budget=budget,
    )
    return _response(result, prefs, {"source": "anchor_product",
                                     "anchor_id": product_id})


@app.get("/products", tags=["catalog"])
def products(q: str = "", gender: Optional[str] = None, slot: Optional[str] = None,
             max_price: Optional[int] = None, limit: int = Query(24, ge=1, le=100)
             ) -> dict[str, Any]:
    """Catalog search - the same substring/facet search the UI uses."""
    found = search_products(query=q, gender=gender, slot=slot,
                            max_price=max_price, limit=limit)
    return {
        "count": len(found),
        "products": [_item_out(p["outfit_slot"], p).model_dump() for p in found],
    }


@app.get("/search/semantic", tags=["catalog"])
def semantic_search(q: str = Query(..., min_length=2),
                    limit: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
    """Vector search over product embeddings.

    Exposed on its own so the retrieval layer can be inspected directly, rather
    than only observed through its effect on recommendations.
    """
    if not semantic_retriever.is_available():
        raise HTTPException(
            status_code=503,
            detail="Semantic index unavailable. Run scripts/build_vector_index.py.",
        )
    started = time.perf_counter()
    hits = semantic_retriever.load_index().search(q, top_k=limit)
    return {
        "query": q,
        "model": semantic_retriever.MODEL_NAME,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "results": [
            {"score": round(score, 4),
             **_item_out(get_product(pid)["outfit_slot"], get_product(pid)).model_dump()}
            for pid, score in hits
        ],
    }


# In-memory only, and cleared on restart. This is a demonstration of where a
# learning loop WOULD attach - not a personalisation feature. Nothing here
# influences ranking, and there are no users.
_feedback_log: list[dict[str, Any]] = []


@app.post("/feedback", tags=["feedback"])
def feedback(request: FeedbackRequest) -> dict[str, Any]:
    """Record a like / dislike / try-another signal.

    Stored in memory and used for nothing. It exists to show the shape of the
    data a future learned ranker would need - which is precisely what this
    dataset lacks. No claim of personalisation is made anywhere.
    """
    unknown = [pid for pid in request.outfit_product_ids if get_product(pid) is None]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown product ids: {unknown}")

    _feedback_log.append({
        "timestamp": time.time(),
        "signal": request.signal,
        "product_ids": request.outfit_product_ids,
        "query": request.query,
    })
    return {
        "recorded": True,
        "total_events": len(_feedback_log),
        "note": "In-memory only. Not used for ranking; no personalisation is "
                "claimed. Demonstrates where a future learned ranker would "
                "attach once real interaction data exists.",
    }


@app.get("/catalog/summary", tags=["catalog"])
def summary() -> dict[str, Any]:
    return catalog_summary()
