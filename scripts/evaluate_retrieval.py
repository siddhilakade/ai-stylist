"""Compare TF-IDF against Sentence Transformer + FAISS retrieval.

There are no relevance judgements for this catalog, so Precision@K, Recall@K,
NDCG and MAP can't be computed without inventing the ground truth they need, and
none is reported.

What is measurable: latency, attribute agreement (for queries naming a checkable
attribute like "formal" or "ethnic", what share of the top-K actually carries
it - a weak but real proxy), and overlap between the two rankings.

Run:  python scripts/evaluate_retrieval.py
Out:  docs/retrieval_evaluation.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import semantic_retriever  # noqa: E402
from src.data import get_product  # noqa: E402
from src.similarity import _model as tfidf_model  # noqa: E402

OUT = ROOT / "docs" / "retrieval_evaluation.json"
TOP_K = 10

# Each probe names an attribute we can verify on the returned products. This is
# a proxy for relevance, not relevance itself - it only checks that retrieval
# respects an attribute the query states outright.
PROBES: list[tuple[str, str]] = [
    ("formal outfit for the office", "formal"),
    ("business meeting clothes", "formal"),
    ("traditional Indian ethnic wear for a wedding", "ethnic"),
    ("festive kurta for diwali", "ethnic"),
    ("gym and running clothes", "sports"),
    ("comfortable sportswear for training", "sports"),
    ("black clothing", "black"),
    ("white top", "white"),
    ("blue denim jeans", "blue"),
    ("relaxed casual weekend outfit", "casual"),
]


def attribute_holds(product: dict, attribute: str) -> bool:
    """Whether a product genuinely carries the attribute the query named."""
    if attribute == "formal":
        return product["formality"] >= 2.75
    if attribute == "ethnic":
        return bool(product["is_ethnic"])
    if attribute == "sports":
        return product["usage"] in ("Sports", "Travel")
    if attribute == "casual":
        return product["formality"] <= 1.5
    return attribute.lower() in str(product["baseColour"]).lower()


def tfidf_search(query: str, top_k: int) -> list[int]:
    """Rank the catalog against a free-text query with the existing TF-IDF model."""
    import numpy as np

    similarity, index, products = tfidf_model()
    # `similarity` is product-to-product. To score a free-text query we reuse the
    # same vectoriser by refitting on the corpus and transforming the query.
    from sklearn.feature_extraction.text import TfidfVectorizer

    from src.similarity import _document

    corpus = [_document(p) for p in products]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query.lower()])
    scores = (matrix @ query_vector.T).toarray().ravel()
    order = np.argsort(-scores)[:top_k]
    return [int(products[i]["id"]) for i in order]


def main() -> None:
    if not semantic_retriever.is_available():
        raise SystemExit("Build the index first: python scripts/build_vector_index.py")

    index = semantic_retriever.load_index()
    # Warm both models so the first probe is not paying load cost.
    semantic_retriever.encode(["warmup"])
    tfidf_search("warmup", 1)

    rows = []
    for query, attribute in PROBES:
        started = time.perf_counter()
        semantic_ids = [pid for pid, _ in index.search(query, top_k=TOP_K)]
        semantic_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        tfidf_ids = tfidf_search(query, TOP_K)
        tfidf_ms = (time.perf_counter() - started) * 1000

        def hit_rate(ids: list[int]) -> float:
            products = [get_product(pid) for pid in ids]
            return sum(attribute_holds(p, attribute) for p in products if p) / max(len(ids), 1)

        overlap = len(set(semantic_ids) & set(tfidf_ids)) / max(len(semantic_ids), 1)
        rows.append({
            "query": query, "attribute": attribute,
            "semantic_hit_rate": round(hit_rate(semantic_ids), 3),
            "tfidf_hit_rate": round(hit_rate(tfidf_ids), 3),
            "semantic_ms": round(semantic_ms, 2),
            "tfidf_ms": round(tfidf_ms, 2),
            "overlap": round(overlap, 3),
        })

    semantic_mean = statistics.mean(r["semantic_hit_rate"] for r in rows)
    tfidf_mean = statistics.mean(r["tfidf_hit_rate"] for r in rows)
    summary = {
        "top_k": TOP_K,
        "probes": len(rows),
        "semantic_attribute_hit_rate": round(semantic_mean, 3),
        "tfidf_attribute_hit_rate": round(tfidf_mean, 3),
        "semantic_latency_ms_median": round(
            statistics.median(r["semantic_ms"] for r in rows), 2),
        "tfidf_latency_ms_median": round(
            statistics.median(r["tfidf_ms"] for r in rows), 2),
        "mean_overlap": round(statistics.mean(r["overlap"] for r in rows), 3),
        "model": semantic_retriever.MODEL_NAME,
        "caveat": "Attribute hit-rate is a PROXY for relevance. There are no "
                  "relevance judgements for this catalog, so Precision@K, "
                  "Recall@K, NDCG and MAP are not computable and are not reported.",
    }

    header = f"{'query':<44}{'attr':<9}{'semantic':>10}{'tfidf':>8}{'overlap':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['query']:<44}{row['attribute']:<9}"
              f"{row['semantic_hit_rate']:>10.2f}{row['tfidf_hit_rate']:>8.2f}"
              f"{row['overlap']:>9.2f}")

    print("\n" + "=" * 64)
    print(f"attribute hit-rate   semantic {semantic_mean:.3f}   "
          f"tfidf {tfidf_mean:.3f}")
    print(f"median latency       semantic "
          f"{summary['semantic_latency_ms_median']:.1f} ms   "
          f"tfidf {summary['tfidf_latency_ms_median']:.1f} ms")
    print(f"mean overlap@{TOP_K}      {summary['mean_overlap']:.3f}")
    print("=" * 64)
    winner = ("semantic" if semantic_mean > tfidf_mean
              else "tfidf" if tfidf_mean > semantic_mean else "tie")
    print(f"Higher attribute agreement: {winner}")

    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
