"""Content-based "similar products" via TF-IDF over product metadata.

Deliberately NOT part of the outfit pipeline. It answers a different question:

    similarity    "what else looks like this?"   -> more of the same
    compatibility "what goes WITH this?"         -> the outfit engine

Conflating the two is the classic failure of fashion recommenders: cosine
similarity on a white shirt returns twelve more white shirts, which is exactly
what someone building an outfit doesn't want. Keeping them separate and clearly
labelled is the point.

Fitted on the deployment catalog and cached for the process.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data import Product, catalog_records


def _document(product: Product) -> str:
    """The text used to represent one product.

    Structured attributes are repeated deliberately: they are more reliable than
    the free-text display name, so weighting them up produces better neighbours.
    """
    return " ".join(
        str(value)
        for value in (
            product["productDisplayName"],
            product["articleType"], product["articleType"],
            product["baseColour"], product["baseColour"],
            product["color_family"],
            product["usage"],
            product["gender"],
            product["outfit_slot"],
            product["formality_tier"],
        )
    ).lower()


@lru_cache(maxsize=1)
def _model() -> tuple[np.ndarray, dict[int, int], tuple[Product, ...]]:
    """Fit TF-IDF once and return (similarity matrix, id->row index, products)."""
    products = catalog_records()
    corpus = [_document(p) for p in products]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # bigrams catch "navy blue", "formal shoes"
        min_df=1,
        sublinear_tf=True,    # dampens the repeated-attribute trick above
    )
    matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(matrix)
    index = {int(p["id"]): i for i, p in enumerate(products)}
    return similarity, index, products


def similar_products(product_id: int, top_k: int = 6, same_slot: bool = True) -> list[Product]:
    """Nearest neighbours by metadata similarity, excluding the product itself."""
    similarity, index, products = _model()
    row = index.get(int(product_id))
    if row is None:
        return []

    scores = similarity[row]
    order = np.argsort(-scores)
    anchor_slot = products[row]["outfit_slot"]

    results: list[Product] = []
    for candidate_row in order:
        if candidate_row == row:
            continue
        candidate = products[candidate_row]
        if same_slot and candidate["outfit_slot"] != anchor_slot:
            continue
        results.append(candidate)
        if len(results) >= top_k:
            break
    return results


def similarity_between(product_id_a: int, product_id_b: int) -> float:
    """Cosine similarity between two catalog products, in [0, 1]."""
    similarity, index, _ = _model()
    row_a, row_b = index.get(int(product_id_a)), index.get(int(product_id_b))
    if row_a is None or row_b is None:
        return 0.0
    return float(similarity[row_a][row_b])
