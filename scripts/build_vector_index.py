"""Build the FAISS index over the product catalog.

Products are embedded in ascending id order and the index is exact
(IndexFlatIP), so there's no training step, no clustering and no random seed -
running this twice produces identical output.

The encoder is pretrained, not fine-tuned: there are no compatibility labels to
fine-tune on, and a general-purpose encoder is the right tool for matching free
text against short product descriptions.

Run:  python scripts/build_vector_index.py
Out:  models/semantic/products.faiss, models/semantic/index_meta.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import catalog_records  # noqa: E402
from src.semantic_retriever import (  # noqa: E402
    EMBEDDING_DIM,
    INDEX_DIR,
    INDEX_PATH,
    META_PATH,
    MODEL_NAME,
    encode,
    product_document,
)


def main() -> None:
    import faiss

    products = sorted(catalog_records(), key=lambda p: int(p["id"]))
    print(f"Catalog: {len(products)} products")
    print(f"Encoder: {MODEL_NAME}")

    documents = [product_document(p) for p in products]
    print("\nExample document:")
    print(f"  {documents[0]}")

    started = time.perf_counter()
    vectors = encode(documents)
    encode_seconds = time.perf_counter() - started
    print(f"\nEncoded {len(vectors)} products in {encode_seconds:.1f}s "
          f"({encode_seconds / len(vectors) * 1000:.1f} ms/product)")

    if vectors.shape[1] != EMBEDDING_DIM:
        raise SystemExit(
            f"Expected {EMBEDDING_DIM} dimensions, got {vectors.shape[1]}. "
            "Update EMBEDDING_DIM in src/semantic_retriever.py."
        )

    # Vectors are L2-normalised by the encoder, so inner product == cosine
    # similarity. Flat = exact search: no approximation and nothing to train.
    # At 536 products an approximate index (IVF/HNSW) would add tuning
    # parameters and recall loss to solve a problem we do not have.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    print(f"Built IndexFlatIP with {index.ntotal} vectors")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps({
        "model_name": MODEL_NAME,
        "dimension": EMBEDDING_DIM,
        "index_type": "IndexFlatIP",
        "normalised": True,
        "product_ids": [int(p["id"]) for p in products],
        "catalog_size": len(products),
        "note": "Row order matches ascending product id. Pretrained encoder, "
                "not fine-tuned. Exact search, no approximation.",
    }, indent=2), encoding="utf-8")

    size_mb = INDEX_PATH.stat().st_size / 1e6
    print(f"\nWrote {INDEX_PATH.relative_to(ROOT)} ({size_mb:.2f} MB)")
    print(f"Wrote {META_PATH.relative_to(ROOT)}")

    # --- smoke check ------------------------------------------------------
    from src.semantic_retriever import load_index

    load_index.cache_clear()
    loaded = load_index()
    print(f"\nReloaded index: {loaded.size} vectors, dim {loaded.dimension}")

    for query in [
        "something understated for a first date",
        "office wear that is not boring",
        "bright festive ethnic outfit",
    ]:
        started = time.perf_counter()
        hits = loaded.search(query, top_k=3)
        ms = (time.perf_counter() - started) * 1000
        from src.data import get_product

        print(f"\n  {query!r}  ({ms:.0f} ms)")
        for pid, score in hits:
            product = get_product(pid)
            print(f"    {score:.3f}  {product['productDisplayName'][:52]}")


if __name__ == "__main__":
    main()
