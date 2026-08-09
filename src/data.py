"""Catalog loading and lookup.

The catalog is a plain CSV produced by `scripts/build_catalog.py`, with product
images stored next to it on disk. No database, no vector store, no network call
at runtime: 520 rows fit comfortably in memory, and every extra moving part
would be one more thing to explain and one more thing to fail in deployment.

All paths are resolved relative to this file, so the app runs identically from
any working directory and on any host.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.csv"

# Images live under static/ because Streamlit serves that directory over HTTP
# (see .streamlit/config.toml). The browser then fetches and caches each image
# once, instead of us inlining megabytes of base64 into every page render.
IMAGE_DIR = PROJECT_ROOT / "static" / "products"
IMAGE_URL_PREFIX = "app/static/products"
PLACEHOLDER_IMAGE = PROJECT_ROOT / "static" / "placeholder.png"

Product = dict[str, Any]

# Columns the rest of the code relies on. Load-time validation catches a stale or
# hand-edited CSV immediately instead of at the first KeyError deep in scoring.
REQUIRED_COLUMNS = (
    "id", "productDisplayName", "brand", "gender", "articleType", "baseColour",
    "usage", "outfit_slot", "formality", "formality_tier", "color_family",
    "is_neutral", "is_ethnic", "price", "mrp", "discount_pct", "image_file",
)


class CatalogError(RuntimeError):
    """Raised when the catalog is missing or malformed."""


@lru_cache(maxsize=1)
def load_catalog() -> pd.DataFrame:
    """Load and validate the deployment catalog (cached for the process)."""
    if not CATALOG_PATH.exists():
        raise CatalogError(
            f"Catalog not found at {CATALOG_PATH}. "
            "Run `python scripts/build_catalog.py` to generate it."
        )

    df = pd.read_csv(CATALOG_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CatalogError(f"Catalog is missing required columns: {missing}")

    df["id"] = df["id"].astype(int)
    df["price"] = df["price"].astype(int)
    df["mrp"] = df["mrp"].astype(int)
    df["discount_pct"] = df["discount_pct"].astype(int)
    df["brand"] = df["brand"].astype(str)
    df["formality"] = df["formality"].astype(float)
    df["is_neutral"] = df["is_neutral"].astype(bool)
    df["is_ethnic"] = df["is_ethnic"].astype(bool)
    df["productDisplayName"] = df["productDisplayName"].astype(str)

    if df["id"].duplicated().any():
        raise CatalogError("Catalog contains duplicate product ids.")

    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def catalog_records() -> tuple[Product, ...]:
    """Row dicts. Scoring runs over plain dicts - far faster than pandas rows."""
    return tuple(load_catalog().to_dict("records"))


@lru_cache(maxsize=1)
def _index_by_id() -> dict[int, Product]:
    return {int(p["id"]): p for p in catalog_records()}


def get_product(product_id: int | str) -> Product | None:
    """Look up one product. Returns None for anything not in the catalog.

    This is the single resolution point for product ids, and it is what makes
    hallucinated products impossible: nothing reaches the UI unless it came out
    of this dict.
    """
    try:
        key = int(product_id)
    except (TypeError, ValueError):
        return None
    return _index_by_id().get(key)


def get_products(product_ids: Iterable[int | str]) -> list[Product]:
    """Resolve many ids, silently dropping any that do not exist."""
    resolved = (get_product(pid) for pid in product_ids)
    return [p for p in resolved if p is not None]


def image_path(product: Product) -> Path:
    """On-disk path of a product image, falling back to a placeholder."""
    candidate = IMAGE_DIR / str(product.get("image_file", ""))
    if candidate.exists():
        return candidate
    return PLACEHOLDER_IMAGE


def image_url(product: Product) -> str:
    """URL the browser uses to fetch a product image."""
    return f"{IMAGE_URL_PREFIX}/{product.get('image_file', '')}"


def product_ids() -> set[int]:
    return set(_index_by_id())


# Display order for browse pages. Catalog order is by product id, which puts
# whatever happens to have a low id first - in practice bags and backpacks. A
# shopper landing on a fashion catalog expects garments first, so listings are
# ordered by slot and then by discount. This is presentation only; it has no
# effect on the recommendation pipeline, which never uses this function.
SLOT_DISPLAY_ORDER = {
    "top": 0, "onepiece": 1, "bottom": 2, "outerwear": 3,
    "footwear": 4, "accessory": 5,
}


def _display_sort_key(product: Product) -> tuple[int, int, int]:
    return (
        SLOT_DISPLAY_ORDER.get(product["outfit_slot"], 9),
        -int(product.get("discount_pct", 0)),
        int(product["id"]),
    )


def search_products(
    query: str = "",
    gender: str | None = None,
    slot: str | None = None,
    max_price: int | None = None,
    limit: int = 60,
) -> list[Product]:
    """Simple substring + facet search, used by the browse/search UI.

    Deliberately not fuzzy or semantic: this is a catalog browser, not the
    recommendation path, and a predictable substring match is easier for an
    evaluator to reason about.
    """
    results = list(catalog_records())

    if gender and gender != "All":
        # Unisex items are shown to everyone; that is what the label means.
        results = [p for p in results if p["gender"] in (gender, "Unisex")]
    if slot and slot != "All":
        results = [p for p in results if p["outfit_slot"] == slot]
    if max_price is not None:
        results = [p for p in results if p["price"] <= max_price]

    tokens = [t for t in query.lower().split() if t]
    if tokens:
        def matches(product: Product) -> bool:
            haystack = " ".join(
                str(product[field]).lower()
                for field in ("productDisplayName", "brand", "articleType",
                              "baseColour", "usage")
            )
            return all(token in haystack for token in tokens)

        results = [p for p in results if matches(p)]

    results.sort(key=_display_sort_key)
    return results[:limit]


def catalog_summary() -> dict[str, Any]:
    """Headline counts, shown in the UI's 'about the data' panel."""
    df = load_catalog()
    return {
        "products": int(len(df)),
        "article_types": int(df["articleType"].nunique()),
        "colors": int(df["baseColour"].nunique()),
        "genders": sorted(df["gender"].unique().tolist()),
        "slots": df["outfit_slot"].value_counts().to_dict(),
        "price_min": int(df["price"].min()),
        "price_max": int(df["price"].max()),
    }
