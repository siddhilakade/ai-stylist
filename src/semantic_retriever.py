"""Semantic candidate retrieval with Sentence Transformer embeddings + FAISS.

Runs after hard filtering, so it only ever narrows an already-legal pool - it
can reorder and shorten, never add a product back or overrule a constraint.

Its job is the part the rule engine can't do: act on free text like "nothing too
flashy", which has no structured field to live in.

Retrieval is per outfit slot (a global top-K could empty a slot), and the
cheapest item in each slot always survives (otherwise narrowing could break a
budget that was satisfiable). IndexFlatIP over normalised vectors is exact, so
the same query always gives the same ranking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.features import OCCASION_PROFILES, SLOT_LABELS

Product = Mapping[str, Any]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "models" / "semantic"
INDEX_PATH = INDEX_DIR / "products.faiss"
META_PATH = INDEX_DIR / "index_meta.json"

# Small, fast, widely used, and good enough for short product descriptions.
# 384 dimensions and ~90 MB, so it loads in about a second and runs on CPU.
# Deliberately NOT fine-tuned: there is no compatibility ground truth to tune on,
# and a pretrained general-purpose encoder is the honest choice here.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# How many candidates to keep per slot. Large enough that the outfit builder
# still has real choice (it needs alternatives to satisfy budget lookahead and
# to produce diverse looks), small enough that retrieval is doing something.
DEFAULT_TOP_K_PER_SLOT = 40


class IndexUnavailable(RuntimeError):
    """The FAISS index or the encoder is missing or unreadable."""


# --------------------------------------------------------------------------
# Text construction
# --------------------------------------------------------------------------

# Article types are stored as plurals ("Shirts", "Backpacks"). Embedding
# "a navy blue backpacks for unisex" is measurably worse than the grammatical
# form, because the encoder was trained on well-formed English.
_ALWAYS_PLURAL = {
    "jeans", "trousers", "shorts", "shoes", "flats", "heels", "sandals",
    "flip flops", "track pants", "leggings", "capris", "sunglasses",
    "casual shoes", "formal shoes", "sports shoes", "sports sandals",
}


def _singular(article_type: str) -> str:
    lowered = article_type.lower()
    if lowered in _ALWAYS_PLURAL or not lowered.endswith("s"):
        return lowered
    return lowered[:-1]


def product_document(product: Product) -> str:
    """The sentence describing a product that gets embedded.

    Written as natural language rather than a bag of fields, because that is the
    distribution the encoder was trained on. "A white formal shirt for men"
    embeds far more usefully than "white|shirt|formal|men".
    """
    tier = str(product.get("formality_tier", "")).lower()
    usage = str(product.get("usage", "")).lower()
    colour = str(product.get("baseColour", "")).lower()
    article = _singular(str(product.get("articleType", "")))
    gender = str(product.get("gender", "")).lower()
    name = str(product.get("productDisplayName", ""))

    parts = [f"A {colour} {article} for {gender}."]
    if tier:
        parts.append(f"A {tier} piece.")
    if usage:
        parts.append(f"Worn for {usage} occasions.")
    if product.get("is_ethnic"):
        parts.append("Traditional Indian ethnic wear, suitable for festivals and weddings.")
    if product.get("is_neutral"):
        parts.append("A neutral shade that coordinates easily.")
    parts.append(name)
    return " ".join(parts)


def preference_query(prefs: Any) -> str:
    """Turn StylePreferences into the sentence we search with.

    `notes` is included deliberately and is the main reason this layer exists:
    it is free text the deterministic engine has no way to act on.
    """
    profile = OCCASION_PROFILES.get(getattr(prefs, "occasion", None))
    occasion = profile["label"].lower() if profile else "everyday"
    gender = getattr(prefs, "gender", "") or ""

    parts = [f"An outfit for {gender.lower()} for {occasion}."]

    style = getattr(prefs, "style", None)
    if style:
        parts.append(f"{style.replace('_', ' ').capitalize()} style.")

    colours = list(getattr(prefs, "preferred_colors", []) or [])
    if colours:
        parts.append(f"In {', '.join(colours)} colours.")

    for item in getattr(prefs, "required_items", []) or []:
        parts.append(f"Includes a {item.describe()}.")

    notes = getattr(prefs, "notes", None)
    if notes:
        # The whole point: an unstructured wish the rule engine cannot use.
        parts.append(str(notes))

    return " ".join(parts)


def slot_query(base_query: str, slot: str) -> str:
    """Specialise the request for one outfit slot."""
    label = SLOT_LABELS.get(slot, slot)
    return f"{base_query} Looking for {label.lower()}."


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------

@dataclass
class SemanticIndex:
    """A loaded FAISS index plus the row -> product id mapping."""

    index: Any
    product_ids: list[int]
    model_name: str
    dimension: int

    @property
    def size(self) -> int:
        return len(self.product_ids)

    def _row_of(self) -> dict[int, int]:
        return {pid: row for row, pid in enumerate(self.product_ids)}

    def search(
        self,
        query: str,
        allowed_ids: Iterable[int] | None = None,
        top_k: int = DEFAULT_TOP_K_PER_SLOT,
    ) -> list[tuple[int, float]]:
        """Rank products by similarity, optionally restricted to `allowed_ids`."""
        return self.search_with_vector(encode([query]), allowed_ids, top_k)

    def search_with_vector(
        self,
        vector: Any,
        allowed_ids: Iterable[int] | None = None,
        top_k: int = DEFAULT_TOP_K_PER_SLOT,
    ) -> list[tuple[int, float]]:
        """Rank against an already-encoded query.

        Split out from `search` so a whole outfit's worth of slot queries can be
        encoded in ONE batched forward pass. Encoding is by far the dominant
        cost here - six separate calls made recommendations ~16x slower than the
        deterministic path, one batched call makes it negligible.

        The `allowed_ids` restriction is what makes this safe to place after hard
        filtering: the result is always a subset of what the constraints already
        permitted.
        """
        import numpy as np

        if allowed_ids is None:
            scores, rows = self.index.search(vector, min(top_k, self.size))
            return [
                (self.product_ids[row], float(score))
                for row, score in zip(rows[0], scores[0]) if row != -1
            ]

        allowed = [pid for pid in dict.fromkeys(allowed_ids)]
        if not allowed:
            return []

        # Exact scoring over just the permitted rows. For a few hundred allowed
        # products this is a single small matrix product - faster than asking
        # FAISS for a large top-N and then discarding most of it, and it can
        # never silently drop a permitted product the way an over-small N would.
        row_of = self._row_of()
        rows = [row_of[pid] for pid in allowed if pid in row_of]
        if not rows:
            return []

        vectors = self.index.reconstruct_batch(np.asarray(rows, dtype="int64"))
        similarities = (vectors @ vector[0]).astype(float)

        order = np.argsort(-similarities)[:top_k]
        return [(allowed[int(i)], float(similarities[int(i)])) for i in order]


@lru_cache(maxsize=1)
def load_encoder() -> Any:
    """Load the Sentence Transformer once per process."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise IndexUnavailable(
            "sentence-transformers is not installed. "
            "pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(MODEL_NAME)


def encode(texts: Sequence[str]) -> Any:
    """Embed texts into L2-normalised float32 vectors.

    Normalising means inner product == cosine similarity, which is why the index
    can be a plain `IndexFlatIP`.
    """
    import numpy as np

    vectors = load_encoder().encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def index_exists() -> bool:
    return INDEX_PATH.is_file() and META_PATH.is_file()


@lru_cache(maxsize=1)
def load_index() -> SemanticIndex:
    """Load the persisted FAISS index (cached; never rebuilt at request time)."""
    if not index_exists():
        raise IndexUnavailable(
            f"No semantic index at {INDEX_PATH}. Build it with "
            "`python scripts/build_vector_index.py`."
        )
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover
        raise IndexUnavailable("faiss-cpu is not installed.") from exc

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_PATH))

    if index.ntotal != len(meta["product_ids"]):
        raise IndexUnavailable(
            "Index and id mapping disagree on size - rebuild the index."
        )
    if meta.get("model_name") != MODEL_NAME:
        # A different encoder produces a different vector space. Silently mixing
        # them would give confident nonsense.
        raise IndexUnavailable(
            f"Index was built with {meta.get('model_name')}, code expects "
            f"{MODEL_NAME}. Rebuild the index."
        )

    return SemanticIndex(
        index=index,
        product_ids=[int(pid) for pid in meta["product_ids"]],
        model_name=meta["model_name"],
        dimension=int(meta["dimension"]),
    )


def is_available() -> bool:
    """Whether semantic retrieval can be attempted at all."""
    if not index_exists():
        return False
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------
# The function the recommender calls
# --------------------------------------------------------------------------

def rerank_candidates(
    products: Sequence[Product],
    prefs: Any,
    top_k_per_slot: int = DEFAULT_TOP_K_PER_SLOT,
) -> tuple[list[Product], dict[str, Any]]:
    """Narrow an already-hard-filtered pool using semantic similarity.

    Returns (kept products, diagnostics). On any failure it returns the input
    unchanged - a missing index must degrade the system to its previous
    behaviour, never break a recommendation.
    """
    diagnostics: dict[str, Any] = {"applied": False}
    if not products:
        return list(products), diagnostics

    try:
        index = load_index()
    except IndexUnavailable as exc:
        diagnostics["reason"] = str(exc)
        return list(products), diagnostics

    base_query = preference_query(prefs)
    by_slot: dict[str, list[Product]] = {}
    for product in products:
        by_slot.setdefault(str(product["outfit_slot"]), []).append(product)

    kept: list[Product] = []
    per_slot: dict[str, int] = {}

    # Slots small enough to keep whole are passed through untouched - there is
    # nothing to narrow, and reordering them would only cost time since the
    # outfit builder scores candidates itself anyway.
    needs_narrowing = [
        slot for slot, items in by_slot.items() if len(items) > top_k_per_slot
    ]
    for slot, slot_products in by_slot.items():
        if slot not in needs_narrowing:
            kept.extend(slot_products)
            per_slot[slot] = len(slot_products)

    if needs_narrowing:
        # One batched forward pass for every slot query.
        vectors = encode([slot_query(base_query, slot) for slot in needs_narrowing])
        for slot, vector in zip(needs_narrowing, vectors):
            slot_products = by_slot[slot]
            ranked = index.search_with_vector(
                vector.reshape(1, -1),
                allowed_ids=[int(p["id"]) for p in slot_products],
                top_k=top_k_per_slot,
            )
            chosen = {pid for pid, _ in ranked}
            # ALWAYS keep the cheapest item in the slot, whatever the encoder
            # thinks of it. Otherwise semantic narrowing can raise the price
            # floor of the whole outfit and turn a budget that was satisfiable
            # into a failure - a retrieval heuristic silently overruling a hard
            # constraint, which is exactly what this layer must never do.
            cheapest = min(slot_products, key=lambda p: int(p["price"]))
            chosen.add(int(cheapest["id"]))
            kept.extend([p for p in slot_products if int(p["id"]) in chosen])
            per_slot[slot] = len(chosen)

    diagnostics.update({
        "applied": True,
        "model": index.model_name,
        "query": base_query,
        "pool_before": len(products),
        "pool_after": len(kept),
        "per_slot": per_slot,
        "top_k_per_slot": top_k_per_slot,
    })
    return kept, diagnostics


def similar_products_semantic(product_id: int, top_k: int = 6) -> list[int]:
    """Nearest neighbours of one product - the semantic counterpart to TF-IDF."""
    index = load_index()
    from src.data import get_product

    product = get_product(product_id)
    if product is None:
        return []
    results = index.search(product_document(product), top_k=top_k + 1)
    return [pid for pid, _ in results if pid != int(product_id)][:top_k]
