"""Inference wrapper for the trained compatibility model.

NOT in the production ranking path. The model was trained and evaluated; it did
not beat the rule baseline on independently labelled pairs (F1 0.545 vs 0.600),
so it was not deployed. See docs/ml_evaluation.md.

It stays loadable and tested so the experiment is reproducible rather than a
claim, and the UI shows its predictions beside the rule scores - labelled as not
affecting ranking. Loading is lazy and cached; the app never trains.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from src.ml.features import FEATURE_COLUMNS, pair_features

Product = Mapping[str, Any]

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "compatibility_model.joblib"

# Stated plainly in one place so the UI and the docs cannot drift from it.
IS_IN_PRODUCTION_PATH = False
EXPERIMENT_SUMMARY = (
    "Random Forest trained on raw product attributes with synthetic "
    "(rule-generated) labels. Generalises to held-out garment types "
    "(ROC-AUC 0.974) but does not beat the rule baseline on independent "
    "author-reviewed pairs, so it does not drive ranking."
)


class ModelUnavailable(RuntimeError):
    """The model artifact is missing or could not be deserialised."""


def model_exists() -> bool:
    return MODEL_PATH.is_file()


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    """Deserialise the trained pipeline bundle (cached for the process)."""
    if not MODEL_PATH.is_file():
        raise ModelUnavailable(
            f"No model at {MODEL_PATH}. Run "
            "`python scripts/train_compatibility_model.py` to create it."
        )
    try:
        import joblib

        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:  # pragma: no cover - corrupt artifact
        raise ModelUnavailable(f"Could not load the model: {exc}") from exc

    missing = set(FEATURE_COLUMNS) - set(bundle.get("feature_columns", []))
    if missing:
        # A model trained against a different feature contract would produce
        # confident nonsense. Fail loudly instead.
        raise ModelUnavailable(
            f"Model was trained on different features; missing {sorted(missing)}. "
            "Retrain with scripts/train_compatibility_model.py."
        )
    return bundle


def predict_compatibility(item_a: Product, item_b: Product) -> float:
    """P(compatible) for one pair, in [0, 1].

    Uses exactly the same `pair_features` the training script used, so inference
    and training can never diverge on feature construction.
    """
    import pandas as pd

    bundle = load_model()
    frame = pd.DataFrame([pair_features(item_a, item_b)], columns=FEATURE_COLUMNS)
    return float(bundle["pipeline"].predict_proba(frame)[0, 1])


def predict_many(pairs: list[tuple[Product, Product]]) -> list[float]:
    """Batched predictions - one vectorised call rather than N."""
    import pandas as pd

    if not pairs:
        return []
    bundle = load_model()
    frame = pd.DataFrame(
        [pair_features(a, b) for a, b in pairs], columns=FEATURE_COLUMNS
    )
    return [float(p) for p in bundle["pipeline"].predict_proba(frame)[:, 1]]


def safe_predict(item_a: Product, item_b: Product) -> float | None:
    """Prediction, or None if the model is unavailable.

    Used by the UI, which must degrade quietly: a missing model artifact is not
    a reason to break a product page.
    """
    try:
        return predict_compatibility(item_a, item_b)
    except Exception:
        return None
