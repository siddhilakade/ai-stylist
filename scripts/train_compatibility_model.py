"""Train the pairwise compatibility model.

Split strategy matters more than the model here. A random split would be close to
meaningless - each product appears in many pairs, so the same garments land on
both sides and the model scores well by memorising combinations. Instead entire
article types are held out, which tests transfer rather than recall.

RandomForest over gradient boosting: competitive with almost no tuning, gives
usable probabilities, exposes feature importances, and every prediction traces
back to readable splits. Depth is capped at 12 on purpose - unlimited depth
memorises individual pairs and destroys transfer.

Run:  python scripts/train_compatibility_model.py
Out:  models/compatibility_model.joblib, models/training_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.features import (  # noqa: E402
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)

DATASET = ROOT / "data" / "ml_training" / "compatibility_pairs.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "compatibility_model.joblib"
REPORT_PATH = MODEL_DIR / "training_report.json"

# Article types held out entirely. Chosen to span slots and formality levels so
# the test set is not accidentally all-easy or all-hard.
HELD_OUT_ARTICLE_TYPES = {
    "Shirts",        # a very common top
    "Formal Shoes",  # the strongest formality anchor in the catalog
    "Skirts",        # a bottom with few training examples
    "Sarees",        # ethnic one-piece
    "Watches",       # accessory
}

RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """One-hot the categoricals, pass numerics through, then a forest."""
    encoder = ColumnTransformer(
        transformers=[
            ("categorical",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # numeric + boolean columns
        verbose_feature_names_out=False,
    )
    forest = RandomForestClassifier(
        n_estimators=300,
        # Depth is capped deliberately. Unlimited depth lets the forest memorise
        # individual product pairs, which inflates the in-distribution score and
        # destroys transfer to held-out garment types.
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("encode", encoder), ("forest", forest)])


def metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4)
        if len(set(y_true)) > 1 else float("nan"),
        "n": int(len(y_true)),
    }


def main() -> None:
    if not DATASET.exists():
        raise SystemExit("Run scripts/build_ml_dataset.py first.")

    frame = pd.read_csv(DATASET)
    print(f"Dataset: {len(frame):,} pairs")

    # --- group split by held-out article types ---------------------------
    is_held_out = (
        frame["articleType_a"].isin(HELD_OUT_ARTICLE_TYPES)
        | frame["articleType_b"].isin(HELD_OUT_ARTICLE_TYPES)
    )
    train, test = frame[~is_held_out], frame[is_held_out]

    print(f"  train (seen types)      : {len(train):,}  "
          f"positives {train['compatible'].mean():.1%}")
    print(f"  test  (held-out types)  : {len(test):,}  "
          f"positives {test['compatible'].mean():.1%}")
    print(f"  held out: {sorted(HELD_OUT_ARTICLE_TYPES)}")

    if train.empty or test.empty:
        raise SystemExit("Split produced an empty side; adjust HELD_OUT_ARTICLE_TYPES.")

    x_train, y_train = train[FEATURE_COLUMNS], train["compatible"]
    x_test, y_test = test[FEATURE_COLUMNS], test["compatible"]

    pipeline = build_pipeline()
    print("\nTraining RandomForest...")
    pipeline.fit(x_train, y_train)

    # --- evaluate ---------------------------------------------------------
    train_proba = pipeline.predict_proba(x_train)[:, 1]
    test_proba = pipeline.predict_proba(x_test)[:, 1]
    train_metrics = metrics(y_train, (train_proba >= 0.5).astype(int), train_proba)
    test_metrics = metrics(y_test, (test_proba >= 0.5).astype(int), test_proba)

    print("\n--- IN-DISTRIBUTION (article types seen in training) ---")
    for key, value in train_metrics.items():
        print(f"  {key:10s} {value}")
    print("\n--- HELD-OUT ARTICLE TYPES (the number that matters) ---")
    for key, value in test_metrics.items():
        print(f"  {key:10s} {value}")
    print("\n" + classification_report(
        y_test, (test_proba >= 0.5).astype(int),
        target_names=["incompatible", "compatible"], zero_division=0))

    # --- feature importance ----------------------------------------------
    encoder = pipeline.named_steps["encode"]
    forest = pipeline.named_steps["forest"]
    names = list(encoder.get_feature_names_out())
    importances = forest.feature_importances_

    # Roll one-hot columns back up to their source feature so the report is
    # readable: "baseColour_a" rather than 31 separate colour columns.
    grouped: dict[str, float] = {}
    for name, importance in zip(names, importances):
        source = next(
            (f for f in FEATURE_COLUMNS if name == f or name.startswith(f + "_")),
            name,
        )
        grouped[source] = grouped.get(source, 0.0) + float(importance)

    top = sorted(grouped.items(), key=lambda kv: -kv[1])
    print("--- FEATURE IMPORTANCE (one-hot columns summed per source feature) ---")
    for name, importance in top:
        print(f"  {name:20s} {importance:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "boolean": BOOLEAN_FEATURES,
            "held_out_article_types": sorted(HELD_OUT_ARTICLE_TYPES),
            "label_provenance": "synthetic/heuristic - generated by src/compatibility.py",
        },
        MODEL_PATH,
        compress=3,
    )

    REPORT_PATH.write_text(json.dumps({
        "model": "RandomForestClassifier(n_estimators=300, max_depth=12, "
                 "min_samples_leaf=5, class_weight=balanced)",
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "held_out_article_types": sorted(HELD_OUT_ARTICLE_TYPES),
        "split_strategy": "group split by article type (not random)",
        "label_provenance": "synthetic/heuristic from src/compatibility.py",
        "in_distribution": train_metrics,
        "held_out": test_metrics,
        "feature_importance": {k: round(v, 4) for k, v in top},
    }, indent=2), encoding="utf-8")

    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} ({size_mb:.2f} MB)")
    print(f"Saved {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
