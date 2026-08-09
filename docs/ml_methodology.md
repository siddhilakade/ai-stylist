# ML methodology

> **Status: experimental. The model is NOT in the production recommendation path.**
> It was built, trained and evaluated; the evaluation did not justify shipping it.
> See [ml_evaluation.md](ml_evaluation.md) for the numbers and
> [decision_log.md](decision_log.md) for the decision.

---

## 1. Why add ML at all

The recommender was entirely rule-based: hand-authored compatibility signals,
transparent weights, greedy assembly. That is defensible and it works — but it
contains no *learned* component, and "AI/ML understanding" is an explicit
assessment dimension.

The honest framing is: **can a model learn compatibility from raw product
attributes, and is it better than the rules I wrote by hand?** That is a real
question with a real answer, and the answer turned out to be *no* — which is
still a result.

## 2. The data problem, stated plainly

`ashraq/fashion-product-images-small` contains **no user-item interactions and no
outfit compatibility labels**. There are no clicks, no purchases, no saves, no
expert annotations. None exist for this catalog.

This rules out entire families of approach:

| Approach | Why not |
|---|---|
| Collaborative filtering | Needs user-item interactions. There are none. Inventing users would be fabricating data. |
| Learning-to-rank | Needs graded relevance judgements per query. None exist. |
| Fine-tuned embedding model | Needs positive/negative outfit pairs from real outfits. None exist. |

What remains is **content-based pairwise compatibility**, trained on labels we
generate ourselves and label as such.

## 3. Labels: synthetic / heuristic

Training labels come from this project's own rule engine
(`src/compatibility.py::pair_compatibility`), thresholded at **0.70**:

```
compatible = 1  if  pair_compatibility(a, b) >= 0.70  else  0
```

**These are synthetic/heuristic labels. They are not user behaviour, not
purchases, not expert annotation.** Every artifact repeats this: the generator
script, the model bundle's `label_provenance` field, and a test that asserts the
provenance string is present.

Dataset: **24,000 pairs** sampled deterministically (SHA-256 of the id pair, not
a PRNG seed, so it reproduces across machines) from 60,178 valid cross-slot
pairs. Balance: **60.7% compatible / 39.3% incompatible**.

Only cross-slot pairs are generated — two shirts are not a "pair" in outfit
terms — and mixed menswear/womenswear pairs are excluded, since the recommender
already forbids them upstream as a hard constraint.

## 4. Avoiding the leakage trap

This is the part that decides whether the experiment means anything.

**The trap:** generate labels from the rule, then train on the rule's own
sub-scores (colour score, formality score, occasion score). The model learns
`label = threshold(weighted_sum(inputs))`, reports ~0.99 accuracy, and has
demonstrated nothing except that a forest can fit a linear threshold.

**What was done instead:** the model sees **raw product attributes only**. No
rule score, and no pair-level quantity derived from one — no colour-compatibility
score, no formality distance, no occasion score. To predict the label it has to
*induce* the compatibility structure from `articleType`, `baseColour`, `usage`
and so on.

This is enforced by a test, not by discipline
([`test_no_rule_derived_feature_leaks_in`](../tests/test_ml.py)).

**Residual leakage, acknowledged:** the labels still originate from the rules, so
the model's target is a heuristic. Training on raw attributes makes the learning
problem genuine, but it does not make the *labels* independent. That is precisely
why the decisive evaluation uses a separately hand-labelled set (§6).

## 5. Features

All features are per-item raw attributes or simple identity checks. Both items in
the pair contribute. Pairs are canonicalised into a fixed slot order first, so
the model cannot learn spurious "item A is the top" artifacts.

| Feature | Type | What it is | Why it matters | Limitation |
|---|---|---|---|---|
| `articleType_a/_b` | categorical | Garment type ("Shirts", "Formal Shoes") | The strongest single cue for what pairs with what | 44 values; rare types have few examples |
| `baseColour_a/_b` | categorical | Raw colour label | Colour is the most visible compatibility signal | Label noise — a brown belt tagged `Black` exists |
| `usage_a/_b` | categorical | Dataset occasion tag | Highest-importance feature in the fitted model | Severely skewed: 77% "Casual" |
| `season_a/_b` | categorical | Season tag | Weak seasonal coherence signal | Near-useless in practice (importance <0.01) |
| `slot_a/_b` | categorical | Outfit slot | Distinguishes top/bottom/footwear roles | Derived, but per-item not per-pair |
| `gender_a/_b` | categorical | Target gender | Guards menswear/womenswear coherence | Already a hard constraint upstream |
| `formality_a/_b` | numeric | Per-item 0–4 formality | Formality mismatch is the most common styling error | Derived from `usage` + a curated table |
| `price_a/_b` | numeric | Item price | Proxy for market tier | **Prices are synthetic** — see README |
| `price_ratio` | numeric | cheaper ÷ dearer | Do the two items sit at the same tier? Nothing in the rules uses price | Meaningless while prices are simulated |
| `is_neutral_a/_b` | boolean | Colour coordinates with anything | Neutrals are the backbone of coordination | Derived; browns/navy counted as neutral |
| `is_ethnic_a/_b` | boolean | Indian ethnic wear | Ethnic wear has its own pairing logic | Derived |
| `same_colour` | boolean | Identical `baseColour` | Detects monochrome pairings | Raw equality, not family-level |
| `same_usage` | boolean | Identical usage tag | Third-highest importance | — |
| `same_season` | boolean | Identical season | Seasonal coherence | Negligible importance |

**24 source features**, expanding to ~180 columns after one-hot encoding.

Note two are *derived* rather than strictly raw (`formality`, `is_neutral`,
`is_ethnic`, `slot`). They are computed **per item**, never per pair, so they
carry no information about the pair's rule score. That distinction is what keeps
them outside the leakage boundary.

## 6. Train/test split — by article type, not at random

A random split would be close to meaningless: each of 536 products appears in
many pairs, so the same garments land on both sides and the model scores well by
memorising specific combinations.

Instead, **entire article types are held out**:

```
held out: Formal Shoes, Sarees, Shirts, Skirts, Watches
train: 16,866 pairs (67.7% positive)   — no held-out type appears
test:   7,134 pairs (44.0% positive)   — at least one garment is an unseen type
```

Chosen to span slots and formality levels so the test set is neither all-easy nor
all-hard. This measures **transfer**: has the model induced a portable notion of
compatibility, or memorised the pairs it saw?

## 7. Model choice

**`RandomForestClassifier`** — 300 trees, `max_depth=12`, `min_samples_leaf=5`,
`class_weight="balanced"`.

| Why | |
|---|---|
| vs. gradient boosting | Competitive with almost no tuning; less prone to overfitting a heuristic target |
| vs. logistic regression | Compatibility is full of interactions ("formal shoes" × "shorts") that a linear model cannot express |
| vs. a neural network | 24k rows of low-cardinality categoricals is exactly where tree ensembles win. A network would be less explainable for no measured gain |

**Depth is capped at 12 deliberately.** Unlimited depth lets the forest memorise
individual product pairs — it inflates the in-distribution score and destroys
transfer to held-out garment types.

Every prediction traces back to readable if/else splits, and feature importances
come out directly — which matters more here than a fractional accuracy gain.

## 8. Training and inference

```
scripts/build_ml_dataset.py        → data/ml_training/compatibility_pairs.csv
scripts/build_human_validation.py  → data/ml_training/human_validation.csv
scripts/train_compatibility_model.py → models/compatibility_model.joblib (3.6 MB)
scripts/evaluate_ml.py             → docs/ml_evaluation_results.json
```

Training and inference are fully separate. **The Streamlit app never trains.** It
lazily loads the serialised pipeline once per process (`lru_cache`) and only when
something asks for a prediction.

Training and inference share one feature function, `pair_features()`, so the two
cannot drift — asserted by
[`test_inference_features_match_training_features`](../tests/test_ml.py).

## 9. What this model is not

- **Not personalised.** It has never seen a user. It scores item pairs.
- **Not trained on real preferences.** Its target is a heuristic I wrote.
- **Not production-ready.** It lost to the baseline; see the evaluation.
- **Not a compatibility oracle.** At best it is a compressed, generalising
  approximation of one set of hand-written styling rules.

## 10. What would make this real

In priority order:

1. **Log interactions.** Which suggested outfits get a click, a save, an
   add-to-bag. That is the missing ingredient, and nothing else substitutes.
2. **Retrain on implicit feedback** — the outfit shown vs. the outfit acted on.
   The scoring function is already a linear model over seven signals; fitting
   those weights to real feedback is a small, well-posed problem.
3. **Independent annotation at scale.** 65 author-judged pairs is not a
   validation set, it is a smoke test. A few thousand pairs from several
   annotators, with inter-annotator agreement reported.
4. **Visual features.** `baseColour` is a single word for a whole garment. CLIP
   embeddings would capture shade, print and texture — the things the current
   feature set is blind to.
