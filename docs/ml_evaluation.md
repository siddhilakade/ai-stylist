# ML evaluation

> Regenerate with `python scripts/evaluate_ml.py`. Raw output:
> [`ml_evaluation_results.json`](ml_evaluation_results.json). Every number below
> is measured, none is illustrative.

**Headline: the ML model did not beat the rule baseline, and neither beat a
trivial baseline. The model was therefore not shipped.**

---

## 1. What can and cannot be measured here

There is no ground truth for outfit compatibility on this catalog — no clicks, no
purchases, no expert labels. So there are two evaluations, and **only the second
one can answer "is the ML better?"**

| | Labels | What it can tell us |
|---|---|---|
| **A. Held-out article types** | Synthetic (rule-generated) | Whether the model *induced* the heuristic from raw attributes and transferred it to unseen garment types |
| **B. Author-reviewed pairs** | Hand-judged, no reference to the rules | Whether either system agrees with considered human judgement |

Evaluation A **cannot** compare the two systems: the rule engine produced those
labels, so it scores a perfect 1.000 by construction. That circularity is
reported rather than hidden.

## 2. Evaluation A — transfer to unseen garment types

Held out entirely from training: **Formal Shoes, Sarees, Shirts, Skirts, Watches**.
Test set = 7,134 pairs where at least one garment is of an unseen type.

| System | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Rule baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **ML model** | **0.941** | **0.949** | **0.914** | **0.931** | **0.974** |

> The rule row is **circular** — it *is* the labelling function. Shown to make
> that explicit, not as a comparison.

For in-distribution pairs (seen garment types) the model scores F1 0.976 /
AUC 0.995; on held-out types it drops to F1 0.931 / AUC 0.974.

**Reading:** the model genuinely generalises. A ~4.5-point F1 drop when facing
garment types it never saw is a real, modest transfer gap — it learned structure,
not just memorised pairs. As an ML exercise, this part worked.

**Feature importance** (one-hot columns summed back to source features):

| Feature | Importance |
|---|---|
| `usage_a` | 0.182 |
| `usage_b` | 0.176 |
| `same_usage` | 0.153 |
| `formality_b` | 0.125 |
| `formality_a` | 0.110 |
| `articleType_a` | 0.065 |
| `articleType_b` | 0.056 |
| `baseColour_a` | 0.016 |
| everything else | < 0.015 each |

Occasion and formality dominate (**74% combined**), which matches the rule
engine's own weighting — occasion + formality carry 0.55 of its compatibility
score. Colour contributes far less than the rules give it (0.30 weight), meaning
the model largely ignores the colour rule and still recovers most of the label.
`price_ratio` — the one feature the rules never use — contributes 0.004, i.e.
nothing.

## 3. Evaluation B — author-reviewed pairs (the real test)

**65 pairs**, labelled by hand from the garment descriptions **without consulting
`pair_compatibility()`**. 39 compatible / 26 not (60% positive rate). 14 pairs
were chosen specifically as cases where the rules were *expected* to be wrong.

| System | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **0. Always-compatible (trivial)** | 0.600 | 0.600 | **1.000** | **0.750** | 0.500 |
| A. Rule baseline | **0.569** | **0.677** | 0.538 | 0.600 | **0.637** |
| B. ML model | 0.538 | 0.667 | 0.462 | 0.545 | 0.561 |
| C. Hybrid (0.6·ML + 0.4·rule) | 0.508 | 0.613 | 0.487 | 0.543 | 0.591 |

Three findings, in order of how uncomfortable they are:

**1. ML lost to the rules.** F1 0.545 vs 0.600, AUC 0.561 vs 0.637. Expected, and
the reason is structural: the model was trained to reproduce the rules, so it can
approximate them but cannot exceed them. It inherits their mistakes and adds
approximation error on top.

**2. The hybrid was worst of the three.** Blending a weaker signal into a
stronger one degraded it. Blending is not free.

**3. Nothing beat "always say compatible" on F1 (0.750).** Both real systems
score *below* a system with no logic at all. This is the finding worth
sitting with, and there are two honest readings:

- On this deliberately adversarial set the trivial baseline is flattered — it
  gets recall 1.000 for free while the set is 60% positive.
- But AUC cuts through that: trivial = 0.500 by definition, rules = 0.637,
  ML = 0.561. **The rules do carry real ranking signal** — just much weaker than
  their confident presentation implies.

## 4. The adversarial subset

The 14 pairs where the rules were predicted to fail:

| System | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Rule baseline | 0.286 | 0.286 | 0.167 |
| ML model | **0.357** | **0.308** | 0.229 |
| Hybrid | 0.286 | 0.286 | 0.188 |

Both systems are **worse than random** here — AUC 0.167 means the rules rank
these pairs almost exactly backwards. The prediction that the rules would fail
these cases was correct. The ML model is marginally less wrong, on 14 examples,
which is not a result anyone should act on.

Concretely, the rules get these wrong:

- **brown shoes with black trousers** — brown is treated as a neutral, so it
  scores well; conventional menswear says never
- **black with navy** — both "neutral", so ~1.0; muddy in daylight
- **head-to-toe one colour** — perfect colour agreement by the rules, reads as a
  uniform to a person
- **blazer with jeans** — penalised as a formality gap, actually a standard
  smart-casual outfit

## 5. Where the two systems disagreed

Only **4 of 65** pairs — the model reproduces the rules almost exactly, which is
what training on rule-generated labels produces.

| Pair | Human | Rule | ML | Who was right |
|---|---|---|---|---|
| Red Kurta + Red Leggings | 0 | 1 (0.73) | 0 (0.47) | **ML** |
| Navy Saree + Black Heels | 1 | 1 (0.70) | 0 (0.38) | Rule |
| Red Saree + Silver Watch | 1 | 1 (0.71) | 0 (0.28) | Rule |
| White Shirt + Navy Tie | 1 | 1 (0.72) | 0 (0.28) | Rule |

The ML model is *more conservative* — it rejects three pairings a person would
happily wear. Note three of the four involve held-out article types (Sarees,
Shirts, Watches), i.e. exactly where its transfer gap lives.

## 6. Decision

**Do not integrate.** Per the pre-registered rule — ship the ML component only if
it demonstrably improves on the baseline — it did not, so:

- `src/compatibility.py` and `src/recommender.py` are **unchanged**
- the rule engine remains the production recommender
- the model is kept, loadable and tested, as a reproducible experiment
- the UI shows its probability in the technical panel, **labelled as not
  affecting ranking**
- [`TestNotInProductionPath`](../tests/test_ml.py) asserts the recommender does
  not import it, and that recommendations are byte-identical with the model loaded

Forcing it in would have made the system worse and the writeup dishonest.

## 7. Limitations of this evaluation

- **65 pairs is a smoke test, not a validation set.** Differences of a few points
  of F1 are well inside noise at this size.
- **Single annotator.** One person's taste, self-labelled, no inter-annotator
  agreement. A different reviewer would move these numbers.
- **Deliberately adversarial.** ~22% of pairs were chosen because the rules were
  expected to fail. Absolute accuracies here are *lower bounds*, not estimates of
  catalog-wide performance.
- **Labels came from the same person who wrote the rules.** Independence of
  *process* (written without consulting the score) is not independence of *mind*.
- **Still no real-world validity.** None of this measures whether anyone would
  wear these outfits. Only user interaction data answers that.

## 8. Why no Precision@K / NDCG / MAP

The assignment permits alternative methodologies where standard metrics do not
apply. They do not apply here:

- **NDCG / MAP / Precision@K** need graded relevance per query. There are no
  relevance judgements, so any such number would be invented.
- **Novelty / serendipity** need popularity or interaction priors. The dataset
  has neither.
- **CTR / conversion / user satisfaction** need users. There are none.

What *is* measured instead — objectively and verifiably — is in
[EVALUATION.md](EVALUATION.md): constraint satisfaction, budget compliance,
explicit-item satisfaction, category completeness, graceful-failure quality,
diversity, catalog coverage and latency, across 34 hand-labelled scenarios.
