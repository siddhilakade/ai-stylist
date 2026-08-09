# Assumptions

Every assumption this project makes that is *not* verifiable from the dataset,
and what happens if it is wrong. Stated separately from
[limitations](../README.md#known-limitations): a limitation is something the
system cannot do; an assumption is something taken on faith to make it work.

---

## Data assumptions

**A1. `usage` is a meaningful occasion signal.**
The dataset's `usage` column (Casual / Formal / Ethnic / Sports / …) is treated
as the primary evidence of what an item is *for*.
*Risk:* it is severely skewed — 77% "Casual", only 67 "Smart Casual" out of
44,072 — and demonstrably noisy (a novelty t-shirt tagged `Formal`).
*Mitigation:* never filtered on directly; mapped to a formality band, then
adjusted by a per-`articleType` table with a ceiling that caps inherently casual
garments. Documented in `src/features.py`.

**A2. `baseColour` describes the garment's dominant colour.**
One word per garment.
*Risk:* it collapses print, shade and texture; and some labels are simply wrong
(a visibly brown belt tagged `Black`). Explicit requests like "black belt"
inherit that error.
*Mitigation:* none possible without visual features. Disclosed.

**A3. `articleType` reliably identifies the garment.**
Used for outfit-slot assignment and for explicit item requests.
*Risk:* low — this is the cleanest column in the dataset.

**A4. A 536-product sample can represent a 44,072-product catalog.**
*Risk:* real. Sparse cells fail — women's formalwear and ethnic footwear in
particular.
*Mitigation:* stratified quota sampling plus a `STAPLES` table guaranteeing the
obvious (gender × garment × colour) combinations. Failures are explained, not
hidden.

**A5. Children's items are excluded, adult items are not children's items.**
*Risk:* the dataset mislabels some children's products with an adult gender.
*Mitigation:* name-pattern screening at build time.

## Domain assumptions

**A6. Conventional styling rules are a reasonable proxy for compatibility.**
The entire compatibility engine rests on this.
*Risk:* significant, and now **measured** — the rules agree with considered human
judgement only ~57% of the time, AUC 0.637
([ml_evaluation.md](ml_evaluation.md)). They will also miss deliberate,
fashion-forward clashing entirely.
*Mitigation:* none. This is the honest ceiling of a rule-based approach without
interaction data.

**A7. Formality can be represented on a single 0–4 axis.**
*Risk:* a simplification. Formality is contextual — "smart casual" in Mumbai and
London differ.
*Why it is still right here:* it reduces "are these compatible?" to a distance,
which is explainable and testable. The alternative — a formality matrix — would
be dozens of unexplained numbers.

**A8. Colour compatibility works at family level.**
~46 colours collapse to 10 families for scoring.
*Important exception:* explicit requests ("black shirt") match **concrete
`baseColour` values**, because black and white are both in the `neutral` family
and a family-level match cannot tell them apart.

**A9. Ethnic wear needs its own axis.**
Indian ethnic wear pairs by its own logic — a kurta goes with jeans, not with
formal trousers.
*Risk:* the `is_ethnic` flag is coarse and derived.

**A10. Menswear and womenswear are not mixed within one outfit.**
Enforced as a hard clash. *Risk:* excludes genuinely androgynous styling.

## Product assumptions

**A11. Users want complete outfits, not ranked items.**
The premise of the whole product.
*Risk:* unvalidated. No user research was done.

**A12. Budget applies to the whole outfit, not per item.**
*Risk:* low — but it is the opposite of how marketplace price filters work, so it
must be signposted in the UI.

**A13. An explicitly named garment outranks an inferred occasion.**
"White shirt" wins over a default casual formality band.
*Risk:* a user may name a garment loosely and be surprised by a strict match.
*Why:* returning a white *top* for "white shirt" is a worse failure.

**A14. Showing zero results beats showing wrong results.**
*Risk:* commercially wrong — empty pages do not convert.
*Why here:* an honest failure is more informative for an evaluator than a
plausible-looking substitution.

## Technical assumptions

**A15. Synthetic prices are acceptable for demonstrating budget logic.**
The dataset has no price or MRP column. Prices, MRPs and discounts are
deterministically generated from `articleType` + a hash of the product id.
*Risk:* the *logic* is real; the *numbers* are not. Any price-derived ML feature
(`price_ratio`) is therefore meaningless.
*Mitigation:* disclosed in the README, the product pages, the How-it-works page
and the ML docs.

**A16. Deterministic recommendations are preferable to stochastic ones.**
Same request → same outfits, always.
*Trade-off:* no exploration, no serendipity. Accepted, because reproducibility is
what makes the system testable.

**A17. Synthetic compatibility labels can support a meaningful ML experiment.**
*Risk:* the central risk of the ML work. The model's target is a heuristic, so it
can at best approximate the rules — which is exactly what the evaluation found.
*Mitigation:* raw-attribute-only features, article-type group split, and an
independent hand-labelled set as the decisive comparison.

**A18. An LLM is unnecessary for correctness.**
The app runs fully without one. *Validated by necessity:* Gemini's project was
blocked mid-build, and the system kept working.

**A19. Author-written styling labels are a usable independence check.**
*Risk:* real. Independence of *process* (written without consulting the rule
score) is not independence of *mind* — the same person wrote both the rules and
the labels.
*Mitigation:* the set was made deliberately adversarial, and the limitation is
stated wherever the numbers appear.
