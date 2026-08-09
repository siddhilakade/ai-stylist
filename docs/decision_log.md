# Decision log

Decisions that shaped the system, with what was rejected and why. Ordered by how
much they matter.

---

### D1 — LLM at the boundaries, deterministic engine in the middle

**Decision.** Gemini/Groq converts language into a validated struct, and narrates
a chosen outfit. It never selects products, scores compatibility, or enforces a
budget.

**Rejected:** giving the model the catalog and asking for an outfit (hallucinated
products, broken budgets, non-deterministic and therefore untestable); LLM
re-ranking of a shortlist (reintroduces non-determinism into the one part that
most needs testing).

**Consequence.** Product ids resolve through exactly one function, and the LLM
response schema has no field that could carry a product — so a hallucinated
product is structurally impossible, not merely discouraged.

---

### D2 — Explicit requests are hard constraints; general preferences are soft

**Decision.** "black shirt" → the top must be a shirt and must be black.
"I like neutral colours" → a ranking preference only. Adjacency decides which.

**Why.** The original build ignored this and returned a *white top* for "black
shirt". Worse, "black shirt" and "white shirt" returned **identical** results,
because both colours collapse into the `neutral` family.

**Consequences.** Explicit colours match concrete `baseColour` values, not
families. A named slot is exempt from the formality band — that band comes from
an *inferred* occasion and could only veto what the user asked for.

---

### D3 — Hand-authored compatibility rules, not a learned model *(revisited in D9)*

**Decision.** Four named signals with documented weights.

**Why.** No compatibility ground truth exists for this catalog. A learned model
would have been unverifiable.

**Cost, now measured:** the rules agree with considered human judgement only
~57% of the time (AUC 0.637). Honest, and worse than their confident presentation
suggests.

---

### D4 — Greedy assembly with budget lookahead, not beam search

**Decision.** Fill the most constrained slot first; before accepting any item,
verify the remaining budget still covers the cheapest option in every unfilled
slot.

**Why.** Without lookahead, greedy buys excellent shoes and then cannot afford
trousers. With it, the algorithm runs in ~4 ms and can be explained end to end in
ninety seconds. Beam search adds a width parameter with no principled way to
choose it.

---

### D5 — Synthetic prices

**Decision.** Generate prices, MRPs and discounts deterministically from
`articleType` + a hash of the product id.

**Why.** The dataset has no price column, and budget is central to the product. A
second dataset would not join reliably.

**Rejected:** dropping budget entirely (removes the most interesting constraint);
sourcing real prices (no reliable join).

**Obligation.** Disclosed in the README, product pages, How-it-works page and ML
docs. Never presented as real.

---

### D6 — Failure is a designed outcome

**Decision.** When constraints cannot be met, return zero outfits and explain
which constraint emptied.

**Why.** Silent substitution is the exact failure mode this architecture exists
to prevent. Commercially the wrong default; for an evaluable prototype, the right
one.

---

### D7 — Higher-resolution images from a community re-export

**Decision.** Re-export images from `benitomartin/...-900x1200` (same catalog,
same ids), store at 720 px, serve as static files.

**Why.** The canonical dataset ships 60×80 thumbnails. A product page renders at
360 CSS px — **720 device px on a HiDPI display** — so anything smaller is being
upscaled on exactly the screen a reviewer uses.

**Rejected:** the full Kaggle dataset (~25 GB, needs credentials, far more
resolution than the UI can display); sharpening the thumbnails (cannot restore
detail that was never captured).

**Follow-on.** At 720 px, base64-inlining a 48-card grid would push ~4 MB through
the websocket per rerun — hence static file serving.

---

### D8 — Guarantee staple catalog coverage

**Decision.** After quota sampling, explicitly add missing staple
(gender × garment × colour) combinations.

**Why.** Proportional sampling produced 28 pairs of trousers and *not one black
pair*, so "black trousers" hit a dead end caused by our sampling rather than by
the data.

---

### D9 — Build the ML model, then **do not ship it**

**Decision.** Train a Random Forest compatibility model. Evaluate it against an
independent hand-labelled set. Keep the rule engine in production because the
model did not beat it.

**Measured:** ML F1 **0.545** vs rule **0.600** vs trivial always-compatible
**0.750**. AUC: rule 0.637, ML 0.561, chance 0.500.

**Why it lost — and why that was predictable.** The labels came from the rules,
so the model's ceiling *is* the rules. It can approximate them, inherit their
mistakes, and add approximation error. It cannot exceed them.

**Why it was still worth building.** It answers a real question with a real
number instead of a claim, it demonstrates the leakage trap and how to avoid it
(raw-attribute features, article-type group split), and it shows the model
genuinely generalises to unseen garment types (AUC 0.974) — the ML worked; the
*labels* were the limitation.

**Rejected:** shipping it anyway to be able to say "ML-powered" (would degrade a
working system and misrepresent the evidence); tuning the decision threshold on
the 65-pair validation set until ML won (overfitting a validation set and
reporting it as a result).

**Enforcement.** A test asserts the recommender does not import the model and
that recommendations are identical with it loaded.

---

### D10 — Groq as the LLM fallback provider

**Decision.** Keep the provider behind the existing two-function seam.

**Why.** The Gemini key's Google Cloud project returned
`403 PERMISSION_DENIED — "Your project has been denied access"` on every model,
across two separate keys. Not a code problem; not fixable from here.

**Validated.** `llama-3.3-70b-versatile` at 371–809 ms was the only model tested
that got the hard/soft constraint split right; the smaller models leaked an
explicit garment colour into the soft palette preference.

---

### D11 — The circuit breaker

**Decision.** After two consecutive LLM failures, stop calling until something
changes.

**Why.** A page render makes one extraction call plus one per outfit. With a dead
provider that is four doomed round-trips and several seconds of stalling for a
result already known to be a fallback.

**Measured:** 1,150 ms for the one real failed call, then 357 ms for three
explanations with no network at all.

---

### D12 — Break scoring ties on the request, not on catalog order

**Symptom.** Different requests kept returning the same products. Across 12
varied requests only 86 distinct products appeared (16% of the catalog), one pair
of grey shoes in 5 of them.

**Diagnosis.** Not a shortage of candidates — pools held 37–41 items per slot. The
scores are *coarse*: `preference_match` takes only 3–12 distinct values across a
whole filtered pool, so on average **10 of 29 candidates tie at the exact
maximum**, and for a formal request it was routinely all of them (20/20 tops,
16/16 bottoms). Ties were settled by `>` keeping the first item in pool order —
i.e. the lowest catalog id. Measured over 31 slot fills, the winner was the
lowest tied id **31 times out of 31**.

**Decision.** Break ties with a hash of (serialised request, product id), scaled
to at most 1e-6.

**Why that is safe.** Every score is rounded to 4dp before assembly, so the
smallest genuine gap between two candidates is 5e-5 — fifty times the nudge. It
can only reorder candidates that are already exactly equal. Determinism is
preserved exactly: the same request yields the same salt and the same outfit.

**Rejected:** random jitter (destroys reproducibility and regression tests);
a cross-request "recently shown" penalty (hidden mutable state, and the same
request would stop being reproducible).

**Cost.** +0.12 ms per request, measured by interleaved A/B over 96 runs each.

---

### D13 — Penalise all-neutral outfits, not just busy ones

**Symptom.** Selected items were **92.7% neutral** against 46.1% in the candidate
pool — the ranker preferred neutrals twice as often as chance.

**Diagnosis.** `color_pair_score` gives neutral+neutral a perfect 1.00, so an
outfit of nothing but neutrals scored a flawless colour signal. "All grey" was
the mathematically optimal answer to almost every request. The busy-outfit
penalty guarded one end of the range and nothing guarded the other.

**Decision.** A mild 0.92 multiplier when an outfit contains *no* strong colour
family, mirroring `BUSY_OUTFIT_PENALTY` at the opposite extreme.

**Why 0.92 and not 0.85.** An all-neutral outfit is a real look, not an error. At
0.92 a single colour accent (0.95) edges out total neutrality (0.92), which
breaks the monopoly — while an all-black outfit still beats a genuine clash and
still wins when it is the better answer on formality, occasion and budget. Both
bounds have tests.

**Result of D12 + D13 together.** Catalog coverage in the evaluation rose from
29.9% to 36.6% (160 → 196 products), preference match 0.919 → 0.953, diversity
0.649 → 0.673. Validity, completeness, constraint satisfaction, budget
compliance, explicit-item satisfaction and graceful failure all stayed at 100%.

---

### D14 — The monotony penalty must not fire on an explicit neutral request

**Symptom.** "College presentation tomorrow — smart casual, black and white,
under ₹3000" returned a white dress, grey shoes and a **green bangle**, ranked
*above* an all-neutral grey-tee / black-skirt / silver-flats look.

**Diagnosis.** Two independent faults compounding:

1. [D13](#d13--penalise-all-neutral-outfits-not-just-busy-ones) was fighting the
   brief. "black and white" resolves to the `neutral` family, so the requested
   outfit *is* all-neutral — and D13 marked it down 0.92 for exactly that. Worse,
   it inverted the accessory gate: dress + shoes scored 1.00 on colour, was
   penalised to 0.92, and adding a green bangle *raised* compatibility to 0.93.
   The guard that should have rejected the bangle (`MAX_COMPATIBILITY_DROP`)
   never saw a drop.
2. The optional-slot gate only ever checked **compatibility**, never **request
   fit**. The bangle scored 0.5025 on preference against ~0.80 for its
   neighbours and still got in, because it sat happily beside two neutrals on
   the four compatibility signals.

Budget is what surfaced it: the dress and shoes left ₹252, and that bangle was
the only accessory that fit.

**Decision.**
- `outfit_signals` / `outfit_compatibility` / `build_outfit` take
  `penalise_monotony`, and the recommender passes `False` when `neutral` is in
  `preferred_colors`. Penalising a user for complying with their own stated
  preference is always wrong.
- Added `OPTIONAL_SLOT_MAX_PREFERENCE_DROP = 0.05`, mirroring the compatibility
  guard on request fit. An optional extra is a bonus; if the only affordable one
  fights the brief, the better outfit is the one without it.

**Result.** The top look becomes white dress + grey shoes, scoring 0.9300 (was
0.8863 with the bangle). Evaluation preference match **0.953 → 0.965**, every
constraint metric unchanged at 100%.

**Known remainder.** Colour is still soft in *required* slots, so a third-choice
look can carry orange flats when nothing neutral is affordable. That is the
correct trade — hard-filtering colour empties slots — and it now only shows up
below two better-ranked, on-brief looks.
