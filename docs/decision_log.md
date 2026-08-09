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
