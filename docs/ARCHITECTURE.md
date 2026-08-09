# Architecture and design decisions

This document records *why* the system is shaped the way it is, including the
things that were deliberately not built. [README.md](../README.md) covers what it
does and how to run it.

---

## 1. Problem statement

A fashion marketplace is optimised for **item-level discovery**: given a product,
find similar products. Search, filters and "you may also like" all answer *"what
is like this?"*

Shoppers routinely have a different question — **outfit-level** and
constraint-bound:

> *"I have a presentation tomorrow. What should I actually wear, that goes
> together, that I can afford?"*

Answering it requires reasoning about **relationships between products across
categories** under **hard constraints** (budget, occasion, gender), and then
**explaining** the result well enough to be trusted. Similarity models actively
work against this: cosine similarity on a white shirt returns twelve more white
shirts, which is precisely what someone building an outfit does not want.

---

## 2. The core architectural decision

> **The LLM handles language. A deterministic engine handles selection.**

This is the decision everything else follows from, so it is worth stating the
alternatives that were rejected.

### Rejected: "Give Gemini the catalog and ask for an outfit"

Tempting, and the fastest thing to build. It fails on five counts:

| Problem | Consequence |
|---|---|
| **Hallucination** | The model invents plausible product names and prices that do not exist in the catalog |
| **Budget violations** | LLMs are unreliable at arithmetic under constraint; "under ₹3,000" becomes ₹3,400 |
| **Non-determinism** | The same request gives different answers, so nothing can be regression-tested |
| **Unexplainability** | "Why these items?" gets a post-hoc rationalisation, not the actual reason |
| **Cost and latency** | Every request becomes a large-context LLM call |

### Rejected: "Let Gemini re-rank the engine's shortlist"

Better, but still gives the model authority over the final answer without giving
it the information to exercise that authority well — and it reintroduces
non-determinism into the one part that most needs to be testable.

### Chosen: LLM at the boundaries, engine in the middle

```
  ┌──────────── LANGUAGE ────────────┐  ┌──── SELECTION ────┐  ┌─── LANGUAGE ───┐
  │                                  │  │                   │  │                │
  │  free text                       │  │  hard filters     │  │  stylist note  │
  │      ▼                           │  │      ▼            │  │       ▲        │
  │  Groq / Gemini (Task A)          │  │  semantic         │  │  Groq/Gemini(B)│
  │      ▼                           │  │  retrieval        │  │       ▲        │
  │  StylePreferences ───────────────┼─►│      ▼            │  │  grounding     │
  │  validated, closed vocabulary    │  │  compatibility    │  │  payload       │
  │                                  │  │      ▼            │  │  (real items,  │
  │  fallback: keyword parser        │  │  greedy assembly  │  │   real scores) │
  │                                  │  │      ▼            │  │                │
  │                                  │  │  ranking ─────────┼─►│                │
  └──────────────────────────────────┘  └───────────────────┘  └────────────────┘
                                          deterministic,
                                          testable, ~57 ms
```

Semantic retrieval sits **inside** the deterministic block on purpose: it is a
pretrained encoder with fixed weights and an exact index, so it is as
reproducible as the rules around it. See §7.5.

The LLM is used where language is genuinely hard, and excluded where correctness
is genuinely hard.

### The structural guarantee

Product ids are resolved through **exactly one function**, `data.get_product()`.
Gemini receives no ids and no catalog, and its response schema has **no field
that could carry a product** — no `product`, `product_id`, `name`, `price` or
`items`. It is not that the model is instructed not to hallucinate a product; it
is that a hallucinated product has nowhere to go.

Tested in [`TestNoHallucination`](../tests/test_recommender.py) and
[`test_schema_has_no_field_that_could_carry_a_product`](../tests/test_llm.py).

---

## 3. Why structured output specifically

`GeminiExtraction` in [`src/schemas.py`](../src/schemas.py) is deliberately a
**separate, flatter model** from `StylePreferences`:

- **Every field is required and non-nullable.** Structured-output APIs handle a
  flat enum/int/bool/string schema far more reliably than one full of
  `anyOf: [T, null]` unions. Sentinels (`"unspecified"`, `0`, `""`) are used
  instead of nulls and translated at the boundary.
- **Every free-choice field is an enum**, generated from the same domain tables
  the engine uses, with import-time assertions that the two cannot drift apart.
  The model cannot emit an occasion the filter does not understand.
- **Validation is all-or-nothing.** An invalid response is discarded entirely
  rather than partially salvaged — a half-parsed preference set is *worse* than a
  keyword-parsed one, because it looks authoritative.

## 4. Why a rule-based fallback exists

`src/nlu.py` is not a consolation prize. It serves three purposes:

1. **The app is never dead.** No key, exhausted quota, or no network — Style Me
   still works, degraded, and the UI says so.
2. **It is the control condition.** Having a baseline is what lets us state
   concretely what the LLM buys. The evaluation reports per-field accuracy for
   both paths side by side.
3. **It makes the LLM's failure mode safe.** Every rejection path falls back here.

---

## 5. Feature engineering decisions

### One formality scale instead of many

The most common real styling error is a **formality mismatch** (formal brogues
with track pants). Putting every item on a single 0–4 axis reduces "are these
compatible?" to a distance, which is trivially explainable and trivially testable.

The dataset labels **77% of items simply "Casual"**, so `usage` alone cannot
discriminate. A ~16-entry per-`articleType` adjustment table does the real work,
calibrated against obvious cases. A separate **ceiling table** defends against
label noise — the dataset genuinely contains a novelty t-shirt tagged
`usage="Formal"`.

### Ethnic wear as its own axis

Indian ethnic wear follows its own pairing logic — a kurta does not go with
formal trousers, but does go with jeans. Squeezing this onto the formality scale
would produce wrong answers in both directions, so it gets a dedicated boolean
and its own compatibility signal.

### `onepiece` as a first-class slot

A dress or saree satisfies top *and* bottom. Modelling that as a second outfit
template (rather than special-casing it inside the builder) keeps the assembly
algorithm uniform and makes women's results genuinely varied.

### Occasion → formality band, not `usage` equality

Filtering `usage == "Smart Casual"` would return 67 items from 44,072. Each
occasion instead maps to a target formality with a tolerance band. An **explicitly
stated** style narrows the band further (1.0 instead of the occasion's 1.25),
because the user saying "smart casual" is stronger evidence than us inferring it.

That last point came directly from testing: with a wide band, a maximally
self-consistent tee/jeans/sneakers combination out-scored the shirt-and-trousers
outfit a "smart casual" request actually asked for.

---

## 6. Scoring decisions

### Why hand-authored rules and not a learned model

There is **no outfit-compatibility ground truth** for this catalog. Options were:

1. Train on Polyvore outfit data → different catalog, different label space, no
   reliable join, and a large unexplainable model in the critical path.
2. Hand-label compatibility → thousands of pairs, and it would only reproduce the
   labeller's rules with extra steps and less transparency.
3. **Encode conventional styling advice as transparent rules** ← chosen.

Option 3's honest cost: the rules encode *conventional* advice and will miss
deliberate, fashion-forward clashing. That is stated as a limitation rather than
hidden.

### Why colour is scored, not filtered

Hard-filtering colour is the fastest way to produce an empty result set. It is
scored instead, with neutrals scoring 0.55 against a colour preference because a
neutral genuinely coordinates with any requested palette. There is a test
asserting colour never changes the size of the filtered set.

### Why the weights are round numbers

Compatibility 0.45 / preference 0.35 / budget 0.20, and within compatibility
0.30 / 0.30 / 0.25 / 0.15. Each has a stated reason. They are **not** tuned
against the evaluation set — fitting seven weights to 34 scenarios would fit
noise and produce a number that looks earned but is not.

---

## 7. Assembly decisions

### Why greedy and not beam search or exact optimisation

Selecting one item per slot to maximise mutual compatibility under a budget is a
maximum-weight clique on a slot-partite graph. Exhaustive search over ~100
candidates per slot is expensive; beam search is better but adds a width
parameter with no principled way to choose it and makes the algorithm markedly
harder to explain.

Greedy with **most-constrained-slot-first** ordering plus **budget lookahead**
gets results that pass every constraint check in the evaluation, in ~4 ms of
assembly time, and can be explained in ninety seconds. For a 536-product catalog
the optimality gap is not what limits recommendation quality — the rules are.

### Budget lookahead is the non-obvious part

Before accepting any candidate, the money left must still cover the cheapest
available item in **every** unfilled slot. Without it, greedy spends the budget on
excellent shoes and then cannot afford trousers. This single check is what makes
naive greedy actually work under a hard budget, and it has a dedicated test.

### Diversity by construction

Outfit *k* may not reuse any item from outfits *1…k−1*. Guaranteed-distinct
results, no inter-outfit similarity metric, no tuning parameter.

That handles diversity *within* one result set. Diversity *between* different
requests turned out to be a separate problem with a separate cause — see
[D12 in the decision log](decision_log.md): the scores are coarse enough that
~10 of 29 candidates routinely tie, and ties were being settled by catalog id.

---

## 7.5 Semantic retrieval

Added after the rule engine was complete and evaluated, which is why it appears
as a distinct layer rather than being woven through.

**Where it sits.** Strictly *after* hard filtering. It receives products that
already satisfy gender, explicit garment and colour, formality band, ethnic rules
and budget, and it may only reorder and shorten that list. It can never add a
product back or overrule a constraint — the direction of the dependency is what
makes it safe to add to a system whose main claim is determinism.

**What it is for.** The part the rule engine structurally cannot do: act on the
free-text remainder of a request. "Nothing too flashy" has no structured field to
live in, and no filter can be written for it.

**Three non-obvious design choices:**

- **Per slot, never global.** A global top-K would happily return 40 tops and
  zero footwear, manufacturing a failure out of a satisfiable request.
- **The cheapest item in each slot always survives.** Otherwise narrowing can
  raise the price floor of the whole outfit and break a budget that was
  satisfiable — a retrieval heuristic silently overruling a hard constraint.
- **Pretrained, not fine-tuned.** There are no compatibility labels to fine-tune
  on. A general-purpose encoder over short product descriptions is the honest
  tool; anything else would be a number without a basis.

**Cost.** This is the expensive part of the system: the encoder forward pass is
~84% of request latency (measured), which is what moved p50 from ~4 ms to ~57 ms.
`IndexFlatIP` search over 536 vectors is 0.3 ms — 0.5% of the request. The index
is not the cost; the embedding is.

**Honest result.** Measured against the TF-IDF baseline it is a **tie** —
attribute hit-rate 0.690 vs 0.680 over 10 probes. The split is the interesting
part: semantic wins on paraphrase ("business meeting clothes" 0.50 vs 0.10),
TF-IDF wins on literal keywords ("formal outfit" 0.30 vs 1.00). Overlap@10 is
0.22, so they rank very differently. Both are kept: semantic for retrieval,
TF-IDF for the "similar products" rail on a product page.

---

## 8. Data decisions

### One dataset, not several

A second dataset for prices would not join reliably to this catalog, and a
multi-dataset benchmark architecture would add a large amount of surface area for
zero product value. Prices are generated deterministically instead, and the fact
that they are synthetic is stated in the README, the app's product pages and the
How-it-works page.

### Stratified sampling, not random

Detailed in the README. The short version: the source is severely skewed, and a
random 536-row sample would leave whole cells empty — making the recommender fail
for reasons that have nothing to do with the recommender.

### Images committed to the repo

About 30 MB total, served as static files. Deployment needs no dataset download, no network call at runtime,
and no dependency on a third party staying up.

---

## 9. What was deliberately not built

| Not built | Why |
|---|---|
| CLIP / image embeddings | Not on the critical path. A working, explainable, evaluated system is worth more than an unfinished embedding pipeline. Listed as the top future improvement. |
| **Hosted** vector database (Pinecone, pgvector, Supabase) | FAISS *was* built — see §7.5 — but as a local index file, not a service. At 536 vectors, search is 0.3 ms in-process; a network round trip is 8–220 ms depending on region. It would make the system measurably slower to solve a problem it does not have. |
| Collaborative filtering | No interaction data exists in the dataset. Building it would mean fabricating users. |
| Learned ranking / LTR | No relevance labels. Same reason. |
| Agents / multi-agent orchestration | The pipeline is a fixed sequence of five deterministic steps. An agent loop would add non-determinism and latency to solve a problem that does not exist here. |
| RAG | The "knowledge" is a 536-row CSV that fits in memory, and the LLM is deliberately never given the catalog. Retrieval augmentation over it would be theatre. |
| Auth / cart / checkout / inventory | Explicitly out of scope; they demonstrate nothing about recommendation quality. |
| Docker | `pip install -r requirements.txt && streamlit run app.py` already works from a clean environment. |

The consistent principle: **every component must survive the question "what does
this buy the user, and can I explain it?"**

---

## 10. Technology choices

| Choice | Why | Cost |
|---|---|---|
| **Streamlit** | Fastest path to an interactive testing UI, which the brief requires; pure Python so the engine and interface share types | Limited layout control; custom CSS needed for a marketplace look |
| **Gemini Flash** | Low latency matters when the call is inline in a UI interaction; native structured output with response schemas | Vendor dependency — mitigated by the fallback path and a `GEMINI_MODEL` env override |
| **Pydantic** | Turns the LLM contract into a validated type, and generates the response schema from the same definition | — |
| **pandas + scikit-learn** | Already the right size for 536 rows; TF-IDF is one import | — |
| **CSV, no database** | 536 rows; a database would be one more thing to explain, provision and fail | Does not scale — correctly, since this is a prototype |

---

## 11. Scaling this to a real catalog

The honest answer to "what breaks at 4 million products?":

1. **Candidate generation** would need an index. Hard filters become a query
   against a product store with indexes on (gender, slot, formality band, price).
2. **Compatibility** is already O(1) per pair and would stay in the ranking layer,
   applied to a few hundred retrieved candidates rather than the whole catalog.
3. **Assembly** is unchanged — it operates on candidate pools, not the catalog.
4. **The LLM calls are unchanged** and remain O(1) per request.
5. **Caching**: extraction results cache on request text; compatibility between
   any two products is deterministic and precomputable per category pair.

The architecture's shape does not change. That is a consequence of keeping the
LLM out of the selection loop — the expensive, non-deterministic component is
already at the edges, where it can be cached and where its failure degrades
gracefully.
