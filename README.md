# AI Stylist
YouTube Explanation Video: https://youtu.be/PyvfDrC5Ysc?si=G-vEfIBjXZEs1bLk
Outfit recommendation for a fashion marketplace. You describe an occasion in
plain English and get back complete, budget-valid outfits with the reasoning
shown.

```
"smart casual for a college presentation, under ₹3000, nothing too flashy"

  Black Shirt        ₹799
  Cream Trousers   ₹1,149
  Grey Sneakers    ₹1,599
  Brown Belt         ₹449
  ─────────────────────────
  Total            ₹3,996  of ₹4,000

  Formality consistent · colours coordinate · within budget
```

Search engines are good at *"what's similar to this?"*. This answers
*"what actually goes **together**, for my occasion, in my budget?"*

---

## How it works

```
free text → LLM → structured preferences → hard constraints
                                                  ↓
                                     semantic retrieval (MiniLM + FAISS)
                                                  ↓
                                     compatibility → outfit assembly
                                     (budget lookahead) → ranking
                                                  ↓
                                     LLM writes the explanation
```

The split matters:

| Layer | Does | Never does |
|---|---|---|
| **LLM** (Groq / Gemini) | Understands the request; explains the result | Picks products, sets prices, enforces budget |
| **Embeddings + FAISS** | Ranks valid candidates by meaning | Overrules a constraint |
| **Rules** | Constraints, compatibility, budget, assembly | — |

The LLM never sees the catalog, and its response schema has no field that could
hold a product id — so a hallucinated product can't reach the screen. Product ids
resolve through one function, `data.get_product()`.

Semantic retrieval runs *after* hard filtering, so it can only narrow an
already-legal pool. Retrieval is per outfit slot (a global top-K could empty a
slot), and the cheapest item in each slot always survives (otherwise narrowing
could break a budget that was satisfiable).

---

## Run it

```bash
pip install -r requirements.txt
python scripts/build_vector_index.py     # one-time, ~40s
streamlit run app.py                     # UI  → localhost:8501
uvicorn api:app --port 8000              # API → localhost:8000/docs
```

Works with no API key — requests fall back to a keyword parser and the UI says
which path answered. For the LLM path:

Create a `.env` file in the project root:

```ini
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

Providers are tried in order and each has its own circuit breaker, so a
rate-limited provider falls through instead of taking the LLM path down.
**Currently Groq (`llama-3.3-70b-versatile`) is active** — the Gemini project
returns 429.

```bash
pip install -r requirements-dev.txt
pytest                                   # 268 tests
python scripts/evaluate.py               # → docs/EVALUATION.md
```

---

## Data

[`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
— 44,072 products, sampled down to **536** with stratified quotas per
(gender × slot × occasion) so every slot has real candidates. Random sampling
would leave whole cells empty: the source is 77% "Casual" and has only 109
women's formal items.

Images are re-exported at 720×960 from a
[higher-resolution mirror](https://huggingface.co/datasets/benitomartin/fashion-product-images-small-900x1200)
of the same catalog (the canonical one ships 60×80 thumbnails) and served as
static files.

⚠️ **Prices are synthetic.** The dataset has no price column, and budget is
central to the product, so prices/MRPs/discounts are generated deterministically
from article type + a hash of the product id. The budget logic is real; the
numbers it operates on are simulated.

---

## Evaluation

34 hand-labelled scenarios, regenerate with `python scripts/evaluate.py`.

| | |
|---|---|
| Every product real (no hallucination) | 100% |
| Explicit item satisfied ("black shirt") | 100% |
| Constraints, completeness, budget | 100% |
| Graceful failure | 100% |
| Preference match (colour) | 0.965 |
| Catalog coverage across the run | 36.4% (195 of 536) |
| Latency p50 / p95 | 73 / 95 ms |

Latency is engine time only, excluding any LLM call, and is regenerated with
the table above — so it tracks whatever machine last ran `scripts/evaluate.py`
and moves by tens of milliseconds between runs. **~84% of it is the Sentence
Transformer forward pass**; `IndexFlatIP` search over 536 vectors is 0.3 ms.
Before semantic retrieval was added the p50 was ~4 ms — that is the price of the
feature, measured rather than estimated.

Catalog coverage is there deliberately: it is the check against a recommender
that always returns the same few items. It sat at 29.9% until the tie-breaking
and colour fixes in [D12–D14](docs/decision_log.md).

No Precision@K / NDCG / MAP — they need relevance labels, and none exist for this
catalog. Inventing them would produce a number that looks earned and isn't.

### Two experiments that didn't pay off

**Semantic vs TF-IDF: a tie.** 0.690 vs 0.680 attribute hit-rate over 10 probes.
The split is the interesting part — semantic wins on paraphrase ("business
meeting clothes" 0.50 vs 0.10), TF-IDF wins on literal keywords ("formal outfit"
0.30 vs 1.00). Overlap@10 is 0.22, so they rank very differently. Both are kept,
each used where it's strong.

**Random Forest compatibility model: built, measured, not shipped.** Trained on
raw product attributes (never on the rule scores — that would just re-learn the
formula) with an article-type group split. It generalised well to unseen garment
types (ROC-AUC 0.974), but on 65 pairs I labelled by hand *without* looking at
the rule scores it lost: F1 0.545 vs the rule baseline's 0.600 — and neither beat
a trivial always-compatible baseline (0.750). Predictable in hindsight: the
training labels came from the rules, so the rules are its ceiling. It stays as a
documented experiment, and a test asserts it can't leak into ranking.

---

## Known limitations

- **No personalisation.** No interaction data exists in the dataset, so no
  collaborative filtering and no learned user model.
- **Compatibility rules are hand-written, not learned** — and measurably
  imperfect: they agree with considered human judgement about 57% of the time.
- **Metadata only, no vision.** Two items labelled "Blue" can be very different
  blues; every print collapses to one `Multi` bucket.
- **536 products** means genuinely sparse cells, so tight constraints can fail.
  Formal requests are the worst case: ~20 tops and ~14 bottoms survive filtering,
  which is thin enough that most candidates score identically.
- **Semantic retrieval is the whole latency budget.** ~84% of a request, taking
  p50 from ~4 ms to tens of ms, for a measured *tie* against TF-IDF. It earns its
  place on free-text requests ("nothing too flashy") and on paraphrase, not on
  throughput.
- **Scores are coarse.** `preference_match` takes only 3–12 distinct values
  across a filtered pool, so ~10 of 29 candidates routinely tie. Tie-breaking is
  now request-derived rather than catalog-order, but the underlying coarseness
  is unfixed — the colour term is a constant whenever no colour is requested.
- The first request pays a one-off encoder load (~30 s cold, seconds when the
  model is already in the local Hugging Face cache); later ones are in the
  tens of milliseconds.

---

## Layout

```
app.py                  Streamlit UI
api.py                  FastAPI service over the same engine
src/
  recommender.py        pipeline: filter → retrieve → assemble → rank
  semantic_retriever.py MiniLM embeddings + FAISS
  compatibility.py      pairwise scoring, hard clashes
  outfit_builder.py     greedy assembly with budget lookahead
  features.py           derived features and domain tables
  schemas.py            Pydantic contracts (the LLM boundary)
  llm_client.py         Groq + Gemini, validation, fallback
  nlu.py                keyword parser (fallback + baseline)
  data.py               catalog; the only product-id resolution point
  similarity.py         TF-IDF similar products
  ui.py                 CSS, product cards, layout primitives
  ml/                   Random Forest experiment
scripts/                catalog build, index build, evaluation
docs/                   architecture, evaluation, assumptions, decisions
tests/                  268 tests
```

More detail: [architecture](docs/ARCHITECTURE.md) ·
[evaluation](docs/EVALUATION.md) · [ML experiment](docs/ml_evaluation.md) ·
[assumptions](docs/assumptions.md) · [decisions](docs/decision_log.md) ·
[marketplace comparison](docs/COMPARISON.md)

---

Independent prototype. Not affiliated with any marketplace; no third-party
branding or assets are used.
"# ai-stylist" 
