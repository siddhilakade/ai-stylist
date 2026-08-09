# Benchmark: AI Stylist vs. how real fashion marketplaces do this

*Addresses the assignment's bonus challenge: benchmark the implementation against
the product that inspired it.*

AI Stylist is modelled on the shopping experience of large Indian fashion
marketplaces — Myntra, Ajio, Nykaa Fashion — which share a recognisable pattern:
a dense white product grid, category-led navigation, a bold call-to-action, and
per-product recommendation strips ("Similar Products", "Customers also viewed",
"Complete the Look").

**This is an independent prototype.** No third-party logo, asset, trademark or
product data is used. The brand name, visual design and all code are original.
The comparison below is about *product and technical decisions*, not appearance.

---

## Similarities — what was deliberately mirrored

| Pattern | Why it was worth copying |
|---|---|
| **White canvas, dense grid, image-first cards** | Fashion is a visual purchase. The proven layout gives the product photo ~70% of the card and pushes text down. |
| **Brand line above description line** | Indian marketplaces lead with brand because it is the strongest purchase signal. The dataset packs both into one string; the card splits them. |
| **Bold single CTA in a saturated accent** | One unambiguous next action per card. |
| **Category-led top navigation** (Men / Women / Kids) | Gender is the first filter in almost every fashion session — it belongs in the chrome, not in a dropdown. |
| **Product detail with attribute table + recommendation strip** | Matches where users already expect "Complete the Look" to live. |
| **Search + facet filters (category, price)** | Table stakes for a catalog browser. |
| **"Complete the Look" as a named feature** | Myntra ships this. It is the closest real-world analogue to what this project is about, which makes it the fairest benchmark. |

---

## Differences — where this project diverges, and why

### 1. Natural language is the primary input, not a search box

Marketplaces take **keyword queries** ("black formal shirt") and return **ranked
items**. AI Stylist takes an **intent** ("I have a presentation tomorrow, smart
casual, under ₹3000") and returns **complete outfits**.

That is a different product. Keyword search cannot express a budget for a *whole
outfit*, an occasion, or a formality level simultaneously — which is exactly the
gap this fills.

### 2. Recommendations are explained, not asserted

A marketplace strip says "Customers also viewed" or nothing at all. Every outfit
here shows:

- the factual bullet points behind it (formality consistency, colour
  coordination, occasion suitability, budget headroom),
- a natural-language stylist note **generated from those same numbers**,
- and an expandable panel with the score breakdown, the four compatibility
  signals, and **the exact JSON payload sent to the LLM**.

Real platforms cannot do this — their recommendations come from learned models
over behavioural data, which are not decomposable into human-readable reasons.
This is a genuine advantage of the rule-based approach, and it is most of why
the approach was chosen.

### 3. Compatibility, not similarity

The industry default is item-to-item collaborative filtering and embedding
similarity. Both answer *"what is like this?"* This project answers *"what goes
**with** this?"* — a cross-category compatibility problem.

TF-IDF similarity is included on the product page as a clearly-labelled,
deliberately-separate feature, precisely to make the distinction visible.

### 4. Budget is a hard constraint on the outfit total

Marketplace price filters are **per item**. Nothing in a typical fashion site
lets you say "₹3,000 for the whole outfit" — yet that is how people actually
shop. Here the budget is enforced across the assembled outfit, with lookahead so
one slot cannot consume it.

### 5. Failure is a designed, explained outcome

Marketplaces almost never return nothing; they broaden the query silently. This
system will return **zero** outfits and explain exactly why — which slot emptied,
what the cheapest feasible outfit would cost, what to relax.

Commercially that is the wrong default (empty pages do not convert). For an
evaluable prototype it is the right one: an honest failure is more informative
than a plausible-looking wrong answer, and silently substituting unrelated
products is the failure mode this architecture exists to prevent.

---

## Where real platforms are decisively better

Stated plainly, because pretending otherwise would be the least credible part of
this document.

| Capability | Them | Here |
|---|---|---|
| **Behavioural personalisation** | Millions of sessions of implicit feedback; recommendations personalised to *you* | No interaction data exists in the dataset. Zero personalisation from history — only explicitly stated preferences. |
| **Catalog scale** | Millions of live SKUs | 536 products. Sparse cells genuinely fail. |
| **Real inventory, price, sizing** | Live stock, real prices, size availability, returns data | Synthetic prices; no stock or sizing at all. |
| **Visual understanding** | CNN/ViT embeddings over real product photography; understands print, texture, cut, fit | Metadata only. Every print collapses into one `Multi` bucket. |
| **Learned compatibility** | Trained on co-purchase and co-view signals — learns that *these* trousers sell with *that* shirt | Hand-authored conventional styling rules. Will miss anything fashion-forward. |
| **Trend and seasonality** | Real-time trend and demand signals | A static `season` column, currently unused for ranking. |
| **A/B infrastructure** | Continuous online experimentation on live conversion | Offline constraint-satisfaction metrics on 34 scenarios. |
| **Cold start at scale** | Content-based bootstrapping into behavioural models | Not applicable — everything is content-based. |

**The honest summary:** on *recommendation power* a real platform wins
overwhelmingly, because it has data this project does not. On *explainability,
constraint satisfaction and outfit-level reasoning* this prototype does something
their production systems mostly do not attempt.

---

## Where this approach would genuinely add value in production

Not as a replacement for a learned recommender — as a **layer on top of one**.

1. **Constraint satisfaction over a learned candidate set.** Let the behavioural
   model produce candidates; let a deterministic assembler enforce budget,
   occasion and formality. Learned models are poor at hard constraints, and a
   budget violation is a trust failure, not a ranking error.
2. **Explanations users will believe.** "These work together because the
   formality matches and the colours are neutral" outperforms "customers also
   viewed" for a decision the shopper is unsure about — and it is exactly what a
   learned model cannot produce.
3. **Cold-start coverage.** Rules work on day one for a new SKU with no
   interaction history, where collaborative filtering has nothing.
4. **Intent capture.** The structured-extraction step is independently useful:
   turning "something for my sister's engagement under 6k" into filters is
   valuable even feeding a conventional search backend.
5. **A safe LLM surface.** The grounding pattern — the model never selects,
   only narrates what a deterministic system selected — generalises to any
   recommendation surface where hallucinated products would be a liability.

---

## Concrete next steps to close the gap

In the order I would actually do them:

1. **Conversational refinement** — "make it cheaper", "swap the shoes", "keep the
   shirt". The engine already supports pinning any item (Complete the Look uses
   it); this is UI and state management, and it is the single biggest UX gap.
2. **CLIP embeddings** for visual compatibility, replacing the coarse
   `baseColour` signal, plus "upload a photo of something you own".
3. **Learn the weights** from implicit feedback once interaction data exists —
   which suggested outfits get items added to bag. The scoring function is
   already a linear model; fitting its seven weights is a small, well-posed
   problem, and keeps the system explainable.
4. **Real catalog integration** — live prices, stock and sizing, with the
   synthetic price layer removed.
5. **Online evaluation** — the offline metrics here measure constraint
   satisfaction, which is necessary but not sufficient. The real question is
   whether people wear the outfits, and only an A/B test answers that.
