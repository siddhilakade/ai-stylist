"""AI Stylist - an explainable, GenAI-powered outfit recommendation prototype.

A demonstration of how an AI styling layer could sit on top of a fashion
marketplace. Independent prototype: not affiliated with, endorsed by, or built
from any real marketplace's assets, branding or data.

Run:  streamlit run app.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="AI Stylist — Outfit Intelligence",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load .env before anything reads GEMINI_API_KEY. The file is git-ignored and is
# only ever read into os.environ - never into the UI.
load_dotenv(Path(__file__).parent / ".env", override=False)

from src import llm_client, ui  # noqa: E402
from src.compatibility import SIGNAL_WEIGHTS, pair_compatibility  # noqa: E402
from src.ml import model as ml_model  # noqa: E402
from src.data import (  # noqa: E402
    CatalogError,
    catalog_records,
    catalog_summary,
    get_product,
    load_catalog,
    search_products,
)
from src.features import (  # noqa: E402
    ALL_SLOTS,
    COLOR_FAMILIES,
    DEFAULT_OCCASION,
    OCCASION_PROFILES,
    SLOT_LABELS,
    STYLE_TARGET_FORMALITY,
)
from src.recommender import (  # noqa: E402
    complete_the_look,
    grounding_payload,
    hard_filter,
    outfit_reasons,
    recommend_outfits,
    score_breakdown,
)
from src.schemas import (  # noqa: E402
    ExtractionResult,
    ExtractionSource,
    StylePreferences,
)
from src.similarity import similar_products  # noqa: E402

EXAMPLE_REQUESTS = [
    "College presentation tomorrow — smart casual, black and white, under ₹3000",
    "Formal outfit for an office meeting, navy tones, budget ₹8000",
    "Something ethnic for my cousin's wedding, around ₹6000",
    "Casual weekend brunch look for men under ₹4000",
]


# --------------------------------------------------------------------------
# Session state and navigation
# --------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "view": "home",
        "request_text": "",
        "prefs": None,
        "extraction": None,
        "result": None,
        "anchor_id": None,
        "wishlist": set(),
        "browse_gender": "Women",
        "browse_query": "",
        "use_llm": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def goto(view: str) -> None:
    # A nav click is a fresh start. Without this the guided finder kept its
    # step, so leaving it on step three and coming back through the nav dropped
    # you straight into "Budget and palette" with no visible way back.
    reset_finder()
    if view in ("men", "women", "kids"):
        st.session_state["browse_gender"] = view.capitalize()
        view = "browse" if view != "kids" else "kids"
    st.session_state["view"] = view
    st.rerun()


def goto_product(product_id: int) -> None:
    st.session_state["anchor_id"] = product_id
    st.session_state["view"] = "product"
    st.rerun()


def occasion_label(prefs: StylePreferences) -> str:
    return OCCASION_PROFILES.get(
        prefs.occasion, OCCASION_PROFILES[DEFAULT_OCCASION]
    )["label"]


# --------------------------------------------------------------------------
# Outfit rendering
# --------------------------------------------------------------------------

def render_outfit(outfit, prefs: StylePreferences, index: int, key_prefix: str) -> None:
    """One complete look: banner, garments, grounded reasons, stylist note."""
    ui.look(outfit, prefs, index, occasion_label(prefs))
    ui.look_shop_row(outfit, goto_product, key_prefix=f"{key_prefix}_{index}")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        ui.why_panel(outfit_reasons(outfit, prefs))
    with right:
        payload = grounding_payload(outfit, prefs)
        with st.spinner("Writing the stylist note…"):
            explanation = llm_client.explain_outfit(
                payload, allow_llm=st.session_state["use_llm"]
            )
        ui.stylist_note(explanation.text, explanation.source)

    # TECHNICAL LEVEL. Collapsed by default so the product surface stays clean,
    # but complete: every number the ranker used, and the exact LLM payload.
    with st.expander("Show scoring details"):
        breakdown_col, signal_col = st.columns(2)
        with breakdown_col:
            st.markdown("**Final score — transparent weighted sum**")
            st.dataframe(score_breakdown(outfit), hide_index=True,
                         use_container_width=True)
            st.caption(f"Final ranking score = **{outfit.final_score:.4f}**")
        with signal_col:
            st.markdown("**Compatibility signals** (averaged over every item pair)")
            st.dataframe(
                [
                    {"signal": label, "score": outfit.signals[key],
                     "weight": SIGNAL_WEIGHTS[key]}
                    for key, label in (
                        ("color", "Colour"), ("formality", "Formality"),
                        ("occasion", "Occasion"), ("ethnic", "Style coherence"),
                    )
                ],
                hide_index=True, use_container_width=True,
            )
            st.caption(
                f"Compatibility **{outfit.compatibility:.3f}** · "
                f"preference match **{outfit.preference_match:.3f}** · "
                f"budget fit **{outfit.budget_fit:.3f}**"
            )

        st.markdown("**How this recommendation was generated**")
        st.markdown(
            f"1. Hard filters removed everything that could not satisfy the "
            f"request (gender, explicit items, formality band, budget).\n"
            f"2. The most constrained slot was filled first "
            f"(`{outfit.anchor_slot}`), with a budget lookahead reserving enough "
            f"for every remaining slot.\n"
            f"3. Each further item maximised "
            f"`0.5 × compatibility + 0.5 × request fit`.\n"
            f"4. The finished outfit was scored by the weighted sum on the left.\n"
            f"5. The LLM was given **only** the payload below to narrate."
        )
        render_ml_experiment(outfit)

        st.markdown("**Exact payload sent to the LLM for the stylist note**")
        st.json(payload, expanded=False)
        st.caption(
            "The LLM receives only this. It never sees the catalog, so it cannot "
            "introduce a product, a price or an attribute the engine did not select."
        )
        if explanation.warning:
            st.caption(f"LLM call detail: `{explanation.warning}`")


def render_ml_experiment(outfit) -> None:
    """Show the trained model's opinion — clearly marked as not driving ranking.

    The model lost to the rule baseline on independently labelled pairs, so it is
    deliberately outside the recommendation path. Displaying its predictions next
    to the rule scores is the honest way to show the experiment: visible and
    inspectable, but with no influence on what was recommended.
    """
    if not ml_model.model_exists():
        return

    pairs = list(combinations(outfit.items.values(), 2))
    probabilities = [ml_model.safe_predict(a, b) for a, b in pairs]
    scored = [(a, b, p) for (a, b), p in zip(pairs, probabilities) if p is not None]
    if not scored:
        return

    st.markdown("**ML compatibility model — experimental, not used for ranking**")
    st.dataframe(
        [
            {
                "pair": f"{a['articleType']} + {b['articleType']}",
                "ML P(compatible)": round(probability, 3),
                "rule score": pair_compatibility(a, b),
            }
            for a, b, probability in scored
        ],
        hide_index=True, use_container_width=True,
    )
    st.caption(
        f"Random Forest trained on raw product attributes with **synthetic "
        f"(rule-generated) labels**. It generalises to held-out garment types "
        f"(ROC-AUC 0.974) but **did not beat the rule baseline** on independently "
        f"hand-labelled pairs (F1 0.545 vs 0.600), so it does not influence these "
        f"recommendations. Full write-up in `docs/ml_evaluation.md`."
    )


def render_failure(result, prefs: StylePreferences) -> None:
    failure = result.failure
    st.error(f"**{failure.message}**\n\n{failure.suggestion}")

    detail = failure.detail or {}
    if failure.reason_code == "budget_infeasible" and detail.get("minimum_cost"):
        covered = (prefs.budget or 0) / detail["minimum_cost"]
        st.progress(
            min(1.0, covered),
            text=f"Your budget covers {covered:.0%} of the cheapest complete "
                 "outfit for this request.",
        )

    counts = result.diagnostics.get("candidates_after_filter", {})
    if counts:
        with st.expander("What survived each hard filter"):
            st.dataframe(
                [{"slot": SLOT_LABELS.get(slot, slot), "candidates": count}
                 for slot, count in counts.items()],
                hide_index=True, use_container_width=False,
            )
            st.caption(
                "This is the honest reason for the failure: the constraint chain "
                "removed every candidate for at least one required slot. The system "
                "returns nothing rather than substituting unrelated products."
            )


def render_understanding(extraction) -> None:
    """What the system understood, and which path produced it."""
    prefs = extraction.preferences
    llm_label = "Understood by " + (extraction.provider or "LLM").title()
    if extraction.model:
        llm_label += f" · {extraction.model}"
    label, css = {
        ExtractionSource.GEMINI: (llm_label, "ais-badge-ok"),
        ExtractionSource.RULE_BASED: ("Built-in engine (no LLM)", "ais-badge-warn"),
        ExtractionSource.FORM: ("Set with form controls", "ais-badge-info"),
    }[extraction.source]

    st.markdown(f'<span class="ais-badge {css}">{label}</span>', unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].metric("For", prefs.gender)
    cols[1].metric("Occasion", occasion_label(prefs))
    cols[2].metric("Style", (prefs.style or "auto").replace("_", " ").title())
    cols[3].metric("Budget", ui.rupees(prefs.budget) if prefs.budget else "Any")
    cols[4].metric(
        "Colours",
        ", ".join(prefs.preferred_colors).title() if prefs.preferred_colors else "Any",
    )
    # Explicit garment requests are shown separately from the soft palette
    # preference, because the difference matters: one is enforced, one is scored.
    if prefs.required_items:
        must = " · ".join(item.describe() for item in prefs.required_items)
        st.markdown(
            f'<div style="margin-top:8px">'
            f'<span class="ais-badge ais-badge-pink">Must include</span>'
            f'<span style="font-weight:700">{must}</span>'
            f'<span style="color:#94969F;font-size:12.5px"> — enforced as a hard '
            f'filter, not a preference</span></div>',
            unsafe_allow_html=True,
        )
    if prefs.notes:
        st.caption(f"Also noted: _{prefs.notes}_")
    if extraction.warning:
        st.info(extraction.warning)


def run_recommendation(prefs: StylePreferences) -> None:
    with st.spinner("Assembling outfits…"):
        st.session_state["result"] = recommend_outfits(prefs)
    st.session_state["prefs"] = prefs


def render_results() -> None:
    """The shared results block used by Home and Style Me."""
    extraction = st.session_state.get("extraction")
    result = st.session_state.get("result")
    if not extraction or not result:
        return

    ui.section("What the system understood", kicker="Before anything was chosen")
    render_understanding(extraction)

    prefs = st.session_state["prefs"]
    for conflict in result.diagnostics.get("conflicts", []):
        st.warning(f"**Conflicting request** — {conflict}")
    for note in result.diagnostics.get("relaxed", []):
        st.info(note)

    if not result.ok:
        render_failure(result, prefs)
        return

    render_look_carousel(result, prefs, key_prefix="res")


def render_look_carousel(result, prefs: StylePreferences, key_prefix: str) -> None:
    """Show one look at a time, with a control to cycle to the next.

    Three full outfits stacked on one page is a wall of text. Showing the
    best-ranked look and letting the user ask for another keeps the page focused
    while still exposing every result the engine produced.
    """
    state_key = f"{key_prefix}_look_index"
    total = len(result.outfits)
    index = st.session_state.get(state_key, 0) % total

    ui.section(
        "Styled for you",
        f"Look {index + 1} of {total} · assembled in {result.latency_ms:.0f} ms",
        kicker="The result",
    )
    render_outfit(result.outfits[index], prefs, index, key_prefix=key_prefix)

    if total > 1:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        next_col, all_col, _ = st.columns([1.25, 1.25, 3.5])
        with next_col:
            if st.button("Show another look  →", key=f"{key_prefix}_next",
                         use_container_width=True):
                st.session_state[state_key] = index + 1
                st.rerun()
        with all_col:
            show_all = st.toggle("Compare all", key=f"{key_prefix}_all")
        if show_all:
            for other_index, outfit in enumerate(result.outfits):
                if other_index == index:
                    continue
                st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                render_outfit(outfit, prefs, other_index, key_prefix=f"{key_prefix}_alt")


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------

def use_example(example: str) -> None:
    """Load an example request into the input boxes and run it.

    This must be an `on_click` callback: Streamlit forbids writing a widget's
    session-state key after that widget has been instantiated in the current
    run, and callbacks execute before the next run builds its widgets.
    """
    st.session_state["request_text"] = example
    for prefix in ("home", "sm"):
        st.session_state[f"{prefix}_request"] = example
    st.session_state["run_now"] = True
    st.session_state["view"] = "style_me"


def style_me_input(key_prefix: str) -> None:
    """The hero natural-language input. The primary GenAI interaction."""
    # Seed through session state rather than `value=`. Passing both a default
    # AND writing the same key from `use_example` makes Streamlit render a
    # warning banner above the input on every run.
    widget_key = f"{key_prefix}_request"
    st.session_state.setdefault(widget_key, st.session_state["request_text"])
    text = st.text_area(
        "request",
        placeholder="I need a smart casual outfit for a college presentation "
                    "under ₹3000, preferably black and white…",
        height=86,
        label_visibility="collapsed",
        key=widget_key,
    )

    action, gender_col, _ = st.columns([1.15, 1.15, 3.2], vertical_alignment="bottom")
    with gender_col:
        default_gender = st.selectbox(
            "If unstated, assume", ["Women", "Men"], key=f"{key_prefix}_gender"
        )
    with action:
        # Deliberately NOT "Style me": that is the nav item's label, and two
        # controls with the same name on one page is ambiguous for users and
        # for accessibility tooling alike.
        go = st.button("Style my outfit", type="primary",
                       use_container_width=True, key=f"{key_prefix}_go")

    st.caption("Or try one of these:")
    example_cols = st.columns(len(EXAMPLE_REQUESTS), gap="small")
    for index, (col, example) in enumerate(zip(example_cols, EXAMPLE_REQUESTS)):
        with col:
            st.button(
                example,
                key=f"ex_{key_prefix}_{index}",
                use_container_width=True,
                on_click=use_example,
                args=(example,),
            )

    if go or st.session_state.pop("run_now", False):
        st.session_state["request_text"] = text or st.session_state["request_text"]
        with st.spinner("Understanding your request…"):
            extraction = llm_client.extract_preferences(
                st.session_state["request_text"],
                default_gender=default_gender,
                allow_llm=st.session_state["use_llm"],
            )
        st.session_state["extraction"] = extraction
        run_recommendation(extraction.preferences)
        if st.session_state["view"] != "style_me":
            st.session_state["view"] = "style_me"
            st.rerun()


def hero_collage_products() -> list:
    """Four colourful pieces, one per slot, for the hero composition."""
    picks, seen = [], set()
    for slot in ("onepiece", "top", "footwear", "bottom"):
        for product in catalog_records():
            if (product["outfit_slot"] == slot and not product["is_neutral"]
                    and product["color_family"] not in seen):
                picks.append(product)
                seen.add(product["color_family"])
                break
    return picks


def view_home() -> None:
    ui.hero(
        kicker="Outfits, not search results",
        headline_html="Dress for the <em>occasion</em>, not the algorithm.",
        body="A marketplace shows you products. This tells you what actually goes "
             "together — a complete outfit, inside your budget, with every reason "
             "it was chosen shown alongside it.",
        pills=[f"<b>{len(catalog_records())}</b> pieces styled",
               "Budget honoured, <b>always</b>",
               "Every choice <b>explained</b>"],
        collage=hero_collage_products(),
    )

    style_me_input("home")

    if st.session_state.get("result"):
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        render_results()
        return

    products = catalog_records()

    # Deliberately NOT filtered to neutrals. The previous build picked
    # `is_neutral` items here, so a fashion homepage opened on four grey
    # garments - the same monotony bias that was fixed in ranking (D13), left
    # sitting in the shop window.
    ui.section(
        "Start anywhere", "Pick a piece and the stylist builds the outfit around it",
        kicker="Shop by category",
    )
    showcase, seen_colours = [], set()
    for slot in ("top", "onepiece", "bottom", "footwear"):
        for product in products:
            colour = product["color_family"]
            if (product["outfit_slot"] == slot and not product["is_neutral"]
                    and colour not in seen_colours):
                showcase.append(product)
                seen_colours.add(colour)
                break
    ui.product_grid(showcase[:4], goto_product, columns=4, key_prefix="home_a")

    # Named for what the filter actually is. This rail was previously labelled
    # "Trending now" while selecting on discount_pct - there is no popularity
    # signal anywhere in this dataset to justify the word "trending".
    ui.section(
        "Biggest reductions in the catalog", kicker="On sale",
        subtitle="Ranked by discount — the dataset carries no popularity signal",
    )
    on_sale = sorted(products, key=lambda p: -int(p.get("discount_pct", 0)))[:4]
    ui.product_grid(on_sale, goto_product, columns=4, key_prefix="home_b")


FINDER_STEPS = ["Who it's for", "The occasion", "Budget & palette"]


def finder_matches(**overrides) -> tuple[int, int]:
    """(pieces surviving the hard filters, of those how many suit the palette).

    Shown live between steps: it lets a user see a request collapsing toward
    zero *before* they commit to it, which beats the most carefully worded
    failure message.

    Two numbers because one would mislead. Colour is deliberately a ranking
    signal and never a hard filter, so the survivor count is identical for
    "blue", "red" and no colour at all — which reads as a broken control. The
    second number is what the palette actually does: how much of the surviving
    pool either sits in a requested family or is a neutral that coordinates
    with it.
    """
    prefs = StylePreferences(**overrides)
    survivors = hard_filter(list(catalog_records()), prefs)
    if not prefs.preferred_colors:
        return len(survivors), 0
    in_palette = sum(
        1 for p in survivors
        if p["color_family"] in prefs.preferred_colors or p["is_neutral"]
    )
    return len(survivors), in_palette


def finder_signature() -> tuple:
    """Everything in the finder that changes the answer."""
    return (
        st.session_state.get("finder_gender"),
        st.session_state.get("finder_occasion"),
        st.session_state.get("f_budget"),
        tuple(st.session_state.get("f_colors") or ()),
        st.session_state.get("f_style"),
        st.session_state.get("f_acc"),
    )


def reset_finder() -> None:
    """Send the guided finder back to step one.

    Called from `goto`, because leaving mid-flow and returning through the nav
    used to resume on step three with no way back except the Back control.
    """
    st.session_state["finder_step"] = 0


def _finder_goto(step: int, **state) -> None:
    st.session_state.update(state)
    st.session_state["finder_step"] = step


def guided_finder() -> None:
    """Three tappable steps instead of a row of dropdowns.

    Buttons rather than selectboxes is a deliberate choice: dropdowns hide their
    options until opened, so users routinely miss them entirely.
    """
    step = int(st.session_state.get("finder_step", 0))
    ui.stepper(step, FINDER_STEPS)

    gender = st.session_state.get("finder_gender", "Women")
    occasion = st.session_state.get("finder_occasion", DEFAULT_OCCASION)

    if step == 0:
        ui.section("Who are you styling?", kicker="Step one")
        cols = st.columns(3, gap="small")
        for col, choice in zip(cols, ["Women", "Men", "Unisex"]):
            with col:
                st.button(
                    choice, key=f"step_who_{choice}", use_container_width=True,
                    on_click=_finder_goto, args=(1,),
                    kwargs={"finder_gender": choice},
                )
        ui.match_count(len(catalog_records()), "pieces in the catalog",
                       "Narrowed at each step, then assembled into complete looks")
        return

    if step == 1:
        ui.section("What's the occasion?", kicker="Step two")
        keys = list(OCCASION_PROFILES)
        for start in range(0, len(keys), 3):
            cols = st.columns(3, gap="small")
            for col, key in zip(cols, keys[start:start + 3]):
                with col:
                    st.button(
                        OCCASION_PROFILES[key]["label"],
                        key=f"step_occ_{key}", use_container_width=True,
                        on_click=_finder_goto, args=(2,),
                        kwargs={"finder_occasion": key},
                    )
        survivors, _ = finder_matches(gender=gender)
        ui.match_count(survivors, f"pieces fit {gender.lower()}")
        st.button("← Back", key="finder_back_1", on_click=_finder_goto, args=(0,))
        return

    ui.section("Budget and palette", kicker="Step three")
    col1, col2, col3 = st.columns(3)
    budget = col1.slider("Budget (₹)", 500, 20000, 5000, step=250, key="f_budget")
    colours = col2.multiselect(
        "Preferred colours", list(COLOR_FAMILIES), format_func=str.title,
        max_selections=3, key="f_colors",
    )
    style = col3.selectbox(
        "Style", ["auto", *STYLE_TARGET_FORMALITY],
        format_func=lambda s: "Let the occasion decide" if s == "auto"
        else s.replace("_", " ").title(),
        key="f_style",
    )
    accessory = st.checkbox("Include an accessory", value=True, key="f_acc")

    resolved_style = None if style == "auto" else style
    survivors, in_palette = finder_matches(
        gender=gender, occasion=occasion, style=resolved_style,
        budget=budget, preferred_colors=colours,
    )
    if colours:
        detail = (
            f"{in_palette} of them sit in your palette or are coordinating "
            "neutrals — colour guides the ranking, it never rules a piece out"
        )
    else:
        detail = "Budget removes single pieces priced above the whole outfit"
    ui.match_count(survivors, "pieces to build from", detail)

    # Filters changed since the last build, so anything still on screen below is
    # stale. Clearing it is better than showing outfits that no longer match the
    # controls the user is looking at.
    if st.session_state.get("finder_built") not in (None, finder_signature()):
        st.session_state["result"] = None
        st.session_state["extraction"] = None
        st.session_state["finder_built"] = None

    go_col, back_col, _ = st.columns([1.4, 1, 3])
    with go_col:
        go = st.button("Build my outfits", type="primary", key="f_go",
                       use_container_width=True)
    with back_col:
        st.button("← Back", key="finder_back_2", on_click=_finder_goto, args=(1,))

    if go:
        prefs = StylePreferences(
            gender=gender, occasion=occasion, style=resolved_style,
            budget=budget, preferred_colors=colours, include_accessory=accessory,
        )
        st.session_state["extraction"] = ExtractionResult(
            preferences=prefs, source=ExtractionSource.FORM
        )
        st.session_state["finder_built"] = finder_signature()
        run_recommendation(prefs)


def view_style_me() -> None:
    tab_ai, tab_form = st.tabs(["Describe it in your own words", "Guided finder"])

    with tab_ai:
        st.markdown(
            '<div class="ais-hero" style="padding:34px 36px 30px">'
            '<div class="ais-kicker">Tell us what you want to wear</div>'
            "<h1 style='font-size:38px'>Say it however you'd say it "
            "to a <em>friend</em>.</h1>"
            "<p>A language model reads your sentence and turns it into structured "
            "preferences. The outfit itself is chosen by a deterministic engine — "
            "so the same request always returns the same look.</p></div>",
            unsafe_allow_html=True,
        )
        style_me_input("sm")

    with tab_form:
        guided_finder()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    render_results()


def view_browse(gender: str) -> None:
    ui.section(f"{gender}'s catalog", "Filter, then pick anything to style around",
               kicker="Browse")

    col1, col2, col3 = st.columns([2, 1.2, 1.2])
    query = col1.text_input("Search this catalog", key="browse_q",
                            value=st.session_state.get("browse_query", ""),
                            placeholder="e.g. black shirt")
    slot = col2.selectbox(
        "Category", ["All", *ALL_SLOTS],
        format_func=lambda s: "All categories" if s == "All" else SLOT_LABELS[s],
        key="browse_slot_sel",
    )
    max_price = col3.slider("Max price (₹)", 250, 7500, 7500, step=250, key="browse_price")

    products = search_products(query=query, gender=gender, slot=slot,
                               max_price=max_price, limit=48)
    st.caption(f"{len(products)} products")
    ui.product_grid(products, goto_product, columns=4, key_prefix=f"browse_{gender}")


def view_search() -> None:
    query = st.session_state.get("browse_query", "")
    ui.section(f"Search results for “{query}”")
    products = search_products(query=query, limit=48)
    st.caption(f"{len(products)} products")
    if not products:
        st.info("Nothing matched. Try a colour, a garment type, or a brand.")
    ui.product_grid(products, goto_product, columns=4, key_prefix="search")


def view_placeholder(title: str, headline: str, body: str) -> None:
    """An intentional 'not in this prototype' page, not an error page."""
    ui.section(title)
    st.markdown(
        f"""
        <div class="ais-hero" style="padding:30px 34px">
          <div class="ais-kicker">Not in this prototype</div>
          <h1 style="font-size:24px">{headline}</h1>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Explore what the stylist does cover:**")
    a, b, c, _ = st.columns([1, 1, 1.3, 2.2])
    with a:
        if st.button("Shop Men", use_container_width=True, key="ph_men"):
            goto("men")
    with b:
        if st.button("Shop Women", use_container_width=True, key="ph_women"):
            goto("women")
    with c:
        if st.button("Style me", type="primary", use_container_width=True,
                     key="ph_style"):
            goto("style_me")


def view_product() -> None:
    product = get_product(st.session_state.get("anchor_id"))
    if product is None:
        st.warning("Product not found.")
        return

    back_col, _ = st.columns([1, 6])
    with back_col:
        if st.button("←  Back", use_container_width=True):
            goto(product["gender"].lower() if product["gender"] != "Unisex" else "women")

    image_col, detail_col = st.columns([1, 1.35], gap="large")
    with image_col:
        st.markdown(
            f'<div class="ais-pdp-img"><img src="{ui.product_image_uri(product)}"/></div>',
            unsafe_allow_html=True,
        )

    with detail_col:
        price = int(product["price"])
        mrp = int(product.get("mrp", price))
        discount = int(product.get("discount_pct", 0))
        price_html = f"<b>{ui.rupees(price)}</b>"
        if discount > 0 and mrp > price:
            price_html += (
                f'<span class="ais-mrp" style="font-size:15px">{ui.rupees(mrp)}</span>'
                f'<span class="ais-off" style="font-size:14px">({discount}% OFF)</span>'
            )

        st.markdown(
            f'<div class="ais-pdp-brand">{ui.brand_of(product)}</div>'
            f'<div class="ais-pdp-name">{ui.short_name_of(product)}</div>'
            f'<div class="ais-pdp-price">{price_html}</div>'
            '<div class="ais-note" style="margin:3px 0 6px 0">'
            "Prices in this prototype are simulated — see “How it works”.</div>",
            unsafe_allow_html=True,
        )

        # Shopper-facing, so it shows the derived attributes but not the raw
        # numbers behind them: "Casual · 0.75/4" leaked the internal formality
        # float onto a product page. The full scoring detail stays in the
        # expander under each recommendation, where it belongs.
        colour = str(product["baseColour"])
        family = str(product["color_family"])
        facts = {
            "Category": SLOT_LABELS[product["outfit_slot"]],
            "Type": product["articleType"],
            "Colour": colour if colour.lower() == family.lower()
                      else f"{colour} · {family} family",
            "Style": product["formality_tier"],
            "Occasion tag": product["usage"],
            "Gender": product["gender"],
        }
        # A single full-width specification table rather than two stacked
        # columns of definition lists: every derived feature the engine actually
        # reasons with, in one scannable block.
        rows = "".join(
            f'<div class="row"><dt>{label}</dt><dd>{value}</dd></div>'
            for label, value in facts.items()
        )
        st.markdown(f'<dl class="ais-attr">{rows}</dl>', unsafe_allow_html=True)

        action1, action2 = st.columns(2)
        wishlisted = int(product["id"]) in st.session_state["wishlist"]
        if action1.button("Build the complete outfit", type="primary",
                          use_container_width=True):
            suggested = int(round(product["price"] * 3.5 / 500) * 500)
            st.session_state["ctl_budget"] = max(1000, min(20000, suggested))
            st.rerun()
        if action2.button("♥  Wishlisted" if wishlisted else "♡  Wishlist",
                          use_container_width=True):
            st.session_state["wishlist"] ^= {int(product["id"])}
            st.rerun()

    ui.section("Complete the look", "Pieces the engine selected to work with this item",
               kicker="Styled around it")
    st.caption(
        "This product is **pinned** as the only candidate for its slot, so it can "
        "never be swapped out. Everything else runs through the same filtering, "
        "scoring and budget logic as Style Me."
    )

    budget_col, _ = st.columns([1.4, 3])
    with budget_col:
        budget = st.slider("Outfit budget (₹)", 1000, 20000, 7000, step=500,
                           key="ctl_budget")

    with st.spinner("Finding pieces that work with this…"):
        result = complete_the_look(int(product["id"]), budget=budget)

    for note in result.diagnostics.get("relaxed", []):
        st.info(note)

    if not result.ok:
        render_failure(result, StylePreferences(budget=budget))
    else:
        prefs = StylePreferences(
            gender=product["gender"] if product["gender"] != "Unisex" else "Women",
            budget=budget,
        )
        render_look_carousel(result, prefs, key_prefix="ctl")

    ui.section("Similar products", "TF-IDF over product metadata — a separate feature",
               kicker="More like this")
    st.caption(
        "Similarity answers “what looks like this?”. Complete the Look answers "
        "“what goes **with** this?”. They are deliberately different features."
    )
    ui.product_grid(similar_products(int(product["id"]), top_k=4), goto_product,
                    columns=4, key_prefix="sim", action_label="View  →")


def view_about() -> None:
    ui.section("How it works")
    st.markdown(
        """
The language model handles language. A deterministic engine handles selection.
Neither does the other's job.

```
  free text ──►  LLM   ──► StylePreferences ──► hard filters ──► candidates
   (user)      (Task A)     validated,           gender            per slot
                            closed vocab         formality             │
                                                 ethnic                ▼
                                                 budget         greedy assembly
                                                                (budget lookahead)
                                                                       │
   stylist note ◄──  LLM   ◄── grounding payload ◄── ranking ◄─────────┘
      (shown)      (Task B)    real items + real     weighted sum
                               computed scores
```
        """
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### What the language model does")
        st.markdown(
            "- **Task A — understanding.** Free text → a closed-vocabulary\n"
            "  `StylePreferences` object via structured output. An invalid\n"
            "  response is rejected and the keyword parser is used instead.\n"
            "- **Task B — explanation.** Given the products the engine *already*\n"
            "  chose plus the real scores, it writes one paragraph."
        )
    with col2:
        st.markdown("#### What it never does")
        st.markdown(
            "- Choose products, or see the catalog at all\n"
            "- Decide compatibility, or set any score\n"
            "- Produce a price, a product name or a product id\n"
            "- Enforce the budget\n\n"
            "Product ids resolve **only** through `data.get_product()`, so a "
            "hallucinated product cannot reach the screen."
        )

    ui.section("Language model status")
    has_key = llm_client.has_api_key()
    tripped = llm_client.circuit_is_open()
    status_col, toggle_col, test_col = st.columns([2, 1, 1], vertical_alignment="bottom")
    with status_col:
        if tripped:
            st.error(
                "**API key configured, but no provider is reachable — running on the "
                "fallback path.**\n\n"
                f"Last error: `{llm_client.circuit_error()}`\n\n"
                "After two consecutive failures the client stops retrying, so a "
                "dead dependency costs one failed call rather than one per outfit. "
                "Everything else still works: requests are parsed with keyword "
                "rules and explanations are generated from the engine's own signals."
            )
        elif has_key:
            # Name the provider that will ACTUALLY answer, not just the first one
            # holding a key. With both configured this panel used to report
            # Gemini while every request was in fact being served by Groq.
            active = llm_client.active_provider()
            configured = llm_client.configured_providers()
            others = [p for p in configured if p != active]
            detail = f" · fallback: {', '.join(others)}" if others else ""
            st.success(
                f"Active provider: **{active}** · model "
                f"`{llm_client.provider_model(active)}`{detail}"
            )
        else:
            st.warning(
                "No `GROQ_API_KEY` or `GEMINI_API_KEY` found. The app runs fully — "
                "requests fall back to a keyword parser and explanations come from "
                "the engine's own signals. Add a key to `.env` for the full GenAI path."
            )
    with toggle_col:
        st.session_state["use_llm"] = st.toggle(
            "Use the language model", value=st.session_state["use_llm"] and has_key,
            disabled=not has_key,
            help="Turn off to compare the LLM path against the rule-based fallback.",
        )
    with test_col:
        if has_key and st.button("Test connection", use_container_width=True):
            llm_client.reset_circuit()
            with st.spinner("Calling the provider…"):
                status = llm_client.check_connection()
            if status["ok"]:
                st.success(f"{status['latency_ms']} ms")
            else:
                st.error(status.get("detail", "Unknown error"))

    ui.section("The catalog")
    summary = catalog_summary()
    cols = st.columns(5)
    cols[0].metric("Products", summary["products"])
    cols[1].metric("Article types", summary["article_types"])
    cols[2].metric("Colours", summary["colors"])
    cols[3].metric("Price range", f"₹{summary['price_min']}–{summary['price_max']:,}")
    cols[4].metric("Slots covered", len(summary["slots"]))
    st.markdown(
        f"""
Sampled deterministically from **`ashraq/fashion-product-images-small`**
(44,072 products) down to **{summary['products']}**, with stratified quotas per
(gender × slot × occasion group) so every slot a request can ask for has real
candidates. See `scripts/build_catalog.py`.

**Honest caveat on prices.** The source dataset has no price or MRP column.
Prices, discounts and struck-through MRPs here are synthetic — derived
deterministically from article type, occasion tag and a hash of the product id.
They are realistic in *distribution* and perfectly reproducible, but they are not
real prices. The budget logic is real; the numbers it operates on are simulated.
        """
    )

    ui.section("Known limitations")
    st.markdown(
        """
- **No personalisation from history.** No interaction data exists in the dataset,
  so there is no collaborative filtering and no learned user model. Everything is
  driven by explicitly stated preferences.
- **Compatibility rules are hand-authored, not learned.** There is no outfit
  compatibility ground truth to train on, so a learned model would be
  unverifiable. The trade-off: the rules encode conventional styling advice and
  will miss deliberate, fashion-forward clashing.
- **Metadata only, no vision.** Compatibility uses `baseColour` and category
  labels, not pixels. Two items labelled "Blue" may be very different blues.
- **Metadata-only imagery.** Product shots are flat catalog images at 720x960,
  displayed at a capped size rather than blown up, but they are not studio
  photography.
- **Small catalog.** 536 products means genuinely sparse cells, so tight
  constraints can legitimately fail — by design, with an explanation.
        """
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

VIEWS = {
    "home": view_home,
    "style_me": view_style_me,
    "browse": lambda: view_browse(st.session_state["browse_gender"]),
    "search": view_search,
    "product": view_product,
    "about": view_about,
    "kids": lambda: view_placeholder(
        "Kids",
        "Kids styling is not available in this prototype",
        "The source dataset does contain Boys' and Girls' products, but they were "
        "deliberately excluded from the demo catalog: children's sizing and "
        "styling rules differ enough from adult wear that the compatibility model "
        "would be making claims it cannot support. Showing the boundary is more "
        "useful than silently returning adult clothing.",
    ),
    "beauty": lambda: view_placeholder(
        "Beauty",
        "Beauty is outside this stylist's scope",
        "The source dataset does include personal care products — lipsticks, "
        "fragrances, nail polish — but they were excluded from the catalog "
        "because they cannot participate in outfit compatibility. There is no "
        "meaningful “goes with” relationship between a lipstick and a pair of "
        "trousers in this metadata, so the engine would have nothing honest to say.",
    ),
}


def main() -> None:
    init_state()
    ui.inject_css()

    try:
        load_catalog()
    except CatalogError as exc:
        st.error(
            f"**Catalog could not be loaded.**\n\n{exc}\n\n"
            "Run `python scripts/build_catalog.py` from the project root."
        )
        st.stop()

    ui.header(goto, wishlist_count=len(st.session_state["wishlist"]))
    VIEWS.get(st.session_state["view"], view_home)()
    ui.footer(goto)


if __name__ == "__main__":
    main()
