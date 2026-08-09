"""Presentation layer: theme, navigation, product cards and outfit cards.

Kept strictly separate from the engine - nothing in this file makes a styling
decision, and nothing in `recommender.py` and friends knows Streamlit exists.
That separation is what lets the whole engine be tested and evaluated headlessly.

The visual language deliberately follows the conventions of Indian fashion
e-commerce: white canvas, dense four-up product grid, brand line above a muted
description line, price with struck-through MRP and a discount call-out, and a
pink primary action. It is an original design under an original brand name - no
third-party logo, wordmark, illustration or asset is reproduced.
"""

from __future__ import annotations

from html import escape
from typing import Any, Mapping

import streamlit as st

from src.data import image_url
from src.features import SLOT_LABELS, extract_short_name

Product = Mapping[str, Any]

# --- palette ---------------------------------------------------------------
PINK = "#FF3F6C"          # primary action / brand accent
INK = "#282C3F"           # headings and prices
BODY = "#535766"          # descriptions
MUTED = "#94969F"         # metadata
LINE = "#EAEAEC"          # hairlines
TILE = "#F5F5F6"          # image tile / chips
DISCOUNT = "#FF905A"      # discount call-out

# Product images are stored at 384x512 (see scripts/upgrade_images.py). Display
# Product images are stored at 720px wide (scripts/upgrade_images.py). These are
# CSS pixels, deliberately kept at half the stored width or less, so that even on
# a 2x HiDPI display the browser is scaling DOWN rather than up.
CARD_IMAGE_WIDTH = 284
DETAIL_IMAGE_WIDTH = 360


def rupees(amount: float | int) -> str:
    return f"₹{int(round(amount)):,}"


def product_image_uri(product: Product) -> str:
    """Static URL for a product image (served by Streamlit from static/)."""
    return image_url(product)


def brand_of(product: Product) -> str:
    return str(product.get("brand") or str(product["productDisplayName"]).split()[0])


def short_name_of(product: Product) -> str:
    return extract_short_name(str(product["productDisplayName"]), brand_of(product))


# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* Streamlit strips <link> tags from markdown, so the webfont is imported
           from inside the stylesheet. If it cannot load (offline), the system
           sans-serif stack below takes over with no visual breakage. */
        @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;600;700;800&display=swap');

        :root {{
            --pink: {PINK}; --ink: {INK}; --body: {BODY}; --muted: {MUTED};
            --line: {LINE}; --tile: {TILE}; --discount: {DISCOUNT};
        }}

        /* ---------- reset Streamlit chrome ---------- */
        [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stStatusWidget"], [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"], #MainMenu, footer {{
            display: none !important;
        }}
        header[data-testid="stHeader"] {{ height: 0; background: transparent; }}
        .stApp {{ background: #FFFFFF; }}
        .block-container {{
            max-width: 1400px; padding: 0.4rem 1.5rem 3.5rem 1.5rem;
        }}
        [data-testid="stVerticalBlock"] {{ gap: 0.35rem; }}
        [data-testid="stHorizontalBlock"] {{ gap: 0.75rem; }}
        hr {{ margin: 0.9rem 0; border-color: var(--line); }}

        html, body, [class*="css"], .stMarkdown, button, input, textarea, select {{
            font-family: "Assistant", -apple-system, BlinkMacSystemFont,
                         "Segoe UI", Roboto, Arial, sans-serif;
            color: var(--ink);
        }}

        /* ---------- header ---------- */
        .ais-logo {{
            display: flex; align-items: center; gap: 9px; padding: 4px 0 0 0;
        }}
        .ais-mark {{
            width: 38px; height: 38px; border-radius: 11px; flex: none;
            background: linear-gradient(135deg, var(--pink) 0%, #FF8A5B 100%);
            color: #FFF; font-weight: 800; font-size: 19px; line-height: 38px;
            text-align: center; letter-spacing: -0.5px;
        }}
        .ais-wordmark {{
            font-size: 17px; font-weight: 800; letter-spacing: -0.4px;
            line-height: 1.05; text-transform: uppercase;
        }}
        .ais-wordmark small {{
            display: block; font-size: 8.5px; font-weight: 700;
            letter-spacing: 1.7px; color: var(--muted); margin-top: 2px;
            white-space: nowrap;
        }}
        [class*="st-key-nav_"] button p {{ white-space: nowrap; }}
        .ais-usernav {{
            display: flex; gap: 26px; justify-content: flex-end;
            align-items: center; padding-top: 6px;
        }}
        .ais-usernav div {{
            text-align: center; font-size: 11px; font-weight: 700;
            color: var(--ink); line-height: 1.15; cursor: default;
        }}
        .ais-usernav span {{ display: block; font-size: 17px; line-height: 1.25; }}
        .ais-usernav b {{
            display: inline-block; background: var(--pink); color: #FFF;
            font-size: 9px; border-radius: 8px; padding: 0 5px; margin-left: 3px;
        }}
        .ais-headrule {{ height: 1px; background: var(--line); margin: 4px 0 14px 0; }}

        /* ---------- buttons ----------
           Streamlit stamps a `st-key-<key>` class on each widget's container,
           which is what lets these rules target one role of button without
           restyling every button in the app. */

        /* nav -> text links with a pink underline on hover */
        [class*="st-key-nav_"] button {{
            border: none !important; background: transparent !important;
            border-radius: 0 !important; border-bottom: 3px solid transparent !important;
            font-size: 13px; font-weight: 700; letter-spacing: .2px;
            text-transform: uppercase; color: var(--ink) !important;
            min-height: 40px; padding: 0 2px !important;
        }}
        [class*="st-key-nav_"] button:hover {{
            color: var(--pink) !important; border-bottom-color: var(--pink) !important;
        }}
        [class*="st-key-nav_"] button p {{ font-size: 13px; font-weight: 700; }}

        /* product-card action -> a quiet accent link glued under the card */
        [class*="st-key-pc_"] {{ margin-top: -6px; }}
        [class*="st-key-pc_"] button {{
            border: none !important; background: transparent !important;
            color: var(--pink) !important; font-size: 11px; font-weight: 800;
            letter-spacing: .7px; text-transform: uppercase; min-height: 30px;
            justify-content: flex-start !important; padding-left: 9px !important;
        }}
        [class*="st-key-pc_"] button p {{ font-size: 11px; font-weight: 800; }}
        [class*="st-key-pc_"] button:hover {{ color: #E12E58 !important; }}

        /* example chips -> soft outlined suggestions */
        [class*="st-key-ex_"] button {{
            border: 1px solid var(--line); background: #FFF;
            color: var(--body) !important; font-weight: 600; font-size: 12px;
            letter-spacing: 0; text-transform: none; min-height: 46px;
            line-height: 1.35; padding: 6px 10px;
        }}
        [class*="st-key-ex_"] button:hover {{
            border-color: var(--pink); color: var(--pink) !important;
            background: #FFF7F9;
        }}
        [class*="st-key-ex_"] button p {{ font-size: 12px; font-weight: 600; }}

        /* ---------- buttons (default) ---------- */
        .stButton > button {{
            border-radius: 4px; border: 1px solid #D4D5D9; background: #FFF;
            color: var(--ink); font-weight: 700; font-size: 12px;
            letter-spacing: .4px; padding: 0.36rem 0.7rem; min-height: 34px;
            transition: all .15s ease;
        }}
        .stButton > button:hover {{
            border-color: var(--pink); color: var(--pink); background: #FFF;
        }}
        .stButton > button:focus:not(:active) {{ color: var(--pink); border-color: var(--pink); }}
        .stButton > button[kind="primary"] {{
            background: var(--pink); border-color: var(--pink); color: #FFF;
            text-transform: uppercase; letter-spacing: .8px; font-weight: 800;
            min-height: 42px; font-size: 13px;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: #E12E58; border-color: #E12E58; color: #FFF;
        }}

        /* ---------- inputs ---------- */
        .stTextInput input, .stTextArea textarea {{
            background: var(--tile); border: 1px solid var(--tile);
            border-radius: 4px; font-size: 14px; color: var(--ink);
        }}
        .stTextInput input:focus, .stTextArea textarea:focus {{
            background: #FFF; border-color: var(--pink); box-shadow: none;
        }}
        .stTextArea textarea::placeholder, .stTextInput input::placeholder {{
            color: var(--muted);
        }}

        /* ---------- hero ---------- */
        .ais-hero {{
            position: relative; overflow: hidden; border-radius: 10px;
            background: linear-gradient(115deg, #FFF0F4 0%, #FFF6F0 48%, #F1F4FF 100%);
            border: 1px solid #F6E3E9; padding: 26px 30px 22px 30px;
            margin-bottom: 14px;
        }}
        .ais-kicker {{
            display: inline-block; background: var(--pink); color: #FFF;
            font-size: 10px; font-weight: 800; letter-spacing: 1.5px;
            padding: 4px 10px; border-radius: 3px; margin-bottom: 10px;
        }}
        .ais-hero h1 {{
            font-size: 30px; font-weight: 800; letter-spacing: -0.8px;
            margin: 0 0 6px 0; line-height: 1.15; color: var(--ink);
        }}
        .ais-hero p {{ font-size: 14.5px; color: var(--body); margin: 0; }}

        /* ---------- product card ---------- */
        .ais-card {{
            border: 1px solid transparent; border-radius: 4px; background: #FFF;
            overflow: hidden; transition: box-shadow .18s ease, border-color .18s ease;
        }}
        .ais-card:hover {{
            box-shadow: 0 4px 14px rgba(40,44,63,.16); border-color: var(--line);
        }}
        /* The dataset's product shots are already cut out on white, so a white
           tile lets the garment sit on the card instead of inside a visible
           grey box. The hairline border keeps the card readable. */
        .ais-imgwrap {{
            background: #FFFFFF; border: 1px solid var(--line); border-radius: 4px;
            display: flex; align-items: center; justify-content: center;
            overflow: hidden; height: 340px;
        }}
        .ais-imgwrap img {{
            max-width: min({CARD_IMAGE_WIDTH}px, 100%); max-height: 332px;
            width: auto; height: auto; object-fit: contain; display: block;
            transition: transform .25s ease;
        }}
        .ais-card:hover .ais-imgwrap img {{ transform: scale(1.035); }}
        .ais-body {{ padding: 8px 9px 10px 9px; }}
        .ais-brand {{
            font-size: 13.5px; font-weight: 700; color: var(--ink);
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            line-height: 1.25;
        }}
        .ais-desc {{
            font-size: 12.5px; color: var(--body); margin-top: 1px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }}
        .ais-pricerow {{ margin-top: 5px; font-size: 13.5px; line-height: 1.3; }}
        .ais-pricerow b {{ font-weight: 700; color: var(--ink); }}
        .ais-mrp {{ color: var(--muted); text-decoration: line-through; margin-left: 5px; font-size: 12px; }}
        .ais-off {{ color: var(--discount); margin-left: 5px; font-size: 12px; font-weight: 600; }}
        .ais-meta {{
            margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px;
        }}
        .ais-chip {{
            font-size: 9.5px; font-weight: 700; letter-spacing: .3px;
            background: var(--tile); color: var(--body);
            padding: 2px 6px; border-radius: 2px; text-transform: uppercase;
        }}
        .ais-chip-accent {{ background: #FFF0F4; color: var(--pink); }}

        /* ---------- outfit result ---------- */
        .ais-look {{
            border: 1px solid var(--line); border-radius: 8px;
            overflow: hidden; margin-bottom: 6px;
        }}
        .ais-lookhead {{
            background: linear-gradient(100deg, #FFF3F6 0%, #FFF8F4 60%, #F4F6FF 100%);
            border-bottom: 1px solid var(--line); padding: 13px 18px;
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 8px;
        }}
        .ais-looktitle {{
            font-size: 14.5px; font-weight: 800; letter-spacing: .3px;
            text-transform: uppercase; color: var(--ink);
        }}
        .ais-lookmeta {{
            font-size: 12.5px; color: var(--body); margin-top: 3px; font-weight: 600;
        }}
        .ais-looktotal {{ text-align: right; }}
        .ais-looktotal b {{ font-size: 21px; font-weight: 800; color: var(--ink); }}
        .ais-looktotal small {{
            display: block; font-size: 11px; color: var(--muted); font-weight: 700;
            letter-spacing: .5px; text-transform: uppercase;
        }}
        .ais-badge {{
            display: inline-block; font-size: 10.5px; font-weight: 800;
            letter-spacing: .5px; padding: 3px 9px; border-radius: 11px;
            margin-right: 6px; text-transform: uppercase;
        }}
        .ais-badge-ok {{ background: #E8F7EF; color: #12805C; }}
        .ais-badge-warn {{ background: #FFF3E4; color: #B25E09; }}
        .ais-badge-info {{ background: #EFF1FE; color: #3E4BC4; }}
        .ais-badge-pink {{ background: #FFF0F4; color: var(--pink); }}

        .ais-plus {{
            text-align: center; color: #D4D5D9; font-size: 20px;
            font-weight: 300; padding-top: 105px;
        }}

        /* ---------- why panel ---------- */
        .ais-why {{
            border: 1px solid var(--line); border-radius: 8px; padding: 15px 17px;
            background: #FFF; height: 100%;
        }}
        .ais-why h4 {{
            font-size: 11px; font-weight: 800; letter-spacing: 1.2px;
            text-transform: uppercase; color: var(--pink); margin: 0 0 10px 0;
        }}
        .ais-why ul {{ margin: 0; padding: 0; list-style: none; }}
        .ais-why li {{
            font-size: 13px; line-height: 1.55; color: var(--body);
            padding: 3px 0 3px 21px; position: relative;
        }}
        .ais-why li:before {{
            content: "✓"; position: absolute; left: 0; top: 3px;
            color: #12805C; font-weight: 800; font-size: 12px;
        }}
        .ais-stylist {{
            border: 1px solid var(--line); border-radius: 8px; padding: 15px 17px;
            background: linear-gradient(180deg, #FFFAFB 0%, #FFF 60%); height: 100%;
        }}
        .ais-stylist p {{
            font-size: 13.5px; line-height: 1.65; color: var(--body); margin: 0;
        }}
        .ais-stylist h4 {{
            font-size: 11px; font-weight: 800; letter-spacing: 1.2px;
            text-transform: uppercase; color: var(--pink); margin: 0 0 9px 0;
        }}
        .ais-src {{
            font-size: 10.5px; color: var(--muted); margin-top: 9px;
            font-weight: 600; letter-spacing: .3px;
        }}

        /* ---------- product detail ---------- */
        .ais-pdp-img {{
            background: #FCFCFD; border: 1px solid var(--line); border-radius: 6px;
            display: flex; align-items: center; justify-content: center;
            padding: 20px; min-height: 470px;
        }}
        .ais-pdp-img img {{
            max-width: min({DETAIL_IMAGE_WIDTH}px, 100%); height: auto;
            border-radius: 3px;
        }}
        .ais-pdp-brand {{ font-size: 24px; font-weight: 800; letter-spacing: -0.4px; }}
        .ais-pdp-name {{ font-size: 17px; color: var(--body); margin-top: 1px; }}
        .ais-pdp-price {{ margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--line); }}
        .ais-pdp-price b {{ font-size: 25px; font-weight: 800; }}
        .ais-attr {{ margin-top: 16px; }}
        .ais-attr dt {{
            font-size: 10.5px; color: var(--muted); text-transform: uppercase;
            letter-spacing: .7px; font-weight: 700;
        }}
        .ais-attr dd {{ margin: 1px 0 11px 0; font-size: 14px; font-weight: 600; }}

        /* ---------- section headings ---------- */
        .ais-sec {{
            display: flex; align-items: baseline; gap: 10px;
            margin: 22px 0 12px 0; padding-bottom: 9px;
            border-bottom: 1px solid var(--line);
        }}
        .ais-sec h3 {{
            font-size: 16px; font-weight: 800; letter-spacing: .3px; margin: 0;
            text-transform: uppercase; color: var(--ink);
        }}
        .ais-sec span {{ font-size: 12.5px; color: var(--muted); font-weight: 600; }}

        /* ---------- misc ---------- */
        .ais-note {{ font-size: 11.5px; color: var(--muted); }}
        .stTabs [data-baseweb="tab"] {{
            font-size: 13px; font-weight: 700; letter-spacing: .3px;
        }}
        .stTabs [aria-selected="true"] {{ color: var(--pink); }}
        div[data-testid="stMetricValue"] {{ font-size: 19px; font-weight: 800; }}
        div[data-testid="stMetricLabel"] {{ font-size: 11px; color: var(--muted); }}
        .streamlit-expanderHeader {{ font-size: 12.5px; font-weight: 700; }}

        @media (max-width: 900px) {{
            .ais-hero h1 {{ font-size: 23px; }}
            .ais-imgwrap {{ height: 190px; }}
            .ais-plus {{ padding-top: 78px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

NAV_ITEMS = [
    ("MEN", "men"),
    ("WOMEN", "women"),
    ("KIDS", "kids"),
    ("HOME", "home"),
    ("BEAUTY", "beauty"),
    # Shorter than "AI Stylist" so the nav never wraps at tablet widths, and it
    # names the action rather than repeating the brand.
    ("STYLE ME", "style_me"),
]


def header(goto, wishlist_count: int = 0, bag_count: int = 0) -> None:
    """Brand bar, primary navigation, search and the user icon cluster."""
    logo_col, nav_col, search_col, user_col = st.columns(
        [1.55, 3.9, 2.25, 1.75], vertical_alignment="center"
    )

    with logo_col:
        st.markdown(
            '<div class="ais-logo">'
            '<div class="ais-mark">AI</div>'
            '<div class="ais-wordmark">Stylist<small>OUTFIT INTELLIGENCE</small></div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with nav_col:
        cols = st.columns(len(NAV_ITEMS), gap="small")
        for col, (label, view) in zip(cols, NAV_ITEMS):
            with col:
                if st.button(label, key=f"nav_{view}", use_container_width=True):
                    goto(view)

    with search_col:
        query = st.text_input(
            "Search",
            key="global_search",
            placeholder="Search for products, brands and more",
            label_visibility="collapsed",
        )
        if query and query != st.session_state.get("_last_search"):
            st.session_state["_last_search"] = query
            st.session_state["browse_query"] = query
            goto("search")

    with user_col:
        st.markdown(
            f"""
            <div class="ais-usernav">
              <div><span>👤</span>Profile</div>
              <div><span>♡</span>Wishlist{f'<b>{wishlist_count}</b>' if wishlist_count else ''}</div>
              <div><span>👜</span>Bag{f'<b>{bag_count}</b>' if bag_count else ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="ais-headrule"></div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="ais-sec"><h3>{escape(title)}</h3>'
        f'<span>{escape(subtitle)}</span></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Product cards
# --------------------------------------------------------------------------

def product_card_html(product: Product, chips: bool = True) -> str:
    """A compact marketplace product card."""
    brand = escape(brand_of(product))
    description = escape(short_name_of(product))
    price = int(product["price"])
    mrp = int(product.get("mrp", price))
    discount = int(product.get("discount_pct", 0))

    price_html = f"<b>{rupees(price)}</b>"
    if discount > 0 and mrp > price:
        price_html += (
            f'<span class="ais-mrp">{rupees(mrp)}</span>'
            f'<span class="ais-off">({discount}% OFF)</span>'
        )

    meta = ""
    if chips:
        meta = (
            '<div class="ais-meta">'
            f'<span class="ais-chip ais-chip-accent">'
            f'{escape(SLOT_LABELS.get(product["outfit_slot"], ""))}</span>'
            f'<span class="ais-chip">{escape(str(product["baseColour"]))}</span>'
            f'<span class="ais-chip">{escape(str(product["formality_tier"]))}</span>'
            "</div>"
        )

    return f"""
    <div class="ais-card">
      <div class="ais-imgwrap">
        <img src="{product_image_uri(product)}" alt="{brand}" loading="lazy"/>
      </div>
      <div class="ais-body">
        <div class="ais-brand">{brand}</div>
        <div class="ais-desc">{description}</div>
        <div class="ais-pricerow">{price_html}</div>
        {meta}
      </div>
    </div>
    """


def product_grid(
    products: list[Product],
    goto_product,
    columns: int = 4,
    key_prefix: str = "grid",
    action_label: str = "Complete the Look  →",
) -> None:
    """Dense product grid; each card carries one action."""
    if not products:
        st.info("No products match these filters. Try widening the price range.")
        return

    for start in range(0, len(products), columns):
        row = products[start:start + columns]
        cols = st.columns(columns, gap="small")
        for col, product in zip(cols, row):
            with col:
                st.markdown(product_card_html(product), unsafe_allow_html=True)
                if st.button(
                    action_label,
                    key=f"pc_{key_prefix}_{product['id']}",
                    use_container_width=True,
                ):
                    goto_product(int(product["id"]))
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Outfit presentation
# --------------------------------------------------------------------------

def look_header(outfit, prefs, index: int, occasion_label: str) -> None:
    """The banner that makes an outfit read as a styled look, not a search result."""
    pieces = " · ".join(
        SLOT_LABELS.get(slot, slot) for slot, _ in outfit.ordered_items
    )
    style_bits = [occasion_label]
    if prefs.style:
        style_bits.insert(0, prefs.style.replace("_", " ").title())
    meta = " · ".join(style_bits) + f" · {pieces}"

    if prefs.budget:
        headroom = prefs.budget - outfit.total_price
        budget_badge = (
            f'<span class="ais-badge ais-badge-ok">✓ Budget matched · '
            f"{rupees(headroom)} left</span>"
        )
        total_sub = f"of {rupees(prefs.budget)} budget"
    else:
        budget_badge = '<span class="ais-badge ais-badge-info">No budget set</span>'
        total_sub = f"{len(outfit.items)} pieces"

    st.markdown(
        f"""
        <div class="ais-lookhead">
          <div>
            <div class="ais-looktitle">✨ Your AI-styled look {index + 1}</div>
            <div class="ais-lookmeta">{escape(meta)}</div>
            <div style="margin-top:8px">
              {budget_badge}
              <span class="ais-badge ais-badge-pink">Match {outfit.final_score:.0%}</span>
            </div>
          </div>
          <div class="ais-looktotal">
            <b>{rupees(outfit.total_price)}</b>
            <small>{escape(total_sub)}</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def outfit_items_row(outfit, goto_product=None, key_prefix: str = "outfit") -> None:
    """The garments of one outfit, left to right, with + separators."""
    items = outfit.ordered_items
    widths: list[float] = []
    for index in range(len(items)):
        if index:
            widths.append(0.16)
        widths.append(1.0)
    cols = st.columns(widths, gap="small")

    position = 0
    for index, (slot, product) in enumerate(items):
        if index:
            with cols[position]:
                st.markdown('<div class="ais-plus">+</div>', unsafe_allow_html=True)
            position += 1
        with cols[position]:
            st.markdown(product_card_html(product), unsafe_allow_html=True)
            if goto_product is not None:
                if st.button(
                    "View product  →",
                    key=f"pc_{key_prefix}_{slot}_{product['id']}",
                    use_container_width=True,
                ):
                    goto_product(int(product["id"]))
        position += 1


def why_panel(reasons: list[str]) -> None:
    bullets = "".join(f"<li>{escape(reason)}</li>" for reason in reasons)
    st.markdown(
        f'<div class="ais-why"><h4>✓ Why this look works</h4><ul>{bullets}</ul></div>',
        unsafe_allow_html=True,
    )


def stylist_note(text: str, source: str) -> None:
    label = (
        "Generated by Gemini from the selected products and their computed scores"
        if source == "gemini"
        else "Generated from the engine's own signals (Gemini not in use)"
    )
    st.markdown(
        f'<div class="ais-stylist"><h4>✨ Stylist note</h4>'
        f"<p>{escape(text)}</p>"
        f'<div class="ais-src">{escape(label)}</div></div>',
        unsafe_allow_html=True,
    )
