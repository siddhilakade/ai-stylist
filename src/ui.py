"""Presentation layer: theme, navigation, product cards and outfit composition.

Kept strictly separate from the engine - nothing in this file makes a styling
decision, and nothing in `recommender.py` and friends knows Streamlit exists.
That separation is what lets the whole engine be tested and evaluated headlessly.

THE VISUAL LANGUAGE
    Editorial fashion rather than dashboard. Three things carry it:

    1. A warm ivory ground with white cards. A white-on-white page gives cards
       no edge, which is what made the earlier build read flat. Every surface
       here sits *on* something.
    2. Two typefaces with different jobs - a display serif for editorial voice
       and a geometric sans for UI. One typeface can only express hierarchy
       through size, which is why single-font layouts read like admin panels.
    3. An elevation and motion system, applied consistently: content rises in
       on load, cards lift on hover, nothing moves without a reason.

    Original design under an original brand name - no third-party logo,
    wordmark, illustration or asset is reproduced.
"""

from __future__ import annotations

from html import escape
from typing import Any, Mapping

import streamlit as st

from src.data import image_url
from src.features import SLOT_LABELS, extract_short_name

Product = Mapping[str, Any]

# --- palette ---------------------------------------------------------------
# Mirrored in the `:root` block below. Kept here as Python constants because a
# few call sites build inline styles; the CSS is the source of truth for
# anything rendered as a stylesheet.
PINK = "#FF3F6C"          # primary action / brand accent
GOLD = "#B08D57"          # editorial kicker, secondary accent
INK = "#17161B"           # headings and prices
BODY = "#55505C"          # descriptions
MUTED = "#9A94A3"         # metadata
LINE = "#EAE4DA"          # hairlines, warm rather than grey
GROUND = "#FAF8F4"        # page background
CARD = "#FFFFFF"          # card surface
DISCOUNT = "#E0673A"      # discount call-out

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
# Written as a plain string rather than an f-string on purpose: CSS is almost
# entirely braces, and escaping every one of them as {{ }} is a reliable source
# of silent breakage. Values that Python also needs are duplicated above.

_CSS = """
<style>
/* Streamlit strips <link> tags from markdown, so webfonts are imported from
   inside the stylesheet. If they cannot load (offline), the system stacks
   below take over with no visual breakage. */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --pink: #FF3F6C; --pink-deep: #E12E58; --gold: #B08D57;
    --ink: #17161B; --body: #55505C; --muted: #9A94A3;
    --line: #EAE4DA; --ground: #FAF8F4; --card: #FFFFFF;
    --discount: #E0673A; --ok: #1F7A5C;

    --serif: 'Cormorant Garamond', Georgia, 'Times New Roman', serif;
    --sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI',
            Roboto, Arial, sans-serif;

    --shadow-sm: 0 2px 8px rgba(23,22,27,.05);
    --shadow: 0 8px 24px rgba(23,22,27,.07);
    --shadow-lg: 0 22px 55px rgba(23,22,27,.14);
    --r: 14px; --r-lg: 22px; --r-sm: 9px;

    --ease: cubic-bezier(.22,.61,.36,1);
}

/* ---------- reset Streamlit chrome ---------- */
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], #MainMenu, footer {
    display: none !important;
}
header[data-testid="stHeader"] { height: 0; background: transparent; }
.stApp { background: var(--ground); }
.block-container { max-width: 1320px; padding: 0.5rem 1.6rem 4rem 1.6rem; }
[data-testid="stVerticalBlock"] { gap: 0.4rem; }
[data-testid="stHorizontalBlock"] { gap: 0.8rem; }
hr { margin: 1rem 0; border-color: var(--line); }

html, body, [class*="css"], .stMarkdown, button, input, textarea, select {
    font-family: var(--sans);
    color: var(--ink);
}

/* ---------- motion ----------
   One entrance animation, applied with a stagger. Streamlit re-runs the whole
   script on every interaction, so these replay on each render - which is why
   they are short and never move layout after settling. */
@keyframes riseIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: none; }
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes drift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes sheen {
    0%   { transform: translateX(-120%) skewX(-18deg); }
    60%  { transform: translateX(320%) skewX(-18deg); }
    100% { transform: translateX(320%) skewX(-18deg); }
}
@keyframes popIn {
    0%   { opacity: 0; transform: scale(.86); }
    70%  { transform: scale(1.04); }
    100% { opacity: 1; transform: scale(1); }
}

.rise { animation: riseIn .5s var(--ease) both; }
.rise-1 { animation-delay: .04s; } .rise-2 { animation-delay: .08s; }
.rise-3 { animation-delay: .12s; } .rise-4 { animation-delay: .16s; }
.rise-5 { animation-delay: .20s; } .rise-6 { animation-delay: .24s; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .001ms !important;
        transition-duration: .001ms !important;
    }
}

/* ---------- header ---------- */
.ais-logo { display: flex; align-items: baseline; gap: 7px; padding: 6px 0 0 0; }
.ais-wordmark {
    font-family: var(--serif); font-size: 31px; font-weight: 700;
    letter-spacing: -.5px; line-height: 1; color: var(--ink);
}
.ais-wordmark i { font-style: italic; color: var(--pink); }
.ais-wordmark small {
    display: block; font-family: var(--sans); font-size: 8px; font-weight: 700;
    letter-spacing: 3.1px; color: var(--muted); margin-top: 4px;
    text-transform: uppercase; white-space: nowrap;
}
[class*="st-key-nav_"] button p { white-space: nowrap; }
.ais-usernav {
    display: flex; gap: 24px; justify-content: flex-end; align-items: center;
    padding-top: 8px;
}
.ais-usernav div {
    text-align: center; font-size: 10.5px; font-weight: 700; color: var(--body);
    line-height: 1.2; cursor: default; transition: color .18s var(--ease);
}
.ais-usernav div:hover { color: var(--pink); }
.ais-usernav span { display: block; font-size: 16px; line-height: 1.3; }
.ais-usernav b {
    display: inline-block; background: var(--pink); color: #FFF; font-size: 9px;
    border-radius: 8px; padding: 0 5px; margin-left: 3px;
}
.ais-headrule {
    height: 1px; margin: 6px 0 18px 0;
    background: linear-gradient(90deg, transparent, var(--line) 12%,
                var(--line) 88%, transparent);
}

/* ---------- buttons ----------
   Streamlit stamps a `st-key-<key>` class on each widget's container, which is
   what lets these rules target one role of button without restyling all. */

[class*="st-key-nav_"] button {
    border: none !important; background: transparent !important;
    border-radius: 0 !important; border-bottom: 2px solid transparent !important;
    font-size: 12px; font-weight: 700; letter-spacing: 1.1px;
    text-transform: uppercase; color: var(--body) !important;
    min-height: 38px; padding: 0 2px !important;
    transition: color .18s var(--ease), border-color .18s var(--ease);
}
[class*="st-key-nav_"] button:hover {
    color: var(--ink) !important; border-bottom-color: var(--pink) !important;
}
[class*="st-key-nav_"] button p { font-size: 12px; font-weight: 700; }

/* product-card action -> a quiet accent link glued under the card */
[class*="st-key-pc_"] { margin-top: -4px; }
[class*="st-key-pc_"] button {
    border: none !important; background: transparent !important;
    color: var(--pink) !important; font-size: 10.5px; font-weight: 800;
    letter-spacing: .9px; text-transform: uppercase; min-height: 30px;
    justify-content: flex-start !important; padding-left: 2px !important;
}
[class*="st-key-pc_"] button p { font-size: 10.5px; font-weight: 800; }
[class*="st-key-pc_"] button:hover { color: var(--pink-deep) !important; }

/* example chips -> soft outlined suggestions */
[class*="st-key-ex_"] button {
    border: 1px solid var(--line); background: var(--card);
    border-radius: var(--r-sm) !important;
    color: var(--body) !important; font-weight: 600; font-size: 12px;
    letter-spacing: 0; text-transform: none; min-height: 52px;
    line-height: 1.4; padding: 8px 12px; box-shadow: var(--shadow-sm);
    transition: all .2s var(--ease);
}
[class*="st-key-ex_"] button:hover {
    border-color: var(--pink); color: var(--ink) !important;
    transform: translateY(-2px); box-shadow: var(--shadow);
}
[class*="st-key-ex_"] button p { font-size: 12px; font-weight: 600; }

/* ---------- buttons (default) ---------- */
.stButton > button {
    border-radius: var(--r-sm); border: 1px solid var(--line);
    background: var(--card); color: var(--ink); font-weight: 700; font-size: 12px;
    letter-spacing: .4px; padding: 0.4rem 0.9rem; min-height: 38px;
    transition: all .2s var(--ease);
}
.stButton > button:hover {
    border-color: var(--pink); color: var(--pink); transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
}
.stButton > button:focus:not(:active) { color: var(--pink); border-color: var(--pink); }
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, var(--pink) 0%, #FF6B4A 100%);
    border: none; text-transform: uppercase; letter-spacing: 1.1px;
    font-weight: 800; min-height: 48px; font-size: 12.5px;
    box-shadow: 0 10px 26px rgba(255,63,108,.30);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 34px rgba(255,63,108,.40);
}
/* The label MUST be set on the inner nodes, not just the button. Streamlit
   derives its own button text colour from `textColor` in config.toml and
   applies it to the inner <p>, which beats a `color` declared on the button -
   so a dark theme text colour rendered near-black type on the pink gradient
   and the primary action became unreadable. */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover *,
.stButton > button[kind="primary"]:focus,
.stButton > button[kind="primary"]:focus * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* Stepper choice cards -> large tappable tiles, not dropdown rows.
   MUST come after `.stButton > button`: both selectors have identical
   specificity (one class/attribute + one type), so source order decides, and
   these rules silently lost their min-height when declared earlier. */
[class*="st-key-step_"] button {
    border: 1.5px solid var(--line) !important; background: var(--card);
    border-radius: var(--r) !important; min-height: 76px;
    font-size: 14px !important; font-weight: 700; letter-spacing: .1px;
    text-transform: none; color: var(--ink) !important;
    box-shadow: var(--shadow-sm); transition: all .2s var(--ease);
}
[class*="st-key-step_"] button p { font-size: 14px !important; font-weight: 700; }
[class*="st-key-step_"] button:hover {
    border-color: var(--pink) !important; transform: translateY(-3px);
    box-shadow: var(--shadow);
}
/* The Back control is a step_ sibling in spirit but must not become a tile. */
[class*="st-key-finder_back_"] button { min-height: 38px; }

/* ---------- inputs ---------- */
.stTextInput input, .stTextArea textarea {
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--r-sm); font-size: 14.5px; color: var(--ink);
    transition: border-color .2s var(--ease), box-shadow .2s var(--ease);
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--pink); box-shadow: 0 0 0 3px rgba(255,63,108,.10);
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder { color: var(--muted); }
.stSlider [data-baseweb="slider"] div[role="slider"] { border-color: var(--pink); }

/* ---------- hero ---------- */
.ais-hero {
    position: relative; overflow: hidden; border-radius: var(--r-lg);
    background: linear-gradient(115deg, #FCEEF1 0%, #FBF3EA 42%, #F0F0FA 100%);
    background-size: 220% 220%; animation: drift 22s ease-in-out infinite;
    border: 1px solid var(--line); padding: 46px 46px 42px 46px;
    margin-bottom: 22px; box-shadow: var(--shadow-sm);
}
.ais-hero:after {
    content: ""; position: absolute; inset: 0 auto 0 0; width: 22%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.55), transparent);
    animation: sheen 5.5s var(--ease) 1s infinite; pointer-events: none;
}
.ais-kicker {
    display: inline-block; font-size: 10px; font-weight: 800; letter-spacing: 2.6px;
    text-transform: uppercase; color: var(--gold); margin-bottom: 14px;
}
.ais-hero h1 {
    font-family: var(--serif); font-size: 55px; font-weight: 600;
    letter-spacing: -1.2px; margin: 0 0 12px 0; line-height: 1.03; color: var(--ink);
    max-width: 15ch;
}
.ais-hero h1 em { font-style: italic; color: var(--pink); }
.ais-hero p {
    font-size: 15.5px; color: var(--body); margin: 0; max-width: 54ch;
    line-height: 1.65;
}
.ais-hero-grid {
    display: grid; grid-template-columns: 1.05fr .95fr; gap: 34px; align-items: center;
    position: relative; z-index: 1;
}
.ais-hero-grid.solo { grid-template-columns: 1fr; }
.ais-hero-grid.solo h1 { max-width: 20ch; }
.ais-hero-grid.solo p { max-width: 62ch; }
/* The dataset ships cut-out product shots on white, so a plain collage would
   read as four floating garments. Each tile gets its own tint, which gives the
   composition the blocks of colour that photography would otherwise provide. */
.ais-collage { display: grid; grid-template-columns: 1fr 1fr; grid-auto-rows: 196px; gap: 13px; }
.ais-collage figure {
    margin: 0; border-radius: var(--r); overflow: hidden; display: flex;
    align-items: center; justify-content: center; padding: 14px;
    box-shadow: var(--shadow); animation: riseIn .6s var(--ease) both;
    transition: transform .3s var(--ease);
}
.ais-collage figure:hover { transform: translateY(-4px) rotate(-1deg); }
.ais-collage figure:nth-child(1) { background: #F6E1E7; animation-delay: .05s; }
.ais-collage figure:nth-child(2) { background: #E7E2F4; animation-delay: .12s; }
.ais-collage figure:nth-child(3) { background: #F2EADB; animation-delay: .19s; }
.ais-collage figure:nth-child(4) { background: #DFEBE7; animation-delay: .26s; }
/* The source shots are cut out on white. `multiply` drops that white into the
   tile's tint, so the garment sits ON the colour instead of inside a white
   rectangle floating on it. Model shots with a grey studio background pick up
   the tint too, which reads as a deliberate duotone rather than a mismatch. */
.ais-collage img {
    max-width: 100%; max-height: 100%; object-fit: contain; display: block;
    mix-blend-mode: multiply;
}

.ais-pills { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 22px; }
.ais-pill {
    display: inline-flex; align-items: center; gap: 7px; background: rgba(255,255,255,.8);
    border: 1px solid var(--line); padding: 7px 15px; border-radius: 999px;
    font-size: 12px; font-weight: 700; color: var(--body); backdrop-filter: blur(6px);
}
.ais-pill b { color: var(--pink); }

/* ---------- section headings (three tier) ---------- */
.ais-sec { margin: 40px 0 18px 0; }
.ais-sec .k {
    font-size: 10px; font-weight: 800; letter-spacing: 2.4px;
    text-transform: uppercase; color: var(--gold); display: block; margin-bottom: 7px;
}
.ais-sec h3 {
    font-family: var(--serif); font-size: 34px; font-weight: 600;
    letter-spacing: -.5px; margin: 0; color: var(--ink); line-height: 1.1;
}
.ais-sec h3 em { font-style: italic; color: var(--pink); }
.ais-sec .sub { font-size: 13.5px; color: var(--muted); margin-top: 6px; font-weight: 500; }

/* ---------- product card ---------- */
.ais-card {
    border: 1px solid var(--line); border-radius: var(--r); background: var(--card);
    overflow: hidden; box-shadow: var(--shadow-sm);
    transition: transform .26s var(--ease), box-shadow .26s var(--ease);
}
.ais-card:hover { transform: translateY(-5px); box-shadow: var(--shadow-lg); }
/* The catalog mixes two kinds of shot: garments cut out on white, and model
   photography on a grey studio backdrop. Side by side in a grid those read as
   two different products. A warm tile plus `multiply` reconciles them - white
   drops out entirely, and the grey backdrop resolves to the same warm tone -
   so the grid looks art-directed rather than scraped. */
.ais-imgwrap {
    background: #F7F2EB; border-bottom: 1px solid var(--line);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden; height: 330px; position: relative;
}
.ais-imgwrap img {
    max-width: min(284px, 100%); max-height: 318px;
    width: auto; height: auto; object-fit: contain; display: block;
    mix-blend-mode: multiply;
    transition: transform .4s var(--ease);
}
.ais-card:hover .ais-imgwrap img { transform: scale(1.055); }
.ais-body { padding: 13px 14px 15px 14px; }
.ais-brand {
    font-size: 14px; font-weight: 700; color: var(--ink); letter-spacing: -.1px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.3;
}
.ais-desc {
    font-size: 12.5px; color: var(--muted); margin-top: 2px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ais-pricerow { margin-top: 9px; font-size: 14.5px; line-height: 1.3; }
.ais-pricerow b { font-weight: 800; color: var(--ink); }
.ais-mrp { color: var(--muted); text-decoration: line-through; margin-left: 6px; font-size: 12px; }
.ais-off { color: var(--discount); margin-left: 6px; font-size: 12px; font-weight: 700; }
/* One line, always. Wrapping chips made neighbouring cards different heights,
   which left the action links under them visibly out of line at narrow widths. */
.ais-meta {
    margin-top: 10px; display: flex; flex-wrap: nowrap; gap: 5px;
    overflow: hidden; height: 21px;
}
.ais-chip { flex: none; }
.ais-chip {
    font-size: 9.5px; font-weight: 700; letter-spacing: .5px; background: var(--ground);
    border: 1px solid var(--line); color: var(--body);
    padding: 3px 8px; border-radius: 999px; text-transform: uppercase;
}
.ais-chip-accent { background: #FFF1F4; border-color: #FBD8E1; color: var(--pink); }

/* ---------- the composed look ----------
   The whole point of the product is that these pieces belong together. Rendering
   them as a row of identical catalog tiles said the opposite, so an outfit is one
   surface with one border, one price and internal size hierarchy. */
.ais-look {
    background: var(--card); border: 1px solid var(--line); border-radius: var(--r-lg);
    box-shadow: var(--shadow); overflow: hidden; margin-bottom: 10px;
    animation: riseIn .55s var(--ease) both;
}
.ais-lookhead {
    display: flex; justify-content: space-between; align-items: flex-end;
    flex-wrap: wrap; gap: 14px; padding: 26px 30px 20px 30px;
    background: linear-gradient(120deg, #FDF3F5 0%, #FBF5EC 55%, #F2F1FB 100%);
    border-bottom: 1px solid var(--line);
}
.ais-looknum {
    font-family: var(--serif); font-size: 40px; font-weight: 600; font-style: italic;
    line-height: .9; color: var(--ink);
}
.ais-lookmeta {
    font-size: 12px; color: var(--body); margin-top: 8px; font-weight: 600;
    letter-spacing: .4px; text-transform: uppercase;
}
.ais-looktotal { text-align: right; }
.ais-looktotal b {
    font-family: var(--serif); font-size: 38px; font-weight: 600; color: var(--ink);
    line-height: 1;
}
.ais-looktotal small {
    display: block; font-size: 10.5px; color: var(--muted); font-weight: 700;
    letter-spacing: 1.2px; text-transform: uppercase; margin-top: 5px;
}
.ais-badges { margin-top: 12px; }
.ais-badge {
    display: inline-block; font-size: 10px; font-weight: 800; letter-spacing: .9px;
    padding: 5px 11px; border-radius: 999px; margin-right: 7px;
    text-transform: uppercase; animation: popIn .4s var(--ease) both;
}
.ais-badge-ok { background: #E4F5EE; color: var(--ok); }
.ais-badge-warn { background: #FDF0E0; color: #9E5A12; }
.ais-badge-info { background: #EDEEFB; color: #3E4BC4; }
.ais-badge-pink { background: #FFF1F4; color: var(--pink); }

/* the flat-lay itself */
.ais-lay { display: grid; grid-template-columns: 1.12fr 1fr; gap: 14px; padding: 26px 30px 30px 30px; }
.ais-lay-rest { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
.ais-lay-rest .ais-it:nth-child(odd):last-child { grid-column: span 2; }
.ais-it {
    position: relative; background: #FFF; border: 1px solid var(--line);
    border-radius: var(--r); overflow: hidden; display: flex; flex-direction: column;
    transition: transform .26s var(--ease), box-shadow .26s var(--ease),
                border-color .26s var(--ease);
    animation: riseIn .5s var(--ease) both;
}
.ais-it:nth-child(1) { animation-delay: .05s; } .ais-it:nth-child(2) { animation-delay: .1s; }
.ais-it:nth-child(3) { animation-delay: .15s; } .ais-it:nth-child(4) { animation-delay: .2s; }
.ais-it:hover { transform: translateY(-4px); box-shadow: var(--shadow); border-color: #E2D6C4; }
.ais-it-img {
    flex: 1; display: flex; align-items: center; justify-content: center;
    padding: 12px; background: #F7F2EB;
}
.ais-it-img img {
    max-width: 100%; max-height: 100%; object-fit: contain; display: block;
    mix-blend-mode: multiply;
}
.ais-it-hero { min-height: 390px; }
.ais-it-hero .ais-it-img img { max-height: 320px; }
.ais-it-sm { min-height: 188px; }
.ais-it-sm .ais-it-img img { max-height: 124px; }
.ais-it-cap {
    border-top: 1px solid var(--line); padding: 9px 12px 11px 12px; background: #FFF;
}
.ais-it-slot {
    font-size: 8.5px; font-weight: 800; letter-spacing: 1.7px; text-transform: uppercase;
    color: var(--gold);
}
.ais-it-name {
    font-size: 12.5px; font-weight: 700; color: var(--ink); margin-top: 3px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ais-it-price { font-size: 12.5px; font-weight: 700; color: var(--body); margin-top: 2px; }

/* ---------- why / stylist panels ---------- */
.ais-why, .ais-stylist {
    border: 1px solid var(--line); border-radius: var(--r); padding: 22px 24px;
    background: var(--card); height: 100%; box-shadow: var(--shadow-sm);
}
.ais-stylist { background: linear-gradient(165deg, #FDF6F8 0%, #FFF 55%); }
.ais-why h4, .ais-stylist h4 {
    font-family: var(--serif); font-size: 22px; font-weight: 600; font-style: italic;
    letter-spacing: -.2px; text-transform: none; color: var(--ink); margin: 0 0 14px 0;
}
.ais-why ul { margin: 0; padding: 0; list-style: none; }
.ais-why li {
    font-size: 13px; line-height: 1.6; color: var(--body);
    padding: 5px 0 5px 24px; position: relative;
}
.ais-why li:before {
    content: ""; position: absolute; left: 0; top: 12px; width: 12px; height: 2px;
    background: var(--pink); border-radius: 2px;
}
.ais-stylist p { font-size: 14px; line-height: 1.75; color: var(--body); margin: 0; }
.ais-src {
    font-size: 10px; color: var(--muted); margin-top: 14px; font-weight: 600;
    letter-spacing: .5px; text-transform: uppercase;
}

/* ---------- stepper ---------- */
.ais-steps { display: flex; align-items: center; justify-content: center; gap: 12px; margin: 6px 0 26px 0; }
.ais-step { display: flex; align-items: center; gap: 9px; }
.ais-step i {
    width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center;
    font-style: normal; font-size: 12px; font-weight: 800; background: #EFE9E0;
    color: var(--muted); transition: all .3s var(--ease);
}
.ais-step span { font-size: 12px; font-weight: 700; color: var(--muted); letter-spacing: .3px; }
.ais-step.on i { background: var(--ink); color: #FFF; transform: scale(1.06); }
.ais-step.on span { color: var(--ink); }
.ais-step.done i { background: var(--pink); color: #FFF; }
.ais-step.done span { color: var(--body); }
.ais-steprule { width: 46px; height: 2px; background: var(--line); border-radius: 2px; }
.ais-count {
    text-align: center; font-size: 13px; color: var(--body); font-weight: 600;
    margin: 4px 0 2px 0;
}
.ais-count b { font-family: var(--serif); font-size: 25px; font-weight: 700; color: var(--pink); font-style: italic; }
.ais-count.warn b { color: var(--discount); }
.ais-count .d { display: block; font-size: 11.5px; color: var(--muted); margin-top: 4px; font-weight: 500; }

/* ---------- product detail ---------- */
/* Same tint + multiply as the grid and the composed look. Left on white, a
   model shot showed its grey studio box inside a white panel - the one place
   in the app where the two shot types were still visibly different. */
.ais-pdp-img {
    background: #F7F2EB; border: 1px solid var(--line); border-radius: var(--r-lg);
    display: flex; align-items: center; justify-content: center; padding: 30px;
    min-height: 480px; box-shadow: var(--shadow-sm);
}
.ais-pdp-img img {
    max-width: min(360px, 100%); height: auto; border-radius: var(--r-sm);
    mix-blend-mode: multiply;
}
.ais-pdp-brand { font-family: var(--serif); font-size: 38px; font-weight: 600; letter-spacing: -.6px; line-height: 1.1; }
.ais-pdp-name { font-size: 16px; color: var(--body); margin-top: 4px; }
.ais-pdp-price { margin-top: 20px; padding-top: 20px; border-top: 1px solid var(--line); }
.ais-pdp-price b { font-size: 30px; font-weight: 800; }
.ais-attr { margin-top: 22px; border-top: 1px solid var(--line); }
.ais-attr .row {
    display: grid; grid-template-columns: 150px 1fr; gap: 14px;
    padding: 11px 0; border-bottom: 1px solid var(--line);
}
.ais-attr dt { font-size: 11.5px; color: var(--muted); font-weight: 600; margin: 0; }
.ais-attr dd { margin: 0; font-size: 13.5px; font-weight: 600; color: var(--ink); }

/* ---------- footer ---------- */
.ais-footrule {
    height: 1px; margin: 56px 0 22px 0;
    background: linear-gradient(90deg, transparent, var(--line) 12%,
                var(--line) 88%, transparent);
}
.ais-foot-mark {
    font-family: var(--serif); font-size: 24px; font-weight: 700; color: var(--ink);
    line-height: 1;
}
.ais-foot-mark i { font-style: italic; color: var(--pink); }
.ais-foot p {
    font-size: 12px; color: var(--muted); margin: 8px 0 0 0; max-width: 62ch;
    line-height: 1.6;
}

/* ---------- misc ---------- */
.ais-note { font-size: 11.5px; color: var(--muted); }
.stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 700; letter-spacing: .3px; }
.stTabs [aria-selected="true"] { color: var(--pink); }
div[data-testid="stMetricValue"] { font-size: 21px; font-weight: 800; }
div[data-testid="stMetricLabel"] { font-size: 11px; color: var(--muted); }
.streamlit-expanderHeader { font-size: 12.5px; font-weight: 700; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: var(--r); background: var(--card); }

/* The nav is six items plus a search field; below this it starts colliding. */
@media (max-width: 1150px) {
    [class*="st-key-nav_"] button p { font-size: 10.5px; letter-spacing: .4px; }
    .ais-wordmark { font-size: 25px; }
    .ais-usernav { gap: 14px; }
    .ais-usernav div { font-size: 9.5px; }
}

@media (max-width: 1000px) {
    .ais-hero { padding: 30px 26px; }
    .ais-hero h1 { font-size: 36px; }
    .ais-sec h3 { font-size: 26px; }
    .ais-lay { grid-template-columns: 1fr; }
    .ais-it-hero { min-height: 300px; }
    .ais-imgwrap { height: 210px; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


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
            '<div class="ais-logo"><div class="ais-wordmark">AI <i>Stylist</i>'
            "<small>Outfit Intelligence</small></div></div>",
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
              <div><span>◍</span>Profile</div>
              <div><span>♡</span>Wishlist{f'<b>{wishlist_count}</b>' if wishlist_count else ''}</div>
              <div><span>▣</span>Bag{f'<b>{bag_count}</b>' if bag_count else ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="ais-headrule"></div>', unsafe_allow_html=True)


def hero(kicker: str, headline_html: str, body: str,
         pills: list[str] | None = None,
         collage: list[Product] | None = None) -> None:
    """The editorial hero. `headline_html` may contain <em> for the accent word."""
    pills_html = ""
    if pills:
        pills_html = (
            '<div class="ais-pills">'
            + "".join(f'<span class="ais-pill">{p}</span>' for p in pills)
            + "</div>"
        )

    collage_html = ""
    if collage:
        figures = "".join(
            f'<figure><img src="{product_image_uri(p)}" '
            f'alt="{escape(brand_of(p))}" loading="lazy"/></figure>'
            for p in collage[:4]
        )
        collage_html = f'<div class="ais-collage">{figures}</div>'

    # Without a collage the two-column grid leaves a dead right-hand half, so
    # the text hero gets its own single-column modifier instead.
    grid_class = "ais-hero-grid" if collage_html else "ais-hero-grid solo"

    st.markdown(
        '<div class="ais-hero">'
        f'<div class="{grid_class}"><div>'
        f'<div class="ais-kicker">{escape(kicker)}</div>'
        f"<h1>{headline_html}</h1>"
        f"<p>{escape(body)}</p>"
        f"{pills_html}</div>"
        f"{collage_html}</div></div>",
        unsafe_allow_html=True,
    )


def footer(goto) -> None:
    """Closing band, and the only route to the How-it-works page.

    `view_about` was registered in the router but nothing linked to it, so the
    page documenting the synthetic prices and the known limitations was
    unreachable - while the product page told shoppers to "see How it works".
    """
    st.markdown('<div class="ais-footrule"></div>', unsafe_allow_html=True)
    brand_col, link_col = st.columns([2.4, 1], vertical_alignment="center")
    with brand_col:
        st.markdown(
            '<div class="ais-foot">'
            '<div class="ais-foot-mark">AI <i>Stylist</i></div>'
            "<p>An independent prototype. Prices are simulated, and the catalog is "
            "a 536-piece sample - both are documented rather than hidden. "
            "Not affiliated with any marketplace.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with link_col:
        if st.button("How it works  →", key="foot_about", use_container_width=True):
            goto("about")


def section(title: str, subtitle: str = "", kicker: str = "") -> None:
    """Three-tier editorial header: kicker, display title, muted subtitle.

    A single bold line can only signal hierarchy by size. The kicker carries
    category, the serif carries voice, the subtitle carries detail - which is
    what makes a page read as edited rather than generated.
    """
    kicker_html = f'<span class="k">{escape(kicker)}</span>' if kicker else ""
    sub_html = f'<div class="sub">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="ais-sec rise">{kicker_html}<h3>{escape(title)}</h3>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def stepper(current: int, labels: list[str]) -> None:
    """Numbered progress indicator for the guided Style Me flow."""
    parts = []
    for index, label in enumerate(labels):
        state = "on" if index == current else ("done" if index < current else "")
        mark = "✓" if index < current else str(index + 1)
        parts.append(
            f'<div class="ais-step {state}"><i>{mark}</i><span>{escape(label)}</span></div>'
        )
        if index < len(labels) - 1:
            parts.append('<div class="ais-steprule"></div>')
    st.markdown(f'<div class="ais-steps">{"".join(parts)}</div>', unsafe_allow_html=True)


def match_count(count: int, noun: str = "pieces to build from",
                detail: str = "") -> None:
    """Live feedback on how many catalog items still satisfy the choices made.

    It lets a user watch the pool shrink toward zero *before* they commit to a
    request the catalog cannot satisfy, which is a better outcome than the best
    failure message.

    `noun` says GARMENTS, not results. Calling these "matches" implied the next
    screen would list them, when the engine's job is to assemble a handful of
    complete outfits out of them - so a count of 173 followed by 3 looks read as
    a bug rather than as the product working.
    """
    tone = " warn" if count < 12 else ""
    detail_html = f'<span class="d">{escape(detail)}</span>' if detail else ""
    st.markdown(
        f'<div class="ais-count{tone}"><b>{count}</b> {escape(noun)}'
        f"{detail_html}</div>",
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
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Outfit presentation
# --------------------------------------------------------------------------

def _look_item_html(slot: str, product: Product, hero: bool = False) -> str:
    """One garment tile, as COMPACT html.

    Deliberately emitted with no newlines and no indentation. Streamlit renders
    through Markdown even with `unsafe_allow_html=True`, and Markdown ends an
    HTML block at the first blank line and treats any 4-space-indented line as a
    code block. A pretty-printed fragment interpolated into a parent template
    therefore breaks out of its own wrapper and prints the next tag as source.
    """
    size_class = "ais-it-hero" if hero else "ais-it-sm"
    return (
        f'<div class="ais-it {size_class}">'
        f'<div class="ais-it-img">'
        f'<img src="{product_image_uri(product)}" '
        f'alt="{escape(brand_of(product))}" loading="lazy"/>'
        f"</div>"
        f'<div class="ais-it-cap">'
        f'<div class="ais-it-slot">{escape(SLOT_LABELS.get(slot, slot))}</div>'
        f'<div class="ais-it-name">{escape(brand_of(product))}</div>'
        f'<div class="ais-it-price">{rupees(int(product["price"]))}</div>'
        f"</div></div>"
    )


def look(outfit, prefs, index: int, occasion_label: str) -> None:
    """One outfit rendered as a single composed surface.

    Emitted as ONE html block rather than through `st.columns`, because the
    column API cannot express the size hierarchy that makes a set of garments
    read as a styled look instead of a row of search results.
    """
    items = outfit.ordered_items
    pieces = " · ".join(SLOT_LABELS.get(slot, slot) for slot, _ in items)
    style_bits = [occasion_label]
    if prefs.style:
        style_bits.insert(0, prefs.style.replace("_", " ").title())
    meta = " · ".join(style_bits) + f" · {pieces}"

    if prefs.budget:
        headroom = prefs.budget - outfit.total_price
        badges = (
            f'<span class="ais-badge ais-badge-ok">Within budget · '
            f"{rupees(headroom)} left</span>"
        )
        total_sub = f"of {rupees(prefs.budget)} budget"
    else:
        badges = '<span class="ais-badge ais-badge-info">No budget set</span>'
        total_sub = f"{len(outfit.items)} pieces"
    badges += f'<span class="ais-badge ais-badge-pink">Match {outfit.final_score:.0%}</span>'

    hero_html = _look_item_html(*items[0], hero=True) if items else ""
    rest_html = "".join(_look_item_html(slot, item) for slot, item in items[1:])

    st.markdown(
        '<div class="ais-look">'
        '<div class="ais-lookhead"><div>'
        f'<div class="ais-looknum">Look {index + 1:02d}</div>'
        f'<div class="ais-lookmeta">{escape(meta)}</div>'
        f'<div class="ais-badges">{badges}</div>'
        "</div>"
        f'<div class="ais-looktotal"><b>{rupees(outfit.total_price)}</b>'
        f"<small>{escape(total_sub)}</small></div>"
        "</div>"
        f'<div class="ais-lay">{hero_html}'
        f'<div class="ais-lay-rest">{rest_html}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def look_shop_row(outfit, goto_product, key_prefix: str = "outfit") -> None:
    """One quiet row of actions under a composed look.

    A button under every garment turned the outfit back into a list of products.
    The actions still exist - they are just no longer part of the composition.
    """
    items = outfit.ordered_items
    # Weighted so the links stay grouped under the look rather than stretching
    # across the full width of the page.
    cols = st.columns([1] * len(items) + [max(1, 7 - len(items))], gap="small")
    for col, (slot, product) in zip(cols, items):
        with col:
            if st.button(
                f"{SLOT_LABELS.get(slot, slot)}  →",
                key=f"pc_{key_prefix}_{slot}_{product['id']}",
                use_container_width=True,
            ):
                goto_product(int(product["id"]))


def why_panel(reasons: list[str]) -> None:
    bullets = "".join(f"<li>{escape(reason)}</li>" for reason in reasons)
    st.markdown(
        f'<div class="ais-why rise"><h4>Why this look works</h4><ul>{bullets}</ul></div>',
        unsafe_allow_html=True,
    )


def stylist_note(text: str, source: str) -> None:
    label = (
        "Written by the language model from the selected products and their computed scores"
        if source == "gemini"
        else "Written from the engine's own signals (no language model in use)"
    )
    st.markdown(
        f'<div class="ais-stylist rise"><h4>Stylist note</h4>'
        f"<p>{escape(text)}</p>"
        f'<div class="ais-src">{escape(label)}</div></div>',
        unsafe_allow_html=True,
    )
