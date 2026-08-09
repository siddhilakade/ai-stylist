"""Smoke tests for the Streamlit interface.

Streamlit's AppTest runs the real script headlessly, so these catch the class of
bug that unit tests cannot: a view that raises on render. They run with the LLM
disabled so they never touch the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.data import catalog_records

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def run_view(view: str, **state) -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.session_state["view"] = view
    app.session_state["use_llm"] = False
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    return app


@pytest.mark.parametrize(
    "view", ["home", "about", "browse", "kids", "beauty", "search"]
)
def test_static_views_render_without_error(view):
    app = run_view(view)
    assert not app.exception, [str(e.value) for e in app.exception]


@pytest.mark.parametrize("gender", ["Men", "Women"])
def test_browse_renders_for_each_gender(gender):
    app = run_view("browse", browse_gender=gender)
    assert not app.exception, [str(e.value) for e in app.exception]


def test_style_me_end_to_end_renders_outfits():
    app = run_view(
        "style_me",
        request_text="I need a formal outfit for an office meeting, navy, budget 8000",
        run_now=True,
    )
    assert not app.exception, [str(e.value) for e in app.exception]
    rendered = " ".join(block.value for block in app.markdown)
    assert "Why this look works" in rendered
    assert "Your AI-styled look 1" in rendered


def test_style_me_reports_failure_instead_of_crashing():
    app = run_view(
        "style_me",
        request_text="formal office outfit for men under 400 rupees",
        run_now=True,
    )
    assert not app.exception, [str(e.value) for e in app.exception]
    assert app.error, "an impossible request should surface an error panel"


def test_product_page_renders_complete_the_look():
    product = next(p for p in catalog_records() if p["outfit_slot"] == "top")
    app = run_view("product", anchor_id=int(product["id"]))
    assert not app.exception, [str(e.value) for e in app.exception]
    rendered = " ".join(block.value for block in app.markdown)
    assert "Complete the look" in rendered
    assert "Similar products" in rendered


def test_product_card_shows_brand_price_and_discount():
    """The card must render real catalog values, never placeholders."""
    from src.ui import product_card_html

    product = next(p for p in catalog_records() if p["discount_pct"] > 0)
    html = product_card_html(product)
    assert f"{product['price']:,}" in html
    assert f"{product['mrp']:,}" in html
    assert f"{product['discount_pct']}% OFF" in html
    assert str(product["brand"]) in html


def test_kids_page_explains_the_gap_rather_than_returning_adult_clothing():
    app = run_view("kids")
    assert not app.exception
    rendered = " ".join(block.value for block in app.markdown)
    assert "not available in this prototype" in rendered
    assert "deliberately excluded" in rendered


def test_beauty_page_is_intentional_not_an_error():
    app = run_view("beauty")
    assert not app.exception
    rendered = " ".join(block.value for block in app.markdown)
    assert "outside this stylist's scope" in rendered
