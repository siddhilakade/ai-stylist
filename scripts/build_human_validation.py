"""Build the independently-judged validation set.

Every label is the author's styling judgement, written from the garment
descriptions WITHOUT consulting the rule engine. Not user data, not expert
annotation - one person's taste, written down.

What makes it useful anyway: it's independent of the rules, so it's the only
place "is the ML better than the rules?" can be asked without the question
answering itself. It's also deliberately adversarial - about a fifth of the pairs
are cases where the rules were expected to be wrong (brown shoes with black
trousers, black with navy, head-to-toe one colour, blazer with jeans).

Run:  python scripts/build_human_validation.py
Out:  data/ml_training/human_validation.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import catalog_records  # noqa: E402
from src.ml.features import canonical_order, pair_features  # noqa: E402

OUT_PATH = ROOT / "data" / "ml_training" / "human_validation.csv"

# (gender, articleType_a, colour_a, articleType_b, colour_b, label, rationale)
# label: 1 = "I would wear these together", 0 = "I would not"
JUDGEMENTS: list[tuple] = [
    # ---- clear positives: conventional, well-established combinations -------
    ("Men", "Shirts", "White", "Trousers", "Navy Blue", 1, "White shirt with navy trousers is the default smart outfit"),
    ("Men", "Shirts", "Blue", "Trousers", "Grey", 1, "Blue shirt and grey trousers, standard office pairing"),
    ("Men", "Shirts", "White", "Jeans", "Blue", 1, "White shirt and blue jeans, universally fine"),
    ("Men", "Trousers", "Navy Blue", "Formal Shoes", "Brown", 1, "Navy with brown leather is a classic"),
    ("Men", "Trousers", "Khaki", "Shirts", "White", 1, "Chinos and white shirt"),
    ("Men", "Tshirts", "White", "Jeans", "Blue", 1, "The most basic casual outfit there is"),
    ("Men", "Blazers", "Black", "Trousers", "Grey", 1, "Black blazer over grey trousers"),
    ("Men", "Shirts", "Pink", "Trousers", "Navy Blue", 1, "Pink shirt with navy is conventional, not loud"),
    ("Men", "Trousers", "Olive", "Casual Shoes", "Brown", 1, "Earth tones agree with each other"),
    ("Women", "Shirts", "White", "Trousers", "Black", 1, "White and black, safest possible pairing"),
    ("Women", "Tops", "Black", "Jeans", "Blue", 1, "Black top and blue denim"),
    ("Women", "Dresses", "Black", "Heels", "Black", 1, "Black dress with black heels"),
    ("Women", "Kurtas", "White", "Leggings", "Red", 1, "White kurta with a coloured legging is standard ethnic wear"),
    ("Women", "Trousers", "Navy Blue", "Heels", "Nude", 1, "Nude heels go with everything"),
    ("Women", "Skirts", "Black", "Shirts", "White", 1, "Black skirt, white shirt"),
    ("Women", "Jeans", "Blue", "Flats", "Brown", 1, "Denim with tan/brown flats"),
    ("Men", "Shirts", "Navy Blue", "Trousers", "Cream", 1, "Navy over cream reads well"),
    ("Women", "Sarees", "Red", "Flats", "Gold", 1, "Gold footwear with a red saree is conventional"),

    # ---- clear negatives: things nobody should wear together ----------------
    ("Men", "Formal Shoes", "Black", "Shorts", "Blue", 0, "Formal shoes with shorts, never"),
    ("Men", "Ties", "Red", "Tshirts", "Black", 0, "A tie with a t-shirt"),
    ("Men", "Blazers", "Black", "Shorts", "Red", 0, "Blazer with bright shorts"),
    ("Women", "Sports Shoes", "Blue", "Sarees", "Red", 0, "Trainers with a saree"),
    ("Men", "Shirts", "Orange", "Trousers", "Green", 0, "Orange and green, both loud, clash"),
    ("Men", "Tshirts", "Purple", "Shorts", "Orange", 0, "Purple and orange together is jarring"),
    ("Women", "Kurtas", "Orange", "Jeans", "Pink", 0, "Orange kurta with pink denim clashes"),
    ("Men", "Shirts", "Red", "Trousers", "Green", 0, "Red and green read as a costume"),
    ("Women", "Dresses", "Magenta", "Flats", "Orange", 0, "Magenta and orange fight each other"),
    ("Men", "Jackets", "Yellow", "Trousers", "Brown", 0, "Bright yellow jacket over brown, poor"),
    ("Women", "Tops", "Green", "Leggings", "Red", 0, "Green and red, no"),
    ("Men", "Shirts", "Yellow", "Jeans", "Grey", 0, "Bright yellow shirt is hard to place here"),

    # ---- cases where the RULE ENGINE IS EXPECTED TO BE WRONG ----------------
    # Rules: brown counts as a neutral, so brown+black scores well.
    # Reality: brown shoes with black trousers is the classic menswear error.
    ("Men", "Trousers", "Black", "Formal Shoes", "Brown", 0, "DISAGREE-WITH-RULES: brown shoes with black trousers"),
    ("Men", "Trousers", "Black", "Casual Shoes", "Brown", 0, "DISAGREE-WITH-RULES: same clash, casual version"),
    ("Women", "Trousers", "Black", "Flats", "Brown", 0, "DISAGREE-WITH-RULES: brown on black"),

    # Rules: black and navy are both 'neutral' -> near-perfect score.
    # Reality: they are close enough to look like a mistake in daylight.
    ("Men", "Shirts", "Black", "Trousers", "Navy Blue", 0, "DISAGREE-WITH-RULES: black and navy muddy each other"),
    ("Men", "Jackets", "Black", "Jeans", "Navy Blue", 0, "DISAGREE-WITH-RULES: black over navy"),

    # Rules: identical colour family -> high score.
    # Reality: head-to-toe one colour reads as a uniform.
    ("Men", "Tshirts", "Grey", "Jeans", "Grey", 0, "DISAGREE-WITH-RULES: head-to-toe grey is flat"),
    ("Men", "Shirts", "Brown", "Trousers", "Brown", 0, "DISAGREE-WITH-RULES: all-brown looks like a uniform"),
    ("Women", "Kurtas", "Red", "Leggings", "Red", 0, "DISAGREE-WITH-RULES: complete red-on-red"),

    # Rules: large formality gap -> penalised.
    # Reality: blazer with jeans is a completely standard smart-casual look.
    ("Men", "Blazers", "Blue", "Jeans", "Blue", 1, "DISAGREE-WITH-RULES: blazer and jeans is standard smart casual"),
    ("Men", "Blazers", "Black", "Jeans", "Black", 1, "DISAGREE-WITH-RULES: black blazer with black jeans works"),
    ("Women", "Blazers", "Black", "Jeans", "Blue", 1, "DISAGREE-WITH-RULES: blazer over denim"),

    # Rules: formal shoes with jeans is a big formality gap.
    # Reality: brown brogues with dark denim is a normal smart-casual choice.
    ("Men", "Formal Shoes", "Brown", "Jeans", "Navy Blue", 1, "DISAGREE-WITH-RULES: brogues with dark denim"),

    # Rules: 'Multi' is penalised generically.
    # Reality: a printed top over plain black is fine - the plain piece anchors it.
    ("Women", "Kurtas", "Multi", "Leggings", "Brown", 1, "DISAGREE-WITH-RULES: print anchored by a plain neutral"),
    ("Men", "Shirts", "Multi", "Trousers", "Navy Blue", 1, "DISAGREE-WITH-RULES: printed shirt with plain navy"),

    # Rules: two prints score the same as one print.
    # Reality: two prints together is the actual mistake.
    ("Women", "Sarees", "Multi", "Flats", "Multi", 0, "Two busy prints together"),

    # ---- borderline / judgement calls --------------------------------------
    ("Men", "Shirts", "Purple", "Trousers", "Grey", 1, "Purple with grey is restrained enough"),
    ("Men", "Kurtas", "White", "Trousers", "Black", 1, "White kurta over black trousers is normal"),
    ("Men", "Kurtas", "Maroon", "Jeans", "Blue", 1, "Kurta with jeans is a common Indian outfit"),
    ("Women", "Sarees", "Navy Blue", "Heels", "Black", 1, "Navy saree with black heels"),
    ("Women", "Dresses", "Red", "Heels", "Black", 1, "Red dress, black heels"),
    ("Women", "Dresses", "Blue", "Flats", "Silver", 1, "Metallic flats lift a plain dress"),
    ("Men", "Shirts", "Green", "Trousers", "Khaki", 1, "Green and khaki are neighbours"),
    ("Men", "Belts", "Brown", "Formal Shoes", "Brown", 1, "Matching leathers, exactly right"),
    ("Men", "Belts", "Black", "Formal Shoes", "Brown", 0, "Belt and shoes must match in leather"),
    ("Men", "Belts", "Brown", "Trousers", "Navy Blue", 1, "Brown belt with navy"),
    ("Women", "Watches", "Gold", "Dresses", "Black", 1, "Gold watch with a black dress"),
    ("Women", "Watches", "Silver", "Sarees", "Red", 1, "Silver reads as neutral jewellery"),
    ("Men", "Ties", "Navy Blue", "Shirts", "White", 1, "Navy tie on a white shirt"),
    ("Men", "Ties", "Pink", "Shirts", "Red", 0, "Pink tie on a red shirt, too close"),
    ("Women", "Jackets", "Red", "Dresses", "Pink", 0, "Red over pink is uncomfortable"),
    ("Women", "Jackets", "Black", "Dresses", "White", 1, "Black jacket over white dress"),
    ("Men", "Sandals", "Brown", "Shorts", "Navy Blue", 1, "Sandals with shorts is the point of sandals"),
    ("Men", "Sandals", "Brown", "Trousers", "Grey", 0, "Sandals with formal-ish trousers"),
    ("Women", "Heels", "Red", "Trousers", "Black", 1, "A red heel as the single accent"),
    ("Women", "Heels", "Turquoise Blue", "Dresses", "Orange", 0, "Turquoise and orange, both shouting"),
]


def find_product(products: list[dict], gender: str, article_type: str, colour: str):
    """Lowest-id catalog product matching the spec, or None."""
    matches = [
        p for p in products
        if p["gender"] == gender
        and p["articleType"] == article_type
        and p["baseColour"] == colour
    ]
    return min(matches, key=lambda p: int(p["id"])) if matches else None


def main() -> None:
    products = list(catalog_records())
    rows, missing = [], []

    for gender, type_a, colour_a, type_b, colour_b, label, rationale in JUDGEMENTS:
        item_a = find_product(products, gender, type_a, colour_a)
        item_b = find_product(products, gender, type_b, colour_b)
        if item_a is None or item_b is None:
            missing.append(f"{gender} {colour_a} {type_a} + {colour_b} {type_b}")
            continue

        a, b = canonical_order(item_a, item_b)
        rows.append({
            "id_a": int(a["id"]), "id_b": int(b["id"]),
            "desc_a": f"{a['baseColour']} {a['articleType']}",
            "desc_b": f"{b['baseColour']} {b['articleType']}",
            **pair_features(a, b),
            "human_label": label,
            "rationale": rationale,
            "expected_rule_disagreement": int("DISAGREE-WITH-RULES" in rationale),
        })

    frame = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    print(f"  pairs resolved      : {len(frame)} of {len(JUDGEMENTS)}")
    print(f"  human_label = 1     : {int(frame['human_label'].sum())}")
    print(f"  human_label = 0     : {len(frame) - int(frame['human_label'].sum())}")
    print(f"  expected disagreements with rules: "
          f"{int(frame['expected_rule_disagreement'].sum())}")
    if missing:
        print(f"\n  {len(missing)} specs had no catalog match and were skipped:")
        for spec in missing:
            print(f"    - {spec}")
    print("\nLabels are the AUTHOR'S styling judgement, written without consulting "
          "the rule engine. Not user data, not expert annotation.")


if __name__ == "__main__":
    main()
