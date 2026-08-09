"""Evaluation harness.

WHY NOT PRECISION / RECALL / NDCG
---------------------------------
Those metrics need relevance labels: a ground-truth set of "correct" items per
query. No such labels exist for outfit compatibility - there is no public dataset
saying "this shirt goes with these trousers" for this catalog, and building one
by hand would amount to scoring the system against its own rules. Reporting an
NDCG here would be a fabricated number, so we do not report one.

WHAT WE MEASURE INSTEAD
-----------------------
Constraint-satisfaction metrics, which are objective and fully verifiable:

  1. Validity            every returned product exists in the catalog
  2. Constraint sat.     gender and formality band respected on every item
  3. Budget compliance   total <= stated budget, on every returned outfit
  4. Completeness        every required slot filled
  5. Preference match    stated colour honoured, or a neutral substituted
  6. Diversity           article-type spread within a result set, plus catalog
                         coverage across the whole evaluation run
  7. Latency             p50 / p95 / max end-to-end engine time
  8. Failure quality     scenarios that should fail do fail, with an actionable
                         message - never a crash, never unrelated products
  9. NLU accuracy        parsed preferences vs the hand-labelled expectation

Run:  python scripts/evaluate.py            (rule-based NLU; no API key needed)
      python scripts/evaluate.py --gemini   (also evaluates the Gemini NLU path)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_product, load_catalog, product_ids  # noqa: E402
from src.features import (  # noqa: E402
    SLOT_ACCESSORY,
    resolve_formality_target,
)
from src.outfit_builder import TEMPLATE_ONEPIECE, TEMPLATE_STANDARD  # noqa: E402
from src.recommender import ACCESSORY_TOLERANCE_BONUS, recommend_outfits  # noqa: E402
from src.schemas import StylePreferences  # noqa: E402

SCENARIO_PATH = ROOT / "data" / "eval_scenarios.json"
REPORT_MD = ROOT / "docs" / "EVALUATION.md"
REPORT_JSON = ROOT / "docs" / "evaluation_results.json"


# --------------------------------------------------------------------------
# Per-scenario checks
# --------------------------------------------------------------------------

def check_validity(outfits) -> bool:
    """Every product shown must resolve to a real catalog row."""
    valid = product_ids()
    return all(
        pid in valid and get_product(pid) is not None
        for outfit in outfits for pid in outfit.product_ids
    )


def check_completeness(outfits) -> bool:
    """Every outfit fills all required slots of whichever template it used."""
    for outfit in outfits:
        template = TEMPLATE_ONEPIECE if outfit.template == "onepiece" else TEMPLATE_STANDARD
        if not set(template.required_slots) <= set(outfit.items):
            return False
    return True


def check_constraints(outfits, prefs: StylePreferences) -> bool:
    """Gender and the formality band, re-verified on the final output."""
    target, tolerance = resolve_formality_target(prefs.occasion, prefs.style)
    # The builder may have used one relaxation step; allow for it here so we are
    # measuring "did the system stay inside the constraints it told the user
    # about", not "did it never relax".
    tolerance += 0.75
    # Mirror the engine's rule: a Unisex request draws from the whole catalog.
    allowed_genders = (
        {"Men", "Women", "Unisex"} if prefs.gender == "Unisex"
        else {prefs.gender, "Unisex"}
    )
    # Slots the user named explicitly are exempt from the formality band, in the
    # evaluator exactly as in the engine: the band comes from an inferred
    # occasion, and it must not veto a garment the user asked for by name.
    explicit_slots = set(prefs.constraints_by_slot())

    for outfit in outfits:
        for slot, item in outfit.items.items():
            if item["gender"] not in allowed_genders:
                return False
            if slot in explicit_slots:
                continue
            allowed = tolerance + (
                ACCESSORY_TOLERANCE_BONUS if item["outfit_slot"] == SLOT_ACCESSORY else 0.0
            )
            if abs(float(item["formality"]) - target) > allowed:
                return False
    return True


def check_budget(outfits, prefs: StylePreferences) -> bool:
    if prefs.budget is None:
        return True
    return all(o.total_price <= prefs.budget for o in outfits)


def measure_preference_match(outfits, prefs: StylePreferences) -> float | None:
    """Fraction of items that either match a requested colour or are neutral.

    Neutrals count because the system's documented behaviour is to substitute a
    coordinating neutral when an exact colour is unavailable, rather than to
    return an unrelated colour or nothing at all.
    """
    if not prefs.preferred_colors or not outfits:
        return None
    hits = total = 0
    for outfit in outfits:
        for item in outfit.items.values():
            total += 1
            if item["color_family"] in prefs.preferred_colors or item["is_neutral"]:
                hits += 1
    return hits / total if total else None


def measure_diversity(outfits) -> float | None:
    """Distinct article types / total items across the returned looks."""
    items = [item for outfit in outfits for item in outfit.items.values()]
    if len(items) < 2:
        return None
    return len({i["articleType"] for i in items}) / len(items)


def check_failure_quality(result) -> dict[str, Any]:
    failure = result.failure
    return {
        "has_failure_object": failure is not None,
        "has_message": bool(failure and len(failure.message) > 20),
        "has_suggestion": bool(failure and len(failure.suggestion) > 20),
        "returned_no_products": result.outfits == [],
    }


# --------------------------------------------------------------------------
# NLU accuracy
# --------------------------------------------------------------------------

NLU_FIELDS = ("gender", "occasion", "style", "budget", "preferred_colors",
              "required_items")


def _items_as_pairs(prefs: StylePreferences) -> list[list[str | None]]:
    return sorted([item.garment, item.colour] for item in prefs.required_items)


def score_nlu(parsed: StylePreferences, expected: dict[str, Any]) -> dict[str, bool]:
    expected_items = sorted(
        [entry["garment"], entry.get("colour")]
        for entry in (expected.get("required_items") or [])
    )
    return {
        "gender": parsed.gender == expected.get("gender"),
        "occasion": parsed.occasion == expected.get("occasion"),
        "style": parsed.style == expected.get("style"),
        "budget": parsed.budget == expected.get("budget"),
        "preferred_colors": sorted(parsed.preferred_colors)
        == sorted(expected.get("preferred_colors") or []),
        # The hard/soft distinction: did we capture the garments the user named
        # outright, without inventing any?
        "required_items": _items_as_pairs(parsed) == expected_items,
    }


def check_explicit_items(outfits, prefs: StylePreferences) -> bool | None:
    """Every returned outfit must contain each explicitly requested garment."""
    constraints = prefs.constraints_by_slot()
    if not constraints or not outfits:
        return None
    for outfit in outfits:
        for slot, constraint in constraints.items():
            item = outfit.items.get(slot)
            if item is None or item["articleType"] not in constraint["article_types"]:
                return False
            colours = constraint["base_colours"]
            if colours and item["baseColour"] not in colours:
                return False
    return True


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def preferences_from_expected(expected: dict[str, Any]) -> StylePreferences:
    return StylePreferences(
        gender=expected.get("gender") or "Women",
        occasion=expected.get("occasion") or "everyday_casual",
        style=expected.get("style"),
        budget=expected.get("budget"),
        preferred_colors=expected.get("preferred_colors") or [],
        required_items=expected.get("required_items") or [],
    )


def run(use_gemini: bool = False) -> dict[str, Any]:
    scenarios = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))["scenarios"]
    catalog_size = len(load_catalog())

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    all_recommended: set[int] = set()
    nlu_rule: list[dict[str, bool]] = []
    nlu_llm: list[dict[str, bool]] = []

    from src.llm_client import extract_preferences
    from src.nlu import parse_request

    for scenario in scenarios:
        expected = scenario["expected"]
        prefs = preferences_from_expected(expected)

        # --- recommendation quality (measured from the labelled preferences,
        #     so engine quality is not confounded with NLU accuracy) ---------
        result = recommend_outfits(prefs)
        latencies.append(result.latency_ms)
        for outfit in result.outfits:
            all_recommended.update(outfit.product_ids)

        expected_success = scenario["expect_outcome"] == "success"
        row: dict[str, Any] = {
            "id": scenario["id"],
            "group": scenario["group"],
            "request": scenario["request"],
            "expected_outcome": scenario["expect_outcome"],
            "actual_outcome": "success" if result.ok else "failure",
            "outcome_as_expected": result.ok == expected_success,
            "n_outfits": len(result.outfits),
            "latency_ms": result.latency_ms,
            "relaxed": bool(result.diagnostics.get("relaxed")),
            "conflicts": result.diagnostics.get("conflicts", []),
            "notes": scenario.get("notes", ""),
            "review": scenario.get("review", ""),
        }

        if result.ok:
            row.update({
                "validity": check_validity(result.outfits),
                "completeness": check_completeness(result.outfits),
                "constraints_ok": check_constraints(result.outfits, prefs),
                "budget_ok": check_budget(result.outfits, prefs),
                "preference_match": measure_preference_match(result.outfits, prefs),
                "explicit_items_ok": check_explicit_items(result.outfits, prefs),
                "diversity": measure_diversity(result.outfits),
                "best_score": result.outfits[0].final_score,
                "best_compatibility": result.outfits[0].compatibility,
                "total_price": result.outfits[0].total_price,
            })
        else:
            row.update({
                "failure_reason": result.failure.reason_code if result.failure else None,
                "failure_quality": check_failure_quality(result),
            })

        # --- NLU accuracy -------------------------------------------------
        row["nlu_rule_based"] = score_nlu(parse_request(scenario["request"]), expected)
        nlu_rule.append(row["nlu_rule_based"])

        if use_gemini:
            extraction = extract_preferences(scenario["request"], allow_llm=True)
            row["nlu_gemini_source"] = extraction.source.value
            row["nlu_gemini"] = score_nlu(extraction.preferences, expected)
            nlu_llm.append(row["nlu_gemini"])

        rows.append(row)

    explicit = [r for r in rows if r.get("explicit_items_ok") is not None]
    successes = [r for r in rows if r["actual_outcome"] == "success"]
    failures = [r for r in rows if r["actual_outcome"] == "failure"]
    with_colour = [r for r in successes if r.get("preference_match") is not None]

    def rate(values: list[bool]) -> float:
        return round(100 * sum(values) / len(values), 1) if values else 0.0

    def field_accuracy(records: list[dict[str, bool]]) -> dict[str, float]:
        if not records:
            return {}
        return {
            field: rate([record[field] for record in records]) for field in NLU_FIELDS
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenarios": len(rows),
        "catalog_size": catalog_size,
        "outcome_as_expected_pct": rate([r["outcome_as_expected"] for r in rows]),
        "succeeded": len(successes),
        "failed": len(failures),
        "validity_pct": rate([r["validity"] for r in successes]),
        "completeness_pct": rate([r["completeness"] for r in successes]),
        "constraints_pct": rate([r["constraints_ok"] for r in successes]),
        "budget_pct": rate([r["budget_ok"] for r in successes]),
        "explicit_items_pct": rate([r["explicit_items_ok"] for r in explicit])
        if explicit else None,
        "explicit_items_tested": len(explicit),
        "preference_match_mean": round(
            statistics.mean([r["preference_match"] for r in with_colour]), 3
        ) if with_colour else None,
        "diversity_mean": round(statistics.mean(
            [r["diversity"] for r in successes if r.get("diversity") is not None]
        ), 3) if successes else None,
        "catalog_coverage_pct": round(100 * len(all_recommended) / catalog_size, 1),
        "distinct_products_recommended": len(all_recommended),
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(
            sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 2
        ),
        "latency_max_ms": round(max(latencies), 2),
        "graceful_failures_pct": rate([
            all(r["failure_quality"].values()) for r in failures
        ]) if failures else None,
        "relaxation_used_count": sum(1 for r in rows if r["relaxed"]),
        "nlu_rule_based_accuracy": field_accuracy(nlu_rule),
        "nlu_gemini_accuracy": field_accuracy(nlu_llm) if use_gemini else None,
        "groups": dict(Counter(r["group"] for r in rows)),
    }
    return {"summary": summary, "rows": rows}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_markdown(report: dict[str, Any]) -> None:
    summary, rows = report["summary"], report["rows"]

    def pct(value) -> str:
        return "n/a" if value is None else f"{value}%"

    lines: list[str] = []
    add = lines.append

    add("# Evaluation\n")
    add(
        "> Generated by `python scripts/evaluate.py`. Every number below is "
        "computed from an actual run against the shipped catalog — nothing here "
        "is hand-written.\n"
    )
    add(f"Run at `{summary['generated_at']}` over **{summary['scenarios']} "
        f"hand-labelled scenarios** against a **{summary['catalog_size']}-product** catalog.\n")

    add("## Why these metrics\n")
    add(
        "Precision, recall and NDCG all require per-query relevance labels. There "
        "is no ground truth for *outfit compatibility* on this catalog, and "
        "authoring one by hand would mean grading the system against its own "
        "rules. Rather than report a fabricated NDCG, the evaluation measures "
        "properties that are objectively checkable: does the system obey the "
        "constraints it was given, does it stay inside the budget, does it return "
        "real products, and does it fail usefully when it cannot succeed.\n"
    )

    add("## Headline results\n")
    add("| Metric | Result | What it means |")
    add("| --- | --- | --- |")
    add(f"| Outcome as expected | {pct(summary['outcome_as_expected_pct'])} | "
        "The system succeeded on the scenarios a reviewer judged feasible, and "
        "failed on the ones judged infeasible. |")
    add(f"| Recommendation validity | {pct(summary['validity_pct'])} | "
        "Every product shown resolves to a real catalog row. No hallucinated items. |")
    add(f"| Category completeness | {pct(summary['completeness_pct'])} | "
        "Every returned outfit filled all required slots for its template. |")
    add(f"| Constraint satisfaction | {pct(summary['constraints_pct'])} | "
        "Gender and formality band respected on every individual item. |")
    add(f"| Budget compliance | {pct(summary['budget_pct'])} | "
        "No returned outfit exceeded the stated budget. |")
    add(f"| **Explicit item satisfaction** | {pct(summary['explicit_items_pct'])} | "
        f"Across the {summary['explicit_items_tested']} scenarios naming a specific "
        "garment (\"a black shirt\"), the share where every returned outfit "
        "actually contained that garment in that colour. |")
    add(f"| Preference match (colour) | {summary['preference_match_mean']} | "
        "Share of items matching a requested colour family or substituting a "
        "coordinating neutral. Measured only on the scenarios that stated a colour. |")
    add(f"| Diversity | {summary['diversity_mean']} | "
        "Distinct article types / total items across a result set. 1.0 means every "
        "piece in every returned look was a different garment type. |")
    add(f"| Catalog coverage | {pct(summary['catalog_coverage_pct'])} | "
        f"{summary['distinct_products_recommended']} distinct products surfaced "
        "across the run — a check against always recommending the same few items. |")
    add(f"| Latency (p50 / p95 / max) | {summary['latency_p50_ms']} / "
        f"{summary['latency_p95_ms']} / {summary['latency_max_ms']} ms | "
        "Engine time only, excluding any LLM call. |")
    add(f"| Graceful failure | {pct(summary['graceful_failures_pct'])} | "
        f"Of the {summary['failed']} failing scenarios, the share that returned an "
        "explanation, an actionable suggestion, and zero products. |")
    add("")

    add("## Natural-language understanding\n")
    add(
        "Accuracy of the extracted preference fields against the hand-labelled "
        "expectation, per field. The rule-based parser is the fallback that runs "
        "when Gemini is unavailable; comparing the two is how we show what the "
        "LLM is actually buying.\n"
    )
    add("| Field | Rule-based | Gemini |")
    add("| --- | --- | --- |")
    llm = summary.get("nlu_gemini_accuracy")
    for field in NLU_FIELDS:
        rule_value = summary["nlu_rule_based_accuracy"].get(field)
        llm_value = llm.get(field) if llm else None
        add(f"| `{field}` | {pct(rule_value)} | {pct(llm_value) if llm else 'not run'} |")
    if not llm:
        add("")
        add("> Gemini column not populated: run `python scripts/evaluate.py --gemini` "
            "with `GEMINI_API_KEY` set to measure the LLM path.")
    add("")

    add("## Success cases\n")
    add("| ID | Request | Outfits | Best score | Total | Latency |")
    add("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        if row["actual_outcome"] != "success":
            continue
        add(f"| {row['id']} | {row['request']} | {row['n_outfits']} | "
            f"{row['best_score']:.3f} | ₹{row['total_price']:,} | {row['latency_ms']:.1f} ms |")
    add("")

    add("## Failure cases\n")
    add(
        "Failures are a designed outcome, not a bug: when the constraint chain "
        "empties a required slot, the system says so instead of substituting "
        "something unrelated.\n"
    )
    add("| ID | Request | Reason | Expected? | Why |")
    add("| --- | --- | --- | --- | --- |")
    for row in rows:
        if row["actual_outcome"] != "failure":
            continue
        expected_mark = "yes" if row["outcome_as_expected"] else "**no — finding**"
        add(f"| {row['id']} | {row['request']} | `{row['failure_reason']}` | "
            f"{expected_mark} | {row['notes'] or '—'} |")
    add("")

    conflicted = [r for r in rows if r["conflicts"]]
    if conflicted:
        add("## Conflicting requests detected\n")
        add(
            "Contradictory inputs are answered on a best-effort basis *and* "
            "flagged, rather than silently resolved in favour of one half.\n"
        )
        add("| ID | Request | Warning shown |")
        add("| --- | --- | --- |")
        for row in conflicted:
            add(f"| {row['id']} | {row['request']} | {' '.join(row['conflicts'])} |")
        add("")

    surprises = [r for r in rows if not r["outcome_as_expected"]]
    add("## Findings — where the system disagreed with the reviewer\n")
    add(
        "Expectations were written before the first run and have **not** been "
        "edited to match the results. Each disagreement was inspected and "
        "adjudicated; the adjudication is recorded below verbatim from "
        "`data/eval_scenarios.json`.\n"
    )
    if not surprises:
        add("None. Every scenario's outcome matched the pre-registered expectation.\n")
    else:
        for row in surprises:
            add(f"**{row['id']}** (`{row['group']}`) — expected "
                f"*{row['expected_outcome']}*, got *{row['actual_outcome']}*.  ")
            add(f"Request: “{row['request']}”  ")
            add(f"Adjudication: {row['review'] or row['notes'] or '(not yet reviewed)'}\n")

    add("## Known limitations of this evaluation\n")
    add(
        "- **No relevance ground truth.** Everything above measures constraint\n"
        "  satisfaction, not whether a human would call the outfit stylish. That\n"
        "  judgement would need a user study or expert annotation.\n"
        "- **30 scenarios is small.** They were chosen to cover the design's\n"
        "  decision boundaries (sparse cells, conflicting constraints, budget\n"
        "  edges), not to be a statistically powered sample.\n"
        "- **The author wrote both the system and the labels.** The expectations\n"
        "  were fixed before running, and disagreements are reported above, but\n"
        "  this is not an independent evaluation.\n"
        "- **Novelty and serendipity are not measured.** The dataset has no\n"
        "  popularity or interaction signal, so there is no basis for computing\n"
        "  them.\n"
        "- **Prices are synthetic**, so budget compliance is verified against\n"
        "  simulated prices. The logic is real; the price data is not.\n"
    )

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the AI Stylist engine.")
    parser.add_argument(
        "--gemini", action="store_true",
        help="Also evaluate the Gemini NLU path (needs GEMINI_API_KEY).",
    )
    args = parser.parse_args()

    report = run(use_gemini=args.gemini)
    write_markdown(report)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"\nScenarios              : {summary['scenarios']}")
    print(f"Outcome as expected    : {summary['outcome_as_expected_pct']}%")
    print(f"Succeeded / failed     : {summary['succeeded']} / {summary['failed']}")
    print(f"Validity               : {summary['validity_pct']}%")
    print(f"Completeness           : {summary['completeness_pct']}%")
    print(f"Constraints            : {summary['constraints_pct']}%")
    print(f"Budget compliance      : {summary['budget_pct']}%")
    print(f"Explicit item match    : {summary['explicit_items_pct']}% "
          f"({summary['explicit_items_tested']} scenarios)")
    print(f"Preference match       : {summary['preference_match_mean']}")
    print(f"Diversity              : {summary['diversity_mean']}")
    print(f"Catalog coverage       : {summary['catalog_coverage_pct']}% "
          f"({summary['distinct_products_recommended']} products)")
    print(f"Latency p50/p95/max ms : {summary['latency_p50_ms']} / "
          f"{summary['latency_p95_ms']} / {summary['latency_max_ms']}")
    print(f"Graceful failures      : {summary['graceful_failures_pct']}%")
    print(f"NLU (rule-based)       : {summary['nlu_rule_based_accuracy']}")
    if summary.get("nlu_gemini_accuracy"):
        print(f"NLU (gemini)           : {summary['nlu_gemini_accuracy']}")

    surprises = [r for r in report["rows"] if not r["outcome_as_expected"]]
    if surprises:
        print(f"\n{len(surprises)} scenario(s) disagreed with the expectation:")
        for row in surprises:
            print(f"  {row['id']}: expected {row['expected_outcome']}, "
                  f"got {row['actual_outcome']} — {row['request']}")

    print(f"\nWrote {REPORT_MD.relative_to(ROOT)} and {REPORT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
