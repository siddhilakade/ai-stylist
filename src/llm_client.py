"""LLM integration (Groq + Gemini) - the two places an LLM genuinely earns its place.

TASK A - Natural language understanding (before the engine)
    Free text  ->  validated `StylePreferences`
    "something smart casual for my college presentation, nothing flashy, under 3k"
    is trivial for a language model and painful for regular expressions. The
    model gets a closed-vocabulary response schema, so the worst it can do is
    pick the wrong enum value - never an unknown one.

TASK B - Grounded explanation (after the engine)
    Chosen products + computed scores  ->  one short stylist paragraph
    The model is handed the outfit the engine already selected together with the
    real numbers behind it, and asked only to phrase them. It never sees the
    catalog, so it has nothing to hallucinate from.

WHAT GEMINI IS NEVER ASKED TO DO
    Choose products. Score compatibility. Decide a price. Enforce a budget.
    Those are deterministic, testable and reproducible, and an LLM would make
    all three of those properties worse.

SECURITY
    The key is read from the environment or Streamlit secrets - never from
    source, never logged, never rendered. Every response is validated before it
    reaches the UI, and every failure degrades to the rule-based path instead of
    surfacing an error.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.nlu import looks_unparseable, parse_request
from src.schemas import ExtractionResult, ExtractionSource, GeminiExtraction

# A low-latency model: this runs inline in a UI interaction, so a "flash" tier
# model is the right trade-off. Overridable via GEMINI_MODEL without a code
# change, so a newer model can be swapped in from the deployment environment.
DEFAULT_MODEL = "gemini-2.0-flash"
REQUEST_TIMEOUT_SECONDS = 20

# Deterministic extraction: we want the same sentence to produce the same
# preferences every time, which matters for both testing and user trust.
EXTRACTION_TEMPERATURE = 0.0
# The explanation is prose, so a little variation is fine - but not much, since
# it must stay faithful to the numbers it is given.
EXPLANATION_TEMPERATURE = 0.4
EXPLANATION_MAX_TOKENS = 220


class GeminiUnavailable(RuntimeError):
    """No usable API key, or the SDK is not installed."""


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

def get_api_key() -> str | None:
    """Read the key from the environment, then from Streamlit secrets.

    Never from a config file in the repo, and never returned to the UI.
    """
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key and key.strip() and key.strip() != "your_key_here":
        return key.strip()

    return _streamlit_secret("GEMINI_API_KEY")


def _streamlit_secret(name: str) -> str | None:
    """Read one key from Streamlit secrets, when that is safe to attempt."""
    if "streamlit" not in sys.modules or not _secrets_are_readable():
        return None
    try:
        secret = sys.modules["streamlit"].secrets.get(name)
        if secret and str(secret).strip():
            return str(secret).strip()
    except Exception:
        pass
    return None


# Streamlit Community Cloud checks out the repository here. Deployed apps have
# NO secrets.toml on disk - secrets are pasted into the dashboard and injected
# into the running process - so the file check below is never satisfied there.
CLOUD_MOUNT = Path("/mount/src")


def _secrets_are_readable() -> bool:
    """Whether it is safe to touch `st.secrets`.

    Locally, reading `st.secrets` with no secrets.toml anywhere makes Streamlit
    render an error box in the app, and a try/except here cannot suppress it -
    so a file has to exist first.

    On Streamlit Community Cloud there is no such file, and checking only for
    one meant a deployed app ignored its configured key and ran on the keyword
    fallback forever. Detecting the cloud mount is what makes both cases work.
    """
    candidates = (
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    )
    if any(path.is_file() for path in candidates):
        return True
    return CLOUD_MOUNT.exists() or bool(os.environ.get("STREAMLIT_RUNTIME_ENV"))


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
# Two interchangeable LLM providers behind one interface. Gemini is tried first
# when configured; Groq is the working fallback. Both fill exactly the same
# validated `GeminiExtraction` schema, so the rest of the system cannot tell
# which one answered - and neither can bypass the validation.
#
# This exists because it had to: partway through the build, the Gemini key's
# Google Cloud project started returning
#   403 PERMISSION_DENIED - "Your project has been denied access"
# on every model, across two separate keys. Provider choice turned out to be an
# operational risk, not a design detail.

GROQ_MODEL = "llama-3.3-70b-versatile"
PROVIDER_GEMINI = "gemini"
PROVIDER_GROQ = "groq"
# Order matters: first configured, non-tripped provider wins.
PROVIDER_ORDER = (PROVIDER_GEMINI, PROVIDER_GROQ)


def get_groq_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if key and key.strip() and key.strip() != "your_key_here":
        return key.strip()
    return _streamlit_secret("GROQ_API_KEY")


def provider_key(provider: str) -> str | None:
    return get_api_key() if provider == PROVIDER_GEMINI else get_groq_key()


def configured_providers() -> list[str]:
    """Providers that have a key, regardless of whether they currently work."""
    return [p for p in PROVIDER_ORDER if provider_key(p)]


def active_provider() -> str | None:
    """The provider that will actually be used for the next call, if any."""
    for provider in configured_providers():
        if not _circuit_open_for(provider):
            return provider
    return None


def provider_model(provider: str | None) -> str:
    if provider == PROVIDER_GROQ:
        return os.environ.get("GROQ_MODEL", GROQ_MODEL)
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def get_model_name() -> str:
    """Model of whichever provider is currently active (for the UI)."""
    return provider_model(active_provider() or PROVIDER_GEMINI)


@lru_cache(maxsize=1)
def _groq_client() -> Any:
    key = get_groq_key()
    if not key:
        raise GeminiUnavailable("GROQ_API_KEY is not set.")
    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise GeminiUnavailable("groq is not installed.") from exc
    return Groq(api_key=key)


@lru_cache(maxsize=1)
def _client() -> Any:
    key = get_api_key()
    if not key:
        raise GeminiUnavailable(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise GeminiUnavailable("google-genai is not installed.") from exc
    return genai.Client(api_key=key)


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------
# A single page render makes one extraction call plus one explanation call per
# outfit. If the API is down, the key is revoked or the quota is exhausted, that
# is four doomed round-trips and four multi-second stalls for a result we
# already know we will fall back on.
#
# After a couple of consecutive failures we stop calling until something
# changes, and say so once. Retrying a known-dead dependency on every render is
# the difference between "degrades gracefully" and "feels broken".

FAILURE_THRESHOLD = 2

# Tracked PER PROVIDER. A blocked Gemini project must not stop us trying Groq -
# which is exactly the situation this project ended up in.
_circuits: dict[str, dict[str, Any]] = {
    PROVIDER_GEMINI: {"failures": 0, "last_error": None},
    PROVIDER_GROQ: {"failures": 0, "last_error": None},
}


def _circuit_open_for(provider: str) -> bool:
    return _circuits[provider]["failures"] >= FAILURE_THRESHOLD


def circuit_is_open() -> bool:
    """True when NO configured provider is usable - i.e. the LLM path is dead."""
    configured = configured_providers()
    return bool(configured) and all(_circuit_open_for(p) for p in configured)


def circuit_error() -> str | None:
    """The most recent error from any provider, for the debug panel."""
    for provider in PROVIDER_ORDER:
        error = _circuits[provider]["last_error"]
        if error:
            return f"[{provider}] {error}"
    return None


def reset_circuit() -> None:
    """Called by the UI's 'Retry' control - re-arms every provider."""
    for state in _circuits.values():
        state["failures"] = 0
        state["last_error"] = None


# What a normal user is told when the LLM is down. A shopper does not need -
# and should not see - an HTTP status code or a provider stack trace; they need
# to know the app still works. The technical detail stays in `circuit_error()`
# for the debug panel and is never rendered inline.
FRIENDLY_UNAVAILABLE = (
    "AI understanding is temporarily unavailable — using the built-in "
    "recommendation engine instead."
)


def _record_failure(exc: Exception, provider: str = PROVIDER_GEMINI) -> None:
    _circuits[provider]["failures"] += 1
    _circuits[provider]["last_error"] = _safe_error(exc)


def _record_success(provider: str = PROVIDER_GEMINI) -> None:
    _circuits[provider]["failures"] = 0
    _circuits[provider]["last_error"] = None


def has_api_key() -> bool:
    """Whether ANY provider is configured, regardless of whether it works."""
    return bool(configured_providers())


def is_available() -> bool:
    """Whether the LLM path is worth attempting right now."""
    return has_api_key() and not circuit_is_open()


def check_connection() -> dict[str, Any]:
    """Diagnostic used by the 'Test connection' control.

    Tries every configured provider so the UI can report which one actually
    works, rather than just that "the LLM" is down.
    """
    providers = configured_providers()
    if not providers:
        return {"ok": False, "detail": "No API key found in environment or secrets."}

    attempts: list[dict[str, Any]] = []
    for provider in providers:
        started = time.perf_counter()
        try:
            if provider == PROVIDER_GROQ:
                reply = _groq_client().chat.completions.create(
                    model=provider_model(provider),
                    messages=[{"role": "user",
                               "content": "Reply with the single word: ok"}],
                    temperature=0.0, max_tokens=10,
                ).choices[0].message.content or ""
            else:
                from google.genai import types

                reply = _client().models.generate_content(
                    model=provider_model(provider),
                    contents="Reply with the single word: ok",
                    config=types.GenerateContentConfig(
                        temperature=0.0, max_output_tokens=10),
                ).text or ""
            _record_success(provider)
            return {
                "ok": True,
                "provider": provider,
                "model": provider_model(provider),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "reply": reply.strip()[:40],
                "attempts": attempts,
            }
        except Exception as exc:
            _record_failure(exc, provider)
            attempts.append({"provider": provider,
                             "model": provider_model(provider),
                             "detail": _safe_error(exc)})

    return {"ok": False, "detail": "; ".join(
        f"{a['provider']}: {a['detail'][:120]}" for a in attempts
    ), "attempts": attempts}


def _safe_error(exc: Exception) -> str:
    """Error text with anything key-shaped stripped out.

    Some SDK errors echo the request URL, which carries the API key as a query
    parameter. This must never reach a log line or the screen.
    """
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"(key=)[A-Za-z0-9_\-]+", r"\1***", text)
    text = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "***", text)
    return text[:400]


# --------------------------------------------------------------------------
# TASK A - preference extraction
# --------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You convert a shopper's free-text request into structured styling preferences \
for an Indian fashion marketplace.

Rules:
- Fill every field. Use the sentinel values when something is not stated:
  style="unspecified", budget_inr=0, preferred_colors=[], notes="".
- budget_inr is the total budget for the WHOLE outfit, in Indian rupees, as a \
plain integer. "3k" means 3000. If no budget is mentioned, use 0.
- Infer the occasion from context: a presentation or viva is "interview", a \
shaadi or festival is "wedding_festive", the gym is "workout".
- Only set `style` when the user actually indicates a formality level. Do not \
infer it from the occasion; the system already knows what each occasion implies.
- There are TWO different kinds of colour statement and they go in DIFFERENT \
fields. Getting this wrong is the most damaging mistake you can make:
    * A colour attached to a specific garment is a HARD requirement. \
"a black shirt" -> required_items: [{"garment": "shirt", "colour": "black"}], \
and preferred_colors stays EMPTY.
    * A general palette wish is a SOFT preference. "I like neutral tones" -> \
preferred_colors: ["neutral"], and required_items stays empty.
- required_items also captures a garment named with no colour: "I need jeans" \
-> [{"garment": "jeans", "colour": ""}]. Only list garments the user actually \
asked for; never add garments to complete an outfit yourself.
- preferred_colors holds colour FAMILIES, not shades: "neutrals" -> ["neutral"], \
"navy tones" -> ["blue"]. required_items colours are specific words like \
"black", "white", "navy", "red".
- Set include_accessory to false only if the user says they do not want one.
- notes: any other stated preference in a few words (e.g. "nothing flashy", \
"prefers loose fit"). Empty string if there is nothing.
- You are NOT choosing products. Never name a brand, a product or a price. \
required_items records the KIND of garment the user asked for, never a \
particular item from a catalogue - you cannot see the catalogue.
"""


def extract_preferences(
    text: str,
    default_gender: str = "Women",
    allow_llm: bool = True,
) -> ExtractionResult:
    """Free text -> validated preferences, via Gemini with a rule-based fallback.

    Every path returns a valid `StylePreferences`; the `source` field records
    which path produced it so the UI can be honest about it.
    """
    text = (text or "").strip()
    rule_based = parse_request(text, default_gender=default_gender)

    if not text:
        return ExtractionResult(
            preferences=rule_based,
            source=ExtractionSource.RULE_BASED,
            warning="No request text provided - showing a general recommendation.",
        )

    if not allow_llm or not is_available():
        if not allow_llm:
            warning = None
        elif circuit_is_open():
            warning = FRIENDLY_UNAVAILABLE
        else:
            warning = (
                "AI understanding is not configured — using the built-in "
                "recommendation engine."
            )
        return ExtractionResult(
            preferences=rule_based,
            source=ExtractionSource.RULE_BASED,
            warning=warning,
            degraded=True,
        )

    # Try each configured provider in order. A provider that fails trips only
    # its own breaker, so a blocked Gemini project falls through to Groq rather
    # than taking the whole LLM path down with it.
    raw, used_provider = None, None
    for provider in configured_providers():
        if _circuit_open_for(provider):
            continue
        try:
            raw = (
                _call_groq_extraction(text, default_gender)
                if provider == PROVIDER_GROQ
                else _call_gemini_extraction(text, default_gender)
            )
            _record_success(provider)
            used_provider = provider
            break
        except Exception as exc:
            _record_failure(exc, provider)

    if raw is None:
        return ExtractionResult(
            preferences=rule_based,
            source=ExtractionSource.RULE_BASED,
            warning=FRIENDLY_UNAVAILABLE,
            degraded=True,
        )

    # Validate before trusting. An invalid or malformed response is discarded
    # entirely rather than partially salvaged - a half-parsed preference set is
    # worse than a keyword-parsed one, because it looks authoritative.
    try:
        extraction = GeminiExtraction.model_validate_json(raw)
        preferences = extraction.to_preferences()
    except Exception as exc:
        return ExtractionResult(
            preferences=rule_based,
            source=ExtractionSource.RULE_BASED,
            raw_model_output=raw[:500],
            warning=(
                "The language model returned a response that did not match the "
                f"expected schema, so keyword rules were used. ({_safe_error(exc)})"
            ),
            degraded=True,
        )

    warning = None
    if looks_unparseable(text, preferences) and preferences.budget is None:
        warning = (
            "This request was hard to interpret - the result may not reflect "
            "what you meant. Try naming an occasion, a budget or a colour."
        )

    return ExtractionResult(
        preferences=preferences,
        source=ExtractionSource.GEMINI,
        provider=used_provider,
        model=provider_model(used_provider),
        raw_model_output=raw[:500],
        warning=warning,
    )


@lru_cache(maxsize=128)
def _call_groq_extraction(text: str, default_gender: str) -> str:
    """Same contract as the Gemini call: raw JSON matching GeminiExtraction.

    Groq exposes JSON mode rather than a typed response-schema parameter, so the
    schema is embedded in the system prompt and the result is validated by the
    same Pydantic model. Validation - not the provider - is what guarantees the
    output is safe to use.
    """
    schema = json.dumps(GeminiExtraction.model_json_schema())
    response = _groq_client().chat.completions.create(
        model=provider_model(PROVIDER_GROQ),
        messages=[
            {"role": "system",
             "content": f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
                        f"Reply with JSON only, matching this schema: {schema}"},
            {"role": "user",
             "content": f"Shopper request: {text!r}\nIf the request does not say "
                        f"who it is for, assume {default_gender}."},
        ],
        response_format={"type": "json_object"},
        temperature=EXTRACTION_TEMPERATURE,
    )
    return (response.choices[0].message.content or "").strip()


@lru_cache(maxsize=128)
def _call_groq_explanation(payload_json: str) -> str:
    response = _groq_client().chat.completions.create(
        model=provider_model(PROVIDER_GROQ),
        messages=[
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": payload_json},
        ],
        temperature=EXPLANATION_TEMPERATURE,
        max_tokens=EXPLANATION_MAX_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()


@lru_cache(maxsize=128)
def _call_gemini_extraction(text: str, default_gender: str) -> str:
    """Raw JSON string from Gemini. Cached: identical text costs one call."""
    from google.genai import types

    response = _client().models.generate_content(
        model=get_model_name(),
        contents=(
            f"Shopper request: {text!r}\n"
            f"If the request does not say who it is for, assume {default_gender}."
        ),
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            temperature=EXTRACTION_TEMPERATURE,
            response_mime_type="application/json",
            # Structured output: the SDK turns this Pydantic model into a
            # response schema, and the API constrains decoding to match it.
            response_schema=GeminiExtraction,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000),
        ),
    )
    return (response.text or "").strip()


# --------------------------------------------------------------------------
# TASK B - grounded explanation
# --------------------------------------------------------------------------

EXPLANATION_SYSTEM_PROMPT = """\
You are a concise personal stylist writing one short paragraph for a shopper.

You will be given a JSON object containing the outfit that has ALREADY been \
chosen for them, and the scores the styling engine computed for it. Your only \
job is to explain, in natural language, why these pieces work together for this \
request.

Hard rules:
- Use ONLY the items, colours and prices in the JSON. Do not add, swap or \
suggest any other garment.
- Do not invent or alter any price. If you mention money, use only \
total_price_inr or an item's exact price_inr.
- Do not invent brand names, fabrics, fits, materials or product details that \
are not in the JSON.
- Refer to items by their garment type and colour ("the black shirt"), not by \
their full catalogue name.
- 2 to 4 sentences. Warm and practical, not salesy. No bullet points, no \
headings, no emoji.
- Mention the budget only if budget_inr is present and not null.
"""


@dataclass
class ExplanationResult:
    text: str
    source: str  # "gemini" | "template"
    latency_ms: float = 0.0
    warning: str | None = None


def explain_outfit(
    payload: dict[str, Any],
    allow_llm: bool = True,
) -> ExplanationResult:
    """Turn the engine's grounding payload into a stylist paragraph.

    `payload` comes from `recommender.grounding_payload` and contains the real
    selected products and the real scores. If Gemini is unavailable or its
    output fails validation, we fall back to a sentence assembled from the same
    facts - so the user always gets a grounded explanation.
    """
    template = _template_explanation(payload)

    if not allow_llm or not is_available():
        return ExplanationResult(text=template, source="template")

    started = time.perf_counter()
    payload_json = json.dumps(payload, sort_keys=True)
    text, last_error = None, None
    for provider in configured_providers():
        if _circuit_open_for(provider):
            continue
        try:
            text = (
                _call_groq_explanation(payload_json)
                if provider == PROVIDER_GROQ
                else _call_gemini_explanation(payload_json)
            )
            _record_success(provider)
            break
        except Exception as exc:
            _record_failure(exc, provider)
            last_error = _safe_error(exc)

    if text is None:
        return ExplanationResult(
            text=template, source="template", warning=last_error
        )

    ok, problem = validate_explanation(text, payload)
    if not ok:
        return ExplanationResult(
            text=template,
            source="template",
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            warning=f"Gemini output rejected by the grounding check: {problem}",
        )

    return ExplanationResult(
        text=text.strip(),
        source="gemini",
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@lru_cache(maxsize=128)
def _call_gemini_explanation(payload_json: str) -> str:
    from google.genai import types

    response = _client().models.generate_content(
        model=get_model_name(),
        contents=payload_json,
        config=types.GenerateContentConfig(
            system_instruction=EXPLANATION_SYSTEM_PROMPT,
            temperature=EXPLANATION_TEMPERATURE,
            max_output_tokens=EXPLANATION_MAX_TOKENS,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000),
        ),
    )
    return (response.text or "").strip()


# --------------------------------------------------------------------------
# Output validation - defence in depth
# --------------------------------------------------------------------------
# The architecture already makes product hallucination structurally impossible:
# Gemini never sees the catalog and never returns an id, so the UI can only
# render products the engine selected. These checks guard the remaining surface
# - the prose itself - against invented prices and links.

# A figure is either explicitly a currency amount, or a bare 3-6 digit number.
# Percentages are excluded: the grounding payload contains score percentages, and
# an explanation is allowed to quote them.
_NUMBER_PATTERN = re.compile(
    r"(?:rs\.?|inr|₹)\s*([\d,]+)|\b(\d{3,6})\b(?!\s*%)", re.I
)
_URL_PATTERN = re.compile(r"https?://|www\.", re.I)
MAX_EXPLANATION_CHARS = 900


def validate_explanation(text: str, payload: dict[str, Any]) -> tuple[bool, str]:
    """Reject an explanation that states a number we did not give it."""
    if not text or not text.strip():
        return False, "empty response"
    if len(text) > MAX_EXPLANATION_CHARS:
        return False, "response too long"
    if _URL_PATTERN.search(text):
        return False, "response contains a link"

    allowed: set[int] = {int(payload.get("total_price_inr", 0))}
    for item in payload.get("selected_items", []):
        allowed.add(int(item.get("price_inr", 0)))
    budget = (payload.get("request") or {}).get("budget_inr")
    if budget:
        allowed.add(int(budget))
    # Also allow any figure that literally appears in the payload we sent, so a
    # product name like "Puma 500" quoted back to us is not a false positive.
    allowed.update(
        int(n) for n in re.findall(r"\d{3,6}", json.dumps(payload))
    )

    for match in _NUMBER_PATTERN.finditer(text):
        raw = match.group(1) or match.group(2)
        try:
            value = int(raw.replace(",", ""))
        except ValueError:
            continue
        if value not in allowed:
            return False, f"mentions the unsupported figure {value}"

    return True, ""


# --------------------------------------------------------------------------
# Template fallback
# --------------------------------------------------------------------------

# Article types are stored as plurals ("Shirts", "Casual Shoes"). Prose needs
# the singular, except for garments that are inherently a pair.
_ALWAYS_PLURAL = {"jeans", "trousers", "shorts", "shoes", "flats", "heels",
                  "sandals", "flip flops", "track pants", "leggings", "capris"}


def _singular(article_type: str) -> str:
    lowered = article_type.lower()
    if lowered in _ALWAYS_PLURAL or not lowered.endswith("s"):
        return lowered
    return lowered[:-1]


def _with_article(colour: str, noun: str) -> str:
    """"black shirt" -> "a black shirt"; "silver flats" -> "silver flats"."""
    phrase = f"{colour} {noun}"
    if noun in _ALWAYS_PLURAL:
        return phrase  # a pair of trousers takes no indefinite article
    return f"{'an' if colour[:1] in 'aeiou' else 'a'} {phrase}"


def _join_naturally(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _template_explanation(payload: dict[str, Any]) -> str:
    """A grounded explanation with no LLM involved at all.

    Built from the same facts Gemini would receive, so the app never has to show
    an error where an explanation should be.
    """
    items = payload.get("selected_items", [])
    if not items:
        return "No outfit was assembled for this request."

    described = _join_naturally([
        _with_article(item["colour"].lower(), _singular(item["article_type"]))
        for item in items
    ])
    occasion = (payload.get("request") or {}).get("occasion", "this occasion")
    signals = payload.get("signals", {})
    total = payload.get("total_price_inr", 0)

    return (
        f"This look brings together {described}, chosen for {occasion}. "
        f"The pieces sit at a consistent formality level "
        f"({signals.get('Formality consistency', 0):.0%} match) and the colours "
        f"coordinate cleanly ({signals.get('Colour coordination', 0):.0%}). "
        f"The full outfit comes to ₹{total:,}."
    )
