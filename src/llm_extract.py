"""LLM-based outcome extraction with caching and cost/latency tracking.

Providers: "anthropic", "openai", and "mock" (offline, for testing the pipeline only).
Every response is cached in cache/<model>/<call_id>.json, so re-running the evaluation
after changing metrics costs nothing. Delete the folder to force fresh calls.
"""
from __future__ import annotations

import json
import os
import random
import re
import time

from src.config import CACHE_DIR, LLM_MODELS, PRICE_PER_MTOK

PROMPT_VERSION = "v1"

SYSTEM = (
    "You extract structured facts from pharmacy benefits-verification phone calls between an "
    "AI patient-care specialist ('assistant') and a human insurance representative ('user'). "
    "Use only what the insurance representative explicitly states. Never guess."
)

INSTRUCTIONS = """Read the call transcript and return ONLY a JSON object with exactly these keys:

{
  "record_found": true | false | null,
  "coverage_status": "ACTIVE" | "INACTIVE" | "NOT_ANSWERED" | null,
  "drug1_formulary": "YES" | "NO" | null,
  "drug1_restrictions": "YES" | "NO" | null,
  "drug2_formulary": "YES" | "NO" | null,
  "drug2_restrictions": "YES" | "NO" | null
}

Rules:
- record_found: did the representative locate the patient's record? null if never discussed.
- coverage_status: only if record_found is true. NOT_ANSWERED if asked but never answered.
- drugN_formulary: is the Nth drug discussed covered / on formulary? Drugs are numbered in the order discussed.
- drugN_restrictions: only if drugN_formulary is "YES". YES if prior authorization, step therapy,
  or quantity limits apply; NO if the representative says there are none.
- Use null for anything that does not apply or is never discussed.

Transcript:
"""


def build_prompt(script: list[dict]) -> str:
    lines = [f"[{i}] {t.get('role','')}: {t.get('content','')}" for i, t in enumerate(script)]
    return INSTRUCTIONS + "\n".join(lines)


def parse_json(text: str) -> dict:
    text = re.sub(r"```(json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


# --------------------------------------------------------------------------- providers
def _call_anthropic(model: str, prompt: str) -> tuple[str, int, int]:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    r = client.messages.create(model=model, max_tokens=4000, system=SYSTEM,
                               messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens


def _call_openai(model: str, prompt: str) -> tuple[str, int, int]:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY
    r = client.chat.completions.create(model=model, temperature=0,
                                       response_format={"type": "json_object"},
                                       messages=[{"role": "system", "content": SYSTEM},
                                                 {"role": "user", "content": prompt}])
    return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens


def _call_mock(gold: dict, seed: str) -> tuple[str, int, int]:
    """Offline stand-in: returns gold labels with ~10% of fields corrupted.
    Exists only to test the pipeline end to end; its scores mean nothing."""
    rng = random.Random(seed)
    out = {}
    for k, v in gold.items():
        if v is not None and rng.random() < 0.10:
            v = {True: False, False: True, "YES": "NO", "NO": "YES",
                 "ACTIVE": "INACTIVE"}.get(v, None)
        out[k] = v
    return json.dumps(out), 0, 0


# --------------------------------------------------------------------------- public API
def extract(call: dict, provider: str, model: str | None = None, gold: dict | None = None) -> dict:
    model = model or LLM_MODELS.get(provider, provider)
    cache = CACHE_DIR / f"{model}_{PROMPT_VERSION}" / f"{call['call_id']}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    prompt = build_prompt(call["script"])
    t0 = time.perf_counter()
    for attempt in range(4):
        try:
            if provider == "anthropic":
                text, tin, tout = _call_anthropic(model, prompt)
            elif provider == "openai":
                text, tin, tout = _call_openai(model, prompt)
            elif provider == "mock":
                text, tin, tout = _call_mock(gold or {}, call["call_id"])
            else:
                raise ValueError(f"unknown provider {provider}")
            break
        except Exception as e:  # retry transient API errors with backoff
            if attempt == 3 or provider == "mock":
                raise
            print(f"  retry {attempt+1} for {call['call_id']}: {e}")
            time.sleep(2 ** attempt)
    latency = time.perf_counter() - t0

    try:
        pred = parse_json(text)
        parse_ok = bool(pred)  # empty or truncated replies count as failures
    except json.JSONDecodeError:
        pred, parse_ok = {}, False

    price = PRICE_PER_MTOK.get(model)
    cost = (tin * price["input"] + tout * price["output"]) / 1e6 if price else None
    rec = {"prediction": pred, "raw": text, "parse_ok": parse_ok,
           "input_tokens": tin, "output_tokens": tout, "latency_s": latency, "cost_usd": cost}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec, indent=1))
    return rec


def provider_available(provider: str) -> bool:
    return provider == "mock" or bool(
        os.getenv({"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[provider]))
