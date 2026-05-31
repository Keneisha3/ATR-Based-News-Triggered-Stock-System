"""Optional LLM headline classifier (sentiment + category + confidence).

Provider is auto-detected from environment variables:
  * ANTHROPIC_API_KEY -> Claude
  * OPENAI_API_KEY    -> OpenAI

If neither key is present (or USE_LLM is False), `available()` returns False and
callers fall back to the offline lexicon in news.py. Uses plain `requests`, so
no extra SDK dependency.
"""

from __future__ import annotations

import json
import os
import re

import requests

from . import config

_VALID_SENTIMENT = {"bullish", "bearish", "neutral"}


def provider() -> str | None:
    if not config.USE_LLM:
        return None
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


def available() -> bool:
    return provider() is not None


def _build_prompt(headlines: list[str]) -> str:
    cats = ", ".join(config.NEWS_CATEGORIES)
    numbered = "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))
    return (
        "You are a financial news classifier. For each numbered stock headline, "
        "return its market sentiment for the mentioned company, a category, and a "
        "0-100 confidence.\n"
        f"sentiment must be one of: bullish, bearish, neutral.\n"
        f"category must be one of: {cats}.\n"
        "Respond with ONLY a JSON array, one object per headline, in order, like:\n"
        '[{"i":0,"sentiment":"bullish","category":"Earnings","confidence":85}]\n\n'
        f"Headlines:\n{numbered}"
    )


def _parse(content: str, n: int) -> list[dict]:
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        raise ValueError("no JSON array in LLM response")
    data = json.loads(match.group(0))
    by_index = {int(o.get("i", k)): o for k, o in enumerate(data)}
    out = []
    for k in range(n):
        o = by_index.get(k, {})
        senti = str(o.get("sentiment", "neutral")).lower()
        if senti not in _VALID_SENTIMENT:
            senti = "neutral"
        cat = o.get("category", "Other")
        if cat not in config.NEWS_CATEGORIES:
            cat = "Other"
        try:
            conf = max(0, min(100, int(o.get("confidence", 50))))
        except (TypeError, ValueError):
            conf = 50
        out.append({"sentiment": senti, "category": cat, "confidence": conf})
    return out


def _call_anthropic(prompt: str) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": config.LLM_ANTHROPIC_MODEL,
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _call_openai(prompt: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.LLM_OPENAI_MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def classify(headlines: list[str]) -> list[dict] | None:
    """Classify headlines in batches. Returns a list aligned to `headlines`,
    or None if the LLM is unavailable or every batch failed."""
    prov = provider()
    if prov is None or not headlines:
        return None

    caller = _call_anthropic if prov == "anthropic" else _call_openai
    results: list[dict] = []
    any_ok = False
    for start in range(0, len(headlines), config.LLM_BATCH_SIZE):
        batch = headlines[start:start + config.LLM_BATCH_SIZE]
        try:
            content = caller(_build_prompt(batch))
            results.extend(_parse(content, len(batch)))
            any_ok = True
        except Exception as exc:
            print(f"  ! LLM batch failed ({prov}): {exc}; using lexicon for these")
            results.extend([{"sentiment": "neutral", "category": "Other",
                             "confidence": 0}] * len(batch))
    return results if any_ok else None
