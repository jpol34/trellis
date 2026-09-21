"""Sequential caller-persona generation via Claude Sonnet.

Personas are generated one at a time, not fanned out concurrently: each new prompt lists every
persona already produced in this batch and is explicitly instructed to differ from all of them.
A persona describes the caller's identity and speaking style only (demographics, tone,
verbosity, background, speech quirks) — never the injected ground-truth values (name, phone,
etc.), which come from `trellis.generation.values` instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from trellis.generation.http import extract_anthropic_text, post_with_retry
from trellis.settings import settings

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


@dataclass(frozen=True)
class Persona:
    age_range: str
    tone: str
    verbosity: str
    background: str
    speech_quirks: str


def _persona_prompt(prior: list[Persona]) -> str:
    lines = [
        "Generate one caller persona for a synthetic property-management phone call "
        "transcript. The persona describes the caller's identity and speaking style only — "
        "never a name, phone number, email, or any other ground-truth value, which is injected "
        "separately.",
        "",
        "Respond strictly as JSON with keys: age_range, tone, verbosity, background, "
        "speech_quirks — no other text. Keep each value to a short phrase or one sentence at "
        "most, not a paragraph.",
    ]
    if prior:
        lines += [
            "",
            "This persona MUST be clearly distinct from every persona already generated in this "
            "batch, listed below. Vary age range, tone, verbosity, background, and speech quirks "
            "so no two personas in the batch read the same.",
        ]
        lines += [
            f"  {i + 1}. age_range={p.age_range!r} tone={p.tone!r} verbosity={p.verbosity!r} "
            f"background={p.background!r} speech_quirks={p.speech_quirks!r}"
            for i, p in enumerate(prior)
        ]
    return "\n".join(lines)


def _parse_persona(text: str) -> Persona:
    """Extracts the persona JSON object from a model response, tolerating surrounding
    prose/markdown fences — models don't reliably return bare JSON despite instructions to.
    Uses raw_decode from the first '{' rather than find('{')/rfind('}'), since a brace inside a
    free-text field value (or trailing prose after the object) would make rfind('}') pick the
    wrong end index."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in persona response: {text!r}")
    try:
        data = json.JSONDecoder().raw_decode(text, start)[0]
    except json.JSONDecodeError as e:
        raise ValueError(f"no valid JSON object found in persona response: {text!r}") from e
    return Persona(
        age_range=data["age_range"],
        tone=data["tone"],
        verbosity=data["verbosity"],
        background=data["background"],
        speech_quirks=data["speech_quirks"],
    )


async def _generate_one(client: httpx.AsyncClient, prior: list[Persona]) -> Persona:
    resp = await post_with_retry(
        client,
        _ANTHROPIC_URL,
        headers={
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": settings.generation_claude_model,
            # Generous relative to the actual JSON payload: some models prepend a `thinking`
            # block before the visible text (see extract_anthropic_text), and a persona whose
            # free-text fields ran long enough to hit a small budget would be silently truncated
            # into invalid JSON rather than erroring clearly.
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": _persona_prompt(prior)}],
        },
        timeout=60.0,
    )
    text = extract_anthropic_text(resp.json())
    return _parse_persona(text)


async def generate_personas(n: int, client: httpx.AsyncClient) -> list[Persona]:
    """Generates `n` personas sequentially. Each call sees every persona generated earlier in
    this batch (see `_persona_prompt`) so the model is explicitly steered away from repeating
    itself — a concurrent fan-out has no such visibility between calls and mitigates mode
    collapse far less effectively."""
    personas: list[Persona] = []
    for _ in range(n):
        personas.append(await _generate_one(client, personas))
    return personas
