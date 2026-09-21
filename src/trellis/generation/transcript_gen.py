"""One-call transcript generation: given a persona, category, call reason, and variability
tier, asks Claude Sonnet for one realistic call transcript that naturally weaves in the
injected ground-truth values (never the distractors, which the model never sees) in language
matching the persona and the variability tier's linguistic/conversational tags.

`generate_scenario_item` is the top-level, CLI-callable entry point: given `category`,
`call_reason`, and `variability_tier`, it generates a persona and a transcript (both requiring
live Anthropic API calls) and returns one complete generated item.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import httpx
import yaml
from faker import Faker

from trellis.generation.distractors import build_candidates
from trellis.generation.http import post_with_retry
from trellis.generation.personas import Persona, generate_personas
from trellis.generation.values import generate_value
from trellis.schema.loader import REPO_ROOT
from trellis.schema.types import CategorySpec
from trellis.settings import settings

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_VARIABILITY_TIERS_YAML = REPO_ROOT / "configs" / "variability_tiers.yaml"


@dataclass(frozen=True)
class FieldGroundTruth:
    field: str
    value: str
    candidates: list[str]
    correct_index: int
    # Best-effort, diagnostic-only recovery of where `value` appears in the transcript text —
    # not guaranteed accurate; no LLM-based fallback is in scope for this ticket.
    char_offset: int | None


@dataclass(frozen=True)
class GeneratedItem:
    transcript: str
    persona: Persona
    ground_truth: list[FieldGroundTruth]


def load_variability_tags(tier: str) -> dict[str, list[str]]:
    tiers = yaml.safe_load(_VARIABILITY_TIERS_YAML.read_text(encoding="utf-8"))
    if tier not in tiers:
        raise ValueError(f"unknown variability tier {tier!r}; known tiers: {sorted(tiers)}")
    return tiers[tier]


def _transcript_prompt(
    category: CategorySpec,
    call_reason: str,
    persona: Persona,
    tags: dict[str, list[str]],
    values: dict[str, str],
) -> str:
    lines = [
        "Write one realistic phone call transcript between a caller and a property-management "
        f"leasing/resident-services agent. Category: {category.category}. Reason for the "
        f"call: {call_reason}.",
        "",
        "Caller persona:",
        f"  age range: {persona.age_range}",
        f"  tone: {persona.tone}",
        f"  verbosity: {persona.verbosity}",
        f"  background: {persona.background}",
        f"  speech quirks: {persona.speech_quirks}",
        "",
        "The following values must appear naturally in the caller's dialogue somewhere in the "
        "call, exactly as given below (do not alter their formatting):",
    ]
    lines += [f"  - {name}: {value}" for name, value in values.items()]
    lines += [
        "",
        "Linguistic style tags to apply: "
        + (", ".join(tags["linguistic"]) or "none — clean, plain speech") + ".",
        "Conversational style tags to apply: "
        + (", ".join(tags["conversational"]) or "none — smooth, single-disclosure turns") + ".",
        "",
        "Write only the transcript itself, formatted as alternating 'Caller:' / 'Agent:' turns. "
        "No preamble, no commentary, no markdown formatting.",
    ]
    return "\n".join(lines)


def _find_char_offset(transcript: str, value: str) -> int | None:
    """Best-effort, diagnostic-only recovery of where `value` appears in `transcript`: exact
    match first, then a case-insensitive fallback."""
    idx = transcript.find(value)
    if idx != -1:
        return idx
    idx = transcript.lower().find(value.lower())
    return idx if idx != -1 else None


async def _call_transcript_model(client: httpx.AsyncClient, prompt: str) -> str:
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
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120.0,
    )
    return resp.json()["content"][0]["text"]


async def generate_item(
    category: CategorySpec,
    call_reason: str,
    variability_tier: str,
    persona: Persona,
    client: httpx.AsyncClient,
    *,
    rng: random.Random | None = None,
    faker: Faker | None = None,
) -> GeneratedItem:
    """Produces one complete generated item for a single scenario cell: injected ground-truth
    values for every field in `category`, a transcript weaving them in naturally, and a
    shuffled candidate list + correct index per field, composable with #4's wire adapter
    (`trellis.reference.wire.candidates_to_criteria`)."""
    if call_reason not in category.scenarios.call_reasons:
        raise ValueError(
            f"{call_reason!r} is not a valid call reason for category {category.category!r}; "
            f"known reasons: {category.scenarios.call_reasons}"
        )
    rng = rng if rng is not None else random.Random()
    faker = faker if faker is not None else Faker()
    tags = load_variability_tags(variability_tier)

    values = {f.name: generate_value(f, faker, rng) for f in category.fields}
    prompt = _transcript_prompt(category, call_reason, persona, tags, values)
    transcript = await _call_transcript_model(client, prompt)

    ground_truth = []
    for f in category.fields:
        value = values[f.name]
        candidates, correct_index = build_candidates(value, f, faker, rng)
        ground_truth.append(
            FieldGroundTruth(
                field=f.name,
                value=value,
                candidates=candidates,
                correct_index=correct_index,
                char_offset=_find_char_offset(transcript, value),
            )
        )

    return GeneratedItem(transcript=transcript, persona=persona, ground_truth=ground_truth)


async def generate_scenario_item(
    category: CategorySpec, call_reason: str, variability_tier: str
) -> dict:
    """CLI-callable entry point: given a category, call reason, and variability tier, generates
    a persona and one complete transcript item end to end, making the necessary live Anthropic
    API calls itself. Requires `settings.anthropic_api_key` (sourced via Strongbox) to be set."""
    async with httpx.AsyncClient() as client:
        persona = (await generate_personas(1, client))[0]
        item = await generate_item(category, call_reason, variability_tier, persona, client)

    return {
        "transcript": item.transcript,
        "persona": item.persona,
        "ground_truth": [
            {
                "field": gt.field,
                "value": gt.value,
                "candidates": gt.candidates,
                "correct_index": gt.correct_index,
                "char_offset": gt.char_offset,
            }
            for gt in item.ground_truth
        ],
    }
