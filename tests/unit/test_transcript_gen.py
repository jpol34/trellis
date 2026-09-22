from __future__ import annotations

import json
import random

import httpx
import pytest
from faker import Faker

from trellis.generation.personas import Persona
from trellis.generation.transcript_gen import (
    FieldGroundTruth,
    generate_item,
    load_variability_tags,
)
from trellis.schema.loader import CATEGORIES_DIR, load_all_categories

CATEGORIES = {spec.category: spec for spec in load_all_categories(CATEGORIES_DIR)}
PROSPECT = CATEGORIES["prospect"]

PERSONA = Persona(
    age_range="30-40",
    tone="friendly",
    verbosity="moderate",
    background="software engineer",
    speech_quirks="none",
)


def _fake_transcript_response(text: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"type": "text", "text": text}]})

    return httpx.MockTransport(handler)


def _seeded_faker(seed: int) -> Faker:
    faker = Faker()
    faker.seed_instance(seed)
    return faker


def test_load_variability_tags_known_tiers():
    for tier in ("clean", "natural", "noisy"):
        tags = load_variability_tags(tier)
        assert set(tags) == {"linguistic", "conversational"}


def test_load_variability_tags_unknown_tier_raises():
    with pytest.raises(ValueError):
        load_variability_tags("nonexistent_tier")


async def test_generate_item_unknown_call_reason_raises():
    async with httpx.AsyncClient(transport=_fake_transcript_response("transcript")) as client:
        with pytest.raises(ValueError):
            await generate_item(PROSPECT, "not_a_real_reason", "clean", PERSONA, client)


async def test_generate_item_produces_ground_truth_for_every_field():
    transport = _fake_transcript_response("Caller: hi. Agent: hi.")
    async with httpx.AsyncClient(transport=transport) as client:
        item = await generate_item(
            PROSPECT,
            "new_inquiry",
            "clean",
            PERSONA,
            client,
            rng=random.Random(0),
            faker=_seeded_faker(0),
        )

    assert item.persona is PERSONA
    assert item.transcript == "Caller: hi. Agent: hi."
    assert len(item.ground_truth) == len(PROSPECT.fields)
    assert {gt.field for gt in item.ground_truth} == {f.name for f in PROSPECT.fields}

    for gt in item.ground_truth:
        assert isinstance(gt, FieldGroundTruth)
        assert gt.value in gt.candidates
        assert gt.candidates[gt.correct_index] == gt.value
        assert len(gt.candidates) in (4, 5)


async def test_generate_item_char_offset_found_when_value_is_in_transcript():
    faker = _seeded_faker(0)
    rng = random.Random(0)
    # Deterministic real value for the name field, embedded verbatim in the fake transcript.
    from trellis.generation.values import generate_value

    name_field = next(f for f in PROSPECT.fields if f.name == "name")
    real_name = generate_value(name_field, _seeded_faker(0), random.Random(0))
    transcript_text = f"Caller: Hi, this is {real_name} calling."

    async with httpx.AsyncClient(transport=_fake_transcript_response(transcript_text)) as client:
        item = await generate_item(
            PROSPECT, "new_inquiry", "clean", PERSONA, client, rng=rng, faker=faker
        )

    name_gt = next(gt for gt in item.ground_truth if gt.field == "name")
    assert name_gt.value == real_name
    assert name_gt.char_offset is not None
    assert transcript_text[name_gt.char_offset : name_gt.char_offset + len(real_name)] == real_name


async def test_generate_item_char_offset_none_when_value_absent_from_transcript():
    async with httpx.AsyncClient(
        transport=_fake_transcript_response("Caller: I'd like to ask about pricing.")
    ) as client:
        item = await generate_item(
            PROSPECT,
            "new_inquiry",
            "clean",
            PERSONA,
            client,
            rng=random.Random(1),
            faker=_seeded_faker(1),
        )

    assert any(gt.char_offset is None for gt in item.ground_truth)


async def test_generate_item_sends_variability_tags_and_values_in_prompt():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["prompt"] = body["messages"][0]["content"]
        return httpx.Response(200, json={"content": [{"type": "text", "text": "transcript"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        item = await generate_item(
            PROSPECT,
            "scheduling_tour",
            "noisy",
            PERSONA,
            client,
            rng=random.Random(2),
            faker=_seeded_faker(2),
        )

    prompt = seen["prompt"]
    assert "scheduling_tour" in prompt
    assert "typos" in prompt  # a noisy-tier linguistic tag
    for gt in item.ground_truth:
        assert gt.value in prompt
