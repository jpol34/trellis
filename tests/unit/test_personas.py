from __future__ import annotations

import json

import httpx

from trellis.generation.personas import Persona, _parse_persona, generate_personas


def _anthropic_response(persona: dict) -> httpx.Response:
    return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(persona)}]})


async def test_generate_personas_returns_requested_count():
    personas = [
        {
            "age_range": f"{20 + i}-{30 + i}",
            "tone": "tone",
            "verbosity": "verbosity",
            "background": "background",
            "speech_quirks": "quirks",
        }
        for i in range(3)
    ]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        resp = _anthropic_response(personas[calls["n"]])
        calls["n"] += 1
        return resp

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await generate_personas(3, client)

    assert len(result) == 3
    assert all(isinstance(p, Persona) for p in result)
    assert calls["n"] == 3


def test_parse_persona_tolerates_brace_in_field_value_and_trailing_prose():
    persona = {
        "age_range": "30-40",
        "tone": "friendly",
        "verbosity": "concise",
        "background": "works as a {full-stack} engineer",
        "speech_quirks": "none",
    }
    text = f"Here you go:\n{json.dumps(persona)}\nLet me know if you need {{more}} info."

    result = _parse_persona(text)

    assert result.background == "works as a {full-stack} engineer"


async def test_generate_personas_is_sequential_and_each_prompt_lists_prior_personas():
    personas = [
        {
            "age_range": "20-30",
            "tone": "chipper",
            "verbosity": "terse",
            "background": "grad student",
            "speech_quirks": "says 'like' often",
        },
        {
            "age_range": "60-70",
            "tone": "gruff",
            "verbosity": "rambling",
            "background": "retired contractor",
            "speech_quirks": "long pauses",
        },
    ]
    seen_prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        seen_prompts.append(prompt)
        return _anthropic_response(personas[len(seen_prompts) - 1])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await generate_personas(2, client)

    assert len(seen_prompts) == 2
    # First prompt has no prior personas to list; second must reference the first persona's
    # attributes so the model is steered away from repeating it.
    assert "chipper" not in seen_prompts[0]
    assert "chipper" in seen_prompts[1]
    assert "distinct" in seen_prompts[1].lower()


async def test_generate_personas_zero_returns_empty_list_and_makes_no_calls():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise AssertionError("should not be called")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await generate_personas(0, client)

    assert result == []
    assert calls["n"] == 0
