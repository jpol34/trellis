"""Manual smoke test for the live generation path (ticket #5).

Not part of `pytest tests/unit` — this makes real Anthropic API calls and costs money. Run it
yourself once `ANTHROPIC_API_KEY` and `GENERATION_CLAUDE_MODEL` are set, e.g.:

    strongbox get ANTHROPIC_API_KEY > $null  # confirms the secret exists in Strongbox
    $env:ANTHROPIC_API_KEY = (Get-StrongboxSecret ANTHROPIC_API_KEY)
    $env:GENERATION_CLAUDE_MODEL = "claude-sonnet-4-5"
    uv run python scripts/smoke_test_generation.py

Prints each generated persona (to eyeball batch distinctness — no two should read the same) and
one full generated transcript item (to eyeball transcript quality and that injected values read
naturally in context).
"""

from __future__ import annotations

import asyncio
import json

import httpx

from trellis.generation.personas import generate_personas
from trellis.generation.transcript_gen import generate_item
from trellis.schema.loader import CATEGORIES_DIR, load_all_categories
from trellis.settings import settings


async def main() -> None:
    if not settings.anthropic_api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Source it via Strongbox first — see this script's "
            "module docstring."
        )
    if not settings.generation_claude_model:
        raise SystemExit("GENERATION_CLAUDE_MODEL is not set.")

    categories = {spec.category: spec for spec in load_all_categories(CATEGORIES_DIR)}
    prospect = categories["prospect"]

    async with httpx.AsyncClient() as client:
        print("--- Generating a batch of 4 personas (eyeball: no two should read the same) ---")
        personas = await generate_personas(4, client)
        for i, p in enumerate(personas):
            print(f"{i + 1}. {p}")

        print()
        print("--- Generating one transcript item for category=prospect, "
              "call_reason=new_inquiry, tier=natural ---")
        item = await generate_item(prospect, "new_inquiry", "natural", personas[0], client)

    print()
    print("Persona used:", item.persona)
    print()
    print("Transcript:")
    print(item.transcript)
    print()
    print("Ground truth (eyeball: each `value` should appear naturally in the transcript above):")
    print(
        json.dumps(
            [
                {
                    "field": gt.field,
                    "value": gt.value,
                    "candidates": gt.candidates,
                    "correct_index": gt.correct_index,
                    "char_offset": gt.char_offset,
                }
                for gt in item.ground_truth
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
