"""Manual, human-run smoke test for the GPT-5.1 reference arm against the real OpenAI API.

Not part of the unit test suite (no OPENAI_API_KEY is available to the agent that wrote this) —
run it yourself after sourcing the key:

    ./scripts/configure-generation-secrets.ps1   # writes OPENAI_API_KEY into .env via Strongbox
    uv run python scripts/smoke_test_gpt51_arm.py

Checks two things the unit tests can only verify against mocked responses: that evidence-first
prompting produces a real quoted excerpt for a field the transcript states, and that a field
genuinely absent from a real transcript makes the arm abstain instead of guessing.
"""

from __future__ import annotations

import asyncio
import sys

from trellis.reference.gpt51_arm import GPT51Arm
from trellis.schema.types import FieldSpec
from trellis.settings import settings

_TRANSCRIPT_WITH_FIELD = """\
Agent: Thanks for calling, can I get a callback number in case we're disconnected?
Caller: Yeah, it's 555-201-4477.
Agent: Got it, and what's the issue you're calling about today?
Caller: My water heater is leaking in the garage.
"""

_TRANSCRIPT_WITHOUT_FIELD = """\
Agent: Thanks for calling, what's going on today?
Caller: My water heater is leaking in the garage.
Agent: Okay, I'll get a technician scheduled. Anything else?
Caller: No, that's it, thanks.
"""

_FIELD = FieldSpec(
    name="callback_number",
    match_type="normalized_phone",
    required=True,
    distractor_strategy="swap_sibling_value",
    value_pool=None,
)


async def main() -> None:
    if not settings.openai_api_key:
        print(
            "OPENAI_API_KEY is not set. Run ./scripts/configure-generation-secrets.ps1 first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    arm = GPT51Arm()

    print(f"Model: {settings.reference_gpt_model}\n")

    print("--- Field present in transcript ---")
    found = await arm.answer(_TRANSCRIPT_WITH_FIELD, _FIELD, None)
    print(found)
    if found["value"] is None:
        print("UNEXPECTED: expected a value, got an abstain.", file=sys.stderr)

    print("\n--- Field absent from transcript ---")
    absent = await arm.answer(_TRANSCRIPT_WITHOUT_FIELD, _FIELD, None)
    print(absent)
    if absent["value"] is not None:
        print("UNEXPECTED: expected an abstain (value=None), got a guessed value.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
