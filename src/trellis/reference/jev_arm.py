"""Reference arm backed by the real typesafe.ai Jev API.

Poses a `choice` System One question over the candidate set it is given, translating between
trellis's positional `candidates` and Jev's dict-keyed criteria via `reference.wire`.
"""

from __future__ import annotations

from typesafe_sdk import AsyncTypeSafeClient

from trellis.reference.base import ArmAnswer, register_arm
from trellis.reference.wire import candidates_to_criteria, criteria_key_to_index
from trellis.schema.types import FieldSpec
from trellis.settings import settings

QUESTION_NAME = "field"


class JevArm:
    name = "jev"
    mode = "closed_set"

    async def answer(
        self, transcript: str, field: FieldSpec, candidates: list[str] | None
    ) -> ArmAnswer:
        if candidates is None:
            raise ValueError("jev arm is closed_set-only and requires candidates")

        criteria = candidates_to_criteria(candidates)
        async with AsyncTypeSafeClient(
            api_key=settings.typesafe_api_key, base_url=settings.jev_base_url or None
        ) as client:
            response = await client.system_one(
                state=transcript,
                questions={
                    QUESTION_NAME: {
                        "type": "choice",
                        "instructions": f"Which value applies for the {field.name!r} field?",
                        "criteria": criteria,
                    }
                },
            )

        answer = response.choices[QUESTION_NAME]
        chosen_index = criteria_key_to_index(answer.choice)
        return ArmAnswer(
            value=candidates[chosen_index],
            chosen_index=chosen_index,
            confidence=answer.confidence,
        )


register_arm(JevArm())
