"""The open-extraction reference arm: given `candidates=None`, extracts a field's value
directly from transcript text by prompting an OpenAI model, rather than choosing among a
pre-built candidate list (that's what the other, closed-set arms do).

The prompt is evidence-first: the model must quote the specific transcript text supporting a
value before stating it, and must use a distinct, explicit abstention output when the field
genuinely isn't mentioned. Without that, an LLM asked to judge absence from a transcript tends
toward over-closure — treating sparse or missing evidence as a confirmed answer instead of
admitting it found nothing (the same failure mode laya-bench's dual-LLM labeling prompt design
guards against; see `laya-bench/src/laya_bench/eval/labeling.py::_build_prompt`).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from trellis.reference.base import ArmAnswer, register_arm
from trellis.reference.http import post_with_retry
from trellis.schema.types import FieldSpec
from trellis.settings import settings

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_INSUFFICIENT_EVIDENCE = "insufficient"


def _build_prompt(transcript: str, field: FieldSpec) -> str:
    return (
        "You are extracting a single field's value from a call transcript for a benchmark "
        "evaluation. Read the transcript below, then answer strictly as JSON in the format "
        "specified — no other text.\n\n"
        f"Field to extract: {field.name}\n\n"
        'First quote the specific passage of the transcript that actually supports the '
        'value, in an "evidence" field — do not paraphrase, and do not cite unrelated text '
        "(e.g. the field's name appearing in an unrelated context doesn't count as evidence). "
        "Then respond with one of:\n"
        '  - {"evidence": "<quoted text>", "status": "found", "value": "<extracted value>", '
        '"confidence": <float 0-1>} if the transcript states the value.\n'
        '  - {"evidence": "", "status": "insufficient"} if the transcript does not actually '
        "state this field's value. Reserve this for genuine absence — not merely because the "
        "field isn't stated in the exact words you expected.\n\n"
        "Transcript:\n"
        f"{transcript}"
    )


def _parse_response(text: str) -> ArmAnswer:
    """Extracts the evidence-first JSON answer from a model's response, tolerating surrounding
    prose/markdown fences — models don't reliably return bare JSON despite instructions to."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in response: {text!r}")
    parsed: dict[str, Any] = json.loads(text[start : end + 1])

    status = parsed.get("status")
    if status == _INSUFFICIENT_EVIDENCE:
        return {"value": None, "chosen_index": None, "confidence": 0.0}
    if status != "found":
        raise ValueError(f"status {status!r} is not 'found' or {_INSUFFICIENT_EVIDENCE!r}")

    value = parsed.get("value")
    if not isinstance(value, str) or not value:
        raise ValueError(f"value {value!r} is not a non-empty string")

    evidence = parsed.get("evidence")
    if not isinstance(evidence, str) or not evidence:
        raise ValueError(f"evidence {evidence!r} is not a non-empty string for a found value")

    confidence = parsed.get("confidence")
    if not isinstance(confidence, int | float) or isinstance(confidence, bool) or not (
        0.0 <= confidence <= 1.0
    ):
        raise ValueError(f"confidence {confidence!r} is not a float in [0, 1]")

    return {"value": value, "chosen_index": None, "confidence": float(confidence)}


class GPT51Arm:
    name = "gpt-5.1"

    async def answer(
        self, transcript: str, field: FieldSpec, candidates: list[str] | None
    ) -> ArmAnswer:
        async with httpx.AsyncClient() as client:
            resp = await post_with_retry(
                client,
                _OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.reference_gpt_model,
                    "messages": [{"role": "user", "content": _build_prompt(transcript, field)}],
                    "response_format": {"type": "json_object"},
                },
                timeout=60.0,
            )
        text = resp.json()["choices"][0]["message"]["content"]
        return _parse_response(text)


register_arm(GPT51Arm())
