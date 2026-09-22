"""Debug tool: capture the real wire-level request/response for each of the three arms
(trellis, jev, gpt-5.1) against one real eval-set record, and save it to a JSON file for
inspection.

Each arm's actual traffic looks different, so this doesn't call the arms through the eval
runner (`validate/run_eval.py`) — it drives each arm's real code path directly and taps the
transport layer to capture what's genuinely sent/received, rather than reconstructing it from
each arm's own pre-request data:

  - gpt-5.1: an `httpx.AsyncClient` with `event_hooks` captures the real HTTP request/response
    around `GPT51Arm`'s call to the OpenAI chat-completions endpoint.
  - jev: `typesafe_sdk.AsyncTypeSafeClient` sends over a custom `httpx2` transport, which
    supports the same `event_hooks` mechanism as `httpx`. `jev_arm.AsyncTypeSafeClient` is
    monkeypatched so `JevArm.answer()`'s own client construction picks up a capturing
    `httpx2.AsyncClient`, without touching `jev_arm.py`.
  - trellis: no network hop. `TrellisModel._agent.system_one` (the `laya.Agent` bound method) is
    wrapped to record the exact `questions` dict passed in and the exact raw dict returned.

Any `Authorization` header captured is redacted before printing/saving. Run it yourself — it
makes one real, billed call each to OpenAI and typesafe.ai:

    uv run python scripts/debug_wire.py

Output: data/debug_wire_capture.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import httpx2

from trellis.generation.corpus_common import read_jsonl
from trellis.reference import jev_arm as jev_arm_module
from trellis.reference.gpt51_arm import GPT51Arm
from trellis.reference.jev_arm import JevArm
from trellis.schema.loader import load_all_categories
from trellis.schema.types import FieldSpec
from trellis.settings import settings

EVAL_SET_PATH = Path("data") / "eval_set" / "eval.jsonl"
OUTPUT_PATH = Path("data") / "debug_wire_capture.json"

# Small, cheap subset of one record's fields — enough to see a realistic multi-question call
# shape per arm without paying for/waiting on all 9 fields a real eval-set record carries.
FIELDS_TO_USE = ["name", "budget", "pet_info"]

_REDACTED = "***REDACTED***"
# Bot-mitigation/session cookies (e.g. Cloudflare's __cf_bm on OpenAI's endpoint) aren't API
# credentials, but they're still live session-identifying values with no reason to persist to
# disk in a debug artifact — redacted alongside the actual auth header.
_SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie"}


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: (_REDACTED if k.lower() in _SENSITIVE_HEADERS else v) for k, v in headers.items()
    }


def _try_json(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace")


def _load_sample_record() -> dict:
    records = read_jsonl(EVAL_SET_PATH)
    for record in records:
        if record.get("category") == "prospect":
            return record
    raise SystemExit(f"no 'prospect' category record found in {EVAL_SET_PATH}")


def _field_specs_for_category(category: str) -> dict[str, FieldSpec]:
    for spec in load_all_categories():
        if spec.category == category:
            return {f.name: f for f in spec.fields}
    raise SystemExit(f"no category config found for {category!r}")


# --------------------------------------------------------------------------------------------
# gpt-5.1: real httpx.AsyncClient with event_hooks tapping the actual request/response.
# --------------------------------------------------------------------------------------------


def _make_capturing_httpx_client(sink: dict[str, list[dict]]) -> httpx.AsyncClient:
    async def on_request(request: httpx.Request) -> None:
        request.read()
        sink["requests"].append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": _redact_headers(dict(request.headers)),
                "body": _try_json(request.content),
            }
        )

    async def on_response(response: httpx.Response) -> None:
        await response.aread()
        sink["responses"].append(
            {
                "status_code": response.status_code,
                "headers": _redact_headers(dict(response.headers)),
                "body": _try_json(response.content),
            }
        )

    return httpx.AsyncClient(event_hooks={"request": [on_request], "response": [on_response]})


async def _capture_gpt51(
    transcript: str, fields: dict[str, FieldSpec]
) -> dict[str, dict]:
    sink: dict[str, list[dict]] = {"requests": [], "responses": []}
    client = _make_capturing_httpx_client(sink)
    arm = GPT51Arm(client=client)

    per_field: dict[str, dict] = {}
    try:
        for name, field in fields.items():
            before = len(sink["requests"])
            answer = await arm.answer(transcript, field, None)
            per_field[name] = {
                "request": sink["requests"][before],
                "response": sink["responses"][before],
                "parsed_answer": answer,
            }
    finally:
        await client.aclose()
    return per_field


# --------------------------------------------------------------------------------------------
# jev: httpx2.AsyncClient (typesafe_sdk's transport) with the same event_hooks mechanism.
# --------------------------------------------------------------------------------------------


def _make_capturing_httpx2_client(sink: dict[str, list[dict]]) -> httpx2.AsyncClient:
    async def on_request(request: httpx2.Request) -> None:
        request.read()
        sink["requests"].append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": _redact_headers(dict(request.headers)),
                "body": _try_json(request.content),
            }
        )

    async def on_response(response: httpx2.Response) -> None:
        await response.aread()
        sink["responses"].append(
            {
                "status_code": response.status_code,
                "headers": _redact_headers(dict(response.headers)),
                "body": _try_json(response.content),
            }
        )

    return httpx2.AsyncClient(event_hooks={"request": [on_request], "response": [on_response]})


async def _capture_jev(
    transcript: str, fields: dict[str, FieldSpec], candidates_by_field: dict[str, list[str]]
) -> dict[str, dict]:
    sink: dict[str, list[dict]] = {"requests": [], "responses": []}
    real_client_cls = jev_arm_module.AsyncTypeSafeClient

    class _CapturingAsyncTypeSafeClient(real_client_cls):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("http_client", _make_capturing_httpx2_client(sink))
            super().__init__(*args, **kwargs)

    jev_arm_module.AsyncTypeSafeClient = _CapturingAsyncTypeSafeClient
    try:
        arm = JevArm()
        per_field: dict[str, dict] = {}
        for name, field in fields.items():
            before = len(sink["requests"])
            answer = await arm.answer(transcript, field, candidates_by_field[name])
            per_field[name] = {
                "request": sink["requests"][before],
                "response": sink["responses"][before],
                "parsed_answer": answer,
            }
        return per_field
    finally:
        jev_arm_module.AsyncTypeSafeClient = real_client_cls


# --------------------------------------------------------------------------------------------
# trellis: in-process, no network — wrap the laya.Agent.system_one call site instead.
# --------------------------------------------------------------------------------------------


def _capture_trellis(
    transcript: str, fields: dict[str, FieldSpec], candidates_by_field: dict[str, list[str]]
) -> dict[str, Any]:
    from trellis.model.trellis_model import TrellisModel

    model = TrellisModel(settings.trellis_checkpoint_path, device=settings.trellis_device)

    captured: dict[str, Any] = {}
    real_system_one = model._agent.system_one

    def capturing_system_one(state: Any, questions: dict) -> dict:
        result = real_system_one(state, questions)
        captured["request"] = {"state": state, "questions": questions}
        captured["response"] = result
        return result

    model._agent.system_one = capturing_system_one  # type: ignore[assignment]

    items = [(fields[name], candidates_by_field[name]) for name in fields]
    results = model.discriminate_batch(transcript, items)
    captured["parsed_answers"] = [
        (str(r) if isinstance(r, Exception) else r.__dict__) for r in results
    ]
    return captured


# --------------------------------------------------------------------------------------------


async def main() -> None:
    missing = [
        name
        for name, val in [
            ("OPENAI_API_KEY", settings.openai_api_key),
            ("TYPESAFE_API_KEY", settings.typesafe_api_key),
            ("TRELLIS_CHECKPOINT_PATH", settings.trellis_checkpoint_path),
        ]
        if not val
    ]
    if missing:
        print(f"Missing required settings: {', '.join(missing)}. See .env.", file=sys.stderr)
        raise SystemExit(1)

    record = _load_sample_record()
    transcript = record["transcript"]
    category = record["category"]
    field_specs = _field_specs_for_category(category)

    record_fields = {f["field"]: f for f in record["fields"]}
    field_names = [name for name in FIELDS_TO_USE if name in record_fields]
    if len(field_names) < 2:
        raise SystemExit(
            f"expected at least 2 of {FIELDS_TO_USE} on record {record['item_id']!r}, "
            f"found {field_names}"
        )

    fields = {name: field_specs[name] for name in field_names}
    candidates_by_field = {name: record_fields[name]["candidates"] for name in field_names}

    print(f"Sample record: {record['item_id']} (category={category})")
    print(f"Fields used: {field_names}\n")

    print("--- Capturing trellis (in-process) ---")
    trellis_capture = _capture_trellis(transcript, fields, candidates_by_field)
    print("done.\n")

    print("--- Capturing jev (real typesafe.ai call) ---")
    jev_capture = await _capture_jev(transcript, fields, candidates_by_field)
    print("done.\n")

    print("--- Capturing gpt-5.1 (real OpenAI call) ---")
    try:
        gpt51_capture: dict[str, Any] = await _capture_gpt51(transcript, fields)
        print("done.\n")
    except Exception as e:  # noqa: BLE001 - a gpt-5.1 failure must not discard trellis/jev data
        print(f"FAILED: {e}\n", file=sys.stderr)
        gpt51_capture = {"error": str(e)}

    output = {
        "sample": {
            "item_id": record["item_id"],
            "category": category,
            "fields_used": field_names,
            "transcript": transcript,
        },
        "trellis": trellis_capture,
        "jev": jev_capture,
        "gpt-5.1": gpt51_capture,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"Wrote capture to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
