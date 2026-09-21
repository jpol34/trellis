"""Translation between the internal positional candidate representation and
the dict-keyed `ChoiceQuestion` wire shape shared by laya and Jev."""

from __future__ import annotations

CANDIDATE_KEY_PREFIX = "candidate_"


def candidates_to_criteria(candidates: list[str]) -> dict[str, str]:
    return {f"{CANDIDATE_KEY_PREFIX}{i}": text for i, text in enumerate(candidates)}


def criteria_key_to_index(key: str) -> int:
    if not key.startswith(CANDIDATE_KEY_PREFIX):
        raise ValueError(f"malformed candidate key: {key!r}")

    suffix = key[len(CANDIDATE_KEY_PREFIX) :]
    if not suffix.isdigit():
        raise ValueError(f"malformed candidate key: {key!r}")

    return int(suffix)
