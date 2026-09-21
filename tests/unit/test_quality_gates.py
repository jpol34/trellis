from __future__ import annotations

import json

from trellis.generation.quality_gates import (
    QualityGateField,
    QualityGateItem,
    run_quality_gates,
)

CELL = dict(category="prospect", call_reason="new_inquiry")

_VARIED_FILLER = [
    "Yeah so, sorry, one sec, let me find that for you.",
    "Hang on, I've got it written down here somewhere.",
    "Sure thing, give me a moment to grab that.",
    "Oh, right, let me think — okay, here it is.",
    "No problem, just a second.",
    "Alright, bear with me a sec.",
    "Sure, hold on, let me check.",
    "One moment, pulling that up now.",
    "Okay yeah, I've got that right here.",
    "Sure, let me just double check that first.",
    "Give me one second here.",
    "Ah okay, here we go.",
    "Yep, one sec, almost there.",
    "Alright, let me grab that real quick.",
    "Sure, hang on just a moment.",
    "Okay, bear with me.",
    "One sec, let me pull it up.",
    "Alright, just a moment please.",
]


def _budget_item(idx: int, tier: str, filler: str, *, templated: bool = False) -> QualityGateItem:
    value = str(1500 + idx * 25)
    if templated:
        sentence = f"My budget is {value}."
    else:
        sentence = f"{filler} My budget's around {value} a month, give or take."
    transcript = f"Caller: {sentence} Agent: Got it, thanks for that."
    offset = transcript.index(value)
    candidates = [value, str(1500 + idx * 25 + 300), str(1500 + idx * 25 - 300), "1200"]
    return QualityGateItem(
        item_id=f"item-{idx}",
        variability_tier=tier,
        transcript=transcript,
        fields=[
            QualityGateField(field="budget", value=value, candidates=candidates, char_offset=offset)
        ],
        **CELL,
    )


def _make_batch() -> list[QualityGateItem]:
    items = []
    # Deliberate near-duplicate pair, same scenario cell, near-identical transcript.
    dup_a_transcript = (
        "Caller: Hi, I'm calling about the two bedroom unit, my budget is 1800 a month. "
        "Agent: Great, thanks."
    )
    dup_b_transcript = (
        "Caller: Hi, I'm calling about the two bedroom unit, my budget is 1800 a month, "
        "give or take. Agent: Great, thanks."
    )
    dup_a = QualityGateItem(
        item_id="dup-a",
        variability_tier="clean",
        transcript=dup_a_transcript,
        fields=[
            QualityGateField(
                field="budget",
                value="1800",
                candidates=["1800", "2100", "1500", "1200"],
                char_offset=dup_a_transcript.index("1800"),
            )
        ],
        **CELL,
    )
    dup_b = QualityGateItem(
        item_id="dup-b",
        variability_tier="clean",
        transcript=dup_b_transcript,
        fields=[
            QualityGateField(
                field="budget",
                value="1800",
                candidates=["1800", "2100", "1500", "1200"],
                char_offset=dup_b_transcript.index("1800"),
            )
        ],
        **CELL,
    )
    items += [dup_a, dup_b]

    # A deliberately templated injection sentence — should be flagged by the leakage check.
    items.append(_budget_item(0, "natural", _VARIED_FILLER[0], templated=True))

    # A deliberate distractor-tell: real value uniquely lacks the hedge markers its distractors
    # all carry.
    hedged_transcript = "Caller: My budget's flexible, but let's say 1950 a month. Agent: Noted."
    hedged_item = QualityGateItem(
        item_id="item-hedged",
        variability_tier="natural",
        transcript=hedged_transcript,
        fields=[
            QualityGateField(
                field="budget",
                value="1950",
                candidates=["around 1800", "about 1900", "maybe 2000", "1950"],
                char_offset=hedged_transcript.index("1950"),
            )
        ],
        **CELL,
    )
    items.append(hedged_item)

    tiers = ["clean", "natural", "noisy"]
    for i in range(1, 17):
        items.append(_budget_item(i, tiers[i % 3], _VARIED_FILLER[i % len(_VARIED_FILLER)]))

    return items


def test_quality_gates_over_batch_of_twenty_produces_all_reports(tmp_path):
    items = _make_batch()
    assert len(items) >= 20

    diversity_path = tmp_path / "diversity_review.json"
    injection_path = tmp_path / "injection_review.json"
    distractor_path = tmp_path / "distractor_review.json"

    report = run_quality_gates(
        items,
        diversity_review_path=diversity_path,
        injection_review_path=injection_path,
        distractor_review_path=distractor_path,
    )

    assert any({p.item_a, p.item_b} == {"dup-a", "dup-b"} for p in report.near_duplicates)
    assert any(f.reason == "templated_pattern" for f in report.injection_flags)
    assert any(f.reason == "real_value_missing_expected_marker" for f in report.distractor_flags)

    for path in (diversity_path, injection_path, distractor_path):
        assert path.exists()
        entries = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        assert len(entries) > 0
        assert all(e["reviewed"] is None for e in entries)


def test_rerun_preserves_already_reviewed_entries_end_to_end(tmp_path):
    items = _make_batch()
    diversity_path = tmp_path / "diversity_review.json"
    injection_path = tmp_path / "injection_review.json"
    distractor_path = tmp_path / "distractor_review.json"

    run_quality_gates(
        items,
        diversity_review_path=diversity_path,
        injection_review_path=injection_path,
        distractor_review_path=distractor_path,
    )

    diversity_entries = json.loads(diversity_path.read_text(encoding="utf-8"))
    for e in diversity_entries:
        e["reviewed"] = True
    diversity_path.write_text(json.dumps(diversity_entries, indent=2), encoding="utf-8")

    # Add one more item to the batch (simulating a growing corpus), in a scenario cell of its
    # own so it can't introduce a new near-duplicate pair, and re-run.
    new_transcript = (
        "Caller: Hey, quick question about the application process, my budget's 3975 a month. "
        "Agent: Sure, happy to walk you through that."
    )
    items.append(
        QualityGateItem(
            item_id="item-99",
            category="prospect",
            call_reason="application_followup",
            variability_tier="clean",
            transcript=new_transcript,
            fields=[
                QualityGateField(
                    field="budget",
                    value="3975",
                    candidates=["3975", "4275", "3675", "3200"],
                    char_offset=new_transcript.index("3975"),
                )
            ],
        )
    )
    run_quality_gates(
        items,
        diversity_review_path=diversity_path,
        injection_review_path=injection_path,
        distractor_review_path=distractor_path,
    )

    diversity_entries_after = json.loads(diversity_path.read_text(encoding="utf-8"))
    assert len(diversity_entries_after) == len(diversity_entries)
    assert all(e["reviewed"] is True for e in diversity_entries_after)
