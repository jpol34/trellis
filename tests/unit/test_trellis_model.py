"""Unit-level tests for `TrellisModel.discriminate_batch`'s chunking/demux logic, via a
monkeypatched fake `laya` agent — no real checkpoint needed (see `tests/integration/
test_trellis_model.py` for the real-checkpoint, single-item regression contract)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import trellis.model.trellis_model as trellis_model_module
from trellis.model.trellis_model import DiscriminationResult, TrellisModel
from trellis.schema.types import FieldSpec

FIELD_A = FieldSpec(
    name="issue_type",
    match_type="canonical_list",
    required=True,
    distractor_strategy="swap_sibling_value",
    value_pool="issue_types",
)
FIELD_B = FieldSpec(
    name="urgency",
    match_type="canonical_list",
    required=True,
    distractor_strategy="swap_sibling_value",
    value_pool="urgency_levels",
)

CANDIDATES_A = ["leak", "noise complaint", "not mentioned"]
CANDIDATES_B = ["high", "low", "not mentioned"]


class _FakeAgent:
    def __init__(self) -> None:
        self.device = SimpleNamespace(type="cpu")
        self.system_one_calls: list[tuple[str, dict]] = []
        # key -> either a normal answer dict, or a callable(key) -> dict, or "malformed" to omit
        self.answer_overrides: dict[str, object] = {}
        # 1-based call index (across this agent's lifetime) that should raise instead of
        # answering, to simulate a later chunk in one `discriminate_batch` call failing outright.
        self.raise_on_call_number: int | None = None

    def system_one(self, transcript: str, questions: dict) -> dict:
        self.system_one_calls.append((transcript, questions))
        if self.raise_on_call_number == len(self.system_one_calls):
            raise RuntimeError("system_one failed")
        answers = {}
        for key, question in questions.items():
            override = self.answer_overrides.get(key)
            if override == "malformed":
                continue  # omit this key entirely, forcing a KeyError downstream
            if override is not None:
                answers[key] = override
                continue
            criteria = question["criteria"]
            first_key = next(iter(criteria))  # candidate_0 - always a valid choice
            answers[key] = {"choice": first_key, "confidence": 0.9}
        return {"answers": answers}


@pytest.fixture
def fake_agent(monkeypatch) -> _FakeAgent:
    agent = _FakeAgent()
    monkeypatch.setattr(trellis_model_module.laya, "load", lambda *a, **k: agent)
    return agent


def _model(fake_agent: _FakeAgent) -> TrellisModel:
    return TrellisModel("unused-checkpoint-dir", device="cpu")


def test_batch_results_returned_in_order(fake_agent):
    model = _model(fake_agent)

    results = model.discriminate_batch(
        "transcript", [(FIELD_A, CANDIDATES_A), (FIELD_B, CANDIDATES_B)]
    )

    assert len(fake_agent.system_one_calls) == 1
    assert all(isinstance(r, DiscriminationResult) for r in results)
    assert results[0].chosen_value == "leak"
    assert results[1].chosen_value == "high"


def test_invalid_item_gets_error_at_its_position_without_a_model_call(fake_agent):
    model = _model(fake_agent)

    results = model.discriminate_batch(
        "transcript", [(FIELD_A, CANDIDATES_A), (FIELD_B, [])]
    )

    assert isinstance(results[0], DiscriminationResult)
    assert isinstance(results[1], ValueError)
    # only the valid item's question was ever sent to the model
    _transcript, questions = fake_agent.system_one_calls[0]
    assert len(questions) == 1


def test_malformed_response_isolated_to_its_own_position(fake_agent):
    key_a = "field_0"
    fake_agent.answer_overrides[key_a] = "malformed"
    model = _model(fake_agent)

    results = model.discriminate_batch(
        "transcript", [(FIELD_A, CANDIDATES_A), (FIELD_B, CANDIDATES_B)]
    )

    assert isinstance(results[0], ValueError)
    assert isinstance(results[1], DiscriminationResult)
    assert results[1].chosen_value == "high"


def test_out_of_range_choice_index_isolated_to_its_own_position(fake_agent):
    key_a = "field_0"
    fake_agent.answer_overrides[key_a] = {"choice": "candidate_99", "confidence": 0.5}
    model = _model(fake_agent)

    results = model.discriminate_batch(
        "transcript", [(FIELD_A, CANDIDATES_A), (FIELD_B, CANDIDATES_B)]
    )

    assert isinstance(results[0], ValueError)
    assert isinstance(results[1], DiscriminationResult)


def test_chunks_at_batch_max_size_boundary(fake_agent, monkeypatch):
    monkeypatch.setattr(trellis_model_module.settings, "trellis_batch_max_size", 3)
    model = _model(fake_agent)
    items = [(FIELD_A, CANDIDATES_A) for _ in range(7)]

    results = model.discriminate_batch("transcript", items)

    assert len(results) == 7
    assert all(isinstance(r, DiscriminationResult) for r in results)
    chunk_sizes = [len(questions) for _transcript, questions in fake_agent.system_one_calls]
    assert chunk_sizes == [3, 3, 1]


def test_later_chunk_failure_does_not_discard_earlier_chunk_results(fake_agent, monkeypatch):
    monkeypatch.setattr(trellis_model_module.settings, "trellis_batch_max_size", 3)
    fake_agent.raise_on_call_number = 2  # the second chunk's system_one call fails outright
    model = _model(fake_agent)
    items = [(FIELD_A, CANDIDATES_A) for _ in range(7)]  # chunks of 3, 3, 1

    results = model.discriminate_batch("transcript", items)

    assert len(results) == 7
    assert isinstance(results[0], DiscriminationResult)  # chunk 1 succeeded
    assert isinstance(results[1], DiscriminationResult)
    assert isinstance(results[2], DiscriminationResult)
    assert isinstance(results[3], RuntimeError)  # chunk 2 failed outright
    assert isinstance(results[4], RuntimeError)
    assert isinstance(results[5], RuntimeError)
    assert isinstance(results[6], DiscriminationResult)  # chunk 3 still ran and succeeded


def test_discriminate_wraps_batch_for_a_single_item(fake_agent):
    model = _model(fake_agent)

    result = model.discriminate("transcript", FIELD_A, CANDIDATES_A)

    assert result.chosen_value == "leak"


def test_discriminate_sends_the_original_unsuffixed_wire_key(fake_agent):
    model = _model(fake_agent)

    model.discriminate("transcript", FIELD_A, CANDIDATES_A)

    _transcript, questions = fake_agent.system_one_calls[0]
    assert list(questions.keys()) == ["field"]


def test_discriminate_reraises_single_item_error(fake_agent):
    model = _model(fake_agent)

    with pytest.raises(ValueError):
        model.discriminate("transcript", FIELD_A, [])
