"""Covers #15's report.py: the closed-set-vs-open-extraction caveat renders prominently (not as
a footnote), and per-category/per-field tables render correctly from `BackendResult` data,
including the "no data" and "errors" branches."""

from __future__ import annotations

from trellis.validate.report import render_markdown
from trellis.validate.run_eval import BackendResult, DatasetResult, ItemResult


def _item(
    field: str,
    state,
    error: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> ItemResult:
    return ItemResult(
        item_id="i1",
        category="prospect",
        field=field,
        predicted_value="x",
        gold_value="x",
        confidence=0.9,
        latency_ms=1.0,
        state=state,
        error=error,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def test_caveat_appears_before_any_data_table() -> None:
    results = {
        "trellis": BackendResult(
            backend_name="trellis",
            mode="closed_set",
            datasets={"prospect": DatasetResult("prospect", [_item("name", "present_correct")])},
        )
    }
    report = render_markdown(results, ["prospect"])

    caveat_idx = report.index("Different tasks, not a ranked leaderboard")
    table_idx = report.index("### prospect")
    assert caveat_idx < table_idx
    assert "closed-set" in report
    assert "open extraction" in report


def test_report_renders_all_five_states_and_per_field_breakdown() -> None:
    items = [
        _item("name", "present_correct"),
        _item("name", "present_incorrect"),
        _item("pet_info", "hallucinated"),
        _item("pet_info", "silently_dropped"),
        _item("pet_info", "correctly_absent"),
    ]
    results = {
        "trellis": BackendResult(
            backend_name="trellis",
            mode="closed_set",
            datasets={"prospect": DatasetResult("prospect", items)},
        )
    }
    report = render_markdown(results, ["prospect"])

    assert "trellis" in report
    assert "#### prospect: accuracy by field" in report
    assert "| name |" in report
    assert "| pet_info |" in report


def test_report_handles_missing_dataset_for_an_arm() -> None:
    results = {
        "trellis": BackendResult(backend_name="trellis", mode="closed_set", datasets={}),
    }
    report = render_markdown(results, ["prospect"])
    assert "no data" in report


def test_report_counts_errors_separately_from_states() -> None:
    items = [
        _item("name", "present_correct"),
        _item("name", None, error="RuntimeError: boom"),
    ]
    results = {
        "gpt-5.1": BackendResult(
            backend_name="gpt-5.1",
            mode="open_extraction",
            datasets={"prospect": DatasetResult("prospect", items)},
        )
    }
    report = render_markdown(results, ["prospect"])
    lines = [line for line in report.splitlines() if line.startswith("| gpt-5.1 |")]
    assert len(lines) == 1
    assert "| 1 |" in lines[0]  # one error counted, doesn't corrupt the accuracy denominator


def test_trellis_cost_is_not_applicable_not_zero() -> None:
    results = {
        "trellis": BackendResult(
            backend_name="trellis",
            mode="closed_set",
            datasets={
                "prospect": DatasetResult(
                    "prospect", [_item("name", "present_correct", input_tokens=100)]
                )
            },
        )
    }
    report = render_markdown(results, ["prospect"])
    lines = [line for line in report.splitlines() if line.startswith("| trellis |")]
    assert lines[0].rstrip("|").rstrip().endswith("n/a")


def test_gpt51_cost_uses_real_input_and_output_rates() -> None:
    # gpt-5.1: $1.25/MTok input, $10.00/MTok output.
    results = {
        "gpt-5.1": BackendResult(
            backend_name="gpt-5.1",
            mode="open_extraction",
            datasets={
                "prospect": DatasetResult(
                    "prospect",
                    [
                        _item(
                            "name",
                            "present_correct",
                            input_tokens=1_000_000,
                            output_tokens=100_000,
                        )
                    ],
                )
            },
        )
    }
    report = render_markdown(results, ["prospect"])
    lines = [line for line in report.splitlines() if line.startswith("| gpt-5.1 |")]
    # 1M input tokens * $1.25 + 100k output tokens * $10.00/M = $1.25 + $1.00 = $2.25
    assert "$2.2500" in lines[0]


def test_jev_cost_ignores_output_tokens_per_free_output_pricing() -> None:
    # jev: $0.042/MTok input, output free.
    results = {
        "jev": BackendResult(
            backend_name="jev",
            mode="closed_set",
            datasets={
                "prospect": DatasetResult(
                    "prospect",
                    [
                        _item(
                            "name",
                            "present_correct",
                            input_tokens=1_000_000,
                            output_tokens=999_999,
                        )
                    ],
                )
            },
        )
    }
    report = render_markdown(results, ["prospect"])
    lines = [line for line in report.splitlines() if line.startswith("| jev |")]
    assert "$0.0420" in lines[0]
