"""Renders the three-arm (trellis / GPT-5.1 / Jev) eval comparison as markdown: one accuracy/
precision-recall/state-breakdown table per category, plus a per-field accuracy breakdown within
each category. Adapted from laya-bench's `eval/report.py::render_markdown`/`_dataset_table`
structure — backends-as-rows tables per dataset — with the dataset axis remapped to trellis's
eval-set categories and an explicit closed-set-vs-open-extraction caveat that trellis's dataset
axis doesn't need (laya-bench's backends are all the same task)."""

from __future__ import annotations

from collections import Counter

from trellis.validate.costs import BACKEND_COSTS, usage_cost
from trellis.validate.metrics import bootstrap_accuracy_ci, latency_percentiles, precision_recall_f1
from trellis.validate.states import FiveState

STATE_ORDER: tuple[FiveState, ...] = (
    "present_correct",
    "present_incorrect",
    "hallucinated",
    "silently_dropped",
    "correctly_absent",
)

_CAVEAT = """> **Different tasks, not a ranked leaderboard.** `trellis` and `jev` are scored on
> **closed-set discrimination**: each is handed the transcript plus a shuffled candidate list
> (the real value, 3 plausible distractors, and — for optional fields — a "not mentioned" option)
> and must pick one. `gpt-5.1` is scored on **open extraction**: it sees only the transcript and
> must produce (or decline to produce) a value from free text, matched against gold via
> `validate/matchers.py`. A closed-set arm's job is strictly easier (bounded candidate set, no
> free-text ambiguity), so a higher closed-set accuracy than gpt-5.1's open-extraction accuracy
> does not mean the closed-set arm is the better model — the two numbers answer different
> questions.
>
> **Cost is real observed token usage, not every arm's cost.** The Cost column is computed from
> each item's actual usage reported by the arm's own API response, at real published rates (see
> `validate/costs.py`) — it's only meaningful for token-billed arms (gpt-5.1, jev). `trellis`'s
> cost is compute (a RunPod GPU or local CPU), not tokens, and is out of scope here — its "n/a"
> means "not applicable," not "$0"."""


def _is_correct(state: FiveState | None) -> bool:
    return state in ("present_correct", "correctly_absent")


def _cost_str(backend_name: str, items: list) -> str:
    cost = BACKEND_COSTS.get(backend_name)
    if cost is None:
        return "n/a"
    total = sum(usage_cost(cost, i.input_tokens, i.output_tokens) for i in items)
    return f"${total:.4f}"


def _arm_row(backend_name: str, mode: str, items: list) -> str:
    valid = [i for i in items if i.error is None]
    errors = len(items) - len(valid)

    if not valid:
        return (
            f"| {backend_name} | {mode} | no data ({errors} errors) | - | - | - | - | - | - | "
            f"{_cost_str(backend_name, valid)} |"
        )

    states = [i.state for i in valid]
    correct = [_is_correct(s) for s in states]
    ci = bootstrap_accuracy_ci(correct)
    prf1 = precision_recall_f1(states)
    counts = Counter(states)
    state_str = " / ".join(f"{counts.get(s, 0)}" for s in STATE_ORDER)

    latencies = [i.latency_ms for i in valid if i.latency_ms is not None]
    pct = latency_percentiles(latencies)

    return (
        f"| {backend_name} | {mode} | {ci.point:.1%} [{ci.low:.1%}, {ci.high:.1%}] | "
        f"{prf1.precision:.3f} | {prf1.recall:.3f} | {prf1.f1:.3f} | {state_str} | "
        f"{errors} | {pct['p50']:.0f} | {_cost_str(backend_name, valid)} |"
    )


def _category_table(category: str, results: dict) -> str:
    lines = [
        f"### {category}",
        "",
        "| Arm | Mode | Accuracy (95% CI) | Precision | Recall | F1 | "
        + " / ".join(STATE_ORDER)
        + " | Errors | p50 ms | Cost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for backend_name, backend_result in results.items():
        dataset_result = backend_result.datasets.get(category)
        items = dataset_result.items if dataset_result else []
        if not items:
            lines.append(
                f"| {backend_name} | {backend_result.mode} | no data | - | - | - | - | - | - | "
                f"{_cost_str(backend_name, items)} |"
            )
            continue
        lines.append(_arm_row(backend_name, backend_result.mode, items))
    return "\n".join(lines)


def _per_field_table(category: str, results: dict) -> str:
    arm_names = list(results.keys())
    fields: list[str] = []
    seen = set()
    for backend_result in results.values():
        dataset_result = backend_result.datasets.get(category)
        if not dataset_result:
            continue
        for item in dataset_result.items:
            if item.field not in seen:
                seen.add(item.field)
                fields.append(item.field)

    lines = [
        f"#### {category}: accuracy by field",
        "",
        "| Field | " + " | ".join(arm_names) + " |",
        "| --- | " + " | ".join("---" for _ in arm_names) + " |",
    ]
    for field_name in fields:
        row = [field_name]
        for arm_name in arm_names:
            dataset_result = results[arm_name].datasets.get(category)
            all_items = dataset_result.items if dataset_result else []
            items = [i for i in all_items if i.field == field_name]
            valid = [i for i in items if i.error is None]
            if not valid:
                errors = len(items)
                row.append("no data" if errors == 0 else f"{errors} errors")
                continue
            n_correct = sum(1 for i in valid if _is_correct(i.state))
            row.append(f"{n_correct}/{len(valid)} ({n_correct / len(valid):.0%})")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(results: dict, categories: list[str]) -> str:
    sections = ["# Trellis three-arm eval report", "", _CAVEAT, ""]
    for category in categories:
        sections.append(_category_table(category, results))
        sections.append("")
        sections.append(_per_field_table(category, results))
        sections.append("")
    return "\n".join(sections)
