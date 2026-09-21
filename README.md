# trellis

A fine-tune of `laya`'s `choice` classification head for closed-set field discrimination on
synthetic property-management call transcripts (residents/prospects categories), benchmarked
against Jev (typesafe.ai) and GPT-5.1.

## Setup

```
uv sync --extra train --extra gen --extra dev
```

## Tests

```
uv run pytest tests/unit
```
