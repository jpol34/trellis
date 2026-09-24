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

## Training

Bring up the training pod with `trellis-pod-up`, then SSH in and start training via `hangar-run`
so the shared heartbeat file stays fresh for as long as training runs:

```
hangar-run python3 -m trellis.train ...
```

`hangar-run` is meant to wrap the training command itself, not an interactive shell — wrapping a
shell would mask genuine human idleness that SSH-idle detection would otherwise catch, since the
shell staying alive keeps the heartbeat fresh regardless of whether anyone's actually using it.

The pod's watchdog (`stack/autostop.py`) stops it on either of two independent conditions: a fixed
`POD_MAX_HOURS` ceiling regardless of activity, or `POD_IDLE_TIMEOUT_S` of no heartbeat and no SSH
activity.
