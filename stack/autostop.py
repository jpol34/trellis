"""Watchdog stopping the training pod on either of two independent conditions.

RunPod Pods run one container per pod (no Docker Compose support), so this runs as PID 1 in the
training container — see stack/entrypoint.sh, which execs into this script directly. That's
deliberate: the watchdog must keep running (and keep the container alive) regardless of what
training or other work is happening in the pod.

Two stop conditions are polled each loop iteration, either one sufficient on its own and neither
resetting the other:

- A fixed wall-clock ceiling (`POD_MAX_HOURS`), activity-independent. Its start timestamp is
  persisted to disk so it survives this process being killed and restarted by RunPod without
  losing track of the deadline (but not a full pod restart, which is fine — a restarted pod means
  the previous deadline no longer applies, and this simply starts a fresh clock).
- hangar's idle-detection (`hangar.idle.IdleWatchdog`), activity-based: it tracks the shared
  heartbeat file (kept fresh by `hangar-run`-wrapped training commands) and SSH/tty activity,
  reporting idle once both have been stale past `POD_IDLE_TIMEOUT_S`.

Adapted from cindrel's `stack/autostop/autostop.py`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import hangar
from hangar.idle import IdleWatchdog

STATE_FILE = Path("/tmp/trellis-autostop-start-time")
POLL_INTERVAL_S = 60
DEFAULT_IDLE_TIMEOUT_S = 3600.0


def _get_or_set_start_time() -> float:
    if STATE_FILE.exists():
        return float(STATE_FILE.read_text().strip())
    now = time.time()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(str(now))
    return now


def _stop_pod(pod_id: str) -> None:
    """Calls `hangar.stop_pod`, swallowing any error so a transient API failure (auth hiccup,
    RunPod-side 5xx, rate limit) can't kill this loop — the next tick retries."""
    try:
        hangar.stop_pod(pod_id)
    except Exception as exc:  # noqa: BLE001 - must not let this loop die
        print(f"[autostop] stop_pod failed, will retry next tick: {exc}")


def _poll_once(
    pod_id: str, start_time: float, max_hours: float, idle_watchdog: IdleWatchdog
) -> None:
    """Checks both stop conditions once and calls `hangar.stop_pod` if either fires. Idempotent,
    so the caller can keep polling on the same condition tick over tick until the pod stopping
    kills this container."""
    elapsed_hours = (time.time() - start_time) / 3600
    if elapsed_hours > max_hours:
        print(f"[autostop] elapsed {elapsed_hours:.2f}h > {max_hours}h — stopping pod {pod_id}")
        _stop_pod(pod_id)
    elif idle_watchdog.poll():
        print(f"[autostop] idle past {idle_watchdog.threshold_s}s — stopping pod {pod_id}")
        _stop_pod(pod_id)


def main() -> None:
    pod_id = os.environ["RUNPOD_POD_ID"]
    api_key = os.environ["RUNPOD_API_KEY"]
    max_hours = float(os.environ.get("POD_MAX_HOURS", "4"))
    idle_timeout_s = float(os.environ.get("POD_IDLE_TIMEOUT_S", str(DEFAULT_IDLE_TIMEOUT_S)))

    hangar.init(api_key)
    idle_watchdog = IdleWatchdog(idle_timeout_s)

    start_time = _get_or_set_start_time()
    print(f"[autostop] watchdog started, ceiling deadline in {max_hours}h from {start_time}")
    print(f"[autostop] idle timeout: {idle_timeout_s}s")

    while True:
        _poll_once(pod_id, start_time, max_hours, idle_watchdog)
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
