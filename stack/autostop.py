"""Hard, activity-independent auto-stop watchdog.

RunPod Pods run one container per pod (no Docker Compose support), so this runs as PID 1 in the
training container — see stack/entrypoint.sh, which execs into this script directly. That's
deliberate: the watchdog must keep running (and keep the container alive) regardless of what
training or other work is happening in the pod, since its whole job is guaranteeing an upper
bound on pod runtime independent of anything else's health.

Persists its start timestamp to disk so it survives being killed and restarted by RunPod without
losing track of the deadline (but not a full pod restart, which is fine — a restarted pod means
the previous deadline no longer applies, and this simply starts a fresh clock).

Adapted from cindrel's `stack/autostop/autostop.py`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

STATE_FILE = Path("/tmp/trellis-autostop-start-time")
POLL_INTERVAL_S = 60
GRAPHQL_URL = "https://api.runpod.io/graphql"


def _get_or_set_start_time() -> float:
    if STATE_FILE.exists():
        return float(STATE_FILE.read_text().strip())
    now = time.time()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(str(now))
    return now


def _stop_pod(pod_id: str, api_key: str) -> None:
    query = f'mutation {{ podStop(input: {{podId: "{pod_id}"}}) {{ id desiredStatus }} }}'
    resp = httpx.post(GRAPHQL_URL, params={"api_key": api_key}, json={"query": query}, timeout=30.0)
    print(f"[autostop] podStop response: {resp.status_code} {resp.text[:300]}")


def main() -> None:
    pod_id = os.environ["RUNPOD_POD_ID"]
    api_key = os.environ["RUNPOD_API_KEY"]
    max_hours = float(os.environ.get("POD_MAX_HOURS", "4"))

    start_time = _get_or_set_start_time()
    print(f"[autostop] watchdog started, deadline in {max_hours}h from {start_time}")

    while True:
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours > max_hours:
            print(f"[autostop] elapsed {elapsed_hours:.2f}h > {max_hours}h — stopping pod {pod_id}")
            _stop_pod(pod_id, api_key)
            # Keep polling (idempotent) in case the stop call needs a retry; the pod stopping
            # will itself kill this container shortly after.
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
