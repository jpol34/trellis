"""`trellis-pod-down` — manually stop the RunPod pod (the autostop watchdog is the safety net,
not the primary stop path)."""

from __future__ import annotations

import hangar
from rich import print

from trellis.settings import settings


def main() -> None:
    if not settings.runpod_pod_id:
        print("[red]RUNPOD_POD_ID is not set — nothing to stop.[/red]")
        raise SystemExit(1)
    hangar.init(settings.runpod_api_key)
    result = hangar.stop_pod(settings.runpod_pod_id)
    print(f"[green]Pod {settings.runpod_pod_id} stopping.[/green] {result}")


if __name__ == "__main__":
    main()
