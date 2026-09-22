"""`trellis-pod-status` — manual sanity check of elapsed runtime vs. the autostop deadline. This
is defense-in-depth only, not a second automated enforcement path."""

from __future__ import annotations

from rich import print

from trellis.pod.runpod_client import pod_status
from trellis.settings import settings


def main() -> None:
    if not settings.runpod_pod_id:
        print("[red]RUNPOD_POD_ID is not set.[/red]")
        raise SystemExit(1)
    status = pod_status(settings.runpod_pod_id)
    uptime_s = (status.get("runtime") or {}).get("uptimeInSeconds", 0)
    uptime_h = uptime_s / 3600
    print(f"Pod {settings.runpod_pod_id}: {status.get('desiredStatus')}, uptime {uptime_h:.2f}h")
    if uptime_h > settings.pod_max_hours * 0.75:
        print(
            f"[yellow]Approaching the {settings.pod_max_hours}h autostop threshold "
            "— the watchdog should fire soon if it hasn't already.[/yellow]"
        )


if __name__ == "__main__":
    main()
