"""`trellis-pod-up` — resume the existing pod if RUNPOD_POD_ID is set, otherwise create a fresh
one. Creating a pod is billable; this prints the hourly-relevant GPU type before doing so."""

from __future__ import annotations

import hangar
from hangar import PodSpec
from rich import print

from trellis.settings import settings


def _pod_spec() -> PodSpec:
    return PodSpec(
        name="trellis-train",
        image=settings.runpod_image,
        gpu_type_id=settings.runpod_gpu_type_id,
        disk_gb=settings.runpod_disk_gb,
        ports=["22/tcp"],
        device_env_key="TRELLIS_DEVICE",
        device_env_value=settings.trellis_device,
        pod_id=settings.runpod_pod_id or None,
        extra_env={
            "POD_MAX_HOURS": str(settings.pod_max_hours),
            "POD_IDLE_TIMEOUT_S": str(settings.pod_idle_timeout_s),
        },
        registry_id=settings.runpod_registry_id,
    )


def main() -> None:
    if not settings.runpod_api_key:
        print("[red]RUNPOD_API_KEY is not set.[/red]")
        raise SystemExit(1)

    hangar.init(settings.runpod_api_key)
    spec = _pod_spec()
    existing_pod_id = settings.runpod_pod_id

    if not existing_pod_id:
        print(
            f"[yellow]Creating a new {settings.runpod_gpu_type_id} pod. "
            "This is billable for as long as it runs.[/yellow]"
        )

    pod_id = hangar.start_pod(spec)

    if existing_pod_id and pod_id == existing_pod_id:
        print(f"[green]Pod {pod_id} starting.[/green]")
    else:
        if existing_pod_id:
            print(
                f"[yellow]Pod {existing_pod_id}'s host has no free GPU capacity — "
                "terminated it and created a fresh one on a different host. This is billable "
                "for as long as it runs.[/yellow]"
            )
        print(f"[green]Pod created: {pod_id}.[/green] Save this as RUNPOD_POD_ID for future runs.")

    print(
        f"The in-pod autostop watchdog stops the pod on either of two independent conditions: "
        f"a fixed ceiling of POD_MAX_HOURS ({settings.pod_max_hours}h) regardless of activity, "
        f"or hangar's idle-detection firing after POD_IDLE_TIMEOUT_S "
        f"({settings.pod_idle_timeout_s}s) of no heartbeat and no SSH activity — see "
        "stack/autostop.py. Start training over SSH with `hangar-run <command>` (not directly) "
        "so the heartbeat flows and idle-detection can see it's alive."
    )


if __name__ == "__main__":
    main()
