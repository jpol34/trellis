"""`trellis-pod-up` — resume the existing pod if RUNPOD_POD_ID is set, otherwise create a fresh
one. Creating a pod is billable; this prints the hourly-relevant GPU type before doing so."""

from __future__ import annotations

from rich import print

from trellis.pod.runpod_client import (
    PodCapacityError,
    create_pod,
    delete_pod,
    pod_action,
    update_pod_env,
)
from trellis.settings import settings


def _pod_env() -> dict[str, str]:
    # autostop.py requires RUNPOD_API_KEY and RUNPOD_POD_ID via os.environ[...] — the pod's
    # watchdog won't start without them.
    return {
        "RUNPOD_API_KEY": settings.runpod_api_key,
        "TRELLIS_DEVICE": "cuda",
        "POD_MAX_HOURS": str(settings.pod_max_hours),
    }


def _create_pod() -> str:
    print(
        f"[yellow]Creating a new {settings.runpod_gpu_type_id} pod. "
        "This is billable for as long as it runs.[/yellow]"
    )
    result = create_pod(
        name="trellis-train",
        image=settings.runpod_image,
        gpu_id=settings.runpod_gpu_type_id,
        disk_gb=settings.runpod_disk_gb,
        ports=["22/tcp"],
        env=_pod_env(),
        registry_id=settings.runpod_registry_id,
    )
    pod_id = result["id"]
    # RUNPOD_POD_ID can't be known until the pod exists, so it's set after creation and the pod
    # is restarted to pick it up — env vars are baked in at container start. The RunPod API's
    # pod-env PATCH replaces the whole env map rather than merging, so this must resend every
    # key from _pod_env(), not just the new one.
    update_pod_env(pod_id, {**_pod_env(), "RUNPOD_POD_ID": pod_id})
    pod_action(pod_id, "restart")
    print(f"[green]Pod created: {pod_id}.[/green] Save this as RUNPOD_POD_ID for future runs.")
    return pod_id


def main() -> None:
    if not settings.runpod_api_key:
        print("[red]RUNPOD_API_KEY is not set.[/red]")
        raise SystemExit(1)

    if settings.runpod_pod_id:
        try:
            pod_action(settings.runpod_pod_id, "start")
            print(f"[green]Pod {settings.runpod_pod_id} starting.[/green]")
        except PodCapacityError:
            print(
                f"[yellow]Pod {settings.runpod_pod_id}'s host has no free GPU capacity — "
                "terminating it and creating a fresh one on a different host.[/yellow]"
            )
            delete_pod(settings.runpod_pod_id)
            _create_pod()
    else:
        _create_pod()

    print(
        f"The in-pod autostop watchdog will stop it automatically after POD_MAX_HOURS "
        f"({settings.pod_max_hours}h) regardless of activity — see stack/autostop.py."
    )


if __name__ == "__main__":
    main()
