"""RunPod API client: GraphQL for start/stop/status of an existing pod (the original, minimal
surface), plus REST v2 for creating/updating/deleting pods — needed because a stopped pod is
pinned to the host it last ran on, and that host sometimes has no free GPU capacity for a
`start`, with no fallback but to terminate and create a fresh pod on a different host.

Adapted from cindrel's `src/cindrel/pod/runpod_client.py`.
"""

from __future__ import annotations

import httpx

from trellis.settings import settings

_GRAPHQL_URL = "https://api.runpod.io/graphql"
_REST_V2_URL = "https://api.runpod.io/v2"


def _run(query: str) -> dict:
    resp = httpx.post(
        _GRAPHQL_URL,
        params={"api_key": settings.runpod_api_key},
        json={"query": query},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"RunPod API error: {data['errors']}")
    return data["data"]


def resume_pod(pod_id: str) -> dict:
    return _run(f'mutation {{ podResume(input: {{podId: "{pod_id}"}}) {{ id desiredStatus }} }}')


def stop_pod(pod_id: str) -> dict:
    return _run(f'mutation {{ podStop(input: {{podId: "{pod_id}"}}) {{ id desiredStatus }} }}')


def pod_status(pod_id: str) -> dict:
    data = _run(
        f'query {{ pod(input: {{podId: "{pod_id}"}}) '
        "{ id desiredStatus runtime { uptimeInSeconds } } }"
    )
    return data["pod"]


class PodCapacityError(RuntimeError):
    """Raised when a pod action fails because its pinned host has no free GPU capacity — the
    caller's only recourse is to terminate the pod and create a fresh one on a different host."""


def _rest_client() -> httpx.Client:
    return httpx.Client(
        base_url=_REST_V2_URL,
        headers={"Authorization": f"Bearer {settings.runpod_api_key}"},
        timeout=30.0,
    )


def get_pod(pod_id: str) -> dict | None:
    """Returns the pod's REST v2 representation, or None if it no longer exists."""
    with _rest_client() as client:
        resp = client.get(f"/pods/{pod_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


def create_pod(
    *,
    name: str,
    image: str,
    gpu_id: str,
    disk_gb: int,
    ports: list[str],
    env: dict[str, str],
    registry_id: str = "",
) -> dict:
    with _rest_client() as client:
        resp = client.post(
            "/pods",
            json={
                "name": name,
                "image": image,
                "cloud": "SECURE",
                "gpu": {"id": gpu_id, "count": 1},
                "disk": disk_gb,
                "ports": ports,
                "startSsh": True,
                "env": env,
                "registry": registry_id or None,
            },
        )
        resp.raise_for_status()
        return resp.json()


def update_pod_env(pod_id: str, env: dict[str, str]) -> dict:
    with _rest_client() as client:
        resp = client.patch(f"/pods/{pod_id}", json={"env": env})
        resp.raise_for_status()
        return resp.json()


def pod_action(pod_id: str, action: str) -> dict:
    """action: 'start' | 'stop' | 'restart' | 'terminate'. Raises PodCapacityError on the
    known "not enough free GPUs on the host machine" failure mode for a `start`."""
    with _rest_client() as client:
        resp = client.post(f"/pods/{pod_id}/action", json={"action": action})
        if resp.status_code == 400 and "not enough free gpus" in resp.text.lower():
            raise PodCapacityError(resp.text)
        resp.raise_for_status()
        return resp.json()


def delete_pod(pod_id: str) -> None:
    with _rest_client() as client:
        resp = client.delete(f"/pods/{pod_id}")
        if resp.status_code not in (200, 204, 404):
            resp.raise_for_status()
