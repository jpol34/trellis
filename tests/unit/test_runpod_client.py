import json

import httpx
import pytest

from trellis.pod.runpod_client import (
    PodCapacityError,
    create_pod,
    delete_pod,
    get_pod,
    pod_action,
    pod_status,
    resume_pod,
    stop_pod,
    update_pod_env,
)
from trellis.settings import settings


@pytest.fixture(autouse=True)
def _api_key():
    original = settings.runpod_api_key
    settings.runpod_api_key = "test-runpod-key"
    yield
    settings.runpod_api_key = original


def _fake_rest_client(monkeypatch, handler):
    def fake_rest_client():
        return httpx.Client(
            base_url="https://api.runpod.io/v2",
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr("trellis.pod.runpod_client._rest_client", fake_rest_client)


def _fake_graphql(monkeypatch, handler):
    def fake_post(url, **kwargs):
        request = httpx.Request("POST", url)
        response = handler(request, kwargs)
        response.request = request
        return response

    monkeypatch.setattr("trellis.pod.runpod_client.httpx.post", fake_post)


def test_pod_action_raises_pod_capacity_error_on_400(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Error: not enough free GPUs on the host machine")

    def fake_rest_client():
        return httpx.Client(
            base_url="https://api.runpod.io/v2",
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr("trellis.pod.runpod_client._rest_client", fake_rest_client)

    with pytest.raises(PodCapacityError):
        pod_action("pod-123", "start")


def test_pod_action_raises_plain_error_on_other_400(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Error: something unrelated")

    _fake_rest_client(monkeypatch, handler)

    with pytest.raises(httpx.HTTPStatusError):
        pod_action("pod-123", "start")


def test_create_pod_sends_expected_body_and_returns_response(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "pod-new"})

    _fake_rest_client(monkeypatch, handler)

    result = create_pod(
        name="trellis-train",
        image="ghcr.io/example/trellis:latest",
        gpu_id="NVIDIA GeForce RTX 4090",
        disk_gb=20,
        ports=["22/tcp"],
        env={"FOO": "bar"},
        registry_id="reg-123",
    )

    assert result == {"id": "pod-new"}
    assert captured["body"]["image"] == "ghcr.io/example/trellis:latest"
    assert captured["body"]["registry"] == "reg-123"
    assert captured["body"]["env"] == {"FOO": "bar"}


def test_create_pod_omits_registry_when_not_set(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "pod-new"})

    _fake_rest_client(monkeypatch, handler)

    create_pod(
        name="trellis-train",
        image="public/image:latest",
        gpu_id="NVIDIA GeForce RTX 4090",
        disk_gb=20,
        ports=["22/tcp"],
        env={},
    )

    assert captured["body"]["registry"] is None


def test_update_pod_env_patches_and_returns_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert json.loads(request.content) == {"env": {"A": "1"}}
        return httpx.Response(200, json={"id": "pod-123"})

    _fake_rest_client(monkeypatch, handler)

    assert update_pod_env("pod-123", {"A": "1"}) == {"id": "pod-123"}


def test_get_pod_returns_none_on_404(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    _fake_rest_client(monkeypatch, handler)

    assert get_pod("pod-missing") is None


def test_get_pod_returns_representation_when_found(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "pod-123", "desiredStatus": "RUNNING"})

    _fake_rest_client(monkeypatch, handler)

    assert get_pod("pod-123") == {"id": "pod-123", "desiredStatus": "RUNNING"}


def test_delete_pod_tolerates_404(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        return httpx.Response(404)

    _fake_rest_client(monkeypatch, handler)

    delete_pod("pod-already-gone")  # must not raise


def test_delete_pod_raises_on_server_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _fake_rest_client(monkeypatch, handler)

    with pytest.raises(httpx.HTTPStatusError):
        delete_pod("pod-123")


def test_resume_pod_sends_graphql_mutation(monkeypatch):
    def handler(request: httpx.Request, kwargs) -> httpx.Response:
        assert "podResume" in kwargs["json"]["query"]
        assert "pod-123" in kwargs["json"]["query"]
        return httpx.Response(
            200, json={"data": {"podResume": {"id": "pod-123", "desiredStatus": "RUNNING"}}}
        )

    _fake_graphql(monkeypatch, handler)

    assert resume_pod("pod-123") == {"podResume": {"id": "pod-123", "desiredStatus": "RUNNING"}}


def test_stop_pod_sends_graphql_mutation(monkeypatch):
    def handler(request: httpx.Request, kwargs) -> httpx.Response:
        assert "podStop" in kwargs["json"]["query"]
        return httpx.Response(
            200, json={"data": {"podStop": {"id": "pod-123", "desiredStatus": "EXITED"}}}
        )

    _fake_graphql(monkeypatch, handler)

    assert stop_pod("pod-123") == {"podStop": {"id": "pod-123", "desiredStatus": "EXITED"}}


def test_pod_status_returns_pod_data(monkeypatch):
    def handler(request: httpx.Request, kwargs) -> httpx.Response:
        assert "pod-123" in kwargs["json"]["query"]
        return httpx.Response(
            200,
            json={
                "data": {
                    "pod": {
                        "id": "pod-123",
                        "desiredStatus": "RUNNING",
                        "runtime": {"uptimeInSeconds": 42},
                    }
                }
            },
        )

    _fake_graphql(monkeypatch, handler)

    result = pod_status("pod-123")
    assert result["id"] == "pod-123"
    assert result["runtime"]["uptimeInSeconds"] == 42


def test_graphql_error_raises_runtime_error(monkeypatch):
    def handler(request: httpx.Request, kwargs) -> httpx.Response:
        return httpx.Response(200, json={"errors": [{"message": "pod not found"}]})

    _fake_graphql(monkeypatch, handler)

    with pytest.raises(RuntimeError):
        pod_status("pod-missing")
