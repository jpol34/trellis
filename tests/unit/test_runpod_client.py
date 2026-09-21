import httpx
import pytest

from trellis.pod.runpod_client import PodCapacityError, pod_action
from trellis.settings import settings


@pytest.fixture(autouse=True)
def _api_key():
    original = settings.runpod_api_key
    settings.runpod_api_key = "test-runpod-key"
    yield
    settings.runpod_api_key = original


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

    def fake_rest_client():
        return httpx.Client(
            base_url="https://api.runpod.io/v2",
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr("trellis.pod.runpod_client._rest_client", fake_rest_client)

    with pytest.raises(httpx.HTTPStatusError):
        pod_action("pod-123", "start")
