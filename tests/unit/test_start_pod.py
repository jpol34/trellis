import hangar
import pytest

from trellis.pod import start_pod
from trellis.settings import settings


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(settings, "runpod_api_key", "test-runpod-key")
    monkeypatch.setattr(settings, "runpod_pod_id", "")
    monkeypatch.setattr(settings, "runpod_image", "ghcr.io/jpol34/trellis-gpu:latest")
    monkeypatch.setattr(settings, "runpod_gpu_type_id", "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(settings, "runpod_disk_gb", 20)
    monkeypatch.setattr(settings, "runpod_registry_id", "")
    monkeypatch.setattr(settings, "trellis_device", "cuda")
    monkeypatch.setattr(settings, "pod_max_hours", 4)


def test_main_exits_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "runpod_api_key", "")

    with pytest.raises(SystemExit):
        start_pod.main()


def test_pod_spec_matches_settings():
    spec = start_pod._pod_spec()

    assert spec.name == "trellis-train"
    assert spec.image == "ghcr.io/jpol34/trellis-gpu:latest"
    assert spec.gpu_type_id == "NVIDIA GeForce RTX 4090"
    assert spec.disk_gb == 20
    assert spec.ports == ["22/tcp"]
    assert spec.device_env_key == "TRELLIS_DEVICE"
    assert spec.device_env_value == "cuda"
    assert spec.extra_env == {"POD_MAX_HOURS": "4"}


def test_main_creates_pod_when_no_pod_id_set(monkeypatch):
    calls = {}

    def fake_start_pod(spec):
        calls["spec"] = spec
        return "pod-new"

    monkeypatch.setattr(hangar, "init", lambda api_key: calls.setdefault("init_key", api_key))
    monkeypatch.setattr(hangar, "start_pod", fake_start_pod)

    start_pod.main()

    assert calls["init_key"] == "test-runpod-key"
    assert calls["spec"].name == "trellis-train"


def test_main_resumes_existing_pod(monkeypatch):
    monkeypatch.setattr(settings, "runpod_pod_id", "pod-existing")
    calls = {}

    monkeypatch.setattr(hangar, "init", lambda api_key: None)
    monkeypatch.setattr(
        hangar, "pod_action", lambda pod_id, action: calls.setdefault("action", (pod_id, action))
    )
    monkeypatch.setattr(hangar, "start_pod", lambda spec: pytest.fail("should not create a pod"))

    start_pod.main()

    assert calls["action"] == ("pod-existing", "start")


def test_main_recreates_pod_on_capacity_error(monkeypatch):
    monkeypatch.setattr(settings, "runpod_pod_id", "pod-existing")
    calls = {"deleted": None}

    def fake_pod_action(pod_id, action):
        raise hangar.PodCapacityError("no capacity")

    monkeypatch.setattr(hangar, "init", lambda api_key: None)
    monkeypatch.setattr(hangar, "pod_action", fake_pod_action)
    monkeypatch.setattr(hangar, "delete_pod", lambda pod_id: calls.__setitem__("deleted", pod_id))
    monkeypatch.setattr(hangar, "start_pod", lambda spec: "pod-fresh")

    start_pod.main()

    assert calls["deleted"] == "pod-existing"
