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
    assert spec.pod_id is None


def test_main_calls_hangar_start_pod_with_built_spec(monkeypatch):
    captured = {}

    def fake_init(api_key):
        captured["api_key"] = api_key

    def fake_start_pod(spec):
        captured["spec"] = spec
        return "pod-new"

    monkeypatch.setattr(hangar, "init", fake_init)
    monkeypatch.setattr(hangar, "start_pod", fake_start_pod)

    start_pod.main()

    assert captured["api_key"] == "test-runpod-key"
    assert captured["spec"].name == "trellis-train"
    assert captured["spec"].pod_id is None


def test_main_passes_existing_pod_id_to_spec(monkeypatch):
    monkeypatch.setattr(settings, "runpod_pod_id", "pod-existing")
    captured = {}

    monkeypatch.setattr(hangar, "init", lambda api_key: None)

    def fake_start_pod(spec):
        captured["spec"] = spec
        return spec.pod_id

    monkeypatch.setattr(hangar, "start_pod", fake_start_pod)

    start_pod.main()

    assert captured["spec"].pod_id == "pod-existing"


def test_main_reports_capacity_fallback_recreate(monkeypatch, capsys):
    monkeypatch.setattr(settings, "runpod_pod_id", "pod-existing")

    monkeypatch.setattr(hangar, "init", lambda api_key: None)
    monkeypatch.setattr(hangar, "start_pod", lambda spec: "pod-fresh")

    start_pod.main()

    out = capsys.readouterr().out
    assert "pod-existing" in out
    assert "pod-fresh" in out
    assert "billable" in out
