import importlib.util
import time
from pathlib import Path

import hangar

_SPEC = importlib.util.spec_from_file_location(
    "autostop", Path(__file__).resolve().parents[2] / "stack" / "autostop.py"
)
autostop = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(autostop)


class _FakeIdleWatchdog:
    def __init__(self, is_idle: bool):
        self.threshold_s = 60.0
        self._is_idle = is_idle
        self.poll_calls = 0

    def poll(self) -> bool:
        self.poll_calls += 1
        return self._is_idle


def test_poll_once_stops_on_ceiling_regardless_of_idle(monkeypatch):
    stopped = []
    monkeypatch.setattr(hangar, "stop_pod", lambda pod_id: stopped.append(pod_id))
    watchdog = _FakeIdleWatchdog(is_idle=False)

    autostop._poll_once("pod-1", start_time=time.time() - 5 * 3600, max_hours=4, idle_watchdog=watchdog)

    assert stopped == ["pod-1"]


def test_poll_once_stops_on_idle_regardless_of_ceiling(monkeypatch):
    stopped = []
    monkeypatch.setattr(hangar, "stop_pod", lambda pod_id: stopped.append(pod_id))
    watchdog = _FakeIdleWatchdog(is_idle=True)

    autostop._poll_once("pod-1", start_time=time.time(), max_hours=4, idle_watchdog=watchdog)

    assert stopped == ["pod-1"]


def test_poll_once_does_not_stop_when_neither_condition_fires(monkeypatch):
    stopped = []
    monkeypatch.setattr(hangar, "stop_pod", lambda pod_id: stopped.append(pod_id))
    watchdog = _FakeIdleWatchdog(is_idle=False)

    autostop._poll_once("pod-1", start_time=time.time(), max_hours=4, idle_watchdog=watchdog)

    assert stopped == []


def test_poll_once_only_stops_once_when_both_conditions_fire(monkeypatch):
    stopped = []
    monkeypatch.setattr(hangar, "stop_pod", lambda pod_id: stopped.append(pod_id))
    watchdog = _FakeIdleWatchdog(is_idle=True)

    autostop._poll_once("pod-1", start_time=time.time() - 5 * 3600, max_hours=4, idle_watchdog=watchdog)

    assert stopped == ["pod-1"]


def test_stop_pod_swallows_errors_so_the_poll_loop_survives(monkeypatch, capsys):
    def failing_stop_pod(pod_id):
        raise RuntimeError("RunPod API error: boom")

    monkeypatch.setattr(hangar, "stop_pod", failing_stop_pod)

    autostop._stop_pod("pod-1")  # must not raise

    assert "boom" in capsys.readouterr().out


def test_get_or_set_start_time_persists_across_calls(tmp_path, monkeypatch):
    state_file = tmp_path / "start-time"
    monkeypatch.setattr(autostop, "STATE_FILE", state_file)

    first = autostop._get_or_set_start_time()
    second = autostop._get_or_set_start_time()

    assert first == second
    assert state_file.exists()
