from __future__ import annotations

from pathlib import Path

from flatrig_retarget_sidecar.motion2motion_retarget import resolve_motion2motion_device


def test_device_prefers_cuda(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("FLATRIG_M2M_DEVICE", raising=False)
    monkeypatch.delenv("FLATRIG_M2M_ALLOW_MPS", raising=False)
    monkeypatch.setattr(
        "flatrig_retarget_sidecar.motion2motion_retarget.resolve_motion2motion_python",
        lambda _dir=None: Path("/tmp/python"),
    )
    monkeypatch.setattr(
        "flatrig_retarget_sidecar.motion2motion_retarget._probe_torch_runtime",
        lambda _python, _cwd: {"torch": "2.9.1", "cuda": True, "mps": False},
    )

    assert resolve_motion2motion_device(tmp_path) == "cuda"


def test_device_keeps_mps_opt_in(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("FLATRIG_M2M_DEVICE", raising=False)
    monkeypatch.delenv("FLATRIG_M2M_ALLOW_MPS", raising=False)
    monkeypatch.setattr(
        "flatrig_retarget_sidecar.motion2motion_retarget.resolve_motion2motion_python",
        lambda _dir=None: Path("/tmp/python"),
    )
    monkeypatch.setattr(
        "flatrig_retarget_sidecar.motion2motion_retarget._probe_torch_runtime",
        lambda _python, _cwd: {"torch": "2.9.1", "cuda": False, "mps": True},
    )

    assert resolve_motion2motion_device(tmp_path) == "cpu"

    monkeypatch.setenv("FLATRIG_M2M_ALLOW_MPS", "1")
    assert resolve_motion2motion_device(tmp_path) == "mps"
