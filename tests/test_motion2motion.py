from __future__ import annotations

from pathlib import Path

from flatrig_retarget_sidecar.motion2motion_retarget import (
    SHARED_VENV_DIR,
    SHARED_VENV_PYTHON,
    resolve_motion2motion_device,
)


def test_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.delenv("FLATRIG_M2M_DEVICE", raising=False)
    monkeypatch.delenv("FLATRIG_M2M_ALLOW_MPS", raising=False)
    monkeypatch.setattr(
        "flatrig_retarget_sidecar.motion2motion_retarget._probe_torch_runtime",
        lambda _python, _cwd: {"torch": "2.9.1", "cuda": True, "mps": False},
    )

    assert resolve_motion2motion_device(Path("/tmp/python")) == "cuda"


def test_device_keeps_mps_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("FLATRIG_M2M_DEVICE", raising=False)
    monkeypatch.delenv("FLATRIG_M2M_ALLOW_MPS", raising=False)
    monkeypatch.setattr(
        "flatrig_retarget_sidecar.motion2motion_retarget._probe_torch_runtime",
        lambda _python, _cwd: {"torch": "2.9.1", "cuda": False, "mps": True},
    )

    assert resolve_motion2motion_device(Path("/tmp/python")) == "cpu"

    monkeypatch.setenv("FLATRIG_M2M_ALLOW_MPS", "1")
    assert resolve_motion2motion_device(Path("/tmp/python")) == "mps"


def test_shared_python_path_is_project_venv() -> None:
    assert SHARED_VENV_DIR.name == ".venv"
    assert SHARED_VENV_PYTHON.parent.name in {"bin", "Scripts"}
    assert SHARED_VENV_PYTHON.parent.parent == SHARED_VENV_DIR
