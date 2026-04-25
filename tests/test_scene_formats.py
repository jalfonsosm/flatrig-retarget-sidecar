from __future__ import annotations

import json

from flatrig_retarget_sidecar.scene_formats import (
    BlenderProbe,
    inspect_3d_source,
    probe_scene_backend,
    probe_scene_backend_impl,
)


def test_probe_scene_backend_prefers_bpy(monkeypatch) -> None:
    monkeypatch.delenv("FLATRIG_RETARGET_SCENE_BACKEND", raising=False)
    monkeypatch.setattr(
        "flatrig_retarget_sidecar.scene_formats.probe_bpy_backend",
        lambda: BlenderProbe(available=True, detail="ready", mode="bpy_module", script="worker.py"),
    )
    monkeypatch.setattr(
        "flatrig_retarget_sidecar.scene_formats.probe_blender_backend",
        lambda: BlenderProbe(
            available=True,
            detail="ready",
            mode="blender_cli",
            executable="blender",
            script="worker.py",
        ),
    )

    probe = probe_scene_backend_impl()
    payload = probe_scene_backend()
    assert probe.mode == "bpy_module"
    assert payload["available"] is True


def test_inspect_source_uses_bpy_worker_when_available(monkeypatch, tmp_path) -> None:
    class Worker:
        @staticmethod
        def inspect_source(source: str) -> dict[str, object]:
            return {"ok": True, "detail": "ready", "source": source}

    monkeypatch.setattr(
        "flatrig_retarget_sidecar.scene_formats.probe_bpy_backend",
        lambda: BlenderProbe(available=True, detail="ready", mode="bpy_module", script="worker.py"),
    )
    monkeypatch.setattr(
        "flatrig_retarget_sidecar.scene_formats._load_bpy_worker",
        lambda: Worker,
    )

    output = tmp_path / "inspect.json"
    result = inspect_3d_source("example.fbx", str(output))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.ok is True
    assert payload["detail"] == "ready"
    assert payload["source"].endswith("example.fbx")
