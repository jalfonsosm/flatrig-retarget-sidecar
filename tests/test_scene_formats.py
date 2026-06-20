from __future__ import annotations

import json

from flatrig.scene_formats import (
    BlenderProbe,
    SceneCommandResult,
    extract_scene,
    inspect_3d_source,
    probe_scene_backend,
    probe_scene_backend_impl,
)


def test_probe_scene_backend_prefers_bpy(monkeypatch) -> None:
    monkeypatch.delenv("FLATRIG_RETARGET_SCENE_BACKEND", raising=False)
    monkeypatch.setattr(
        "flatrig.scene_formats.probe_bpy_backend",
        lambda: BlenderProbe(available=True, detail="ready", mode="bpy_module", script="worker.py"),
    )
    monkeypatch.setattr(
        "flatrig.scene_formats.probe_blender_backend",
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
        "flatrig.scene_formats.probe_bpy_backend",
        lambda: BlenderProbe(available=True, detail="ready", mode="bpy_module", script="worker.py"),
    )
    monkeypatch.setattr(
        "flatrig.scene_formats._load_bpy_worker",
        lambda: Worker,
    )

    output = tmp_path / "inspect.json"
    result = inspect_3d_source("example.fbx", str(output))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.ok is True
    assert payload["detail"] == "ready"
    assert payload["source"].endswith("example.fbx")


def test_extract_scene_forwards_base_color_texture_output(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "flatrig.scene_formats.probe_scene_backend_impl",
        lambda: BlenderProbe(
            available=True,
            detail="ready",
            mode="bpy_module",
            script="worker.py",
        ),
    )

    def run_worker(command, source, output, extra_args):
        captured.update(
            command=command,
            source=source,
            output=output,
            extra_args=extra_args,
        )
        return SceneCommandResult(ok=True, detail="ready")

    monkeypatch.setattr(
        "flatrig.scene_formats._run_bpy_command_with_args",
        run_worker,
    )
    texture_path = tmp_path / "preview diffuse.png"

    result = extract_scene(
        "example.fbx",
        str(tmp_path / "preview.json"),
        base_color_texture_output=str(texture_path),
    )

    assert result.ok is True
    extra_args = captured["extra_args"]
    option_index = extra_args.index("--base-color-texture-output")
    assert extra_args[option_index + 1] == str(texture_path)
