from __future__ import annotations

import json

import pytest

from flatrig.scene_formats import (
    BlenderProbe,
    SceneCommandResult,
    bake_predicted_rig,
    export_3d_rest_bvh,
    extract_animations,
    extract_mesh_targets,
    extract_scene,
    inspect_3d_source,
    probe_scene_backend,
    probe_scene_backend_impl,
    render_sprites,
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


@pytest.mark.parametrize(
    ("call_scene_command", "expected_command"),
    [
        (
            lambda output: extract_scene(
                "source.fbx",
                output,
                view_dir="-0.707107,-0.698325,0.111097",
                view_up="0,0.157115,0.98758",
            ),
            "extract-scene",
        ),
        (
            lambda output: extract_animations(
                "source.fbx",
                output,
                view_dir="-0.707107,-0.698325,0.111097",
                view_up="0,0.157115,0.98758",
            ),
            "extract-animations",
        ),
        (
            lambda output: export_3d_rest_bvh(
                "source.fbx",
                output,
                bvh_output="rest.bvh",
                view_dir="-0.707107,-0.698325,0.111097",
                view_up="0,0.157115,0.98758",
            ),
            "export-3d-rest-bvh",
        ),
        (
            lambda output: render_sprites(
                "source.fbx",
                output,
                parts_json="parts.json",
                images_dir="images",
                view_dir="-0.707107,-0.698325,0.111097",
                view_up="0,0.157115,0.98758",
            ),
            "render-sprites",
        ),
        (
            lambda output: extract_mesh_targets(
                "source.fbx",
                output,
                target_spec="targets.json",
                view_dir="-0.707107,-0.698325,0.111097",
                view_up="0,0.157115,0.98758",
            ),
            "extract-mesh-targets",
        ),
    ],
)
def test_projection_commands_forward_custom_view_vectors_as_single_args(
    monkeypatch, tmp_path, call_scene_command, expected_command
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
        captured.update(command=command, source=source, output=output, extra_args=extra_args)
        return SceneCommandResult(ok=True, detail="ready")

    monkeypatch.setattr("flatrig.scene_formats._run_bpy_command_with_args", run_worker)

    result = call_scene_command(str(tmp_path / "scene.json"))

    assert result.ok is True
    assert captured["command"] == expected_command
    extra_args = captured["extra_args"]
    assert "--view-dir=-0.707107,-0.698325,0.111097" in extra_args
    assert "--view-up=0,0.157115,0.98758" in extra_args
    assert "--view-dir" not in extra_args
    assert "--view-up" not in extra_args


def test_bake_predicted_rig_forwards_original_mesh_path(monkeypatch, tmp_path) -> None:
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
        captured.update(command=command, source=source, output=output, extra_args=extra_args)
        return SceneCommandResult(ok=True, detail="ready")

    monkeypatch.setattr("flatrig.scene_formats._run_bpy_command_with_args", run_worker)
    mesh_path = tmp_path / "source mesh.glb"

    result = bake_predicted_rig(
        "prediction.npz",
        str(tmp_path / "report.json"),
        fbx_output=str(tmp_path / "rigged.fbx"),
        mesh_path=str(mesh_path),
    )

    assert result.ok is True
    assert captured["command"] == "bake-predicted-rig"
    extra_args = captured["extra_args"]
    option_index = extra_args.index("--mesh-path")
    assert extra_args[option_index + 1] == str(mesh_path.resolve())
