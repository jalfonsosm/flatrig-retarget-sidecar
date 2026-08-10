from __future__ import annotations

import json

import pytest

from flatrig.scene_formats import (
    BlenderProbe,
    SceneCommandResult,
    bake_predicted_rig,
    export_3d_rest_bvh,
    extract_animations,
    extract_scene,
    inspect_3d_source,
    probe_scene_backend,
    probe_scene_backend_impl,
    render_sprites,
)


def test_windows_program_files_candidates_prefer_numeric_version(
    monkeypatch, tmp_path
) -> None:
    import flatrig.scene_formats as scene_formats

    for name in ("ProgramW6432", "PROGRAMFILES", "PROGRAMFILES(X86)"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
    installs = [
        tmp_path / "Blender Foundation" / version / "blender.exe"
        for version in ("Blender 4.3", "Blender 4.10", "Blender 5.2")
    ]
    for executable in installs:
        executable.parent.mkdir(parents=True)
        executable.touch()

    candidates = scene_formats._windows_program_files_blender_candidates()

    assert candidates == [installs[2], installs[1], installs[0]]


def test_resolve_blender_executable_uses_windows_app_path(
    monkeypatch, tmp_path
) -> None:
    import flatrig.scene_formats as scene_formats

    executable = tmp_path / "Blender 5.2" / "blender.exe"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.delenv("FLATRIG_RETARGET_BLENDER", raising=False)
    monkeypatch.setattr(scene_formats, "ROOT_DIR", tmp_path / "missing-sidecar-root")
    monkeypatch.setattr(scene_formats.shutil, "which", lambda _name: None)
    monkeypatch.setattr(scene_formats.sys, "platform", "win32")
    monkeypatch.setattr(
        scene_formats,
        "_windows_registry_blender_candidates",
        lambda: [executable],
    )
    monkeypatch.setattr(
        scene_formats,
        "_windows_program_files_blender_candidates",
        lambda: [],
    )

    assert scene_formats.resolve_blender_executable() == executable.resolve()


def test_probe_scene_backend_prefers_bpy(monkeypatch) -> None:
    monkeypatch.delenv("FLATRIG_RETARGET_SCENE_BACKEND", raising=False)
    monkeypatch.setattr("flatrig.scene_formats.sys.platform", "linux")
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


def test_probe_scene_backend_prefers_blender_cli_on_windows(monkeypatch) -> None:
    monkeypatch.delenv("FLATRIG_RETARGET_SCENE_BACKEND", raising=False)
    monkeypatch.setattr("flatrig.scene_formats.sys.platform", "win32")
    monkeypatch.setattr(
        "flatrig.scene_formats.probe_blender_backend",
        lambda: BlenderProbe(
            available=True,
            detail="ready",
            mode="blender_cli",
            executable="blender.exe",
            script="worker.py",
        ),
    )

    def unexpected_bpy_probe() -> BlenderProbe:
        raise AssertionError("Windows auto mode must not import bpy when Blender is available")

    monkeypatch.setattr(
        "flatrig.scene_formats.probe_bpy_backend",
        unexpected_bpy_probe,
    )

    probe = probe_scene_backend_impl()

    assert probe.mode == "blender_cli"
    assert probe.executable == "blender.exe"


def test_inspect_source_uses_bpy_worker_when_available(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FLATRIG_RETARGET_SCENE_BACKEND", "bpy")

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


def test_render_sprites_emits_no_render_quality_toggle(monkeypatch, tmp_path) -> None:
    """Sprite render quality is a fixed internal setting, not a caller option.

    The renderer tunes Eevee itself (see ``texture.DEFAULT_SPRITE_RENDER_SAMPLES``),
    so no speed/quality flag may leak back into the worker command line.
    """
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

    result = render_sprites(
        "source.fbx",
        str(tmp_path / "sprites.json"),
        parts_json="parts.json",
        images_dir="images",
    )

    assert result.ok is True
    assert captured["command"] == "render-sprites"
    assert "--fast-render" not in captured["extra_args"]
