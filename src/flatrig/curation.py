"""Public Blender-worker commands used by offline asset curation.

The private repository keeps catalog policy and declarative rig proportions.
This module owns only the process boundary for the Blender operations needed to
inspect, export, and edit those assets.  It intentionally returns the same
``SceneCommandResult`` shape as the primary scene-format API.
"""

from __future__ import annotations

from pathlib import Path

from flatrig.scene_formats import ROOT_DIR, SceneCommandResult, run_blender_worker_command


CURATION_WORKER_SCRIPT = ROOT_DIR / "tools" / "blender_curation_io.py"


def _run_curation_worker(
    command: str,
    source: str,
    output: str,
    extra_args: list[str] | None = None,
) -> SceneCommandResult:
    return run_blender_worker_command(
        CURATION_WORKER_SCRIPT,
        command,
        source,
        output,
        extra_args,
    )


def inspect_curation_source(
    source: str,
    output: str,
    *,
    source_root: str,
) -> SceneCommandResult:
    """List source actions while preserving Mesh2Motion library relocation."""
    return _run_curation_worker(
        "inspect-curation-source",
        source,
        output,
        ["--source-root", str(Path(source_root).expanduser().resolve())],
    )


def export_curation_fbx(
    source: str,
    output: str,
    *,
    source_root: str,
    fbx_output: str,
    action_name: str | None = None,
    embed_textures: bool = False,
    clean_control_bones: bool = False,
) -> SceneCommandResult:
    """Export a curated armature/mesh pair with the selected action baked."""
    extra_args = [
        "--source-root",
        str(Path(source_root).expanduser().resolve()),
        "--fbx-output",
        str(Path(fbx_output).expanduser().resolve()),
    ]
    if action_name:
        extra_args.extend(["--action-name", action_name])
    if embed_textures:
        extra_args.append("--embed-textures")
    if clean_control_bones:
        extra_args.append("--clean-control-bones")
    return _run_curation_worker("export-curation-fbx", source, output, extra_args)


def rewrite_armature_rest_pose(
    source: str,
    output: str,
    *,
    fbx_output: str,
    rest_spec: str,
    action_name: str | None = None,
) -> SceneCommandResult:
    """Apply a declarative edit-bone rest pose and export it as FBX."""
    extra_args = [
        "--fbx-output",
        str(Path(fbx_output).expanduser().resolve()),
        "--rest-spec",
        str(Path(rest_spec).expanduser().resolve()),
    ]
    if action_name:
        extra_args.extend(["--action-name", action_name])
    return _run_curation_worker("rewrite-armature-rest-pose", source, output, extra_args)
