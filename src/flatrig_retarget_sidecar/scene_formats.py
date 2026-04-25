"""3D scene inspection and conversion helpers exposed by the public sidecar."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from flatrig_retarget_sidecar.spine_import import ROOT_DIR

ENV_BLENDER = "FLATRIG_RETARGET_BLENDER"
ENV_SCENE_BACKEND = "FLATRIG_RETARGET_SCENE_BACKEND"
DEFAULT_MACOS_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
BLENDER_SCRIPT = ROOT_DIR / "tools" / "blender_scene_io.py"


@dataclass(slots=True)
class BlenderProbe:
    available: bool
    detail: str
    mode: str | None = None
    executable: str | None = None
    script: str | None = None


@dataclass(slots=True)
class SceneCommandResult:
    ok: bool
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)


def resolve_blender_executable() -> Path | None:
    raw = os.environ.get(ENV_BLENDER)
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return candidate.resolve()

    bundled_candidates = [
        ROOT_DIR / "runtime" / "blender" / "blender",
        ROOT_DIR / "runtime" / "blender" / "bin" / "blender",
        ROOT_DIR / "runtime" / "blender" / "Blender.app" / "Contents" / "MacOS" / "Blender",
        ROOT_DIR / "runtime" / "blender" / "blender.exe",
    ]
    for candidate in bundled_candidates:
        if candidate.exists():
            return candidate.resolve()

    resolved = shutil.which("blender")
    if resolved:
        return Path(resolved).resolve()

    if DEFAULT_MACOS_BLENDER.exists():
        return DEFAULT_MACOS_BLENDER.resolve()

    applications_dir = DEFAULT_MACOS_BLENDER.parent.parent.parent
    if applications_dir.exists():
        for candidate in sorted(applications_dir.glob("Blender*.app/Contents/MacOS/Blender")):
            if candidate.exists():
                return candidate.resolve()
    return None


def _load_bpy_worker():
    spec = importlib_util.spec_from_file_location(
        "flatrig_sidecar_blender_scene_io", BLENDER_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("The public sidecar is missing tools/blender_scene_io.py.")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_bpy_backend() -> BlenderProbe:
    if not BLENDER_SCRIPT.exists():
        return BlenderProbe(
            available=False,
            detail="The public sidecar is missing tools/blender_scene_io.py.",
            mode="bpy_module",
            script=str(BLENDER_SCRIPT),
        )
    try:
        import bpy  # type: ignore  # noqa: F401
    except Exception as exc:
        return BlenderProbe(
            available=False,
            detail=f"Python bpy module unavailable: {exc}",
            mode="bpy_module",
            script=str(BLENDER_SCRIPT),
        )
    return BlenderProbe(
        available=True,
        detail="ready",
        mode="bpy_module",
        script=str(BLENDER_SCRIPT),
    )


def probe_blender_backend() -> BlenderProbe:
    blender = resolve_blender_executable()
    if blender is None:
        return BlenderProbe(
            available=False,
            detail="bpy is unavailable and no bundled Blender fallback was found.",
            mode="blender_cli",
            script=str(BLENDER_SCRIPT),
        )
    if not BLENDER_SCRIPT.exists():
        return BlenderProbe(
            available=False,
            detail="The public sidecar is missing tools/blender_scene_io.py.",
            mode="blender_cli",
            executable=str(blender),
            script=str(BLENDER_SCRIPT),
        )
    return BlenderProbe(
        available=True,
        detail="ready",
        mode="blender_cli",
        executable=str(blender),
        script=str(BLENDER_SCRIPT),
    )


def probe_scene_backend_impl() -> BlenderProbe:
    preferred = (os.environ.get(ENV_SCENE_BACKEND) or "auto").strip().lower()
    if preferred == "bpy":
        return probe_bpy_backend()
    if preferred == "blender":
        return probe_blender_backend()

    bpy_probe = probe_bpy_backend()
    if bpy_probe.available:
        return bpy_probe
    blender_probe = probe_blender_backend()
    if blender_probe.available:
        return blender_probe

    detail = bpy_probe.detail
    if blender_probe.detail and blender_probe.detail != bpy_probe.detail:
        detail = f"{bpy_probe.detail}; {blender_probe.detail}"
    return BlenderProbe(
        available=False,
        detail=detail,
        mode="auto",
        executable=blender_probe.executable,
        script=str(BLENDER_SCRIPT),
    )


def _run_bpy_command(command: str, source: str, output: str) -> SceneCommandResult:
    probe = probe_bpy_backend()
    if not probe.available:
        return SceneCommandResult(
            ok=False,
            detail=probe.detail,
            payload={"ok": False, "detail": probe.detail},
        )

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        worker = _load_bpy_worker()
        source_path = str(Path(source).expanduser().resolve())
        if command == "inspect":
            payload = worker.inspect_source(source_path)
        elif command == "convert":
            payload = worker.convert_source(source_path, str(output_path))
        else:  # pragma: no cover - internal contract
            raise ValueError(f"Unsupported scene command: {command}")
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        payload = {"ok": False, "detail": str(exc)}
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return SceneCommandResult(
            ok=False,
            detail=str(exc),
            payload=payload,
            command=[
                sys.executable,
                str(BLENDER_SCRIPT),
                command,
                source,
                "--output",
                str(output_path),
            ],
        )

    detail = str(payload.get("detail") or "ok")
    return SceneCommandResult(
        ok=bool(payload.get("ok", False)),
        detail=detail,
        payload=payload,
        command=[
            sys.executable,
            str(BLENDER_SCRIPT),
            command,
            source,
            "--output",
            str(output_path),
        ],
    )


def _run_bpy_command_with_args(
    command: str,
    source: str,
    output: str,
    extra_args: list[str] | None = None,
) -> SceneCommandResult:
    """Run the Blender worker with the managed bpy interpreter."""
    probe = probe_bpy_backend()
    if not probe.available:
        return SceneCommandResult(
            ok=False,
            detail=probe.detail,
            payload={"ok": False, "detail": probe.detail},
        )

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        sys.executable,
        str(BLENDER_SCRIPT),
        "--",
        command,
        str(Path(source).expanduser().resolve()),
        "--output",
        str(output_path),
    ]
    if extra_args:
        argv.extend(extra_args)

    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
    )

    payload: dict[str, Any]
    if output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {"ok": False, "detail": "Could not read output JSON."}
    else:
        payload = {
            "ok": False,
            "detail": "The bpy worker did not create the expected output JSON.",
        }

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or payload.get("detail") or "bpy worker command failed."
        payload = {
            **payload,
            "ok": False,
            "detail": detail,
            "stdout": stdout,
            "stderr": stderr,
        }
        return SceneCommandResult(ok=False, detail=detail, payload=payload, command=argv)

    detail = str(payload.get("detail") or "ok")
    return SceneCommandResult(
        ok=bool(payload.get("ok", False)),
        detail=detail,
        payload=payload,
        command=argv,
    )


def _run_blender_command(command: str, source: str, output: str) -> SceneCommandResult:
    probe = probe_blender_backend()
    if not probe.available or probe.executable is None or probe.script is None:
        return SceneCommandResult(
            ok=False,
            detail=probe.detail,
            payload={"ok": False, "detail": probe.detail},
        )

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        probe.executable,
        "--background",
        "--factory-startup",
        "--python",
        probe.script,
        "--",
        command,
        str(Path(source).expanduser().resolve()),
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
    )

    payload: dict[str, Any]
    if output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {"ok": False, "detail": "Blender wrote an unreadable JSON payload."}
    else:
        payload = {
            "ok": False,
            "detail": "The Blender bridge did not create the expected output JSON.",
        }

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or payload.get("detail") or "Blender command failed."
        payload = {
            **payload,
            "ok": False,
            "detail": detail,
            "stdout": stdout,
            "stderr": stderr,
        }
        return SceneCommandResult(ok=False, detail=detail, payload=payload, command=argv)

    detail = str(payload.get("detail") or "ok")
    return SceneCommandResult(
        ok=bool(payload.get("ok", False)), detail=detail, payload=payload, command=argv
    )


def inspect_3d_source(source: str, output: str) -> SceneCommandResult:
    probe = probe_scene_backend_impl()
    if probe.mode == "bpy_module" and probe.available:
        return _run_bpy_command("inspect", source, output)
    return _run_blender_command("inspect", source, output)


def convert_3d_source(source: str, output: str) -> SceneCommandResult:
    probe = probe_scene_backend_impl()
    if probe.mode == "bpy_module" and probe.available:
        return _run_bpy_command("convert", source, output)
    return _run_blender_command("convert", source, output)


def _run_blender_command_with_args(
    command: str,
    source: str,
    output: str,
    extra_args: list[str] = None,
) -> SceneCommandResult:
    """Run a Blender CLI command with extra arguments."""
    probe = probe_blender_backend()
    if not probe.available:
        return SceneCommandResult(
            ok=False,
            detail=probe.detail,
            payload={"ok": False, "detail": probe.detail},
        )

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        probe.executable or sys.executable,
        "--background",
        "--factory-startup",
        "--python",
        probe.script,
        "--",
        command,
        str(Path(source).expanduser().resolve()),
        "--output",
        str(output_path),
    ]
    if extra_args:
        argv.extend(extra_args)

    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
    )

    payload: dict[str, Any]
    if output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {"ok": False, "detail": "Could not read output JSON."}
    else:
        payload = {
            "ok": False,
            "detail": "The Blender command did not create the expected output JSON.",
        }

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or payload.get("detail") or "Blender command failed."
        payload = {
            **payload,
            "ok": False,
            "detail": detail,
            "stdout": stdout,
            "stderr": stderr,
        }
        return SceneCommandResult(ok=False, detail=detail, payload=payload, command=argv)

    detail = str(payload.get("detail") or "ok")
    return SceneCommandResult(
        ok=bool(payload.get("ok", False)),
        detail=detail,
        payload=payload,
        command=argv,
    )


def extract_scene(
    source: str,
    output: str,
    view_preset: str = "side",
    view_dir: str = None,
    view_up: str = None,
    view_roll: float = 0.0,
    source_frame: int = None,
    projection_space: str = "world",
) -> SceneCommandResult:
    """Extract scene data (mesh, bones, weights) using projection."""
    extra_args = [
        "--view-preset",
        view_preset,
        "--projection-space",
        projection_space,
    ]
    if view_dir:
        extra_args.extend(["--view-dir", view_dir])
    if view_up:
        extra_args.extend(["--view-up", view_up])
    if view_roll != 0.0:
        extra_args.extend(["--view-roll", str(view_roll)])
    if source_frame is not None:
        extra_args.extend(["--source-frame", str(source_frame)])

    probe = probe_scene_backend_impl()
    if probe.mode == "bpy_module" and probe.available:
        return _run_bpy_command_with_args("extract-scene", source, output, extra_args)
    return _run_blender_command_with_args("extract-scene", source, output, extra_args)


def extract_animations(
    source: str,
    output: str,
    view_preset: str = "side",
    view_dir: str = None,
    view_up: str = None,
    view_roll: float = 0.0,
    source_frame: int = None,
    projection_space: str = "world",
    animation_names: list = None,
    fps: float = 30.0,
    frame_start: int = None,
    frame_end: int = None,
    sample_substeps: int = 2,
    optimize_animation_keys: bool = True,
    force_loop_closing_keys: bool = False,
    pose_mode: str = "full",
    pose_blend: float = 1.0,
    rotation_flatten: float = 0.0,
    rotation_flatten_scope: str = "all",
    stretch_guard_enabled: bool = False,
    stretch_guard_max_scale: float = 1.75,
    stretch_guard_strength: float = 0.65,
    ik_leaf_refine_enabled: bool = False,
    ik_leaf_strength: float = 0.35,
    ik_leaf_iterations: int = 6,
    ik_leaf_max_chain_length: int = 3,
    ik_leaf_preserve_scale: float = 0.65,
    drop_problematic_frames: bool = False,
    preserve_root_motion: bool = False,
    preserve_root_rotation: bool = False,
) -> SceneCommandResult:
    """Extract animations using projection."""
    extra_args = [
        "--view-preset",
        view_preset,
        "--projection-space",
        projection_space,
        "--fps",
        str(fps),
        "--sample-substeps",
        str(sample_substeps),
        "--pose-mode",
        pose_mode,
        "--pose-blend",
        str(pose_blend),
        "--rotation-flatten",
        str(rotation_flatten),
        "--rotation-flatten-scope",
        rotation_flatten_scope,
    ]
    if view_dir:
        extra_args.extend(["--view-dir", view_dir])
    if view_up:
        extra_args.extend(["--view-up", view_up])
    if view_roll != 0.0:
        extra_args.extend(["--view-roll", str(view_roll)])
    if source_frame is not None:
        extra_args.extend(["--source-frame", str(source_frame)])
    if animation_names:
        for name in animation_names:
            extra_args.extend(["--animation", name])
    if frame_start is not None:
        extra_args.extend(["--frame-start", str(frame_start)])
    if frame_end is not None:
        extra_args.extend(["--frame-end", str(frame_end)])
    if not optimize_animation_keys:
        extra_args.append("--no-optimize-animation-keys")
    if force_loop_closing_keys:
        extra_args.append("--force-loop-closing-keys")
    if stretch_guard_enabled:
        extra_args.append("--stretch-guard-enabled")
        extra_args.extend(["--stretch-guard-max-scale", str(stretch_guard_max_scale)])
        extra_args.extend(["--stretch-guard-strength", str(stretch_guard_strength)])
    if ik_leaf_refine_enabled:
        extra_args.append("--ik-leaf-refine-enabled")
        extra_args.extend(["--ik-leaf-strength", str(ik_leaf_strength)])
        extra_args.extend(["--ik-leaf-iterations", str(ik_leaf_iterations)])
        extra_args.extend(["--ik-leaf-max-chain-length", str(ik_leaf_max_chain_length)])
        extra_args.extend(["--ik-leaf-preserve-scale", str(ik_leaf_preserve_scale)])
    if drop_problematic_frames:
        extra_args.append("--drop-problematic-frames")
    if preserve_root_motion:
        extra_args.append("--preserve-root-motion")
    if preserve_root_rotation:
        extra_args.append("--preserve-root-rotation")

    probe = probe_scene_backend_impl()
    if probe.mode == "bpy_module" and probe.available:
        return _run_bpy_command_with_args("extract-animations", source, output, extra_args)
    return _run_blender_command_with_args("extract-animations", source, output, extra_args)


def render_sprites(
    source: str,
    output: str,
    parts_json: str,
    images_dir: str,
    view_preset: str = "side",
    view_dir: str = None,
    view_up: str = None,
    view_roll: float = 0.0,
    source_frame: int = None,
    projection_space: str = "world",
    resolution: int = 2048,
    bind_frame: int = 0,
) -> SceneCommandResult:
    """Render sprites using projection."""
    extra_args = [
        "--view-preset",
        view_preset,
        "--projection-space",
        projection_space,
        "--parts-json",
        parts_json,
        "--images-dir",
        images_dir,
        "--resolution",
        str(resolution),
    ]
    if view_dir:
        extra_args.extend(["--view-dir", view_dir])
    if view_up:
        extra_args.extend(["--view-up", view_up])
    if view_roll != 0.0:
        extra_args.extend(["--view-roll", str(view_roll)])
    if source_frame is not None:
        extra_args.extend(["--source-frame", str(source_frame)])
    if bind_frame > 0:
        extra_args.extend(["--bind-frame", str(bind_frame)])

    probe = probe_scene_backend_impl()
    if probe.mode == "bpy_module" and probe.available:
        return _run_bpy_command_with_args("render-sprites", source, output, extra_args)
    return _run_blender_command_with_args("render-sprites", source, output, extra_args)


def probe_scene_backend() -> dict[str, Any]:
    return asdict(probe_scene_backend_impl())
