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


def _run_bpy_command(
    command: str, source: str, output: str, target_format: str | None = None
) -> SceneCommandResult:
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
            payload = worker.convert_source(source_path, str(output_path), target_format or "glb")
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


def _run_blender_command(
    command: str, source: str, output: str, target_format: str | None = None
) -> SceneCommandResult:
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
    if target_format:
        argv.extend(["--target-format", target_format])

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


def convert_3d_source(source: str, output: str, target_format: str = "glb") -> SceneCommandResult:
    probe = probe_scene_backend_impl()
    if probe.mode == "bpy_module" and probe.available:
        return _run_bpy_command("convert", source, output, target_format=target_format)
    return _run_blender_command("convert", source, output, target_format=target_format)


def probe_scene_backend() -> dict[str, Any]:
    return asdict(probe_scene_backend_impl())
