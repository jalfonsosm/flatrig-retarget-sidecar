"""Motion2Motion integration helpers for cross-rig retargeting."""

from __future__ import annotations

import contextlib
import copy
import functools
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from flatrig_retarget_sidecar.spine_import import (
    ROOT_DIR,
    SpinePackage,
    build_spine_package,
)
from flatrig_retarget_sidecar.spine_motion2motion_bridge import (
    ExportedSpineBvh,
    build_exported_motion2motion_mapping,
    export_spine_animation_to_bvh,
)

M2M_LABEL = "Motion2Motion"
M2M_DIR = (ROOT_DIR / "workflow" / "external" / "Motion2Motion_codes").resolve()
SHARED_VENV_DIR = (ROOT_DIR / ".venv").resolve()
SHARED_VENV_PYTHON = (
    SHARED_VENV_DIR / "Scripts" / "python.exe"
    if sys.platform.startswith("win")
    else SHARED_VENV_DIR / "bin" / "python"
)
M2M_RUNNER = "run_M2M.py"
M2M_DEFAULT_CONFIG = "configs/default.yaml"
ENV_M2M_DEVICE = "FLATRIG_M2M_DEVICE"
ENV_M2M_ALLOW_MPS = "FLATRIG_M2M_ALLOW_MPS"
ENV_M2M_MATCHING_ALPHA = "FLATRIG_M2M_MATCHING_ALPHA"
M2M_FALLBACK_TARGET_ANIMATIONS = ("idle", "walk", "run", "animation")
NUMERIC_EPSILON = 1e-8
M2M_SUFFIX_RE = re.compile(r"__[^\\s]{3}$")
TOKEN_SPLIT_RE = re.compile(r"(?<!^)(?=[A-Z])|[^A-Za-z0-9]+")
LEFT_TOKENS = {"l", "lf", "left", "lhs"}
RIGHT_TOKENS = {"r", "rt", "right", "rhs"}
STOP_TOKENS = {
    "bone",
    "joint",
    "ctrl",
    "control",
    "deform",
    "def",
    "rig",
    "slot",
    "mesh",
    "attachment",
    "node",
}


@dataclass(slots=True)
class Motion2MotionProbe:
    available: bool
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Motion2MotionBvhRetargetResult:
    output_bvh: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Motion2MotionSpineRetargetResult:
    animation_name: str
    animation: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenericSkeletonNode:
    name: str
    bvh_name: str
    parent_name: str | None
    parent_index: int
    depth: int
    child_count: int
    descendant_count: int
    world_x: float
    world_y: float
    end_x: float
    end_y: float
    length: float


@dataclass(slots=True)
class GenericSkeletonDescription:
    label: str
    root_name: str
    root_bvh_name: str
    names: list[str]
    bvh_names: list[str]
    parent_indices: list[int]
    nodes: list[GenericSkeletonNode]


@dataclass(slots=True)
class GenericSparseChain:
    names: list[str]
    start_name: str
    end_name: str
    parent_name: str | None
    side: str
    is_main: bool
    attachment_index: int
    start_depth: int
    total_length: float
    span_length: float
    straightness: float
    centroid_x: float
    centroid_y: float
    direction_x: float
    direction_y: float
    name_tokens: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NormalizedBvhSource:
    source_label: str
    normalized_bvh: str
    skeleton: GenericSkeletonDescription
    original_names: list[str]
    normalized_names: list[str]
    matching_names: list[str]
    original_to_matching: dict[str, str]
    matching_to_bvh: dict[str, str]


def _probe_torch_runtime(python_executable: Path, cwd: Path) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [
                str(python_executable),
                "-c",
                (
                    "import json, torch; "
                    "payload = {"
                    "'torch': torch.__version__, "
                    "'cuda': bool(getattr(torch, 'cuda', None) and torch.cuda.is_available()), "
                    "'mps': bool(("
                    "getattr(getattr(torch, 'backends', None), 'mps', None) "
                    "and torch.backends.mps.is_available()"
                    ") or (getattr(torch, 'mps', None) and torch.mps.is_available()))"
                    "}; "
                    "print(json.dumps(payload))"
                ),
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None

    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _motion2motion_can_use_mps() -> bool:
    raw = str(os.environ.get(ENV_M2M_ALLOW_MPS) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_motion2motion_device(python_executable: Path | None = None) -> str:
    raw = str(os.environ.get(ENV_M2M_DEVICE) or "").strip()
    if raw:
        return raw

    resolved_python = python_executable or SHARED_VENV_PYTHON
    if python_executable is not None or resolved_python.exists():
        runtime = _probe_torch_runtime(resolved_python, M2M_DIR if M2M_DIR.exists() else ROOT_DIR)
        if runtime:
            if runtime.get("cuda"):
                return "cuda"
            if runtime.get("mps"):
                # Upstream Motion2Motion currently trips CPU/GPU tensor mismatches on Apple MPS.
                # Keep MPS opt-in until the backend is patched upstream or locally.
                if _motion2motion_can_use_mps():
                    return "mps"
                return "cpu"

    if shutil.which("nvidia-smi"):
        return "cuda"
    return "cpu"


def resolve_matching_alpha(default: float = 0.9) -> float:
    raw = str(os.environ.get(ENV_M2M_MATCHING_ALPHA) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def probe_motion2motion_backend() -> Motion2MotionProbe:
    m2m_dir = M2M_DIR
    if not m2m_dir.exists():
        return Motion2MotionProbe(
            available=False,
            detail=(
                f"{M2M_LABEL} checkout not found at {m2m_dir}. "
                "Run tools/install_motion2motion_backend.py."
            ),
            metadata={"m2m_dir": str(m2m_dir)},
        )

    runner_path = m2m_dir / M2M_RUNNER
    config_path = m2m_dir / M2M_DEFAULT_CONFIG
    if not runner_path.exists() or not config_path.exists():
        missing = [
            str(path.relative_to(m2m_dir))
            for path in (runner_path, config_path)
            if not path.exists()
        ]
        return Motion2MotionProbe(
            available=False,
            detail=(f"{M2M_LABEL} checkout is incomplete. Missing: {', '.join(missing)}"),
            metadata={"m2m_dir": str(m2m_dir)},
        )

    python_executable = SHARED_VENV_PYTHON
    if not python_executable.exists():
        return Motion2MotionProbe(
            available=False,
            detail=(
                f"The shared Python environment was not found at {python_executable}. "
                "Run tools/install_motion2motion_backend.py --install-deps."
            ),
            metadata={
                "m2m_dir": str(m2m_dir),
                "venv_dir": str(SHARED_VENV_DIR),
                "python": str(python_executable),
            },
        )

    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import imageio, torch, unfoldNd, yaml; print(torch.__version__)",
        ],
        cwd=m2m_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    runtime = _probe_torch_runtime(python_executable, m2m_dir)
    if runtime is None or completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        return Motion2MotionProbe(
            available=False,
            detail=(
                f"{M2M_LABEL} checkout was found, but runtime dependencies are missing. "
                f"Probe error: {stderr}"
            ),
            metadata={
                "m2m_dir": str(m2m_dir),
                "venv_dir": str(SHARED_VENV_DIR),
                "python": str(python_executable),
            },
        )

    torch_version = str(runtime.get("torch") or "unknown")
    cuda_available = bool(runtime.get("cuda"))
    mps_available = bool(runtime.get("mps"))
    return Motion2MotionProbe(
        available=True,
        detail=(
            f"{M2M_LABEL} checkout and runtime dependencies were detected. "
            "The backend is ready for BVH sparse-correspondence retargeting."
        ),
        metadata={
            "m2m_dir": str(m2m_dir),
            "venv_dir": str(SHARED_VENV_DIR),
            "python": str(python_executable),
            "torch": torch_version,
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            "device": resolve_motion2motion_device(python_executable),
            "mps_auto_disabled": mps_available and not _motion2motion_can_use_mps(),
            "device_policy": (
                "auto prefers CUDA, then CPU. Apple MPS stays opt-in for Motion2Motion because "
                "the upstream runtime still hits CPU/GPU tensor mismatches on MPS."
            ),
        },
    )


def inspect_bvh_source(source_bvh: str | Path) -> dict[str, Any]:
    animation = _load_bvh_animation(source_bvh)
    frame_count = int(np.asarray(animation.positions).shape[0])
    frame_time = float(animation.frametime)
    duration = max(frame_time, frame_count * frame_time)
    names = [str(name) for name in animation.names]
    root_name = _strip_motion2motion_suffix(names[0]) if names else "root"
    return {
        "frame_count": frame_count,
        "frame_time": frame_time,
        "duration": duration,
        "joint_count": len(names),
        "root_name": root_name,
        "available_animations": [Path(str(source_bvh)).stem or "animation"],
    }


def retarget_bvh_pair(
    source_bvh: str | Path,
    target_bvh: str | Path,
    mapping_file: str | Path,
    *,
    output_bvh: str | Path | None = None,
    device: str | None = None,
    matching_alpha: float | None = None,
) -> Motion2MotionBvhRetargetResult:
    probe = probe_motion2motion_backend()
    if not probe.available:
        raise RuntimeError(f"{M2M_LABEL} backend unavailable: {probe.detail}")

    m2m_dir = M2M_DIR
    python_executable = SHARED_VENV_PYTHON
    if not python_executable.exists():
        raise RuntimeError(
            f"{M2M_LABEL} backend probe passed, but the shared Python executable is missing "
            f"at {python_executable}."
        )

    source_path = Path(source_bvh).expanduser().resolve()
    target_path = Path(target_bvh).expanduser().resolve()
    mapping_path = Path(mapping_file).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source BVH: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target BVH: {target_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing sparse mapping file: {mapping_path}")

    resolved_device = str(device or resolve_motion2motion_device()).strip() or "cpu"
    resolved_matching_alpha = float(
        matching_alpha if matching_alpha is not None else resolve_matching_alpha()
    )

    with tempfile.TemporaryDirectory(
        prefix="m2m_run_", dir=str(ROOT_DIR / "workflow")
    ) as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        command = [
            str(python_executable),
            str((m2m_dir / M2M_RUNNER).resolve()),
            "-e",
            str(target_path),
            "-d",
            resolved_device,
            "--source",
            str(source_path),
            "--mapping_file",
            str(mapping_path),
            "--output_dir",
            str(temp_dir),
            "--sparse_retargeting",
            "--matching_alpha",
            str(resolved_matching_alpha),
        ]
        completed = subprocess.run(
            command,
            cwd=m2m_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"{M2M_LABEL} retarget failed: {stderr}")

        outputs = sorted(temp_dir.rglob("*_syn.bvh"), key=lambda path: path.stat().st_mtime)
        if not outputs:
            raise RuntimeError(
                f"{M2M_LABEL} finished without producing the expected BVH output in {temp_dir}."
            )
        synthesized_bvh = outputs[-1]
        if output_bvh is None:
            final_output_path = target_path.with_name(target_path.stem + "_m2m_retarget.bvh")
        else:
            final_output_path = Path(output_bvh).expanduser().resolve()
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(synthesized_bvh, final_output_path)

        return Motion2MotionBvhRetargetResult(
            output_bvh=str(final_output_path),
            diagnostics={
                "backend_label": M2M_LABEL,
                "source_bvh": str(source_path),
                "target_bvh": str(target_path),
                "mapping_file": str(mapping_path),
                "output_bvh": str(final_output_path),
                "device": resolved_device,
                "matching_alpha": resolved_matching_alpha,
                "runner": str((m2m_dir / M2M_RUNNER).resolve()),
                "venv_dir": str(SHARED_VENV_DIR),
                "python": str(python_executable),
                "stdout_tail": _tail_lines(completed.stdout, 40),
            },
        )


def retarget_spine_animation(
    source: SpinePackage,
    target: SpinePackage,
    animation_name: str,
    *,
    target_animation_name: str | None = None,
    matching_alpha: float | None = None,
    mapping_file: str | Path | None = None,
) -> Motion2MotionSpineRetargetResult:
    if animation_name not in source.animations:
        available = ", ".join(sorted(source.animations)) or "<none>"
        raise ValueError(
            f"Missing source animation '{animation_name}' in {source.source_label}. "
            f"Available animations: {available}"
        )

    source_duration = _infer_source_animation_duration(source, animation_name)
    target_package, resolved_target_animation_name, synthesized_target_rest = (
        _resolve_target_package_with_exemplar(
            target,
            preferred_animation_name=target_animation_name,
            source_duration=source_duration,
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="m2m_spine_", dir=str(ROOT_DIR / "workflow")
    ) as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        source_bvh_path = temp_dir / "source.bvh"
        source_meta_path = temp_dir / "source.meta.json"
        target_bvh_path = temp_dir / "target.bvh"
        target_meta_path = temp_dir / "target.meta.json"
        mapping_path = temp_dir / "mapping.json"
        retargeted_bvh_path = temp_dir / "retargeted.bvh"

        source_metadata = export_spine_animation_to_bvh(
            source,
            animation_name,
            source_bvh_path,
            metadata_path=source_meta_path,
            positions_mode="all",
        )
        target_metadata = export_spine_animation_to_bvh(
            target_package,
            resolved_target_animation_name,
            target_bvh_path,
            metadata_path=target_meta_path,
            positions_mode="all",
        )
        mapping_payload = build_exported_motion2motion_mapping(
            source,
            target_package,
            mapping_file=mapping_file,
        )
        mapping_path.write_text(json.dumps(mapping_payload, indent=2) + "\n", encoding="utf-8")

        bvh_result = retarget_bvh_pair(
            source_bvh_path,
            target_bvh_path,
            mapping_path,
            output_bvh=retargeted_bvh_path,
            matching_alpha=matching_alpha,
        )
        clip = bvh_to_spine_animation(
            retargeted_bvh_path,
            target_package,
            target_metadata,
        )

        diagnostics = {
            "backend_label": M2M_LABEL,
            "source_animation_name": animation_name,
            "target_exemplar_animation_name": resolved_target_animation_name,
            "target_exemplar_synthesized": synthesized_target_rest,
            "source_source_label": source.source_label,
            "target_source_label": target.source_label,
            "source_non_root_translate_bones": source_metadata.motion2motion_non_root_translate_bones,
            "source_ignored_scale_bones": source_metadata.ignored_scale_bones,
            "target_non_root_translate_bones": target_metadata.motion2motion_non_root_translate_bones,
            "target_ignored_scale_bones": target_metadata.ignored_scale_bones,
            "source_frame_count": source_metadata.frame_count,
            "target_frame_count": target_metadata.frame_count,
            "mapping_pair_count": len(mapping_payload.get("mapping") or []),
            "mapping_root_joint": mapping_payload.get("root_joint"),
            "mapping_mode": "manual" if mapping_file else "auto",
            "mapping_file": str(Path(mapping_file).expanduser()) if mapping_file else None,
            "output_bvh": bvh_result.output_bvh,
            "bvh_result": dict(bvh_result.diagnostics),
            "result_bone_count": len(clip.get("bones") or {}),
        }
        return Motion2MotionSpineRetargetResult(
            animation_name=animation_name,
            animation=clip,
            diagnostics=diagnostics,
        )


def retarget_bvh_to_spine_animation(
    source_bvh: str | Path,
    target: SpinePackage,
    *,
    animation_name: str | None = None,
    target_animation_name: str | None = None,
    matching_alpha: float | None = None,
    mapping_file: str | Path | None = None,
) -> Motion2MotionSpineRetargetResult:
    source_path = Path(source_bvh).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source BVH: {source_path}")

    source_inspection = inspect_bvh_source(source_path)
    source_duration = float(source_inspection.get("duration") or (1.0 / 30.0))
    resolved_animation_name = (
        str(animation_name or "").strip()
        or source_inspection.get("available_animations", [source_path.stem])[0]
        or source_path.stem
    )
    target_package, resolved_target_animation_name, synthesized_target_rest = (
        _resolve_target_package_with_exemplar(
            target,
            preferred_animation_name=target_animation_name,
            source_duration=source_duration,
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="m2m_bvh_spine_", dir=str(ROOT_DIR / "workflow")
    ) as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        normalized_source = normalize_bvh_for_motion2motion(
            source_path,
            temp_dir / "source_normalized.bvh",
        )
        target_bvh_path = temp_dir / "target.bvh"
        target_meta_path = temp_dir / "target.meta.json"
        retargeted_bvh_path = temp_dir / "retargeted.bvh"
        mapping_path = temp_dir / "mapping.json"

        target_metadata = export_spine_animation_to_bvh(
            target_package,
            resolved_target_animation_name,
            target_bvh_path,
            metadata_path=target_meta_path,
            positions_mode="all",
        )
        target_skeleton = build_generic_skeleton_from_exported_spine_bvh(target_metadata)
        if mapping_file:
            raw_mapping = json.loads(Path(mapping_file).expanduser().read_text(encoding="utf-8"))
            mapping_payload = _normalize_generic_user_mapping_payload(
                raw_mapping,
                normalized_source.skeleton,
                target_skeleton,
                target_metadata.root_matching_name,
            )
            mapping_diagnostics = {
                "manual": True,
                "mapping_pair_count": len(mapping_payload.get("mapping") or []),
                "source": str(Path(mapping_file).expanduser()),
            }
        else:
            mapping_payload, mapping_diagnostics = build_auto_sparse_mapping_payload(
                normalized_source.skeleton,
                target_skeleton,
                root_joint=target_metadata.root_matching_name,
            )
        mapping_path.write_text(json.dumps(mapping_payload, indent=2) + "\n", encoding="utf-8")

        bvh_result = retarget_bvh_pair(
            normalized_source.normalized_bvh,
            target_bvh_path,
            mapping_path,
            output_bvh=retargeted_bvh_path,
            matching_alpha=matching_alpha,
        )
        clip = bvh_to_spine_animation(
            retargeted_bvh_path,
            target_package,
            target_metadata,
        )

        diagnostics = {
            "backend_label": M2M_LABEL,
            "source_animation_name": resolved_animation_name,
            "target_exemplar_animation_name": resolved_target_animation_name,
            "target_exemplar_synthesized": synthesized_target_rest,
            "source_source_label": str(source_path),
            "target_source_label": target.source_label,
            "source_frame_count": int(source_inspection.get("frame_count") or 0),
            "target_frame_count": target_metadata.frame_count,
            "source_joint_count": int(source_inspection.get("joint_count") or 0),
            "mapping_pair_count": len(mapping_payload.get("mapping") or []),
            "mapping_root_joint": mapping_payload.get("root_joint"),
            "mapping_mode": "manual" if mapping_file else "auto",
            "mapping_file": str(Path(mapping_file).expanduser()) if mapping_file else None,
            "mapping_diagnostics": mapping_diagnostics,
            "output_bvh": bvh_result.output_bvh,
            "bvh_result": dict(bvh_result.diagnostics),
            "result_bone_count": len((clip.get("bones") or {})),
        }
        return Motion2MotionSpineRetargetResult(
            animation_name=resolved_animation_name,
            animation=clip,
            diagnostics=diagnostics,
        )


def bvh_to_spine_animation(
    bvh_path: str | Path,
    target_package: SpinePackage,
    target_metadata: ExportedSpineBvh,
) -> dict[str, Any]:
    animation = _load_bvh_animation(bvh_path)
    joint_metadata_by_bvh_name = {joint.bvh_name: joint for joint in target_metadata.joints}
    target_bone_lookup = target_package.bones_by_name
    frame_count = int(animation.rotations.shape[0])
    frametime = float(animation.frametime)
    parents = list(animation.parents)
    names = [str(name) for name in animation.names]
    offsets = np.asarray(animation.offsets, dtype=np.float64)
    positions = np.asarray(animation.positions, dtype=np.float64)
    rotations = np.asarray(animation.rotations, dtype=np.float64)

    track_map: dict[str, dict[str, list[dict[str, float]]]] = {}
    previous_rotation_values: dict[str, float] = {}

    for frame_index in range(frame_count):
        time_value = round(frame_index * frametime, 4)
        world_cache: list[dict[str, Any] | None] = [None] * len(names)
        local_cache_2d: dict[str, dict[str, float]] = {}

        for joint_index, bvh_name in enumerate(names):
            joint_metadata = joint_metadata_by_bvh_name.get(bvh_name)
            if joint_metadata is None:
                continue

            rotation_deg = rotations[frame_index, joint_index]
            local_rotation_3d = _euler_xyz_degrees_to_matrix(rotation_deg)
            if parents[joint_index] < 0:
                head_3d = np.asarray(positions[frame_index, joint_index], dtype=np.float64)
                world_rotation_3d = local_rotation_3d
            else:
                parent_state = world_cache[parents[joint_index]]
                if parent_state is None:
                    continue
                head_3d = (
                    parent_state["head_3d"]
                    + parent_state["world_rotation_3d"] @ offsets[joint_index]
                )
                world_rotation_3d = parent_state["world_rotation_3d"] @ local_rotation_3d

            world_head_2d = np.asarray(head_3d[:2], dtype=np.float64)
            projected_x_axis_2d = np.asarray(world_rotation_3d[:2, 0], dtype=np.float64)
            axis_norm = float(np.linalg.norm(projected_x_axis_2d))
            spine_name = joint_metadata.spine_name

            if spine_name and spine_name in target_bone_lookup:
                setup_bone = target_bone_lookup[spine_name]
                fallback_axis = np.asarray(setup_bone.world_basis[0], dtype=np.float64)
            else:
                fallback_axis = np.array((1.0, 0.0), dtype=np.float64)
            if axis_norm <= NUMERIC_EPSILON:
                projected_x_axis_2d = fallback_axis
                axis_norm = float(np.linalg.norm(projected_x_axis_2d))
            if axis_norm <= NUMERIC_EPSILON:
                projected_x_axis_2d = np.array((1.0, 0.0), dtype=np.float64)
            else:
                projected_x_axis_2d = projected_x_axis_2d / axis_norm

            if parents[joint_index] < 0:
                local_x = float(world_head_2d[0])
                local_y = float(world_head_2d[1])
                local_rotation_deg = math.degrees(
                    math.atan2(projected_x_axis_2d[1], projected_x_axis_2d[0])
                )
                world_basis_2d = _build_basis_2d(local_rotation_deg)
            else:
                parent_state = world_cache[parents[joint_index]]
                assert parent_state is not None
                local_position = _safe_inverse_2x2(parent_state["world_basis_2d"]) @ (
                    world_head_2d - parent_state["head_2d"]
                )
                local_axis = _safe_inverse_2x2(parent_state["world_basis_2d"]) @ projected_x_axis_2d
                local_rotation_deg = math.degrees(math.atan2(local_axis[1], local_axis[0]))
                local_x = float(local_position[0])
                local_y = float(local_position[1])
                world_basis_2d = parent_state["world_basis_2d"] @ _build_basis_2d(
                    local_rotation_deg
                )

            world_cache[joint_index] = {
                "head_3d": head_3d,
                "head_2d": world_head_2d,
                "world_rotation_3d": world_rotation_3d,
                "world_basis_2d": _orthonormalize_2x2(world_basis_2d),
                "spine_name": spine_name,
            }
            if not spine_name or spine_name not in target_bone_lookup:
                continue
            local_cache_2d[spine_name] = {
                "x": local_x,
                "y": local_y,
                "rotation": _normalize_angle(local_rotation_deg),
            }

        for spine_name, pose in local_cache_2d.items():
            setup_bone = target_bone_lookup[spine_name]
            raw_rotation = float(pose["rotation"]) - float(setup_bone.rotation)
            rel_rotation = _unwrap_angle_near(
                _normalize_angle(raw_rotation),
                previous_rotation_values.get(spine_name),
            )
            previous_rotation_values[spine_name] = rel_rotation
            rel_x = float(pose["x"]) - float(setup_bone.x)
            rel_y = float(pose["y"]) - float(setup_bone.y)

            timelines = track_map.setdefault(
                spine_name,
                {"rotate": [], "translate": [], "scale": []},
            )
            timelines["rotate"].append(
                {
                    "time": time_value,
                    "angle": round(rel_rotation, 2),
                    "value": round(rel_rotation, 2),
                }
            )
            if abs(rel_x) > 1e-4 or abs(rel_y) > 1e-4 or setup_bone.parent is None:
                timelines["translate"].append(
                    {
                        "time": time_value,
                        "x": round(rel_x, 4),
                        "y": round(rel_y, 4),
                    }
                )

    compressed_bones: dict[str, Any] = {}
    for spine_name, timelines in track_map.items():
        rotate = _compress_timeline_keys(timelines["rotate"], ("angle", "value"), tolerance=0.01)
        translate = _compress_timeline_keys(timelines["translate"], ("x", "y"), tolerance=1e-4)
        payload: dict[str, Any] = {}
        if rotate:
            payload["rotate"] = rotate
        if translate:
            payload["translate"] = translate
        if payload:
            compressed_bones[spine_name] = payload
    return {"bones": compressed_bones}


def _resolve_target_package_with_exemplar(
    target: SpinePackage,
    *,
    preferred_animation_name: str | None,
    source_duration: float,
) -> tuple[SpinePackage, str, bool]:
    if preferred_animation_name and preferred_animation_name in target.animations:
        return target, preferred_animation_name, False

    available_names = list(target.animations.keys())
    for candidate in M2M_FALLBACK_TARGET_ANIMATIONS:
        if candidate in target.animations:
            return target, candidate, False
    if available_names:
        return target, available_names[0], False

    target_root = _select_target_root_name(target)
    synthetic_name = "__sidecar_rest__"
    cloned_payload = copy.deepcopy(target.payload)
    cloned_payload.setdefault("animations", {})
    cloned_payload["animations"][synthetic_name] = {
        "bones": {
            target_root: {
                "translate": [
                    {"time": 0.0, "x": 0.0, "y": 0.0},
                    {"time": round(max(source_duration, 1.0 / 30.0), 4), "x": 0.0, "y": 0.0},
                ],
                "rotate": [
                    {"time": 0.0, "angle": 0.0, "value": 0.0},
                    {
                        "time": round(max(source_duration, 1.0 / 30.0), 4),
                        "angle": 0.0,
                        "value": 0.0,
                    },
                ],
            }
        }
    }
    synthetic_target = build_spine_package(cloned_payload, source_label=target.source_label)
    return synthetic_target, synthetic_name, True


def _select_target_root_name(target: SpinePackage) -> str:
    for bone in target.bones:
        if bone.parent is None:
            return bone.name
    if target.bones:
        return target.bones[0].name
    raise ValueError("Target Spine package does not contain any bones.")


def _infer_source_animation_duration(source: SpinePackage, animation_name: str) -> float:
    animation = source.animations.get(animation_name) or {}
    return max(1.0 / 30.0, _scan_animation_duration(animation))


def _scan_animation_duration(animation_payload: dict[str, Any]) -> float:
    max_time = 0.0
    stack = [animation_payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            time_value = current.get("time")
            if isinstance(time_value, (int, float)):
                max_time = max(max_time, float(time_value))
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return max_time


def _tail_lines(text: str | None, max_lines: int) -> list[str]:
    lines = [line.rstrip() for line in (text or "").splitlines() if line.rstrip()]
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _euler_xyz_degrees_to_matrix(values: np.ndarray) -> np.ndarray:
    x_rad, y_rad, z_rad = [math.radians(float(value)) for value in values]
    cx, cy, cz = math.cos(x_rad), math.cos(y_rad), math.cos(z_rad)
    sx, sy, sz = math.sin(x_rad), math.sin(y_rad), math.sin(z_rad)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rx @ ry @ rz


def _build_basis_2d(rotation_deg: float) -> np.ndarray:
    rotation_rad = math.radians(rotation_deg)
    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)
    return np.array(
        [
            [cos_r, -sin_r],
            [sin_r, cos_r],
        ],
        dtype=np.float64,
    )


def _safe_inverse_2x2(matrix: np.ndarray) -> np.ndarray:
    det = float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    if abs(det) <= NUMERIC_EPSILON:
        return np.eye(2, dtype=np.float64)
    return np.linalg.inv(matrix)


def _orthonormalize_2x2(matrix: np.ndarray) -> np.ndarray:
    x_axis = np.array((matrix[0, 0], matrix[1, 0]), dtype=np.float64)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= NUMERIC_EPSILON:
        x_axis = np.array((1.0, 0.0), dtype=np.float64)
    else:
        x_axis = x_axis / x_norm
    y_axis = np.array((-x_axis[1], x_axis[0]), dtype=np.float64)
    return np.array(
        [
            [x_axis[0], y_axis[0]],
            [x_axis[1], y_axis[1]],
        ],
        dtype=np.float64,
    )


def _normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def _unwrap_angle_near(value: float, reference: float | None) -> float:
    if reference is None:
        return float(value)
    candidate = float(value)
    while candidate - reference > 180.0:
        candidate -= 360.0
    while candidate - reference < -180.0:
        candidate += 360.0
    return candidate


def _compress_timeline_keys(
    keys: list[dict[str, float]],
    fields: tuple[str, ...],
    *,
    tolerance: float,
) -> list[dict[str, float]]:
    if len(keys) <= 2:
        return list(keys)

    result = [keys[0]]
    for index in range(1, len(keys) - 1):
        previous = result[-1]
        current = keys[index]
        next_key = keys[index + 1]
        if all(
            abs(float(current[field]) - float(previous[field])) <= tolerance
            and abs(float(next_key[field]) - float(current[field])) <= tolerance
            for field in fields
        ):
            continue
        result.append(current)
    result.append(keys[-1])
    return result


def _collect_generic_sparse_chains(
    skeleton: GenericSkeletonDescription,
) -> list[GenericSparseChain]:
    by_name = {node.name: node for node in skeleton.nodes}
    children: dict[str, list[str]] = {node.name: [] for node in skeleton.nodes}
    for node in skeleton.nodes:
        if node.parent_name:
            children.setdefault(node.parent_name, []).append(node.name)
    main_path = _compute_generic_main_path(skeleton.root_name, by_name, children)
    main_path_set = set(main_path)
    chains: list[GenericSparseChain] = []

    if len(main_path) > 1:
        chains.append(
            _build_generic_chain(
                skeleton,
                names=main_path,
                root_name=skeleton.root_name,
                is_main=True,
                attachment_index=0,
            )
        )

    for index, bone_name in enumerate(main_path):
        next_main = main_path[index + 1] if index + 1 < len(main_path) else None
        for child_name in children.get(bone_name, []):
            if child_name == next_main:
                continue
            chain_names = _walk_generic_linear_chain(
                child_name,
                children,
                stop_names=main_path_set,
            )
            chains.append(
                _build_generic_chain(
                    skeleton,
                    names=chain_names,
                    root_name=skeleton.root_name,
                    is_main=False,
                    attachment_index=index,
                )
            )

    if not chains:
        chains.append(
            _build_generic_chain(
                skeleton,
                names=[skeleton.root_name],
                root_name=skeleton.root_name,
                is_main=True,
                attachment_index=0,
            )
        )
    return chains


def _compute_generic_main_path(
    root_name: str,
    by_name: dict[str, GenericSkeletonNode],
    children: dict[str, list[str]],
) -> list[str]:
    path = [root_name]
    current_name = root_name
    while children.get(current_name):
        ranked_children = sorted(
            children[current_name],
            key=lambda name: (
                int(by_name[name].descendant_count),
                int(by_name[name].child_count),
                float(by_name[name].length),
            ),
            reverse=True,
        )
        current_name = ranked_children[0]
        path.append(current_name)
    return path


def _walk_generic_linear_chain(
    start_name: str,
    children: dict[str, list[str]],
    *,
    stop_names: set[str],
) -> list[str]:
    chain = [start_name]
    current_name = start_name
    while True:
        current_children = children.get(current_name, [])
        if len(current_children) != 1:
            break
        next_name = current_children[0]
        if next_name in stop_names:
            break
        chain.append(next_name)
        current_name = next_name
    return chain


def _build_generic_chain(
    skeleton: GenericSkeletonDescription,
    *,
    names: list[str],
    root_name: str,
    is_main: bool,
    attachment_index: int,
) -> GenericSparseChain:
    by_name = {node.name: node for node in skeleton.nodes}
    root = by_name[root_name]
    start = by_name[names[0]]
    end = by_name[names[-1]]
    points = [(by_name[name].world_x, by_name[name].world_y) for name in names]
    centroid_x = sum(point[0] for point in points) / max(len(points), 1)
    centroid_y = sum(point[1] for point in points) / max(len(points), 1)
    total_length = sum(max(float(by_name[name].length), 1e-6) for name in names)
    start_point = (start.world_x, start.world_y)
    end_point = (end.end_x, end.end_y)
    span_length = _distance_2d(start_point, end_point)
    straightness = span_length / max(total_length, 1e-6)
    direction_x, direction_y = _normalize_vector_2d(
        end_point[0] - start_point[0],
        end_point[1] - start_point[1],
    )
    side = _infer_generic_chain_side(
        names, by_name=by_name, root_x=float(root.world_x), centroid_x=centroid_x
    )
    name_tokens: list[str] = []
    for name in names[:2] + names[-2:]:
        name_tokens.extend(_name_tokens(name))
    return GenericSparseChain(
        names=list(names),
        start_name=start.name,
        end_name=end.name,
        parent_name=start.parent_name,
        side=side,
        is_main=is_main,
        attachment_index=int(attachment_index),
        start_depth=int(start.depth),
        total_length=float(total_length),
        span_length=float(span_length),
        straightness=float(straightness),
        centroid_x=float(centroid_x),
        centroid_y=float(centroid_y),
        direction_x=float(direction_x),
        direction_y=float(direction_y),
        name_tokens=sorted(set(name_tokens)),
    )


def _greedy_match_generic_chain_pairs(
    source_chains: list[GenericSparseChain],
    target_chains: list[GenericSparseChain],
    *,
    mirror: bool,
    min_score: float,
) -> tuple[list[tuple[GenericSparseChain, GenericSparseChain, float]], float]:
    used_targets: set[int] = set()
    pairs: list[tuple[GenericSparseChain, GenericSparseChain, float]] = []
    total_score = 0.0
    for source_chain in source_chains:
        best_index = -1
        best_score = -1.0
        for index, target_chain in enumerate(target_chains):
            if index in used_targets:
                continue
            score = _score_generic_chain_pair(source_chain, target_chain, mirror=mirror)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0 or best_score < min_score:
            continue
        used_targets.add(best_index)
        pairs.append((source_chain, target_chains[best_index], best_score))
        total_score += best_score
    return pairs, total_score


def _score_generic_chain_pair(
    source: GenericSparseChain,
    target: GenericSparseChain,
    *,
    mirror: bool,
) -> float:
    if source.is_main != target.is_main:
        return 0.0
    target_side = target.side
    if mirror:
        if target_side == "left":
            target_side = "right"
        elif target_side == "right":
            target_side = "left"
    name_score = _token_similarity(source.name_tokens, target.name_tokens)
    length_score = _ratio_score(source.total_length, target.total_length)
    depth_score = 1.0 / (1.0 + abs(source.start_depth - target.start_depth))
    attachment_score = 1.0 / (1.0 + abs(source.attachment_index - target.attachment_index))
    straightness_score = 1.0 - min(abs(source.straightness - target.straightness), 1.0)
    side_score = _side_similarity(source.side, target_side)
    direction_score = 0.5 + 0.5 * abs(
        source.direction_x * target.direction_x + source.direction_y * target.direction_y
    )
    score = (
        0.18 * name_score
        + 0.26 * length_score
        + 0.14 * depth_score
        + 0.16 * attachment_score
        + 0.10 * straightness_score
        + 0.08 * side_score
        + 0.08 * direction_score
    )
    return float(max(0.0, min(1.0, score)))


def _dedupe_generic_pairs(
    pairs: list[tuple[str, str, float, str]],
    *,
    max_pairs: int,
) -> list[tuple[str, str, float, str]]:
    seen_source: set[str] = set()
    seen_target: set[str] = set()
    deduped: list[tuple[str, str, float, str]] = []
    for source_name, target_name, score, reason in sorted(
        pairs,
        key=lambda item: (-item[2], item[3], item[1], item[0]),
    ):
        if source_name in seen_source or target_name in seen_target:
            continue
        deduped.append((source_name, target_name, score, reason))
        seen_source.add(source_name)
        seen_target.add(target_name)
        if len(deduped) >= max(1, int(max_pairs)):
            break
    deduped.sort(key=lambda item: (0 if item[3] == "root" else 1, item[3], item[1], item[0]))
    return deduped


def _strip_motion2motion_suffix(name: str) -> str:
    stripped = M2M_SUFFIX_RE.sub("", str(name or "").strip())
    return stripped or "joint"


def _sanitize_matching_name(name: str, used_names: set[str]) -> str:
    base = "_".join(
        part for part in TOKEN_SPLIT_RE.split(_strip_motion2motion_suffix(name)) if part
    )
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_").lower() or "joint"
    candidate = base
    counter = 2
    while candidate in used_names:
        candidate = f"{base}_{counter}"
        counter += 1
    used_names.add(candidate)
    return candidate


def _infer_generic_chain_side(
    names: list[str],
    *,
    by_name: dict[str, GenericSkeletonNode],
    root_x: float,
    centroid_x: float,
) -> str:
    side_votes = [_infer_side_from_name(name) for name in names]
    side_votes = [side for side in side_votes if side != "unknown"]
    if side_votes:
        left_votes = sum(1 for side in side_votes if side == "left")
        right_votes = sum(1 for side in side_votes if side == "right")
        if left_votes > right_votes:
            return "left"
        if right_votes > left_votes:
            return "right"
    delta_x = centroid_x - root_x
    if delta_x <= -0.25:
        return "left"
    if delta_x >= 0.25:
        return "right"
    if any(by_name[name].depth <= 1 for name in names):
        return "center"
    return "unknown"


def _infer_side_from_name(name: str) -> str:
    tokens = _split_tokens(name)
    if any(token in LEFT_TOKENS for token in tokens):
        return "left"
    if any(token in RIGHT_TOKENS for token in tokens):
        return "right"
    return "unknown"


def _name_tokens(name: str) -> list[str]:
    tokens: list[str] = []
    for token in _split_tokens(name):
        if token in LEFT_TOKENS or token in RIGHT_TOKENS or token in STOP_TOKENS:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return tokens


def _split_tokens(name: str) -> list[str]:
    base = _strip_motion2motion_suffix(str(name or "").split(":")[-1])
    return [part.strip().lower() for part in TOKEN_SPLIT_RE.split(base) if part.strip()]


def _token_similarity(source_tokens: list[str], target_tokens: list[str]) -> float:
    source_set = set(source_tokens)
    target_set = set(target_tokens)
    if not source_set or not target_set:
        return 0.0
    overlap = len(source_set & target_set)
    union = len(source_set | target_set)
    return overlap / max(union, 1)


def _ratio_score(a: float, b: float) -> float:
    safe_a = max(float(a), 1e-6)
    safe_b = max(float(b), 1e-6)
    return math.exp(-abs(math.log(safe_a / safe_b)))


def _side_similarity(a: str, b: str) -> float:
    if a == "unknown" or b == "unknown":
        return 0.6
    if a == "center" and b == "center":
        return 1.0
    if a == "center" or b == "center":
        return 0.45
    return 1.0 if a == b else 0.1


def _normalize_vector_2d(x: float, y: float) -> tuple[float, float]:
    length = math.hypot(x, y)
    if length <= 1e-6:
        return 1.0, 0.0
    return x / length, y / length


def _distance_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def normalize_bvh_for_motion2motion(
    source_bvh: str | Path,
    output_bvh: str | Path,
) -> NormalizedBvhSource:
    source_path = Path(source_bvh).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Missing BVH source: {source_path}")
    animation = _load_bvh_animation(source_path)
    original_names = [str(name) for name in animation.names]
    used_names: set[str] = set()
    matching_names: list[str] = []
    normalized_names: list[str] = []
    original_to_matching: dict[str, str] = {}
    matching_to_bvh: dict[str, str] = {}

    for index, original_name in enumerate(original_names):
        matching_name = _sanitize_matching_name(
            _strip_motion2motion_suffix(original_name),
            used_names,
        )
        bvh_name = f"{matching_name}__{index:03d}"
        matching_names.append(matching_name)
        normalized_names.append(bvh_name)
        original_to_matching[original_name] = matching_name
        matching_to_bvh[matching_name] = bvh_name

    contents = source_path.read_text(encoding="utf-8", errors="replace")
    lines = contents.splitlines()
    normalized_lines: list[str] = []
    joint_cursor = 0
    joint_pattern = re.compile(r"^(\s*)(ROOT|JOINT)(\s+)(\S+)(\s*)$")
    for line in lines:
        match = joint_pattern.match(line)
        if match and joint_cursor < len(normalized_names):
            normalized_lines.append(
                f"{match.group(1)}{match.group(2)}{match.group(3)}"
                f"{normalized_names[joint_cursor]}{match.group(5)}"
            )
            joint_cursor += 1
            continue
        normalized_lines.append(line)
    if joint_cursor != len(normalized_names):
        raise RuntimeError(
            f"Failed to normalize BVH names for {source_path}: "
            f"rewrote {joint_cursor}/{len(normalized_names)} joints."
        )

    output_path = Path(output_bvh).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")

    skeleton = build_generic_skeleton_description(
        label=str(source_path),
        matching_names=matching_names,
        bvh_names=normalized_names,
        parent_indices=[int(parent_index) for parent_index in animation.parents],
        offsets=np.asarray(animation.offsets, dtype=np.float64),
    )
    return NormalizedBvhSource(
        source_label=str(source_path),
        normalized_bvh=str(output_path),
        skeleton=skeleton,
        original_names=original_names,
        normalized_names=normalized_names,
        matching_names=matching_names,
        original_to_matching=original_to_matching,
        matching_to_bvh=matching_to_bvh,
    )


def build_generic_skeleton_from_exported_spine_bvh(
    metadata: ExportedSpineBvh,
) -> GenericSkeletonDescription:
    return build_generic_skeleton_description(
        label=metadata.source_label,
        matching_names=[joint.matching_name for joint in metadata.joints],
        bvh_names=[joint.bvh_name for joint in metadata.joints],
        parent_indices=[int(joint.parent_index) for joint in metadata.joints],
        offsets=np.asarray(
            [[joint.offset_x, joint.offset_y, joint.offset_z] for joint in metadata.joints],
            dtype=np.float64,
        ),
    )


def _generic_matching_lookup(skeleton: GenericSkeletonDescription) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for node in skeleton.nodes:
        for key in (
            node.name,
            node.bvh_name,
            _strip_motion2motion_suffix(node.bvh_name),
        ):
            if key is not None and str(key):
                lookup[str(key)] = node.name
    return lookup


def _normalize_generic_user_mapping_payload(
    raw_payload: dict[str, Any],
    source_skeleton: GenericSkeletonDescription,
    target_skeleton: GenericSkeletonDescription,
    root_joint: str,
) -> dict[str, Any]:
    raw_pairs = raw_payload.get("mapping")
    if not isinstance(raw_pairs, list):
        raw_pairs = raw_payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("Mapping file must contain a 'mapping' or 'pairs' array.")

    source_lookup = _generic_matching_lookup(source_skeleton)
    target_lookup = _generic_matching_lookup(target_skeleton)
    requested_root = raw_payload.get("root_joint") or raw_payload.get("target_root") or root_joint

    pairs: list[dict[str, str]] = []
    used_source: set[str] = set()
    used_target: set[str] = set()

    def add_pair(source_name: str, target_name: str) -> None:
        if not source_name or not target_name:
            return
        if source_name in used_source or target_name in used_target:
            return
        pairs.append({"source": source_name, "target": target_name})
        used_source.add(source_name)
        used_target.add(target_name)

    add_pair(source_skeleton.root_name, target_lookup.get(str(requested_root), str(requested_root)))
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict):
            continue
        raw_source = (
            raw_pair.get("source")
            or raw_pair.get("source_joint")
            or raw_pair.get("from")
            or raw_pair.get("source_bone")
        )
        raw_target = (
            raw_pair.get("target")
            or raw_pair.get("target_joint")
            or raw_pair.get("to")
            or raw_pair.get("target_bone")
        )
        if raw_source is None or raw_target is None:
            continue
        add_pair(
            source_lookup.get(str(raw_source), str(raw_source)),
            target_lookup.get(str(raw_target), str(raw_target)),
        )

    if not pairs:
        raise ValueError("Mapping file did not contain any usable source/target pairs.")

    return {
        "source_name": str(raw_payload.get("source_name") or source_skeleton.label),
        "target_name": str(raw_payload.get("target_name") or target_skeleton.label),
        "root_joint": target_lookup.get(str(requested_root), str(requested_root)),
        "mapping": pairs,
    }


def build_auto_sparse_mapping_payload(
    source: GenericSkeletonDescription,
    target: GenericSkeletonDescription,
    *,
    root_joint: str,
    max_pairs: int = 12,
    min_chain_score: float = 0.38,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_chains = _collect_generic_sparse_chains(source)
    target_chains = _collect_generic_sparse_chains(target)
    main_source = next((chain for chain in source_chains if chain.is_main), None)
    main_target = next((chain for chain in target_chains if chain.is_main), None)
    branch_source = [chain for chain in source_chains if not chain.is_main]
    branch_target = [chain for chain in target_chains if not chain.is_main]

    direct_pairs, direct_score = _greedy_match_generic_chain_pairs(
        branch_source,
        branch_target,
        mirror=False,
        min_score=min_chain_score,
    )
    mirrored_pairs, mirrored_score = _greedy_match_generic_chain_pairs(
        branch_source,
        branch_target,
        mirror=True,
        min_score=min_chain_score,
    )
    use_mirror = mirrored_score > direct_score
    matched_chain_pairs = mirrored_pairs if use_mirror else direct_pairs

    pair_candidates: list[tuple[str, str, float, str]] = [
        (source.root_name, target.root_name, 1.0, "root")
    ]
    if (
        main_source
        and main_target
        and main_source.end_name != source.root_name
        and main_target.end_name != target.root_name
    ):
        pair_candidates.append(
            (
                main_source.end_name,
                main_target.end_name,
                _score_generic_chain_pair(main_source, main_target, mirror=use_mirror),
                "main_chain_leaf",
            )
        )
    for source_chain, target_chain, score in matched_chain_pairs:
        pair_candidates.append(
            (source_chain.start_name, target_chain.start_name, score, "chain_start")
        )
        if (
            source_chain.end_name != source_chain.start_name
            and target_chain.end_name != target_chain.start_name
        ):
            pair_candidates.append(
                (
                    source_chain.end_name,
                    target_chain.end_name,
                    max(0.0, score - 0.05),
                    "chain_end",
                )
            )

    selected_pairs = _dedupe_generic_pairs(pair_candidates, max_pairs=max_pairs)
    payload = {
        "source_name": Path(source.label).stem or "source",
        "target_name": Path(target.label).stem or "target",
        "root_joint": root_joint,
        "mapping": [
            {"source": source_name, "target": target_name}
            for source_name, target_name, _, _ in selected_pairs
        ],
    }
    diagnostics = {
        "mirror_x": use_mirror,
        "source_root": source.root_name,
        "target_root": target.root_name,
        "source_chain_count": len(source_chains),
        "target_chain_count": len(target_chains),
        "selected_pairs": [
            {
                "source": source_name,
                "target": target_name,
                "score": round(score, 4),
                "reason": reason,
            }
            for source_name, target_name, score, reason in selected_pairs
        ],
        "chain_pairs": [
            {
                "source_chain": source_chain.names,
                "target_chain": target_chain.names,
                "score": round(score, 4),
            }
            for source_chain, target_chain, score in matched_chain_pairs
        ],
    }
    return payload, diagnostics


def build_generic_skeleton_description(
    *,
    label: str,
    matching_names: list[str],
    bvh_names: list[str],
    parent_indices: list[int],
    offsets: np.ndarray,
) -> GenericSkeletonDescription:
    if not matching_names:
        raise ValueError("Skeleton description requires at least one joint.")
    if len(matching_names) != len(parent_indices) or len(matching_names) != len(bvh_names):
        raise ValueError("Skeleton description arrays must have the same length.")
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape[0] != len(matching_names) or offsets.shape[1] < 2:
        raise ValueError("Skeleton offsets must have shape [joint_count, >=2].")

    children: dict[int, list[int]] = {index: [] for index in range(len(matching_names))}
    roots: list[int] = []
    for index, parent_index in enumerate(parent_indices):
        if parent_index is None or int(parent_index) < 0:
            roots.append(index)
            continue
        children[int(parent_index)].append(index)

    if not roots:
        raise ValueError("Could not infer a root joint from the BVH hierarchy.")

    world_positions = np.zeros((len(matching_names), 2), dtype=np.float64)
    for index, parent_index in enumerate(parent_indices):
        if parent_index is None or int(parent_index) < 0:
            continue
        world_positions[index] = world_positions[int(parent_index)] + offsets[index, :2]

    descendant_counts = [0] * len(matching_names)
    depths = [0] * len(matching_names)

    def visit(index: int, depth: int) -> int:
        depths[index] = depth
        total = 0
        for child_index in children.get(index, []):
            total += 1 + visit(child_index, depth + 1)
        descendant_counts[index] = total
        return total

    for root_index in roots:
        visit(root_index, 0)

    root_index = max(
        roots,
        key=lambda index: (
            descendant_counts[index],
            len(children.get(index, [])),
        ),
    )
    nodes: list[GenericSkeletonNode] = []
    for index, matching_name in enumerate(matching_names):
        child_indices = children.get(index, [])
        if child_indices:
            primary_child = max(
                child_indices,
                key=lambda child_index: (
                    descendant_counts[child_index],
                    len(children.get(child_index, [])),
                    float(np.linalg.norm(offsets[child_index])),
                ),
            )
            length = max(
                float(np.linalg.norm(offsets[primary_child, :2])),
                float(np.linalg.norm(offsets[primary_child])),
                1e-6,
            )
            axis = _normalize_vector_2d(
                float(offsets[primary_child, 0]),
                float(offsets[primary_child, 1]),
            )
        else:
            length = max(
                float(np.linalg.norm(offsets[index, :2])),
                float(np.linalg.norm(offsets[index])),
                1e-6,
            )
            axis = _normalize_vector_2d(
                float(offsets[index, 0]),
                float(offsets[index, 1]),
            )
        nodes.append(
            GenericSkeletonNode(
                name=matching_name,
                bvh_name=bvh_names[index],
                parent_name=matching_names[parent_indices[index]]
                if int(parent_indices[index]) >= 0
                else None,
                parent_index=int(parent_indices[index]),
                depth=depths[index],
                child_count=len(child_indices),
                descendant_count=descendant_counts[index],
                world_x=float(world_positions[index, 0]),
                world_y=float(world_positions[index, 1]),
                end_x=float(world_positions[index, 0] + axis[0] * length),
                end_y=float(world_positions[index, 1] + axis[1] * length),
                length=float(length),
            )
        )

    return GenericSkeletonDescription(
        label=label,
        root_name=matching_names[root_index],
        root_bvh_name=bvh_names[root_index],
        names=list(matching_names),
        bvh_names=list(bvh_names),
        parent_indices=[int(parent_index) for parent_index in parent_indices],
        nodes=nodes,
    )


@contextlib.contextmanager
def _prepended_sys_path(path: Path) -> Iterator[None]:
    path_value = str(path)
    sys.path.insert(0, path_value)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(path_value)


@functools.lru_cache(maxsize=1)
def _load_m2m_bvh_io_module():
    m2m_dir = M2M_DIR
    if not m2m_dir.exists():
        raise RuntimeError(
            f"{M2M_LABEL} checkout not found at {m2m_dir}. "
            "Run tools/install_motion2motion_backend.py first."
        )
    with _prepended_sys_path(m2m_dir):
        return importlib.import_module("dataset.bvh.bvh_io")


def _load_bvh_animation(path: str | Path):
    module = _load_m2m_bvh_io_module()
    return module.load(str(Path(path).expanduser().resolve()))
