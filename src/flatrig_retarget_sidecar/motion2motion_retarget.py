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
    build_sample_times,
    evaluate_local_pose_map,
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
    force_mapping_review: bool = False,
    mapping_quality_threshold: float = 0.55,
) -> Motion2MotionSpineRetargetResult:
    if animation_name not in source.animations:
        available = ", ".join(sorted(source.animations)) or "<none>"
        raise ValueError(
            f"Missing source animation '{animation_name}' in {source.source_label}. "
            f"Available animations: {available}"
        )

    source_duration = _infer_source_animation_duration(source, animation_name)
    mapping_payload = build_exported_motion2motion_mapping(
        source,
        target,
        mapping_file=mapping_file,
    )

    print(f"[DEBUG_APPEND_MAPPING] retarget_spine_animation: mapping_file={'NONE' if mapping_file is None else str(mapping_file)} animation={animation_name} root_joint={mapping_payload.get('root_joint')}", flush=True)

    mapping_pair_count = len(mapping_payload.get("mapping") or [])
    source_bone_count = len([b for b in (source.bones or []) if isinstance(b, dict)])
    target_bone_count = len([b for b in (target.bones or []) if isinstance(b, dict)])
    expected_pairs = max(1, min(12, source_bone_count, target_bone_count))
    coverage = min(1.0, mapping_pair_count / expected_pairs) if not mapping_file else 1.0
    mapping_quality = coverage
    mapping_review_required = bool(
        force_mapping_review or mapping_quality < mapping_quality_threshold
    )
    if mapping_review_required:
        return Motion2MotionSpineRetargetResult(
            animation_name=animation_name,
            animation={},
            diagnostics={
                "backend_label": M2M_LABEL,
                "mapping_review_required": True,
                "mapping_payload": mapping_payload,
                "mapping_pairs": list(mapping_payload.get("mapping") or []),
                "mapping_pair_count": mapping_pair_count,
                "mapping_root_joint": mapping_payload.get("root_joint"),
                "mapping_quality_score": round(mapping_quality, 4),
                "mapping_quality_threshold": mapping_quality_threshold,
                "force_mapping_review": force_mapping_review,
                "mapping_mode": "manual" if mapping_file else "auto",
                "mapping_file": str(Path(mapping_file).expanduser()) if mapping_file else None,
                "mapping_metadata": mapping_payload.get("metadata") or {},
                "source_bone_count": source_bone_count,
                "target_bone_count": target_bone_count,
                "source_source_label": source.source_label,
                "target_source_label": target.source_label,
                "preflight_mapping_review": True,
            },
        )

    (
        target_package,
        resolved_target_animation_name,
        synthesized_target_rest,
        target_exemplar_mode,
    ) = _resolve_target_package_with_exemplar(
        target,
        source_animation_name=animation_name,
        preferred_animation_name=target_animation_name,
        source_duration=source_duration,
    )
    source_loop_closed = _source_animation_loop_closed(source, animation_name, source_duration)

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
            sample_duration=source_duration,
        )
        mapping_path.write_text(json.dumps(mapping_payload, indent=2) + "\n", encoding="utf-8")

        bvh_result = retarget_bvh_pair(
            source_bvh_path,
            target_bvh_path,
            mapping_path,
            output_bvh=retargeted_bvh_path,
            matching_alpha=matching_alpha,
        )
        # Build the set of TARGET spine bone names that the user actually paired
        # in the joint mapping. M2M synthesizes rotations for every joint of
        # the target rig, but the unmapped ones (e.g. fingers, toes when only
        # the main biped chain was paired) are noise — they don't correspond
        # to any source bone, so they jitter and bleed into sprite skinning.
        # Restricting emission to mapped bones keeps the rest at setup pose.
        mapped_target_bones = _resolve_mapped_target_spine_bones(
            mapping_payload, target_metadata
        )
        clip = bvh_to_spine_animation(
            retargeted_bvh_path,
            target_package,
            target_metadata,
            mapped_target_bones=mapped_target_bones,
        )
        if source_loop_closed:
            _force_spine_clip_loop_closure(clip, source_duration)

        # Surface rig family detection in diagnostics (informational only — the
        # 2D Spine path doesn't auto-bypass M2M because Spine bone naming is too
        # ambiguous for safe Mixamo detection; the 3D path in retarget_3d.py is
        # where the Mixamo fast-path actually fires).
        from flatrig_retarget_sidecar.rig_identity import detect_rig_family

        diagnostics = {
            "backend_label": M2M_LABEL,
            "source_animation_name": animation_name,
            "source_duration": source_duration,
            "source_loop_closed": source_loop_closed,
            "target_exemplar_animation_name": resolved_target_animation_name,
            "target_exemplar_synthesized": synthesized_target_rest,
            "target_exemplar_mode": target_exemplar_mode,
            "source_source_label": source.source_label,
            "target_source_label": target.source_label,
            "source_non_root_translate_bones": source_metadata.motion2motion_non_root_translate_bones,
            "source_ignored_scale_bones": source_metadata.ignored_scale_bones,
            "target_non_root_translate_bones": target_metadata.motion2motion_non_root_translate_bones,
            "target_ignored_scale_bones": target_metadata.ignored_scale_bones,
            "source_frame_count": source_metadata.frame_count,
            "target_frame_count": target_metadata.frame_count,
            "mapping_pair_count": len(mapping_payload.get("mapping") or []),
            "mapping_pairs": list(mapping_payload.get("mapping") or []),
            "mapping_root_joint": mapping_payload.get("root_joint"),
            "mapping_mode": "manual" if mapping_file else "auto",
            "mapping_file": str(Path(mapping_file).expanduser()) if mapping_file else None,
            "output_bvh": bvh_result.output_bvh,
            "bvh_result": dict(bvh_result.diagnostics),
            "result_bone_count": len(clip.get("bones") or {}),
            "rig_family_source": detect_rig_family(list(source.bones or [])),
            "rig_family_target": detect_rig_family(list(target.bones or [])),
            "bypass_reason": None,
        }
        return Motion2MotionSpineRetargetResult(
            animation_name=animation_name,
            animation=clip,
            diagnostics=diagnostics,
        )


def _spine_clip_has_motion(clip: dict[str, Any], tolerance: float = 1e-5) -> bool:
    for timelines in (clip.get("bones") or {}).values():
        if not isinstance(timelines, dict):
            continue
        for frames in timelines.values():
            if not isinstance(frames, list) or len(frames) < 2:
                continue
            first = _timeline_numeric_values(frames[0])
            for frame in frames[1:]:
                current = _timeline_numeric_values(frame)
                if len(current) != len(first):
                    return True
                if any(abs(a - b) > tolerance for a, b in zip(first, current)):
                    return True
    return False


def _spine_clip_has_pose_or_motion(clip: dict[str, Any], tolerance: float = 1e-5) -> bool:
    for timelines in (clip.get("bones") or {}).values():
        if not isinstance(timelines, dict):
            continue
        for frames in timelines.values():
            if not isinstance(frames, list) or not frames:
                continue
            for frame in frames:
                values = _timeline_numeric_values(frame)
                if any(abs(value) > tolerance for value in values):
                    return True
    return _spine_clip_has_motion(clip, tolerance=tolerance)


def _timeline_numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        values: list[float] = []
        for item in value:
            values.extend(_timeline_numeric_values(item))
        return values
    if isinstance(value, dict):
        values: list[float] = []
        for key, item in value.items():
            if str(key).lower() in {"time", "curve", "c", "c2", "c3", "c4"}:
                continue
            values.extend(_timeline_numeric_values(item))
        return values
    return []


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
    (
        target_package,
        resolved_target_animation_name,
        synthesized_target_rest,
        target_exemplar_mode,
    ) = (
        _resolve_target_package_with_exemplar(
            target,
            source_animation_name=resolved_animation_name,
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
            sample_duration=source_duration,
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
        mapped_target_bones = _resolve_mapped_target_spine_bones(
            mapping_payload, target_metadata
        )
        clip = bvh_to_spine_animation(
            retargeted_bvh_path,
            target_package,
            target_metadata,
            mapped_target_bones=mapped_target_bones,
        )

        diagnostics = {
            "backend_label": M2M_LABEL,
            "source_animation_name": resolved_animation_name,
            "target_exemplar_animation_name": resolved_target_animation_name,
            "target_exemplar_synthesized": synthesized_target_rest,
            "target_exemplar_mode": target_exemplar_mode,
            "source_source_label": str(source_path),
            "target_source_label": target.source_label,
            "source_frame_count": int(source_inspection.get("frame_count") or 0),
            "target_frame_count": target_metadata.frame_count,
            "source_joint_count": int(source_inspection.get("joint_count") or 0),
            "mapping_pair_count": len(mapping_payload.get("mapping") or []),
            "mapping_pairs": list(mapping_payload.get("mapping") or []),
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


def _resolve_mapped_target_spine_bones(
    mapping_payload: dict[str, Any],
    target_metadata: ExportedSpineBvh,
) -> set[str]:
    """Translate the joint-mapping editor's target names to spine bone names.

    The mapping payload stores target joints by their Motion2Motion-sanitized
    `matching_name` (e.g. "mixamorig_Hips" — colon stripped). Spine animations
    reference bones by their original name (e.g. "mixamorig:Hips"). This
    helper walks the target_metadata.joints catalog to convert one set into
    the other so the bvh→spine converter can filter on spine bone names.
    """
    matching_to_spine: dict[str, str] = {}
    for joint in target_metadata.joints:
        if joint.spine_name and joint.matching_name:
            matching_to_spine[str(joint.matching_name)] = str(joint.spine_name)

    mapped: set[str] = set()
    for pair in mapping_payload.get("mapping") or []:
        target_name = (
            pair.get("target")
            if isinstance(pair, dict)
            else (pair[1] if isinstance(pair, (list, tuple)) and len(pair) > 1 else None)
        )
        if not target_name:
            continue
        target_name = str(target_name)
        spine_name = matching_to_spine.get(target_name)
        if spine_name is None:
            # Some mapping files already store spine names directly (e.g. when
            # the user authored them by hand); accept those too.
            spine_name = target_name
        mapped.add(spine_name)

    # The root joint (if any) should also be considered mapped — it's the
    # implicit pinning point of the retarget.
    root_matching = mapping_payload.get("root_joint")
    if isinstance(root_matching, str):
        mapped.add(matching_to_spine.get(root_matching, root_matching))
    all_target_spine_names = set(str(joint.spine_name) for joint in target_metadata.joints if joint.spine_name)
    excluded = all_target_spine_names - mapped
    print(f"[DEBUG_APPEND_MAPPING] _resolve_mapped_target_spine_bones: mapped_targets={sorted(mapped)} excluded_targets={sorted(excluded)} total_target_spine_bones={len(all_target_spine_names)}", flush=True)
    return mapped


def bvh_to_spine_animation(
    bvh_path: str | Path,
    target_package: SpinePackage,
    target_metadata: ExportedSpineBvh,
    *,
    mapped_target_bones: set[str] | None = None,
) -> dict[str, Any]:
    """Convert a retargeted BVH to a Spine animation clip.

    `mapped_target_bones` lists the target spine bone names that were
    explicitly paired in the joint mapping editor. Bones NOT in this set
    receive a "best guess" rotation from Motion2Motion (it has to fill in
    every joint of the target rig), but those guesses are noise — there's no
    source motion driving them. They surface as one-frame jitter on bones
    like fingers and toes, which then leaks into sprites that share weights
    with those bones (skinning blends parent + unmapped child motion). To
    keep the animation faithful to what the user actually mapped, we skip
    keyframe emission for unmapped bones — Spine then holds them at setup
    pose. Pass None to keep the legacy behaviour and emit every bone.
    """
    animation = _load_bvh_animation(bvh_path)
    joint_metadata_by_bvh_name = {joint.bvh_name: joint for joint in target_metadata.joints}
    target_bone_lookup = target_package.bones_by_name
    frame_count = int(animation.rotations.shape[0])
    frametime = float(animation.frametime)
    parents = list(animation.parents)
    names = [str(name) for name in animation.names]
    # Slot bones: ordered list of (slot_index, slot_name, bone_spine_name).
    # Used at the end of the loop to build a per-frame drawOrder timeline so
    # sprites swap layer when the rig's left/right limbs cross the camera
    # axis. Without this the rear sprite stays behind in the static slot
    # order and the animation looks like the crossing leg "bounces back" or
    # "doesn't touch the ground" — exactly the user-reported symptom.
    slot_bones: list[tuple[int, str, str]] = []
    for slot_index, slot in enumerate(target_package.slots or []):
        slot_name = getattr(slot, "name", None)
        bone_name = getattr(slot, "bone", None) or getattr(slot, "bone_name", None)
        if isinstance(slot_name, str) and isinstance(bone_name, str):
            slot_bones.append((slot_index, slot_name, bone_name))
    # Per frame we'll record the depth (camera-axis projection) of each slot's
    # bone. We compute the proper sign once from the setup pose so the
    # ordering convention adapts to whichever side the camera looks at.
    slot_depths_per_frame: list[dict[str, float]] = []
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
            # Skip bones that the user didn't explicitly pair in the mapping
            # editor. Their values would be Motion2Motion guesses driven by no
            # actual source motion — visible as jitter that bleeds into sprite
            # skinning. The bone stays at setup pose in Spine.
            if mapped_target_bones is not None and spine_name not in mapped_target_bones:
                continue
            local_cache_2d[spine_name] = {
                "x": local_x,
                "y": local_y,
                "rotation": _normalize_angle(local_rotation_deg),
            }

        # Snapshot the depth of every slot bone for this frame. Depth here is
        # the world X coordinate of the bone (the lateral axis in the rig's
        # world space, perpendicular to the side-view camera). The sign is
        # normalized later from the setup pose so we don't have to know the
        # camera orientation up front.
        per_slot_depth: dict[str, float] = {}
        for _slot_index, slot_name, bone_name in slot_bones:
            depth: float | None = None
            for joint_index, bvh_name in enumerate(names):
                cached = world_cache[joint_index]
                if cached is None or cached.get("spine_name") != bone_name:
                    continue
                head_3d = cached.get("head_3d")
                if head_3d is not None:
                    depth = float(head_3d[0])
                break
            if depth is not None:
                per_slot_depth[slot_name] = depth
        slot_depths_per_frame.append(per_slot_depth)

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

    draw_order_timeline = _build_dynamic_draw_order_timeline(
        slot_bones=slot_bones,
        slot_depths_per_frame=slot_depths_per_frame,
        target_package=target_package,
        frametime=frametime,
    )
    result: dict[str, Any] = {"bones": compressed_bones}
    if draw_order_timeline:
        result["drawOrder"] = draw_order_timeline
    return result


def _build_dynamic_draw_order_timeline(
    *,
    slot_bones: list[tuple[int, str, str]],
    slot_depths_per_frame: list[dict[str, float]],
    target_package: SpinePackage,
    frametime: float,
) -> list[dict[str, Any]]:
    """Build a per-frame drawOrder track that swaps slots when their bones
    cross each other along the camera-facing axis.

    Without this the appended Spine animation inherits the static slot order
    of the setup pose, so when the rear leg's bone crosses in front of the
    front leg's bone (or any pair of slots crosses), the rear sprite stays
    visually behind — the user perceives this as the crossing limb "bouncing
    back" or "not touching the ground". The convention sign is auto-detected
    from the setup pose so this works regardless of which side the camera
    looks at.
    """
    if not slot_bones or not slot_depths_per_frame:
        return []

    # Detect the camera-axis sign by counting which sign of X best agrees
    # with the setup slot order in the FIRST frame: the more pairs that come
    # out in the same order as the setup, the better the chosen sign. This
    # adapts to rigs viewed from either side without hardcoding orientation.
    if not slot_depths_per_frame:
        return []
    first_depths = slot_depths_per_frame[0]
    setup_order = [(idx, name) for idx, name, _ in slot_bones]

    def order_agreement(sign: float) -> int:
        # Count how many adjacent setup pairs are kept in order when sorted
        # by sign * depth.
        ordered = sorted(
            setup_order,
            key=lambda item: (
                sign * first_depths.get(item[1], float(item[0])),
                item[0],
            ),
        )
        ordered_indices = [idx for idx, _ in ordered]
        # Number of inversions vs. the identity (setup) order.
        agreements = 0
        for i, idx in enumerate(ordered_indices):
            if idx == i:
                agreements += 1
        return agreements

    sign = +1.0 if order_agreement(+1.0) >= order_agreement(-1.0) else -1.0

    timeline: list[dict[str, Any]] = []
    previous_offsets: dict[str, int] | None = None
    for frame_index, depths in enumerate(slot_depths_per_frame):
        # Order the slots by depth (back-to-front). Slots without depth fall
        # back to their setup index so they keep their original layer.
        def slot_depth(item: tuple[int, str]) -> tuple[float, int]:
            idx, name = item
            depth = depths.get(name)
            if depth is None:
                return (float(idx), idx)  # stable fallback by setup index
            return (sign * depth, idx)

        ordered = sorted(setup_order, key=slot_depth)
        # Compute offset = new_position - setup_position. Spine's drawOrder
        # only requires entries whose offset is non-zero.
        new_position_by_slot: dict[str, int] = {}
        for new_pos, (_orig_idx, name) in enumerate(ordered):
            new_position_by_slot[name] = new_pos
        offsets: dict[str, int] = {}
        for slot_index, slot_name in setup_order:
            new_pos = new_position_by_slot[slot_name]
            delta = new_pos - slot_index
            if delta != 0:
                offsets[slot_name] = delta

        # Skip emission when nothing changed since the previous keyframe; Spine
        # holds the previous drawOrder until the next key.
        if offsets == previous_offsets:
            continue
        previous_offsets = offsets

        offsets_payload = [
            {"slot": name, "offset": offset}
            for name, offset in sorted(offsets.items())
        ]
        timeline.append(
            {
                "time": round(frame_index * frametime, 4),
                "offsets": offsets_payload,
            }
        )

    # If the timeline never deviates from setup, don't emit it.
    if all(not entry.get("offsets") for entry in timeline):
        return []
    return timeline


def _resolve_target_package_with_exemplar(
    target: SpinePackage,
    *,
    source_animation_name: str | None = None,
    preferred_animation_name: str | None,
    source_duration: float,
) -> tuple[SpinePackage, str, bool, str]:
    if preferred_animation_name and preferred_animation_name in target.animations:
        return target, preferred_animation_name, False, "preferred"

    matched_name = _select_matching_target_animation(
        source_animation_name,
        list(target.animations.keys()),
    )
    if matched_name:
        return target, matched_name, False, "matched"

    target_root = _select_target_root_name(target)
    synthetic_name = "__sidecar_rest__"
    duration = round(max(source_duration, 1.0 / 30.0), 4)
    sample_times = build_sample_times(duration, 30.0)
    cloned_payload = copy.deepcopy(target.payload)
    cloned_payload.setdefault("animations", {})
    cloned_payload["animations"][synthetic_name] = {
        "bones": {
            target_root: {
                "translate": [
                    {"time": round(float(time_value), 4), "x": 0.0, "y": 0.0}
                    for time_value in sample_times
                ],
                "rotate": [
                    {
                        "time": round(float(time_value), 4),
                        "angle": 0.0,
                        "value": 0.0,
                    }
                    for time_value in sample_times
                ],
            }
        }
    }
    synthetic_target = build_spine_package(cloned_payload, source_label=target.source_label)
    return synthetic_target, synthetic_name, True, "synthetic"


def _select_matching_target_animation(
    source_animation_name: str | None,
    available_names: list[str],
) -> str | None:
    source_tokens = _animation_name_tokens(source_animation_name)
    if not source_tokens:
        return None

    best_name: str | None = None
    best_score = 0
    for candidate in available_names:
        candidate_tokens = _animation_name_tokens(candidate)
        if not candidate_tokens:
            continue
        overlap = source_tokens & candidate_tokens
        score = len(overlap) * 10
        source_key = "".join(sorted(source_tokens))
        candidate_key = "".join(sorted(candidate_tokens))
        if source_key and source_key in candidate_key:
            score += 3
        if candidate_key and candidate_key in source_key:
            score += 3
        if score > best_score:
            best_score = score
            best_name = candidate
    return best_name if best_score > 0 else None


def _animation_name_tokens(name: str | None) -> set[str]:
    raw_tokens = re.split(r"[^A-Za-z0-9]+", str(name or "").lower())
    return {
        token
        for token in raw_tokens
        if len(token) >= 2 and token not in {"armature", "mixamo", "com", "layer0"}
    }


def _select_target_root_name(target: SpinePackage) -> str:
    for bone in target.bones:
        if bone.parent is None:
            return bone.name
    if target.bones:
        return target.bones[0].name
    raise ValueError("Target Spine package does not contain any bones.")


def _infer_source_animation_duration(source: SpinePackage, animation_name: str) -> float:
    animation = source.animations.get(animation_name) or {}
    duration = _scan_animation_duration(animation)
    if duration <= NUMERIC_EPSILON:
        return 1.0
    return max(1.0 / 30.0, duration)


def _source_animation_loop_closed(
    source: SpinePackage,
    animation_name: str,
    duration: float,
    *,
    tolerance: float = 0.5,
) -> bool:
    animation = source.animations.get(animation_name)
    if not isinstance(animation, dict) or duration <= NUMERIC_EPSILON:
        return False
    if not _animation_has_timed_keys(animation):
        return False

    start_pose = evaluate_local_pose_map(source, animation_name, 0.0)
    end_pose = evaluate_local_pose_map(source, animation_name, duration)
    for bone in source.bones:
        start = start_pose.get(bone.name) or {}
        end = end_pose.get(bone.name) or {}
        if abs(_normalize_angle(float(end.get("rotation", 0.0)) - float(start.get("rotation", 0.0)))) > tolerance:
            return False
        for pose_field in ("x", "y"):
            if (
                abs(float(end.get(pose_field, 0.0)) - float(start.get(pose_field, 0.0)))
                > tolerance
            ):
                return False
    return True


def _animation_has_timed_keys(animation_payload: dict[str, Any]) -> bool:
    stack = [animation_payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            time_value = current.get("time")
            if isinstance(time_value, (int, float)) and float(time_value) > NUMERIC_EPSILON:
                return True
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return False


def _force_spine_clip_loop_closure(clip: dict[str, Any], duration: float) -> None:
    """Append a closing keyframe at `duration` mirroring the first keyframe.

    Previously this would also REPLACE an existing keyframe at end_time with
    the start values, which silently overwrote the last frame produced by the
    retarget — visible as a jump at the end of the loop because the retarget's
    last frame might disagree with the first frame by a non-trivial amount.
    Now we only ADD a closing keyframe when none is already present near
    end_time; the retarget's own last frame is always preserved. The loop may
    not be perfectly closed in that case (Spine will jump from the real last
    value back to the start at loop boundary), which is the lesser evil
    compared to discarding retargeted motion data.
    """
    if duration <= NUMERIC_EPSILON:
        return
    end_time = round(float(duration), 4)
    for timelines in (clip.get("bones") or {}).values():
        if not isinstance(timelines, dict):
            continue
        for timeline_name, keys in list(timelines.items()):
            if not isinstance(keys, list) or not keys:
                continue
            start_key = {
                key: copy.deepcopy(value)
                for key, value in (keys[0] or {}).items()
                if key not in {"time", "curve", "c", "c2", "c3", "c4"}
            }
            if not start_key:
                continue
            if abs(float((keys[-1] or {}).get("time", 0.0)) - end_time) <= 1e-4:
                # A keyframe already exists at the loop boundary — keep the
                # retargeted values intact even if they don't perfectly match
                # the start key.
                continue
            keys.append({"time": end_time, **start_key})


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
    # Blender's Euler(..., "XYZ").to_matrix() uses this composition for column vectors.
    # The BVH exporter writes Blender XYZ Euler angles, so import must mirror it exactly.
    return rz @ ry @ rx


def _build_basis_2d(rotation_deg: float, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    rotation_rad = math.radians(rotation_deg)
    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)
    return np.array(
        [
            [cos_r * scale_x, -sin_r * scale_y],
            [sin_r * scale_x, cos_r * scale_y],
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
    """Drop keyframes whose value is essentially the same as both real neighbors.

    The previous version compared against `result[-1]` (the last KEPT key) and
    `keys[index+1]` (the next RAW key). With slow drift between frames each
    `current` would still be within tolerance of the (unchanged) last-kept
    key, so the algorithm could drop arbitrarily many consecutive frames.
    Spine then interpolates linearly across those deleted frames and the
    animation loses motion detail in chunks — visually a "jump" of several
    frames. Comparing against the REAL immediate neighbors fixes that:
    a key only gets dropped when its true predecessor, itself and successor
    are nearly identical, i.e. when the local motion really is flat.
    """
    if len(keys) <= 2:
        return list(keys)

    result = [keys[0]]
    for index in range(1, len(keys) - 1):
        previous = keys[index - 1]
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
    source_named_side = _named_lateral_side(source.names)
    target_named_side = _named_lateral_side(target.names)
    if mirror:
        target_named_side = _opposite_lateral_side(target_named_side)
    if (
        source_named_side in {"left", "right"}
        and target_named_side in {"left", "right"}
        and source_named_side != target_named_side
    ):
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
        if token == "mixamorig" or re.fullmatch(r"mixamorig\d+", token):
            continue
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
    min_chain_score: float = 0.45,
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
    suppress_mirror_by_named_sides = _has_named_lateral_chains(
        branch_source
    ) and _has_named_lateral_chains(branch_target)
    use_mirror = False if suppress_mirror_by_named_sides else mirrored_score > direct_score
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
        main_score = _score_generic_chain_pair(main_source, main_target, mirror=use_mirror)
        if main_score >= min_chain_score:
            pair_candidates.append(
                (
                    main_source.end_name,
                    main_target.end_name,
                    main_score,
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
        "mirror_suppressed_by_named_sides": suppress_mirror_by_named_sides,
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


def _has_named_lateral_chains(chains: list[GenericSparseChain]) -> bool:
    named_sides = set()
    for chain in chains:
        side = _named_lateral_side(chain.names)
        if side in {"left", "right"}:
            named_sides.add(side)
    return "left" in named_sides and "right" in named_sides


def _named_lateral_side(names: list[str]) -> str:
    sides = [_infer_side_from_name(name) for name in names]
    left_count = sum(1 for side in sides if side == "left")
    right_count = sum(1 for side in sides if side == "right")
    if left_count > right_count:
        return "left"
    if right_count > left_count:
        return "right"
    return "unknown"


def _opposite_lateral_side(side: str) -> str:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return side


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
