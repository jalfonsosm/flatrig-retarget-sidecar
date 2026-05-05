"""3D source animation retargeting into a FlatRig projected target rig."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from flatrig_retarget_sidecar.motion2motion_retarget import (
    M2M_LABEL,
    NUMERIC_EPSILON,
    _build_basis_2d,
    _compress_timeline_keys,
    _euler_xyz_degrees_to_matrix,
    _load_bvh_animation,
    _normalize_angle,
    _orthonormalize_2x2,
    _safe_inverse_2x2,
    _strip_motion2motion_suffix,
    _unwrap_angle_near,
    build_auto_sparse_mapping_payload,
    build_generic_skeleton_description,
    retarget_bvh_pair,
)
from flatrig_retarget_sidecar.scene_formats import (
    export_3d_animation_bvh,
    export_3d_rest_bvh,
    inspect_3d_source,
)
from flatrig_retarget_sidecar.spine_import import ROOT_DIR


@dataclass(slots=True)
class Motion2Motion3DRetargetResult:
    animations: list[dict[str, Any]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def retarget_3d_animations_to_model(
    source: str | Path,
    target_model: str | Path,
    *,
    animation_names: list[str] | None = None,
    output: str | Path | None = None,
    mapping_file: str | Path | None = None,
    matching_alpha: float | None = None,
    quality_threshold: float = 0.55,
    force_mapping_review: bool = False,
    view_preset: str = "side",
    view_dir: str | None = None,
    view_up: str | None = None,
    view_roll: float = 0.0,
    source_frame: int | None = None,
    projection_space: str = "world",
    fps: float = 30.0,
    frame_start: int | None = None,
    frame_end: int | None = None,
    include_preview_3d: bool = False,
) -> dict[str, Any]:
    """Retarget 3D source actions onto the target model rig, then emit FlatRig animation JSON."""
    source_path = Path(source).expanduser().resolve()
    target_path = Path(target_model).expanduser().resolve()
    requested_names = [str(name) for name in (animation_names or []) if str(name).strip()]

    with tempfile.TemporaryDirectory(prefix="m2m_3d_", dir=str(ROOT_DIR / "workflow")) as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        clip_names = requested_names or _inspect_source_action_names(source_path, temp_dir)
        if not clip_names:
            return _write_optional_output(
                output,
                {
                    "ok": False,
                    "detail": "No pose animation actions found in source.",
                    "source": str(source_path),
                    "target": str(target_path),
                    "animations": [],
                },
            )

        animations: list[dict[str, Any]] = []
        preview_3d_animations: list[dict[str, Any]] = []
        clip_diagnostics: list[dict[str, Any]] = []
        review_required = False
        target_joint_count = 0

        for clip_name in clip_names:
            clip_result = _retarget_one_3d_clip(
                source_path,
                target_path,
                clip_name,
                temp_dir=temp_dir,
                mapping_file=mapping_file,
                matching_alpha=matching_alpha,
                quality_threshold=quality_threshold,
                force_mapping_review=force_mapping_review,
                view_preset=view_preset,
                view_dir=view_dir,
                view_up=view_up,
                view_roll=view_roll,
                source_frame=source_frame,
                projection_space=projection_space,
                fps=fps,
                frame_start=frame_start,
                frame_end=frame_end,
                include_preview_3d=include_preview_3d,
            )
            clip_diagnostics.append(clip_result["diagnostics"])
            target_joint_count = max(
                target_joint_count,
                int(clip_result["diagnostics"].get("target_joint_count") or 0),
            )
            if clip_result.get("animation"):
                animations.append(clip_result["animation"])
            if clip_result.get("preview_3d"):
                preview_3d_animations.append(clip_result["preview_3d"])
            review_required = review_required or bool(
                clip_result["diagnostics"].get("mapping_review_required")
            )

        ok = bool(animations)
        payload = {
            "ok": ok,
            "detail": "retargeted" if ok else "No animations were retargeted.",
            "source": str(source_path),
            "target": str(target_path),
            "animations": animations,
            "diagnostics": {
                "backend_label": M2M_LABEL,
                "source": str(source_path),
                "target": str(target_path),
                "target_joint_count": target_joint_count,
                "requested_animation_count": len(clip_names),
                "retargeted_animation_count": len(animations),
                "mapping_review_required": review_required,
                "mapping_quality_threshold": float(quality_threshold),
                "force_mapping_review": bool(force_mapping_review),
                "mapping_editor_schema": "flatrig.joint_mapping.v1",
                "clips": clip_diagnostics,
            },
        }
        if include_preview_3d:
            payload["preview_3d_animations"] = preview_3d_animations
        return _write_optional_output(output, payload)


def _retarget_one_3d_clip(
    source_path: Path,
    target_path: Path,
    clip_name: str,
    *,
    temp_dir: Path,
    mapping_file: str | Path | None,
    matching_alpha: float | None,
    quality_threshold: float,
    force_mapping_review: bool,
    view_preset: str,
    view_dir: str | None,
    view_up: str | None,
    view_roll: float,
    source_frame: int | None,
    projection_space: str,
    fps: float,
    frame_start: int | None,
    frame_end: int | None,
    include_preview_3d: bool,
) -> dict[str, Any]:
    safe_clip_name = _safe_file_stem(clip_name)
    source_bvh = temp_dir / f"source_{safe_clip_name}.bvh"
    source_meta_path = temp_dir / f"source_{safe_clip_name}.meta.json"
    target_bvh = temp_dir / f"target_rest_{safe_clip_name}.bvh"
    target_meta_path = temp_dir / f"target_rest_{safe_clip_name}.meta.json"
    mapping_path = temp_dir / f"mapping_{safe_clip_name}.json"
    retargeted_bvh = temp_dir / f"retargeted_{safe_clip_name}.bvh"

    diagnostics: dict[str, Any] = {
        "animation_name": clip_name,
        "source": str(source_path),
        "target": str(target_path),
    }
    source_result = export_3d_animation_bvh(
        str(source_path),
        str(source_meta_path),
        bvh_output=str(source_bvh),
        animation_name=clip_name,
        fps=fps,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    if not source_result.ok:
        diagnostics.update({"ok": False, "detail": source_result.detail, "stage": "source_export"})
        return {"animation": None, "diagnostics": diagnostics}

    source_metadata = dict(source_result.payload)
    source_skeleton = build_generic_skeleton_from_exported_3d_metadata(source_metadata)
    target_result = export_3d_rest_bvh(
        str(target_path),
        str(target_meta_path),
        bvh_output=str(target_bvh),
        view_preset=view_preset,
        view_dir=view_dir,
        view_up=view_up,
        view_roll=view_roll,
        source_frame=source_frame,
        projection_space=projection_space,
        fps=fps,
        frame_count=int(source_metadata.get("frame_count") or 2),
    )
    if not target_result.ok:
        diagnostics.update(
            {
                "ok": False,
                "detail": f"Target BVH export failed: {target_result.detail}",
                "stage": "target_export",
                "target_export": target_result.payload,
            }
        )
        return {"animation": None, "diagnostics": diagnostics}
    target_metadata = dict(target_result.payload)
    target_skeleton = build_generic_skeleton_from_exported_3d_metadata(target_metadata)
    mapping_payload, mapping_diagnostics, mapping_mode = _resolve_mapping_payload(
        source_metadata,
        target_metadata,
        source_skeleton,
        target_skeleton,
        mapping_file=mapping_file,
    )
    mapping_quality = _estimate_mapping_quality(
        mapping_payload,
        mapping_diagnostics,
        source_joint_count=len(source_metadata.get("joints") or []),
        target_joint_count=len(target_metadata.get("joints") or []),
        manual=mapping_mode == "manual",
    )
    mapping_review_required = bool(force_mapping_review or mapping_quality < quality_threshold)
    mapping_path.write_text(json.dumps(mapping_payload, indent=2) + "\n", encoding="utf-8")

    try:
        bvh_result = retarget_bvh_pair(
            source_bvh,
            target_bvh,
            mapping_path,
            output_bvh=retargeted_bvh,
            matching_alpha=matching_alpha,
        )
        animation = bvh_to_flatrig_animation(
            retargeted_bvh,
            target_metadata,
            animation_name=clip_name,
        )
        preview_3d = (
            bvh_to_preview_3d_animation(
                retargeted_bvh,
                target_metadata,
                animation_name=clip_name,
            )
            if include_preview_3d
            else None
        )
    except Exception as exc:
        diagnostics.update(
            {
                "ok": False,
                "detail": str(exc),
                "stage": "motion2motion",
                "mapping_mode": mapping_mode,
                "mapping_quality_score": mapping_quality,
                "mapping_review_required": mapping_review_required,
                "target_bvh": str(target_bvh),
                "target_joint_count": len(target_metadata.get("joints") or []),
            }
        )
        return {"animation": None, "preview_3d": None, "diagnostics": diagnostics}

    diagnostics.update(
        {
            "ok": True,
            "detail": "retargeted",
            "stage": "complete",
            "source_bvh": str(source_bvh),
            "target_bvh": str(target_bvh),
            "retargeted_bvh": str(retargeted_bvh),
            "mapping_file": str(mapping_path),
            "mapping_mode": mapping_mode,
            "target_joint_count": len(target_metadata.get("joints") or []),
            "mapping_pair_count": len(mapping_payload.get("mapping") or []),
            "mapping_root_joint": mapping_payload.get("root_joint"),
            "mapping_quality_score": mapping_quality,
            "mapping_review_required": mapping_review_required,
            "mapping_diagnostics": mapping_diagnostics,
            "bvh_result": dict(bvh_result.diagnostics),
            "result_bone_count": len(animation.get("bones") or {}),
        }
    )
    return {"animation": animation, "preview_3d": preview_3d, "diagnostics": diagnostics}


def build_generic_skeleton_from_exported_3d_metadata(
    metadata: dict[str, Any],
):
    joints = list(metadata.get("joints") or [])
    return build_generic_skeleton_description(
        label=str(metadata.get("source") or metadata.get("source_label") or "3d"),
        matching_names=[str(joint.get("matching_name") or joint.get("name") or "joint") for joint in joints],
        bvh_names=[str(joint.get("bvh_name") or joint.get("matching_name") or "joint") for joint in joints],
        parent_indices=[int(joint.get("parent_index", -1)) for joint in joints],
        offsets=np.asarray([joint.get("offset") or [0.0, 0.0, 0.0] for joint in joints], dtype=np.float64),
    )


def bvh_to_flatrig_animation(
    bvh_path: str | Path,
    target_metadata: dict[str, Any],
    *,
    animation_name: str,
) -> dict[str, Any]:
    animation = _load_bvh_animation(bvh_path)
    frame_count = int(animation.rotations.shape[0])
    frametime = float(animation.frametime)
    names = [str(name) for name in animation.names]
    parents = [int(parent_index) for parent_index in animation.parents]
    offsets = np.asarray(animation.offsets, dtype=np.float64)
    positions = np.asarray(animation.positions, dtype=np.float64)
    rotations = np.asarray(animation.rotations, dtype=np.float64)

    joint_metadata_by_bvh_name = _joint_metadata_lookup(target_metadata)
    setup_bones = {
        str(bone.get("name")): dict(bone)
        for bone in target_metadata.get("bones_2d") or []
        if bone.get("name") is not None
    }
    projection_basis = _projection_basis(target_metadata)
    children = _metadata_children(target_metadata)
    track_map: dict[str, dict[str, list[dict[str, float]]]] = {}
    previous_rotation_values: dict[str, float] = {}

    for frame_index in range(frame_count):
        time_value = round(frame_index * frametime, 4)
        world_cache: list[dict[str, Any] | None] = [None] * len(names)
        local_cache_2d: dict[str, dict[str, float]] = {}

        for joint_index, bvh_name in enumerate(names):
            joint_metadata = joint_metadata_by_bvh_name.get(bvh_name) or joint_metadata_by_bvh_name.get(
                _strip_motion2motion_suffix(bvh_name)
            )
            if joint_metadata is None:
                continue

            local_position = np.asarray(positions[frame_index, joint_index], dtype=np.float64)
            local_offset = np.asarray(offsets[joint_index], dtype=np.float64)
            local_rotation_3d = _euler_xyz_degrees_to_matrix(rotations[frame_index, joint_index])
            parent_index = parents[joint_index]

            if parent_index < 0:
                head_3d = local_offset + local_position
                world_rotation_3d = local_rotation_3d
            else:
                parent_state = world_cache[parent_index]
                if parent_state is None:
                    continue
                head_3d = (
                    parent_state["head_3d"]
                    + parent_state["world_rotation_3d"] @ (local_offset + local_position)
                )
                world_rotation_3d = parent_state["world_rotation_3d"] @ local_rotation_3d

            tail_offset = _joint_tail_offset(joint_metadata, children)
            tail_3d = head_3d + world_rotation_3d @ tail_offset
            world_head_2d = projection_basis @ head_3d
            world_tail_2d = projection_basis @ tail_3d
            projected_axis_2d = np.asarray(world_tail_2d - world_head_2d, dtype=np.float64)
            axis_norm = float(np.linalg.norm(projected_axis_2d))
            original_name = joint_metadata.get("name")
            setup_bone = setup_bones.get(str(original_name)) if original_name else None
            fallback_axis = _setup_bone_world_axis(setup_bone, setup_bones)
            setup_length = float(setup_bone.get("length", 1.0)) if setup_bone else 1.0
            local_scale_x = axis_norm / max(1e-4, setup_length)
            local_scale_y = 1.0

            if axis_norm <= NUMERIC_EPSILON:
                projected_axis_2d = fallback_axis
                axis_norm = float(np.linalg.norm(projected_axis_2d))
            if axis_norm <= NUMERIC_EPSILON:
                projected_axis_2d = np.array((1.0, 0.0), dtype=np.float64)
            else:
                projected_axis_2d = projected_axis_2d / axis_norm

            if parent_index < 0:
                local_x = float(world_head_2d[0])
                local_y = float(world_head_2d[1])
                local_rotation_deg = math.degrees(
                    math.atan2(projected_axis_2d[1], projected_axis_2d[0])
                )
                world_basis_2d = _build_basis_2d(local_rotation_deg, local_scale_x, local_scale_y)
            else:
                parent_state = world_cache[parent_index]
                assert parent_state is not None
                parent_basis = parent_state["world_basis_2d"]
                # Use the full parent basis for position (matches setup extraction)
                local_position_2d = _safe_inverse_2x2(parent_basis) @ (
                    world_head_2d - parent_state["head_2d"]
                )
                # For rotation, respect inherit mode: NoScale bones use the rigid
                # (orthonormalized) parent basis, matching extract_bone_hierarchy.
                inherit_mode = setup_bone.get("inherit", "Normal") if setup_bone else "Normal"
                if inherit_mode == "NoScale":
                    rotation_basis = _orthonormalize_2x2(parent_basis)
                else:
                    rotation_basis = parent_basis
                local_axis = _safe_inverse_2x2(rotation_basis) @ projected_axis_2d
                local_rotation_deg = math.degrees(math.atan2(local_axis[1], local_axis[0]))
                local_x = float(local_position_2d[0])
                local_y = float(local_position_2d[1])
                world_basis_2d = rotation_basis @ _build_basis_2d(local_rotation_deg, local_scale_x, local_scale_y)

            world_cache[joint_index] = {
                "head_3d": head_3d,
                "head_2d": world_head_2d,
                "world_rotation_3d": world_rotation_3d,
                "world_basis_2d": world_basis_2d,
                "original_name": original_name,
            }

            if original_name and str(original_name) in setup_bones:
                local_cache_2d[str(original_name)] = {
                    "x": local_x,
                    "y": local_y,
                    "rotation": _normalize_angle(local_rotation_deg),
                    "scale_x": local_scale_x,
                    "scale_y": local_scale_y,
                }

        for bone_name, pose in local_cache_2d.items():
            setup_bone = setup_bones[bone_name]
            raw_rotation = float(pose["rotation"]) - float(setup_bone.get("rotation", 0.0))
            rel_rotation = _unwrap_angle_near(
                _normalize_angle(raw_rotation),
                previous_rotation_values.get(bone_name),
            )
            previous_rotation_values[bone_name] = rel_rotation
            rel_x = float(pose["x"]) - float(setup_bone.get("x", 0.0))
            rel_y = float(pose["y"]) - float(setup_bone.get("y", 0.0))
            timelines = track_map.setdefault(bone_name, {"rotate": [], "translate": []})
            timelines["rotate"].append(
                {
                    "time": time_value,
                    "angle": round(rel_rotation, 2),
                    "value": round(rel_rotation, 2),
                }
            )
            setup_scale_x = float(setup_bone.get("scaleX", 1.0))
            setup_scale_y = float(setup_bone.get("scaleY", 1.0))
            rel_scale_x = float(pose["scale_x"]) / max(1e-4, setup_scale_x)
            rel_scale_y = float(pose["scale_y"]) / max(1e-4, setup_scale_y)
            if abs(rel_scale_x - 1.0) > 1e-4 or abs(rel_scale_y - 1.0) > 1e-4:
                scale_track = timelines.setdefault("scale", [])
                scale_track.append(
                    {
                        "time": time_value,
                        "x": round(rel_scale_x, 4),
                        "y": round(rel_scale_y, 4),
                    }
                )

            if abs(rel_x) > 1e-4 or abs(rel_y) > 1e-4 or setup_bone.get("parent") is None:
                timelines["translate"].append(
                    {
                        "time": time_value,
                        "x": round(rel_x, 4),
                        "y": round(rel_y, 4),
                    }
                )

    compressed_bones: dict[str, Any] = {}
    for bone_name, timelines in track_map.items():
        rotate = _compress_timeline_keys(timelines["rotate"], ("angle", "value"), tolerance=0.01)
        translate = _compress_timeline_keys(timelines["translate"], ("x", "y"), tolerance=1e-4)
        payload: dict[str, Any] = {}
        if rotate:
            payload["rotate"] = rotate
        if translate:
            payload["translate"] = translate
        if payload:
            compressed_bones[bone_name] = payload

    return {
        "name": animation_name,
        "duration": round(max(0.0, (frame_count - 1) * frametime), 4),
        "bones": compressed_bones,
    }


def bvh_to_preview_3d_animation(
    bvh_path: str | Path,
    target_metadata: dict[str, Any],
    *,
    animation_name: str,
) -> dict[str, Any]:
    """Convert retargeted BVH into compact 3D joint samples for the web preview."""
    animation = _load_bvh_animation(bvh_path)
    frame_count = int(animation.rotations.shape[0])
    frametime = float(animation.frametime)
    names = [str(name) for name in animation.names]
    parents = [int(parent_index) for parent_index in animation.parents]
    offsets = np.asarray(animation.offsets, dtype=np.float64)
    positions = np.asarray(animation.positions, dtype=np.float64)
    rotations = np.asarray(animation.rotations, dtype=np.float64)

    joint_metadata_by_bvh_name = _joint_metadata_lookup(target_metadata)
    children = _metadata_children(target_metadata)
    frames: list[dict[str, Any]] = []

    for frame_index in range(frame_count):
        frame_bones: list[dict[str, Any]] = []
        world_cache: list[dict[str, Any] | None] = [None] * len(names)

        for joint_index, bvh_name in enumerate(names):
            joint_metadata = joint_metadata_by_bvh_name.get(bvh_name) or joint_metadata_by_bvh_name.get(
                _strip_motion2motion_suffix(bvh_name)
            )

            local_position = np.asarray(positions[frame_index, joint_index], dtype=np.float64)
            local_offset = np.asarray(offsets[joint_index], dtype=np.float64)
            local_rotation_3d = _euler_xyz_degrees_to_matrix(rotations[frame_index, joint_index])
            parent_index = parents[joint_index]

            if parent_index < 0:
                head_3d = local_offset + local_position
                world_rotation_3d = local_rotation_3d
            else:
                parent_state = world_cache[parent_index]
                if parent_state is None:
                    world_cache[joint_index] = None
                    continue
                head_3d = (
                    parent_state["head_3d"]
                    + parent_state["world_rotation_3d"] @ (local_offset + local_position)
                )
                world_rotation_3d = parent_state["world_rotation_3d"] @ local_rotation_3d

            world_cache[joint_index] = {
                "head_3d": head_3d,
                "world_rotation_3d": world_rotation_3d,
            }
            if joint_metadata is None:
                continue

            tail_offset = _joint_tail_offset(joint_metadata, children)
            tail_3d = head_3d + world_rotation_3d @ tail_offset
            original_name = joint_metadata.get("name")
            if original_name is None:
                continue
            frame_bones.append(
                {
                    "name": str(original_name),
                    "head": _vector3_to_list(head_3d),
                    "tail": _vector3_to_list(tail_3d),
                }
            )

        frames.append(
            {
                "time": round(frame_index * frametime, 4),
                "bones": frame_bones,
            }
        )

    return {
        "name": animation_name,
        "duration": round(max(0.0, (frame_count - 1) * frametime), 4),
        "fps": round((1.0 / frametime) if frametime > 0 else 30.0, 4),
        "frame_count": frame_count,
        "frames": frames,
    }


def _vector3_to_list(vector) -> list[float]:
    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    return [round(float(arr[0]), 6), round(float(arr[1]), 6), round(float(arr[2]), 6)]


def _inspect_source_action_names(source_path: Path, temp_dir: Path) -> list[str]:
    output_path = temp_dir / "source_inspect.json"
    result = inspect_3d_source(str(source_path), str(output_path))
    if not result.ok:
        return []
    actions = result.payload.get("actions") or []
    names = []
    for action in actions:
        name = action.get("name") if isinstance(action, dict) else None
        if name:
            names.append(str(name))
    return names


def _resolve_mapping_payload(
    source_metadata: dict[str, Any],
    target_metadata: dict[str, Any],
    source_skeleton,
    target_skeleton,
    *,
    mapping_file: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    root_joint = str(target_metadata.get("root_matching_name") or target_skeleton.root_name)
    if mapping_file:
        raw = json.loads(Path(mapping_file).expanduser().read_text(encoding="utf-8"))
        payload = _normalize_user_mapping_payload(raw, source_metadata, target_metadata, root_joint)
        diagnostics = {
            "manual": True,
            "mapping_pair_count": len(payload.get("mapping") or []),
            "source": str(Path(mapping_file).expanduser()),
        }
        return payload, diagnostics, "manual"

    payload, diagnostics = build_auto_sparse_mapping_payload(
        source_skeleton,
        target_skeleton,
        root_joint=root_joint,
    )
    return payload, diagnostics, "auto"


def _normalize_user_mapping_payload(
    raw_payload: dict[str, Any],
    source_metadata: dict[str, Any],
    target_metadata: dict[str, Any],
    root_joint: str,
) -> dict[str, Any]:
    source_lookup = _matching_name_lookup(source_metadata)
    target_lookup = _matching_name_lookup(target_metadata)
    raw_pairs = raw_payload.get("mapping")
    if not isinstance(raw_pairs, list):
        raw_pairs = raw_payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("Mapping file must contain a 'mapping' or 'pairs' array.")

    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
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
        source_name = source_lookup.get(str(raw_source), str(raw_source))
        target_name = target_lookup.get(str(raw_target), str(raw_target))
        key = (source_name, target_name)
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"source": source_name, "target": target_name})

    if not pairs:
        raise ValueError("Mapping file did not contain any usable source/target pairs.")

    requested_root = raw_payload.get("root_joint") or raw_payload.get("target_root") or root_joint
    return {
        "source_name": str(raw_payload.get("source_name") or Path(str(source_metadata.get("source") or "source")).stem),
        "target_name": str(raw_payload.get("target_name") or Path(str(target_metadata.get("source") or "target")).stem),
        "root_joint": target_lookup.get(str(requested_root), str(requested_root)),
        "mapping": pairs,
    }


def _estimate_mapping_quality(
    mapping_payload: dict[str, Any],
    mapping_diagnostics: dict[str, Any],
    *,
    source_joint_count: int,
    target_joint_count: int,
    manual: bool,
) -> float:
    if manual:
        return 1.0
    selected_pairs = mapping_diagnostics.get("selected_pairs") or []
    scores = [
        float(pair.get("score"))
        for pair in selected_pairs
        if isinstance(pair, dict) and isinstance(pair.get("score"), (int, float))
    ]
    average_score = sum(scores) / len(scores) if scores else 0.0
    pair_count = len(mapping_payload.get("mapping") or [])
    expected_pairs = max(1, min(12, source_joint_count, target_joint_count))
    coverage = min(1.0, pair_count / expected_pairs)
    return round(max(0.0, min(1.0, 0.75 * average_score + 0.25 * coverage)), 4)


def _matching_name_lookup(metadata: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for joint in metadata.get("joints") or []:
        matching = str(joint.get("matching_name") or "")
        if not matching:
            continue
        for key in (
            joint.get("name"),
            joint.get("bvh_name"),
            joint.get("matching_name"),
            _strip_motion2motion_suffix(str(joint.get("bvh_name") or "")),
        ):
            if key is not None and str(key):
                lookup[str(key)] = matching
    return lookup


def _joint_metadata_lookup(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for joint in metadata.get("joints") or []:
        joint_dict = dict(joint)
        for key in (
            joint.get("bvh_name"),
            joint.get("matching_name"),
            joint.get("name"),
            _strip_motion2motion_suffix(str(joint.get("bvh_name") or "")),
        ):
            if key is not None and str(key):
                lookup[str(key)] = joint_dict
    return lookup


def _metadata_children(metadata: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    children: dict[int, list[dict[str, Any]]] = {}
    for joint in metadata.get("joints") or []:
        parent_index = int(joint.get("parent_index", -1))
        if parent_index >= 0:
            children.setdefault(parent_index, []).append(dict(joint))
    return children


def _joint_tail_offset(joint_metadata: dict[str, Any], children: dict[int, list[dict[str, Any]]]) -> np.ndarray:
    tail_offset = np.asarray(joint_metadata.get("tail_offset") or [0.0, 0.0, 0.0], dtype=np.float64)
    if float(np.linalg.norm(tail_offset)) > NUMERIC_EPSILON:
        return tail_offset
    child_offsets = children.get(int(joint_metadata.get("index", -1))) or []
    for child in child_offsets:
        offset = np.asarray(child.get("offset") or [0.0, 0.0, 0.0], dtype=np.float64)
        if float(np.linalg.norm(offset)) > NUMERIC_EPSILON:
            return offset
    return np.array((1.0, 0.0, 0.0), dtype=np.float64)


def _projection_basis(target_metadata: dict[str, Any]) -> np.ndarray:
    """Return the 2x3 projection basis from view metadata.

    The fallback maps 3D-X to 2D-X and 3D-Z to 2D-Y, matching Blender's Z-up
    "front" view convention.  The previous fallback of [[1,0,0],[0,1,0]] mapped
    to the XY (top-down) plane, producing stretched or rotated results when the
    view metadata was missing.
    """
    # Front-view default: right = +X, up = +Z  (Blender Z-up convention)
    _FRONT_VIEW_FALLBACK = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    view_payload = target_metadata.get("view") or {}
    basis = np.asarray(view_payload.get("basis_2d") or _FRONT_VIEW_FALLBACK, dtype=np.float64)
    if basis.shape != (2, 3):
        return np.array(_FRONT_VIEW_FALLBACK, dtype=np.float64)
    return basis


def _setup_bone_world_axis(
    setup_bone: dict[str, Any] | None,
    setup_bones: dict[str, dict[str, Any]],
) -> np.ndarray:
    if setup_bone is None:
        return np.array((1.0, 0.0), dtype=np.float64)
    rotation = float(setup_bone.get("rotation", 0.0))
    parent_name = setup_bone.get("parent")
    while parent_name and parent_name in setup_bones:
        parent = setup_bones[parent_name]
        rotation += float(parent.get("rotation", 0.0))
        parent_name = parent.get("parent")
    radians = math.radians(rotation)
    return np.array((math.cos(radians), math.sin(radians)), dtype=np.float64)


def _safe_file_stem(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
    safe = safe.strip("._")
    return safe or "animation"


def _write_optional_output(output: str | Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
