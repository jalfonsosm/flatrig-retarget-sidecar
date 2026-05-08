"""Bridge Spine clips into Motion2Motion-friendly BVH and sparse mappings."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flatrig_retarget_sidecar.sparse_mapping import build_motion2motion_mapping_payload
from flatrig_retarget_sidecar.spine_import import SpineBoneRecord, SpinePackage

NAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]+")
EPSILON = 1e-8


@dataclass(slots=True)
class ExportedBvhJoint:
    index: int
    spine_name: str | None
    matching_name: str
    bvh_name: str
    parent_index: int
    parent_bvh_name: str | None
    offset_x: float
    offset_y: float
    offset_z: float
    length: float
    synthetic: bool = False


@dataclass(slots=True)
class ExportedSpineBvh:
    source_label: str
    animation_name: str
    duration: float
    fps: float
    frame_count: int
    root_bvh_name: str
    root_matching_name: str
    root_spine_name: str | None
    positions_mode: str
    original_to_bvh: dict[str, str]
    bvh_to_original: dict[str, str | None]
    joints: list[ExportedBvhJoint]
    motion2motion_non_root_translate_bones: list[str]
    ignored_scale_bones: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_label": self.source_label,
            "animation_name": self.animation_name,
            "duration": round(self.duration, 6),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "root_bvh_name": self.root_bvh_name,
            "root_matching_name": self.root_matching_name,
            "root_spine_name": self.root_spine_name,
            "positions_mode": self.positions_mode,
            "original_to_bvh": dict(self.original_to_bvh),
            "bvh_to_original": dict(self.bvh_to_original),
            "motion2motion_non_root_translate_bones": list(
                self.motion2motion_non_root_translate_bones
            ),
            "ignored_scale_bones": list(self.ignored_scale_bones),
            "joints": [asdict(joint) for joint in self.joints],
        }


def export_spine_animation_to_bvh(
    package: SpinePackage,
    animation_name: str,
    output_path: str | Path,
    *,
    fps: float = 30.0,
    positions_mode: str = "all",
    metadata_path: str | Path | None = None,
    sample_duration: float | None = None,
) -> ExportedSpineBvh:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if positions_mode not in {"all", "root"}:
        raise ValueError("positions_mode must be 'all' or 'root'")
    if animation_name not in package.animations:
        available = ", ".join(sorted(package.animations)) or "<none>"
        raise KeyError(
            f"Animation '{animation_name}' not found in {package.source_label}. "
            f"Available animations: {available}"
        )

    animation = package.animations[animation_name]
    animation_duration = compute_animation_duration(animation)
    duration = (
        max(float(sample_duration), 1.0 / fps)
        if sample_duration is not None
        else animation_duration
    )
    sample_times = build_sample_times(duration, fps)
    exported_joints, original_to_bvh, bvh_to_original = build_bvh_joint_layout(package)
    joint_index_by_spine_name = {
        joint.spine_name: joint.index for joint in exported_joints if joint.spine_name is not None
    }

    positions: list[list[float]] = []
    rotations: list[list[float]] = []
    for time_value in sample_times:
        sample_time = remap_sample_time(
            time_value,
            export_duration=duration,
            animation_duration=animation_duration,
        )
        local_pose_map = evaluate_local_pose_map(package, animation_name, sample_time)
        frame_positions: list[float] = []
        frame_rotations: list[float] = []
        for joint in exported_joints:
            if joint.synthetic:
                frame_positions.extend((0.0, 0.0, 0.0))
                frame_rotations.extend((0.0, 0.0, 0.0))
                continue
            pose = local_pose_map[joint.spine_name or ""]
            x = float(pose["x"])
            y = float(pose["y"])
            if positions_mode == "root" and joint.parent_index >= 0:
                x = 0.0
                y = 0.0
            frame_positions.extend((x, y, 0.0))
            frame_rotations.extend((0.0, 0.0, float(pose["rotation"])))
        positions.append(frame_positions)
        rotations.append(frame_rotations)

    metadata = ExportedSpineBvh(
        source_label=package.source_label,
        animation_name=animation_name,
        duration=duration,
        fps=fps,
        frame_count=len(sample_times),
        root_bvh_name=exported_joints[0].bvh_name,
        root_matching_name=exported_joints[0].matching_name,
        root_spine_name=exported_joints[0].spine_name,
        positions_mode=positions_mode,
        original_to_bvh=original_to_bvh,
        bvh_to_original=bvh_to_original,
        joints=exported_joints,
        motion2motion_non_root_translate_bones=_find_non_root_translate_bones(
            package,
            animation_name,
            joint_index_by_spine_name,
        ),
        ignored_scale_bones=_find_scale_timeline_bones(package, animation_name),
    )
    write_planar_bvh(
        output_path,
        metadata.joints,
        positions,
        rotations,
        fps=fps,
        positions_mode=positions_mode,
    )
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata.to_metadata(), indent=2) + "\n",
            encoding="utf-8",
        )
    return metadata


def build_exported_motion2motion_mapping(
    source: SpinePackage,
    target: SpinePackage,
    *,
    max_pairs: int = 20,
    min_chain_score: float = 0.45,
    mapping_file: str | Path | None = None,
) -> dict[str, Any]:
    source_joints, _, _ = build_bvh_joint_layout(source)
    target_joints, _, _ = build_bvh_joint_layout(target)
    if mapping_file:
        return _build_user_exported_motion2motion_mapping(
            mapping_file,
            source,
            target,
            source_joints,
            target_joints,
        )

    source_original_to_matching = {
        joint.spine_name: joint.matching_name
        for joint in source_joints
        if joint.spine_name is not None
    }
    target_original_to_matching = {
        joint.spine_name: joint.matching_name
        for joint in target_joints
        if joint.spine_name is not None
    }
    payload = build_motion2motion_mapping_payload(
        source,
        target,
        max_pairs=max_pairs,
        min_chain_score=min_chain_score,
    )
    rewritten_mapping: list[dict[str, str]] = []
    used_source: set[str] = set()
    used_target: set[str] = set()

    source_export_root = source_joints[0].matching_name
    target_export_root = target_joints[0].matching_name

    def add_pair(source_name: str | None, target_name: str | None) -> None:
        if not source_name or not target_name:
            return
        if source_name in used_source or target_name in used_target:
            return
        rewritten_mapping.append({"source": source_name, "target": target_name})
        used_source.add(source_name)
        used_target.add(target_name)

    add_pair(source_export_root, target_export_root)

    semantic_pairs = _suggest_semantic_biped_mapping(source, target)
    for source_name, target_name in semantic_pairs:
        if len(rewritten_mapping) >= max_pairs:
            break
        add_pair(
            source_original_to_matching.get(source_name),
            target_original_to_matching.get(target_name),
        )

    semantic_mode = len(semantic_pairs) >= 4
    for pair in payload["mapping"]:
        if len(rewritten_mapping) >= max_pairs:
            break
        if semantic_mode and (
            _semantic_biped_role(pair["source"]) is None
            or _semantic_biped_role(pair["target"]) is None
        ):
            continue
        source_name = source_original_to_matching.get(pair["source"])
        target_name = target_original_to_matching.get(pair["target"])
        add_pair(source_name, target_name)

    return {
        "source_name": payload["source_name"],
        "target_name": payload["target_name"],
        "root_joint": target_export_root,
        "mapping": rewritten_mapping,
        "metadata": {
            "source_export_root": source_export_root,
            "target_export_root": target_export_root,
            "source_export_root_bvh_name": source_joints[0].bvh_name,
            "target_export_root_bvh_name": target_joints[0].bvh_name,
            "source_original_root": _select_primary_root(source).name,
            "target_original_root": _select_primary_root(target).name,
            "source_bvh_joint_count": len(source_joints),
            "target_bvh_joint_count": len(target_joints),
            "semantic_pair_count": len(semantic_pairs),
        },
    }


def _suggest_semantic_biped_mapping(
    source: SpinePackage,
    target: SpinePackage,
) -> list[tuple[str, str]]:
    """Prefer obvious biped bone pairs before falling back to sparse geometry.

    The sparse matcher is useful for unknown 2D rigs, but generated FlatRig
    targets commonly expose Mixamo-style names. In that case, semantic anchors
    avoid unstable matches such as head -> HeadTop_End or foot -> Toe_End.
    """

    source_by_role = _semantic_role_lookup(source)
    target_by_role = _semantic_role_lookup(target)
    role_pairs = [
        ("hips", "hips"),
        ("chest", "chest"),
        ("torso", "chest"),
        ("torso", "torso"),
        ("neck", "neck"),
        ("head", "head"),
        ("front_upper_arm", "right_upper_arm"),
        ("front_forearm", "right_forearm"),
        ("front_hand", "right_hand"),
        ("rear_upper_arm", "left_upper_arm"),
        ("rear_forearm", "left_forearm"),
        ("rear_hand", "left_hand"),
        ("front_thigh", "right_thigh"),
        ("front_shin", "right_shin"),
        ("front_foot", "right_foot"),
        ("rear_thigh", "left_thigh"),
        ("rear_shin", "left_shin"),
        ("rear_foot", "left_foot"),
        ("left_upper_arm", "left_upper_arm"),
        ("left_forearm", "left_forearm"),
        ("left_hand", "left_hand"),
        ("right_upper_arm", "right_upper_arm"),
        ("right_forearm", "right_forearm"),
        ("right_hand", "right_hand"),
        ("left_thigh", "left_thigh"),
        ("left_shin", "left_shin"),
        ("left_foot", "left_foot"),
        ("right_thigh", "right_thigh"),
        ("right_shin", "right_shin"),
        ("right_foot", "right_foot"),
    ]

    pairs: list[tuple[str, str]] = []
    used_source: set[str] = set()
    used_target: set[str] = set()
    for source_role, target_role in role_pairs:
        source_name = source_by_role.get(source_role)
        target_name = target_by_role.get(target_role)
        if not source_name or not target_name:
            continue
        if source_name in used_source or target_name in used_target:
            continue
        pairs.append((source_name, target_name))
        used_source.add(source_name)
        used_target.add(target_name)
    return pairs


def _semantic_role_lookup(package: SpinePackage) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for bone in package.bones:
        role = _semantic_biped_role(bone.name)
        if role:
            lookup.setdefault(role, bone.name)
    return lookup


def _semantic_biped_role(name: str) -> str | None:
    key = _semantic_name_key(name)
    if not key:
        return None

    if key == "root":
        return "root"
    if key in {"hip", "hips", "pelvis"}:
        return "hips"
    if key in {"torso", "chest", "spine2", "spine3", "upperbody"}:
        return "chest"
    if key in {"spine", "spine1"}:
        return "torso"
    if key == "neck":
        return "neck"
    if key == "head":
        return "head"
    if "headtop" in key or key.endswith("end"):
        return None

    side = _semantic_side(key)
    if side is None:
        return None

    if "shoulder" in key:
        return f"{side}_shoulder"
    if "forearm" in key or "lowerarm" in key or "bracer" in key:
        return f"{side}_forearm"
    if "upperarm" in key or key.endswith("arm"):
        return f"{side}_upper_arm"
    if "hand" in key or "fist" in key:
        return f"{side}_hand"
    if "upleg" in key or "upperleg" in key or "thigh" in key:
        return f"{side}_thigh"
    if "shin" in key or ("leg" in key and "upleg" not in key and "upperleg" not in key):
        return f"{side}_shin"
    if "foot" in key:
        return f"{side}_foot"
    return None


def _semantic_side(key: str) -> str | None:
    if key.startswith("front"):
        return "front"
    if key.startswith("rear") or key.startswith("back"):
        return "rear"
    if key.startswith("left"):
        return "left"
    if key.startswith("right"):
        return "right"
    return None


def _semantic_name_key(name: str) -> str:
    leaf = str(name).split(":")[-1]
    leaf = re.sub(r"^mixamorig\d*[_-]?", "", leaf, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", leaf.lower())


def _strip_motion2motion_suffix(name: str) -> str:
    return re.sub(r"__\d{3}$", "", str(name))


def _exported_matching_lookup(joints: list[ExportedBvhJoint]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for joint in joints:
        for key in (
            joint.spine_name,
            joint.matching_name,
            joint.bvh_name,
            _strip_motion2motion_suffix(joint.bvh_name),
        ):
            if key is not None and str(key):
                lookup[str(key)] = joint.matching_name
    return lookup


def _build_user_exported_motion2motion_mapping(
    mapping_file: str | Path,
    source: SpinePackage,
    target: SpinePackage,
    source_joints: list[ExportedBvhJoint],
    target_joints: list[ExportedBvhJoint],
) -> dict[str, Any]:
    raw_payload = json.loads(Path(mapping_file).expanduser().read_text(encoding="utf-8"))
    raw_pairs = raw_payload.get("mapping")
    if not isinstance(raw_pairs, list):
        raw_pairs = raw_payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("Mapping file must contain a 'mapping' or 'pairs' array.")

    source_lookup = _exported_matching_lookup(source_joints)
    target_lookup = _exported_matching_lookup(target_joints)
    source_export_root = source_joints[0].matching_name
    target_export_root = target_joints[0].matching_name
    requested_root = raw_payload.get("root_joint") or raw_payload.get("target_root")
    root_joint = target_lookup.get(str(requested_root), target_export_root) if requested_root else target_export_root

    rewritten_mapping: list[dict[str, str]] = []
    used_source: set[str] = set()
    used_target: set[str] = set()

    def add_pair(source_name: str, target_name: str) -> None:
        if not source_name or not target_name:
            return
        if source_name in used_source or target_name in used_target:
            return
        rewritten_mapping.append({"source": source_name, "target": target_name})
        used_source.add(source_name)
        used_target.add(target_name)

    add_pair(source_export_root, root_joint)
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

    if not rewritten_mapping:
        raise ValueError("Mapping file did not contain any usable source/target pairs.")

    return {
        "source_name": str(raw_payload.get("source_name") or source.source_label),
        "target_name": str(raw_payload.get("target_name") or target.source_label),
        "root_joint": root_joint,
        "mapping": rewritten_mapping,
        "metadata": {
            "manual": True,
            "mapping_file": str(Path(mapping_file).expanduser()),
            "source_export_root": source_export_root,
            "target_export_root": target_export_root,
            "source_bvh_joint_count": len(source_joints),
            "target_bvh_joint_count": len(target_joints),
        },
    }


def compute_animation_duration(animation_payload: dict[str, Any]) -> float:
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


def build_sample_times(duration: float, fps: float) -> list[float]:
    if duration <= EPSILON:
        duration = 1.0
    frame_count = max(2, int(math.ceil(duration * fps)) + 1)
    times = [min(duration, index / fps) for index in range(frame_count)]
    if abs(times[-1] - duration) > 1e-6:
        times.append(duration)
    return times


def remap_sample_time(
    time_value: float,
    *,
    export_duration: float,
    animation_duration: float,
) -> float:
    if animation_duration <= EPSILON:
        return 0.0
    if export_duration <= EPSILON:
        return min(max(0.0, time_value), animation_duration)
    if abs(export_duration - animation_duration) <= 1e-6:
        return min(max(0.0, time_value), animation_duration)
    alpha = min(1.0, max(0.0, time_value / export_duration))
    return alpha * animation_duration


def evaluate_local_pose_map(
    package: SpinePackage,
    animation_name: str,
    time_value: float,
) -> dict[str, dict[str, float]]:
    animation_payload = package.animations.get(animation_name) or {}
    animation_bones = animation_payload.get("bones") or {}
    pose_by_name: dict[str, dict[str, float]] = {}

    for bone in package.bones:
        pose = {
            "x": bone.x,
            "y": bone.y,
            "rotation": bone.rotation,
            "scaleX": bone.scale_x,
            "scaleY": bone.scale_y,
        }
        timelines = animation_bones.get(bone.name)
        if timelines:
            rotate = sample_timeline(timelines.get("rotate"), ("value", "angle"), time_value)
            if rotate is not None:
                pose["rotation"] += get_rotate_key_value(rotate)
            translate = sample_timeline(timelines.get("translate"), ("x", "y"), time_value)
            if translate is not None:
                pose["x"] += float(translate.get("x", 0.0))
                pose["y"] += float(translate.get("y", 0.0))
            scale = sample_timeline(timelines.get("scale"), ("x", "y"), time_value)
            if scale is not None:
                pose["scaleX"] = float(scale.get("x", pose["scaleX"]))
                pose["scaleY"] = float(scale.get("y", pose["scaleY"]))
        pose["rotation"] = normalize_angle(float(pose["rotation"]))
        pose_by_name[bone.name] = pose

    return pose_by_name


def sample_timeline(
    keys: Any,
    fields: tuple[str, ...],
    time_value: float,
) -> dict[str, float] | None:
    if not isinstance(keys, list) or not keys:
        return None

    if time_value <= float(keys[0].get("time", 0.0)):
        return {field: _resolve_key_value(keys[0], field) for field in fields}

    for index in range(len(keys) - 1):
        current = keys[index] or {}
        next_key = keys[index + 1] or {}
        next_time = float(next_key.get("time", 0.0))
        if time_value > next_time:
            continue
        current_time = float(current.get("time", 0.0))
        span = next_time - current_time
        alpha = (time_value - current_time) / span if span > EPSILON else 0.0
        return {
            field: _resolve_key_value(current, field)
            + (_resolve_key_value(next_key, field) - _resolve_key_value(current, field)) * alpha
            for field in fields
        }

    last = keys[-1] or {}
    return {field: _resolve_key_value(last, field) for field in fields}


def get_rotate_key_value(key: dict[str, float]) -> float:
    if "value" in key and isinstance(key["value"], (int, float)):
        return float(key["value"])
    if "angle" in key and isinstance(key["angle"], (int, float)):
        return float(key["angle"])
    return 0.0


def build_bvh_joint_layout(
    package: SpinePackage,
) -> tuple[list[ExportedBvhJoint], dict[str, str], dict[str, str | None]]:
    primary_root = _select_primary_root(package)
    root_like_bones = _find_root_like_bones(package)

    used_names: set[str] = set()
    joints: list[ExportedBvhJoint] = []
    original_to_bvh: dict[str, str] = {}
    bvh_to_original: dict[str, str | None] = {}
    name_to_index: dict[str, int] = {}

    if len(root_like_bones) > 1:
        synthetic_matching_name = _sanitize_bvh_name("sidecar_root", used_names)
        synthetic_name = _motion2motion_export_name(synthetic_matching_name, 0)
        joints.append(
            ExportedBvhJoint(
                index=0,
                spine_name=None,
                matching_name=synthetic_matching_name,
                bvh_name=synthetic_name,
                parent_index=-1,
                parent_bvh_name=None,
                offset_x=0.0,
                offset_y=0.0,
                offset_z=0.0,
                length=0.0,
                synthetic=True,
            )
        )
        bvh_to_original[synthetic_name] = None
        name_to_index[synthetic_name] = 0
        for root_bone in sorted(
            root_like_bones, key=lambda bone: (bone.name != primary_root.name, bone.index)
        ):
            _append_bvh_subtree(
                root_bone,
                package,
                parent_index=0,
                parent_bvh_name=synthetic_name,
                joints=joints,
                used_names=used_names,
                original_to_bvh=original_to_bvh,
                bvh_to_original=bvh_to_original,
                name_to_index=name_to_index,
            )
    else:
        _append_bvh_subtree(
            primary_root,
            package,
            parent_index=-1,
            parent_bvh_name=None,
            joints=joints,
            used_names=used_names,
            original_to_bvh=original_to_bvh,
            bvh_to_original=bvh_to_original,
            name_to_index=name_to_index,
        )

    return joints, original_to_bvh, bvh_to_original


def write_planar_bvh(
    output_path: str | Path,
    joints: list[ExportedBvhJoint],
    positions: list[list[float]],
    rotations: list[list[float]],
    *,
    fps: float,
    positions_mode: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joint_children = _build_joint_children(joints)
    lines = ["HIERARCHY"]
    _write_joint_hierarchy(
        lines,
        joints=joints,
        joint_children=joint_children,
        joint_index=0,
        depth=0,
        positions_mode=positions_mode,
    )
    lines.append("MOTION")
    lines.append(f"Frames: {len(positions)}")
    lines.append(f"Frame Time: {1.0 / fps:.8f}")

    for frame_positions, frame_rotations in zip(positions, rotations, strict=True):
        values: list[str] = []
        cursor = 0
        for joint in joints:
            position_triplet = frame_positions[cursor : cursor + 3]
            rotation_triplet = frame_rotations[cursor : cursor + 3]
            cursor += 3
            if joint.parent_index < 0 or positions_mode == "all":
                values.extend(f"{value:.6f}" for value in position_triplet)
            values.extend(f"{value:.6f}" for value in rotation_triplet)
        lines.append(" ".join(values))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_joint_hierarchy(
    lines: list[str],
    *,
    joints: list[ExportedBvhJoint],
    joint_children: dict[int, list[int]],
    joint_index: int,
    depth: int,
    positions_mode: str,
) -> None:
    joint = joints[joint_index]
    indent = "\t" * depth
    label = "ROOT" if joint.parent_index < 0 else "JOINT"
    lines.append(f"{indent}{label} {joint.bvh_name}")
    lines.append(f"{indent}{{")
    channel_indent = f"{indent}\t"
    lines.append(
        f"{channel_indent}OFFSET {joint.offset_x:.6f} {joint.offset_y:.6f} {joint.offset_z:.6f}"
    )
    if joint.parent_index < 0 or positions_mode == "all":
        lines.append(
            f"{channel_indent}CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
        )
    else:
        lines.append(f"{channel_indent}CHANNELS 3 Xrotation Yrotation Zrotation")

    children = joint_children.get(joint_index) or []
    if children:
        for child_index in children:
            _write_joint_hierarchy(
                lines,
                joints=joints,
                joint_children=joint_children,
                joint_index=child_index,
                depth=depth + 1,
                positions_mode=positions_mode,
            )
    else:
        end_indent = f"{indent}\t"
        end_offset = joint.length if abs(joint.length) > EPSILON else 0.0
        lines.append(f"{end_indent}End Site")
        lines.append(f"{end_indent}{{")
        lines.append(f"{end_indent}\tOFFSET {end_offset:.6f} 0.000000 0.000000")
        lines.append(f"{end_indent}}}")

    lines.append(f"{indent}}}")


def _append_bvh_subtree(
    bone: SpineBoneRecord,
    package: SpinePackage,
    *,
    parent_index: int,
    parent_bvh_name: str | None,
    joints: list[ExportedBvhJoint],
    used_names: set[str],
    original_to_bvh: dict[str, str],
    bvh_to_original: dict[str, str | None],
    name_to_index: dict[str, int],
) -> None:
    if bone.name in original_to_bvh:
        return
    index = len(joints)
    matching_name = _sanitize_bvh_name(bone.name, used_names)
    bvh_name = _motion2motion_export_name(matching_name, index)
    if parent_index < 0:
        offset_x = 0.0
        offset_y = 0.0
        offset_z = 0.0
    else:
        offset_x = bone.x
        offset_y = bone.y
        offset_z = 0.0
    joints.append(
        ExportedBvhJoint(
            index=index,
            spine_name=bone.name,
            matching_name=matching_name,
            bvh_name=bvh_name,
            parent_index=parent_index,
            parent_bvh_name=parent_bvh_name,
            offset_x=offset_x,
            offset_y=offset_y,
            offset_z=offset_z,
            length=bone.length,
        )
    )
    original_to_bvh[bone.name] = bvh_name
    bvh_to_original[bvh_name] = bone.name
    name_to_index[bvh_name] = index

    children = [
        package.bones_by_name[child_name]
        for child_name in bone.children
        if child_name in package.bones_by_name
    ]
    children.sort(key=lambda child: child.index)
    for child in children:
        _append_bvh_subtree(
            child,
            package,
            parent_index=index,
            parent_bvh_name=bvh_name,
            joints=joints,
            used_names=used_names,
            original_to_bvh=original_to_bvh,
            bvh_to_original=bvh_to_original,
            name_to_index=name_to_index,
        )


def _find_non_root_translate_bones(
    package: SpinePackage,
    animation_name: str,
    joint_index_by_spine_name: dict[str, int],
) -> list[str]:
    animation_payload = package.animations.get(animation_name) or {}
    animation_bones = animation_payload.get("bones") or {}
    result: list[str] = []
    for bone_name, timelines in animation_bones.items():
        if not isinstance(timelines, dict):
            continue
        joint_index = joint_index_by_spine_name.get(bone_name)
        if joint_index is None or joint_index <= 0:
            continue
        if timelines.get("translate"):
            result.append(bone_name)
    return sorted(result)


def _find_scale_timeline_bones(package: SpinePackage, animation_name: str) -> list[str]:
    animation_payload = package.animations.get(animation_name) or {}
    animation_bones = animation_payload.get("bones") or {}
    result = [
        bone_name
        for bone_name, timelines in animation_bones.items()
        if isinstance(timelines, dict) and timelines.get("scale")
    ]
    return sorted(result)


def _find_root_like_bones(package: SpinePackage) -> list[SpineBoneRecord]:
    bones_by_name = package.bones_by_name
    roots = [
        bone for bone in package.bones if bone.parent is None or bone.parent not in bones_by_name
    ]
    if roots:
        return roots
    if package.bones:
        return [package.bones[0]]
    raise ValueError("Spine package does not contain any bones.")


def _select_primary_root(package: SpinePackage) -> SpineBoneRecord:
    roots = _find_root_like_bones(package)
    roots.sort(key=lambda bone: (bone.depth, bone.index))
    return roots[0]


def _build_joint_children(joints: list[ExportedBvhJoint]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for joint in joints:
        if joint.parent_index >= 0:
            result.setdefault(joint.parent_index, []).append(joint.index)
    return result


def _sanitize_bvh_name(name: str, used_names: set[str]) -> str:
    candidate = NAME_SANITIZE_RE.sub("_", name).strip("_")
    if not candidate:
        candidate = "bone"
    if candidate[0].isdigit():
        candidate = f"b_{candidate}"
    unique_name = candidate
    suffix = 2
    while unique_name in used_names:
        unique_name = f"{candidate}_{suffix}"
        suffix += 1
    used_names.add(unique_name)
    return unique_name


def _motion2motion_export_name(matching_name: str, index: int) -> str:
    return f"{matching_name}__{index:03d}"


def _resolve_key_value(key: dict[str, Any], field: str) -> float:
    value = key.get(field)
    if field == "value" and not isinstance(value, (int, float)):
        alt = key.get("angle")
        if isinstance(alt, (int, float)):
            return float(alt)
    if field == "angle" and not isinstance(value, (int, float)):
        alt = key.get("value")
        if isinstance(alt, (int, float)):
            return float(alt)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def normalize_angle(angle_degrees: float) -> float:
    value = math.fmod(angle_degrees, 360.0)
    if value <= -180.0:
        value += 360.0
    elif value > 180.0:
        value -= 360.0
    return value
