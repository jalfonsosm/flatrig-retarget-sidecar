"""Blender worker for scene inspection and normalization."""

from __future__ import annotations

# ruff: noqa: I001

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    # bpy must be imported before bmesh/mathutils in the managed bpy runtime.
    import bpy
    import mathutils
    import bmesh
except ImportError:
    bpy = None
    mathutils = None
    bmesh = None
import numpy as np

# ============================================================================
# Constants
# ============================================================================

SEGMENT_EPSILON = 1e-8
TERMINAL_CHAIN_ROOT_RATIO = 0.6
TERMINAL_CHAIN_MAX_LENGTH_RATIO = 0.8
TERMINAL_CHAIN_PARENT_RATIO = 1.5
TERMINAL_CHAIN_MAX_SPAN = 6
VECTOR_EPSILON = 1e-8
M2M_SUFFIX_RE = re.compile(r"__[^\\s]{3}$")

VIEW_PRESETS = {
    "front": {
        "view_dir": (0.0, -1.0, 0.0),
        "right_axis": (1.0, 0.0, 0.0),
        "up_axis": (0.0, 0.0, 1.0),
    },
    "back": {
        "view_dir": (0.0, 1.0, 0.0),
        "right_axis": (1.0, 0.0, 0.0),
        "up_axis": (0.0, 0.0, 1.0),
    },
    "side": {
        "view_dir": (-1.0, 0.0, 0.0),
        "right_axis": (0.0, -1.0, 0.0),
        "up_axis": (0.0, 0.0, 1.0),
    },
    "side_r": {
        "view_dir": (1.0, 0.0, 0.0),
        "right_axis": (0.0, 1.0, 0.0),
        "up_axis": (0.0, 0.0, 1.0),
    },
    "top": {
        "view_dir": (0.0, 0.0, -1.0),
        "right_axis": (1.0, 0.0, 0.0),
        "up_axis": (0.0, 1.0, 0.0),
    },
    "bottom": {
        "view_dir": (0.0, 0.0, 1.0),
        "right_axis": (1.0, 0.0, 0.0),
        "up_axis": (0.0, 1.0, 0.0),
    },
    "three_quarter": {
        "view_dir": (1.0, 1.0, 0.0),
        "up_hint": (0.0, 0.0, 1.0),
    },
    "three_quarter_r": {
        "view_dir": (-1.0, 1.0, 0.0),
        "up_hint": (0.0, 0.0, 1.0),
    },
    "three_quarter_back": {
        "view_dir": (1.0, -1.0, 0.0),
        "up_hint": (0.0, 0.0, 1.0),
    },
    "three_quarter_back_r": {
        "view_dir": (-1.0, -1.0, 0.0),
        "up_hint": (0.0, 0.0, 1.0),
    },
    "isometric": {
        "view_dir": (-1.0, -1.0, -1.0),
        "up_hint": (0.0, 0.0, 1.0),
    },
    "isometric_r": {
        "view_dir": (1.0, -1.0, -1.0),
        "up_hint": (0.0, 0.0, 1.0),
    },
}

VIEW_ALIASES = {
    "3q": "three_quarter",
    "3q_r": "three_quarter_r",
    "3q_back": "three_quarter_back",
    "3q_back_r": "three_quarter_back_r",
    "3/4": "three_quarter",
    "3/4_r": "three_quarter_r",
    "iso": "isometric",
    "iso_r": "isometric_r",
}

VIEW_PRESET_NAMES = frozenset(VIEW_PRESETS.keys())

VECTOR_EPSILON = 1e-8
WORLD_UP = np.array((0.0, 0.0, 1.0), dtype=np.float64)
WORLD_Y = np.array((0.0, 1.0, 0.0), dtype=np.float64)
WORLD_X = np.array((1.0, 0.0, 0.0), dtype=np.float64)


def _normalize_vector(vector, *, label):
    """Normalize a numpy vector."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{label} must contain exactly 3 values.")
    length = float(np.linalg.norm(vector))
    if length <= VECTOR_EPSILON:
        raise ValueError(f"{label} must be non-zero.")
    return vector / length


def _axis_angle_rotation(axis, angle_degrees):
    """Compute 3x3 rotation matrix from axis-angle."""
    axis = _normalize_vector(axis, label="axis")
    angle_radians = math.radians(float(angle_degrees))
    cos_value = math.cos(angle_radians)
    sin_value = math.sin(angle_radians)
    outer = np.outer(axis, axis)
    cross = np.array(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=np.float64,
    )
    return cos_value * np.eye(3) + sin_value * cross + (1.0 - cos_value) * outer


def _finalize_view_config(view_name, preset_name, view_dir, right_axis, up_axis, roll_degrees):
    """Finalize and return a view configuration dict."""
    view_dir = _normalize_vector(view_dir, label="view_dir")
    right_axis = _normalize_vector(right_axis, label="right_axis")
    up_axis = _normalize_vector(up_axis, label="up_axis")
    depth_axis = _normalize_vector(-view_dir, label="depth_axis")
    basis_2d = np.stack((right_axis, up_axis), axis=0)
    basis_3d = np.stack((right_axis, up_axis, depth_axis), axis=0)
    return {
        "name": str(view_name or preset_name or "view"),
        "preset": preset_name,
        "mode": "preset" if preset_name is not None else "custom",
        "view_dir": view_dir,
        "right_axis": right_axis,
        "up_axis": up_axis,
        "depth_axis": depth_axis,
        "basis_2d": basis_2d,
        "basis_3d": basis_3d,
        "roll_degrees": float(roll_degrees),
    }


def _resolve_view_preset_name(view_name):
    """Resolve view name to preset name."""
    resolved = str(view_name or "side").strip().lower()
    if not resolved:
        return "side"
    resolved = VIEW_ALIASES.get(resolved, resolved)
    if resolved == "custom":
        raise ValueError("view 'custom' requires --view-dir to be provided.")
    if resolved not in VIEW_PRESETS:
        available = ", ".join(sorted(VIEW_PRESET_NAMES | set(VIEW_ALIASES)))
        raise ValueError(f"Unknown view '{view_name}'. Available presets: {available}")
    return resolved


def _build_explicit_view_config(view_name, view_dir, right_axis, up_axis, preset_name=None):
    """Build view config from explicitly provided vectors."""
    view_dir = _normalize_vector(view_dir, label="view_dir")
    right_axis = _normalize_vector(right_axis, label="right_axis")
    up_axis = _normalize_vector(up_axis, label="up_axis")
    if abs(float(np.dot(view_dir, right_axis))) > 1e-6:
        raise ValueError(f"View '{view_name}' has non-orthogonal view_dir/right_axis.")
    if abs(float(np.dot(view_dir, up_axis))) > 1e-6:
        raise ValueError(f"View '{view_name}' has non-orthogonal view_dir/up_axis.")
    if abs(float(np.dot(right_axis, up_axis))) > 1e-6:
        raise ValueError(f"View '{view_name}' has non-orthogonal right_axis/up_axis.")
    return _finalize_view_config(
        view_name=view_name,
        preset_name=preset_name,
        view_dir=view_dir,
        right_axis=right_axis,
        up_axis=up_axis,
        roll_degrees=0.0,
    )


def _build_view_config_from_direction(view_name, view_dir, up_hint=None, preset_name=None):
    """Build view config from direction vectors."""
    view_dir = _normalize_vector(view_dir, label="view_dir")
    up_hint_vec = _normalize_vector(up_hint if up_hint is not None else WORLD_UP, label="up_hint")
    if abs(float(np.dot(view_dir, up_hint_vec))) >= 1.0 - 1e-5:
        for fallback in (WORLD_Y, WORLD_X, -WORLD_X):
            fallback_norm = _normalize_vector(fallback, label="fallback_up")
            if abs(float(np.dot(view_dir, fallback_norm))) < 1.0 - 1e-5:
                up_hint_vec = fallback_norm
                break
    right_axis = np.cross(view_dir, up_hint_vec)
    right_axis = _normalize_vector(right_axis, label="right_axis")
    up_axis = np.cross(right_axis, view_dir)
    up_axis = _normalize_vector(up_axis, label="up_axis")
    return _finalize_view_config(
        view_name=view_name,
        preset_name=preset_name,
        view_dir=view_dir,
        right_axis=right_axis,
        up_axis=up_axis,
        roll_degrees=0.0,
    )


def _apply_roll_to_view_config(view_cfg, roll_degrees):
    """Apply roll rotation to view config."""
    roll_degrees = float(roll_degrees)
    if abs(roll_degrees) <= VECTOR_EPSILON:
        return dict(view_cfg)
    rotation = _axis_angle_rotation(view_cfg["view_dir"], roll_degrees)
    right_axis = rotation @ np.asarray(view_cfg["right_axis"], dtype=np.float64)
    up_axis = rotation @ np.asarray(view_cfg["up_axis"], dtype=np.float64)
    return _finalize_view_config(
        view_name=str(view_cfg.get("name") or "view"),
        preset_name=view_cfg.get("preset"),
        view_dir=view_cfg["view_dir"],
        right_axis=right_axis,
        up_axis=up_axis,
        roll_degrees=float(view_cfg.get("roll_degrees", 0.0)) + roll_degrees,
    )


def get_view_config(view_name="side", *, view_dir=None, up_hint=None, roll_degrees=0.0):
    """Return a normalized view configuration for preset or custom directions."""
    roll_degrees = float(roll_degrees or 0.0)
    if view_dir is not None:
        config = _build_view_config_from_direction(
            view_name="custom" if not str(view_name or "").strip() else str(view_name).strip(),
            view_dir=view_dir,
            up_hint=up_hint,
            preset_name=None,
        )
    else:
        preset_name = _resolve_view_preset_name(view_name)
        preset = VIEW_PRESETS[preset_name]
        if "right_axis" in preset and "up_axis" in preset:
            config = _build_explicit_view_config(
                view_name=preset_name,
                view_dir=np.array(preset["view_dir"], dtype=np.float64),
                right_axis=np.array(preset["right_axis"], dtype=np.float64),
                up_axis=np.array(preset["up_axis"], dtype=np.float64),
                preset_name=preset_name,
            )
        else:
            config = _build_view_config_from_direction(
                view_name=preset_name,
                view_dir=np.array(preset["view_dir"], dtype=np.float64),
                up_hint=np.array(preset["up_hint"], dtype=np.float64)
                if "up_hint" in preset
                else None,
                preset_name=preset_name,
            )
    if abs(roll_degrees) > VECTOR_EPSILON:
        config = _apply_roll_to_view_config(config, roll_degrees)
    return config


def _project_projection_space_direction(direction_3d, view_cfg):
    """Project a 3D direction onto the 2D view plane."""
    direction_3d = np.asarray(direction_3d, dtype=np.float64)
    basis_2d = np.asarray(view_cfg["basis_2d"], dtype=np.float64)
    if direction_3d.ndim == 1:
        return basis_2d @ direction_3d
    return direction_3d @ basis_2d.T


def project_point_ortho(point_3d, view_cfg, projection_inverse=None):
    """Project a 3D point into the configured 2D plane."""
    projected = _transform_point_to_projection_space(
        point_3d,
        projection_inverse=projection_inverse,
    )
    projected_2d = _project_projection_space_direction(projected, view_cfg)
    return (float(projected_2d[0]), float(projected_2d[1]))


def _transform_point_to_projection_space(point_3d, projection_inverse=None):
    """Transform world-space point into projection space."""
    point_3d = np.asarray(point_3d, dtype=np.float64)
    if projection_inverse is None:
        return point_3d

    squeeze = False
    if point_3d.ndim == 1:
        point_3d = point_3d[np.newaxis, :]
        squeeze = True

    point_h = np.concatenate(
        (point_3d, np.ones((point_3d.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    projection_inverse = np.asarray(projection_inverse, dtype=np.float64)

    if projection_inverse.ndim == 2:
        transformed = (projection_inverse @ point_h.T).T[:, :3]
    else:
        transformed = np.einsum("nij,nj->ni", projection_inverse, point_h)[:, :3]

    if squeeze:
        return transformed[0]
    return transformed


def point_depth(point_3d, view_cfg, projection_inverse=None):
    """Return a scalar depth where larger values are closer to the camera."""
    projected = _transform_point_to_projection_space(
        point_3d,
        projection_inverse=projection_inverse,
    )
    return float(np.dot(projected, np.asarray(view_cfg["depth_axis"], dtype=np.float64)))


def compute_projection_frame(points_2d, margin=0.06):
    """Build a square projection frame around a set of 2D points."""
    if len(points_2d) == 0:
        return {
            "center_x": 0.0,
            "center_y": 0.0,
            "span": 1.0,
            "min_x": -0.5,
            "max_x": 0.5,
            "min_y": -0.5,
            "max_y": 0.5,
            "margin": margin,
        }

    points_2d = np.asarray(points_2d, dtype=np.float64)
    if points_2d.ndim == 1:
        points_2d = points_2d[np.newaxis, :]

    xs = points_2d[:, 0]
    ys = points_2d[:, 1]
    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())

    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    span = max(width, height) * (1.0 + margin * 2.0)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    half = span * 0.5

    return {
        "center_x": center_x,
        "center_y": center_y,
        "span": span,
        "min_x": center_x - half,
        "max_x": center_x + half,
        "min_y": center_y - half,
        "max_y": center_y + half,
        "margin": margin,
    }


def project_points_to_uv(points_2d, frame):
    """Map projected 2D points to UVs inside the given projection frame."""
    span = max(frame["span"], 1e-6)
    u = (points_2d[:, 0] - frame["min_x"]) / span
    v = 1.0 - ((points_2d[:, 1] - frame["min_y"]) / span)
    return np.stack((u, v), axis=1)


def _set_scene_armatures_rest_pose(scene):
    """Temporarily force armatures to rest pose for bind/setup extraction."""
    previous = []
    for scene_obj in scene.objects:
        if scene_obj.type != "ARMATURE" or scene_obj.data is None:
            continue
        previous.append((scene_obj.data, scene_obj.data.pose_position))
        scene_obj.data.pose_position = "REST"
    if previous:
        bpy.context.view_layer.update()
    return previous


def _restore_scene_armature_pose_positions(previous):
    for armature_data, pose_position in previous:
        armature_data.pose_position = pose_position
    if previous:
        bpy.context.view_layer.update()


def _neutral_setup_role(name: str) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "", str(name).split(":")[-1].lower())
    if not key:
        return None
    if "shoulder" in key or "clavicle" in key:
        return None
    if "left" not in key and "right" not in key:
        return None
    # Arms and hands — straighten in an A-pose-like silhouette so the segmenter
    # can carve clean part outlines and z-order can resolve front/back arms.
    if "hand" in key or "wrist" in key:
        return "hand"
    if "forearm" in key or "lowerarm" in key:
        return "forearm"
    if "upperarm" in key or "uparm" in key or key.endswith("arm"):
        return "upper_arm"
    # Legs straightened too — but NOT feet/toes. Rotating the foot tip in world
    # space breaks rigs where "forward" isn't a fixed axis (Mixamo, some FBXs
    # arrive Y-up, others Z-up after Blender's import conversion). Leaving feet
    # and toes alone preserves whatever orientation the rig author chose, which
    # is almost always the right one for a static bind pose.
    if "shin" in key or "calf" in key or "lowerleg" in key or "loleg" in key:
        return "lower_leg"
    if "thigh" in key or "upleg" in key or "upperleg" in key or key.endswith("leg"):
        return "upper_leg"
    return None


def _neutral_setup_side(name: str) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "", str(name).split(":")[-1].lower())
    if "left" in key:
        return "left"
    if "right" in key:
        return "right"
    return None


# A-pose direction tables. We use armature world coordinates with the standard
# Blender post-import convention: +X = right, +Y = forward, +Z = up. Brazos
# salen hacia abajo y ~30° hacia afuera (silueta A clásica) en lugar de pegados
# al cuerpo (que era el bug previo: arms hugged the hips). Las piernas se dejan
# verticales — no se separan — para no quedar en split.
#
# sin30 ≈ 0.5, cos30 ≈ 0.866. Vector down-and-outward, normalized.
_ARM_OUTWARD_LEFT = (-0.5, 0.0, -0.866)
_ARM_OUTWARD_RIGHT = (0.5, 0.0, -0.866)


def _neutral_setup_direction(role: str, side: str | None) -> tuple[float, float, float] | None:
    """World-space direction the bone tip should point to in the A-pose.

    Returns None when the role/side combo should not be touched (e.g. feet,
    toes, or a sided role on an unsided bone), in which case the caller skips
    the rotation entirely.
    """
    if role in ("upper_arm", "forearm", "hand"):
        if side == "left":
            return _ARM_OUTWARD_LEFT
        if side == "right":
            return _ARM_OUTWARD_RIGHT
        return None
    if role in ("upper_leg", "lower_leg"):
        return (0.0, 0.0, -1.0)
    return None


def _clear_armature_animation_for_setup(armature_obj):
    animation_data = getattr(armature_obj, "animation_data", None)
    if animation_data is None:
        return
    animation_data.action = None
    for track in getattr(animation_data, "nla_tracks", []) or []:
        track.mute = True


def _reset_armature_pose(armature_obj):
    if armature_obj is None or getattr(armature_obj, "pose", None) is None:
        return
    if armature_obj.data is not None:
        armature_obj.data.pose_position = "POSE"
    for pose_bone in armature_obj.pose.bones:
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)
        if pose_bone.rotation_mode == "QUATERNION":
            pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        elif pose_bone.rotation_mode == "AXIS_ANGLE":
            pose_bone.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        else:
            pose_bone.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()


def _rotate_pose_bone_world_direction(armature_obj, pose_bone, desired_direction_world) -> bool:
    head_world, tail_world = _pose_bone_world_head_tail(armature_obj, pose_bone)
    current_direction = tail_world - head_world
    if float(current_direction.length) <= VECTOR_EPSILON:
        return False
    desired_direction = mathutils.Vector(desired_direction_world)
    if float(desired_direction.length) <= VECTOR_EPSILON:
        return False
    delta = current_direction.normalized().rotation_difference(desired_direction.normalized())
    pivot_to_origin = mathutils.Matrix.Translation(-head_world)
    pivot_back = mathutils.Matrix.Translation(head_world)
    rotate_world = delta.to_matrix().to_4x4()
    current_world_matrix = armature_obj.matrix_world @ pose_bone.matrix
    new_world_matrix = pivot_back @ rotate_world @ pivot_to_origin @ current_world_matrix
    pose_bone.matrix = armature_obj.matrix_world.inverted() @ new_world_matrix
    bpy.context.view_layer.update()
    return True


def _apply_neutral_down_setup_pose(armature_obj) -> dict[str, object]:
    """Pose compatible humanoid limbs into a neutral A-pose silhouette.

    Arms hang down, legs are vertical, feet/toes point forward. Used as the
    sprite/setup pose for source models that don't bring an animation, replacing
    the bare T/rest pose that produced thin-arm side renders.
    """
    if armature_obj is None or getattr(armature_obj, "pose", None) is None:
        return {"mode": "none", "posed_bone_count": 0}

    _clear_armature_animation_for_setup(armature_obj)
    _reset_armature_pose(armature_obj)

    # Order matters: rotate root segments before their children so the child
    # rotation is computed against the already-posed parent.
    role_order = {
        "upper_arm": 0,
        "forearm":   1,
        "hand":      2,
        "upper_leg": 3,
        "lower_leg": 4,
    }

    candidates = []
    for pose_bone in armature_obj.pose.bones:
        role = _neutral_setup_role(pose_bone.name)
        if role is None:
            continue
        side = _neutral_setup_side(pose_bone.name)
        direction = _neutral_setup_direction(role, side)
        if direction is None:
            continue
        candidates.append((role_order.get(role, 99), pose_bone.name, role, direction, pose_bone))
    candidates.sort(key=lambda item: (item[0], item[1]))

    posed_count = 0
    posed_roles: set[str] = set()
    for _order, _name, role, direction, pose_bone in candidates:
        try:
            if _rotate_pose_bone_world_direction(armature_obj, pose_bone, direction):
                posed_count += 1
                posed_roles.add(role)
        except Exception:
            continue

    return {
        "mode": "neutral_down",
        "posed_bone_count": posed_count,
        "posed_roles": sorted(posed_roles),
    }


def _should_use_neutral_setup_pose(source_frame=None, use_rest_pose=False) -> bool:
    """Disabled. The synthetic A-pose path was removed on 2026-05-09 — the
    bind pose now comes exclusively from the input model's first animation
    frame (or the optimization clip's first frame when the model has none).
    Kept as a no-op stub so existing callers don't change shape; will be
    deleted once `_apply_neutral_down_setup_pose` is unreachable from any
    code path.
    """
    return False


def _apply_auto_setup_pose(armature_obj, source_frame=None, use_rest_pose=False) -> dict[str, object]:
    # The A-pose path is intentionally inactive. We rely on the scene already
    # being at the chosen render frame (via _select_sprite_render_frame /
    # _resolve_setup_frame) and don't override pose bones here.
    return {"mode": "rest_pose" if use_rest_pose else "frame", "posed_bone_count": 0}


def _armature_uniform_scale(armature_obj) -> float:
    if armature_obj is None:
        return 1.0
    scale = armature_obj.matrix_world.to_scale()
    values = [abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z))]
    values = [value for value in values if value > VECTOR_EPSILON]
    if not values:
        return 1.0
    return sum(values) / len(values)


def _matrix3_to_json(matrix):
    return [
        [float(matrix[row][col]) for col in range(3)]
        for row in range(3)
    ]


def _matrix3_from_json(values):
    if values is None:
        return mathutils.Matrix.Identity(3)
    return mathutils.Matrix(
        (
            (float(values[0][0]), float(values[0][1]), float(values[0][2])),
            (float(values[1][0]), float(values[1][1]), float(values[1][2])),
            (float(values[2][0]), float(values[2][1]), float(values[2][2])),
        )
    )


def _armature_world_rotation(armature_obj):
    if armature_obj is None:
        return mathutils.Matrix.Identity(3)
    return armature_obj.matrix_world.to_quaternion().to_matrix()


def extract_2d_mesh(
    mesh_obj,
    view_cfg,
    projection_frame=None,
    source_frame=None,
    projection_inverse=None,
    use_rest_pose=False,
):
    """Extract the bind-pose mesh projected to 2D."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    if source_frame is None:
        source_frame = scene.frame_start
    scene.frame_set(source_frame)
    rest_pose_state = _set_scene_armatures_rest_pose(scene) if use_rest_pose else []
    depsgraph.update()

    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = None
    try:
        eval_mesh = eval_obj.to_mesh()
        world_mat = eval_obj.matrix_world

        # Extract vertices in world space
        vertices_3d = np.empty((len(eval_mesh.vertices), 3), dtype=np.float64)
        for index, vert in enumerate(eval_mesh.vertices):
            world_co = world_mat @ vert.co
            vertices_3d[index] = (world_co.x, world_co.y, world_co.z)

        # Project to 2D
        vertices_2d_list = []
        for i in range(len(vertices_3d)):
            pt = vertices_3d[i]
            projected = _transform_point_to_projection_space(pt, projection_inverse=projection_inverse)
            basis_2d = np.asarray(view_cfg["basis_2d"], dtype=np.float64)
            proj_2d = basis_2d @ projected
            vertices_2d_list.append([proj_2d[0], proj_2d[1]])
        vertices_2d = np.array(vertices_2d_list, dtype=np.float64)

        if projection_frame is None:
            projection_frame = compute_projection_frame(vertices_2d)
        fallback_uvs = project_points_to_uv(vertices_2d, projection_frame)

        # Compute depths
        depths = np.array(
            [
                point_depth(vertices_3d[i], view_cfg, projection_inverse=projection_inverse)
                for i in range(len(vertices_3d))
            ],
            dtype=np.float64,
        )

        bm = bmesh.new()
        try:
            bm.from_mesh(eval_mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bm.faces.ensure_lookup_table()
            uv_layer = bm.loops.layers.uv.active

            output_vertices_2d = []
            output_vertices_3d = []
            output_uvs = []
            output_depths = []
            source_vertex_indices = []
            triangles = []
            triangle_keys = []
            vertex_remap = {}

            for face in bm.faces:
                if len(face.loops) != 3:
                    continue
                tri = []
                source_tri = []
                for loop in face.loops:
                    source_index = int(loop.vert.index)
                    if source_index < 0 or source_index >= len(vertices_2d):
                        continue
                    source_tri.append(source_index)
                    if uv_layer is not None:
                        loop_uv = loop[uv_layer].uv
                        source_uv = (float(loop_uv.x), float(loop_uv.y))
                    else:
                        source_uv = (
                            float(fallback_uvs[source_index][0]),
                            float(fallback_uvs[source_index][1]),
                        )
                    remap_key = (
                        source_index,
                        round(source_uv[0], 8),
                        round(source_uv[1], 8),
                    )
                    output_index = vertex_remap.get(remap_key)
                    if output_index is None:
                        output_index = len(output_vertices_2d)
                        vertex_remap[remap_key] = output_index
                        output_vertices_2d.append(vertices_2d[source_index].tolist())
                        output_vertices_3d.append(vertices_3d[source_index].tolist())
                        output_uvs.append([source_uv[0], source_uv[1]])
                        output_depths.append(float(depths[source_index]))
                        source_vertex_indices.append(source_index)
                    tri.append(output_index)
                if len(tri) != 3 or len(set(tri)) != 3:
                    continue
                triangles.append(tri)
                triangle_keys.append(tuple(sorted(source_tri)))
        finally:
            bm.free()

        # Extract vertex groups (weights)
        vertex_groups = {}
        group_names = {group.index: group.name for group in mesh_obj.vertex_groups}

        for vert in mesh_obj.data.vertices:
            for group in vert.groups:
                group_name = group_names.get(group.group, f"group_{group.group}")
                vertex_groups.setdefault(group_name, []).append((vert.index, group.weight))

        return {
            "vertices_2d": output_vertices_2d,
            "vertices_3d": output_vertices_3d,
            "triangles": triangles,
            "triangle_keys": triangle_keys,
            "uvs": output_uvs,
            "depths": output_depths,
            "vertex_groups": vertex_groups,
            "source_vertex_indices": source_vertex_indices,
            "projection_frame": projection_frame,
        }
    finally:
        if eval_mesh is not None:
            eval_obj.to_mesh_clear()
        _restore_scene_armature_pose_positions(rest_pose_state)


def parse_args() -> argparse.Namespace:
    try:
        separator_index = sys.argv.index("--")
        script_args = sys.argv[separator_index + 1 :]
    except ValueError:
        script_args = []

    parser = argparse.ArgumentParser(description="Run the public 3D scene worker.")
    parser.add_argument(
        "command",
        choices=(
            "inspect",
            "convert",
            "extract-scene",
            "extract-animations",
            "render-sprites",
            "export-3d-animation-bvh",
            "export-3d-rest-bvh",
        ),
    )
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--view-preset",
        default="front",
        choices=list(VIEW_PRESETS.keys()),
        help="View preset for projection (front, back, side, side_r, top, bottom)",
    )
    parser.add_argument("--view-dir", default=None, help="Custom view direction as 'x,y,z' tuple")
    parser.add_argument("--view-up", default=None, help="Custom view up hint as 'x,y,z' tuple")
    parser.add_argument("--view-roll", type=float, default=0.0, help="View roll in degrees")
    parser.add_argument(
        "--source-frame", type=int, default=None, help="Source frame for pose evaluation"
    )
    parser.add_argument(
        "--use-rest-pose",
        action="store_true",
        default=False,
        help="Evaluate setup mesh and bones in armature rest pose",
    )
    # Animation extraction parameters
    parser.add_argument(
        "--projection-space",
        default="world",
        choices=("world", "root"),
        help="Projection space used by the Python pipeline",
    )
    parser.add_argument(
        "--animation",
        dest="animation_names",
        action="append",
        default=[],
        help="Animation name (can be specified multiple times)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Target animation FPS")
    parser.add_argument("--frame-start", type=int, default=None, help="First frame to sample")
    parser.add_argument("--frame-end", type=int, default=None, help="Last frame to sample")
    parser.add_argument("--frame-count", type=int, default=None, help="Frame count for rest BVH export")
    parser.add_argument("--sample-substeps", type=int, default=2, help="Subsamples per frame")
    parser.add_argument(
        "--no-optimize-animation-keys",
        dest="optimize_animation_keys",
        action="store_false",
        default=True,
    )
    parser.add_argument("--force-loop-closing-keys", action="store_true", default=False)
    parser.add_argument(
        "--pose-mode",
        default="full",
        choices=("full", "rotation_only", "local_rotation", "blend"),
        help="Pose extraction mode",
    )
    parser.add_argument(
        "--pose-blend", type=float, default=1.0, help="Blend amount for pose-mode=blend"
    )
    parser.add_argument(
        "--rotation-flatten", type=float, default=0.0, help="Rotation flatten amount"
    )
    parser.add_argument("--rotation-flatten-scope", default="all", help="Rotation flatten scope")
    parser.add_argument("--stretch-guard-enabled", action="store_true", default=False)
    parser.add_argument("--stretch-guard-max-scale", type=float, default=1.75)
    parser.add_argument("--stretch-guard-strength", type=float, default=0.65)
    parser.add_argument("--ik-leaf-refine-enabled", action="store_true", default=False)
    parser.add_argument("--ik-leaf-strength", type=float, default=0.35)
    parser.add_argument("--ik-leaf-iterations", type=int, default=6)
    parser.add_argument("--ik-leaf-max-chain-length", type=int, default=3)
    parser.add_argument("--ik-leaf-preserve-scale", type=float, default=0.65)
    parser.add_argument("--drop-problematic-frames", action="store_true", default=False)
    parser.add_argument("--preserve-root-motion", action="store_true", default=False)
    parser.add_argument("--preserve-root-rotation", action="store_true", default=False)
    parser.add_argument("--bvh-output", help="Path where a BVH export should be written")
    # Sprite rendering parameters
    parser.add_argument(
        "--parts-json", help="JSON file with part triangle keys and projection frames"
    )
    parser.add_argument("--images-dir", help="Directory where rendered part PNGs will be written")
    parser.add_argument(
        "--resolution", type=int, default=2048, help="Render resolution for each part image"
    )
    parser.add_argument(
        "--bind-frame", type=int, default=0, help="Frame to use for bind-pose sprite rendering"
    )
    parser.add_argument(
        "--mesh-target-vertices",
        type=int,
        default=5000,
        help="Target vertex count for source mesh reduction",
    )
    parser.add_argument(
        "--no-mesh-reduction",
        dest="mesh_reduction",
        action="store_false",
        default=True,
        help="Disable source mesh reduction",
    )
    return parser.parse_args(script_args)


def import_model(filepath: str) -> None:
    extension = Path(filepath).suffix.lower()
    if extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath, use_custom_props=False)
        return
    if extension in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=filepath)
        return
    raise ValueError(f"Unsupported format: {extension}. Use .fbx, .glb, or .gltf.")


def find_mesh_and_armature():
    mesh_obj = None
    armature_obj = None

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            if mesh_obj is None or len(obj.data.vertices) > len(mesh_obj.data.vertices):
                mesh_obj = obj
        elif obj.type == "ARMATURE":
            armature_obj = obj

    if mesh_obj and mesh_obj.parent and mesh_obj.parent.type == "ARMATURE":
        armature_obj = mesh_obj.parent

    if mesh_obj and armature_obj is None:
        for modifier in mesh_obj.modifiers:
            if modifier.type == "ARMATURE" and modifier.object:
                armature_obj = modifier.object
                break

    return mesh_obj, armature_obj


def is_pose_action(action) -> bool:
    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:
        slots = getattr(action, "slots", None) or []
        return any(
            str(getattr(slot, "target_id_type", "") or "").upper() in {"OBJECT", "ARMATURE"}
            for slot in slots
        )
    for fcurve in fcurves:
        path = str(fcurve.data_path or "")
        if path.startswith('pose.bones["') or path.startswith("pose.bones['"):
            return True
        if path in {
            "location",
            "rotation_euler",
            "rotation_quaternion",
            "rotation_axis_angle",
            "scale",
        }:
            return True
    return False


def list_actions(armature_obj) -> list[dict[str, object]]:
    actions = []
    seen = set()
    active_name = None
    if armature_obj and armature_obj.animation_data and armature_obj.animation_data.action:
        active_name = armature_obj.animation_data.action.name

    for action in bpy.data.actions:
        if not is_pose_action(action) or action.name in seen:
            continue
        seen.add(action.name)
        start, end = action.frame_range
        actions.append(
            {
                "name": action.name,
                "frame_start": int(round(start)),
                "frame_end": int(round(end)),
                "is_active": action.name == active_name,
            }
        )

    actions.sort(key=lambda item: (not bool(item["is_active"]), str(item["name"]).lower()))
    return actions


def inspect_source(source_path: str) -> dict[str, object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)
    mesh_obj, armature_obj = find_mesh_and_armature()

    payload: dict[str, object] = {
        "ok": True,
        "detail": "ready",
        "source": source_path,
        "format": Path(source_path).suffix.lower().lstrip("."),
        "source_space": "3d",
        "supports_character_build": True,
        "supports_animation_append": True,
        "mesh": None,
        "armature": None,
        "actions": [],
        "normalized_format": "glb"
        if Path(source_path).suffix.lower() == ".fbx"
        else Path(source_path).suffix.lower().lstrip("."),
    }

    if mesh_obj is not None:
        payload["mesh"] = {
            "name": mesh_obj.name,
            "vertex_count": int(len(mesh_obj.data.vertices)),
            "triangle_count": int(len(mesh_obj.data.polygons)),
        }

    if armature_obj is not None:
        payload["armature"] = {
            "name": armature_obj.name,
            "bone_count": int(len(armature_obj.data.bones)),
            "bone_names": sorted(bone.name for bone in armature_obj.data.bones),
        }
        payload["actions"] = list_actions(armature_obj)

    return payload


def convert_source(source_path: str, output_path: str) -> dict[str, object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    export_path = output
    if export_path.suffix.lower() != ".glb":
        export_path = export_path.with_suffix(".glb")

    bpy.ops.export_scene.gltf(
        filepath=str(export_path),
        export_format="GLB",
        export_yup=True,
        export_animations=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )

    inspected = inspect_source(str(export_path))
    return {
        "ok": True,
        "detail": "converted",
        "source": source_path,
        "output": str(export_path),
        "target_format": "glb",
        "inspection": inspected,
    }


def get_world_matrix(obj) -> list[list[float]]:
    """Get the world matrix of an object as a 4x4 list."""
    matrix = obj.matrix_world
    return [
        [matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]],
        [matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]],
        [matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]],
        [matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]],
    ]


def get_bone_world_matrix(armature_obj, bone_name: str) -> list[list[float]]:
    """Get the world matrix of a bone in armature space."""
    if armature_obj.type != "ARMATURE":
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    pose_bone = armature_obj.pose.bones.get(bone_name)
    if pose_bone is None:
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    # Get the bone's matrix in world space
    world_matrix = armature_obj.matrix_world @ pose_bone.matrix
    return [
        [world_matrix[0][0], world_matrix[0][1], world_matrix[0][2], world_matrix[0][3]],
        [world_matrix[1][0], world_matrix[1][1], world_matrix[1][2], world_matrix[1][3]],
        [world_matrix[2][0], world_matrix[2][1], world_matrix[2][2], world_matrix[2][3]],
        [world_matrix[3][0], world_matrix[3][1], world_matrix[3][2], world_matrix[3][3]],
    ]


def extract_mesh_data(mesh_obj) -> dict[str, Any]:
    """Extract mesh vertices, normals, UVs, and weights from a mesh object."""
    if mesh_obj is None:
        return {"vertex_count": 0, "vertices": [], "triangles": []}

    mesh = mesh_obj.data
    if hasattr(mesh, "calc_loop_triangles"):
        mesh.calc_loop_triangles()

    # Get vertex groups for skinning weights
    vertex_groups = {}
    for i, vg in enumerate(mesh_obj.vertex_groups):
        vertex_groups[vg.name] = i

    vertices = []
    normals = []
    weights = []  # List of (bone_index, weight) pairs per vertex

    mesh.calc_loop_triangles()

    for vert in mesh.vertices:
        # Position
        co = vert.co
        vertices.extend([co.x, co.y, co.z])

        # Normal
        no = vert.normal
        normals.extend([no.x, no.y, no.z])

        # Weights - collect from vertex groups
        vertex_weights = []
        for group in vert.groups:
            if group.weight > 0.001:  # Skip very small weights
                vertex_weights.append((group.group, group.weight))
        if not vertex_weights:
            vertex_weights = [(0, 1.0)]  # Default to first bone with full weight
        weights.append(vertex_weights)

    # Extract UVs from the first UV map
    uv_layer = None
    if hasattr(mesh, "uv_layers") and mesh.uv_layers:
        uv_layer = mesh.uv_layers[0].data

    # Build triangles from loop triangles
    triangles = []
    tri_uvs = []
    for tri in mesh.loop_triangles:
        triangles.extend([tri.vertices[0], tri.vertices[1], tri.vertices[2]])
        if uv_layer:
            tri_uvs.extend(
                [
                    [uv_layer[tri.loops[0]].uv.x, uv_layer[tri.loops[0]].uv.y],
                    [uv_layer[tri.loops[1]].uv.x, uv_layer[tri.loops[1]].uv.y],
                    [uv_layer[tri.loops[2]].uv.x, uv_layer[tri.loops[2]].uv.y],
                ]
            )

    result = {
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.loop_triangles),
        "vertices": vertices,
        "normals": normals,
        "triangles": triangles,
        "weights": weights,
    }

    if uv_layer:
        result["uvs"] = tri_uvs if tri_uvs else []

    # Extract base color texture from materials
    base_color_data = extract_base_color_texture(mesh_obj)
    if base_color_data:
        result["base_color_rgba"] = base_color_data["rgba"]
        result["base_color_width"] = base_color_data["width"]
        result["base_color_height"] = base_color_data["height"]
        result["base_color_channels"] = base_color_data["channels"]

    return result


def extract_base_color_texture(mesh_obj) -> dict[str, Any] | None:
    """Extract base color texture data from a mesh object's materials.

    Returns a dict with rgba (flattened), width, height, and channels (4 for RGBA).
    Returns None if no valid texture is found.
    """
    if mesh_obj is None:
        return None

    mesh = mesh_obj.data
    if not hasattr(mesh, "materials") or not mesh.materials:
        return None

    # Try to get the first material with a base color texture
    for slot in mesh.materials:
        if slot is None:
            continue

        material = slot
        # Handle material slots in Blender 5.0+
        if hasattr(slot, "material"):
            material = slot.material

        if material is None:
            continue

        # Check for Principled BSDF shader and its Base Color input
        if hasattr(material, "node_tree") and material.node_tree:
            nodes = material.node_tree.nodes
            for node in nodes:
                if node.type == "BSDF_PRINCIPLED":
                    # Try to get the Base Color input
                    base_color_input = node.inputs.get("Base Color")
                    if base_color_input and base_color_input.links:
                        link = base_color_input.links[0]
                        from_node = link.from_node

                        # Check if it's an image texture
                        if from_node and from_node.type == "TEX_IMAGE":
                            image = from_node.image
                            if image and image.size[0] > 0 and image.size[1] > 0:
                                # Read pixels - they come as RGBA
                                width, height = image.size
                                pixels = list(image.pixels)

                                # Convert to flattened RGBA list
                                # Image pixels are in [r,g,b,a, r,g,b,a, ...] format
                                rgba = []
                                for i in range(0, len(pixels), 4):
                                    rgba.append(int(pixels[i] * 255))  # R
                                    rgba.append(int(pixels[i + 1] * 255))  # G
                                    rgba.append(int(pixels[i + 2] * 255))  # B
                                    rgba.append(int(pixels[i + 3] * 255))  # A

                                return {
                                    "rgba": rgba,
                                    "width": width,
                                    "height": height,
                                    "channels": 4,
                                }

        # Fallback: try to find any image texture node
        if hasattr(material, "node_tree") and material.node_tree:
            nodes = material.node_tree.nodes
            for node in nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image and image.size[0] > 0 and image.size[1] > 0:
                        width, height = image.size
                        pixels = list(image.pixels)

                        rgba = []
                        for i in range(0, len(pixels), 4):
                            rgba.append(int(pixels[i] * 255))
                            rgba.append(int(pixels[i + 1] * 255))
                            rgba.append(int(pixels[i + 2] * 255))
                            rgba.append(int(pixels[i + 3] * 255))

                        return {
                            "rgba": rgba,
                            "width": width,
                            "height": height,
                            "channels": 4,
                        }

    return None


def _mesh_triangle_count(mesh_obj) -> int:
    mesh = mesh_obj.data
    if hasattr(mesh, "calc_loop_triangles"):
        mesh.calc_loop_triangles()
    return int(len(getattr(mesh, "loop_triangles", []) or mesh.polygons))


def reduce_mesh_object(mesh_obj, target_vertices=5000, enabled=True) -> dict[str, object]:
    """Reduce the source mesh in Blender before extraction.

    Decimate runs inside Blender so UV layers and vertex-group weights remain on
    the mesh that the native pipeline receives.
    """
    source_vertex_count = int(len(mesh_obj.data.vertices)) if mesh_obj is not None else 0
    source_triangle_count = _mesh_triangle_count(mesh_obj) if mesh_obj is not None else 0
    target_vertices = int(target_vertices or 0)
    report = {
        "enabled": bool(enabled),
        "applied": False,
        "mode": "sidecar_blender_decimate",
        "target_vertices": target_vertices,
        "source_vertex_count": source_vertex_count,
        "source_triangle_count": source_triangle_count,
        "output_vertex_count": source_vertex_count,
        "output_triangle_count": source_triangle_count,
        "reason": "disabled" if not enabled else "not_run",
    }

    if not enabled:
        return report
    if mesh_obj is None:
        report["reason"] = "no_mesh"
        return report
    if target_vertices <= 0:
        report["reason"] = "no_target"
        return report
    if source_vertex_count <= target_vertices:
        report["reason"] = "source_under_target"
        return report

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    current_vertices = source_vertex_count
    try:
        for pass_index in range(4):
            if current_vertices <= target_vertices:
                break
            ratio = max(0.01, min(1.0, float(target_vertices) / max(float(current_vertices), 1.0)))
            modifier = mesh_obj.modifiers.new(
                name=f"FlatRig_SourceMeshReduction_{pass_index + 1}",
                type="DECIMATE",
            )
            modifier.decimate_type = "COLLAPSE"
            modifier.ratio = ratio
            if hasattr(modifier, "use_collapse_triangulate"):
                modifier.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            bpy.context.view_layer.update()
            current_vertices = int(len(mesh_obj.data.vertices))
    except Exception as exc:
        raise RuntimeError(f"Source mesh reduction failed: {exc}") from exc

    mesh_obj.data.update()
    output_vertex_count = int(len(mesh_obj.data.vertices))
    output_triangle_count = _mesh_triangle_count(mesh_obj)
    report.update(
        {
            "applied": output_vertex_count < source_vertex_count,
            "output_vertex_count": output_vertex_count,
            "output_triangle_count": output_triangle_count,
            "reason": "target_reached"
            if output_vertex_count <= target_vertices
            else "best_effort",
        }
    )
    return report


# ============================================================================
# Projection Helpers
# ============================================================================


def _find_root_bone_name(armature):
    """Find the root bone name (bone with no parent)."""
    for bone in armature.data.bones:
        if bone.parent is None:
            return bone.name
    return None


def get_projection_reference_inverse(
    armature,
    projection_space="world",
    use_rest_pose=False,
    root_bone_name=None,
    reference_root_matrix=None,
):
    """Return the inverse transform for the requested projection space."""
    if projection_space != "root" or armature is None:
        return None

    if root_bone_name is None:
        root_bone_name = _find_root_bone_name(armature)
    if root_bone_name is None:
        return None

    if use_rest_pose:
        root_bone = armature.data.bones[root_bone_name]
        current_matrix = armature.matrix_world @ root_bone.matrix_local
    else:
        root_bone = armature.pose.bones[root_bone_name]
        current_matrix = armature.matrix_world @ root_bone.matrix

    current_matrix = np.array(current_matrix, dtype=np.float64)

    if reference_root_matrix is None:
        reference_root_matrix = current_matrix
    else:
        reference_root_matrix = np.asarray(reference_root_matrix, dtype=np.float64)

    current_rotation = orthonormalize_3x3(current_matrix[:3, :3])
    reference_rotation = orthonormalize_3x3(reference_root_matrix[:3, :3])
    projection_matrix = np.eye(4, dtype=np.float64)
    projection_matrix[:3, :3] = current_rotation @ reference_rotation.T
    projection_matrix[:3, 3] = current_matrix[:3, 3]

    return np.linalg.inv(projection_matrix)


def orthonormalize_3x3(matrix):
    """Extract the closest rigid rotation from a 3x3 transform using SVD."""
    matrix = np.asarray(matrix, dtype=np.float64)
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


# ============================================================================
# Skeleton Helpers
# ============================================================================


def safe_inverse_2x2(matrix, epsilon=None):
    """Invert a 2x2 matrix, falling back to the identity for degenerate cases."""
    if epsilon is None:
        epsilon = SEGMENT_EPSILON
    det = float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    if abs(det) <= epsilon:
        return np.eye(2, dtype=np.float64)
    return np.linalg.inv(matrix)


def orthonormalize_2x2(matrix, epsilon=None):
    """Orthonormalize a 2x2 matrix preserving handedness."""
    if epsilon is None:
        epsilon = SEGMENT_EPSILON
    det = float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    x_axis = np.array((matrix[0, 0], matrix[1, 0]), dtype=np.float64)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= epsilon:
        x_axis = np.array((1.0, 0.0), dtype=np.float64)
    else:
        x_axis = x_axis / x_norm
    y_axis = np.array((-x_axis[1], x_axis[0]), dtype=np.float64)
    if det < 0.0:
        y_axis = -y_axis
    return np.array(
        [[x_axis[0], y_axis[0]], [x_axis[1], y_axis[1]]],
        dtype=np.float64,
    )


def _build_2d_basis(rotation_deg, scale_x=1.0, scale_y=1.0):
    """Build the 2x2 matrix for a local bone transform."""
    rotation_rad = math.radians(rotation_deg)
    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)
    return np.array(
        [[cos_r * scale_x, -sin_r * scale_y], [sin_r * scale_x, cos_r * scale_y]],
        dtype=np.float64,
    )


def _normalize_angle(angle):
    """Normalize an angle to the [-180, 180] range."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def _bone_is_connected(rest_bone, epsilon=1e-4):
    """Return True when a child bone is effectively attached to its parent's tail."""
    if rest_bone.parent is None:
        return False
    if getattr(rest_bone, "use_connect", False):
        return True
    return (rest_bone.head_local - rest_bone.parent.tail_local).length <= epsilon


def _topological_sort(armature):
    """Sort bone names so parents always come before children."""
    sorted_bones = []
    visited = set()

    def visit(bone):
        if bone.name in visited:
            return
        if bone.parent:
            visit(bone.parent)
        visited.add(bone.name)
        sorted_bones.append(bone.name)

    for bone in armature.data.bones:
        visit(bone)

    return sorted_bones


def _sanitize_motion2motion_name(name, used_names):
    """Build a stable Motion2Motion matching name from a Blender bone name."""
    base = M2M_SUFFIX_RE.sub("", str(name or "")).strip()
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "joint"
    if base[0].isdigit():
        base = f"joint_{base}"

    candidate = base
    suffix = 2
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _motion2motion_export_name(matching_name, index):
    return f"{matching_name}__{int(index):03d}"


def _vector_to_json(vector):
    return [float(vector[0]), float(vector[1]), float(vector[2])]


def _view_config_to_json(view_cfg):
    return {
        "name": str(view_cfg.get("name") or ""),
        "preset": view_cfg.get("preset"),
        "mode": view_cfg.get("mode"),
        "view_dir": _vector_to_json(view_cfg["view_dir"]),
        "right_axis": _vector_to_json(view_cfg["right_axis"]),
        "up_axis": _vector_to_json(view_cfg["up_axis"]),
        "depth_axis": _vector_to_json(view_cfg["depth_axis"]),
        "basis_2d": np.asarray(view_cfg["basis_2d"], dtype=np.float64).tolist(),
        "basis_3d": np.asarray(view_cfg["basis_3d"], dtype=np.float64).tolist(),
        "roll_degrees": float(view_cfg.get("roll_degrees", 0.0)),
    }


def _build_3d_bvh_layout(armature_obj, source_frame=None, use_rest_pose=True):
    """Return Motion2Motion-friendly BVH joints for a Blender armature."""
    scene = bpy.context.scene
    if source_frame is not None:
        scene.frame_set(int(source_frame))
    rest_pose_state = _set_scene_armatures_rest_pose(scene) if use_rest_pose else []
    bpy.context.view_layer.update()

    bone_order = _topological_sort(armature_obj)
    bone_order_index = {name: index for index, name in enumerate(bone_order)}
    armature_scale = _armature_uniform_scale(armature_obj)
    armature_world = armature_obj.matrix_world.copy()
    armature_linear = armature_world.to_3x3()
    armature_rotation = _armature_world_rotation(armature_obj)
    root_bones = [
        bone
        for bone in armature_obj.data.bones
        if bone.parent is None and bone.name in bone_order_index
    ]
    root_bones.sort(key=lambda bone: bone_order_index[bone.name])
    use_synthetic_root = len(root_bones) > 1

    used_names = set()
    joints = []
    original_to_bvh = {}
    bvh_to_original = {}
    original_to_matching = {}
    matching_to_bvh = {}
    name_to_index = {}

    def bone_world_head_tail(rest_bone):
        if not use_rest_pose:
            pose_bone = armature_obj.pose.bones.get(rest_bone.name)
            if pose_bone is not None:
                return armature_world @ pose_bone.head, armature_world @ pose_bone.tail
        return armature_world @ rest_bone.head_local, armature_world @ rest_bone.tail_local

    try:
        if use_synthetic_root:
            matching_name = _sanitize_motion2motion_name("sidecar_root", used_names)
            bvh_name = _motion2motion_export_name(matching_name, 0)
            joints.append(
                {
                    "index": 0,
                    "name": None,
                    "matching_name": matching_name,
                    "bvh_name": bvh_name,
                    "parent_index": -1,
                    "parent_bvh_name": None,
                    "offset": [0.0, 0.0, 0.0],
                    "head": [0.0, 0.0, 0.0],
                    "tail": [0.0, 0.0, 0.0],
                    "tail_offset": [1.0, 0.0, 0.0],
                    "length": 0.0,
                    "synthetic": True,
                }
            )
            matching_to_bvh[matching_name] = bvh_name
            bvh_to_original[bvh_name] = None

        for bone_name in bone_order:
            rest_bone = armature_obj.data.bones[bone_name]
            index = len(joints)
            matching_name = _sanitize_motion2motion_name(rest_bone.name, used_names)
            bvh_name = _motion2motion_export_name(matching_name, index)
            parent_index = -1
            parent_bvh_name = None
            if rest_bone.parent is not None:
                parent_index = name_to_index.get(rest_bone.parent.name, -1)
            elif use_synthetic_root:
                parent_index = 0
            if parent_index >= 0:
                parent_bvh_name = joints[parent_index]["bvh_name"]

            head_vec, tail_vec = bone_world_head_tail(rest_bone)
            if rest_bone.parent is None:
                offset_vec = head_vec
            else:
                parent_head_vec, _parent_tail_vec = bone_world_head_tail(rest_bone.parent)
                offset_vec = head_vec - parent_head_vec
            tail_offset_vec = tail_vec - head_vec
            length = float(tail_offset_vec.length)

            joint = {
                "index": index,
                "name": rest_bone.name,
                "matching_name": matching_name,
                "bvh_name": bvh_name,
                "parent_index": int(parent_index),
                "parent_bvh_name": parent_bvh_name,
                "offset": _vector_to_json(offset_vec),
                "head": _vector_to_json(head_vec),
                "tail": _vector_to_json(tail_vec),
                "tail_offset": _vector_to_json(tail_offset_vec),
                "length": length,
                "synthetic": False,
            }
            joints.append(joint)
            name_to_index[rest_bone.name] = index
            original_to_bvh[rest_bone.name] = bvh_name
            bvh_to_original[bvh_name] = rest_bone.name
            original_to_matching[rest_bone.name] = matching_name
            matching_to_bvh[matching_name] = bvh_name

        return {
            "joints": joints,
            "original_to_bvh": original_to_bvh,
            "bvh_to_original": bvh_to_original,
            "original_to_matching": original_to_matching,
            "matching_to_bvh": matching_to_bvh,
            "root_bvh_name": joints[0]["bvh_name"] if joints else None,
            "root_matching_name": joints[0]["matching_name"] if joints else None,
            "coordinate_scale": armature_scale,
            "coordinate_linear": _matrix3_to_json(armature_linear),
            "coordinate_rotation": _matrix3_to_json(armature_rotation),
        }
    finally:
        _restore_scene_armature_pose_positions(rest_pose_state)


def _build_joint_children(joints):
    children = {int(joint["index"]): [] for joint in joints}
    for joint in joints:
        parent_index = int(joint.get("parent_index", -1))
        if parent_index >= 0:
            children.setdefault(parent_index, []).append(int(joint["index"]))
    return children


def _write_3d_joint_hierarchy(lines, joints, joint_children, joint_index, depth):
    joint = joints[joint_index]
    indent = "\t" * depth
    label = "ROOT" if int(joint.get("parent_index", -1)) < 0 else "JOINT"
    lines.append(f"{indent}{label} {joint['bvh_name']}")
    lines.append(f"{indent}{{")
    channel_indent = f"{indent}\t"
    offset = joint.get("offset") or [0.0, 0.0, 0.0]
    lines.append(f"{channel_indent}OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
    lines.append(
        f"{channel_indent}CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
    )

    children = joint_children.get(joint_index) or []
    if children:
        for child_index in children:
            _write_3d_joint_hierarchy(lines, joints, joint_children, child_index, depth + 1)
    else:
        tail_offset = joint.get("tail_offset") or [1.0, 0.0, 0.0]
        lines.append(f"{channel_indent}End Site")
        lines.append(f"{channel_indent}{{")
        lines.append(
            f"{channel_indent}\tOFFSET {tail_offset[0]:.6f} {tail_offset[1]:.6f} {tail_offset[2]:.6f}"
        )
        lines.append(f"{channel_indent}}}")

    lines.append(f"{indent}}}")


def _write_3d_bvh(output_path, joints, positions, rotations, fps):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not joints:
        raise ValueError("Cannot write BVH without joints.")

    joint_children = _build_joint_children(joints)
    lines = ["HIERARCHY"]
    _write_3d_joint_hierarchy(lines, joints, joint_children, 0, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {len(positions)}")
    lines.append(f"Frame Time: {1.0 / fps:.8f}")

    for frame_positions, frame_rotations in zip(positions, rotations, strict=True):
        values = []
        cursor = 0
        for _joint in joints:
            position_triplet = frame_positions[cursor : cursor + 3]
            rotation_triplet = frame_rotations[cursor : cursor + 3]
            cursor += 3
            values.extend(f"{value:.6f}" for value in position_triplet)
            values.extend(f"{value:.6f}" for value in rotation_triplet)
        lines.append(" ".join(values))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _matrix_xyz_euler_degrees(matrix):
    euler = matrix.to_euler("XYZ")
    return [math.degrees(float(euler.x)), math.degrees(float(euler.y)), math.degrees(float(euler.z))]


def _vector_from_json(values, fallback=(0.0, 0.0, 0.0)):
    values = values or fallback
    return mathutils.Vector((float(values[0]), float(values[1]), float(values[2])))


def _rotation_between_vectors(source, target):
    source_vec = mathutils.Vector(source)
    target_vec = mathutils.Vector(target)
    if source_vec.length <= VECTOR_EPSILON or target_vec.length <= VECTOR_EPSILON:
        return mathutils.Matrix.Identity(3)
    source_vec.normalize()
    target_vec.normalize()
    dot = max(-1.0, min(1.0, float(source_vec.dot(target_vec))))
    if dot > 1.0 - 1e-8:
        return mathutils.Matrix.Identity(3)
    if dot < -1.0 + 1e-8:
        axis = source_vec.cross(mathutils.Vector((1.0, 0.0, 0.0)))
        if axis.length <= VECTOR_EPSILON:
            axis = source_vec.cross(mathutils.Vector((0.0, 1.0, 0.0)))
        axis.normalize()
        return mathutils.Matrix.Rotation(math.pi, 3, axis)
    return source_vec.rotation_difference(target_vec).to_matrix()


def _pose_bone_world_head_tail(armature_obj, pose_bone):
    armature_world = armature_obj.matrix_world
    return armature_world @ pose_bone.head, armature_world @ pose_bone.tail


def _sample_action_frames(action, scene, fps, frame_start=None, frame_end=None):
    scene_fps = float(scene.render.fps) / max(float(scene.render.fps_base), VECTOR_EPSILON)
    start, end = action.frame_range if action is not None else (scene.frame_start, scene.frame_end)
    if frame_start is not None:
        start = float(frame_start)
    if frame_end is not None:
        end = float(frame_end)
    if end < start:
        end = start
    duration_seconds = max((float(end) - float(start)) / max(scene_fps, VECTOR_EPSILON), 1.0 / fps)
    frame_count = max(2, int(math.floor(duration_seconds * fps)) + 1)
    frame_step = scene_fps / fps
    return [float(start) + index * frame_step for index in range(frame_count)]


def _set_scene_frame_float(scene, frame_value):
    frame_int = int(math.floor(float(frame_value)))
    subframe = float(frame_value) - float(frame_int)
    scene.frame_set(frame_int, subframe=subframe)
    bpy.context.view_layer.update()


def _collect_3d_bvh_frames(armature_obj, layout, sample_frames, fps):
    scene = bpy.context.scene
    positions = []
    rotations = []
    joints = list(layout["joints"])
    for frame_value in sample_frames:
        _set_scene_frame_float(scene, frame_value)
        frame_positions = []
        frame_rotations = []
        world_cache = [None] * len(joints)
        for joint in joints:
            joint_index = int(joint.get("index", len(frame_positions) // 3))
            parent_index = int(joint.get("parent_index", -1))
            if joint.get("synthetic"):
                world_cache[joint_index] = {
                    "head": _vector_from_json(joint.get("head")),
                    "rotation": mathutils.Matrix.Identity(3),
                }
                frame_positions.extend((0.0, 0.0, 0.0))
                frame_rotations.extend((0.0, 0.0, 0.0))
                continue
            pose_bone = armature_obj.pose.bones.get(joint["name"])
            if pose_bone is None:
                world_cache[joint_index] = None
                frame_positions.extend((0.0, 0.0, 0.0))
                frame_rotations.extend((0.0, 0.0, 0.0))
                continue

            rest_offset = _vector_from_json(joint.get("offset"))
            tail_offset = _vector_from_json(joint.get("tail_offset"), fallback=(1.0, 0.0, 0.0))
            head_world, tail_world = _pose_bone_world_head_tail(armature_obj, pose_bone)
            posed_tail_axis_world = tail_world - head_world

            parent_state = world_cache[parent_index] if 0 <= parent_index < len(world_cache) else None
            if parent_state is not None:
                parent_rotation = parent_state["rotation"]
                parent_rotation_inv = parent_rotation.inverted()
                local_position = parent_rotation_inv @ (head_world - parent_state["head"]) - rest_offset
                desired_axis_parent = parent_rotation_inv @ posed_tail_axis_world
                local_rotation = _rotation_between_vectors(tail_offset, desired_axis_parent)
                world_rotation = parent_rotation @ local_rotation
            else:
                local_position = head_world - rest_offset
                local_rotation = _rotation_between_vectors(tail_offset, posed_tail_axis_world)
                world_rotation = local_rotation

            world_cache[joint_index] = {
                "head": head_world,
                "rotation": world_rotation,
            }
            frame_positions.extend((
                float(local_position.x),
                float(local_position.y),
                float(local_position.z),
            ))
            frame_rotations.extend(_matrix_xyz_euler_degrees(local_rotation))
        positions.append(frame_positions)
        rotations.append(frame_rotations)
    return positions, rotations


def _rest_3d_bvh_frames(layout, frame_count=2):
    frame_count = max(2, int(frame_count or 2))
    frame_positions = []
    frame_rotations = []
    for _joint in layout["joints"]:
        frame_positions.extend((0.0, 0.0, 0.0))
        frame_rotations.extend((0.0, 0.0, 0.0))
    return (
        [list(frame_positions) for _ in range(frame_count)],
        [list(frame_rotations) for _ in range(frame_count)],
    )


def _resolve_action_for_export(armature_obj, animation_names):
    requested = [str(name) for name in (animation_names or []) if str(name).strip()]
    actions = [action for action in bpy.data.actions if is_pose_action(action)]
    if requested:
        wanted = requested[0]
        for action in actions:
            if action.name == wanted:
                return action
        wanted_lower = wanted.lower()
        for action in actions:
            action_name_lower = str(action.name).lower()
            action_tokens = [token.lower() for token in str(action.name).split("|")]
            if wanted_lower == action_name_lower or wanted_lower in action_tokens:
                return action
            if action_name_lower.endswith("|" + wanted_lower) or wanted_lower in action_name_lower:
                return action
        available = ", ".join(sorted(action.name for action in actions)) or "<none>"
        raise ValueError(f"Animation '{wanted}' not found. Available animations: {available}")
    if armature_obj.animation_data and armature_obj.animation_data.action:
        return armature_obj.animation_data.action
    if actions:
        return sorted(actions, key=lambda action: str(action.name).lower())[0]
    return None


def _select_sprite_render_frame(armature_obj, source_frame=None) -> int:
    """Pick the frame used as bind pose for sprite/setup extraction.

    Policy (decided 2026-05-09):
      * Explicit positive source_frame wins.
      * Otherwise, return the **first frame** of the input model's action — not
        a percentage of the duration. The first frame is what the user can
        guarantee looks good (the previous 35%-of-duration heuristic landed on
        unpredictable poses and was the original "thin arms" bug).
      * If no action exists, fall back to scene.frame_start. The intended
        long-term fallback is the first frame of a curated single-clip
        optimization animation; that wiring is pending — see
        TODO(single-clip-optimization) below.
    """
    if source_frame is not None and int(source_frame) > 0:
        return int(source_frame)

    scene = bpy.context.scene
    if armature_obj is None:
        return int(scene.frame_start or 1)

    action = _resolve_action_for_export(armature_obj, [])
    if action is not None:
        start, _end = action.frame_range
        return int(round(float(start)))

    # TODO(single-clip-optimization): when the curated single optimization clip
    # is wired through, load its first frame here instead of falling back to
    # scene.frame_start. This is the path taken when the input model brings no
    # animation of its own.
    return int(scene.frame_start or 1)


def _resolve_setup_frame(
    armature_obj,
    source_frame=None,
    use_rest_pose=False,
    neutral_auto_pose=False,
) -> int:
    """Resolve the shared setup frame for mesh, bones, target BVH, and sprites.

    source_frame=-1 means "auto": choose a stable in-action pose when possible.
    Rest-pose extraction still evaluates at a concrete scene frame so evaluated
    meshes and projection metadata stay deterministic.
    """
    scene = bpy.context.scene
    if source_frame is not None:
        source_frame = int(source_frame)
        if source_frame > 0:
            return source_frame
        if source_frame < 0 and neutral_auto_pose:
            return int(scene.frame_start or 1)
        if source_frame < 0 and not use_rest_pose:
            return _select_sprite_render_frame(armature_obj)
    return int(scene.frame_start or 1)


def export_3d_animation_bvh_cli(
    source_path: str,
    output_path: str,
    bvh_output: str,
    animation_names: list = None,
    fps: float = 30.0,
    frame_start: int = None,
    frame_end: int = None,
) -> dict[str, object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)
    _mesh_obj, armature_obj = find_mesh_and_armature()
    if armature_obj is None:
        return {"ok": False, "detail": "No armature found in scene"}
    if fps <= 0.0:
        return {"ok": False, "detail": "fps must be > 0"}

    action = _resolve_action_for_export(armature_obj, animation_names)
    if action is None:
        return {"ok": False, "detail": "No pose animation actions found in scene"}
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    armature_obj.animation_data.action = action

    layout = _build_3d_bvh_layout(armature_obj)
    sample_frames = _sample_action_frames(
        action,
        bpy.context.scene,
        fps,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    positions, rotations = _collect_3d_bvh_frames(armature_obj, layout, sample_frames, fps)
    _write_3d_bvh(bvh_output, layout["joints"], positions, rotations, fps)

    payload = {
        "ok": True,
        "detail": "exported",
        "source": source_path,
        "output": str(Path(output_path)),
        "bvh_output": str(Path(bvh_output)),
        "animation_name": action.name,
        "duration": round((len(sample_frames) - 1) / fps, 4),
        "fps": float(fps),
        "frame_count": len(sample_frames),
        "frame_time": 1.0 / fps,
        "positions_mode": "all",
        **layout,
    }
    return payload


def export_3d_rest_bvh_cli(
    source_path: str,
    output_path: str,
    bvh_output: str,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
    source_frame: int = None,
    use_rest_pose: bool = False,
    projection_space: str = "world",
    fps: float = 30.0,
    frame_count: int = None,
) -> dict[str, object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)
    _mesh_obj, armature_obj = find_mesh_and_armature()
    if armature_obj is None:
        return {"ok": False, "detail": "No armature found in scene"}
    if fps <= 0.0:
        return {"ok": False, "detail": "fps must be > 0"}

    setup_frame = _resolve_setup_frame(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
        neutral_auto_pose=True,
    )
    bpy.context.scene.frame_set(setup_frame)
    setup_pose = _apply_auto_setup_pose(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
    )
    view_cfg = get_view_config(
        view_name=view_preset,
        view_dir=tuple(view_dir) if view_dir is not None else None,
        up_hint=tuple(view_up) if view_up is not None else None,
        roll_degrees=view_roll,
    )
    bones_2d = extract_bone_hierarchy(
        armature_obj,
        view_cfg,
        source_frame=setup_frame,
        use_rest_pose=use_rest_pose,
        projection_space=projection_space,
        projection_reference_root=None,
    )
    # The "rest" BVH that Motion2Motion uses as the target reference must be
    # built from the actual 3D rest pose of the rig, NOT from the bind frame
    # the user picked for sprite rendering. The source BVH (input animation)
    # is also built in rest pose (default of `_build_3d_bvh_layout`) and M2M
    # assumes both rigs share that convention. Building the target layout from
    # the bind frame (e.g. frame 1 of a Walk action) made M2M interpret every
    # rotation against a posed reference, producing visibly broken retargets
    # whenever the bind frame wasn't the rest pose. Fixed 2026-05-09.
    layout = _build_3d_bvh_layout(
        armature_obj,
        source_frame=None,
        use_rest_pose=True,
    )
    positions, rotations = _rest_3d_bvh_frames(layout, frame_count=frame_count)
    _write_3d_bvh(bvh_output, layout["joints"], positions, rotations, fps)

    return {
        "ok": True,
        "detail": "exported",
        "source": source_path,
        "output": str(Path(output_path)),
        "bvh_output": str(Path(bvh_output)),
        "animation_name": "__target_rest__",
        "duration": round((len(positions) - 1) / fps, 4),
        "fps": float(fps),
        "frame_count": len(positions),
        "frame_time": 1.0 / fps,
        "positions_mode": "all",
        "projection_space": projection_space,
        "setup_frame": setup_frame,
        "setup_pose": setup_pose,
        "use_rest_pose": bool(use_rest_pose),
        "retarget_use_rest_pose": bool(use_rest_pose),
        "view": _view_config_to_json(view_cfg),
        "bones_2d": bones_2d,
        **layout,
    }


def _default_inherit_mode(record):
    """Determine inherit mode based on terminal chain status."""
    if record.get("terminal_chain"):
        return "NoScale"
    return "Normal"


def _basis_inverse_for_inherit(parent_state, inherit_mode):
    """Get the basis inverse considering inherit mode."""
    basis = parent_state["matrix"]
    if inherit_mode == "NoScale":
        basis = parent_state["rigid_matrix"]
    return safe_inverse_2x2(basis)


def _compose_world_matrix(parent_state, local_rotation, scale_x, inherit_mode):
    """Compose world matrix from parent state and local transform."""
    parent_basis = parent_state["matrix"]
    if inherit_mode == "NoScale":
        parent_basis = parent_state["rigid_matrix"]
    return parent_basis @ _build_2d_basis(local_rotation, scale_x=scale_x)


def _should_start_terminal_chain(record, by_name, children):
    """Determine if a bone should start a terminal chain."""
    if record["parent"] is None:
        return False
    if record["child_count"] > 1:
        return False
    if record["linear_chain_length"] < 2:
        return False
    if record["length_ratio"] > TERMINAL_CHAIN_ROOT_RATIO:
        return False

    parent = by_name[record["parent"]]
    if (
        record["parent_child_count"] <= 1
        and record["parent_length_ratio"] < TERMINAL_CHAIN_PARENT_RATIO
    ):
        return False
    if parent["length_ratio"] <= record["length_ratio"] and record["parent_child_count"] <= 1:
        return False
    return True


def _annotate_bone_topology(records):
    """Attach generic topology metadata to bone records."""
    by_name = {record["name"]: record for record in records}
    children = {record["name"]: [] for record in records}
    for record in records:
        if record["parent"]:
            children[record["parent"]].append(record["name"])

    positive_lengths = sorted(
        record["length"] for record in records if record["length"] > SEGMENT_EPSILON
    )
    median_length = float(np.median(positive_lengths)) if positive_lengths else 1.0
    median_length = max(median_length, SEGMENT_EPSILON)
    best_path_cache = {}

    leaf_cache = {}
    linear_cache = {}

    def leaf_distance(name):
        if name in leaf_cache:
            return leaf_cache[name]
        kids = children[name]
        if not kids:
            leaf_cache[name] = 0
        else:
            leaf_cache[name] = 1 + min(leaf_distance(child) for child in kids)
        return leaf_cache[name]

    def linear_chain_length(name):
        if name in linear_cache:
            return linear_cache[name]
        kids = children[name]
        if len(kids) != 1:
            linear_cache[name] = 1
        else:
            linear_cache[name] = 1 + linear_chain_length(kids[0])
        return linear_cache[name]

    def best_path(name):
        if name in best_path_cache:
            return best_path_cache[name]
        own_length = max(float(by_name[name]["length"]), 0.0)
        kids = children[name]
        if not kids:
            best_path_cache[name] = ([name], own_length)
            return best_path_cache[name]

        best_child_path = []
        best_child_score = -1.0
        for child_name in kids:
            child_path, child_score = best_path(child_name)
            if child_score > best_child_score:
                best_child_path = child_path
                best_child_score = child_score
        best_path_cache[name] = ([name] + best_child_path, own_length + max(best_child_score, 0.0))
        return best_path_cache[name]

    for record in records:
        name = record["name"]
        parent = by_name.get(record["parent"])
        record["child_count"] = len(children[name])
        record["parent_child_count"] = len(children[parent["name"]]) if parent else 0
        record["leaf_distance"] = leaf_distance(name)
        record["linear_chain_length"] = linear_chain_length(name)
        record["length_ratio"] = record["length"] / median_length if median_length else 1.0
        if parent and record["length"] > SEGMENT_EPSILON:
            record["parent_length_ratio"] = parent["length"] / record["length"]
        else:
            record["parent_length_ratio"] = 1.0
        record["main_chain"] = False
        record["terminal_chain"] = False
        record["terminal_chain_root"] = False
        record["terminal_chain_order"] = -1

    roots = [record["name"] for record in records if record["parent"] is None]
    best_root_path = []
    best_root_score = -1.0
    for root_name in roots:
        path, score = best_path(root_name)
        if score > best_root_score:
            best_root_path = path
            best_root_score = score
    main_chain_names = set(best_root_path)
    for record in records:
        record["main_chain"] = record["name"] in main_chain_names

    for record in records:
        if record["terminal_chain"]:
            continue
        if not _should_start_terminal_chain(record, by_name, children):
            continue
        current_name = record["name"]
        order = 0
        while True:
            current = by_name[current_name]
            current["terminal_chain"] = True
            current["terminal_chain_root"] = order == 0
            current["terminal_chain_order"] = order
            kids = children[current_name]
            if len(kids) != 1 or order + 1 >= TERMINAL_CHAIN_MAX_SPAN:
                break
            next_record = by_name[kids[0]]
            if next_record["length_ratio"] > TERMINAL_CHAIN_MAX_LENGTH_RATIO:
                break
            current_name = next_record["name"]
            order += 1

    for record in records:
        record["inherit"] = _default_inherit_mode(record)


def extract_bone_hierarchy(
    armature,
    view_cfg,
    source_frame=None,
    use_rest_pose=False,
    projection_space="world",
    projection_reference_root=None,
):
    """Extract bones in setup pose and project to 2D.

    Returns a list of bone dicts ordered so parents come before children.
    """
    scene = bpy.context.scene
    if source_frame is None:
        source_frame = scene.frame_start
    scene.frame_set(source_frame)
    bpy.context.view_layer.update()
    projection_inverse = get_projection_reference_inverse(
        armature,
        projection_space=projection_space,
        use_rest_pose=use_rest_pose,
        reference_root_matrix=projection_reference_root,
    )

    bone_order = _topological_sort(armature)
    records = []

    for idx, bone_name in enumerate(bone_order):
        pose_bone = armature.pose.bones[bone_name]
        rest_bone = armature.data.bones[bone_name]

        if use_rest_pose:
            head_world = armature.matrix_world @ rest_bone.head_local
            tail_world = armature.matrix_world @ rest_bone.tail_local
        else:
            head_world = armature.matrix_world @ pose_bone.head
            tail_world = armature.matrix_world @ pose_bone.tail

        head_2d = np.array(
            project_point_ortho(head_world, view_cfg, projection_inverse=projection_inverse),
            dtype=np.float64,
        )
        tail_2d = np.array(
            project_point_ortho(tail_world, view_cfg, projection_inverse=projection_inverse),
            dtype=np.float64,
        )
        segment = tail_2d - head_2d
        length = float(np.linalg.norm(segment))
        parent_name = rest_bone.parent.name if rest_bone.parent else None

        records.append(
            {
                "name": bone_name,
                "parent": parent_name,
                "index": idx,
                "head": head_2d,
                "segment": segment,
                "length": length,
                "rotation_world": math.degrees(math.atan2(segment[1], segment[0]))
                if length > SEGMENT_EPSILON
                else 0.0,
                "connected": _bone_is_connected(rest_bone),
            }
        )

    _annotate_bone_topology(records)

    bones = []
    world_cache = {}

    for record in records:
        bone_name = record["name"]
        head_vector = record["head"]
        segment = record["segment"]
        length = record["length"]
        inherit_mode = record["inherit"]
        parent_name = record["parent"]

        if parent_name:
            parent_state = world_cache[parent_name]
            inv_parent = safe_inverse_2x2(parent_state["matrix"])
            local_position = inv_parent @ (head_vector - parent_state["head"])
            if length > SEGMENT_EPSILON:
                world_x_axis = segment / length
            else:
                world_x_axis = np.array((1.0, 0.0), dtype=np.float64)
            local_basis_inverse = _basis_inverse_for_inherit(parent_state, inherit_mode)
            local_x_axis = local_basis_inverse @ world_x_axis
            local_rotation = math.degrees(math.atan2(local_x_axis[1], local_x_axis[0]))
            local_x = float(local_position[0])
            local_y = float(local_position[1])
            world_matrix = _compose_world_matrix(parent_state, local_rotation, 1.0, inherit_mode)
        else:
            local_x = float(head_vector[0])
            local_y = float(head_vector[1])
            local_rotation = record["rotation_world"]
            world_matrix = _build_2d_basis(local_rotation, scale_x=1.0)

        bone = {
            "name": bone_name,
            "parent": parent_name,
            "index": record["index"],
            "x": round(local_x, 4),
            "y": round(local_y, 4),
            "rotation": round(local_rotation, 2),
            "length": round(length, 4),
            "connected": record["connected"],
            "inherit": inherit_mode,
            "child_count": record["child_count"],
            "parent_child_count": record["parent_child_count"],
            "leaf_distance": record["leaf_distance"],
            "linear_chain_length": record["linear_chain_length"],
            "length_ratio": round(record["length_ratio"], 4),
            "parent_length_ratio": round(record["parent_length_ratio"], 4),
            "main_chain": bool(record["main_chain"]),
            "terminal_chain": record["terminal_chain"],
            "terminal_chain_root": record["terminal_chain_root"],
            "terminal_chain_order": record["terminal_chain_order"],
        }
        bones.append(bone)
        world_cache[bone_name] = {
            "head": head_vector,
            "matrix": world_matrix,
            "rigid_matrix": orthonormalize_2x2(world_matrix),
        }

    return bones


def _weights_to_json(weights):
    """Convert weight dicts to JSON-serializable format."""
    serialized = []
    for weight_dict in weights:
        pairs = []
        for bone_index, weight_value in sorted((weight_dict or {}).items()):
            pairs.append([int(bone_index), float(weight_value)])
        serialized.append(pairs)
    return serialized


def extract_bone_hierarchy_3d(armature, source_frame=None, use_rest_pose=False):
    """Extract 3D bone heads/tails in world space for preview/debug rendering."""
    if armature is None:
        return []

    scene = bpy.context.scene
    if source_frame is None:
        source_frame = scene.frame_start
    scene.frame_set(source_frame)
    bpy.context.view_layer.update()

    bones = []
    for idx, bone_name in enumerate(_topological_sort(armature)):
        rest_bone = armature.data.bones[bone_name]
        pose_bone = armature.pose.bones.get(bone_name)
        if use_rest_pose or pose_bone is None:
            head_world = armature.matrix_world @ rest_bone.head_local
            tail_world = armature.matrix_world @ rest_bone.tail_local
        else:
            head_world = armature.matrix_world @ pose_bone.head
            tail_world = armature.matrix_world @ pose_bone.tail
        bones.append(
            {
                "name": bone_name,
                "parent": rest_bone.parent.name if rest_bone.parent else None,
                "index": idx,
                "head": _vector_to_json(head_world),
                "tail": _vector_to_json(tail_world),
                "length": float((tail_world - head_world).length),
            }
        )
    return bones


def _timeline_duration(timeline: dict[str, object]) -> float:
    duration = 0.0
    for key in ("rotate", "translate", "scale", "shear"):
        for record in timeline.get(key) or []:
            duration = max(duration, float(record.get("time", 0.0)))
    return duration


def _animation_duration(animation_payload: dict[str, object]) -> float:
    duration = 0.0
    for timeline in (animation_payload.get("bones") or {}).values():
        duration = max(duration, _timeline_duration(timeline or {}))
    return duration


def _serialize_animations(animations: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    records = []
    for animation_name, payload in animations.items():
        payload = payload or {}
        records.append(
            {
                "name": str(animation_name),
                "duration": round(_animation_duration(payload), 4),
                "bones": dict(payload.get("bones") or {}),
                "frame_filter": dict(payload.get("frame_filter") or {}),
            }
        )
    records.sort(key=lambda record: str(record.get("name", "")).lower())
    return records


def extract_scene_cli(
    source_path: str,
    output_path: str,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
    source_frame: int = None,
    use_rest_pose: bool = False,
    projection_space: str = "world",
    mesh_reduction: bool = True,
    mesh_target_vertices: int = 5000,
) -> dict[str, object]:
    """CLI wrapper for scene extraction (mesh + bones + weights).

    This combines extract_2d_mesh and extract_bone_hierarchy with weight transfer.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    mesh_obj, armature_obj = find_mesh_and_armature()
    if mesh_obj is None:
        return {"ok": False, "detail": "No mesh found in scene"}

    setup_frame = _resolve_setup_frame(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
        neutral_auto_pose=True,
    )
    bpy.context.scene.frame_set(setup_frame)
    setup_pose = _apply_auto_setup_pose(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
    )
    mesh_reduction_report = reduce_mesh_object(
        mesh_obj,
        target_vertices=mesh_target_vertices,
        enabled=mesh_reduction,
    )

    # Build view configuration
    view_cfg = get_view_config(
        view_name=view_preset,
        view_dir=tuple(view_dir) if view_dir is not None else None,
        up_hint=tuple(view_up) if view_up is not None else None,
        roll_degrees=view_roll,
    )

    # Extract bone hierarchy
    bones = []
    if armature_obj is not None:
        bones = extract_bone_hierarchy(
            armature_obj,
            view_cfg,
            source_frame=setup_frame,
            use_rest_pose=use_rest_pose,
            projection_space=projection_space,
            projection_reference_root=None,
        )
    bones_3d = extract_bone_hierarchy_3d(
        armature_obj,
        source_frame=setup_frame,
        use_rest_pose=use_rest_pose,
    )

    # Extract mesh
    mesh_data = extract_2d_mesh(
        mesh_obj,
        view_cfg,
        source_frame=setup_frame,
        use_rest_pose=use_rest_pose,
    )
    mesh_reduction_report["output_vertex_count"] = len(mesh_data.get("vertices_2d") or [])
    mesh_reduction_report["output_triangle_count"] = len(mesh_data.get("triangles") or [])
    if (
        mesh_reduction_report.get("applied")
        and int(mesh_reduction_report.get("target_vertices") or 0) > 0
        and int(mesh_reduction_report["output_vertex_count"])
        > int(mesh_reduction_report["target_vertices"])
        and mesh_reduction_report.get("reason") == "target_reached"
    ):
        mesh_reduction_report["reason"] = "target_reached_before_uv_split"
    mesh_data["mesh_reduction"] = mesh_reduction_report
    base_color_data = extract_base_color_texture(mesh_obj)
    if base_color_data:
        mesh_data["base_color_rgba"] = base_color_data["rgba"]
        mesh_data["base_color_width"] = base_color_data["width"]
        mesh_data["base_color_height"] = base_color_data["height"]
        mesh_data["base_color_channels"] = base_color_data["channels"]

    # Transfer weights
    bone_name_to_index = {
        str(bone["name"]): int(bone["index"]) for bone in bones if bone.get("name") is not None
    }

    # Get weight transfer function if available
    try:
        from flatrig.mesh import transfer_3d_weights_to_2d

        base_source_weights = (
            transfer_3d_weights_to_2d(mesh_obj, bone_name_to_index)
            if bone_name_to_index
            else [{} for _ in range(len(mesh_data.get("vertices_2d") or []))]
        )
        source_indices = mesh_data.get("source_vertex_indices") or list(range(len(base_source_weights)))
        source_weights = [
            base_source_weights[int(source_index)]
            if 0 <= int(source_index) < len(base_source_weights)
            else {}
            for source_index in source_indices
        ]
    except ImportError:
        # Fallback if transfer function not available
        source_weights = [{} for _ in range(len(mesh_data.get("vertices_2d") or []))]

    return {
        "ok": True,
        "detail": "extracted",
        "source": source_path,
        "setup_frame": setup_frame,
        "setup_pose": setup_pose,
        "use_rest_pose": bool(use_rest_pose),
        "mesh": mesh_data,
        "bones": bones,
        "bones_3d": bones_3d,
        "source_weights": _weights_to_json(source_weights),
        "mesh_reduction": mesh_reduction_report,
    }


def extract_animations_cli(
    source_path: str,
    output_path: str,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
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
) -> dict[str, object]:
    """CLI wrapper for animation extraction.

    This extracts animations using the bone hierarchy and animation functions.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    mesh_obj, armature_obj = find_mesh_and_armature()
    if armature_obj is None:
        return {"ok": False, "detail": "No armature found in scene"}

    setup_frame = _resolve_setup_frame(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=False,
    )
    # Build view configuration
    view_cfg = get_view_config(
        view_name=view_preset,
        view_dir=tuple(view_dir) if view_dir is not None else None,
        up_hint=tuple(view_up) if view_up is not None else None,
        roll_degrees=view_roll,
    )

    # Extract bone hierarchy
    bones = extract_bone_hierarchy(
        armature_obj,
        view_cfg,
        source_frame=setup_frame,
        projection_space=projection_space,
        projection_reference_root=None,
    )

    # Get animation extraction function if available
    try:
        from flatrig.animation import extract_bone_animations

        animations = extract_bone_animations(
            armature_obj,
            bones,
            view_cfg,
            fps=fps,
            frame_start=frame_start,
            frame_end=frame_end,
            sample_substeps=sample_substeps,
            optimize_animation_keys=optimize_animation_keys,
            force_loop_closing_keys=force_loop_closing_keys,
            projection_space=projection_space,
            pose_mode=pose_mode,
            pose_blend=pose_blend,
            rotation_flatten={"amount": rotation_flatten, "scope": rotation_flatten_scope}
            if rotation_flatten > 0
            else None,
            stretch_guard={
                "enabled": True,
                "max_scale": stretch_guard_max_scale,
                "strength": stretch_guard_strength,
                "bones": "all",
            }
            if stretch_guard_enabled
            else None,
            leaf_ik_refine={
                "enabled": True,
                "strength": ik_leaf_strength,
                "iterations": ik_leaf_iterations,
                "max_chain_length": ik_leaf_max_chain_length,
                "preserve_scale": ik_leaf_preserve_scale,
            }
            if ik_leaf_refine_enabled
            else None,
            problem_frame_filter={"enabled": True} if drop_problematic_frames else None,
            projection_reference_root=None,
            preserve_root_motion=preserve_root_motion,
            preserve_root_rotation=preserve_root_rotation,
            action_names=animation_names or [],
        )
    except ImportError as e:
        return {"ok": False, "detail": f"Animation extraction not available: {e}"}

    return {
        "ok": True,
        "detail": "extracted",
        "source": source_path,
        "setup_frame": setup_frame,
        "bones": bones,
        "animations": _serialize_animations(animations),
    }


def render_sprites_cli(
    source_path: str,
    output_path: str,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
    source_frame: int = None,
    use_rest_pose: bool = False,
    projection_space: str = "world",
    parts_json: str = None,
    images_dir: str = None,
    resolution: int = 2048,
    bind_frame: int = 0,
    mesh_reduction: bool = True,
    mesh_target_vertices: int = 5000,
) -> dict[str, object]:
    """CLI wrapper for sprite rendering.

    This renders part sprites using the projection and sprite functions.
    """
    import json as json_module

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    mesh_obj, armature_obj = find_mesh_and_armature()
    if mesh_obj is None:
        return {"ok": False, "detail": "No mesh found in scene"}

    mesh_reduction_report = reduce_mesh_object(
        mesh_obj,
        target_vertices=mesh_target_vertices,
        enabled=mesh_reduction,
    )

    if not parts_json or not images_dir:
        return {"ok": False, "detail": "parts-json and images-dir are required"}

    parts_path = Path(parts_json).expanduser().resolve()
    output_dir = Path(images_dir).expanduser().resolve()

    if not parts_path.exists():
        return {"ok": False, "detail": f"parts-json not found: {parts_path}"}

    parts = json_module.loads(parts_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if bind_frame > 0:
        render_frame = int(bind_frame)
        setup_pose = {"mode": "frame", "posed_bone_count": 0}
    else:
        render_frame = _resolve_setup_frame(
            armature_obj,
            source_frame=(source_frame if bind_frame == 0 else -1),
            use_rest_pose=use_rest_pose,
            neutral_auto_pose=True,
        )
        setup_pose = None
    bpy.context.scene.frame_set(render_frame)
    if setup_pose is None:
        setup_pose = _apply_auto_setup_pose(
            armature_obj,
            source_frame=(source_frame if bind_frame == 0 else -1),
            use_rest_pose=use_rest_pose,
        )
    bpy.context.view_layer.update()

    # Build view configuration
    view_cfg = get_view_config(
        view_name=view_preset,
        view_dir=tuple(view_dir) if view_dir is not None else None,
        up_hint=tuple(view_up) if view_up is not None else None,
        roll_degrees=view_roll,
    )

    # Get projection reference
    projection_reference_root = None
    if armature_obj:
        world_matrix = armature_obj.matrix_world
        projection_reference_root = [
            [world_matrix[0][0], world_matrix[0][1], world_matrix[0][2], world_matrix[0][3]],
            [world_matrix[1][0], world_matrix[1][1], world_matrix[1][2], world_matrix[1][3]],
            [world_matrix[2][0], world_matrix[2][1], world_matrix[2][2], world_matrix[2][3]],
            [world_matrix[3][0], world_matrix[3][1], world_matrix[3][2], world_matrix[3][3]],
        ]

    # Get render_part_sprite function if available
    try:
        from flatrig.projection import get_projection_reference_matrix
        from flatrig.texture import render_part_sprite

        projection_matrix = (
            get_projection_reference_matrix(
                armature_obj,
                projection_space=projection_space,
                use_rest_pose=use_rest_pose,
                reference_root_matrix=projection_reference_root,
            )
            if armature_obj
            else None
        )

        renders = []
        for part in parts:
            attachment_name = str(part.get("attachment_name") or part.get("name") or "part")
            part_output = output_dir / f"{attachment_name}.png"
            ok = render_part_sprite(
                mesh_obj,
                view_cfg,
                [tuple(key) for key in (part.get("triangle_keys") or [])],
                dict(part.get("projection_frame") or {}),
                str(part_output),
                resolution=resolution,
                depth_center=float(part.get("mean_depth", 0.0) or 0.0),
                bind_frame=render_frame,
                use_rest_pose=use_rest_pose,
                projection_matrix=projection_matrix,
            )
            renders.append(
                {
                    "name": str(part.get("name") or attachment_name),
                    "attachment_name": attachment_name,
                    "ok": bool(ok),
                    "detail": "rendered" if ok else "render failed",
                    "output_path": str(part_output),
                    "width": resolution,
                    "height": resolution,
                }
            )

        return {
            "ok": True,
            "detail": "rendered",
            "source": source_path,
            "render_frame": render_frame,
            "setup_pose": setup_pose,
            "use_rest_pose": bool(use_rest_pose),
            "mesh_reduction": mesh_reduction_report,
            "renders": renders,
        }
    except ImportError as e:
        return {"ok": False, "detail": f"Sprite rendering not available: {e}"}


def main() -> None:
    args = parse_args()
    source_path = str(Path(args.source).expanduser().resolve())
    output_path = Path(args.output).expanduser().resolve()

    payload: dict[str, object]

    if args.command == "inspect":
        payload = inspect_source(source_path)
    elif args.command == "convert":
        payload = convert_source(source_path, str(output_path))
    elif args.command == "extract-scene":
        view_dir = None
        view_up = None
        if args.view_dir:
            view_dir = tuple(float(x) for x in args.view_dir.split(","))
        if args.view_up:
            view_up = tuple(float(x) for x in args.view_up.split(","))

        payload = extract_scene_cli(
            source_path,
            str(output_path),
            view_preset=args.view_preset,
            view_dir=view_dir,
            view_up=view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            use_rest_pose=args.use_rest_pose,
            projection_space=args.projection_space,
            mesh_reduction=args.mesh_reduction,
            mesh_target_vertices=args.mesh_target_vertices,
        )
    elif args.command == "extract-animations":
        view_dir = None
        view_up = None
        if args.view_dir:
            view_dir = tuple(float(x) for x in args.view_dir.split(","))
        if args.view_up:
            view_up = tuple(float(x) for x in args.view_up.split(","))

        payload = extract_animations_cli(
            source_path,
            str(output_path),
            view_preset=args.view_preset,
            view_dir=view_dir,
            view_up=view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            projection_space=args.projection_space,
            animation_names=args.animation_names,
            fps=args.fps,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            sample_substeps=args.sample_substeps,
            optimize_animation_keys=args.optimize_animation_keys,
            force_loop_closing_keys=args.force_loop_closing_keys,
            pose_mode=args.pose_mode,
            pose_blend=args.pose_blend,
            rotation_flatten=args.rotation_flatten,
            rotation_flatten_scope=args.rotation_flatten_scope,
            stretch_guard_enabled=args.stretch_guard_enabled,
            stretch_guard_max_scale=args.stretch_guard_max_scale,
            stretch_guard_strength=args.stretch_guard_strength,
            ik_leaf_refine_enabled=args.ik_leaf_refine_enabled,
            ik_leaf_strength=args.ik_leaf_strength,
            ik_leaf_iterations=args.ik_leaf_iterations,
            ik_leaf_max_chain_length=args.ik_leaf_max_chain_length,
            ik_leaf_preserve_scale=args.ik_leaf_preserve_scale,
            drop_problematic_frames=args.drop_problematic_frames,
            preserve_root_motion=args.preserve_root_motion,
            preserve_root_rotation=args.preserve_root_rotation,
        )
    elif args.command == "export-3d-animation-bvh":
        if not args.bvh_output:
            raise ValueError("--bvh-output is required for export-3d-animation-bvh")
        payload = export_3d_animation_bvh_cli(
            source_path,
            str(output_path),
            bvh_output=args.bvh_output,
            animation_names=args.animation_names,
            fps=args.fps,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
    elif args.command == "export-3d-rest-bvh":
        if not args.bvh_output:
            raise ValueError("--bvh-output is required for export-3d-rest-bvh")
        view_dir = None
        view_up = None
        if args.view_dir:
            view_dir = tuple(float(x) for x in args.view_dir.split(","))
        if args.view_up:
            view_up = tuple(float(x) for x in args.view_up.split(","))

        payload = export_3d_rest_bvh_cli(
            source_path,
            str(output_path),
            bvh_output=args.bvh_output,
            view_preset=args.view_preset,
            view_dir=view_dir,
            view_up=view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            use_rest_pose=args.use_rest_pose,
            projection_space=args.projection_space,
            fps=args.fps,
            frame_count=args.frame_count,
        )
    elif args.command == "render-sprites":
        view_dir = None
        view_up = None
        if args.view_dir:
            view_dir = tuple(float(x) for x in args.view_dir.split(","))
        if args.view_up:
            view_up = tuple(float(x) for x in args.view_up.split(","))

        payload = render_sprites_cli(
            source_path,
            str(output_path),
            view_preset=args.view_preset,
            view_dir=view_dir,
            view_up=view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            use_rest_pose=args.use_rest_pose,
            projection_space=args.projection_space,
            parts_json=args.parts_json,
            images_dir=args.images_dir,
            resolution=args.resolution,
            bind_frame=args.bind_frame,
            mesh_reduction=args.mesh_reduction,
            mesh_target_vertices=args.mesh_target_vertices,
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - Blender runtime
        import traceback

        payload = {"ok": False, "detail": str(exc), "traceback": traceback.format_exc()}
        try:
            args = parse_args()
            output_path = Path(args.output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        raise SystemExit(1) from exc
