"""Blender worker for scene inspection and normalization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
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


def extract_2d_mesh(
    mesh_obj,
    view_cfg,
    projection_frame=None,
    source_frame=None,
    projection_inverse=None,
):
    """Extract the bind-pose mesh projected to 2D."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    if source_frame is None:
        source_frame = scene.frame_start
    scene.frame_set(source_frame)
    depsgraph.update()

    eval_obj = mesh_obj.evaluated_get(depsgraph)
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
    uvs = project_points_to_uv(vertices_2d, projection_frame)

    # Compute depths
    depths = np.array(
        [
            point_depth(vertices_3d[i], view_cfg, projection_inverse=projection_inverse)
            for i in range(len(vertices_3d))
        ],
        dtype=np.float64,
    )

    # Triangulate mesh
    bm = bmesh.new()
    bm.from_mesh(eval_mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    triangles = []
    triangle_keys = []
    for face in bm.faces:
        tri = [vert.index for vert in face.verts]
        triangles.append(tri)
        triangle_keys.append(tuple(sorted(tri)))
    bm.free()
    eval_obj.to_mesh_clear()

    # Extract vertex groups (weights)
    vertex_groups = {}
    group_names = {group.index: group.name for group in mesh_obj.vertex_groups}

    for vert in mesh_obj.data.vertices:
        for group in vert.groups:
            group_name = group_names.get(group.group, f"group_{group.group}")
            vertex_groups.setdefault(group_name, []).append((vert.index, group.weight))

    return {
        "vertices_2d": vertices_2d.tolist(),
        "vertices_3d": vertices_3d.tolist(),
        "triangles": triangles,
        "triangle_keys": triangle_keys,
        "uvs": uvs.tolist(),
        "depths": depths.tolist(),
        "vertex_groups": vertex_groups,
        "projection_frame": projection_frame,
    }


def parse_args() -> argparse.Namespace:
    try:
        separator_index = sys.argv.index("--")
        script_args = sys.argv[separator_index + 1 :]
    except ValueError:
        script_args = []

    parser = argparse.ArgumentParser(description="Run the public 3D scene worker.")
    parser.add_argument(
        "command",
        choices=("inspect", "convert", "extract-scene", "extract-animations", "render-sprites"),
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
    projection_space: str = "world",
) -> dict[str, object]:
    """CLI wrapper for scene extraction (mesh + bones + weights).

    This combines extract_2d_mesh and extract_bone_hierarchy with weight transfer.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    mesh_obj, armature_obj = find_mesh_and_armature()
    if mesh_obj is None:
        return {"ok": False, "detail": "No mesh found in scene"}

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
            source_frame=source_frame,
            projection_space=projection_space,
            projection_reference_root=None,
        )

    # Extract mesh
    mesh_data = extract_2d_mesh(
        mesh_obj,
        view_cfg,
        source_frame=source_frame,
    )

    # Transfer weights
    bone_name_to_index = {
        str(bone["name"]): int(bone["index"]) for bone in bones if bone.get("name") is not None
    }

    # Get weight transfer function if available
    try:
        from flatrig.mesh import transfer_3d_weights_to_2d

        source_weights = (
            transfer_3d_weights_to_2d(mesh_obj, bone_name_to_index)
            if bone_name_to_index
            else [{} for _ in range(len(mesh_data.get("vertices_2d") or []))]
        )
    except ImportError:
        # Fallback if transfer function not available
        source_weights = [{} for _ in range(len(mesh_data.get("vertices_2d") or []))]

    return {
        "ok": True,
        "detail": "extracted",
        "source": source_path,
        "mesh": mesh_data,
        "bones": bones,
        "source_weights": _weights_to_json(source_weights),
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
        source_frame=source_frame,
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
    projection_space: str = "world",
    parts_json: str = None,
    images_dir: str = None,
    resolution: int = 2048,
    bind_frame: int = 0,
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

    if not parts_json or not images_dir:
        return {"ok": False, "detail": "parts-json and images-dir are required"}

    parts_path = Path(parts_json).expanduser().resolve()
    output_dir = Path(images_dir).expanduser().resolve()

    if not parts_path.exists():
        return {"ok": False, "detail": f"parts-json not found: {parts_path}"}

    parts = json_module.loads(parts_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

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
                bind_frame=bind_frame if bind_frame > 0 else source_frame or 0,
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
            projection_space=args.projection_space,
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
            projection_space=args.projection_space,
            parts_json=args.parts_json,
            images_dir=args.images_dir,
            resolution=args.resolution,
            bind_frame=args.bind_frame,
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
