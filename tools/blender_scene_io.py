"""Blender worker for scene inspection and normalization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

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
VISIBILITY_RASTER_EPSILON = 1e-6

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
    up_hint_vec = _normalize_vector(
        up_hint if up_hint is not None else WORLD_UP, label="up_hint"
    )
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
                up_hint=np.array(preset["up_hint"], dtype=np.float64) if "up_hint" in preset else None,
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
    """Project a 3D point into the configured 2D plane.
    """
    projected = _transform_point_to_projection_space(
        point_3d,
        projection_inverse=projection_inverse,
    )
    projected_2d = _project_projection_space_direction(projected, view_cfg)
    return (float(projected_2d[0]), float(projected_2d[1]))


def project_points_ortho(points_3d, view_cfg, projection_inverse=None):
    """Vectorized orthographic projection for an array of 3D points."""
    projected = transform_points_to_projection_space(
        points_3d,
        projection_inverse=projection_inverse,
    )
    right_axis = np.asarray(view_cfg["right_axis"], dtype=np.float64)
    up_axis = np.asarray(view_cfg["up_axis"], dtype=np.float64)
    return np.stack((projected @ right_axis, projected @ up_axis), axis=1)


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


def transform_points_to_projection_space(points_3d, projection_inverse=None):
    """Transform world-space points into projection space."""
    return _transform_point_to_projection_space(
        points_3d,
        projection_inverse=projection_inverse,
    )


def transform_point_from_projection_space(point_3d, projection_matrix=None):
    """Transform a point from projection-local space back to world space."""
    point_3d = np.asarray(point_3d, dtype=np.float64)
    if projection_matrix is None:
        return point_3d

    point_h = np.concatenate((point_3d, np.array((1.0,), dtype=np.float64)))
    return (np.asarray(projection_matrix, dtype=np.float64) @ point_h)[:3]


def transform_direction_to_projection_space(direction_3d, projection_inverse=None):
    """Transform a direction vector into projection-local space."""
    direction_3d = np.asarray(direction_3d, dtype=np.float64)
    if projection_inverse is None:
        return direction_3d
    rotation = np.asarray(projection_inverse, dtype=np.float64)[:3, :3]
    return rotation @ direction_3d


def transform_direction_from_projection_space(direction_3d, projection_matrix=None):
    """Transform a direction vector from projection-local space to world space."""
    direction_3d = np.asarray(direction_3d, dtype=np.float64)
    if projection_matrix is None:
        return direction_3d
    rotation = np.asarray(projection_matrix, dtype=np.float64)[:3, :3]
    return rotation @ direction_3d


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


def compose_projection_plane_point(x, y, depth, view_cfg):
    """Build a 3D point in projection space from 2D screen coordinates plus depth."""
    return (
        np.asarray(view_cfg["right_axis"], dtype=np.float64) * float(x)
        + np.asarray(view_cfg["up_axis"], dtype=np.float64) * float(y)
        + np.asarray(view_cfg["depth_axis"], dtype=np.float64) * float(depth)
    )


def get_evaluated_mesh(obj, depsgraph):
    """Get the evaluated deformed mesh after Blender modifiers."""
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    return eval_obj, mesh


def get_evaluated_vertex_positions(obj, depsgraph):
    """Get world-space vertex positions after armature deformation."""
    eval_obj, mesh = get_evaluated_mesh(obj, depsgraph)
    world_mat = eval_obj.matrix_world

    positions = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    for index, vert in enumerate(mesh.vertices):
        world_co = world_mat @ vert.co
        positions[index] = (world_co.x, world_co.y, world_co.z)

    eval_obj.to_mesh_clear()
    return positions


def ensure_polygon_normals(mesh):
    """Populate polygon normals when running on Blender versions that need it."""
    if hasattr(mesh, "calc_normals"):
        mesh.calc_normals()


def compute_vertex_visibility(obj, depsgraph, view_cfg, projection_inverse=None):
    """Cheap visibility estimate based on front-facing polygons."""
    eval_obj, mesh = get_evaluated_mesh(obj, depsgraph)
    world_mat = eval_obj.matrix_world
    view_dir = mathutils.Vector(np.asarray(view_cfg["view_dir"], dtype=np.float64))

    visible = np.zeros(len(mesh.vertices), dtype=bool)
    ensure_polygon_normals(mesh)

    for poly in mesh.polygons:
        world_normal = (world_mat.to_3x3() @ poly.normal).normalized()
        if projection_inverse is not None:
            world_normal = mathutils.Vector(
                transform_direction_to_projection_space(world_normal, projection_inverse)
            ).normalized()
        if world_normal.dot(view_dir) < 0:
            for vertex_index in poly.vertices:
                visible[vertex_index] = True

    eval_obj.to_mesh_clear()
    return visible


def compute_front_facing_vertex_visibility_from_triangles(
    positions_projection_space,
    triangles,
    view_cfg,
):
    """Mark vertices that belong to any front-facing triangle."""
    projected_positions = np.asarray(positions_projection_space, dtype=np.float64)
    triangle_array = np.asarray(triangles, dtype=np.int32)
    visible = np.zeros(projected_positions.shape[0], dtype=bool)
    if triangle_array.size == 0:
        return visible

    p0 = projected_positions[triangle_array[:, 0]]
    p1 = projected_positions[triangle_array[:, 1]]
    p2 = projected_positions[triangle_array[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    normal_lengths = np.linalg.norm(normals, axis=1)
    view_dir = np.asarray(view_cfg["view_dir"], dtype=np.float64)
    facing = (normal_lengths > VISIBILITY_RASTER_EPSILON) & (
        np.einsum("ij,j->i", normals, view_dir) < -VISIBILITY_RASTER_EPSILON
    )
    if np.any(facing):
        visible[np.unique(triangle_array[facing].reshape(-1))] = True
    return visible


def compute_surface_vertex_visibility_from_triangles(
    positions_projection_space,
    positions_2d,
    triangles,
    view_cfg,
    *,
    triangle_groups=None,
    raster_size=256,
):
    """Approximate visible-surface vertices with a small orthographic z-buffer."""
    projected_positions = np.asarray(positions_projection_space, dtype=np.float64)
    projected_points_2d = np.asarray(positions_2d, dtype=np.float64)
    triangle_array = np.asarray(triangles, dtype=np.int32)
    raster_size = max(16, int(raster_size))
    visible = np.zeros(projected_positions.shape[0], dtype=bool)
    if triangle_array.size == 0 or projected_points_2d.size == 0:
        return visible

    front_facing = compute_front_facing_vertex_visibility_from_triangles(
        projected_positions,
        triangles,
        view_cfg,
    )
    if not np.any(front_facing):
        return visible

    p0 = projected_positions[triangle_array[:, 0]]
    p1 = projected_positions[triangle_array[:, 1]]
    p2 = projected_positions[triangle_array[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    normal_lengths = np.linalg.norm(normals, axis=1)
    view_dir = np.asarray(view_cfg["view_dir"], dtype=np.float64)
    front_triangle_mask = (normal_lengths > VISIBILITY_RASTER_EPSILON) & (
        np.einsum("ij,j->i", normals, view_dir) < -VISIBILITY_RASTER_EPSILON
    )
    if not np.any(front_triangle_mask):
        return visible

    depth_axis = np.asarray(view_cfg["depth_axis"], dtype=np.float64)
    depths = projected_positions @ depth_axis
    frame = compute_projection_frame(projected_points_2d, margin=0.02)
    span = max(float(frame["span"]), VISIBILITY_RASTER_EPSILON)
    raster_points = np.empty_like(projected_points_2d, dtype=np.float64)
    raster_points[:, 0] = (
        (projected_points_2d[:, 0] - float(frame["min_x"])) / span * (raster_size - 1)
    )
    raster_points[:, 1] = (
        (projected_points_2d[:, 1] - float(frame["min_y"])) / span * (raster_size - 1)
    )

    if triangle_groups:
        groups = [np.asarray(group, dtype=np.int32) for group in triangle_groups if len(group) > 0]
    else:
        groups = [np.nonzero(front_triangle_mask)[0].astype(np.int32)]

    for group in groups:
        visible_indices = _raster_visible_triangle_indices(
            raster_points,
            depths,
            triangle_array,
            group[front_triangle_mask[group]],
            raster_size=raster_size,
        )
        if visible_indices.size <= 0:
            continue
        visible[np.unique(triangle_array[visible_indices].reshape(-1))] = True

    return visible


def _raster_visible_triangle_indices(
    raster_points,
    depths,
    triangle_array,
    triangle_indices,
    *,
    raster_size,
):
    triangle_indices = np.asarray(triangle_indices, dtype=np.int32)
    if triangle_indices.size <= 0:
        return np.empty(0, dtype=np.int32)

    depth_buffer = np.full((raster_size, raster_size), -np.inf, dtype=np.float64)
    triangle_buffer = np.full((raster_size, raster_size), -1, dtype=np.int32)

    for triangle_index in triangle_indices:
        i0, i1, i2 = triangle_array[int(triangle_index)]
        x0, y0 = raster_points[i0]
        x1, y1 = raster_points[i1]
        x2, y2 = raster_points[i2]
        z0 = float(depths[i0])
        z1 = float(depths[i1])
        z2 = float(depths[i2])

        min_x = max(0, int(math.floor(min(x0, x1, x2))))
        max_x = min(raster_size - 1, int(math.ceil(max(x0, x1, x2))))
        min_y = max(0, int(math.floor(min(y0, y1, y2))))
        max_y = min(raster_size - 1, int(math.ceil(max(y0, y1, y2))))
        if min_x > max_x or min_y > max_y:
            continue

        denominator = ((y1 - y2) * (x0 - x2)) + ((x2 - x1) * (y0 - y2))
        if abs(denominator) <= VISIBILITY_RASTER_EPSILON:
            _mark_triangle_centroid_visibility(
                triangle_buffer,
                depth_buffer,
                triangle_index=int(triangle_index),
                points=((x0, y0), (x1, y1), (x2, y2)),
                depths=(z0, z1, z2),
            )
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        w0 = (((y1 - y2) * (grid_x - x2)) + ((x2 - x1) * (grid_y - y2))) / denominator
        w1 = (((y2 - y0) * (grid_x - x2)) + ((x0 - x2) * (grid_y - y2))) / denominator
        w2 = 1.0 - w0 - w1
        inside = (
            (w0 >= -VISIBILITY_RASTER_EPSILON)
            & (w1 >= -VISIBILITY_RASTER_EPSILON)
            & (w2 >= -VISIBILITY_RASTER_EPSILON)
        )

        if not np.any(inside):
            _mark_triangle_centroid_visibility(
                triangle_buffer,
                depth_buffer,
                triangle_index=int(triangle_index),
                points=((x0, y0), (x1, y1), (x2, y2)),
                depths=(z0, z1, z2),
            )
            continue

        patch_depth = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
        patch_triangles = triangle_buffer[min_y : max_y + 1, min_x : max_x + 1]
        depth_values = (w0 * z0) + (w1 * z1) + (w2 * z2)
        update = inside & (depth_values > patch_depth + VISIBILITY_RASTER_EPSILON)
        if np.any(update):
            patch_depth[update] = depth_values[update]
            patch_triangles[update] = int(triangle_index)

    return np.unique(triangle_buffer[triangle_buffer >= 0])


def _mark_triangle_centroid_visibility(
    triangle_buffer,
    depth_buffer,
    *,
    triangle_index,
    points,
    depths,
):
    centroid_x = int(round(sum(point[0] for point in points) / 3.0))
    centroid_y = int(round(sum(point[1] for point in points) / 3.0))
    if (
        centroid_x < 0
        or centroid_y < 0
        or centroid_y >= depth_buffer.shape[0]
        or centroid_x >= depth_buffer.shape[1]
    ):
        return
    centroid_depth = float(sum(depths) / 3.0)
    if centroid_depth > depth_buffer[centroid_y, centroid_x] + VISIBILITY_RASTER_EPSILON:
        depth_buffer[centroid_y, centroid_x] = centroid_depth
        triangle_buffer[centroid_y, centroid_x] = int(triangle_index)


def build_visibility_map(
    obj,
    frame_start,
    frame_end,
    view_cfg,
    frame_step=1,
    projection_space="world",
    armature_obj=None,
    projection_reference_root=None,
    triangles=None,
    visibility_mode="all",
    visibility_triangle_groups=None,
    visibility_raster_size=256,
):
    """Build a per-vertex, per-frame visibility and position map in Blender."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frame_step = max(1, int(frame_step))
    frames = list(range(frame_start, frame_end + 1, frame_step))
    if frames[-1] != frame_end:
        frames.append(frame_end)

    scene.frame_set(frames[0])
    depsgraph.update()
    positions_3d_0 = get_evaluated_vertex_positions(obj, depsgraph)
    num_frames = len(frames)
    num_verts = positions_3d_0.shape[0]

    projection_inverses = None
    if projection_space == "root":
        projection_inverses = np.zeros((num_frames, 4, 4), dtype=np.float64)

    visibility = np.zeros((num_frames, num_verts), dtype=bool)
    positions_3d = np.zeros((num_frames, num_verts, 3), dtype=np.float64)
    positions_2d = np.zeros((num_frames, num_verts, 2), dtype=np.float64)

    for frame_index, frame in enumerate(frames):
        scene.frame_set(frame)
        depsgraph.update()
        projection_inverse = get_projection_reference_inverse(
            armature_obj,
            projection_space=projection_space,
            reference_root_matrix=projection_reference_root,
        )
        if projection_inverses is not None and projection_inverse is not None:
            projection_inverses[frame_index] = projection_inverse

        frame_positions_3d = get_evaluated_vertex_positions(obj, depsgraph)
        positions_3d[frame_index] = frame_positions_3d
        projection_space_positions = transform_points_to_projection_space(
            frame_positions_3d,
            projection_inverse=projection_inverse,
        )
        frame_positions_2d = np.stack(
            (
                projection_space_positions @ np.asarray(view_cfg["right_axis"], dtype=np.float64),
                projection_space_positions @ np.asarray(view_cfg["up_axis"], dtype=np.float64),
            ),
            axis=1,
        )
        positions_2d[frame_index] = frame_positions_2d

        resolved_mode = str(visibility_mode or "all").strip().lower()
        if resolved_mode == "all":
            visibility[frame_index] = True
        elif resolved_mode == "mesh_surface":
            visibility[frame_index] = compute_surface_vertex_visibility_from_triangles(
                projection_space_positions,
                frame_positions_2d,
                triangles if triangles is not None else [],
                view_cfg,
                raster_size=visibility_raster_size,
            )
        elif resolved_mode == "sprite_surface":
            visibility[frame_index] = compute_surface_vertex_visibility_from_triangles(
                projection_space_positions,
                frame_positions_2d,
                triangles if triangles is not None else [],
                view_cfg,
                triangle_groups=visibility_triangle_groups,
                raster_size=visibility_raster_size,
            )
        elif triangles is not None:
            visibility[frame_index] = compute_front_facing_vertex_visibility_from_triangles(
                projection_space_positions,
                triangles,
                view_cfg,
            )
        else:
            visibility[frame_index] = compute_vertex_visibility(
                obj,
                depsgraph,
                view_cfg,
                projection_inverse=projection_inverse,
            )

    return {
        "frames": frames,
        "visible": visibility,
        "positions_2d": positions_2d,
        "positions_3d": positions_3d,
        "projection_inverses": projection_inverses,
    }


def get_bind_pose_2d(obj, view_cfg, projection_inverse=None):
    """Get the bind-pose vertex positions projected to 2D."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene.frame_set(scene.frame_start)
    depsgraph.update()

    positions_3d = get_evaluated_vertex_positions(obj, depsgraph)
    return project_points_ortho(positions_3d, view_cfg, projection_inverse=projection_inverse)


def _pick_render_engine(scene):
    engine_property = scene.render.bl_rna.properties.get("engine")
    available = {
        item.identifier
        for item in (engine_property.enum_items if engine_property is not None else [])
    }
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH", "CYCLES"):
        if candidate in available:
            return candidate
    return scene.render.engine


def _frame_to_world_center(projection_frame, view_cfg, depth_center=0.0, projection_matrix=None):
    coords = compose_projection_plane_point(
        projection_frame["center_x"],
        projection_frame["center_y"],
        depth_center,
        view_cfg,
    )
    world_coords = transform_point_from_projection_space(
        coords,
        projection_matrix=projection_matrix,
    )
    return mathutils.Vector(np.asarray(world_coords, dtype=np.float64))


def setup_orthographic_camera(
    view_cfg,
    projection_frame,
    depth_center=0.0,
    distance=10.0,
    camera_name="flatRig_Camera",
    projection_matrix=None,
):
    """Create an orthographic camera that matches a projection frame."""
    center = _frame_to_world_center(
        projection_frame,
        view_cfg,
        depth_center=depth_center,
        projection_matrix=projection_matrix,
    )
    right_axis = mathutils.Vector(
        transform_direction_from_projection_space(
            view_cfg["right_axis"],
            projection_matrix=projection_matrix,
        )
    ).normalized()
    up_axis = mathutils.Vector(
        transform_direction_from_projection_space(
            view_cfg["up_axis"],
            projection_matrix=projection_matrix,
        )
    ).normalized()
    view_dir = mathutils.Vector(
        transform_direction_from_projection_space(
            view_cfg["view_dir"],
            projection_matrix=projection_matrix,
        )
    ).normalized()
    camera_location = center - view_dir * distance

    z_axis = (-view_dir).normalized()
    camera_matrix = mathutils.Matrix(
        (
            (right_axis.x, up_axis.x, z_axis.x, camera_location.x),
            (right_axis.y, up_axis.y, z_axis.y, camera_location.y),
            (right_axis.z, up_axis.z, z_axis.z, camera_location.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.name = camera_name
    camera.matrix_world = camera_matrix
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = projection_frame["span"]
    bpy.context.scene.camera = camera
    return camera


def render_projected_sprite(scene, output_path, resolution=2048):
    """Render the current scene from its active orthographic camera."""
    print(f"[flatrig_texture] Rendering sprite at {resolution}x{resolution}...")

    scene.render.engine = _pick_render_engine(scene)
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = output_path

    try:
        bpy.ops.render.render(write_still=True)
        print(f"[flatrig_texture] Sprite rendered to: {output_path}")
        return True
    except Exception as exc:
        print(f"[flatrig_texture] Render failed: {exc}")
        _create_placeholder_atlas(output_path, resolution)
        return True


def _build_unlit_material(original_material):
    """Create a temporary emission-only copy that preserves base-color textures."""
    material = original_material.copy()
    material.use_nodes = True
    if hasattr(material, "use_backface_culling"):
        material.use_backface_culling = True
    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = False
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output_node = next(
        (node for node in nodes if node.bl_idname == "ShaderNodeOutputMaterial"),
        None,
    )
    principled_node = next(
        (node for node in nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"),
        None,
    )

    if output_node is None:
        output_node = nodes.new("ShaderNodeOutputMaterial")

    emission_node = nodes.new("ShaderNodeEmission")
    emission_node.name = "_flatrig_emission"
    emission_node.inputs["Strength"].default_value = 1.0

    if principled_node is not None:
        base_input = principled_node.inputs["Base Color"]
        if base_input.links:
            source_socket = base_input.links[0].from_socket
            links.new(source_socket, emission_node.inputs["Color"])
        else:
            emission_node.inputs["Color"].default_value = base_input.default_value
    else:
        emission_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    for link in list(output_node.inputs["Surface"].links):
        links.remove(link)
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
    return material


def _apply_unlit_materials(objects):
    """Swap materials to temporary unlit copies and return a restore token."""
    restore_info = []
    created_materials = []

    for obj in objects:
        if obj.type != "MESH":
            continue

        original_materials = list(obj.data.materials)
        replacement_materials = []
        for material in original_materials:
            if material is None:
                replacement_materials.append(None)
                continue
            unlit_material = _build_unlit_material(material)
            replacement_materials.append(unlit_material)
            created_materials.append(unlit_material)

        obj.data.materials.clear()
        for material in replacement_materials:
            obj.data.materials.append(material)
        restore_info.append((obj, original_materials))

    return restore_info, created_materials


def _restore_materials(restore_info, created_materials):
    """Restore original materials and delete temporary unlit copies."""
    for obj, original_materials in restore_info:
        obj.data.materials.clear()
        for material in original_materials:
            obj.data.materials.append(material)

    for material in created_materials:
        bpy.data.materials.remove(material, do_unlink=True)


def render_preview_sprite(
    obj,
    view_cfg,
    projection_frame,
    output_path,
    resolution=2048,
    depth_center=0.0,
    bind_frame=None,
    projection_matrix=None,
):
    """Render a full-body preview that matches the exported projection."""
    scene = bpy.context.scene
    if bind_frame is not None:
        scene.frame_set(bind_frame)
        bpy.context.view_layer.update()
    armatures = [scene_obj for scene_obj in scene.objects if scene_obj.type == "ARMATURE"]
    mesh_objects = [scene_obj for scene_obj in scene.objects if scene_obj.type == "MESH"]
    hidden_armatures = []

    for armature in armatures:
        hidden_armatures.append((armature, armature.hide_render))
        armature.hide_render = True

    camera = setup_orthographic_camera(
        view_cfg,
        projection_frame,
        depth_center=depth_center,
        camera_name="flatRig_PreviewCamera",
        projection_matrix=projection_matrix,
    )
    restore_info, created_materials = _apply_unlit_materials(mesh_objects)

    try:
        return render_projected_sprite(scene, output_path, resolution=resolution)
    finally:
        _restore_materials(restore_info, created_materials)
        bpy.data.objects.remove(camera, do_unlink=True)
        for armature, previous_state in hidden_armatures:
            armature.hide_render = previous_state


def render_part_sprite(
    source_obj,
    view_cfg,
    triangle_keys,
    projection_frame,
    output_path,
    resolution=1024,
    depth_center=0.0,
    bind_frame=None,
    projection_matrix=None,
):
    """Render a cropped sprite for one body part."""
    scene = bpy.context.scene
    if bind_frame is not None:
        scene.frame_set(bind_frame)
        bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = source_obj.evaluated_get(depsgraph)
    render_mesh = bpy.data.meshes.new_from_object(
        eval_obj,
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )

    bm = bmesh.new()
    bm.from_mesh(render_mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    wanted = {tuple(key) for key in triangle_keys}
    delete_faces = []
    for face in bm.faces:
        tri = tuple(sorted(vert.index for vert in face.verts))
        if tri not in wanted:
            delete_faces.append(face)

    if delete_faces:
        bmesh.ops.delete(bm, geom=delete_faces, context="FACES")

    fill_holes = False

    if fill_holes:
        boundary_edges = [edge for edge in bm.edges if len(edge.link_faces) == 1]
        hole_faces = []
        if boundary_edges:
            res = bmesh.ops.holes_fill(bm, edges=boundary_edges)
            if "faces" in res and res["faces"]:
                hole_faces = res["faces"]
                geom_to_hull = list({v for f in hole_faces for v in f.verts})
                hull_res = bmesh.ops.convex_hull(bm, input=geom_to_hull, use_existing_faces=True)
                new_hull_faces = [f for f in hull_res.get("geom", []) if isinstance(f, bmesh.types.BMFace)]
                for face in new_hull_faces:
                    if face not in hole_faces:
                        hole_faces.append(face)

    if not bm.faces:
        bm.free()
        bpy.data.meshes.remove(render_mesh, do_unlink=True)
        return False

    bm.to_mesh(render_mesh)
    bm.free()

    render_obj = bpy.data.objects.new(f"{source_obj.name}_flatrig_part", render_mesh)
    render_obj.matrix_world = source_obj.matrix_world.copy()
    scene.collection.objects.link(render_obj)

    if fill_holes:
        hole_mat_index = len(render_obj.data.materials)
        hole_mat = bpy.data.materials.new(name="flatrig_hole_mask")
        render_obj.data.materials.append(hole_mat)

        for polygon in render_obj.data.polygons:
            if polygon.index >= len(render_obj.data.polygons) - len(hole_faces):
                polygon.material_index = hole_mat_index

    restore_info, created_materials = _apply_unlit_materials([render_obj])

    camera = setup_orthographic_camera(
        view_cfg,
        projection_frame,
        depth_center=depth_center,
        camera_name="flatRig_PartCamera",
        projection_matrix=projection_matrix,
    )

    hidden_objects = []
    for scene_obj in scene.objects:
        if scene_obj in (render_obj, camera):
            continue
        hidden_objects.append((scene_obj, scene_obj.hide_render))
        scene_obj.hide_render = True

    try:
        if fill_holes:
            success = render_projected_sprite(scene, output_path, resolution=resolution)
            mask_path = str(output_path).rsplit(".", 1)[0] + "_mask.png"

            mask_materials = []
            for index, material in enumerate(render_obj.data.materials):
                if material is None:
                    mask_materials.append(None)
                    continue

                mask_mat = bpy.data.materials.new(name=f"flatrig_mask_mat_{index}")
                mask_mat.use_nodes = True
                if hasattr(mask_mat, "use_backface_culling"):
                    mask_mat.use_backface_culling = True
                if hasattr(mask_mat, "show_transparent_back"):
                    mask_mat.show_transparent_back = False
                nodes = mask_mat.node_tree.nodes
                links = mask_mat.node_tree.links
                nodes.clear()

                output_node = nodes.new("ShaderNodeOutputMaterial")
                emission_node = nodes.new("ShaderNodeEmission")
                emission_node.inputs["Strength"].default_value = 1.0

                if index == hole_mat_index:
                    emission_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
                else:
                    emission_node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 0.0)

                links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

                mass_blend = getattr(mask_mat, "blend_method", None)
                if mass_blend is not None:
                    mask_mat.blend_method = "CLIP"

                mask_materials.append(mask_mat)

            render_obj.data.materials.clear()
            for material in mask_materials:
                render_obj.data.materials.append(material)

            scene.render.film_transparent = False
            original_bg_color = None
            bg_node = None
            if scene.world is not None:
                scene.world.use_nodes = True
                bg_node = scene.world.node_tree.nodes.get("Background")
                if bg_node:
                    original_bg_color = tuple(bg_node.inputs["Color"].default_value)
                    bg_node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)

            render_projected_sprite(scene, mask_path, resolution=resolution)

            scene.render.film_transparent = True
            if bg_node and original_bg_color is not None:
                bg_node.inputs["Color"].default_value = original_bg_color

            for material in mask_materials:
                if material:
                    bpy.data.materials.remove(material, do_unlink=True)

            return success

        return render_projected_sprite(scene, output_path, resolution=resolution)
    finally:
        _restore_materials(restore_info, created_materials)
        bpy.data.objects.remove(render_obj, do_unlink=True)
        bpy.data.meshes.remove(render_mesh, do_unlink=True)
        bpy.data.objects.remove(camera, do_unlink=True)
        for scene_obj, previous_state in hidden_objects:
            scene_obj.hide_render = previous_state


def _create_placeholder_atlas(output_path, resolution=2048):
    """Create a plain placeholder image if rendering fails."""
    print("[flatrig_texture] Creating placeholder atlas...")
    image = bpy.data.images.new("placeholder", width=resolution, height=resolution)
    image.pixels = [1.0] * (resolution * resolution * 4)
    image.filepath_raw = output_path
    image.file_format = "PNG"
    image.save()
    print(f"[flatrig_texture] Placeholder saved to: {output_path}")


def extract_2d_mesh(
    mesh_obj,
    view_cfg,
    projection_frame=None,
    source_frame=None,
    projection_inverse=None,
):
    """Extract the bind-pose mesh projected to 2D.
    """
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
        [point_depth(vertices_3d[i], view_cfg, projection_inverse=projection_inverse) for i in range(len(vertices_3d))],
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

    parser = argparse.ArgumentParser(description="Inspect or convert 3D sources for flatRig.")
    parser.add_argument("command", choices=("inspect", "inspect-3d-source", "convert", "extract-bone-hierarchy", "extract-2d-mesh"))
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-format", default="glb", choices=("glb",))
    # Projection parameters for extract-bone-hierarchy
    parser.add_argument("--view-preset", default="front",
                        choices=list(VIEW_PRESETS.keys()) + list(VIEW_PRESETS.keys()),
                        help="View preset for projection (front, back, side, side_r, top, bottom)")
    parser.add_argument("--view-dir", default=None,
                        help="Custom view direction as 'x,y,z' tuple")
    parser.add_argument("--view-up", default=None,
                        help="Custom view up hint as 'x,y,z' tuple")
    parser.add_argument("--view-roll", type=float, default=0.0,
                        help="View roll in degrees")
    parser.add_argument("--source-frame", type=int, default=None,
                        help="Source frame for pose evaluation")
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


def reset_blender_scene() -> None:
    """Reset Blender to an empty scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def get_blender_scene():
    """Return the active Blender scene."""
    return bpy.context.scene


def get_blender_depsgraph():
    """Return the active evaluated dependency graph."""
    return bpy.context.evaluated_depsgraph_get()


def update_blender_view_layer() -> None:
    """Update the active Blender view layer."""
    bpy.context.view_layer.update()


def get_blender_action(action_name: str):
    """Return a Blender action by name."""
    return bpy.data.actions.get(action_name)


def import_extra_animation_source(filepath: str) -> list[str]:
    """Import an animation source and return newly created action names."""
    existing_action_names = {action.name for action in bpy.data.actions}
    import_model(filepath)
    return [
        action.name
        for action in bpy.data.actions
        if action.name not in existing_action_names
    ]


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


def list_armature_action_objects(armature_obj):
    """Return Blender action objects relevant to an armature, active action first."""
    actions = []
    seen = set()
    if armature_obj and armature_obj.animation_data and armature_obj.animation_data.action:
        active = armature_obj.animation_data.action
        actions.append(active)
        seen.add(active.name)
    for action in bpy.data.actions:
        if action.name in seen or not is_pose_action(action):
            continue
        actions.append(action)
        seen.add(action.name)
    return actions


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


def convert_source(source_path: str, output_path: str, target_format: str) -> dict[str, object]:
    if target_format != "glb":
        raise ValueError(f"Unsupported target format: {target_format}")

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
        "target_format": target_format,
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
        [world_matrix[3][0], world_matrix[3][1], world_matrix[3][2], world_matrix[3][3]]
    ]


def sample_bone_world_matrices(armature, frames, bone_names=None):
    """Sample per-bone world matrices for a specific list of frames."""
    scene = bpy.context.scene
    if bone_names is None:
        bone_names = [bone.name for bone in armature.pose.bones]
    result = np.zeros((len(frames), len(bone_names), 4, 4), dtype=np.float64)

    for frame_index, frame in enumerate(frames):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        for bone_index, bone_name in enumerate(bone_names):
            pose_bone = armature.pose.bones[bone_name]
            world_matrix = armature.matrix_world @ pose_bone.matrix
            result[frame_index, bone_index] = np.array(world_matrix, dtype=np.float64)

    return result


def sample_projected_bone_segments_2d(
    armature,
    view_cfg,
    frame_start,
    frame_end,
    bone_names=None,
    projection_space="world",
    projection_reference_root=None,
):
    """Sample projected 2D head/tail segments for each bone on every frame."""
    scene = bpy.context.scene
    if bone_names is None:
        bone_names = [bone.name for bone in armature.pose.bones]

    frames = []
    fps = scene.render.fps

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        projection_inverse = get_projection_reference_inverse(
            armature,
            projection_space=projection_space,
            reference_root_matrix=projection_reference_root,
        )

        heads = []
        tails = []
        for bone_name in bone_names:
            pose_bone = armature.pose.bones[bone_name]
            head_world = armature.matrix_world @ pose_bone.head
            tail_world = armature.matrix_world @ pose_bone.tail
            head_2d = project_point_ortho(head_world, view_cfg, projection_inverse=projection_inverse)
            tail_2d = project_point_ortho(tail_world, view_cfg, projection_inverse=projection_inverse)
            heads.append([round(head_2d[0], 4), round(head_2d[1], 4)])
            tails.append([round(tail_2d[0], 4), round(tail_2d[1], 4)])

        frames.append(
            {
                "frame": frame,
                "time": round((frame - frame_start) / fps, 4),
                "heads": heads,
                "tails": tails,
            }
        )

    return {
        "fps": fps,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "bones": list(bone_names),
        "frames": frames,
    }


def extract_mesh_data(mesh_obj) -> dict[str, Any]:
    """Extract mesh vertices, normals, UVs, and weights from a mesh object."""
    if mesh_obj is None:
        return {"vertex_count": 0, "vertices": [], "triangles": []}
    
    mesh = mesh_obj.data
    if hasattr(mesh, "calc_loop_triangles"):
        mesh.calc_loop_triangles()
    
    # Get the inverse bind matrix if the mesh is skinned
    inverse_bind_matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    armature_obj = None
    if mesh_obj.parent and mesh_obj.parent.type == "ARMATURE":
        armature_obj = mesh_obj.parent
    elif mesh_obj.modifiers:
        for mod in mesh_obj.modifiers:
            if mod.type == "ARMATURE" and mod.object:
                armature_obj = mod.object
                break
    
    # Get vertex groups for skinning weights
    vertex_groups = {}
    for i, vg in enumerate(mesh_obj.vertex_groups):
        vertex_groups[vg.name] = i
    
    vertices = []
    normals = []
    uvs = []
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
            tri_uvs.extend([
                [uv_layer[tri.loops[0]].uv.x, uv_layer[tri.loops[0]].uv.y],
                [uv_layer[tri.loops[1]].uv.x, uv_layer[tri.loops[1]].uv.y],
                [uv_layer[tri.loops[2]].uv.x, uv_layer[tri.loops[2]].uv.y],
            ])
    
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
                                    rgba.append(int(pixels[i] * 255))      # R
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
    if record["parent_child_count"] <= 1 and record["parent_length_ratio"] < TERMINAL_CHAIN_PARENT_RATIO:
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
    
    positive_lengths = sorted(record["length"] for record in records if record["length"] > SEGMENT_EPSILON)
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


def extract_2d_mesh_cli(
    source_path: str,
    output_path: str,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
    source_frame: int = None,
) -> dict[str, object]:
    """CLI wrapper for extract_2d_mesh that matches Python's pipeline.
    
    This function imports the model, finds the mesh and armature, and calls extract_2d_mesh
    to get the proper 2D mesh projection that Python uses.
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
    
    # Call extract_2d_mesh (same as Python pipeline)
    mesh_data = extract_2d_mesh(
        mesh_obj,
        view_cfg,
        source_frame=source_frame,
    )
    
    return {
        "ok": True,
        "detail": "extracted",
        "source": source_path,
        "view_preset": view_preset,
        "view_roll": view_roll,
        "mesh": mesh_data,
    }


def extract_bone_hierarchy_cli(
    source_path: str,
    output_path: str,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
    source_frame: int = None,
) -> dict[str, object]:
    """CLI wrapper for extract_bone_hierarchy that matches Python's pipeline.
    
    This function imports the model, finds the armature, and calls extract_bone_hierarchy
    to get the proper 2D bone hierarchy that Python uses.
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
    
    # Call extract_bone_hierarchy (same as Python pipeline)
    bones = extract_bone_hierarchy(
        armature_obj,
        view_cfg,
        source_frame=source_frame,
        use_rest_pose=False,
        projection_space="world",
        projection_reference_root=None,
    )
    
    return {
        "ok": True,
        "detail": "extracted",
        "source": source_path,
        "view_preset": view_preset,
        "view_roll": view_roll,
        "bones": bones,
    }


def main() -> None:
    args = parse_args()
    source_path = str(Path(args.source).expanduser().resolve())
    output_path = Path(args.output).expanduser().resolve()

    payload: dict[str, object]
    
    if args.command == "inspect" or args.command == "inspect-3d-source":
        payload = inspect_source(source_path)
    elif args.command == "convert":
        payload = convert_source(source_path, str(output_path), args.target_format)
    elif args.command == "extract-bone-hierarchy":
        # Parse view_dir and view_up if provided
        view_dir = None
        view_up = None
        if args.view_dir:
            view_dir = tuple(float(x) for x in args.view_dir.split(","))
        if args.view_up:
            view_up = tuple(float(x) for x in args.view_up.split(","))
        
        payload = extract_bone_hierarchy_cli(
            source_path,
            str(output_path),
            view_preset=args.view_preset,
            view_dir=view_dir,
            view_up=view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
        )
    elif args.command == "extract-2d-mesh":
        # Parse view_dir and view_up if provided
        view_dir = None
        view_up = None
        if args.view_dir:
            view_dir = tuple(float(x) for x in args.view_dir.split(","))
        if args.view_up:
            view_up = tuple(float(x) for x in args.view_up.split(","))
        
        payload = extract_2d_mesh_cli(
            source_path,
            str(output_path),
            view_preset=args.view_preset,
            view_dir=view_dir,
            view_up=view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
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
