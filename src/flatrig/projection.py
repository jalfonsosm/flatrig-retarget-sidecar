"""
Orthographic projection helpers for the Blender worker.

All 2D geometry, UV generation and preview renders must share the same
projection frame. This module centralizes that logic.
"""

import math

import bpy
import numpy as np
from mathutils import Vector

from flatrig._sidecar_import import (
    compute_projection_frame,
    orthonormalize_3x3,
)

WORLD_UP = np.array((0.0, 0.0, 1.0), dtype=np.float64)
WORLD_Y = np.array((0.0, 1.0, 0.0), dtype=np.float64)
WORLD_X = np.array((1.0, 0.0, 0.0), dtype=np.float64)
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
VIEW_PRESET_NAMES = tuple(VIEW_PRESETS.keys())


def serialize_view_config(view_cfg):
    """Return a JSON-safe representation of a resolved view configuration."""
    return {
        "name": str(view_cfg.get("name") or "view"),
        "preset": view_cfg.get("preset"),
        "mode": str(view_cfg.get("mode") or "preset"),
        "roll_degrees": round(float(view_cfg.get("roll_degrees", 0.0)), 4),
        "view_dir": _serialize_vector(view_cfg.get("view_dir")),
        "right_axis": _serialize_vector(view_cfg.get("right_axis")),
        "up_axis": _serialize_vector(view_cfg.get("up_axis")),
        "depth_axis": _serialize_vector(view_cfg.get("depth_axis")),
    }


def compose_projection_plane_point(x, y, depth, view_cfg):
    """Build a 3D point in projection space from 2D screen coordinates plus depth."""
    return (
        np.asarray(view_cfg["right_axis"], dtype=np.float64) * float(x)
        + np.asarray(view_cfg["up_axis"], dtype=np.float64) * float(y)
        + np.asarray(view_cfg["depth_axis"], dtype=np.float64) * float(depth)
    )


def get_projection_reference_matrix(
    armature,
    projection_space="world",
    use_rest_pose=False,
    root_bone_name=None,
    reference_root_matrix=None,
):
    """Return the transform that defines the requested projection space."""
    if projection_space != "root" or armature is None:
        return None

    current_root_matrix = get_root_world_matrix(
        armature,
        use_rest_pose=use_rest_pose,
        root_bone_name=root_bone_name,
    )
    if current_root_matrix is None:
        return None

    if reference_root_matrix is None:
        reference_root_matrix = current_root_matrix

    current_rotation = orthonormalize_3x3(current_root_matrix[:3, :3])
    reference_rotation = orthonormalize_3x3(
        np.asarray(reference_root_matrix, dtype=np.float64)[:3, :3]
    )
    projection_matrix = np.eye(4, dtype=np.float64)
    projection_matrix[:3, :3] = current_rotation @ reference_rotation.T
    projection_matrix[:3, 3] = current_root_matrix[:3, 3]

    return projection_matrix


def get_projection_reference_inverse(
    armature,
    projection_space="world",
    use_rest_pose=False,
    root_bone_name=None,
    reference_root_matrix=None,
):
    """Return the inverse transform for the requested projection space."""
    reference_matrix = get_projection_reference_matrix(
        armature,
        projection_space=projection_space,
        use_rest_pose=use_rest_pose,
        root_bone_name=root_bone_name,
        reference_root_matrix=reference_root_matrix,
    )
    if reference_matrix is None:
        return None
    return np.linalg.inv(reference_matrix)


def get_root_world_matrix(armature, use_rest_pose=False, root_bone_name=None):
    """Return the current root bone transform in world space."""
    if armature is None:
        return None
    if root_bone_name is None:
        root_bone_name = _find_root_bone_name(armature)
    if root_bone_name is None:
        return None

    if use_rest_pose:
        root_bone = armature.data.bones[root_bone_name]
        matrix = armature.matrix_world @ root_bone.matrix_local
    else:
        root_bone = armature.pose.bones[root_bone_name]
        matrix = armature.matrix_world @ root_bone.matrix
    return np.array(matrix, dtype=np.float64)


def project_point_ortho(point_3d, view_cfg, projection_inverse=None):
    """Project a 3D point into the configured 2D plane."""
    projected = transform_points_to_projection_space(
        point_3d,
        projection_inverse=projection_inverse,
    )
    projected_2d = _project_projection_space_direction(projected, view_cfg)
    return float(projected_2d[0]), float(projected_2d[1])


def project_points_ortho(points_3d, view_cfg, projection_inverse=None):
    """Vectorized orthographic projection for an array of 3D points."""
    points_3d = transform_points_to_projection_space(
        points_3d,
        projection_inverse=projection_inverse,
    )
    right_axis = np.asarray(view_cfg["right_axis"], dtype=np.float64)
    up_axis = np.asarray(view_cfg["up_axis"], dtype=np.float64)
    return np.stack((points_3d @ right_axis, points_3d @ up_axis), axis=1)


def project_direction_ortho(direction_3d, view_cfg, projection_inverse=None):
    """Project a 3D direction vector into the configured 2D plane."""
    direction_3d = transform_direction_to_projection_space(
        direction_3d,
        projection_inverse=projection_inverse,
    )
    return _project_projection_space_direction(direction_3d, view_cfg)


def point_depth(point_3d, view_cfg, projection_inverse=None):
    """Return a scalar depth where larger values are closer to the camera."""
    projected = transform_points_to_projection_space(
        point_3d,
        projection_inverse=projection_inverse,
    )
    return float(np.dot(projected, np.asarray(view_cfg["depth_axis"], dtype=np.float64)))


def transform_points_to_projection_space(points_3d, projection_inverse=None):
    """Transform world-space points into the active projection space."""
    points_3d = np.asarray(points_3d, dtype=np.float64)
    if projection_inverse is None:
        return points_3d

    squeeze = False
    if points_3d.ndim == 1:
        points_3d = points_3d[np.newaxis, :]
        squeeze = True

    points_h = np.concatenate(
        (points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    projection_inverse = np.asarray(projection_inverse, dtype=np.float64)

    if projection_inverse.ndim == 2:
        transformed = (projection_inverse @ points_h.T).T[:, :3]
    else:
        transformed = np.einsum("nij,nj->ni", projection_inverse, points_h)[:, :3]

    if squeeze:
        return transformed[0]
    return transformed


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


def get_evaluated_mesh(obj, depsgraph):
    """Get the evaluated (deformed) mesh after modifiers."""
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    return eval_obj, mesh


def get_evaluated_vertex_positions(obj, depsgraph):
    """Get world-space vertex positions after armature deformation."""
    eval_obj, mesh = get_evaluated_mesh(obj, depsgraph)
    world_mat = eval_obj.matrix_world

    positions = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    for i, vert in enumerate(mesh.vertices):
        world_co = world_mat @ vert.co
        positions[i] = (world_co.x, world_co.y, world_co.z)

    eval_obj.to_mesh_clear()
    return positions


def ensure_polygon_normals(mesh):
    """Populate polygon normals when running on Blender versions that need it."""
    if hasattr(mesh, "calc_normals"):
        mesh.calc_normals()


def compute_vertex_visibility(obj, depsgraph, view_cfg, projection_inverse=None):
    """Cheap visibility estimate based on front-facing polygons.

    This is not a true z-buffer test, but it is still useful as a frame mask
    for the optimizer. Slot splitting handles the larger occlusion problem.
    """
    eval_obj, mesh = get_evaluated_mesh(obj, depsgraph)
    world_mat = eval_obj.matrix_world
    view_dir = Vector(np.asarray(view_cfg["view_dir"], dtype=np.float64))

    visible = np.zeros(len(mesh.vertices), dtype=bool)
    ensure_polygon_normals(mesh)

    for poly in mesh.polygons:
        world_normal = (world_mat.to_3x3() @ poly.normal).normalized()
        if projection_inverse is not None:
            world_normal = Vector(
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
    """Approximate visible-surface vertices with a small orthographic z-buffer.

    `triangle_groups=None` rasterizes the whole mesh as one surface.
    When groups are provided, each group is rasterized independently and the
    resulting visible vertices are unioned. This approximates "render each
    sprite alone" visibility for segmented exports.
    """
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


def estimate_part_draw_order_from_triangles(
    positions_3d,
    triangles,
    triangle_part_indices,
    view_cfg,
    *,
    projection_inverse=None,
    raster_size=256,
    base_order=None,
):
    """Estimate a back-to-front part order from rasterized surface overlap.

    The order is derived from the visible front-most surface and pairwise
    overlap votes between parts. This is intended for slot ordering where a
    single global z-order per frame must be inferred from the 3D model.
    """
    triangle_array = np.asarray(triangles, dtype=np.int32)
    part_indices = np.asarray(triangle_part_indices, dtype=np.int32)
    if triangle_array.size == 0 or part_indices.size == 0:
        return {
            "ordered_part_indices": list(base_order or []),
            "scores": {},
            "visible_depth_means": {},
            "visible_pixel_counts": {},
            "overlap_pixel_count": 0,
        }

    raster_size = max(16, int(raster_size))
    projection_positions = transform_points_to_projection_space(
        positions_3d,
        projection_inverse=projection_inverse,
    )
    projected_points_2d = project_points_ortho(
        positions_3d,
        view_cfg,
        projection_inverse=projection_inverse,
    )
    unique_parts = sorted({int(part_index) for part_index in part_indices if int(part_index) >= 0})
    if not unique_parts:
        return {
            "ordered_part_indices": list(base_order or []),
            "scores": {},
            "visible_depth_means": {},
            "visible_pixel_counts": {},
            "overlap_pixel_count": 0,
        }

    local_part_index = {
        int(part_index): int(local_index) for local_index, part_index in enumerate(unique_parts)
    }
    local_triangle_parts = np.asarray(
        [local_part_index.get(int(part_index), -1) for part_index in part_indices],
        dtype=np.int32,
    )

    p0 = projection_positions[triangle_array[:, 0]]
    p1 = projection_positions[triangle_array[:, 1]]
    p2 = projection_positions[triangle_array[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    normal_lengths = np.linalg.norm(normals, axis=1)
    view_dir = np.asarray(view_cfg["view_dir"], dtype=np.float64)
    front_triangle_mask = (
        (local_triangle_parts >= 0)
        & (normal_lengths > VISIBILITY_RASTER_EPSILON)
        & (np.einsum("ij,j->i", normals, view_dir) < -VISIBILITY_RASTER_EPSILON)
    )
    if not np.any(front_triangle_mask):
        fallback = list(base_order or unique_parts)
        return {
            "ordered_part_indices": fallback,
            "scores": {int(part_index): 0.0 for part_index in unique_parts},
            "visible_depth_means": {int(part_index): float("-inf") for part_index in unique_parts},
            "visible_pixel_counts": {int(part_index): 0 for part_index in unique_parts},
            "overlap_pixel_count": 0,
        }

    depth_axis = np.asarray(view_cfg["depth_axis"], dtype=np.float64)
    depths = projection_positions @ depth_axis
    frame = compute_projection_frame(projected_points_2d, margin=0.02)
    span = max(float(frame["span"]), VISIBILITY_RASTER_EPSILON)
    raster_points = np.empty_like(projected_points_2d, dtype=np.float64)
    raster_points[:, 0] = (
        (projected_points_2d[:, 0] - float(frame["min_x"])) / span * (raster_size - 1)
    )
    raster_points[:, 1] = (
        (projected_points_2d[:, 1] - float(frame["min_y"])) / span * (raster_size - 1)
    )

    num_parts = len(unique_parts)
    depth_buffer = np.full((raster_size, raster_size), -np.inf, dtype=np.float64)
    top_part_buffer = np.full((raster_size, raster_size), -1, dtype=np.int32)
    part_coverage = np.zeros((num_parts, raster_size, raster_size), dtype=bool)
    part_depth_buffer = np.full((num_parts, raster_size, raster_size), -np.inf, dtype=np.float64)

    for triangle_index in np.nonzero(front_triangle_mask)[0]:
        local_part = int(local_triangle_parts[int(triangle_index)])
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
            _mark_triangle_centroid_part_visibility(
                top_part_buffer,
                depth_buffer,
                part_coverage[local_part],
                part_depth_buffer[local_part],
                local_part=local_part,
                points=((x0, y0), (x1, y1), (x2, y2)),
                depths=(z0, z1, z2),
            )
            continue

        denominator = ((y1 - y2) * (x0 - x2)) + ((x2 - x1) * (y0 - y2))
        if abs(denominator) <= VISIBILITY_RASTER_EPSILON:
            _mark_triangle_centroid_part_visibility(
                top_part_buffer,
                depth_buffer,
                part_coverage[local_part],
                part_depth_buffer[local_part],
                local_part=local_part,
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
            _mark_triangle_centroid_part_visibility(
                top_part_buffer,
                depth_buffer,
                part_coverage[local_part],
                part_depth_buffer[local_part],
                local_part=local_part,
                points=((x0, y0), (x1, y1), (x2, y2)),
                depths=(z0, z1, z2),
            )
            continue

        part_coverage_patch = part_coverage[local_part, min_y : max_y + 1, min_x : max_x + 1]
        part_coverage_patch |= inside
        part_depth_patch = part_depth_buffer[local_part, min_y : max_y + 1, min_x : max_x + 1]

        patch_depth = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
        patch_parts = top_part_buffer[min_y : max_y + 1, min_x : max_x + 1]
        depth_values = (w0 * z0) + (w1 * z1) + (w2 * z2)
        visible_part_pixels = inside & (depth_values > part_depth_patch + VISIBILITY_RASTER_EPSILON)
        if np.any(visible_part_pixels):
            part_depth_patch[visible_part_pixels] = depth_values[visible_part_pixels]
        update = inside & (depth_values > patch_depth + VISIBILITY_RASTER_EPSILON)
        if np.any(update):
            patch_depth[update] = depth_values[update]
            patch_parts[update] = local_part

    visible_depth_means = {}
    visible_pixel_counts = {}
    local_scores = np.zeros(num_parts, dtype=np.float64)
    overlap_pixel_count = 0
    for local_index, part_index in enumerate(unique_parts):
        top_mask = top_part_buffer == local_index
        visible_pixel_counts[int(part_index)] = int(np.count_nonzero(top_mask))
        if np.any(top_mask):
            visible_depth_means[int(part_index)] = float(np.mean(depth_buffer[top_mask]))
        else:
            visible_depth_means[int(part_index)] = float("-inf")

    for local_a in range(num_parts):
        for local_b in range(local_a + 1, num_parts):
            overlap = np.isfinite(part_depth_buffer[local_a]) & np.isfinite(
                part_depth_buffer[local_b]
            )
            if not np.any(overlap):
                continue
            overlap_count = int(np.count_nonzero(overlap))
            overlap_pixel_count += overlap_count
            depth_delta = part_depth_buffer[local_a][overlap] - part_depth_buffer[local_b][overlap]
            a_front = int(np.count_nonzero(depth_delta > VISIBILITY_RASTER_EPSILON))
            b_front = int(np.count_nonzero(depth_delta < -VISIBILITY_RASTER_EPSILON))
            margin = a_front - b_front
            if margin == 0:
                continue
            weight = float(margin) / float(max(overlap_count, 1))
            local_scores[local_a] += weight
            local_scores[local_b] -= weight

    base_rank = {
        int(part_index): rank for rank, part_index in enumerate(base_order or unique_parts)
    }
    ordered_part_indices = sorted(
        unique_parts,
        key=lambda part_index: (
            float(local_scores[local_part_index[int(part_index)]]),
            float(visible_depth_means.get(int(part_index), float("-inf"))),
            int(base_rank.get(int(part_index), len(unique_parts))),
        ),
    )

    return {
        "ordered_part_indices": [int(part_index) for part_index in ordered_part_indices],
        "scores": {
            int(part_index): float(local_scores[local_part_index[int(part_index)]])
            for part_index in unique_parts
        },
        "visible_depth_means": visible_depth_means,
        "visible_pixel_counts": visible_pixel_counts,
        "overlap_pixel_count": int(overlap_pixel_count),
    }


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


def _mark_triangle_centroid_part_visibility(
    top_part_buffer,
    depth_buffer,
    part_coverage,
    part_depth_buffer,
    *,
    local_part,
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
    part_coverage[centroid_y, centroid_x] = True
    if centroid_depth > part_depth_buffer[centroid_y, centroid_x] + VISIBILITY_RASTER_EPSILON:
        part_depth_buffer[centroid_y, centroid_x] = centroid_depth
    if centroid_depth > depth_buffer[centroid_y, centroid_x] + VISIBILITY_RASTER_EPSILON:
        depth_buffer[centroid_y, centroid_x] = centroid_depth
        top_part_buffer[centroid_y, centroid_x] = int(local_part)


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
    """Build a per-vertex, per-frame visibility and position map."""
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


def _find_root_bone_name(armature):
    for bone in armature.data.bones:
        if bone.parent is None:
            return bone.name
    return None


def _resolve_view_preset_name(view_name):
    resolved = str(view_name or "side").strip().lower()
    if not resolved:
        return "side"
    resolved = VIEW_ALIASES.get(resolved, resolved)
    if resolved == "custom":
        raise ValueError("view 'custom' requires --view-dir to be provided.")
    if resolved not in VIEW_PRESETS:
        available = ", ".join(sorted(set(VIEW_PRESET_NAMES) | set(VIEW_ALIASES)))
        raise ValueError(f"Unknown view '{view_name}'. Available presets: {available}")
    return resolved


def _build_explicit_view_config(view_name, view_dir, right_axis, up_axis, preset_name=None):
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
    view_dir = _normalize_vector(view_dir, label="view_dir")
    up_hint = _normalize_vector(up_hint if up_hint is not None else WORLD_UP, label="up_hint")
    if abs(float(np.dot(view_dir, up_hint))) >= 1.0 - 1e-5:
        for fallback in (WORLD_Y, WORLD_X, -WORLD_X):
            fallback = _normalize_vector(fallback, label="fallback_up")
            if abs(float(np.dot(view_dir, fallback))) < 1.0 - 1e-5:
                up_hint = fallback
                break

    right_axis = np.cross(view_dir, up_hint)
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


def _finalize_view_config(view_name, preset_name, view_dir, right_axis, up_axis, roll_degrees):
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


def _project_projection_space_direction(direction_3d, view_cfg):
    direction_3d = np.asarray(direction_3d, dtype=np.float64)
    basis_2d = np.asarray(view_cfg["basis_2d"], dtype=np.float64)
    if direction_3d.ndim == 1:
        return basis_2d @ direction_3d
    return direction_3d @ basis_2d.T


def _normalize_vector(vector, *, label):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{label} must contain exactly 3 values.")
    length = float(np.linalg.norm(vector))
    if length <= VECTOR_EPSILON:
        raise ValueError(f"{label} must be non-zero.")
    return vector / length


def _axis_angle_rotation(axis, angle_degrees):
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
    return cos_value * np.eye(3, dtype=np.float64) + sin_value * cross + (1.0 - cos_value) * outer


def _serialize_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return [round(float(component), 6) for component in vector.tolist()]
