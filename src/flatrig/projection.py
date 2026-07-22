"""
Blender-facing orthographic projection helpers for the sidecar worker.

The pure projection / visibility / draw-order math lives privately in
``flatrig_private.projection_math`` (Cython-obfuscated at build time) and is
re-exported here so existing ``from flatrig.projection import ...`` call sites
keep working. This module keeps only the Blender (``bpy``/depsgraph) bound
helpers: evaluated mesh access, vertex sampling and the per-frame visibility map.
"""

import bpy
import numpy as np
from mathutils import Vector

from flatrig._sidecar_import import orthonormalize_3x3

# Pure projection math (no ``bpy``). Re-exported so callers that still do
# ``from flatrig.projection import project_points_ortho`` (etc.) resolve here.
from flatrig_private.projection_math import *  # noqa: F401,F403

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
