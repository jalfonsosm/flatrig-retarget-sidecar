"""Small projection bridge safe for Blender's isolated embedded Python.

The product's projection and visibility algorithms remain in
``flatrig_private``. These coordinate transforms are the Blender-bound subset
needed to place an orthographic sprite camera, so the public worker must be
able to import them without the application's CPython virtual environment.
"""

from __future__ import annotations

import numpy as np

from flatrig._sidecar_import import orthonormalize_3x3


def compose_projection_plane_point(x, y, depth, view_cfg):
    """Build a projection-space point from screen coordinates and depth."""

    return (
        np.asarray(view_cfg["right_axis"], dtype=np.float64) * float(x)
        + np.asarray(view_cfg["up_axis"], dtype=np.float64) * float(y)
        + np.asarray(view_cfg["depth_axis"], dtype=np.float64) * float(depth)
    )


def transform_point_from_projection_space(point_3d, projection_matrix=None):
    """Transform a projection-space point back to world space."""

    point_3d = np.asarray(point_3d, dtype=np.float64)
    if projection_matrix is None:
        return point_3d
    point_h = np.concatenate((point_3d, np.array((1.0,), dtype=np.float64)))
    return (np.asarray(projection_matrix, dtype=np.float64) @ point_h)[:3]


def transform_direction_from_projection_space(direction_3d, projection_matrix=None):
    """Transform a projection-space direction back to world space."""

    direction_3d = np.asarray(direction_3d, dtype=np.float64)
    if projection_matrix is None:
        return direction_3d
    rotation = np.asarray(projection_matrix, dtype=np.float64)[:3, :3]
    return rotation @ direction_3d


def get_projection_reference_matrix(
    armature,
    projection_space="world",
    use_rest_pose=False,
    root_bone_name=None,
    reference_root_matrix=None,
):
    """Return the root-space transform used to place the sprite camera."""

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


def _find_root_bone_name(armature):
    for bone in armature.data.bones:
        if bone.parent is None:
            return bone.name
    return None
