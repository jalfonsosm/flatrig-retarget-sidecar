VECTOR_EPSILON = 1e-8

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None

import math
import re
import json
from blender_io.bone_utils import _find_root_bone_name
import numpy as np
from collections import defaultdict

__all__ = [
    "_rest_local_quat",
    "_quat_angle_degrees",
    "_matrix3_to_np",
    "_rotation_between_np",
    "_twist_about_y_np",
    "_armature_uniform_scale",
    "_world_rotation_3x3",
    "_armature_world_rotation",
    "get_projection_reference_inverse",
    "orthonormalize_3x3",
    "safe_inverse_2x2",
    "orthonormalize_2x2",
    "_build_2d_basis",
    "_normalize_angle",
    "_matrix_xyz_euler_degrees",
    "_rotation_between_vectors"
]



def _rest_local_quat(data_bone):
    """Rest orientation of a (data) bone relative to its parent, as a quaternion."""
    if data_bone.parent is not None:
        matrix = data_bone.parent.matrix_local.inverted() @ data_bone.matrix_local
    else:
        matrix = data_bone.matrix_local
    return matrix.to_quaternion()


def _quat_angle_degrees(a, b) -> float:
    dot = abs(
        float(a.w) * float(b.w)
        + float(a.x) * float(b.x)
        + float(a.y) * float(b.y)
        + float(a.z) * float(b.z)
    )
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _matrix3_to_np(matrix) -> np.ndarray:
    """mathutils 3x3 (or the 3x3 block of a 4x4) -> numpy 3x3."""
    return np.array(
        [[float(matrix[row][col]) for col in range(3)] for row in range(3)],
        dtype=np.float64,
    )


def _rotation_between_np(source, target) -> np.ndarray:
    """Minimal 3x3 rotation taking unit-ish vector `source` onto `target`.

    No twist beyond what is needed to align the two directions — so a bone's
    roll (and therefore how its mesh is mounted) is preserved.
    """
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return np.eye(3, dtype=np.float64)
    a = a / na
    b = b / nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-9:
        # Opposite directions: rotate 180° about any axis perpendicular to a.
        helper = np.array((1.0, 0.0, 0.0)) if abs(a[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
        axis = np.cross(a, helper)
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        x, y, z = axis
        return 2.0 * np.array(((x * x, x * y, x * z),
                               (x * y, y * y, y * z),
                               (x * z, y * z, z * z))) - np.eye(3)
    skew = np.array(((0.0, -v[2], v[1]),
                     (v[2], 0.0, -v[0]),
                     (-v[1], v[0], 0.0)), dtype=np.float64)
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + c))


def _twist_about_y_np(rot3) -> np.ndarray:
    """Twist component of a 3x3 rotation about the bone axis (+Y), as 3x3.

    Swing-twist decomposition keeping only the rotation about +Y and discarding
    any swing that tilts Y. Used to graft the donor's roll *motion* onto a
    direction-matched target bone without disturbing its aimed direction — the
    direction-only copy preserves the target's rest roll and therefore drops the
    donor's wrist/finger twist (12-60deg on this fixture). At the donor's rest
    frame this is identity (no twist), so a static roll mismatch never spins the
    target's mesh (the auto-rig "face backwards" regression stays fixed).
    """
    if mathutils is None:
        return np.eye(3, dtype=np.float64)
    q = mathutils.Matrix(np.asarray(rot3, dtype=np.float64).tolist()).to_quaternion()
    n = math.hypot(q.w, q.y)
    if n < 1e-9:
        return np.eye(3, dtype=np.float64)
    twist = mathutils.Quaternion((q.w / n, 0.0, q.y / n, 0.0))
    return _matrix3_to_np(twist.to_matrix())


def _armature_uniform_scale(armature_obj) -> float:
    if armature_obj is None:
        return 1.0
    scale = armature_obj.matrix_world.to_scale()
    values = [abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z))]
    values = [value for value in values if value > VECTOR_EPSILON]
    if not values:
        return 1.0
    return sum(values) / len(values)


def _world_rotation_3x3(matrix4):
    """Pure-rotation 3x3 from a world matrix, stripping object scale.

    Bone world matrices built from ``armature.matrix_world @ bone.matrix``
    inherit the armature object's scale (Mixamo imports carry a uniform
    0.01 cm->m factor). Native LBS treats this 3x3 as an orthonormal bind/
    frame rotation, so any embedded scale collapses every skinned vertex
    toward the bone head and corrupts the depth-based draw order. Extract
    the rotation via the quaternion, which normalizes the scale away.
    """
    return matrix4.to_quaternion().to_matrix()


def _armature_world_rotation(armature_obj):
    if armature_obj is None:
        return mathutils.Matrix.Identity(3)
    return armature_obj.matrix_world.to_quaternion().to_matrix()


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


def _matrix_xyz_euler_degrees(matrix):
    euler = matrix.to_euler("XYZ")
    return [
        math.degrees(float(euler.x)),
        math.degrees(float(euler.y)),
        math.degrees(float(euler.z)),
    ]

SEGMENT_EPSILON = 1e-8



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
