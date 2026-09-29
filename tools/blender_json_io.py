"""Stable JSON conversion helpers for the Blender worker.

Pure data-shape converters between Blender math types (``mathutils``
matrices/vectors/quaternions) and the JSON payloads exchanged with the
native pipeline. No scene access, no side effects; ``bpy``/``mathutils``
are optional so the converters that only need plain sequences stay
testable outside Blender.
"""

from __future__ import annotations

import numpy as np

try:
    import mathutils
except ImportError:  # pragma: no cover - exercised outside Blender only
    mathutils = None


def quat_to_stable_json(quat):
    """Serialize a quaternion normalized to a canonical (w >= 0) form."""
    quat = quat.copy()
    quat.normalize()
    if quat.w < 0.0:
        quat.negate()
    return [float(quat.w), float(quat.x), float(quat.y), float(quat.z)]


def matrix3_to_json(matrix):
    return [[float(matrix[row][col]) for col in range(3)] for row in range(3)]


def matrix3_from_json(values):
    if values is None:
        return mathutils.Matrix.Identity(3)
    return mathutils.Matrix(
        (
            (float(values[0][0]), float(values[0][1]), float(values[0][2])),
            (float(values[1][0]), float(values[1][1]), float(values[1][2])),
            (float(values[2][0]), float(values[2][1]), float(values[2][2])),
        )
    )


def matrix4_to_json(matrix):
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def vector_to_json(vector):
    return [float(vector[0]), float(vector[1]), float(vector[2])]


def vector_from_json(values, fallback=(0.0, 0.0, 0.0)):
    values = values or fallback
    return mathutils.Vector((float(values[0]), float(values[1]), float(values[2])))


def view_config_to_json(view_cfg):
    return {
        "name": str(view_cfg.get("name") or ""),
        "preset": view_cfg.get("preset"),
        "mode": view_cfg.get("mode"),
        "view_dir": vector_to_json(view_cfg["view_dir"]),
        "right_axis": vector_to_json(view_cfg["right_axis"]),
        "up_axis": vector_to_json(view_cfg["up_axis"]),
        "depth_axis": vector_to_json(view_cfg["depth_axis"]),
        "basis_2d": np.asarray(view_cfg["basis_2d"], dtype=np.float64).tolist(),
        "basis_3d": np.asarray(view_cfg["basis_3d"], dtype=np.float64).tolist(),
        "roll_degrees": float(view_cfg.get("roll_degrees", 0.0)),
        "auto_lateral_flip": bool(view_cfg.get("auto_lateral_flip", False)),
        "lateral_sign": view_cfg.get("lateral_sign"),
    }


def weights_to_json(weights):
    """Convert weight dicts to JSON-serializable format."""
    serialized = []
    for weight_dict in weights:
        pairs = []
        for bone_index, weight_value in sorted((weight_dict or {}).items()):
            pairs.append([int(bone_index), float(weight_value)])
        serialized.append(pairs)
    return serialized
