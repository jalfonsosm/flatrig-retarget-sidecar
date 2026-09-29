"""View presets and orthographic projection helpers for the Blender worker."""

from __future__ import annotations

import math

import numpy as np

VECTOR_EPSILON = 1e-8
WORLD_UP = np.array((0.0, 0.0, 1.0), dtype=np.float64)
WORLD_Y = np.array((0.0, 1.0, 0.0), dtype=np.float64)
WORLD_X = np.array((1.0, 0.0, 0.0), dtype=np.float64)

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


def get_scene_view_config(
    view_name="side",
    *,
    view_dir=None,
    up_hint=None,
    roll_degrees=0.0,
    armature_obj=None,
):
    """Return a view config for the scene.

    Orientation is resolved geometrically at import time by
    ``normalize_model_orientation`` (it rotates the rig to the canonical forward
    using the structural *deform* root, ignoring any control rig). There is no
    name-based left/right mirror flip here: the anatomical handedness of a
    bilaterally symmetric humanoid is not recoverable from geometry alone
    (its only asymmetry reference is bone names), so guessing it from ``_l``/``_r``
    tokens was rig-format-specific and is intentionally gone. ``armature_obj`` is
    retained for callers and any future geometry-based view adaptation.
    """
    _ = armature_obj
    config = get_view_config(
        view_name=view_name,
        view_dir=view_dir,
        up_hint=up_hint,
        roll_degrees=roll_degrees,
    )
    config["auto_lateral_flip"] = False
    config["lateral_sign"] = None
    return config


def _project_projection_space_direction(direction_3d, view_cfg):
    """Project a 3D direction onto the 2D view plane."""
    direction_3d = np.asarray(direction_3d, dtype=np.float64)
    basis_2d = np.asarray(view_cfg["basis_2d"], dtype=np.float64)
    if direction_3d.ndim == 1:
        return basis_2d @ direction_3d
    return direction_3d @ basis_2d.T


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


def project_point_ortho(point_3d, view_cfg, projection_inverse=None):
    """Project a 3D point into the configured 2D plane."""
    projected = _transform_point_to_projection_space(
        point_3d,
        projection_inverse=projection_inverse,
    )
    projected_2d = _project_projection_space_direction(projected, view_cfg)
    return (float(projected_2d[0]), float(projected_2d[1]))


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
