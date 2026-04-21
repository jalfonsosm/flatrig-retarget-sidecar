"""Blender worker for scene inspection and normalization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import bpy
import mathutils
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
}


def _normalize_vector_3d(vector, label="vector"):
    """Normalize a 3D vector."""
    v = mathutils.Vector(vector)
    length = v.length
    if length < 1e-10:
        raise ValueError(f"{label} cannot be zero vector")
    return v.normalized()


def _build_view_config(
    view_preset: str = "front",
    view_dir: Optional[tuple] = None,
    view_up: Optional[tuple] = None,
    view_roll: float = 0.0,
) -> dict:
    """Build a view configuration dict compatible with projection helpers."""
    if view_dir is not None:
        view_dir = _normalize_vector_3d(view_dir, "view_dir")
        right_axis = view_dir.orthogonal()
        up_axis = view_dir.cross(right_axis)
    else:
        preset = VIEW_PRESETS.get(view_preset, VIEW_PRESETS["front"])
        view_dir = mathutils.Vector(preset["view_dir"])
        right_axis = mathutils.Vector(preset["right_axis"])
        up_axis = mathutils.Vector(preset["up_axis"])
    
    if view_up is not None:
        up_hint = mathutils.Vector(view_up).normalized()
        right_axis = up_hint.cross(view_dir).normalized()
        if right_axis.length < 1e-6:
            right_axis = view_dir.orthogonal()
        up_axis = view_dir.cross(right_axis).normalized()
    
    # Apply roll
    if abs(view_roll) > 1e-6:
        roll_rad = math.radians(view_roll)
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
        new_right = mathutils.Vector([
            cos_r * right_axis.x - sin_r * up_axis.x,
            cos_r * right_axis.y - sin_r * up_axis.y,
            cos_r * right_axis.z - sin_r * up_axis.z,
        ])
        new_up = mathutils.Vector([
            sin_r * right_axis.x + cos_r * up_axis.x,
            sin_r * right_axis.y + cos_r * up_axis.y,
            sin_r * right_axis.z + cos_r * up_axis.z,
        ])
        right_axis = new_right.normalized()
        up_axis = new_up.normalized()
    
    # Build 3D basis matrix (right, up, -view)
    basis_3d = mathutils.Matrix([
        [right_axis.x, right_axis.y, right_axis.z],
        [up_axis.x, up_axis.y, up_axis.z],
        [-view_dir.x, -view_dir.y, -view_dir.z],
    ])
    
    # Build 2D basis (x, y) from right and up
    basis_2d = mathutils.Matrix([
        [right_axis.x, right_axis.y],
        [up_axis.x, up_axis.y],
    ])
    
    return {
        "view_dir": (view_dir.x, view_dir.y, view_dir.z),
        "right_axis": (right_axis.x, right_axis.y, right_axis.z),
        "up_axis": (up_axis.x, up_axis.y, up_axis.z),
        "depth_axis": (-view_dir.x, -view_dir.y, -view_dir.z),
        "basis_2d": [[basis_2d[0][0], basis_2d[0][1]], [basis_2d[1][0], basis_2d[1][1]]],
        "basis_3d": [[basis_3d[0][0], basis_3d[0][1], basis_3d[0][2]],
                     [basis_3d[1][0], basis_3d[1][1], basis_3d[1][2]],
                     [basis_3d[2][0], basis_3d[2][1], basis_3d[2][2]]],
        "roll_degrees": view_roll,
    }


def _project_projection_space_direction(direction_3d, view_cfg):
    """Project a 3D direction onto the 2D view plane."""
    basis_2d = view_cfg["basis_2d"]
    dx = direction_3d[0]
    dy = direction_3d[1]
    dz = direction_3d[2]
    return (
        basis_2d[0][0] * dx + basis_2d[0][1] * dy,
        basis_2d[1][0] * dx + basis_2d[1][1] * dy,
    )


def project_point_ortho(point_3d, view_cfg, projection_inverse=None):
    """Project a 3D point into the configured 2D plane.
    """
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
    
    point_h = np.concatenate(
        (point_3d, np.array((1.0,), dtype=np.float64)),
        axis=0,
    )
    projection_inverse = np.asarray(projection_inverse, dtype=np.float64)
    if projection_inverse.ndim == 2:
        transformed = projection_inverse @ point_h
    else:
        transformed = np.einsum("ij,j->i", projection_inverse, point_h)
    return transformed[:3]


def parse_args() -> argparse.Namespace:
    try:
        separator_index = sys.argv.index("--")
        script_args = sys.argv[separator_index + 1 :]
    except ValueError:
        script_args = []

    parser = argparse.ArgumentParser(description="Inspect or convert 3D sources for flatRig.")
    parser.add_argument("command", choices=("inspect", "convert", "load-scene"))
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-format", default="glb", choices=("glb",))
    # Projection parameters for load-scene
    parser.add_argument("--view-preset", default="front",
                        choices=list(VIEW_PRESETS.keys()) + list(VIEW_PRESETS.keys()),
                        help="View preset for projection (front, back, side, side_r, top, bottom)")
    parser.add_argument("--view-dir", default=None,
                        help="Custom view direction as 'x,y,z' tuple")
    parser.add_argument("--view-up", default=None,
                        help="Custom view up hint as 'x,y,z' tuple")
    parser.add_argument("--view-roll", type=float, default=0.0,
                        help="View roll in degrees")
    parser.add_argument("--projected", action="store_true", default=False,
                        help="Output projected 2D data instead of world-space 3D")
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


def extract_skeleton_data_with_projection(
    armature_obj,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
) -> dict[str, Any]:
    """Extract skeleton hierarchy with projected 2D bone transforms.
    
    This function outputs bones with x, y (projected 2D positions) and rotation
    instead of world-space 3D coordinates.
    
    Args:
        armature_obj: Blender armature object
        view_preset: View preset name (front, back, side, side_r, top, bottom)
        view_dir: Custom view direction vector (overrides preset)
        view_up: Custom up hint vector
        view_roll: View roll in degrees
    
    Returns:
        dict with bone_count and bones list, where each bone has:
        - name: bone name
        - parent: parent bone name (None for root)
        - index: bone index
        - head: [x, y] projected 2D head position
        - segment: [dx, dy] segment direction
        - length: bone length
        - rotation_world: world rotation in degrees
        - connected: whether bone is connected to parent
        - x, y: local 2D position (relative to parent)
        - rotation: local rotation in degrees
    """
    if armature_obj is None:
        return {"bone_count": 0, "bones": []}
    
    # Build view configuration
    view_cfg = _build_view_config(
        view_preset=view_preset,
        view_dir=view_dir,
        view_up=view_up,
        view_roll=view_roll,
    )
    
    # Get all bones in topological order (parents before children)
    bone_order = _topological_sort_bones(armature_obj)
    
    bones_data = []
    world_cache = {}  # For computing local positions
    
    for idx, bone_name in enumerate(bone_order):
        bone = armature_obj.data.bones[bone_name]
        
        # Get parent name and index
        parent_name = None
        parent_index = -1
        if bone.parent:
            parent_name = bone.parent.name
            for i, b in enumerate(armature_obj.data.bones):
                if b.name == parent_name:
                    parent_index = i
                    break
        
        # Get world-space head and tail
        head_world = armature_obj.matrix_world @ bone.head_local
        tail_world = armature_obj.matrix_world @ bone.tail_local
        
        # Project to 2D
        head_2d = project_point_ortho([head_world.x, head_world.y, head_world.z], view_cfg)
        tail_2d = project_point_ortho([tail_world.x, tail_world.y, tail_world.z], view_cfg)
        
        # Compute segment direction and length in 2D
        seg_dx = tail_2d[0] - head_2d[0]
        seg_dy = tail_2d[1] - head_2d[1]
        seg_length = math.hypot(seg_dx, seg_dy)
        world_rotation = math.degrees(math.atan2(seg_dy, seg_dx)) if seg_length > 1e-6 else 0.0
        
        # Compute local position and rotation relative to parent
        if parent_name is not None and parent_name in world_cache:
            parent_state = world_cache[parent_name]
            parent_head_2d = parent_state["head_2d"]
            parent_matrix = parent_state["matrix"]
            
            # inv_parent @ (head - parent_head)
            det = parent_matrix[0] * parent_matrix[3] - parent_matrix[1] * parent_matrix[2]
            if abs(det) < 1e-6:
                det = 1.0
            inv_parent = [
                parent_matrix[3] / det, -parent_matrix[1] / det,
                -parent_matrix[2] / det, parent_matrix[0] / det
            ]
            
            dx = head_2d[0] - parent_head_2d[0]
            dy = head_2d[1] - parent_head_2d[1]
            local_x = inv_parent[0] * dx + inv_parent[2] * dy
            local_y = inv_parent[1] * dx + inv_parent[3] * dy
            
            # Compute world_x_axis and local_x_axis
            if seg_length > 1e-6:
                world_x_axis = [seg_dx / seg_length, seg_dy / seg_length]
            else:
                world_x_axis = [1.0, 0.0]
            
            # local_x_axis = inv_parent @ world_x_axis
            local_x_axis = [
                inv_parent[0] * world_x_axis[0] + inv_parent[2] * world_x_axis[1],
                inv_parent[1] * world_x_axis[0] + inv_parent[3] * world_x_axis[1],
            ]
            local_rotation = math.degrees(math.atan2(local_x_axis[1], local_x_axis[0]))
            
            # Build world matrix for child bone
            cos_r = math.cos(math.radians(local_rotation))
            sin_r = math.sin(math.radians(local_rotation))
            world_matrix = [
                cos_r * parent_matrix[0] - sin_r * parent_matrix[2],
                cos_r * parent_matrix[1] - sin_r * parent_matrix[3],
                sin_r * parent_matrix[0] + cos_r * parent_matrix[2],
                sin_r * parent_matrix[1] + cos_r * parent_matrix[3],
            ]
        else:
            # Root bone: local = world
            local_x = head_2d[0]
            local_y = head_2d[1]
            local_rotation = world_rotation
            
            # Build world matrix for root bone (2x2 rotation + translation encoded)
            cos_r = math.cos(math.radians(world_rotation))
            sin_r = math.sin(math.radians(world_rotation))
            world_matrix = [cos_r, -sin_r, sin_r, cos_r]
        
        bones_data.append({
            "name": bone.name,
            "parent": parent_name,
            "index": idx,
            "head": list(head_2d),
            "segment": [round(seg_dx, 4), round(seg_dy, 4)],
            "length": round(seg_length, 4),
            "rotation_world": round(world_rotation, 2),
            "connected": (seg_length > 1e-4),
            "x": round(local_x, 4),
            "y": round(local_y, 4),
            "rotation": round(local_rotation, 2),
            "world_matrix": world_matrix,
        })
        
        # Cache world state for children
        world_cache[bone.name] = {
            "head_2d": head_2d,
            "matrix": world_matrix,
        }
    
    return {
        "bone_count": len(bones_data),
        "bones": bones_data,
        "view_preset": view_preset,
        "view_roll": view_roll,
    }


def _topological_sort_bones(armature_obj) -> list[str]:
    """Sort bone names so parents come before children."""
    bones = list(armature_obj.data.bones)
    bone_index = {bone.name: i for i, bone in enumerate(bones)}
    
    # Build adjacency list
    children = {bone.name: [] for bone in bones}
    in_degree = {bone.name: 0 for bone in bones}
    
    for bone in bones:
        if bone.parent:
            children[bone.parent.name].append(bone.name)
            in_degree[bone.name] += 1
    
    # Kahn's algorithm
    queue = [name for name, degree in in_degree.items() if degree == 0]
    result = []
    
    while queue:
        # Sort for consistent ordering
        queue.sort(key=lambda n: bone_index[n])
        name = queue.pop(0)
        result.append(name)
        
        for child in children[name]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    
    return result


def extract_skeleton_data(armature_obj) -> dict[str, Any]:
    """Extract skeleton hierarchy and bone transforms (world-space).
    
    This function outputs bones with world-space 3D coordinates (world_position,
    world_tail, local_rotation).
    """
    if armature_obj is None:
        return {"bone_count": 0, "bones": []}
    
    bones_data = []
    
    for bone in armature_obj.data.bones:
        # Get parent index
        parent_index = -1
        if bone.parent:
            for i, b in enumerate(armature_obj.data.bones):
                if b.name == bone.parent.name:
                    parent_index = i
                    break
        
        # Get local transform and convert to WORLD space
        head_world = armature_obj.matrix_world @ bone.head_local
        tail_world = armature_obj.matrix_world @ bone.tail_local
        
        world_head = [head_world.x, head_world.y, head_world.z]
        world_tail = [tail_world.x, tail_world.y, tail_world.z]
        
        # Get roll - bone.roll was removed in Blender 5.0
        # Calculate roll from the Y axis of the bone matrix
        try:
            roll = bone.roll
        except AttributeError:
            # Blender 5.0+: calculate roll from matrix
            y_axis = mathutils.Vector((0, 1, 0))
            y_axis = y_axis @ bone.matrix.to_3x3()
            roll = math.atan2(y_axis.x, y_axis.y)
        
        # Get world matrix for this bone
        world_matrix = get_bone_world_matrix(armature_obj, bone.name)
        
        bones_data.append({
            "name": bone.name,
            "parent_index": parent_index,
            "world_position": world_head,
            "world_tail": world_tail,
            "local_rotation": roll,
            "local_scale": [bone.length, bone.length, bone.length],
            "world_matrix": world_matrix,
        })
    
    return {
        "bone_count": len(bones_data),
        "bones": bones_data,
    }


def get_action_fcurves(action) -> list:
    """Get fcurves from an action, handling Blender 5.0's layered animation system.
    
    In Blender 5.0+, animation data is stored in:
    - action.layers[].strips[].channelbags[].fcurves
    instead of action.fcurves directly.
    """
    # First try the old API (for compatibility with older Blender versions)
    fcurves = getattr(action, "fcurves", None)
    if fcurves is not None and len(fcurves) > 0:
        return list(fcurves)
    
    # Blender 5.0+ layered animation system
    if hasattr(action, "layers"):
        for layer in action.layers:
            if hasattr(layer, "strips"):
                for strip in layer.strips:
                    if hasattr(strip, "channelbags"):
                        for channelbag in strip.channelbags:
                            fc = getattr(channelbag, "fcurves", None)
                            if fc is not None and len(fc) > 0:
                                return list(fc)
    
    return []


def extract_animation_data(armature_obj) -> dict[str, Any]:
    """Extract animation clips from the armature.
    
    Handles Blender 5.0 API changes where action.fcurves may be None
    and animation data is accessed via action.layers[].strips[].channelbags[].fcurves.
    """
    if armature_obj is None:
        return {"animation_count": 0, "animations": []}
    
    animations = []
    
    for action in bpy.data.actions:
        if not is_pose_action(action):
            continue
        
        # Get frame range
        frame_start, frame_end = action.frame_range
        
        # Extract keyframes for bones
        keyframes = []
        
        # Get fcurves using the helper function
        fcurves = get_action_fcurves(action)
        
        # Group fcurves by bone
        bone_keyframes: dict[str, list[dict[str, Any]]] = {}
        for fcurve in fcurves:
            data_path = str(fcurve.data_path or "")
            
            # Parse pose.bones["BoneName"].property
            if not (data_path.startswith('pose.bones["') or data_path.startswith("pose.bones['")):
                continue
            
            # Extract bone name
            try:
                brace_start = data_path.index('["') + 2
                brace_end = data_path.index('"]')
                if brace_end < 0:
                    brace_end = data_path.index("']")
                bone_name = data_path[brace_start:brace_end]
            except (ValueError, IndexError):
                continue
            
            # Get property being animated
            prop = data_path[brace_end + 3:] if brace_end + 3 < len(data_path) else ""
            
            # Get array index if applicable
            array_index = fcurve.array_index
            
            # Sample keyframes
            try:
                keyframe_points = fcurve.keyframe_points
                for kf in keyframe_points:
                    frame = int(round(kf.co.x))
                    value = kf.co.y
                    
                    # Group by bone
                    if bone_name not in bone_keyframes:
                        bone_keyframes[bone_name] = []
                    bone_keyframes[bone_name].append({
                        "bone_name": bone_name,
                        "frame": frame,
                        "property": prop,
                        "array_index": array_index,
                        "value": value,
                    })
            except AttributeError:
                # Blender 5.0+: keyframe_points may not exist, try reading the entire points array
                try:
                    points = fcurve.points
                    for pt in points:
                        frame, value = pt.co
                        frame = int(round(frame))
                        if bone_name not in bone_keyframes:
                            bone_keyframes[bone_name] = []
                        bone_keyframes[bone_name].append({
                            "bone_name": bone_name,
                            "frame": frame,
                            "property": prop,
                            "array_index": array_index,
                            "value": value,
                        })
                except Exception:
                    pass
        
        # Flatten bone_keyframes into keyframes list
        for bone_name, kf_list in bone_keyframes.items():
            keyframes.extend(kf_list)
        
        animations.append({
            "name": action.name,
            "frame_start": int(round(frame_start)),
            "frame_end": int(round(frame_end)),
            "keyframes": keyframes,
        })
    
    return {
        "animation_count": len(animations),
        "animations": animations,
    }


def load_scene(
    source_path: str,
    output_path: str,
    projected: bool = False,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
) -> dict[str, object]:
    """Load a 3D source and export the full scene data as JSON.
    
    This function uses Blender's native import which properly handles all
    transform hierarchies.
    
    Args:
        source_path: Path to the 3D source file
        output_path: Path for the output JSON (not used in function, for CLI compatibility)
        projected: If True, output projected 2D skeleton data
        view_preset: View preset for projection (front, back, side, side_r, top, bottom)
        view_dir: Custom view direction tuple
        view_up: Custom view up hint tuple
        view_roll: View roll in degrees
    
    Returns:
        dict with scene data including mesh, skeleton, and animations
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)
    
    mesh_obj, armature_obj = find_mesh_and_armature()
    
    # Choose skeleton extraction method based on projected flag
    if projected:
        skeleton_data = extract_skeleton_data_with_projection(
            armature_obj,
            view_preset=view_preset,
            view_dir=view_dir,
            view_up=view_up,
            view_roll=view_roll,
        )
    else:
        skeleton_data = extract_skeleton_data(armature_obj)
    
    scene_data = {
        "ok": True,
        "detail": "loaded",
        "source": source_path,
        "format": Path(source_path).suffix.lower().lstrip("."),
        "projected": projected,
        "mesh": extract_mesh_data(mesh_obj),
        "skeleton": skeleton_data,
        "animations": extract_animation_data(armature_obj),
    }
    
    return scene_data


def main() -> None:
    args = parse_args()
    source_path = str(Path(args.source).expanduser().resolve())
    output_path = Path(args.output).expanduser().resolve()

    payload: dict[str, object]
    
    if args.command == "inspect":
        payload = inspect_source(source_path)
    elif args.command == "convert":
        payload = convert_source(source_path, str(output_path), args.target_format)
    elif args.command == "load-scene":
        # Parse view_dir and view_up if provided
        view_dir = None
        view_up = None
        if args.view_dir:
            view_dir = tuple(float(x) for x in args.view_dir.split(","))
        if args.view_up:
            view_up = tuple(float(x) for x in args.view_up.split(","))
        
        payload = load_scene(
            source_path,
            str(output_path),
            projected=args.projected,
            view_preset=args.view_preset,
            view_dir=view_dir,
            view_up=view_up,
            view_roll=args.view_roll,
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
