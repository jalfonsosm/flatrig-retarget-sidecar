"""Blender worker for scene inspection and normalization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
import mathutils


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
    
    return result


def extract_skeleton_data(armature_obj) -> dict[str, Any]:
    """Extract skeleton hierarchy and bone transforms."""
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
        
        # Get local transform
        head = bone.head_local
        tail = bone.tail_local
        
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
            "local_position": [head.x, head.y, head.z],
            "local_rotation": roll,
            "local_scale": [bone.length, bone.length, bone.length],
            "world_matrix": world_matrix,
        })
    
    return {
        "bone_count": len(bones_data),
        "bones": bones_data,
    }


def extract_animation_data(armature_obj) -> dict[str, Any]:
    """Extract animation clips from the armature.
    
    Handles Blender 5.0 API changes where action.fcurves may be None
    and animation data is accessed differently.
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
        
        # Get fcurves - API changed in Blender 5.0
        # Blender 5.0 stores animation data differently
        fcurves = getattr(action, "fcurves", None) or []
        
        # If no fcurves directly on action, try to get them from id_data
        if not fcurves:
            id_data = getattr(action, "id_data", None)
            if id_data is not None:
                fcurves = getattr(id_data, "fcurves", None) or []
        
        # Blender 5.0+: check if animation data is on the armature itself
        if not fcurves and armature_obj and armature_obj.animation_data:
            anim_data = armature_obj.animation_data
            if hasattr(anim_data, "action") and anim_data.action:
                fcurves = getattr(anim_data.action, "fcurves", None) or []
        
        # If still no fcurves, check the action's curves directly via all_items
        if not fcurves:
            try:
                if hasattr(action, "curves"):
                    fcurves = list(action.curves) or []
            except (TypeError, AttributeError):
                pass
        
        if not fcurves:
            # Last resort: try to get FCurves via the Blender RNA path
            try:
                if hasattr(action, "bl_rna"):
                    fcurve_rna = action.bl_rna.properties.get("fcurves")
                    if fcurve_rna:
                        fcurves = getattr(action, "fcurves", []) or []
            except Exception:
                pass
        
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


def load_scene(source_path: str, output_path: str) -> dict[str, object]:
    """Load a 3D source and export the full scene data as JSON.
    
    This function uses Blender's
    native import which properly handles all transform hierarchies.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)
    
    mesh_obj, armature_obj = find_mesh_and_armature()
    
    scene_data = {
        "ok": True,
        "detail": "loaded",
        "source": source_path,
        "format": Path(source_path).suffix.lower().lstrip("."),
        "mesh": extract_mesh_data(mesh_obj),
        "skeleton": extract_skeleton_data(armature_obj),
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
        payload = load_scene(source_path, str(output_path))
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
