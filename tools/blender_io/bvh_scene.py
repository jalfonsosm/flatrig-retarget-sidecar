"""Blender armature sampling helpers for BVH export."""

from __future__ import annotations

# ruff: noqa: I001

import math

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None

from blender_io.bone_utils import _bvh_export_name, _sanitize_bvh_name, _topological_sort
from blender_io.math_utils import (
    VECTOR_EPSILON,
    _armature_uniform_scale,
    _armature_world_rotation,
    _matrix_xyz_euler_degrees,
    _rotation_between_vectors,
    _world_rotation_3x3,
)
from blender_io.scene_pose import (
    _restore_scene_armature_pose_positions,
    _set_scene_armatures_rest_pose,
)
from blender_json_io import (
    matrix3_to_json as _matrix3_to_json,
    vector_from_json as _vector_from_json,
    vector_to_json as _vector_to_json,
)


def _build_3d_bvh_layout(armature_obj, source_frame=None, use_rest_pose=True):
    """Return portable BVH joints for a Blender armature."""
    scene = bpy.context.scene
    if source_frame is not None:
        scene.frame_set(int(source_frame))
    rest_pose_state = _set_scene_armatures_rest_pose(scene) if use_rest_pose else []
    bpy.context.view_layer.update()

    bone_order = _topological_sort(armature_obj)
    bone_order_index = {name: index for index, name in enumerate(bone_order)}
    armature_scale = _armature_uniform_scale(armature_obj)
    armature_world = armature_obj.matrix_world.copy()
    armature_linear = armature_world.to_3x3()
    armature_rotation = _armature_world_rotation(armature_obj)
    root_bones = [
        bone
        for bone in armature_obj.data.bones
        if bone.parent is None and bone.name in bone_order_index
    ]
    root_bones.sort(key=lambda bone: bone_order_index[bone.name])
    use_synthetic_root = len(root_bones) > 1

    used_names = set()
    joints = []
    original_to_bvh = {}
    bvh_to_original = {}
    original_to_matching = {}
    matching_to_bvh = {}
    name_to_index = {}

    def bone_world_head_tail(rest_bone):
        if not use_rest_pose:
            pose_bone = armature_obj.pose.bones.get(rest_bone.name)
            if pose_bone is not None:
                return armature_world @ pose_bone.head, armature_world @ pose_bone.tail
        return armature_world @ rest_bone.head_local, armature_world @ rest_bone.tail_local

    try:
        if use_synthetic_root:
            matching_name = _sanitize_bvh_name("sidecar_root", used_names)
            bvh_name = _bvh_export_name(matching_name, 0)
            joints.append(
                {
                    "index": 0,
                    "name": None,
                    "matching_name": matching_name,
                    "bvh_name": bvh_name,
                    "parent_index": -1,
                    "parent_bvh_name": None,
                    "offset": [0.0, 0.0, 0.0],
                    "head": [0.0, 0.0, 0.0],
                    "tail": [0.0, 0.0, 0.0],
                    "tail_offset": [1.0, 0.0, 0.0],
                    "length": 0.0,
                    "synthetic": True,
                }
            )
            matching_to_bvh[matching_name] = bvh_name
            bvh_to_original[bvh_name] = None

        for bone_name in bone_order:
            rest_bone = armature_obj.data.bones[bone_name]
            index = len(joints)
            matching_name = _sanitize_bvh_name(rest_bone.name, used_names)
            bvh_name = _bvh_export_name(matching_name, index)
            parent_index = -1
            parent_bvh_name = None
            if rest_bone.parent is not None:
                parent_index = name_to_index.get(rest_bone.parent.name, -1)
            elif use_synthetic_root:
                parent_index = 0
            if parent_index >= 0:
                parent_bvh_name = joints[parent_index]["bvh_name"]

            head_vec, tail_vec = bone_world_head_tail(rest_bone)
            if rest_bone.parent is None:
                offset_vec = head_vec
            else:
                parent_head_vec, _parent_tail_vec = bone_world_head_tail(rest_bone.parent)
                offset_vec = head_vec - parent_head_vec
            tail_offset_vec = tail_vec - head_vec
            length = float(tail_offset_vec.length)
            rest_world_rotation = _world_rotation_3x3(
                armature_world @ rest_bone.matrix_local
            )

            joint = {
                "index": index,
                "name": rest_bone.name,
                "matching_name": matching_name,
                "bvh_name": bvh_name,
                "parent_index": int(parent_index),
                "parent_bvh_name": parent_bvh_name,
                "offset": _vector_to_json(offset_vec),
                "head": _vector_to_json(head_vec),
                "tail": _vector_to_json(tail_vec),
                "tail_offset": _vector_to_json(tail_offset_vec),
                "rest_world_rotation": _matrix3_to_json(rest_world_rotation),
                "length": length,
                "synthetic": False,
            }
            joints.append(joint)
            name_to_index[rest_bone.name] = index
            original_to_bvh[rest_bone.name] = bvh_name
            bvh_to_original[bvh_name] = rest_bone.name
            original_to_matching[rest_bone.name] = matching_name
            matching_to_bvh[matching_name] = bvh_name

        return {
            "joints": joints,
            "original_to_bvh": original_to_bvh,
            "bvh_to_original": bvh_to_original,
            "original_to_matching": original_to_matching,
            "matching_to_bvh": matching_to_bvh,
            "root_bvh_name": joints[0]["bvh_name"] if joints else None,
            "root_matching_name": joints[0]["matching_name"] if joints else None,
            "coordinate_scale": armature_scale,
            "coordinate_linear": _matrix3_to_json(armature_linear),
            "coordinate_rotation": _matrix3_to_json(armature_rotation),
        }
    finally:
        _restore_scene_armature_pose_positions(rest_pose_state)


def _pose_bone_world_head_tail(armature_obj, pose_bone):
    armature_world = armature_obj.matrix_world
    return armature_world @ pose_bone.head, armature_world @ pose_bone.tail


def _sample_action_frames(action, scene, fps, frame_start=None, frame_end=None):
    scene_fps = float(scene.render.fps) / max(float(scene.render.fps_base), VECTOR_EPSILON)
    start, end = action.frame_range if action is not None else (scene.frame_start, scene.frame_end)
    if frame_start is not None:
        start = float(frame_start)
    if frame_end is not None:
        end = float(frame_end)
    if end < start:
        end = start
    duration_seconds = max((float(end) - float(start)) / max(scene_fps, VECTOR_EPSILON), 1.0 / fps)
    frame_count = max(2, int(math.floor(duration_seconds * fps)) + 1)
    frame_step = scene_fps / fps
    return [float(start) + index * frame_step for index in range(frame_count)]


def _set_scene_frame_float(scene, frame_value):
    frame_int = int(math.floor(float(frame_value)))
    subframe = float(frame_value) - float(frame_int)
    scene.frame_set(frame_int, subframe=subframe)
    bpy.context.view_layer.update()


def _collect_3d_bvh_frames(armature_obj, layout, sample_frames, fps):
    scene = bpy.context.scene
    positions = []
    rotations = []
    joints = list(layout["joints"])
    for frame_value in sample_frames:
        _set_scene_frame_float(scene, frame_value)
        frame_positions = []
        frame_rotations = []
        world_cache = [None] * len(joints)
        for joint in joints:
            joint_index = int(joint.get("index", len(frame_positions) // 3))
            parent_index = int(joint.get("parent_index", -1))
            if joint.get("synthetic"):
                world_cache[joint_index] = {
                    "head": _vector_from_json(joint.get("head")),
                    "rotation": mathutils.Matrix.Identity(3),
                }
                frame_positions.extend((0.0, 0.0, 0.0))
                frame_rotations.extend((0.0, 0.0, 0.0))
                continue
            pose_bone = armature_obj.pose.bones.get(joint["name"])
            if pose_bone is None:
                world_cache[joint_index] = None
                frame_positions.extend((0.0, 0.0, 0.0))
                frame_rotations.extend((0.0, 0.0, 0.0))
                continue

            rest_offset = _vector_from_json(joint.get("offset"))
            tail_offset = _vector_from_json(joint.get("tail_offset"), fallback=(1.0, 0.0, 0.0))
            head_world, tail_world = _pose_bone_world_head_tail(armature_obj, pose_bone)
            posed_tail_axis_world = tail_world - head_world

            parent_state = (
                world_cache[parent_index] if 0 <= parent_index < len(world_cache) else None
            )
            if parent_state is not None:
                parent_rotation = parent_state["rotation"]
                parent_rotation_inv = parent_rotation.inverted()
                local_position = (
                    parent_rotation_inv @ (head_world - parent_state["head"]) - rest_offset
                )
                desired_axis_parent = parent_rotation_inv @ posed_tail_axis_world
                local_rotation = _rotation_between_vectors(tail_offset, desired_axis_parent)
                world_rotation = parent_rotation @ local_rotation
            else:
                local_position = head_world - rest_offset
                local_rotation = _rotation_between_vectors(tail_offset, posed_tail_axis_world)
                world_rotation = local_rotation

            world_cache[joint_index] = {
                "head": head_world,
                "rotation": world_rotation,
            }
            frame_positions.extend(
                (
                    float(local_position.x),
                    float(local_position.y),
                    float(local_position.z),
                )
            )
            frame_rotations.extend(_matrix_xyz_euler_degrees(local_rotation))
        positions.append(frame_positions)
        rotations.append(frame_rotations)
    return positions, rotations
