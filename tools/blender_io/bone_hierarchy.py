"""2D/3D bone hierarchy extraction for the Blender worker.

Extracted from ``blender_scene_io.py`` as one cohesive block: topology
annotation, inherit-mode helpers and the public hierarchy extraction functions.
"""

from __future__ import annotations

# ruff: noqa: I001

import math

try:
    import bpy
except ImportError:
    bpy = None

import numpy as np

from blender_io.bone_utils import _bone_is_connected, _topological_sort
from blender_io.math_utils import (
    SEGMENT_EPSILON,
    _build_2d_basis,
    _world_rotation_3x3,
    get_projection_reference_inverse,
    orthonormalize_2x2,
    safe_inverse_2x2,
)
from blender_json_io import (
    matrix3_to_json as _matrix3_to_json,
    vector_to_json as _vector_to_json,
)
from blender_view import project_point_ortho

TERMINAL_CHAIN_ROOT_RATIO = 0.6
TERMINAL_CHAIN_MAX_LENGTH_RATIO = 0.8
TERMINAL_CHAIN_PARENT_RATIO = 1.5
TERMINAL_CHAIN_MAX_SPAN = 6


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

    if record["child_count"] == 1:
        child_name = children[record["name"]][0]
        child = by_name[child_name]
        if child["length_ratio"] > TERMINAL_CHAIN_ROOT_RATIO:
            return False

    parent = by_name[record["parent"]]
    if (
        record["parent_child_count"] <= 1
        and record["parent_length_ratio"] < TERMINAL_CHAIN_PARENT_RATIO
    ):
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

    positive_lengths = sorted(
        record["length"] for record in records if record["length"] > SEGMENT_EPSILON
    )
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


def extract_setup_bone_hierarchy(
    armature,
    view_cfg,
    *,
    source_frame=None,
    use_rest_pose=False,
    projection_space="world",
    projection_reference_root=None,
    bind_borrow_info=None,
):
    """Extract setup bones from the target rig's retargeted donor pose.

    `_copy_source_pose_to_target` writes rotations only and explicitly clears
    pose-bone locations/scales. Evaluated heads/tails therefore come from the
    target rig's own rest offsets under donor rotations, which is the pose the
    mesh is rendered in. Do not splice rest-projected 2D lengths into this pose:
    projection foreshortening changes with joint rotation and the resulting
    hybrid skeleton no longer lines up with the rendered sprites.
    """
    bind_borrow_info = bind_borrow_info or {}
    if bind_borrow_info.get("applied") and not use_rest_pose:
        bind_borrow_info["setup_pose_mode"] = "target_retargeted_pose_no_translations"
        bind_borrow_info["setup_morphology_source"] = "target_fk_offsets"
        bind_borrow_info["setup_rotation_source"] = "donor_retarget"
    return extract_bone_hierarchy(
        armature,
        view_cfg,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
        projection_space=projection_space,
        projection_reference_root=projection_reference_root,
    )


def extract_bone_hierarchy_3d(armature, source_frame=None, use_rest_pose=False):
    """Extract 3D bone heads/tails + world rotations for skinning/preview.

    The `world_rotation` field is a 3x3 row-major matrix giving the bone's
    world-space orientation in the evaluated pose. Native callers use this as
    the bind matrix when doing linear blend skinning of sprite vertices —
    deriving the bind rotation from head/tail alone loses the bone roll, which
    causes z-fighting on parts whose vertices are off the bone axis.
    """
    if armature is None:
        return []

    scene = bpy.context.scene
    if source_frame is None:
        source_frame = scene.frame_start
    scene.frame_set(source_frame)
    bpy.context.view_layer.update()

    armature_world = armature.matrix_world
    bones = []
    for idx, bone_name in enumerate(_topological_sort(armature)):
        rest_bone = armature.data.bones[bone_name]
        pose_bone = armature.pose.bones.get(bone_name)
        if use_rest_pose or pose_bone is None:
            head_world = armature_world @ rest_bone.head_local
            tail_world = armature_world @ rest_bone.tail_local
            # In rest pose the bone matrix is its rest local matrix (in armature
            # space), so apply armature world to get world rotation.
            bone_local_matrix = rest_bone.matrix_local
            world_matrix_3x3 = _world_rotation_3x3(armature_world @ bone_local_matrix)
        else:
            head_world = armature_world @ pose_bone.head
            tail_world = armature_world @ pose_bone.tail
            # pose_bone.matrix is the bone's transform in armature space for
            # the current evaluated pose; armature.matrix_world brings it to
            # world coordinates.
            world_matrix_3x3 = _world_rotation_3x3(armature_world @ pose_bone.matrix)
        bones.append(
            {
                "name": bone_name,
                "parent": rest_bone.parent.name if rest_bone.parent else None,
                "index": idx,
                "head": _vector_to_json(head_world),
                "tail": _vector_to_json(tail_world),
                "length": float((tail_world - head_world).length),
                "world_rotation": _matrix3_to_json(world_matrix_3x3),
            }
        )
    return bones


def extract_setup_bone_hierarchy_3d(
    armature,
    *,
    source_frame=None,
    use_rest_pose=False,
    bind_borrow_info=None,
):
    """Extract 3D setup bones from the target rig's retargeted donor pose."""
    bind_borrow_info = bind_borrow_info or {}
    if bind_borrow_info.get("applied") and not use_rest_pose:
        bind_borrow_info["setup_3d_pose_mode"] = "target_retargeted_pose_no_translations"
        bind_borrow_info["setup_3d_morphology_source"] = "target_fk_offsets"
        bind_borrow_info["setup_3d_rotation_source"] = "donor_retarget"
    return extract_bone_hierarchy_3d(
        armature,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
    )
