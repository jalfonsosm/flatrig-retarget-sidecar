"""
Animation timeline extraction for Spine 2D (Blender worker side).

Extracts bone animation keyframes from the 3D armature and converts them to
Spine's relative-to-setup-pose format. The pure 2D math (projection
stabilization, leaf-IK, root motion math, keyframe reduction, exported-pose
sampling) lives privately in ``flatrig_private.animation_math`` (Cython
obfuscated at build time) and is re-exported here so existing
``from flatrig.animation import ...`` call sites keep working. This module keeps
only the Blender (``bpy``/armature/``scene.frame_set``) bound sampling and
orchestration.
"""

import math

import bpy
import numpy as np

from flatrig._sidecar_import import orthonormalize_2x2, orthonormalize_3x3, safe_inverse_2x2
from flatrig.projection import get_projection_reference_inverse
from flatrig_private.projection_math import (
    project_direction_ortho,
    project_point_ortho,
    transform_direction_to_projection_space,
)

# Pure 2D animation math (no ``bpy``). Re-exported so callers that still do
# ``from flatrig.animation import _stabilize_frame_local_poses_2d`` (etc.)
# resolve here, and so the ``bpy``-bound functions below find the helpers
# (``_clamp``, ``_build_2d_basis``, ``BONE_SCALE_EPSILON``, ...) in module scope.
from flatrig_private.animation_math import *  # noqa: F401,F403


def extract_bone_animations(
    armature,
    bones_setup,
    view_cfg,
    fps=30.0,
    frame_start=None,
    frame_end=None,
    sample_substeps=2,
    optimize_animation_keys=True,
    force_loop_closing_keys=False,
    projection_space="world",
    pose_mode="full",
    pose_blend=1.0,
    rotation_flatten=None,
    connected_translation=None,
    stretch_guard=None,
    leaf_ik_refine=None,
    problem_frame_filter=None,
    projection_reference_root=None,
    preserve_root_motion=False,
    preserve_root_rotation=False,
    action_names=None,
):
    """Extract one or more source actions and convert them to Spine animations."""
    animation_data = armature.animation_data_create()
    original_action = animation_data.action
    available_actions = {action.name: action for action in _list_armature_actions(armature)}
    selected_actions = []

    if action_names:
        missing_actions = [name for name in action_names if name not in available_actions]
        if missing_actions:
            available_names = ", ".join(sorted(available_actions)) or "<none>"
            missing_names = ", ".join(missing_actions)
            raise ValueError(
                f"Missing source animation(s): {missing_names}. Available actions: {available_names}"
            )
        selected_actions = [available_actions[name] for name in action_names]
    elif original_action is not None:
        selected_actions = [original_action]
    elif available_actions:
        selected_actions = [available_actions[name] for name in sorted(available_actions)]
    else:
        selected_actions = [None]

    animations = {}
    scene = bpy.context.scene
    original_frame = scene.frame_current
    original_subframe = scene.frame_subframe

    try:
        for action in selected_actions:
            local_frame_start = frame_start
            local_frame_end = frame_end
            animation_name = None
            if action is not None:
                animation_data.action = action
                animation_name = action.name
                action_start, action_end = action.frame_range
                if local_frame_start is None:
                    local_frame_start = int(round(action_start))
                if local_frame_end is None:
                    local_frame_end = int(round(action_end))
            extracted = _extract_current_action_bone_animation(
                armature,
                bones_setup,
                view_cfg,
                fps=fps,
                frame_start=local_frame_start,
                frame_end=local_frame_end,
                sample_substeps=sample_substeps,
                optimize_animation_keys=optimize_animation_keys,
                force_loop_closing_keys=force_loop_closing_keys,
                projection_space=projection_space,
                pose_mode=pose_mode,
                pose_blend=pose_blend,
                rotation_flatten=rotation_flatten,
                connected_translation=connected_translation,
                stretch_guard=stretch_guard,
                leaf_ik_refine=leaf_ik_refine,
                problem_frame_filter=problem_frame_filter,
                projection_reference_root=projection_reference_root,
                preserve_root_motion=preserve_root_motion,
                preserve_root_rotation=preserve_root_rotation,
                animation_name=animation_name,
            )
            animations.update(extracted)
    finally:
        animation_data.action = original_action
        scene.frame_set(original_frame, subframe=original_subframe)
        bpy.context.view_layer.update()

    return animations


def _is_pose_action(action):
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


def _list_armature_actions(armature):
    actions = []
    seen = set()
    active_name = (
        armature.animation_data.action.name
        if armature.animation_data and armature.animation_data.action
        else None
    )
    if active_name and armature.animation_data.action:
        actions.append(armature.animation_data.action)
        seen.add(active_name)
    for action in bpy.data.actions:
        if action.name in seen or not _is_pose_action(action):
            continue
        actions.append(action)
        seen.add(action.name)
    return actions


def _extract_current_action_bone_animation(
    armature,
    bones_setup,
    view_cfg,
    fps=30.0,
    frame_start=None,
    frame_end=None,
    sample_substeps=2,
    optimize_animation_keys=True,
    force_loop_closing_keys=False,
    projection_space="world",
    pose_mode="full",
    pose_blend=1.0,
    rotation_flatten=None,
    connected_translation=None,
    stretch_guard=None,
    leaf_ik_refine=None,
    problem_frame_filter=None,
    projection_reference_root=None,
    preserve_root_motion=False,
    preserve_root_rotation=False,
    animation_name=None,
):
    """Extract the currently active action and convert it to one Spine animation."""
    def _vec_to_json(v):
        return [float(v[0]), float(v[1]), float(v[2])]

    def _mat_to_json(m):
        return [[float(m[row][col]) for col in range(3)] for row in range(3)]

    scene = bpy.context.scene
    if frame_start is None:
        frame_start = scene.frame_start
    if frame_end is None:
        frame_end = scene.frame_end
    pose_mode = _normalize_pose_mode(pose_mode)
    pose_blend = _clamp(float(pose_blend), 0.0, 1.0)
    rotation_flatten = _prepare_rotation_flatten(rotation_flatten)
    connected_translation = _prepare_connected_translation(connected_translation)
    leaf_ik_refine = _prepare_leaf_ik_refine(leaf_ik_refine)
    problem_frame_filter = _prepare_problem_frame_filter(problem_frame_filter)
    leaf_ik_chains = _build_leaf_ik_chains(bones_setup, leaf_ik_refine)

    setup_local = {bone["name"]: bone for bone in bones_setup}
    root_motion_bone_name = next(
        (bone["name"] for bone in bones_setup if bone["parent"] is None), None
    )
    root_motion_reference_2d = _compute_root_motion_reference_2d(
        armature,
        view_cfg,
        frame_start,
        root_motion_bone_name,
        enabled=preserve_root_motion and projection_space == "root",
    )
    root_rotation_reference_deg = _compute_root_rotation_reference_2d(
        armature,
        view_cfg,
        frame_start,
        root_motion_bone_name,
        enabled=preserve_root_rotation and projection_space == "root",
    )
    local_rotation_reference = None
    if pose_mode == "local_rotation":
        scene.frame_set(frame_start, subframe=0.0)
        bpy.context.view_layer.update()
        reference_projection_inverse = get_projection_reference_inverse(
            armature,
            projection_space=projection_space,
            reference_root_matrix=projection_reference_root,
        )
        local_rotation_reference = _build_local_rotation_reference(
            armature,
            bones_setup,
            view_cfg,
            projection_inverse=reference_projection_inverse,
        )

    anim_name = animation_name or "animation"
    if anim_name == "animation" and armature.animation_data and armature.animation_data.action:
        anim_name = armature.animation_data.action.name

    bone_timelines = {}
    previous_rotation_values = {}
    previous_stable_poses = None
    previous_leaf_ik_poses = None
    previous_sample_metrics = None

    for bone_info in bones_setup:
        name = bone_info["name"]
        bone_timelines[name] = {"rotate": [], "translate": [], "scale": []}

    sample_points = []
    substeps = max(1, sample_substeps)
    for frame in range(frame_start, frame_end):
        for substep in range(substeps):
            sample_points.append(
                (
                    frame,
                    substep / substeps,
                    (frame - frame_start + substep / substeps) / fps,
                )
            )
    sample_points.append((frame_end, 0.0, (frame_end - frame_start) / fps))

    sample_records = []
    for frame, subframe, time in sample_points:
        scene.frame_set(frame, subframe=subframe)
        bpy.context.view_layer.update()
        projection_inverse = get_projection_reference_inverse(
            armature,
            projection_space=projection_space,
            reference_root_matrix=projection_reference_root,
        )
        projected_segments = _sample_projected_bone_segments_2d(
            armature,
            bones_setup,
            view_cfg,
            projection_inverse=projection_inverse,
        )
        root_motion_offset = _compute_root_motion_offset_2d(
            armature,
            view_cfg,
            root_motion_bone_name,
            root_motion_reference_2d,
            enabled=preserve_root_motion and projection_space == "root",
        )
        root_rotation_offset = _compute_root_rotation_offset_2d(
            armature,
            view_cfg,
            root_motion_bone_name,
            root_rotation_reference_deg,
            enabled=preserve_root_rotation and projection_space == "root",
        )
        if pose_mode == "blend":
            raw_rotation_poses = _compute_frame_local_bone_poses_2d(
                armature,
                bones_setup,
                view_cfg,
                projected_segments=projected_segments,
                projection_inverse=projection_inverse,
                pose_mode="rotation_only",
                rotation_flatten=rotation_flatten,
            )
            raw_full_poses = _compute_frame_local_bone_poses_2d(
                armature,
                bones_setup,
                view_cfg,
                projected_segments=projected_segments,
                projection_inverse=projection_inverse,
                pose_mode="full",
                rotation_flatten=rotation_flatten,
            )
            stable_full_poses = _stabilize_frame_local_poses_2d(
                raw_full_poses,
                previous_stable_poses,
                bones_setup,
                stretch_guard=stretch_guard,
            )
            frame_poses = _blend_frame_local_poses_2d(
                raw_rotation_poses,
                stable_full_poses,
                pose_blend,
            )
            previous_stable_poses = {name: pose.copy() for name, pose in stable_full_poses.items()}
        else:
            raw_frame_poses = _compute_frame_local_bone_poses_2d(
                armature,
                bones_setup,
                view_cfg,
                projected_segments=projected_segments,
                projection_inverse=projection_inverse,
                pose_mode=pose_mode,
                rotation_flatten=rotation_flatten,
                local_rotation_reference=local_rotation_reference,
            )
            if _is_rotation_pose_mode(pose_mode):
                frame_poses = raw_frame_poses
            else:
                frame_poses = _stabilize_frame_local_poses_2d(
                    raw_frame_poses,
                    previous_stable_poses,
                    bones_setup,
                    stretch_guard=stretch_guard,
                )
                previous_stable_poses = {name: pose.copy() for name, pose in frame_poses.items()}

        if (
            not _is_rotation_pose_mode(pose_mode)
            and leaf_ik_refine.get("enabled", False)
            and leaf_ik_chains
        ):
            frame_poses = _refine_frame_local_poses_with_leaf_ik(
                frame_poses,
                previous_leaf_ik_poses,
                bones_setup,
                projected_segments,
                leaf_ik_chains,
                leaf_ik_refine,
            )
            previous_leaf_ik_poses = {name: pose.copy() for name, pose in frame_poses.items()}
        elif leaf_ik_refine.get("enabled", False):
            previous_leaf_ik_poses = {name: pose.copy() for name, pose in frame_poses.items()}

        current_sample_metrics = _collect_problem_frame_metrics(
            bones_setup,
            projected_segments,
            frame_poses,
        )
        frame_filter = _evaluate_problem_frame_sample(
            current_sample_metrics,
            previous_sample_metrics,
            problem_frame_filter,
            time_value=time,
            fps=fps,
        )
        armature_world = armature.matrix_world
        bones_3d = []
        for bone_info in bones_setup:
            pose_bone = armature.pose.bones.get(bone_info["name"])
            if pose_bone is None:
                continue
            bones_3d.append(
                {
                    "name": bone_info["name"],
                    "head": _vec_to_json(armature_world @ pose_bone.head),
                    "tail": _vec_to_json(armature_world @ pose_bone.tail),
                    # ``matrix_world`` carries the armature object's scale
                    # (Mixamo imports are uniform 0.01 cm->m); native LBS treats
                    # this 3x3 as an orthonormal rotation, so strip the scale via
                    # the quaternion or every skinned vertex collapses toward the
                    # bone head and the depth-based draw order is garbage.
                    "rotation": _mat_to_json(
                        (armature_world @ pose_bone.matrix).to_quaternion().to_matrix()
                    ),
                }
            )

        sample_records.append(
            {
                "frame": int(frame),
                "subframe": float(subframe),
                "time": float(time),
                "frame_poses": {name: pose.copy() for name, pose in frame_poses.items()},
                "root_motion_offset": (
                    float(root_motion_offset[0]),
                    float(root_motion_offset[1]),
                ),
                "root_rotation_offset": float(root_rotation_offset),
                "frame_filter": frame_filter,
                "bones_3d": bones_3d,
            }
        )
        previous_sample_metrics = {
            "time": float(time),
            "bones": current_sample_metrics,
        }

    duration = max(0.0, (frame_end - frame_start) / fps)
    sample_records, filter_summary = _select_problem_frame_samples(
        sample_records,
        duration,
        problem_frame_filter,
    )

    for sample_record in sample_records:
        time = float(sample_record["time"])
        frame_poses = sample_record["frame_poses"]
        root_motion_offset = sample_record["root_motion_offset"]
        root_rotation_offset = sample_record["root_rotation_offset"]
        for bone_info in bones_setup:
            name = bone_info["name"]
            current = frame_poses[name]
            setup = setup_local[name]

            raw_rel_rotation = current["rotation"] - setup["rotation"]
            if name == root_motion_bone_name:
                raw_rel_rotation += root_rotation_offset
            rel_rotation = _unwrap_angle_near(
                raw_rel_rotation,
                previous_rotation_values.get(name),
            )
            previous_rotation_values[name] = rel_rotation

            allow_translate = _bone_allows_translate(setup, connected_translation, pose_mode)
            if allow_translate:
                rel_x = current["x"] - setup.get("x", 0.0)
                rel_y = current["y"] - setup.get("y", 0.0)
                if name == root_motion_bone_name:
                    rel_x += root_motion_offset[0]
                    rel_y += root_motion_offset[1]
            else:
                rel_x = 0.0
                rel_y = 0.0

            scale = 1.0 if _is_rotation_pose_mode(pose_mode) else current["scale_x"]
            rotation_value = round(rel_rotation, 2)
            bone_timelines[name]["rotate"].append(
                {
                    "time": round(time, 4),
                    "angle": rotation_value,
                    "value": rotation_value,
                }
            )
            if allow_translate:
                bone_timelines[name]["translate"].append(
                    {
                        "time": round(time, 4),
                        "x": round(rel_x, 4),
                        "y": round(rel_y, 4),
                    }
                )
            if setup["length"] > BONE_SCALE_EPSILON and abs(scale - 1.0) > 0.001:
                bone_timelines[name]["scale"].append(
                    {
                        "time": round(time, 4),
                        "x": round(scale, 4),
                        "y": 1.0,
                    }
                )

    for name in bone_timelines:
        timelines = bone_timelines[name]
        if optimize_animation_keys:
            timelines = _optimize_keyframes(timelines)
        if force_loop_closing_keys:
            timelines = _force_loop_closing_keys(timelines, duration)
        bone_timelines[name] = timelines

    preview_frames = [
        {"time": round(float(rec["time"]), 4), "bones": rec.get("bones_3d") or []}
        for rec in sample_records
    ]
    return {
        anim_name: {
            "bones": bone_timelines,
            "frame_filter": filter_summary,
            "preview_3d": {"name": anim_name, "frames": preview_frames},
        }
    }


def _compute_frame_local_bone_poses_2d(
    armature,
    bones_setup,
    view_cfg,
    projected_segments=None,
    projection_inverse=None,
    pose_mode="full",
    rotation_flatten=None,
    local_rotation_reference=None,
    decouple_scale=False,
):
    """Compute current local 2D transforms by inverting the parent 2D basis.

    When ``decouple_scale`` is True (full pose mode only) each bone's WORLD
    basis is forced orthogonal — rotation + scale_x (foreshortening) along the
    bone, scale_y = 1, no shear — and the LOCAL transform is solved to
    (rot, scale_x, scale_y, shear_y) that achieves it under the parent's
    (already orthogonal) world basis. This cancels the shear a non-uniform
    parent would otherwise impose on its children (the "underwater" ripple).
    """
    frame_poses = {}
    world_cache = {}

    for bone_info in bones_setup:
        bone_name = bone_info["name"]
        projected = (projected_segments or {}).get(bone_name)
        if projected is None:
            projected = _sample_projected_bone_segments_2d(
                armature,
                [bone_info],
                view_cfg,
                projection_inverse=projection_inverse,
            )[bone_name]
        head_2d = np.asarray(projected["head"], dtype=np.float64)
        segment = np.asarray(projected["segment"], dtype=np.float64).copy()
        current_length = float(projected["length"])
        setup_length = (
            float(bone_info["length"]) if bone_info["length"] > BONE_SCALE_EPSILON else 1.0
        )
        segment, current_length = _apply_rotation_flatten(
            segment,
            current_length,
            bone_info,
            rotation_flatten,
        )
        rigid_segment = segment.copy()
        rigid_length = current_length
        if rigid_length > BONE_SCALE_EPSILON:
            rigid_segment = rigid_segment / rigid_length
        else:
            setup_rotation = math.radians(float(bone_info.get("rotation", 0.0)))
            rigid_segment = np.array(
                (math.cos(setup_rotation), math.sin(setup_rotation)),
                dtype=np.float64,
            )
        inherit_mode = bone_info.get("inherit", "Normal")
        local_scale_y = 1.0
        local_shear_y = 0.0

        if bone_info["parent"]:
            parent_state = world_cache[bone_info["parent"]]
            if _is_rotation_pose_mode(pose_mode):
                local_x = float(bone_info.get("x", 0.0))
                local_y = float(bone_info.get("y", 0.0))
                if pose_mode == "local_rotation":
                    local_rotation = _compute_local_rotation_pose_angle(
                        bone_info,
                        armature,
                        view_cfg,
                        projection_inverse=projection_inverse,
                        local_rotation_reference=local_rotation_reference,
                        fallback_segment=rigid_segment,
                        fallback_parent_rigid=parent_state["rigid_matrix"],
                    )
                else:
                    rigid_inverse = safe_inverse_2x2(
                        parent_state["rigid_matrix"], BONE_SCALE_EPSILON
                    )
                    local_x_axis = rigid_inverse @ rigid_segment
                    local_rotation = math.degrees(math.atan2(local_x_axis[1], local_x_axis[0]))
                local_scale_x = 1.0
                world_matrix = parent_state["rigid_matrix"] @ _build_2d_basis(local_rotation, 1.0)
                world_head = parent_state["head"] + (
                    parent_state["matrix"] @ np.array((local_x, local_y), dtype=np.float64)
                )
            else:
                inv_parent = safe_inverse_2x2(parent_state["matrix"], BONE_SCALE_EPSILON)
                local_position = inv_parent @ (head_2d - parent_state["head"])
                if setup_length > BONE_SCALE_EPSILON:
                    world_x_axis = segment / setup_length
                else:
                    world_x_axis = np.array((1.0, 0.0), dtype=np.float64)
                local_scale_y = 1.0
                local_shear_y = 0.0
                if decouple_scale:
                    # Declare an orthogonal target world basis (rotation +
                    # foreshortening along the bone, scale_y = 1, no shear) and
                    # solve the local transform under the parent's already-
                    # orthogonal world basis. The resulting shear_y exactly
                    # cancels the parent's non-uniform contribution, so the
                    # chain stays orthogonal and children don't ripple.
                    world_scale = float(np.linalg.norm(world_x_axis))
                    if world_scale <= BONE_SCALE_EPSILON:
                        world_scale = 1.0
                    world_scale = max(0.1, min(10.0, world_scale))
                    world_rotation = math.degrees(
                        math.atan2(world_x_axis[1], world_x_axis[0])
                    )
                    target_world = _build_2d_basis(world_rotation, world_scale, 1.0)
                    rotation_basis = (
                        parent_state["rigid_matrix"]
                        if inherit_mode == "NoScale"
                        else parent_state["matrix"]
                    )
                    local_matrix = (
                        safe_inverse_2x2(rotation_basis, BONE_SCALE_EPSILON) @ target_world
                    )
                    (
                        local_rotation,
                        local_scale_x,
                        local_scale_y,
                        local_shear_y,
                    ) = _decompose_local_basis_2d(local_matrix)
                    world_matrix = target_world
                else:
                    local_basis_inverse = _basis_inverse_for_inherit(parent_state, inherit_mode)
                    local_x_axis = local_basis_inverse @ world_x_axis
                    local_scale_x = float(np.linalg.norm(local_x_axis))
                    if local_scale_x <= BONE_SCALE_EPSILON:
                        local_scale_x = 1.0
                    local_rotation = math.degrees(math.atan2(local_x_axis[1], local_x_axis[0]))
                    world_matrix = _compose_world_matrix(
                        parent_state,
                        local_rotation,
                        local_scale_x,
                        inherit_mode,
                    )
                local_x = float(local_position[0])
                local_y = float(local_position[1])
                world_head = head_2d
        else:
            local_x = float(head_2d[0])
            local_y = float(head_2d[1])
            local_scale_x = (
                1.0
                if _is_rotation_pose_mode(pose_mode)
                else (current_length / setup_length if setup_length > BONE_SCALE_EPSILON else 1.0)
            )
            if pose_mode == "local_rotation":
                local_rotation = _compute_local_rotation_pose_angle(
                    bone_info,
                    armature,
                    view_cfg,
                    projection_inverse=projection_inverse,
                    local_rotation_reference=local_rotation_reference,
                    fallback_segment=rigid_segment,
                    fallback_parent_rigid=None,
                )
            elif current_length > BONE_SCALE_EPSILON:
                rotation_segment = rigid_segment if _is_rotation_pose_mode(pose_mode) else segment
                local_rotation = math.degrees(math.atan2(rotation_segment[1], rotation_segment[0]))
            else:
                local_rotation = float(bone_info["rotation"])
            world_matrix = _build_2d_basis(local_rotation, local_scale_x)
            world_head = np.array((local_x, local_y), dtype=np.float64)

        frame_poses[bone_name] = {
            "x": local_x,
            "y": local_y,
            "rotation": _normalize_angle(local_rotation),
            "length": current_length,
            "scale_x": local_scale_x,
            "scale_y": local_scale_y,
            "shear_y": local_shear_y,
        }
        world_cache[bone_name] = {
            "head": world_head,
            "matrix": world_matrix,
            "rigid_matrix": orthonormalize_2x2(world_matrix, BONE_SCALE_EPSILON),
        }

    return frame_poses


def _sample_projected_bone_segments_2d(
    armature,
    bones_setup,
    view_cfg,
    projection_inverse=None,
):
    projected = {}
    for bone_info in bones_setup:
        bone_name = bone_info["name"]
        pose_bone = armature.pose.bones[bone_name]
        head_world = armature.matrix_world @ pose_bone.head
        tail_world = armature.matrix_world @ pose_bone.tail
        world_segment = np.array(
            (
                float(tail_world.x - head_world.x),
                float(tail_world.y - head_world.y),
                float(tail_world.z - head_world.z),
            ),
            dtype=np.float64,
        )
        world_length = float(np.linalg.norm(world_segment))
        head_2d = np.array(
            project_point_ortho(head_world, view_cfg, projection_inverse=projection_inverse),
            dtype=np.float64,
        )
        tail_2d = np.array(
            project_point_ortho(tail_world, view_cfg, projection_inverse=projection_inverse),
            dtype=np.float64,
        )
        segment = tail_2d - head_2d
        planar_ratio = 1.0
        if world_length > BONE_SCALE_EPSILON:
            direction_screen = project_direction_ortho(
                world_segment / world_length,
                view_cfg,
                projection_inverse=projection_inverse,
            )
            planar_ratio = math.hypot(float(direction_screen[0]), float(direction_screen[1]))
        projected[bone_name] = {
            "head": head_2d,
            "tail": tail_2d,
            "segment": segment,
            "length": float(np.linalg.norm(segment)),
            "world_length": world_length,
            "planar_ratio": _clamp(float(planar_ratio), 0.0, 1.0),
        }
    return projected


def _build_local_rotation_reference(
    armature,
    bones_setup,
    view_cfg,
    projection_inverse=None,
):
    reference = {}
    for bone_info in bones_setup:
        bone_name = bone_info["name"]
        direction_world = _sample_bone_direction_world(armature, bone_name)
        if direction_world is None:
            continue
        direction_projection = transform_direction_to_projection_space(
            direction_world,
            projection_inverse=projection_inverse,
        )
        parent_projection_rotation = _sample_parent_projection_rotation(
            armature,
            bone_info,
            projection_inverse=projection_inverse,
        )
        if parent_projection_rotation is None:
            local_direction = direction_projection
        else:
            local_direction = parent_projection_rotation.T @ direction_projection
        reference[bone_name] = {
            "local_direction": _normalize_vector_3d(local_direction),
        }
    return reference


def _compute_local_rotation_pose_angle(
    bone_info,
    armature,
    view_cfg,
    projection_inverse=None,
    local_rotation_reference=None,
    fallback_segment=None,
    fallback_parent_rigid=None,
):
    fallback_rotation = _compute_fallback_rotation_pose_angle(
        bone_info,
        fallback_segment=fallback_segment,
        fallback_parent_rigid=fallback_parent_rigid,
    )
    reference = (local_rotation_reference or {}).get(bone_info["name"])
    if not reference:
        return fallback_rotation

    direction_world = _sample_bone_direction_world(armature, bone_info["name"])
    if direction_world is None:
        return fallback_rotation
    direction_projection = transform_direction_to_projection_space(
        direction_world,
        projection_inverse=projection_inverse,
    )
    parent_projection_rotation = _sample_parent_projection_rotation(
        armature,
        bone_info,
        projection_inverse=projection_inverse,
    )
    screen_normal = _projection_screen_normal(view_cfg)

    if parent_projection_rotation is None:
        current_local_direction = direction_projection
        local_plane_normal = screen_normal
    else:
        current_local_direction = parent_projection_rotation.T @ direction_projection
        local_plane_normal = parent_projection_rotation.T @ screen_normal

    reference_local_direction = np.asarray(reference["local_direction"], dtype=np.float64)
    reference_planar = _project_vector_to_plane(reference_local_direction, local_plane_normal)
    current_planar = _project_vector_to_plane(current_local_direction, local_plane_normal)
    if (
        float(np.linalg.norm(reference_planar)) <= BONE_SCALE_EPSILON
        or float(np.linalg.norm(current_planar)) <= BONE_SCALE_EPSILON
    ):
        return fallback_rotation

    delta = _signed_angle_around_normal_3d(reference_planar, current_planar, local_plane_normal)
    return _normalize_angle(float(bone_info.get("rotation", 0.0)) + delta)


def _sample_bone_direction_world(armature, bone_name):
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        return None
    head_world = armature.matrix_world @ pose_bone.head
    tail_world = armature.matrix_world @ pose_bone.tail
    direction = np.array(
        (
            float(tail_world.x - head_world.x),
            float(tail_world.y - head_world.y),
            float(tail_world.z - head_world.z),
        ),
        dtype=np.float64,
    )
    length = float(np.linalg.norm(direction))
    if length <= BONE_SCALE_EPSILON:
        return None
    return direction / length


def _sample_parent_projection_rotation(armature, bone_info, projection_inverse=None):
    parent_name = bone_info.get("parent")
    if not parent_name:
        return None
    parent_pose_bone = armature.pose.bones.get(parent_name)
    if parent_pose_bone is None:
        return None
    parent_world_matrix = armature.matrix_world @ parent_pose_bone.matrix
    parent_rotation = orthonormalize_3x3(np.asarray(parent_world_matrix.to_3x3(), dtype=np.float64))
    if projection_inverse is not None:
        projection_rotation = np.asarray(projection_inverse, dtype=np.float64)[:3, :3]
        parent_rotation = projection_rotation @ parent_rotation
    return orthonormalize_3x3(parent_rotation)


def _compute_root_motion_offset_2d(
    armature,
    view_cfg,
    root_bone_name,
    root_motion_reference_2d,
    enabled=False,
):
    """Project current root head motion in the exported 2D view space."""
    if not enabled or root_bone_name is None or root_motion_reference_2d is None:
        return 0.0, 0.0
    pose_bone = armature.pose.bones.get(root_bone_name)
    if pose_bone is None:
        return 0.0, 0.0
    root_head_world = armature.matrix_world @ pose_bone.head
    current = project_point_ortho(root_head_world, view_cfg)
    return (
        float(current[0] - root_motion_reference_2d[0]),
        float(current[1] - root_motion_reference_2d[1]),
    )


def _compute_root_motion_reference_2d(
    armature, view_cfg, frame_start, root_bone_name, enabled=False
):
    """Capture the root head reference position in the final 2D projection space."""
    if not enabled or root_bone_name is None:
        return None

    scene = bpy.context.scene
    current_frame = scene.frame_current
    current_subframe = getattr(scene, "frame_subframe", 0.0)
    try:
        scene.frame_set(frame_start)
        bpy.context.view_layer.update()
        pose_bone = armature.pose.bones.get(root_bone_name)
        if pose_bone is None:
            return None
        root_head_world = armature.matrix_world @ pose_bone.head
        projected = project_point_ortho(root_head_world, view_cfg)
        return float(projected[0]), float(projected[1])
    finally:
        scene.frame_set(current_frame, subframe=current_subframe)
        bpy.context.view_layer.update()


def _compute_root_rotation_offset_2d(
    armature,
    view_cfg,
    root_bone_name,
    root_rotation_reference_deg,
    enabled=False,
):
    """Project current root segment angle into the final 2D view space."""
    if not enabled or root_bone_name is None or root_rotation_reference_deg is None:
        return 0.0
    pose_bone = armature.pose.bones.get(root_bone_name)
    if pose_bone is None:
        return 0.0
    head_world = armature.matrix_world @ pose_bone.head
    tail_world = armature.matrix_world @ pose_bone.tail
    head_2d = project_point_ortho(head_world, view_cfg)
    tail_2d = project_point_ortho(tail_world, view_cfg)
    dx = float(tail_2d[0]) - float(head_2d[0])
    dy = float(tail_2d[1]) - float(head_2d[1])
    if abs(dx) <= BONE_SCALE_EPSILON and abs(dy) <= BONE_SCALE_EPSILON:
        return 0.0
    current_rotation = math.degrees(math.atan2(dy, dx))
    return _normalize_angle(current_rotation - root_rotation_reference_deg)


def _compute_root_rotation_reference_2d(
    armature,
    view_cfg,
    frame_start,
    root_bone_name,
    enabled=False,
):
    """Capture the reference projected 2D angle of the root segment."""
    if not enabled or root_bone_name is None:
        return None

    scene = bpy.context.scene
    current_frame = scene.frame_current
    current_subframe = getattr(scene, "frame_subframe", 0.0)
    try:
        scene.frame_set(frame_start)
        bpy.context.view_layer.update()
        pose_bone = armature.pose.bones.get(root_bone_name)
        if pose_bone is None:
            return None
        head_world = armature.matrix_world @ pose_bone.head
        tail_world = armature.matrix_world @ pose_bone.tail
        head_2d = project_point_ortho(head_world, view_cfg)
        tail_2d = project_point_ortho(tail_world, view_cfg)
        dx = float(tail_2d[0]) - float(head_2d[0])
        dy = float(tail_2d[1]) - float(head_2d[1])
        if abs(dx) <= BONE_SCALE_EPSILON and abs(dy) <= BONE_SCALE_EPSILON:
            return 0.0
        return math.degrees(math.atan2(dy, dx))
    finally:
        scene.frame_set(current_frame, subframe=current_subframe)
        bpy.context.view_layer.update()
