"""
Animation timeline extraction for Spine 2D.

Extracts bone animation keyframes from the 3D armature and converts them
to Spine's relative-to-setup-pose format.
"""

import copy
import math
import re

import bpy
import numpy as np

from flatrig._sidecar_import import orthonormalize_2x2, orthonormalize_3x3, safe_inverse_2x2
from flatrig.projection import (
    get_projection_reference_inverse,
    project_direction_ortho,
    project_point_ortho,
    transform_direction_to_projection_space,
)

BONE_SCALE_EPSILON = 1e-3
SHORT_BONE_LENGTH = 0.08
COLLAPSED_SCALE_THRESHOLD = 0.45
SOFT_SCALE_MIN = 0.2
SOFT_SCALE_MAX = 2.5
ROTATION_FLATTEN_MIN_RATIO = 0.35
ROTATION_FLATTEN_FULL_RATIO = 0.9


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


def sample_exported_bone_world_matrices_2d(bones_setup, animation_payload, sample_times):
    """Evaluate exported local timelines into per-frame 2D world transforms."""
    return _sample_exported_bone_context_2d(bones_setup, animation_payload, sample_times)[
        "world_matrices"
    ]


def refine_exported_scale_y_timelines(
    bones_setup,
    animation_payload,
    sample_times,
    target_positions_2d,
    bind_positions_2d,
    weights,
    *,
    visibility_mask=None,
    setup_bone_world_matrices_2d,
    passes=2,
    regularization=0.2,
    temporal_smoothness=0.35,
    min_scale_y=0.35,
    max_scale_y=2.85,
    weight_threshold=1e-5,
):
    """Refine exported local scaleY timelines against the final 2D target."""
    resolved_sample_times = [float(value) for value in (sample_times or [])]
    if not bones_setup or not resolved_sample_times or not weights:
        return {
            "applied": False,
            "reason": "missing_inputs",
        }

    target_positions_2d = np.asarray(target_positions_2d, dtype=np.float64)
    bind_positions_2d = np.asarray(bind_positions_2d, dtype=np.float64)
    setup_bone_world_matrices_2d = np.asarray(setup_bone_world_matrices_2d, dtype=np.float64)
    if target_positions_2d.ndim != 3 or target_positions_2d.shape[2] != 2:
        raise ValueError("target_positions_2d must be shaped [frames, verts, 2].")
    if bind_positions_2d.ndim != 2 or bind_positions_2d.shape[1] != 2:
        raise ValueError("bind_positions_2d must be shaped [verts, 2].")
    if target_positions_2d.shape[0] != len(resolved_sample_times):
        raise ValueError("sample_times must match the target frame count.")
    if target_positions_2d.shape[1] != bind_positions_2d.shape[0]:
        raise ValueError("bind_positions_2d must match the target vertex count.")

    if visibility_mask is None:
        valid_mask = np.ones(target_positions_2d.shape[:2], dtype=bool)
    else:
        valid_mask = np.asarray(visibility_mask, dtype=bool)
        if valid_mask.shape != target_positions_2d.shape[:2]:
            raise ValueError("visibility_mask must match the target [frames, verts] shape.")

    influence_cache, influences_by_bone = _build_export_scale_y_influence_cache(
        bind_positions_2d,
        weights,
        setup_bone_world_matrices_2d,
        weight_threshold=weight_threshold,
    )
    if not any(influences_by_bone):
        return {
            "applied": False,
            "reason": "no_influences",
        }

    before_context = _sample_exported_bone_context_2d(
        bones_setup,
        animation_payload,
        resolved_sample_times,
    )
    before_prediction = _compute_weighted_export_positions(
        before_context["world_matrices"],
        influence_cache,
        bind_positions_2d,
    )
    before_summary = _summarize_export_fit_error(
        before_prediction,
        target_positions_2d,
        valid_mask,
    )

    changed_bones = set()
    applied_passes = 0
    max_outer_passes = max(1, int(passes))
    for _ in range(max_outer_passes):
        applied_passes += 1
        context = _sample_exported_bone_context_2d(
            bones_setup,
            animation_payload,
            resolved_sample_times,
        )
        predicted_positions = _compute_weighted_export_positions(
            context["world_matrices"],
            influence_cache,
            bind_positions_2d,
        )
        pass_before_summary = _summarize_export_fit_error(
            predicted_positions,
            target_positions_2d,
            valid_mask,
        )
        scale_updates = {}
        scale_priors = {}

        for bone_index, vertex_influences in enumerate(influences_by_bone):
            if not vertex_influences:
                continue

            current_scale_y = context["local_scale_y"][:, bone_index].copy()
            diag = np.zeros(len(resolved_sample_times), dtype=np.float64)
            rhs = np.zeros(len(resolved_sample_times), dtype=np.float64)

            world_heads = context["world_heads"][:, bone_index]
            world_x_dirs = context["world_x_dirs"][:, bone_index]
            world_y_dirs = context["world_y_dirs"][:, bone_index]
            local_scale_x_values = context["local_scale_x"][:, bone_index]

            for vertex_index, weight_value, local_x, local_y in vertex_influences:
                if abs(local_y) <= 1e-8:
                    continue
                visible_frames = np.where(valid_mask[:, vertex_index])[0]
                if visible_frames.size == 0:
                    continue

                head_values = world_heads[visible_frames]
                x_dirs = world_x_dirs[visible_frames]
                y_dirs = world_y_dirs[visible_frames]
                scale_x = local_scale_x_values[visible_frames]
                current_sy = current_scale_y[visible_frames]

                constant_term = float(weight_value) * (
                    head_values + x_dirs * (scale_x[:, np.newaxis] * float(local_x))
                )
                scale_term = float(weight_value) * y_dirs * float(local_y)
                current_contrib = constant_term + scale_term * current_sy[:, np.newaxis]
                other_contrib = predicted_positions[visible_frames, vertex_index] - current_contrib
                target_term = target_positions_2d[visible_frames, vertex_index] - other_contrib
                fitted_rhs = target_term - constant_term

                diag[visible_frames] += np.sum(scale_term * scale_term, axis=1)
                rhs[visible_frames] += np.sum(scale_term * fitted_rhs, axis=1)

            if float(np.max(diag)) <= 1e-8:
                continue

            solved_scale_y = _solve_temporal_scale_series(
                diag,
                rhs,
                current_scale_y,
                regularization=float(regularization),
                temporal_smoothness=float(temporal_smoothness),
                min_value=float(min_scale_y),
                max_value=float(max_scale_y),
            )
            if np.allclose(solved_scale_y, current_scale_y, atol=1e-4):
                continue

            delta_scale_y = solved_scale_y - current_scale_y
            scale_updates[bone_index] = solved_scale_y
            scale_priors[bone_index] = current_scale_y

            for vertex_index, weight_value, _local_x, local_y in vertex_influences:
                if abs(local_y) <= 1e-8:
                    continue
                visible_frames = np.where(valid_mask[:, vertex_index])[0]
                if visible_frames.size == 0:
                    continue
                scale_term = float(weight_value) * world_y_dirs[visible_frames] * float(local_y)
                predicted_positions[visible_frames, vertex_index] += (
                    scale_term * delta_scale_y[visible_frames, np.newaxis]
                )

        if not scale_updates:
            break

        animation_snapshot = copy.deepcopy((animation_payload or {}).get("bones") or {})
        accepted = False
        for blend_factor in (1.0, 0.5, 0.25):
            blended_updates = {
                bone_index: scale_priors[bone_index]
                + (solved_scale_y - scale_priors[bone_index]) * float(blend_factor)
                for bone_index, solved_scale_y in scale_updates.items()
            }
            animation_payload["bones"] = copy.deepcopy(animation_snapshot)
            _apply_scale_y_updates_to_animation(
                bones_setup,
                animation_payload,
                resolved_sample_times,
                context["local_scale_x"],
                blended_updates,
            )
            candidate_context = _sample_exported_bone_context_2d(
                bones_setup,
                animation_payload,
                resolved_sample_times,
            )
            candidate_prediction = _compute_weighted_export_positions(
                candidate_context["world_matrices"],
                influence_cache,
                bind_positions_2d,
            )
            candidate_summary = _summarize_export_fit_error(
                candidate_prediction,
                target_positions_2d,
                valid_mask,
            )
            if candidate_summary["mean_error"] < pass_before_summary["mean_error"] - 1e-6:
                changed_bones.update(int(bone_index) for bone_index in blended_updates.keys())
                accepted = True
                break

        if not accepted:
            animation_payload["bones"] = animation_snapshot
            break

    after_context = _sample_exported_bone_context_2d(
        bones_setup,
        animation_payload,
        resolved_sample_times,
    )
    after_prediction = _compute_weighted_export_positions(
        after_context["world_matrices"],
        influence_cache,
        bind_positions_2d,
    )
    after_summary = _summarize_export_fit_error(
        after_prediction,
        target_positions_2d,
        valid_mask,
    )
    return {
        "applied": bool(changed_bones),
        "reason": "refined" if changed_bones else "no_change",
        "passes": int(applied_passes),
        "changed_bone_count": int(len(changed_bones)),
        "changed_bones": [bones_setup[index]["name"] for index in sorted(changed_bones)],
        "mean_error_before": round(float(before_summary["mean_error"]), 4),
        "mean_error_after": round(float(after_summary["mean_error"]), 4),
        "p95_error_before": round(float(before_summary["p95_error"]), 4),
        "p95_error_after": round(float(after_summary["p95_error"]), 4),
        "max_error_before": round(float(before_summary["max_error"]), 4),
        "max_error_after": round(float(after_summary["max_error"]), 4),
        "visible_vertex_samples": int(after_summary["visible_vertex_samples"]),
    }


def _sample_exported_bone_context_2d(bones_setup, animation_payload, sample_times):
    """Evaluate exported local timelines into per-frame 2D world transforms."""
    bone_timelines = dict((animation_payload or {}).get("bones") or {})
    sample_count = len(sample_times)
    bone_count = len(bones_setup)
    world_matrices = np.zeros((sample_count, bone_count, 3, 3), dtype=np.float64)
    world_heads = np.zeros((sample_count, bone_count, 2), dtype=np.float64)
    world_x_dirs = np.zeros((sample_count, bone_count, 2), dtype=np.float64)
    world_y_dirs = np.zeros((sample_count, bone_count, 2), dtype=np.float64)
    local_scale_x_values = np.ones((sample_count, bone_count), dtype=np.float64)
    local_scale_y_values = np.ones((sample_count, bone_count), dtype=np.float64)

    for sample_index, sample_time in enumerate(sample_times):
        world_cache = {}
        for bone_info in bones_setup:
            bone_name = bone_info["name"]
            local_pose = _evaluate_exported_local_pose(
                bone_info,
                bone_timelines.get(bone_name) or {},
                float(sample_time),
            )
            local_x = float(local_pose["x"])
            local_y = float(local_pose["y"])
            local_rotation = float(local_pose["rotation"])
            local_scale_x = float(local_pose["scale_x"])
            local_scale_y = float(local_pose["scale_y"])
            local_shear_x = float(local_pose["shear_x"])
            local_shear_y = float(local_pose["shear_y"])
            parent_name = bone_info.get("parent")
            inherit_mode = bone_info.get("inherit", "Normal")
            x_axis_angle = math.radians(local_rotation + local_shear_x)
            y_axis_angle = math.radians(local_rotation + 90.0 + local_shear_y)

            local_basis = _build_2d_basis_with_shear(
                local_rotation,
                local_scale_x,
                local_scale_y,
                local_shear_x,
                local_shear_y,
            )
            if parent_name:
                parent_state = world_cache[parent_name]
                world_head = parent_state["head"] + (
                    parent_state["matrix"] @ np.array((local_x, local_y), dtype=np.float64)
                )
                parent_basis = parent_state["matrix"]
                if inherit_mode == "NoScale":
                    parent_basis = parent_state["rigid_matrix"]
                world_matrix = parent_basis @ local_basis
            else:
                world_head = np.array((local_x, local_y), dtype=np.float64)
                parent_basis = np.eye(2, dtype=np.float64)
                world_matrix = local_basis

            transform = np.eye(3, dtype=np.float64)
            transform[:2, :2] = world_matrix
            transform[:2, 2] = world_head
            bone_index = int(bone_info["index"])
            world_matrices[sample_index, bone_index] = transform
            world_heads[sample_index, bone_index] = world_head
            world_x_dirs[sample_index, bone_index] = parent_basis @ np.array(
                (math.cos(x_axis_angle), math.sin(x_axis_angle)),
                dtype=np.float64,
            )
            world_y_dirs[sample_index, bone_index] = parent_basis @ np.array(
                (math.cos(y_axis_angle), math.sin(y_axis_angle)),
                dtype=np.float64,
            )
            local_scale_x_values[sample_index, bone_index] = local_scale_x
            local_scale_y_values[sample_index, bone_index] = local_scale_y
            world_cache[bone_name] = {
                "head": world_head,
                "matrix": world_matrix,
                "rigid_matrix": orthonormalize_2x2(world_matrix, BONE_SCALE_EPSILON),
            }

    return {
        "world_matrices": world_matrices,
        "world_heads": world_heads,
        "world_x_dirs": world_x_dirs,
        "world_y_dirs": world_y_dirs,
        "local_scale_x": local_scale_x_values,
        "local_scale_y": local_scale_y_values,
    }


def _build_export_scale_y_influence_cache(
    bind_positions_2d,
    weights,
    setup_bone_world_matrices_2d,
    *,
    weight_threshold,
):
    bind_positions_h = np.concatenate(
        (bind_positions_2d, np.ones((bind_positions_2d.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    setup_inverse = np.linalg.inv(np.asarray(setup_bone_world_matrices_2d, dtype=np.float64))
    influence_cache = []
    influences_by_bone = [[] for _ in range(setup_inverse.shape[0])]

    for vertex_index, weight_dict in enumerate(weights):
        vertex_influences = []
        for bone_index, weight_value in sorted((weight_dict or {}).items()):
            resolved_weight = float(weight_value)
            if resolved_weight <= float(weight_threshold):
                continue
            local_bind = setup_inverse[int(bone_index)] @ bind_positions_h[vertex_index]
            local_x = float(local_bind[0])
            local_y = float(local_bind[1])
            vertex_influences.append((int(bone_index), resolved_weight, local_x, local_y))
            influences_by_bone[int(bone_index)].append(
                (int(vertex_index), resolved_weight, local_x, local_y)
            )
        influence_cache.append(vertex_influences)

    return influence_cache, influences_by_bone


def _compute_weighted_export_positions(world_matrices, influence_cache, bind_positions_2d):
    predicted_positions = np.repeat(
        np.asarray(bind_positions_2d, dtype=np.float64)[np.newaxis, :, :],
        world_matrices.shape[0],
        axis=0,
    )
    for vertex_index, influences in enumerate(influence_cache):
        if not influences:
            continue
        predicted_vertex = np.zeros((world_matrices.shape[0], 2), dtype=np.float64)
        for bone_index, weight_value, local_x, local_y in influences:
            local_bind = np.array((local_x, local_y, 1.0), dtype=np.float64)
            transformed = np.einsum(
                "fij,j->fi",
                world_matrices[:, int(bone_index)],
                local_bind,
            )
            predicted_vertex += float(weight_value) * transformed[:, :2]
        predicted_positions[:, vertex_index] = predicted_vertex
    return predicted_positions


def _solve_temporal_scale_series(
    diag,
    rhs,
    prior,
    *,
    regularization,
    temporal_smoothness,
    min_value,
    max_value,
):
    count = int(len(diag))
    if count <= 0:
        return np.asarray(prior, dtype=np.float64)

    matrix = np.zeros((count, count), dtype=np.float64)
    vector = np.asarray(rhs, dtype=np.float64) + float(regularization) * np.asarray(
        prior,
        dtype=np.float64,
    )
    diagonal = np.asarray(diag, dtype=np.float64) + float(regularization)

    if count > 1 and float(temporal_smoothness) > 0.0:
        smoothness = float(temporal_smoothness)
        diagonal[0] += smoothness
        diagonal[-1] += smoothness
        if count > 2:
            diagonal[1:-1] += smoothness * 2.0
        for index in range(count - 1):
            matrix[index, index + 1] = -smoothness
            matrix[index + 1, index] = -smoothness

    np.fill_diagonal(matrix, diagonal + 1e-8)
    solved = np.linalg.solve(matrix, vector)
    return np.clip(solved, float(min_value), float(max_value))


def _apply_scale_y_updates_to_animation(
    bones_setup,
    animation_payload,
    sample_times,
    local_scale_x,
    scale_updates,
):
    animation_bones = animation_payload.setdefault("bones", {})
    for bone_index, solved_scale_y in scale_updates.items():
        bone_info = bones_setup[int(bone_index)]
        bone_name = bone_info["name"]
        timelines = animation_bones.setdefault(
            bone_name,
            {"rotate": [], "translate": [], "scale": []},
        )
        base_scale_x = max(abs(float(bone_info.get("scale_x", 1.0))), 1e-8)
        base_scale_y = max(abs(float(bone_info.get("scale_y", 1.0))), 1e-8)
        scale_keyframes = [
            {
                "time": round(float(time_value), 4),
                "x": round(float(local_scale_x[sample_index, bone_index]) / base_scale_x, 4),
                "y": round(float(solved_scale_y[sample_index]) / base_scale_y, 4),
            }
            for sample_index, time_value in enumerate(sample_times)
        ]
        if all(
            abs(float(keyframe.get("x", 1.0)) - 1.0) <= 0.001
            and abs(float(keyframe.get("y", 1.0)) - 1.0) <= 0.001
            for keyframe in scale_keyframes
        ):
            timelines.pop("scale", None)
            continue
        timelines["scale"] = _optimize_timeline_2d(
            scale_keyframes,
            tolerance=0.001,
            value_keys=("x", "y"),
        )


def _summarize_export_fit_error(predicted_positions, target_positions_2d, valid_mask):
    errors = np.linalg.norm(
        np.asarray(predicted_positions, dtype=np.float64)
        - np.asarray(target_positions_2d, dtype=np.float64),
        axis=2,
    )
    mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(mask):
        return {
            "mean_error": 0.0,
            "p95_error": 0.0,
            "max_error": 0.0,
            "visible_vertex_samples": 0,
        }
    values = errors[mask]
    return {
        "mean_error": float(np.mean(values)),
        "p95_error": float(np.percentile(values, 95)),
        "max_error": float(np.max(values)),
        "visible_vertex_samples": int(mask.sum()),
    }


def _is_pose_action(action):
    for fcurve in action.fcurves:
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

    return {anim_name: {"bones": bone_timelines, "frame_filter": filter_summary}}


def _compute_frame_local_bone_poses_2d(
    armature,
    bones_setup,
    view_cfg,
    projected_segments=None,
    projection_inverse=None,
    pose_mode="full",
    rotation_flatten=None,
    local_rotation_reference=None,
):
    """Compute current local 2D transforms by inverting the parent 2D basis."""
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


def _compute_fallback_rotation_pose_angle(
    bone_info,
    fallback_segment=None,
    fallback_parent_rigid=None,
):
    rigid_segment = np.asarray(
        fallback_segment if fallback_segment is not None else (1.0, 0.0), dtype=np.float64
    )
    if bone_info.get("parent"):
        rigid_inverse = safe_inverse_2x2(
            np.asarray(
                fallback_parent_rigid
                if fallback_parent_rigid is not None
                else np.eye(2, dtype=np.float64),
                dtype=np.float64,
            ),
            BONE_SCALE_EPSILON,
        )
        local_x_axis = rigid_inverse @ rigid_segment
        return math.degrees(math.atan2(local_x_axis[1], local_x_axis[0]))
    return math.degrees(math.atan2(rigid_segment[1], rigid_segment[0]))


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


def _projection_screen_normal(view_cfg):
    return np.asarray(view_cfg["depth_axis"], dtype=np.float64)


def _project_vector_to_plane(vector, plane_normal):
    vector = np.asarray(vector, dtype=np.float64)
    plane_normal = _normalize_vector_3d(plane_normal)
    normal_length = float(np.linalg.norm(plane_normal))
    if normal_length <= BONE_SCALE_EPSILON:
        return vector.copy()
    return vector - plane_normal * float(np.dot(vector, plane_normal))


def _normalize_vector_3d(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= BONE_SCALE_EPSILON:
        return vector.copy()
    return vector / length


def _signed_angle_around_normal_3d(reference_vector, current_vector, plane_normal):
    reference_unit = _normalize_vector_3d(reference_vector)
    current_unit = _normalize_vector_3d(current_vector)
    normal_unit = _normalize_vector_3d(plane_normal)
    if (
        float(np.linalg.norm(reference_unit)) <= BONE_SCALE_EPSILON
        or float(np.linalg.norm(current_unit)) <= BONE_SCALE_EPSILON
        or float(np.linalg.norm(normal_unit)) <= BONE_SCALE_EPSILON
    ):
        return 0.0
    cross_value = np.cross(reference_unit, current_unit)
    signed_cross = float(np.dot(cross_value, normal_unit))
    dot_value = _clamp(float(np.dot(reference_unit, current_unit)), -1.0, 1.0)
    return math.degrees(math.atan2(signed_cross, dot_value))


def _prepare_problem_frame_filter(problem_frame_filter):
    config = {
        "enabled": False,
        "min_flatness": 0.82,
        "collapsed_planar_ratio": 0.65,
        "max_collapsed_ratio": 0.35,
        "min_scale": 0.4,
        "max_scale": 1.85,
        "max_extreme_scale_ratio": 0.2,
        "max_rotation_step": 70.0,
        "max_scale_step": 0.45,
    }
    if not problem_frame_filter:
        return config
    config["enabled"] = bool(problem_frame_filter.get("enabled", False))
    return config


def _collect_problem_frame_metrics(bones_setup, projected_segments, frame_poses):
    metrics = {}
    for bone_info in bones_setup:
        bone_name = bone_info["name"]
        projected = (projected_segments or {}).get(bone_name) or {}
        current_pose = (frame_poses or {}).get(bone_name) or {}
        setup_length = max(float(bone_info.get("length", 0.0)), BONE_SCALE_EPSILON)
        weight = max(0.25, float(bone_info.get("length_ratio", 1.0)))
        if bone_info.get("terminal_chain"):
            weight *= 0.5
        metrics[bone_name] = {
            "weight": weight,
            "planar_ratio": _clamp(float(projected.get("planar_ratio", 1.0)), 0.0, 1.0),
            "scale": float(projected.get("length", 0.0)) / setup_length,
            "rotation": float(current_pose.get("rotation", bone_info.get("rotation", 0.0))),
        }
    return metrics


def _evaluate_problem_frame_sample(
    current_metrics,
    previous_sample_metrics,
    filter_config,
    *,
    time_value,
    fps,
):
    total_weight = 0.0
    weighted_flatness = 0.0
    collapsed_weight = 0.0
    extreme_scale_weight = 0.0
    weighted_rotation_step = 0.0
    weighted_scale_step = 0.0

    delta_frames = 0.0
    if previous_sample_metrics is not None:
        previous_time = float(previous_sample_metrics.get("time", time_value))
        delta_frames = max(
            (float(time_value) - previous_time) * float(fps),
            1e-6,
        )

    for bone_name, bone_metrics in (current_metrics or {}).items():
        weight = max(float(bone_metrics.get("weight", 0.0)), 0.0)
        total_weight += weight

        planar_ratio = _clamp(float(bone_metrics.get("planar_ratio", 1.0)), 0.0, 1.0)
        scale_value = float(bone_metrics.get("scale", 1.0))
        weighted_flatness += planar_ratio * weight
        if planar_ratio < float(filter_config["collapsed_planar_ratio"]):
            collapsed_weight += weight
        if scale_value < float(filter_config["min_scale"]) or scale_value > float(
            filter_config["max_scale"]
        ):
            extreme_scale_weight += weight

        if previous_sample_metrics is None:
            continue
        previous_bone_metrics = (previous_sample_metrics.get("bones") or {}).get(bone_name)
        if not previous_bone_metrics:
            continue
        rotation_delta = abs(
            _normalize_angle(
                float(bone_metrics.get("rotation", 0.0))
                - float(previous_bone_metrics.get("rotation", 0.0))
            )
        )
        scale_delta = abs(
            float(bone_metrics.get("scale", 1.0)) - float(previous_bone_metrics.get("scale", 1.0))
        )
        weighted_rotation_step += (rotation_delta / delta_frames) * weight
        weighted_scale_step += (scale_delta / delta_frames) * weight

    total_weight = max(total_weight, 1e-6)
    flatness = weighted_flatness / total_weight
    collapsed_ratio = collapsed_weight / total_weight
    extreme_scale_ratio = extreme_scale_weight / total_weight
    if previous_sample_metrics is not None:
        rotation_step = weighted_rotation_step / total_weight
        scale_step = weighted_scale_step / total_weight
    else:
        rotation_step = 0.0
        scale_step = 0.0

    issues = []
    if flatness < float(filter_config["min_flatness"]):
        issues.append("flatness")
    if collapsed_ratio > float(filter_config["max_collapsed_ratio"]):
        issues.append("collapse")
    if extreme_scale_ratio > float(filter_config["max_extreme_scale_ratio"]):
        issues.append("scale")
    if previous_sample_metrics is not None and rotation_step > float(
        filter_config["max_rotation_step"]
    ):
        issues.append("rotation_step")
    if previous_sample_metrics is not None and scale_step > float(filter_config["max_scale_step"]):
        issues.append("scale_step")

    score = flatness
    score -= 0.75 * collapsed_ratio
    score -= 0.5 * extreme_scale_ratio
    if previous_sample_metrics is not None:
        score -= 0.15 * min(
            rotation_step / max(float(filter_config["max_rotation_step"]), 1e-6),
            2.0,
        )
        score -= 0.15 * min(
            scale_step / max(float(filter_config["max_scale_step"]), 1e-6),
            2.0,
        )

    return {
        "keep": not issues,
        "issues": list(issues),
        "flatness": round(float(flatness), 4),
        "collapsed_ratio": round(float(collapsed_ratio), 4),
        "extreme_scale_ratio": round(float(extreme_scale_ratio), 4),
        "rotation_step": round(float(rotation_step), 4),
        "scale_step": round(float(scale_step), 4),
        "delta_frames": round(float(delta_frames), 4),
        "score": round(float(score), 4),
    }


def _select_problem_frame_samples(sample_records, duration, filter_config):
    summary = {
        "enabled": bool(filter_config.get("enabled", False)),
        "sampled_count": len(sample_records or []),
        "kept_count": len(sample_records or []),
        "dropped_count": 0,
        "kept_ratio": 1.0 if sample_records else 0.0,
        "dropped_ratio": 0.0,
        "synthetic_hold_count": 0,
        "forced_keep": False,
        "issues": {},
        "thresholds": {
            "min_flatness": round(float(filter_config.get("min_flatness", 0.0)), 4),
            "collapsed_planar_ratio": round(
                float(filter_config.get("collapsed_planar_ratio", 0.0)),
                4,
            ),
            "max_collapsed_ratio": round(float(filter_config.get("max_collapsed_ratio", 0.0)), 4),
            "min_scale": round(float(filter_config.get("min_scale", 0.0)), 4),
            "max_scale": round(float(filter_config.get("max_scale", 0.0)), 4),
            "max_extreme_scale_ratio": round(
                float(filter_config.get("max_extreme_scale_ratio", 0.0)),
                4,
            ),
            "max_rotation_step": round(float(filter_config.get("max_rotation_step", 0.0)), 4),
            "max_scale_step": round(float(filter_config.get("max_scale_step", 0.0)), 4),
        },
    }
    if not sample_records:
        return [], summary
    if not filter_config.get("enabled", False):
        return [_clone_sample_record(record) for record in sample_records], summary

    kept_sample_records = [
        record for record in sample_records if (record.get("frame_filter") or {}).get("keep", True)
    ]
    dropped_sample_records = [
        record
        for record in sample_records
        if not (record.get("frame_filter") or {}).get("keep", True)
    ]

    if not kept_sample_records:
        fallback_record = max(
            sample_records,
            key=lambda record: float((record.get("frame_filter") or {}).get("score", -1e9)),
        )
        kept_sample_records = [fallback_record]
        dropped_sample_records = [
            record for record in sample_records if record is not fallback_record
        ]
        summary["forced_keep"] = True

    issues = {}
    for record in dropped_sample_records:
        for issue in set((record.get("frame_filter") or {}).get("issues") or []):
            issues[issue] = issues.get(issue, 0) + 1

    padded_records = []
    first_record = kept_sample_records[0]
    last_record = kept_sample_records[-1]
    if float(first_record.get("time", 0.0)) > 1e-4:
        padded_records.append(
            _clone_sample_record(first_record, time_value=0.0, synthetic_hold=True)
        )
        summary["synthetic_hold_count"] += 1
    padded_records.extend(_clone_sample_record(record) for record in kept_sample_records)
    if duration - float(last_record.get("time", 0.0)) > 1e-4:
        padded_records.append(
            _clone_sample_record(last_record, time_value=duration, synthetic_hold=True)
        )
        summary["synthetic_hold_count"] += 1

    summary["kept_count"] = len(kept_sample_records)
    summary["dropped_count"] = len(dropped_sample_records)
    summary["kept_ratio"] = round(
        len(kept_sample_records) / max(1, len(sample_records)),
        4,
    )
    summary["dropped_ratio"] = round(
        len(dropped_sample_records) / max(1, len(sample_records)),
        4,
    )
    summary["issues"] = issues
    return padded_records, summary


def _clone_sample_record(record, time_value=None, synthetic_hold=False):
    frame_filter = dict(record.get("frame_filter") or {})
    if "issues" in frame_filter:
        frame_filter["issues"] = list(frame_filter["issues"])
    if synthetic_hold:
        frame_filter["synthetic_hold"] = True
    return {
        "frame": int(record.get("frame", 0)),
        "subframe": float(record.get("subframe", 0.0)),
        "time": float(record.get("time", 0.0) if time_value is None else time_value),
        "frame_poses": {
            bone_name: dict(pose or {})
            for bone_name, pose in (record.get("frame_poses") or {}).items()
        },
        "root_motion_offset": (
            float((record.get("root_motion_offset") or (0.0, 0.0))[0]),
            float((record.get("root_motion_offset") or (0.0, 0.0))[1]),
        ),
        "root_rotation_offset": float(record.get("root_rotation_offset", 0.0)),
        "frame_filter": frame_filter,
    }


def _prepare_rotation_flatten(rotation_flatten):
    config = {
        "amount": 0.0,
        "scope": "all",
        "bones": "",
        "matcher": None,
    }
    if not rotation_flatten:
        return config

    amount = _clamp(float(rotation_flatten.get("amount", 0.0)), 0.0, 1.0)
    scope = rotation_flatten.get("scope", "all")
    bones = str(rotation_flatten.get("bones", "") or "").strip()
    config["amount"] = amount
    config["scope"] = scope
    config["bones"] = bones
    if scope == "custom":
        config["matcher"] = _compile_bone_matcher(bones)
    return config


def _prepare_connected_translation(connected_translation):
    config = {
        "scope": "none",
        "bones": "",
        "matcher": None,
    }
    if not connected_translation:
        return config

    scope = str(connected_translation.get("scope", "none") or "none")
    bones = str(connected_translation.get("bones", "") or "").strip()
    config["scope"] = scope
    config["bones"] = bones
    if scope == "custom":
        config["matcher"] = _compile_bone_matcher(bones)
    return config


def _prepare_leaf_ik_refine(leaf_ik_refine):
    config = {
        "enabled": False,
        "strength": 0.35,
        "iterations": 6,
        "max_chain_length": 3,
        "preserve_scale": 0.65,
    }
    if not leaf_ik_refine:
        return config

    config["enabled"] = bool(leaf_ik_refine.get("enabled", False))
    config["strength"] = _clamp(float(leaf_ik_refine.get("strength", config["strength"])), 0.0, 1.0)
    config["iterations"] = max(
        1, min(12, int(leaf_ik_refine.get("iterations", config["iterations"])))
    )
    config["max_chain_length"] = max(
        2,
        min(4, int(leaf_ik_refine.get("max_chain_length", config["max_chain_length"]))),
    )
    config["preserve_scale"] = _clamp(
        float(leaf_ik_refine.get("preserve_scale", config["preserve_scale"])),
        0.0,
        1.0,
    )
    return config


def _build_leaf_ik_chains(bones_setup, leaf_ik_refine):
    if not leaf_ik_refine.get("enabled", False):
        return []

    bones_by_name = {bone["name"]: bone for bone in bones_setup}
    children_by_name = {bone["name"]: [] for bone in bones_setup}
    for bone in bones_setup:
        parent_name = bone.get("parent")
        if parent_name in children_by_name:
            children_by_name[parent_name].append(bone["name"])

    candidates = []
    for bone_info in bones_setup:
        if not _is_leaf_ik_effector_candidate(bone_info):
            continue
        chain_bones = _build_leaf_ik_chain_for_effector(
            bone_info["name"],
            bones_by_name,
            children_by_name,
            leaf_ik_refine["max_chain_length"],
        )
        if len(chain_bones) < 2:
            continue
        total_setup_length = sum(
            max(float(bones_by_name[bone_name].get("length", 0.0)), 0.0)
            for bone_name in chain_bones
        )
        if total_setup_length <= SHORT_BONE_LENGTH:
            continue
        candidates.append(
            {
                "effector": bone_info["name"],
                "bones": chain_bones,
                "total_setup_length": total_setup_length,
                "score": _leaf_ik_effector_score(bone_info),
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    used_bones = set()
    selected_effectors = []
    for candidate in candidates:
        candidate_bones = set(candidate["bones"])
        if used_bones & candidate_bones:
            continue
        if any(
            _is_ancestor_bone(bones_by_name, effect_name, candidate["effector"])
            or _is_ancestor_bone(bones_by_name, candidate["effector"], effect_name)
            for effect_name in selected_effectors
        ):
            continue
        selected.append(candidate)
        selected_effectors.append(candidate["effector"])
        used_bones.update(candidate["bones"])
    return selected


def _is_leaf_ik_effector_candidate(bone_info):
    if not bone_info.get("parent"):
        return False

    length = float(bone_info.get("length", 0.0))
    length_ratio = float(bone_info.get("length_ratio", 0.0))
    child_count = int(bone_info.get("child_count", 0))
    descendant_count = int(bone_info.get("descendant_count", child_count))
    leaf_count = int(bone_info.get("leaf_count", 1 if child_count == 0 else 0))
    leaf_distance = int(bone_info.get("leaf_distance", 99))

    if length <= BONE_SCALE_EPSILON:
        return False
    if child_count == 0:
        return length >= SHORT_BONE_LENGTH or length_ratio >= 0.6
    if child_count > 1:
        return leaf_count >= 2 and descendant_count <= 24 and length_ratio >= 1.0
    return (
        descendant_count <= 4
        and leaf_distance <= 2
        and (length >= SHORT_BONE_LENGTH or length_ratio >= 1.0)
    )


def _leaf_ik_effector_score(bone_info):
    child_count = int(bone_info.get("child_count", 0))
    leaf_count = int(bone_info.get("leaf_count", 1 if child_count == 0 else 0))
    descendant_count = int(bone_info.get("descendant_count", child_count))
    leaf_distance = int(bone_info.get("leaf_distance", 99))
    length_ratio = float(bone_info.get("length_ratio", 1.0))
    branching_bonus = 2.5 if child_count > 1 else 0.0
    return (
        2.0 * min(length_ratio, 10.0)
        + 0.65 * min(leaf_count, 8)
        + 0.12 * min(descendant_count, 24)
        + branching_bonus
        - 0.18 * min(max(leaf_distance - 1, 0), 6)
    )


def _build_leaf_ik_chain_for_effector(
    effector_name,
    bones_by_name,
    children_by_name,
    max_chain_length,
):
    chain = [effector_name]
    current_name = effector_name
    while len(chain) < max_chain_length:
        parent_name = bones_by_name[current_name].get("parent")
        if not parent_name:
            break
        if len(children_by_name.get(parent_name, [])) > 1 and len(chain) >= 2:
            break
        chain.append(parent_name)
        current_name = parent_name
    chain.reverse()
    return chain


def _is_ancestor_bone(bones_by_name, candidate_ancestor, bone_name):
    current_name = bone_name
    while current_name:
        if current_name == candidate_ancestor:
            return True
        current_bone = bones_by_name.get(current_name)
        if current_bone is None:
            return False
        current_name = current_bone.get("parent")
    return False


def _refine_frame_local_poses_with_leaf_ik(
    frame_poses,
    previous_poses,
    bones_setup,
    projected_segments,
    leaf_ik_chains,
    leaf_ik_refine,
):
    if not leaf_ik_refine.get("enabled", False) or not leaf_ik_chains:
        return frame_poses

    refined = {name: pose.copy() for name, pose in frame_poses.items()}
    bones_by_name = {bone["name"]: bone for bone in bones_setup}
    strength = float(leaf_ik_refine.get("strength", 0.0))
    preserve_scale = float(leaf_ik_refine.get("preserve_scale", 0.0))
    iterations = int(leaf_ik_refine.get("iterations", 1))
    max_angle_step = 18.0 + 42.0 * strength

    for chain in leaf_ik_chains:
        effector_name = chain["effector"]
        projected = projected_segments.get(effector_name)
        if not projected:
            continue

        working_poses = {name: pose.copy() for name, pose in refined.items()}
        for bone_name in chain["bones"]:
            current_scale = abs(float(working_poses[bone_name].get("scale_x", 1.0)))
            anchored_scale = _clamp(
                _lerp(current_scale, 1.0, preserve_scale),
                0.55,
                SOFT_SCALE_MAX,
            )
            working_poses[bone_name]["scale_x"] = anchored_scale

        world_cache = _compute_pose_world_cache(working_poses, bones_setup)
        root_head = world_cache[chain["bones"][0]]["head"]
        chain_reach = sum(
            max(float(bones_by_name[bone_name].get("length", 0.0)), BONE_SCALE_EPSILON)
            * abs(float(working_poses[bone_name].get("scale_x", 1.0)))
            for bone_name in chain["bones"]
        )
        if chain_reach <= BONE_SCALE_EPSILON:
            continue

        target_tail = _clip_ik_target_to_reach(
            np.asarray(projected["tail"], dtype=np.float64),
            root_head,
            chain_reach,
        )
        current_tail = world_cache[effector_name]["tail"]
        tolerance = max(0.0025, chain_reach * 0.02)
        current_scale_error = max(
            abs(abs(float(frame_poses[bone_name].get("scale_x", 1.0))) - 1.0)
            for bone_name in chain["bones"]
        )
        if (
            float(np.linalg.norm(current_tail - target_tail)) <= tolerance
            and current_scale_error < 0.15
        ):
            continue

        _solve_leaf_ik_chain_ccd(
            working_poses,
            bones_setup,
            chain["bones"],
            effector_name,
            target_tail,
            iterations=iterations,
            max_angle_step=max_angle_step,
            tolerance=tolerance,
        )

        for bone_name in chain["bones"]:
            original_pose = refined[bone_name]
            solved_pose = working_poses[bone_name]
            reference_pose = (previous_poses or {}).get(bone_name) or original_pose
            reference_rotation = float(reference_pose.get("rotation", original_pose["rotation"]))
            original_rotation = _unwrap_angle_near(
                float(original_pose["rotation"]), reference_rotation
            )
            solved_rotation = _unwrap_angle_near(float(solved_pose["rotation"]), reference_rotation)
            refined[bone_name]["rotation"] = _normalize_angle(
                _lerp_angle(original_rotation, solved_rotation, strength)
            )
            refined[bone_name]["scale_x"] = _clamp(
                _lerp(
                    abs(float(original_pose.get("scale_x", 1.0))),
                    abs(float(solved_pose.get("scale_x", 1.0))),
                    strength,
                ),
                SOFT_SCALE_MIN,
                SOFT_SCALE_MAX,
            )

    return refined


def _solve_leaf_ik_chain_ccd(
    frame_poses,
    bones_setup,
    chain_bones,
    effector_name,
    target_tail,
    *,
    iterations,
    max_angle_step,
    tolerance,
):
    if not chain_bones:
        return

    for _ in range(max(1, iterations)):
        world_cache = _compute_pose_world_cache(frame_poses, bones_setup)
        current_tail = world_cache[effector_name]["tail"]
        if float(np.linalg.norm(current_tail - target_tail)) <= tolerance:
            break

        for bone_name in reversed(chain_bones):
            world_cache = _compute_pose_world_cache(frame_poses, bones_setup)
            joint_head = world_cache[bone_name]["head"]
            current_tail = world_cache[effector_name]["tail"]
            current_vector = current_tail - joint_head
            target_vector = target_tail - joint_head
            if (
                float(np.linalg.norm(current_vector)) <= BONE_SCALE_EPSILON
                or float(np.linalg.norm(target_vector)) <= BONE_SCALE_EPSILON
            ):
                continue

            angle_delta = _signed_angle_between_vectors(current_vector, target_vector)
            angle_delta = _clamp(angle_delta, -max_angle_step, max_angle_step)
            if abs(angle_delta) <= 1e-3:
                continue

            frame_poses[bone_name]["rotation"] = _normalize_angle(
                float(frame_poses[bone_name]["rotation"]) + angle_delta
            )


def _compute_pose_world_cache(frame_poses, bones_setup):
    world_cache = {}
    for bone_info in bones_setup:
        bone_name = bone_info["name"]
        pose = frame_poses[bone_name]
        local_x = float(pose.get("x", bone_info.get("x", 0.0)))
        local_y = float(pose.get("y", bone_info.get("y", 0.0)))
        local_rotation = float(pose.get("rotation", bone_info.get("rotation", 0.0)))
        local_scale_x = float(pose.get("scale_x", 1.0))
        parent_name = bone_info.get("parent")
        inherit_mode = bone_info.get("inherit", "Normal")

        if parent_name:
            parent_state = world_cache[parent_name]
            world_head = parent_state["head"] + (
                parent_state["matrix"] @ np.array((local_x, local_y), dtype=np.float64)
            )
            world_matrix = _compose_world_matrix(
                parent_state,
                local_rotation,
                local_scale_x,
                inherit_mode,
            )
        else:
            world_head = np.array((local_x, local_y), dtype=np.float64)
            world_matrix = _build_2d_basis(local_rotation, local_scale_x)

        tail_offset = world_matrix @ np.array(
            (max(float(bone_info.get("length", 0.0)), 0.0), 0.0),
            dtype=np.float64,
        )
        world_cache[bone_name] = {
            "head": world_head,
            "tail": world_head + tail_offset,
            "matrix": world_matrix,
            "rigid_matrix": orthonormalize_2x2(world_matrix, BONE_SCALE_EPSILON),
        }

    return world_cache


def _clip_ik_target_to_reach(target, root_head, chain_reach):
    offset = target - root_head
    distance = float(np.linalg.norm(offset))
    if distance <= BONE_SCALE_EPSILON or distance <= chain_reach:
        return target
    return root_head + (offset / distance) * chain_reach


def _signed_angle_between_vectors(current_vector, target_vector):
    current_norm = float(np.linalg.norm(current_vector))
    target_norm = float(np.linalg.norm(target_vector))
    if current_norm <= BONE_SCALE_EPSILON or target_norm <= BONE_SCALE_EPSILON:
        return 0.0

    current_unit = current_vector / current_norm
    target_unit = target_vector / target_norm
    cross_value = float(current_unit[0] * target_unit[1] - current_unit[1] * target_unit[0])
    dot_value = _clamp(float(np.dot(current_unit, target_unit)), -1.0, 1.0)
    return math.degrees(math.atan2(cross_value, dot_value))


def _normalize_pose_mode(pose_mode):
    mode = str(pose_mode or "full")
    if mode not in {"full", "rotation_only", "local_rotation", "blend"}:
        return "full"
    return mode


def _is_rotation_pose_mode(pose_mode):
    return _normalize_pose_mode(pose_mode) in {"rotation_only", "local_rotation"}


def _evaluate_exported_local_pose(bone_info, timelines, sample_time):
    base_x = float(bone_info.get("x", 0.0))
    base_y = float(bone_info.get("y", 0.0))
    base_rotation = float(bone_info.get("rotation", 0.0))
    base_scale_x = float(bone_info.get("scale_x", 1.0))
    base_scale_y = float(bone_info.get("scale_y", 1.0))
    base_shear_x = float(bone_info.get("shear_x", 0.0))
    base_shear_y = float(bone_info.get("shear_y", 0.0))

    translate_x, translate_y = _sample_exported_timeline_2d(
        timelines.get("translate") or [],
        sample_time,
        default_x=0.0,
        default_y=0.0,
    )
    scale_x, scale_y = _sample_exported_timeline_2d(
        timelines.get("scale") or [],
        sample_time,
        default_x=1.0,
        default_y=1.0,
    )
    shear_x, shear_y = _sample_exported_timeline_2d(
        timelines.get("shear") or [],
        sample_time,
        default_x=0.0,
        default_y=0.0,
    )
    rotation_offset = _sample_exported_timeline_1d(
        timelines.get("rotate") or [],
        sample_time,
        default_value=0.0,
        value_key="angle",
        is_angle=True,
    )

    return {
        "x": base_x + translate_x,
        "y": base_y + translate_y,
        "rotation": _normalize_angle(base_rotation + rotation_offset),
        "scale_x": base_scale_x * scale_x,
        "scale_y": base_scale_y * scale_y,
        "shear_x": base_shear_x + shear_x,
        "shear_y": base_shear_y + shear_y,
    }


def _sample_exported_timeline_1d(
    keyframes,
    sample_time,
    *,
    default_value,
    value_key,
    is_angle=False,
):
    if not keyframes:
        return float(default_value)

    first = keyframes[0]
    first_time = float(first.get("time", 0.0))
    first_value = float(first.get(value_key, default_value))
    if sample_time <= first_time + 1e-8:
        return first_value if first_time <= 1e-4 else float(default_value)

    last = keyframes[-1]
    last_time = float(last.get("time", first_time))
    if sample_time >= last_time - 1e-8:
        return float(last.get(value_key, default_value))

    for prev_key, next_key in zip(keyframes, keyframes[1:]):
        prev_time = float(prev_key.get("time", 0.0))
        next_time = float(next_key.get("time", prev_time))
        if sample_time > next_time + 1e-8:
            continue

        prev_value = float(prev_key.get(value_key, default_value))
        next_value = float(next_key.get(value_key, prev_value))
        if next_time <= prev_time + 1e-8:
            return next_value
        weight = (sample_time - prev_time) / (next_time - prev_time)
        if is_angle:
            return _lerp_angle(prev_value, next_value, weight)
        return _lerp(prev_value, next_value, weight)

    return float(last.get(value_key, default_value))


def _sample_exported_timeline_2d(
    keyframes,
    sample_time,
    *,
    default_x,
    default_y,
):
    if not keyframes:
        return float(default_x), float(default_y)

    first = keyframes[0]
    first_time = float(first.get("time", 0.0))
    first_x = float(first.get("x", default_x))
    first_y = float(first.get("y", default_y))
    if sample_time <= first_time + 1e-8:
        if first_time <= 1e-4:
            return first_x, first_y
        return float(default_x), float(default_y)

    last = keyframes[-1]
    last_time = float(last.get("time", first_time))
    if sample_time >= last_time - 1e-8:
        return float(last.get("x", default_x)), float(last.get("y", default_y))

    for prev_key, next_key in zip(keyframes, keyframes[1:]):
        prev_time = float(prev_key.get("time", 0.0))
        next_time = float(next_key.get("time", prev_time))
        if sample_time > next_time + 1e-8:
            continue

        prev_x = float(prev_key.get("x", default_x))
        prev_y = float(prev_key.get("y", default_y))
        next_x = float(next_key.get("x", prev_x))
        next_y = float(next_key.get("y", prev_y))
        if next_time <= prev_time + 1e-8:
            return next_x, next_y
        weight = (sample_time - prev_time) / (next_time - prev_time)
        return _lerp(prev_x, next_x, weight), _lerp(prev_y, next_y, weight)

    return float(last.get("x", default_x)), float(last.get("y", default_y))


def _blend_frame_local_poses_2d(rotation_poses, full_poses, blend_weight):
    if blend_weight <= 0.0:
        return {name: pose.copy() for name, pose in rotation_poses.items()}
    if blend_weight >= 1.0:
        return {name: pose.copy() for name, pose in full_poses.items()}

    blended = {}
    for name, rotation_pose in rotation_poses.items():
        full_pose = full_poses.get(name, rotation_pose)
        blended[name] = {
            "x": _lerp(float(rotation_pose["x"]), float(full_pose["x"]), blend_weight),
            "y": _lerp(float(rotation_pose["y"]), float(full_pose["y"]), blend_weight),
            "rotation": _normalize_angle(
                _lerp_angle(
                    float(rotation_pose["rotation"]), float(full_pose["rotation"]), blend_weight
                )
            ),
            "length": float(rotation_pose["length"]),
            "scale_x": _lerp(
                float(rotation_pose["scale_x"]), float(full_pose["scale_x"]), blend_weight
            ),
        }
    return blended


def _compile_bone_matcher(spec):
    if not spec:
        return lambda bone_name: False

    tokens = [token.strip() for token in re.split(r"[;,]+", spec) if token.strip()]
    if len(tokens) > 1:
        normalized = {token.casefold() for token in tokens}

        def match_token_list(bone_name):
            short_name = bone_name.split(":")[-1].casefold()
            full_name = bone_name.casefold()
            return short_name in normalized or full_name in normalized

        return match_token_list

    try:
        pattern = re.compile(spec, re.IGNORECASE)
        return lambda bone_name: bool(pattern.search(bone_name))
    except re.error:
        needle = spec.casefold()
        return (
            lambda bone_name: needle in bone_name.casefold()
            or needle in bone_name.split(":")[-1].casefold()
        )


def _bone_allows_translate(bone_info, connected_translation, pose_mode):
    if bone_info.get("parent") is None:
        return True
    if _is_rotation_pose_mode(pose_mode):
        return False
    if not bone_info.get("connected", False):
        return True
    return _connected_translation_applies(bone_info, connected_translation)


def _connected_translation_applies(bone_info, connected_translation):
    scope = (connected_translation or {}).get("scope", "none")
    if scope == "none":
        return False
    if scope == "all":
        return True
    if scope == "custom":
        matcher = (connected_translation or {}).get("matcher")
        return bool(matcher and matcher(bone_info["name"]))
    if scope == "terminal":
        return bool(bone_info.get("terminal_chain") or bone_info.get("terminal_chain_root"))
    if scope == "limbs":
        if bone_info.get("main_chain"):
            return False
        return bool(
            bone_info.get("terminal_chain")
            or bone_info.get("terminal_chain_root")
            or bone_info.get("leaf_distance", 99) <= 6
            or bone_info.get("parent_child_count", 0) > 1
        )
    return False


def _rotation_flatten_applies(bone_info, rotation_flatten):
    if not rotation_flatten or rotation_flatten.get("amount", 0.0) <= 0.0:
        return False
    if float(bone_info.get("length", 0.0)) <= BONE_SCALE_EPSILON:
        return False

    scope = rotation_flatten.get("scope", "all")
    if scope == "custom":
        matcher = rotation_flatten.get("matcher")
        return bool(matcher and matcher(bone_info["name"]))

    if bone_info.get("parent") is None:
        return False
    if scope == "all":
        return True
    if scope == "limbs":
        if bone_info.get("main_chain"):
            return False
        if bone_info.get("terminal_chain"):
            return False
        return bool(
            bone_info.get("leaf_distance", 99) <= 6 or bone_info.get("parent_child_count", 0) > 1
        )
    return False


def _apply_rotation_flatten(segment, current_length, bone_info, rotation_flatten):
    if not _rotation_flatten_applies(bone_info, rotation_flatten):
        return segment, current_length

    target_length = max(float(bone_info.get("length", current_length)), BONE_SCALE_EPSILON)
    amount = _clamp(float(rotation_flatten.get("amount", 0.0)), 0.0, 1.0)
    length_ratio = current_length / max(target_length, BONE_SCALE_EPSILON)
    confidence = _clamp(
        (length_ratio - ROTATION_FLATTEN_MIN_RATIO)
        / max(ROTATION_FLATTEN_FULL_RATIO - ROTATION_FLATTEN_MIN_RATIO, BONE_SCALE_EPSILON),
        0.0,
        1.0,
    )
    if _is_terminal_chain_bone(bone_info) or bone_info.get("leaf_distance", 99) <= 2:
        confidence *= 0.5
    effective_amount = amount * confidence
    if effective_amount <= 0.0:
        return segment, current_length

    flattened_length = _lerp(current_length, target_length, effective_amount)
    if flattened_length <= BONE_SCALE_EPSILON:
        return segment, current_length
    if current_length <= BONE_SCALE_EPSILON:
        return segment, current_length
    return segment * (flattened_length / current_length), flattened_length


def _stabilize_frame_local_poses_2d(
    frame_poses,
    previous_poses,
    bones_setup,
    stretch_guard=None,
):
    """Regularize unstable local transforms caused by depth collapse in the projection."""
    if not previous_poses:
        if not stretch_guard or not stretch_guard.get("enabled", False):
            return frame_poses
        return {
            bone_info["name"]: _apply_stretch_guard(
                frame_poses[bone_info["name"]].copy(),
                frame_poses[bone_info["name"]],
                bone_info,
                stretch_guard,
            )
            for bone_info in bones_setup
        }

    stabilized = {}
    for bone_info in bones_setup:
        name = bone_info["name"]
        raw = frame_poses[name]
        prev = previous_poses.get(name, raw)
        parent_name = bone_info.get("parent")
        parent_scale = stabilized[parent_name]["scale_x"] if parent_name in stabilized else 1.0
        instability = _bone_instability_score(raw, prev, bone_info, parent_scale)

        if instability <= 0.0:
            stabilized[name] = _apply_stretch_guard(raw.copy(), prev, bone_info, stretch_guard)
            continue

        anchor_weight = min(0.85, 0.2 + instability * 0.55)
        if bone_info.get("terminal_chain_root"):
            anchor_weight = min(0.9, anchor_weight + 0.1)
        pos_weight = min(
            0.9,
            anchor_weight + max(0.0, (COLLAPSED_SCALE_THRESHOLD - abs(parent_scale))) * 0.5,
        )

        blended = raw.copy()
        blended["rotation"] = _normalize_angle(
            _lerp_angle(raw["rotation"], prev["rotation"], anchor_weight)
        )

        scale_anchor = _clamp(prev["scale_x"], SOFT_SCALE_MIN, SOFT_SCALE_MAX)
        if _is_terminal_chain_bone(bone_info):
            scale_anchor = _clamp(scale_anchor, 0.5, 1.75)
        blended["scale_x"] = _lerp(raw["scale_x"], scale_anchor, anchor_weight)
        blended["scale_x"] = _clamp(blended["scale_x"], SOFT_SCALE_MIN, SOFT_SCALE_MAX)

        if not bone_info.get("connected", False):
            if bone_info.get("terminal_chain_root") or (
                _is_terminal_chain_bone(bone_info) and abs(parent_scale) < 0.4
            ):
                anchor_x = bone_info.get("x", 0.0)
                anchor_y = bone_info.get("y", 0.0)
            else:
                anchor_x = prev["x"]
                anchor_y = prev["y"]
            blended["x"] = _lerp(raw["x"], anchor_x, pos_weight)
            blended["y"] = _lerp(raw["y"], anchor_y, pos_weight)

        stabilized[name] = _apply_stretch_guard(blended, prev, bone_info, stretch_guard)

    return stabilized


def _apply_stretch_guard(pose, previous_pose, bone_info, stretch_guard):
    """Soft-limit excessive 2D scale on selected bones."""
    if not stretch_guard or not stretch_guard.get("enabled", False):
        return pose
    if not _stretch_guard_applies(bone_info, stretch_guard):
        return pose

    max_scale = max(1.0, float(stretch_guard.get("max_scale", SOFT_SCALE_MAX)))
    strength = _clamp(float(stretch_guard.get("strength", 0.0)), 0.0, 1.0)
    scale_value = float(pose["scale_x"])
    scale_abs = abs(scale_value)

    if scale_abs <= max_scale or strength <= 0.0:
        return pose

    target_scale = math.copysign(max_scale, scale_value)
    excess_ratio = _clamp((scale_abs - max_scale) / max(max_scale, BONE_SCALE_EPSILON), 0.0, 1.0)
    effective_strength = _clamp(strength + (1.0 - strength) * excess_ratio, 0.0, 1.0)
    pose["scale_x"] = _lerp(scale_value, target_scale, effective_strength)

    if previous_pose is not None:
        prev_scale = float(previous_pose.get("scale_x", pose["scale_x"]))
        if abs(prev_scale) <= max_scale:
            pose["scale_x"] = _lerp(pose["scale_x"], prev_scale, 0.15 * strength)

    pose["scale_x"] = _clamp(pose["scale_x"], -SOFT_SCALE_MAX, SOFT_SCALE_MAX)
    return pose


def _stretch_guard_applies(bone_info, stretch_guard):
    scope = stretch_guard.get("bones", "all")
    is_terminal = _is_terminal_chain_bone(bone_info)
    if scope == "terminal":
        return is_terminal
    if scope == "nonterminal":
        return not is_terminal
    return True


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


def _bone_instability_score(raw, prev, bone_info, parent_scale):
    """Estimate how ambiguous the current 2D solve is for a bone."""
    length = float(bone_info.get("length", 0.0))
    length_ratio = float(bone_info.get("length_ratio", 1.0))
    leaf_distance = int(bone_info.get("leaf_distance", 99))
    short_bone = (
        _is_terminal_chain_bone(bone_info)
        or length <= SHORT_BONE_LENGTH
        or (length_ratio <= 0.55 and leaf_distance <= 3)
    )
    if not short_bone:
        return 0.0

    scale_now = abs(float(raw["scale_x"]))
    scale_prev = abs(float(prev["scale_x"]))
    scale_jump = abs(scale_now - scale_prev)
    rotation_jump = abs(_normalize_angle(raw["rotation"] - prev["rotation"]))
    collapse = _clamp(
        (COLLAPSED_SCALE_THRESHOLD - scale_now) / COLLAPSED_SCALE_THRESHOLD,
        0.0,
        1.0,
    )
    stretch = _clamp((scale_now - 1.75) / 1.5, 0.0, 1.0)
    jump = _clamp((scale_jump - 0.35) / 1.25, 0.0, 1.0)
    parent_collapse = _clamp((0.55 - abs(parent_scale)) / 0.55, 0.0, 1.0)
    rotation = _clamp((rotation_jump - 40.0) / 120.0, 0.0, 1.0)

    offset = 0.0
    if not bone_info.get("connected", False):
        setup_x = float(bone_info.get("x", 0.0))
        setup_y = float(bone_info.get("y", 0.0))
        offset = math.dist((raw["x"], raw["y"]), (setup_x, setup_y))
        offset = _clamp(offset / max(length * 6.0, 0.08), 0.0, 1.0)

    score = (
        0.28 * collapse
        + 0.22 * stretch
        + 0.18 * jump
        + 0.18 * parent_collapse
        + 0.08 * rotation
        + 0.06 * offset
    )

    if bone_info.get("terminal_chain_root"):
        score += 0.1 * max(collapse, parent_collapse, offset)
    elif _is_terminal_chain_bone(bone_info):
        score += 0.05 * max(collapse, jump)

    return _clamp(score, 0.0, 1.0)


def _is_terminal_chain_bone(bone_info):
    return bool(bone_info.get("terminal_chain"))


def _unwrap_angle_near(angle, reference):
    """Shift an angle by whole turns so it stays close to a reference value."""
    if reference is None:
        return angle
    value = angle
    while value - reference > 180:
        value -= 360
    while value - reference < -180:
        value += 360
    return value


def _normalize_angle(angle):
    """Normalize an angle to the [-180, 180] range."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def _lerp(start, end, weight):
    return start + (end - start) * weight


def _lerp_angle(current, anchor, weight):
    return current + _normalize_angle(anchor - current) * weight


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _compose_world_matrix(parent_state, local_rotation, scale_x, inherit_mode):
    parent_basis = parent_state["matrix"]
    if inherit_mode == "NoScale":
        parent_basis = parent_state["rigid_matrix"]
    return parent_basis @ _build_2d_basis(local_rotation, scale_x)


def _basis_inverse_for_inherit(parent_state, inherit_mode):
    basis = parent_state["matrix"]
    if inherit_mode == "NoScale":
        basis = parent_state["rigid_matrix"]
    return safe_inverse_2x2(basis, BONE_SCALE_EPSILON)


def _build_2d_basis(rotation_deg, scale_x=1.0, scale_y=1.0):
    """Build the 2x2 matrix used by the 2D runtime for a local bone transform."""
    rotation_rad = math.radians(rotation_deg)
    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)
    return np.array(
        [
            [cos_r * scale_x, -sin_r * scale_y],
            [sin_r * scale_x, cos_r * scale_y],
        ],
        dtype=np.float64,
    )


def _build_2d_basis_with_shear(
    rotation_deg,
    scale_x=1.0,
    scale_y=1.0,
    shear_x_deg=0.0,
    shear_y_deg=0.0,
):
    """Build the 2x2 matrix for exported Spine-style local transforms."""
    rotation_rad = math.radians(float(rotation_deg))
    x_axis_angle = rotation_rad + math.radians(float(shear_x_deg))
    y_axis_angle = rotation_rad + (math.pi * 0.5) + math.radians(float(shear_y_deg))
    return np.array(
        [
            [math.cos(x_axis_angle) * scale_x, math.cos(y_axis_angle) * scale_y],
            [math.sin(x_axis_angle) * scale_x, math.sin(y_axis_angle) * scale_y],
        ],
        dtype=np.float64,
    )


def _optimize_keyframes(timelines):
    """Remove keyframes that don't contribute to the animation."""
    optimized = {}

    for timeline_name, keyframes in timelines.items():
        if not keyframes:
            continue

        if timeline_name == "rotate":
            optimized[timeline_name] = _optimize_timeline(
                keyframes,
                value_key="value",
                tolerance=0.5,
            )
        elif timeline_name == "translate":
            optimized[timeline_name] = _optimize_timeline_2d(keyframes, tolerance=0.001)
        elif timeline_name == "scale":
            optimized[timeline_name] = _optimize_timeline_2d(
                keyframes,
                tolerance=0.001,
                value_keys=("x", "y"),
            )

    return optimized


def _force_loop_closing_keys(timelines, duration):
    """Ensure looped timelines close explicitly at both t=0 and t=end."""
    closed = {}
    for timeline_name, keyframes in timelines.items():
        if not keyframes:
            continue
        closed[timeline_name] = _force_loop_closing_timeline(
            keyframes,
            timeline_name,
            duration,
        )
    return closed


def _force_loop_closing_timeline(keyframes, timeline_name, duration):
    if not keyframes:
        return keyframes

    result = [dict(keyframe) for keyframe in keyframes]
    start_key = _loop_start_key_for_timeline(result, timeline_name)
    end_key = dict(start_key)
    end_key["time"] = round(duration, 4)

    if result[0]["time"] > 1e-4:
        result.insert(0, start_key)
    else:
        result[0] = start_key

    if abs(result[-1]["time"] - duration) <= 1e-4:
        result[-1] = end_key
    else:
        result.append(end_key)

    return result


def _loop_start_key_for_timeline(keyframes, timeline_name):
    if timeline_name == "rotate":
        if abs(keyframes[0]["time"]) <= 1e-4:
            start_value = float(keyframes[0].get("value", keyframes[0].get("angle", 0.0)))
        else:
            start_value = 0.0
        start_value = round(start_value, 2)
        return {"time": 0.0, "angle": start_value, "value": start_value}

    if timeline_name == "translate":
        if abs(keyframes[0]["time"]) <= 1e-4:
            start_x = float(keyframes[0].get("x", 0.0))
            start_y = float(keyframes[0].get("y", 0.0))
        else:
            start_x = 0.0
            start_y = 0.0
        return {"time": 0.0, "x": round(start_x, 4), "y": round(start_y, 4)}

    if timeline_name == "scale":
        if abs(keyframes[0]["time"]) <= 1e-4:
            start_x = float(keyframes[0].get("x", 1.0))
            start_y = float(keyframes[0].get("y", 1.0))
        else:
            start_x = 1.0
            start_y = 1.0
        return {"time": 0.0, "x": round(start_x, 4), "y": round(start_y, 4)}

    return dict(keyframes[0])


def _optimize_timeline(keyframes, value_key, tolerance):
    """Remove redundant keyframes from a single-value timeline."""
    if len(keyframes) <= 2:
        return keyframes

    result = [keyframes[0]]
    for index in range(1, len(keyframes) - 1):
        prev_val = keyframes[index - 1][value_key]
        curr_val = keyframes[index][value_key]
        next_val = keyframes[index + 1][value_key]
        if abs(curr_val - prev_val) > tolerance or abs(curr_val - next_val) > tolerance:
            result.append(keyframes[index])

    result.append(keyframes[-1])
    return result


def _optimize_timeline_2d(keyframes, tolerance, value_keys=("x", "y")):
    """Remove redundant keyframes from a 2-value timeline."""
    if len(keyframes) <= 2:
        return keyframes

    result = [keyframes[0]]
    for index in range(1, len(keyframes) - 1):
        changed = False
        for key in value_keys:
            prev_val = keyframes[index - 1][key]
            curr_val = keyframes[index][key]
            next_val = keyframes[index + 1][key]
            if abs(curr_val - prev_val) > tolerance or abs(curr_val - next_val) > tolerance:
                changed = True
                break
        if changed:
            result.append(keyframes[index])

    result.append(keyframes[-1])
    return result
