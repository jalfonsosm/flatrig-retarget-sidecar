"""Rig-agnostic orientation helpers for imported Blender scenes."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None


def structural_root_bone(armature_obj, deform_bone_names_fn: Callable | None = None):
    """Choose the top-level bone that owns the largest part of the deform rig."""
    roots = [bone for bone in armature_obj.data.bones if bone.parent is None]
    if not roots:
        return None

    deform_names = (
        deform_bone_names_fn(armature_obj)
        if deform_bone_names_fn is not None
        else set()
    )

    def deform_descendant_count(root):
        pending = list(root.children)
        count = 0
        while pending:
            child = pending.pop()
            if not deform_names or child.name in deform_names:
                count += 1
            pending.extend(child.children)
        return count

    def subtree_has_deform(root):
        if not deform_names:
            return True
        if root.name in deform_names:
            return True
        pending = list(root.children)
        while pending:
            child = pending.pop()
            if child.name in deform_names:
                return True
            pending.extend(child.children)
        return False

    candidates = [root for root in roots if subtree_has_deform(root)] or roots
    return max(candidates, key=lambda bone: (deform_descendant_count(bone), float(bone.length)))


def _horizontal_direction(vector):
    horizontal = mathutils.Vector((float(vector.x), float(vector.y), 0.0))
    if horizontal.length < 1e-6:
        return None
    return horizontal.normalized()


def _children_forward_world(root_head, child_heads, sign_hint):
    """Infer a facing normal from the widest opposing pair of root children."""
    if len(child_heads) < 2:
        return None
    child_vectors = []
    for child_head in child_heads:
        vector = child_head - root_head
        if vector.length >= 1e-6:
            child_vectors.append(vector.normalized())
    if len(child_vectors) < 2:
        return None

    best_pair = None
    best_score = -float("inf")
    for index, first in enumerate(child_vectors):
        for second in child_vectors[index + 1 :]:
            first_xy = _horizontal_direction(first)
            second_xy = _horizontal_direction(second)
            if first_xy is None or second_xy is None:
                continue
            opposition = -first_xy.dot(second_xy)
            vertical_match = 1.0 - min(1.0, abs(float(first.z - second.z)))
            score = opposition + 0.5 * vertical_match
            if score > best_score:
                best_score = score
                best_pair = (first, second)
    if best_pair is None:
        return None

    lateral = _horizontal_direction(best_pair[0] - best_pair[1])
    if lateral is None:
        return None
    up = mathutils.Vector((0.0, 0.0, 1.0))
    forward = up.cross(lateral)
    if forward.length < 1e-6:
        return None

    if sign_hint is not None and forward.dot(sign_hint) < 0.0:
        forward.negate()
    return forward.normalized()


def _forward_from_root(root_world, root_head, child_heads):
    """Resolve facing from one structural root without semantic joint names."""
    root_axis = root_world.col[1].normalized()
    if abs(float(root_axis.z)) < 0.75:
        forward = _horizontal_direction(root_axis)
        if forward is not None:
            return forward

    sign_hint = _horizontal_direction(root_world.col[2])
    forward = _children_forward_world(root_head, child_heads, sign_hint)
    if forward is not None:
        return forward
    return sign_hint


def rig_forward_world(armature_obj, deform_bone_names_fn: Callable | None = None):
    """Return a rig-agnostic horizontal forward direction, or ``None``."""
    world = armature_obj.matrix_world
    root_bone = structural_root_bone(armature_obj, deform_bone_names_fn)
    if root_bone is None:
        return None

    root_world = (world @ root_bone.matrix_local).to_3x3()
    root_head = world @ root_bone.head_local
    child_heads = [world @ child.head_local for child in root_bone.children]
    return _forward_from_root(root_world, root_head, child_heads)


def posed_rig_forward_world(armature_obj, deform_bone_names_fn: Callable | None = None):
    """Return the structural root's facing in the currently evaluated pose."""
    data_root = structural_root_bone(armature_obj, deform_bone_names_fn)
    if data_root is None:
        return None
    pose_root = armature_obj.pose.bones.get(data_root.name)
    if pose_root is None:
        return None

    world = armature_obj.matrix_world
    root_world = (world @ pose_root.matrix).to_3x3()
    root_head = world @ pose_root.head
    child_heads = [world @ child.head for child in pose_root.children]
    return _forward_from_root(root_world, root_head, child_heads)


def _mesh_horizontal_principal_axes(mesh_objects):
    """Principal horizontal (XY) axes of the combined mesh point cloud."""
    points = []
    for mesh_obj in mesh_objects:
        world = mesh_obj.matrix_world
        verts = mesh_obj.data.vertices
        count = len(verts)
        if count == 0:
            continue
        step = max(1, count // 4000)
        for index in range(0, count, step):
            co = world @ verts[index].co
            points.append((float(co.x), float(co.y)))
    if len(points) < 8:
        return None
    cloud = np.asarray(points, dtype=np.float64)
    cloud -= cloud.mean(axis=0)
    covariance = cloud.T @ cloud
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    depth_axis = eigenvectors[:, 0]
    lateral_axis = eigenvectors[:, 1]
    return depth_axis, lateral_axis, float(eigenvalues[0]), float(eigenvalues[1])


def _mesh_facing_sign_along(mesh_objects, axis_2d) -> float:
    """Sign (+1/-1) placing `axis_2d` toward the head's horizontal protrusion."""
    points = []
    for mesh_obj in mesh_objects:
        world = mesh_obj.matrix_world
        for vert in mesh_obj.data.vertices:
            co = world @ vert.co
            points.append((float(co.x), float(co.y), float(co.z)))
    if len(points) < 8:
        return 1.0
    cloud = np.asarray(points, dtype=np.float64)
    z = cloud[:, 2]
    z_span = float(z.max() - z.min())
    if z_span < 1e-9:
        return 1.0
    top = cloud[z > z.max() - 0.30 * z_span][:, :2]
    if len(top) < 3:
        return 1.0
    centre = top.mean(axis=0)
    relative = top - centre
    distances = np.linalg.norm(relative, axis=1)
    tip = relative[int(distances.argmax())]
    projection = float(tip @ np.asarray(axis_2d, dtype=np.float64))
    if abs(projection) < 1e-9:
        return 1.0
    return 1.0 if projection > 0.0 else -1.0


def _disambiguate_forward_with_mesh(forward, mesh_objects):
    """Correct a skeleton-inferred forward that points along the mesh's lateral."""
    axes = _mesh_horizontal_principal_axes(mesh_objects)
    if axes is None:
        return forward
    depth_axis, lateral_axis, depth_var, lateral_var = axes
    forward_2d = np.array((float(forward.x), float(forward.y)), dtype=np.float64)
    norm = float(np.linalg.norm(forward_2d))
    if norm < 1e-9 or lateral_var < 1e-9:
        return forward
    forward_2d /= norm
    align_lateral = abs(float(forward_2d @ lateral_axis))
    align_depth = abs(float(forward_2d @ depth_axis))
    if lateral_var > 1.6 * depth_var and align_lateral > align_depth:
        sign = _mesh_facing_sign_along(mesh_objects, depth_axis)
        corrected = depth_axis * sign
        result = mathutils.Vector((float(corrected[0]), float(corrected[1]), 0.0))
        if result.length > 1e-9:
            print(
                "[blender_scene_io] mesh-disambiguated facing: skeleton forward "
                f"{tuple(round(v, 2) for v in forward)} aligned with the mesh "
                f"lateral axis; using narrow axis {tuple(round(v, 2) for v in result.normalized())}"
            )
            return result.normalized()
    return forward


def _feet_forward_sign(armature_obj, forward):
    """Return ``forward`` with its front/back sign corrected by the feet."""
    if armature_obj is None or forward is None:
        return forward
    bones = list(armature_obj.data.bones)
    if not bones:
        return forward
    world = armature_obj.matrix_world
    heads = [world @ bone.head_local for bone in bones]
    tails = [world @ bone.tail_local for bone in bones]
    zs = [p.z for p in heads] + [p.z for p in tails]
    z_min, z_max = min(zs), max(zs)
    if z_max - z_min < 1e-6:
        return forward
    low_cut = z_min + 0.25 * (z_max - z_min)
    feet = mathutils.Vector((0.0, 0.0, 0.0))
    count = 0
    for head, tail in zip(heads, tails):
        if max(head.z, tail.z) > low_cut:
            continue
        direction = tail - head
        if direction.length < 1e-6:
            continue
        direction = direction.normalized()
        if abs(direction.z) > 0.6:
            continue
        feet += mathutils.Vector((direction.x, direction.y, 0.0))
        count += 1
    if count == 0 or feet.length < 1e-6:
        return forward
    feet.normalize()
    forward_2d = mathutils.Vector((forward.x, forward.y, 0.0))
    if forward_2d.length < 1e-6:
        return forward
    forward_2d.normalize()
    alignment = forward_2d.x * feet.x + forward_2d.y * feet.y
    if abs(alignment) < 0.3:
        return forward
    if alignment < 0.0:
        print(
            "[blender_scene_io] flipped forward sign using feet direction "
            f"(feet={tuple(round(float(v), 2) for v in feet)}, "
            f"was={tuple(round(float(v), 2) for v in forward)})"
        )
        return mathutils.Vector((-forward.x, -forward.y, forward.z))
    return forward


def normalize_model_orientation(
    objects=None,
    *,
    target_forward=(0.0, -1.0, 0.0),
    deform_bone_names_fn: Callable | None = None,
) -> float:
    """Rotate just-imported rig objects about world Z so the rig faces canonical -Y."""
    if objects is None:
        objects = list(bpy.context.scene.objects)
    if not objects:
        return 0.0
    armature_obj = next((obj for obj in objects if obj.type == "ARMATURE"), None)
    if armature_obj is None:
        return 0.0
    forward = rig_forward_world(armature_obj, deform_bone_names_fn)
    if forward is None:
        return 0.0
    mesh_objects = [obj for obj in objects if obj.type == "MESH" and len(obj.data.vertices) > 0]
    if mesh_objects:
        forward = _disambiguate_forward_with_mesh(forward, mesh_objects)
    forward = _feet_forward_sign(armature_obj, forward)
    target = mathutils.Vector((float(target_forward[0]), float(target_forward[1]), 0.0))
    if target.length < 1e-6:
        return 0.0
    target.normalize()
    cross_z = forward.x * target.y - forward.y * target.x
    dot = max(-1.0, min(1.0, forward.x * target.x + forward.y * target.y))
    angle = math.atan2(cross_z, dot)
    if abs(angle) < math.radians(1.0):
        return 0.0
    pivot = armature_obj.matrix_world.translation.copy()
    rotation = (
        mathutils.Matrix.Translation(pivot)
        @ mathutils.Matrix.Rotation(angle, 4, "Z")
        @ mathutils.Matrix.Translation(-pivot)
    )
    imported = set(objects)
    for obj in objects:
        if obj.parent is None or obj.parent not in imported:
            obj.matrix_world = rotation @ obj.matrix_world
    bpy.context.view_layer.update()
    return math.degrees(angle)
