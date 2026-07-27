try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None

import math
import re
import json
import numpy as np
from collections import defaultdict

from .math_utils import *
from .bone_utils import *

__all__ = [
    "build_target_fk_cache",
    "_can_use_same_rig_local_pose_transfer",
    "_copy_source_local_pose_to_target",
    "_copy_source_pose_to_target"
]



def build_target_fk_cache(target_arm) -> dict:
    """Parent-first pose-bone order + each bone's rest rotation relative to its
    parent (numpy 3x3). Constant for a given armature, so the per-frame pose
    copy can reuse it instead of rederiving the topology every frame."""
    order = []
    seen = set()

    def visit(bone):
        if bone.name in seen:
            return
        if bone.parent is not None:
            visit(bone.parent)
        seen.add(bone.name)
        order.append(bone)

    for bone in target_arm.pose.bones:
        visit(bone)

    rest_local = {}
    for bone in order:
        if bone.parent is not None:
            rl = bone.parent.bone.matrix_local.inverted() @ bone.bone.matrix_local
        else:
            rl = bone.bone.matrix_local
        rest_local[bone.name] = _matrix3_to_np(rl.to_3x3())
    # Static rest world rotation per bone (parent chain under an identity root).
    # `order` is parent-first, so each parent is resolved before its children.
    # Used by the twist transfer to re-anchor the donor's roll on the target's
    # own rest mounting (independent of the posed parent -> no double-count).
    rest_world = {}
    for bone in order:
        parent_rw = rest_world.get(bone.parent.name) if bone.parent is not None else None
        if parent_rw is None:
            parent_rw = np.eye(3, dtype=np.float64)
        rest_world[bone.name] = parent_rw @ rest_local[bone.name]
    return {"order": order, "rest_local": rest_local, "rest_world": rest_world}


def _can_use_same_rig_local_pose_transfer(source_arm, target_arm, bone_map) -> bool:
    """Return true only when local pose channels are safe to copy verbatim.

    Same-rig animation should not pass through the direction-only retargeter:
    Blender already has the correct local channels, including any real
    translate/scale keys. The guard is structural: every source and target pose
    bone must map exactly once, parent links must agree under that mapping, and
    the mapped bones must share the same local rest orientation. Equal names are
    not enough: local animation channels are expressed in each bone's parent
    frame, so copying them is only mathematically valid when those frames match.
    """
    if source_arm is None or target_arm is None or not bone_map:
        return False
    source_bones = source_arm.pose.bones
    target_bones = target_arm.pose.bones
    mapped = [
        (src_name, tgt_name)
        for src_name, tgt_name in bone_map.items()
        if source_bones.get(src_name) is not None and target_bones.get(tgt_name) is not None
    ]
    if len(mapped) != len(source_bones) or len(mapped) != len(target_bones):
        return False

    for src_name, tgt_name in mapped:
        src_bone = source_bones.get(src_name)
        tgt_bone = target_bones.get(tgt_name)
        if src_bone is None or tgt_bone is None:
            return False

        src_parent = src_bone.parent.name if src_bone.parent is not None else None
        tgt_parent = tgt_bone.parent.name if tgt_bone.parent is not None else None
        mapped_src_parent = bone_map.get(src_parent) if src_parent is not None else None
        if src_parent is None or tgt_parent is None:
            if src_parent != tgt_parent:
                return False
        elif mapped_src_parent != tgt_parent:
            return False
        src_rest = _rest_local_quat(src_bone.bone)
        tgt_rest = _rest_local_quat(tgt_bone.bone)
        if _quat_angle_degrees(src_rest, tgt_rest) > 2.0:
            return False
    return True


def _copy_source_local_pose_to_target(source_arm, target_arm, bone_map) -> int:
    """Copy evaluated local pose channels from source to target.

    This is only valid for the guarded same-rig path. It preserves real source
    action location/rotation/scale keys while keeping the target armature's own
    rest head/tail positions and bone lengths.
    """
    for tgt_bone in target_arm.pose.bones:
        tgt_bone.location = (0.0, 0.0, 0.0)
        tgt_bone.rotation_mode = "QUATERNION"
        tgt_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        tgt_bone.scale = (1.0, 1.0, 1.0)

    posed = 0
    for src_name, tgt_name in bone_map.items():
        src_bone = source_arm.pose.bones.get(src_name)
        tgt_bone = target_arm.pose.bones.get(tgt_name)
        if src_bone is None or tgt_bone is None:
            continue
        tgt_bone.matrix_basis = src_bone.matrix_basis.copy()
        posed += 1
    return posed


def _copy_source_pose_to_target(
    source_arm, target_arm, bone_map, *, copy_root_location=False, fk_cache=None,
    transfer_twist=True,
) -> int:
    """Reproduce the source pose on the target by matching each mapped bone's
    world DIRECTION (plus the donor's twist motion about that direction) while
    keeping the target rig's own rest roll, solved top-down.

    Both rigs are facing-normalized at import, so pointing each target bone
    along its source bone's world direction follows the donor's motion. The
    target bone's REST ROLL is preserved (only the minimal rotation that aligns
    the directions is applied) — this is what makes it work on rigs whose bone
    axes differ from the donor's:

    * A raw local-channel copy applies the donor's swing about the target's
      differently-rolled local axis → a forward/back walk swung the auto-rigged
      piggy's legs sideways (plane rotated 90°).
    * Copying the donor's FULL ABSOLUTE world rotation forces the donor's static
      roll onto the target; since the piggy's auto-rig mounts its mesh against
      different local axes, that spun the head/face around (animated correctly
      but faced backwards).
    * Matching only the world DIRECTION fixes the swing AND preserves how the
      target mounts its mesh, but it DISCARDS the donor's twist about the bone
      axis. On a renamed-but-not-reoriented rig (e.g. a Mixamo target whose rest
      rolls differ 90-180° from a Mannequin donor) that lost twist reaches
      12-60° on upper arms / hands / fingers and visibly deforms hands.

    `transfer_twist` adds back only the donor's twist MOTION about the aimed
    axis (swing-twist split, `_twist_about_y_np`). It is rest-relative: the twist
    is the difference between the direction-only frame and the donor's
    rest-compensated rotation re-anchored on the target's own static rest, so at
    the donor's rest frame it is identity (no static roll is forced → the piggy
    face-spin stays fixed) and it degenerates to an exact copy when the rolls
    already agree (Mixamo Walk.fbx / Ch36 stay correct).

    The bind borrow and `_extract_transferred_animation` both call this so the
    bind/setup pose stays consistent with the transferred clip at the setup
    frame (the exported rig composes world = setup ∘ delta).
    """
    cache = fk_cache or build_target_fk_cache(target_arm)
    order = cache["order"]
    rest_local = cache["rest_local"]
    rest_world_static = cache.get("rest_world", {})

    inverse_map = {}
    for src_name, tgt_name in bone_map.items():
        inverse_map.setdefault(tgt_name, src_name)

    aw_src = _matrix3_to_np(source_arm.matrix_world.to_3x3())
    aw_tgt_inv = np.linalg.inv(_matrix3_to_np(target_arm.matrix_world.to_3x3()))
    bone_y = np.array((0.0, 1.0, 0.0), dtype=np.float64)

    posed = 0
    target_world = {}  # bone name -> resolved world rotation in target armature space
    for tgt_bone in order:
        parent = tgt_bone.parent
        parent_world = (
            target_world.get(parent.name) if parent is not None else None
        )
        if parent_world is None:
            parent_world = np.eye(3, dtype=np.float64)
        rest = rest_local[tgt_bone.name]
        rest_world = parent_world @ rest  # target bone's rest orientation under posed parent

        src_name = inverse_map.get(tgt_bone.name)
        src_bone = source_arm.pose.bones.get(src_name) if src_name else None
        if src_bone is not None:
            # Source bone full rotation and direction (+Y) in target armature space.
            src_pose_w = aw_tgt_inv @ (aw_src @ _matrix3_to_np(src_bone.matrix.to_3x3()))
            src_dir = src_pose_w @ bone_y
            rest_dir = rest_world @ bone_y
            align = _rotation_between_np(rest_dir, src_dir)
            desired_world = align @ rest_world  # reorient direction, keep target roll
            tgt_rest_static = rest_world_static.get(tgt_bone.name)
            if transfer_twist and tgt_rest_static is not None:
                # Graft the donor's twist about the aimed axis. rc_world is the
                # donor's rotation-from-rest re-anchored on the target's static
                # rest; keep only the twist of (direction-only)^-1 @ rc_world so
                # the aimed direction is untouched.
                src_rest_w = aw_tgt_inv @ (
                    aw_src @ _matrix3_to_np(src_bone.bone.matrix_local.to_3x3())
                )
                rc_world = (src_pose_w @ np.linalg.inv(src_rest_w)) @ tgt_rest_static
                twist = _twist_about_y_np(np.linalg.inv(desired_world) @ rc_world)
                desired_world = desired_world @ twist
            posed += 1
        else:
            # Unmapped target bone: leave it at its rest orientation.
            desired_world = rest_world

        basis = np.linalg.inv(rest_world) @ desired_world
        tgt_bone.rotation_mode = "QUATERNION"
        tgt_bone.rotation_quaternion = mathutils.Matrix(basis.tolist()).to_quaternion()
        if copy_root_location and tgt_bone.parent is None and src_bone is not None:
            tgt_bone.location = src_bone.location.copy()
        else:
            tgt_bone.location = (0.0, 0.0, 0.0)
        tgt_bone.scale = (1.0, 1.0, 1.0)
        target_world[tgt_bone.name] = desired_world

    return posed