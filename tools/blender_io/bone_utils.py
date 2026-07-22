try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None

import math
import re
_MIXAMO_PREFIX_RE = re.compile(r"^mixamorig\d*[:_]?", re.IGNORECASE)
import json

try:
    from blender_orientation import (
        rig_forward_world as _orientation_rig_forward_world,
        structural_root_bone as _orientation_structural_root_bone,
        posed_rig_forward_world as _orientation_posed_rig_forward_world,
    )
except ImportError:
    pass
import numpy as np
from collections import defaultdict

__all__ = [
    "_pose_bone_stem",
    "_pose_bone_tokens",
    "_pose_bone_side",
    "_pose_bone_role",
    "_role_fallbacks",
    "_stem_pose_bone_map",
    "_canonical_target_for_bone_name",
    "is_humanoid_biped",
    "_structural_root_bone",
    "_rig_forward_world",
    "_posed_rig_forward_world",
    "_mesh_uses_armature",
    "_weighted_bone_names",
    "_deform_bone_names",
    "_find_root_bone_name",
    "_bone_is_connected",
    "_topological_sort",
    "_sanitize_bvh_name",
    "_bvh_export_name"
]

# Per-process cache: the deform-bone classification scans every skin weight, so
# we compute it once per armature.
_DEFORM_BONE_NAME_CACHE: dict = {}




def _pose_bone_stem(name):
    colon = name.find(":")
    return name[colon + 1 :] if colon >= 0 else name


def _pose_bone_tokens(name):
    stem = _pose_bone_stem(str(name or ""))
    stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem)
    return [token for token in stem.lower().split("_") if token]


def _pose_bone_side(tokens):
    joined = "".join(tokens)
    if "left" in tokens or "l" in tokens or joined.startswith("left"):
        return "l"
    if "right" in tokens or "r" in tokens or joined.startswith("right"):
        return "r"
    return None


def _pose_bone_role(name):
    """Infer a coarse humanoid role for cross-format pose transfer.

    Exact name matching remains the primary path. This role fallback is only for
    common humanoid rigs whose names differ by convention (Mixamo, UE-style,
    KayKit, VRoid). The pose copy still writes rotations onto the target's own
    bones, preserving target joint positions and lengths.
    """
    tokens = _pose_bone_tokens(name)
    if not tokens:
        return None
    joined = "".join(tokens)
    side = _pose_bone_side(tokens)

    if "end" in tokens or "leaf" in tokens or "top" in tokens:
        return None
    if any(token in tokens for token in ("hips", "pelvis", "waist")):
        return "hips"
    if "head" in tokens:
        return "head"
    if "neck" in tokens:
        return "neck"

    if "spine" in tokens or joined.startswith("spine"):
        compact_suffix = None
        if len(tokens) == 1 and joined.startswith("spine") and joined != "spine":
            compact_suffix = joined[len("spine") :]
        if (
            "03" in tokens
            or "3" in tokens
            or compact_suffix in {"2", "02", "3", "03"}
        ):
            return "spine2"
        if "02" in tokens or "2" in tokens or compact_suffix in {"1", "01"}:
            return "spine1"
        if "01" in tokens or "1" in tokens:
            return "spine0"
        return "spine0"
    if "upperchest" in joined or ("upper" in tokens and "chest" in tokens):
        return "spine2"
    if "chest" in tokens or "torso" in tokens:
        return "spine2"

    if side is None:
        return None
    prefix = f"{side}_"

    if "clavicle" in tokens or "shoulder" in tokens:
        return prefix + "shoulder"
    if (
        "forearm" in joined
        or "lowerarm" in joined
        or ("fore" in tokens and "arm" in tokens)
        or ("lower" in tokens and "arm" in tokens)
    ):
        return prefix + "lower_arm"
    if "upperarm" in joined or ("upper" in tokens and "arm" in tokens):
        return prefix + "upper_arm"
    if "arm" in tokens or joined.endswith("arm"):
        return prefix + "upper_arm"
    if "hand" in tokens or "wrist" in tokens:
        return prefix + "hand"

    if "upleg" in joined or ("up" in tokens and "leg" in tokens) or "thigh" in tokens:
        return prefix + "thigh"
    if (
        ("lower" in tokens and "leg" in tokens)
        or "calf" in tokens
        or "shin" in tokens
        or (joined.endswith("leg") and "up" not in tokens)
    ):
        return prefix + "calf"
    if "foot" in tokens:
        return prefix + "foot"
    if "toe" in tokens or "toes" in tokens or "toebase" in joined:
        return prefix + "toe"

    return None


def _role_fallbacks(role):
    if role == "spine1":
        return ("spine1", "spine2", "spine0")
    if role == "spine2":
        return ("spine2", "spine1", "spine0")
    return (role,)


def _stem_pose_bone_map(source_arm, target_arm):
    """Map source pose-bone names to target pose-bone names.

    Exact namespace-stripped stem matching is preferred. If that misses, fall
    back to a coarse humanoid role map so a Mixamo donor can pose common
    non-Mixamo humanoid rigs without copying donor translations or bone lengths.
    Shared so the bind borrow and per-frame transfer never disagree.
    """
    bone_map = {}
    target_by_stem = {}
    for tgt_bone in target_arm.pose.bones:
        stem = _pose_bone_stem(tgt_bone.name).lower()
        if stem and stem not in target_by_stem:
            target_by_stem[stem] = tgt_bone.name
    for src_bone in source_arm.pose.bones:
        stem = _pose_bone_stem(src_bone.name).lower()
        tgt_name = target_by_stem.get(stem)
        if tgt_name is not None:
            bone_map[src_bone.name] = tgt_name

    used_targets = set(bone_map.values())
    target_by_role = {}
    for tgt_bone in target_arm.pose.bones:
        if tgt_bone.name in used_targets:
            continue
        role = _pose_bone_role(tgt_bone.name)
        if role and role not in target_by_role:
            target_by_role[role] = tgt_bone.name

    semantic_matches = 0
    for src_bone in source_arm.pose.bones:
        if src_bone.name in bone_map:
            continue
        role = _pose_bone_role(src_bone.name)
        if not role:
            continue
        for candidate_role in _role_fallbacks(role):
            tgt_name = target_by_role.get(candidate_role)
            if tgt_name is None or tgt_name in used_targets:
                continue
            bone_map[src_bone.name] = tgt_name
            used_targets.add(tgt_name)
            semantic_matches += 1
            break

    if semantic_matches:
        print(
            "[blender_scene_io] added "
            f"{semantic_matches} semantic humanoid pose-map match(es) "
            f"({len(bone_map)} total)."
        )
    return bone_map


def _canonical_target_for_bone_name(name: str):
    leaf = _strip_mixamo_prefix(name)
    if leaf in _CANONICAL_ALIASES:
        return _CANONICAL_ALIASES[leaf]
    key = re.sub(r"[^a-z0-9]+", "_", str(leaf).lower()).strip("_")
    return _CANONICAL_ALIAS_KEYS.get(key) or _CANONICAL_ALIAS_KEYS.get(key.replace("_", ""))


def is_humanoid_biped(bone_names) -> bool:
    """Only biped humanoids may be reduced; quadruped/winged/limbless are left as
    is (their names resolve almost nothing under the map)."""
    resolved = {_canonical_target_for_bone_name(n) for n in bone_names}
    resolved.discard(None)
    return _CANON_HUMANOID_CORE.issubset(resolved) and (
        "spine.1" in resolved or "spine.2" in resolved or "spine.3" in resolved
    )


def _structural_root_bone(armature_obj):
    return _orientation_structural_root_bone(armature_obj, _deform_bone_names)


def _rig_forward_world(armature_obj):
    return _orientation_rig_forward_world(armature_obj, _deform_bone_names)


def _posed_rig_forward_world(armature_obj):
    return _orientation_posed_rig_forward_world(armature_obj, _deform_bone_names)


def _weighted_bone_names(mesh_objects):
    weighted_names = set()
    for mesh_obj in mesh_objects:
        group_names = {group.index: group.name for group in mesh_obj.vertex_groups}
        for vertex in mesh_obj.data.vertices:
            for assignment in vertex.groups:
                if float(assignment.weight) <= 1e-6:
                    continue
                name = group_names.get(assignment.group)
                if name:
                    weighted_names.add(name)
    return weighted_names


def _deform_bone_names(armature_obj, mesh_objects=None):
    """Name-agnostic deform skeleton: bones that skin the mesh, plus their
    structural ancestors (pass-through connectors up to the root) and their
    descendant tips (unweighted leaf/``*_leaf`` bones that hang off a deform
    bone).

    This is the rig-format-independent way to separate the deform skeleton from
    a control rig: control bones (IK/pole/driver) carry no skin weight and live
    in their own subtree, so they are excluded, while the genuine deform tree —
    including tips that different authoring tools weight inconsistently — is kept
    whole. Two rigs that share a skeleton (e.g. mesh2motion vs quaternius) then
    yield the *same* deform set even though one wraps it in a control rig.
    Returns an empty set when there are no skin weights (BVH / anim-only rigs).
    """
    if armature_obj is None:
        return set()
    cache_key = getattr(armature_obj, "name", None)
    if cache_key:
        cached = _DEFORM_BONE_NAME_CACHE.get(cache_key)
        if cached is not None:
            return cached

    if mesh_objects is None:
        mesh_objects = _meshes_using_armature(armature_obj)
    weighted = _weighted_bone_names(mesh_objects) if mesh_objects else set()

    bones = list(armature_obj.data.bones)
    parent_of = {bone.name: (bone.parent.name if bone.parent else None) for bone in bones}
    children_of: dict = {}
    for name, parent in parent_of.items():
        children_of.setdefault(parent, []).append(name)

    deform = set(weighted)
    # Structural ancestors of every weighted bone (keeps pass-through connectors
    # such as the root above the pelvis).
    for name in weighted:
        cursor = parent_of.get(name)
        while cursor is not None and cursor not in deform:
            deform.add(cursor)
            cursor = parent_of.get(cursor)
    # Descendant tips of weighted bones (keeps unweighted leaves so tool-specific
    # leaf weighting differences do not change the deform set).
    stack = list(weighted)
    while stack:
        name = stack.pop()
        for child in children_of.get(name, []):
            if child not in deform:
                deform.add(child)
                stack.append(child)

    if cache_key:
        _DEFORM_BONE_NAME_CACHE[cache_key] = deform
    return deform


def _find_root_bone_name(armature):
    """Find the root bone name (bone with no parent)."""
    for bone in armature.data.bones:
        if bone.parent is None:
            return bone.name
    return None


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


def _sanitize_bvh_name(name, used_names):
    """Build a stable external matcher matching name from a Blender bone name."""
    base = BVH_SUFFIX_RE.sub("", str(name or "")).strip()
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "joint"
    if base[0].isdigit():
        base = f"joint_{base}"

    candidate = base
    suffix = 2
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _bvh_export_name(matching_name, index):
    return f"{matching_name}__{int(index):03d}"

def _meshes_using_armature(armature_obj):
    """Meshes skinned by ``armature_obj`` (parented or via an armature modifier)."""
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not getattr(obj.data, "vertices", None):
            continue
        if len(obj.data.vertices) == 0:
            continue
        if obj.parent is armature_obj or any(
            modifier.type == "ARMATURE" and modifier.object is armature_obj
            for modifier in obj.modifiers
        ):
            meshes.append(obj)
    return meshes

def _mesh_uses_armature(mesh_obj, armature_obj):
    if mesh_obj.parent is armature_obj:
        return True
    return any(
        modifier.type == "ARMATURE" and modifier.object is armature_obj
        for modifier in mesh_obj.modifiers
    )


BVH_SUFFIX_RE = re.compile(r"__[^\\s]{3}$")

def _strip_mixamo_prefix(name: str) -> str:
    stem = str(name or "")
    if ":" in stem:
        stem = stem.rsplit(":", 1)[-1]
    return _MIXAMO_PREFIX_RE.sub("", stem)



_CANONICAL_ALIASES = {
    # UE mannequin / Quaternius / Mesh2Motion deform names.
    "root": "core", "pelvis": "core",
    "spine_01": "spine.1", "spine_02": "spine.2", "spine_03": "spine.3",
    "neck_01": "neck", "head": "head",
    "clavicle_l": "collar.l", "clavicle_r": "collar.r",
    "upperarm_l": "shoulder.l", "upperarm_r": "shoulder.r",
    "lowerarm_l": "elbow.l", "lowerarm_r": "elbow.r",
    "hand_l": "wrist.l", "hand_r": "wrist.r",
    "thigh_l": "hip.l", "thigh_r": "hip.r",
    "calf_l": "knee.l", "calf_r": "knee.r",
    "foot_l": "ankle.l", "foot_r": "ankle.r",
    "ball_l": "foot.l", "ball_r": "foot.r",
    # HumanML3D / SMPL-style names.
    "left_hip": "hip.l", "right_hip": "hip.r",
    "spine1": "spine.1", "spine2": "spine.2", "spine3": "spine.3",
    "left_knee": "knee.l", "right_knee": "knee.r",
    "left_ankle": "ankle.l", "right_ankle": "ankle.r",
    "left_foot": "foot.l", "right_foot": "foot.r",
    "left_collar": "collar.l", "right_collar": "collar.r",
    "left_shoulder": "shoulder.l", "right_shoulder": "shoulder.r",
    "left_elbow": "elbow.l", "right_elbow": "elbow.r",
    "left_wrist": "wrist.l", "right_wrist": "wrist.r",
    # Mixamo names for callers that bypass import-time mannequin normalization.
    "Hips": "core", "Spine": "spine.1", "Spine1": "spine.2", "Spine2": "spine.3",
    "Neck": "neck", "Head": "head",
    "LeftShoulder": "collar.l", "RightShoulder": "collar.r",
    "LeftArm": "shoulder.l", "RightArm": "shoulder.r",
    "LeftForeArm": "elbow.l", "RightForeArm": "elbow.r",
    "LeftHand": "wrist.l", "RightHand": "wrist.r",
    "LeftUpLeg": "hip.l", "RightUpLeg": "hip.r",
    "LeftLeg": "knee.l", "RightLeg": "knee.r",
    "LeftFoot": "ankle.l", "RightFoot": "ankle.r",
    "LeftToeBase": "foot.l", "RightToeBase": "foot.r",
    # Legacy FlatRig/KayKit names for transitional already-reduced assets.
    "hips": "core", "spine": "spine.1", "chest": "spine.3",
    "upperarm.l": "shoulder.l", "upperarm.r": "shoulder.r",
    "lowerarm.l": "elbow.l", "lowerarm.r": "elbow.r",
    "wrist.l": "wrist.l", "wrist.r": "wrist.r",
    "hand.l": "wrist.l", "hand.r": "wrist.r",
    "upperleg.l": "hip.l", "upperleg.r": "hip.r",
    "lowerleg.l": "knee.l", "lowerleg.r": "knee.r",
    "foot.l": "ankle.l", "foot.r": "ankle.r",
    "toes.l": "foot.l", "toes.r": "foot.r",
    # Common auto-rig aliases.
    "upper_chest": "spine.3",
    "left_upper_leg": "hip.l", "right_upper_leg": "hip.r",
    "left_lower_leg": "knee.l", "right_lower_leg": "knee.r",
    "left_arm": "shoulder.l", "right_arm": "shoulder.r",
    "left_forearm": "elbow.l", "right_forearm": "elbow.r",
    "left_hand": "wrist.l", "right_hand": "wrist.r",
    # VRoid aliases.
    "jbipchest": "spine.2",
    "jbipupperchest": "spine.3",
    "jbiplupperarm": "shoulder.l",
    "jbiprupperarm": "shoulder.r",
    "jbipllowerarm": "elbow.l",
    "jbiprlowerarm": "elbow.r",
    "jbiplhand": "wrist.l",
    "jbiprhand": "wrist.r",
    "jbiplupperleg": "hip.l",
    "jbiprupperleg": "hip.r",
    "jbipllowerleg": "knee.l",
    "jbiprlowerleg": "knee.r",
    "jbiplfoot": "ankle.l",
    "jbiprfoot": "ankle.r",
    "jbipltoes": "foot.l",
    "jbiprtoes": "foot.r",
}

_CANONICAL_ALIAS_KEYS = {}
for _alias_name, _canon_name in _CANONICAL_ALIASES.items():
    _key = re.sub(r"[^a-z0-9]+", "_", _strip_mixamo_prefix(_alias_name).lower()).strip("_")
    _CANONICAL_ALIAS_KEYS[_key] = _canon_name
    _CANONICAL_ALIAS_KEYS[_key.replace("_", "")] = _canon_name
_CANON_HUMANOID_CORE = {"core", "head", "shoulder.l", "shoulder.r", "hip.l", "hip.r"}