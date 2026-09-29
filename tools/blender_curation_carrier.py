"""Blender-only implementation of declarative armature rest-pose rewrites."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy  # type: ignore


def _finite_vec3(value: object, *, field: str, bone_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} for bone '{bone_name}' must be a three-number array")
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} for bone '{bone_name}' must contain numbers") from exc
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{field} for bone '{bone_name}' must contain finite numbers")
    return vector


def _read_rest_spec(rest_spec_path: str) -> tuple[dict[str, dict], dict[str, str | None], set[str]]:
    path = Path(rest_spec_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read rest-pose spec {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Rest-pose spec must be a JSON object")
    raw_bones = payload.get("bones")
    raw_parents = payload.get("parents")
    raw_connected = payload.get("connected")
    if not isinstance(raw_bones, dict) or not raw_bones:
        raise ValueError("Rest-pose spec requires a non-empty 'bones' object")
    if not isinstance(raw_parents, dict):
        raise ValueError("Rest-pose spec requires a 'parents' object")
    if not isinstance(raw_connected, list) or not all(isinstance(name, str) for name in raw_connected):
        raise ValueError("Rest-pose spec requires a 'connected' string array")

    bones: dict[str, dict] = {}
    for name, raw_bone in raw_bones.items():
        if not isinstance(name, str) or not name or not isinstance(raw_bone, dict):
            raise ValueError("Every bone entry needs a non-empty string name and object value")
        head = _finite_vec3(raw_bone.get("head"), field="head", bone_name=name)
        tail = _finite_vec3(raw_bone.get("tail"), field="tail", bone_name=name)
        try:
            roll = float(raw_bone.get("roll_deg"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"roll_deg for bone '{name}' must be numeric") from exc
        if not math.isfinite(roll):
            raise ValueError(f"roll_deg for bone '{name}' must be finite")
        if math.dist(head, tail) <= 1e-12:
            raise ValueError(f"bone '{name}' has a zero-length rest segment")
        bones[name] = {"head": head, "tail": tail, "roll_deg": roll}

    if set(raw_parents) != set(bones):
        raise ValueError("The parents object must contain exactly the rest-pose bone names")
    parents: dict[str, str | None] = {}
    for name, parent in raw_parents.items():
        if parent is not None and (not isinstance(parent, str) or parent not in bones):
            raise ValueError(f"parent for bone '{name}' must be null or another rest-pose bone")
        if parent == name:
            raise ValueError(f"bone '{name}' cannot parent itself")
        parents[name] = parent

    connected = set(raw_connected)
    if not connected.issubset(bones):
        raise ValueError("Connected bone names must all occur in the rest-pose bone set")
    return bones, parents, connected


def _load_single_armature(source_path: str):
    source = Path(source_path).expanduser().resolve()
    if source.suffix.lower() != ".fbx":
        raise ValueError(f"Rest-pose rewriting currently requires an FBX source, got {source}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source))
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected exactly one armature in {source}, found {len(armatures)}")
    return source, armatures[0]


def _apply_rest(armature_obj, bones: dict[str, dict], parents: dict[str, str | None], connected: set[str]) -> None:
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature_obj.data.edit_bones

    missing = sorted(set(bones) - {bone.name for bone in edit_bones})
    extra = sorted({bone.name for bone in edit_bones} - set(bones))
    if missing or extra:
        raise RuntimeError(f"Carrier bone mismatch. missing={missing} unexpected={extra}")

    # Detach first: connected children otherwise move while a parent tail is
    # being rewritten.
    for bone in edit_bones:
        bone.use_connect = False
    for name, spec in bones.items():
        bone = edit_bones[name]
        bone.head = spec["head"]
        bone.tail = spec["tail"]
        bone.roll = math.radians(spec["roll_deg"])
    for name, parent in parents.items():
        edit_bones[name].parent = edit_bones[parent] if parent else None
    for name in bones:
        edit_bones[name].use_connect = name in connected
    bpy.ops.object.mode_set(mode="OBJECT")


def rewrite_armature_rest_pose(
    source_path: str,
    fbx_output: str,
    rest_spec_path: str,
    *,
    action_name: str | None = None,
) -> dict[str, object]:
    """Apply a JSON rest-pose specification and export the rewritten carrier."""
    bones, parents, connected = _read_rest_spec(rest_spec_path)
    source, armature = _load_single_armature(source_path)
    _apply_rest(armature, bones, parents, connected)

    if armature.animation_data and armature.animation_data.action and action_name:
        armature.animation_data.action.name = action_name

    destination = Path(fbx_output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(destination),
        use_selection=False,
        object_types={"ARMATURE", "MESH"},
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_space_transform=True,
        bake_anim_use_all_actions=True,
        bake_anim_force_startend_keying=True,
        bake_anim_simplify_factor=0.0,
    )
    return {
        "ok": True,
        "detail": "ready",
        "source": str(source),
        "fbx_output": str(destination),
        "bone_count": len(bones),
    }
