"""Blender-only scene operations used while curating third-party assets.

This worker module deliberately contains mechanics only: loading a source,
fixing linked texture paths, selecting an action, dropping control rig data,
and exporting FBX.  Catalog selection and destination policy remain in the
private maintenance scripts that call the public sidecar CLI.
"""

from __future__ import annotations

from pathlib import Path

import bpy  # type: ignore


HELPER_NAME_TOKENS = (
    "shape_",
    "custom bone",
    "bone shape",
    "ico_sphere",
    "icosphere",
    "icosahedron",
)

CONTROL_BONE_PREFIXES = (
    "DRV_",
    "DRV ",
    "CTRL_",
    "CTRL ",
    "POLE",
    "POLEARM",
    "IK_",
    "MCH_",
    "MCH-",
)


def _relocate_libraries(source_root: Path) -> None:
    """Reconnect Mesh2Motion linked libraries after opening a ``.blend`` file."""
    search_dirs = [
        source_root / "rigs",
        source_root / "rig-variations",
        source_root / "3d-models",
        source_root / "CC0-packs" / "Quaternius",
    ]
    for library in list(bpy.data.libraries):
        filename = Path(str(library.filepath).replace("\\", "/")).name
        for directory in search_dirs:
            candidate = directory / filename
            if candidate.exists():
                library.filepath = str(candidate)
                library.reload()
                break


def _load_curation_source(source_path: str, source_root: str) -> Path:
    """Load either a source FBX or a Mesh2Motion ``.blend`` scene."""
    source = Path(source_path).expanduser().resolve()
    root = Path(source_root).expanduser().resolve()
    if source.suffix.lower() == ".fbx":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(source))
        return source

    bpy.ops.wm.open_mainfile(filepath=str(source))
    _relocate_libraries(root)
    return source


def _short_action_name(name: str) -> str:
    return name.split("|")[-1]


def inspect_curation_source(source_path: str, source_root: str) -> dict[str, object]:
    """Return source action names without applying the runtime scene policy."""
    source = _load_curation_source(source_path, source_root)
    return {
        "ok": True,
        "detail": "ready",
        "source": str(source),
        "format": source.suffix.lower().lstrip("."),
        "actions": [
            {"name": action.name, "short_name": _short_action_name(action.name)}
            for action in bpy.data.actions
        ],
    }


def _resolve_texture_path(source_root: Path, source_blend: Path, raw_path: str) -> Path | None:
    normalized = str(raw_path or "").replace("\\", "/").strip()
    if not normalized:
        return None

    candidates: list[Path] = []
    if normalized.startswith("//"):
        relative = normalized[2:]
        candidates.append(source_blend.parent / relative)
        candidates.append(source_root / relative)
    else:
        path = Path(normalized)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(source_blend.parent / normalized)
            candidates.append(source_root / normalized)

    basename = Path(normalized).name
    if basename:
        candidates.extend(
            [
                source_blend.parent / basename,
                source_root / "3d-models" / basename,
                source_root / "3d-models" / "CC-SA" / basename,
                source_root / "rig-variations" / basename,
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    if basename:
        matches = sorted(path for path in source_root.rglob(basename) if path.is_file())
        if matches:
            return matches[0]
    return None


def _repair_image_paths(source_root: Path, source_blend: Path) -> list[str]:
    repaired: list[str] = []
    for image in bpy.data.images:
        if image.source != "FILE" or image.packed_file:
            continue
        resolved = _resolve_texture_path(
            source_root,
            source_blend,
            image.filepath or image.filepath_raw,
        )
        if resolved is None:
            continue
        image.filepath = str(resolved)
        image.filepath_raw = str(resolved)
        try:
            image.reload()
        except RuntimeError as exc:
            repaired.append(
                f"could not reload texture '{image.name}' from {resolved}: {exc}"
            )
    return repaired


def _object_is_visible(obj) -> bool:
    try:
        return bool(obj.visible_get())
    except Exception:
        return not bool(getattr(obj, "hide_viewport", False))


def _should_keep_object(obj) -> bool:
    if not _object_is_visible(obj):
        return False
    if obj.type == "ARMATURE":
        return True
    if obj.type != "MESH":
        return False
    vertices = getattr(getattr(obj, "data", None), "vertices", [])
    if len(vertices) <= 0:
        return False
    name = f"{obj.name} {getattr(obj.data, 'name', '')}".lower()
    return not any(token in name for token in HELPER_NAME_TOKENS)


def _is_control_bone_name(name: str) -> bool:
    return name.upper().startswith(CONTROL_BONE_PREFIXES)


def _mesh_uses_armature(mesh_obj, armature_obj) -> bool:
    if mesh_obj.parent is armature_obj:
        return True
    return any(
        modifier.type == "ARMATURE" and modifier.object is armature_obj
        for modifier in mesh_obj.modifiers
    )


def _weighted_bone_names(mesh_objects) -> set[str]:
    weighted: set[str] = set()
    for mesh_obj in mesh_objects:
        groups = mesh_obj.vertex_groups
        for vertex in mesh_obj.data.vertices:
            for group_ref in vertex.groups:
                if group_ref.weight <= 1e-8:
                    continue
                if 0 <= group_ref.group < len(groups):
                    weighted.add(groups[group_ref.group].name)
    return weighted


def _deform_skeleton_names(armature_obj, mesh_objects) -> set[str]:
    weighted = _weighted_bone_names(mesh_objects)
    if not weighted:
        return set()

    bones = list(armature_obj.data.bones)
    parent_of = {bone.name: (bone.parent.name if bone.parent else None) for bone in bones}
    children_of: dict[str | None, list[str]] = {}
    for name, parent in parent_of.items():
        children_of.setdefault(parent, []).append(name)

    keep = set(weighted)
    for name in list(weighted):
        cursor = parent_of.get(name)
        while cursor is not None:
            keep.add(cursor)
            cursor = parent_of.get(cursor)

    stack = list(weighted)
    while stack:
        name = stack.pop()
        for child in children_of.get(name, []):
            if child in keep or _is_control_bone_name(child):
                continue
            keep.add(child)
            stack.append(child)

    return {
        name
        for name in keep
        if name in parent_of and (name in weighted or not _is_control_bone_name(name))
    }


def _remove_action_curves_for_bones(removed_names: set[str]) -> int:
    if not removed_names:
        return 0
    removed_curves = 0
    markers = [f'pose.bones["{name}"]' for name in removed_names]
    for action in list(bpy.data.actions):
        for fcurve in list(action.fcurves):
            if any(marker in fcurve.data_path for marker in markers):
                action.fcurves.remove(fcurve)
                removed_curves += 1
    return removed_curves


def clean_control_bones_in_scene() -> dict[str, int]:
    """Remove unweighted control bones and their corresponding animation data."""
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    report = {
        "armatures": 0,
        "removed_bones": 0,
        "removed_fcurves": 0,
        "removed_vertex_groups": 0,
    }
    if not armatures or not meshes:
        return report

    previous_active = bpy.context.view_layer.objects.active
    previous_mode = previous_active.mode if previous_active is not None else "OBJECT"
    if previous_active is not None and previous_active.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    try:
        for armature_obj in armatures:
            attached_meshes = [
                mesh_obj for mesh_obj in meshes if _mesh_uses_armature(mesh_obj, armature_obj)
            ]
            if not attached_meshes and len(armatures) == 1:
                attached_meshes = meshes
            if not attached_meshes:
                continue

            keep_names = _deform_skeleton_names(armature_obj, attached_meshes)
            if not keep_names:
                continue
            existing_names = {bone.name for bone in armature_obj.data.bones}
            remove_names = existing_names - keep_names
            if not remove_names:
                continue

            report["armatures"] += 1
            report["removed_fcurves"] += _remove_action_curves_for_bones(remove_names)

            for mesh_obj in attached_meshes:
                for name in sorted(remove_names):
                    group = mesh_obj.vertex_groups.get(name)
                    if group is None:
                        continue
                    mesh_obj.vertex_groups.remove(group)
                    report["removed_vertex_groups"] += 1

            bpy.ops.object.select_all(action="DESELECT")
            armature_obj.select_set(True)
            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode="EDIT")
            edit_bones = armature_obj.data.edit_bones

            def nearest_kept_parent(edit_bone):
                parent = edit_bone.parent
                while parent is not None and parent.name not in keep_names:
                    parent = parent.parent
                return parent

            for bone in list(edit_bones):
                if bone.name not in keep_names:
                    continue
                if bone.parent is not None and bone.parent.name not in keep_names:
                    bone.parent = nearest_kept_parent(bone)
                    bone.use_connect = False

            def depth(edit_bone) -> int:
                value = 0
                parent = edit_bone.parent
                while parent is not None:
                    value += 1
                    parent = parent.parent
                return value

            for bone in sorted(
                [bone for bone in edit_bones if bone.name in remove_names],
                key=depth,
                reverse=True,
            ):
                report["removed_bones"] += 1
                edit_bones.remove(bone)
            bpy.ops.object.mode_set(mode="OBJECT")
    finally:
        active = bpy.context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if previous_active is not None:
            bpy.ops.object.select_all(action="DESELECT")
            previous_active.select_set(True)
            bpy.context.view_layer.objects.active = previous_active
            if previous_mode != "OBJECT":
                bpy.ops.object.mode_set(mode=previous_mode)

    return report


def _select_action(action_name: str | None) -> tuple[object | None, list[str]]:
    action = bpy.data.actions.get(action_name) if action_name else None
    notes: list[str] = []
    if action_name and action is None:
        action = next(
            (
                candidate
                for candidate in bpy.data.actions
                if _short_action_name(candidate.name) == action_name
            ),
            None,
        )
    if action_name and action is None:
        if len(bpy.data.actions) == 1:
            action = next(iter(bpy.data.actions))
            notes.append(
                f"action '{action_name}' not found; using sole action '{action.name}'"
            )
        else:
            available = ", ".join(action.name for action in bpy.data.actions)
            raise RuntimeError(f"Action '{action_name}' not found: {available}")
    if action is None and bpy.data.actions:
        action = next(iter(bpy.data.actions))
    return action, notes


def _bind_action_to_armatures(action) -> None:
    for obj in list(bpy.data.objects):
        keep = _should_keep_object(obj)
        if keep and obj.type == "ARMATURE" and action is not None:
            obj.animation_data_create()
            obj.animation_data.action = action
            # Blender 4.4+/5.0 actions can be slotted.  Assigning the action
            # alone leaves the pose at rest and exports a static clip.
            slots = getattr(action, "slots", None)
            if slots:
                try:
                    obj.animation_data.action_slot = slots[0]
                except Exception:
                    pass
        if not keep:
            bpy.data.objects.remove(obj, do_unlink=True)


def export_curation_fbx(
    source_path: str,
    source_root: str,
    fbx_output: str,
    *,
    action_name: str | None = None,
    embed_textures: bool = False,
    clean_control_bones: bool = False,
) -> dict[str, object]:
    """Load, trim, and export an offline animation asset as FBX."""
    source = _load_curation_source(source_path, source_root)
    root = Path(source_root).expanduser().resolve()
    destination = Path(fbx_output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    if embed_textures:
        warnings.extend(_repair_image_paths(root, source))

    action, notes = _select_action(action_name)
    _bind_action_to_armatures(action)

    clean_report: dict[str, int] | None = None
    if clean_control_bones:
        clean_report = clean_control_bones_in_scene()

    if action is not None:
        frame_start, frame_end = action.frame_range
        bpy.context.scene.frame_start = int(frame_start)
        bpy.context.scene.frame_end = int(frame_end)

    if not any(obj.type == "ARMATURE" for obj in bpy.data.objects):
        raise RuntimeError(f"No exportable armature found in {source}")
    if not any(obj.type == "MESH" for obj in bpy.data.objects):
        raise RuntimeError(f"No exportable mesh found in {source}")

    bpy.ops.export_scene.fbx(
        filepath=str(destination),
        use_selection=False,
        object_types={"ARMATURE", "MESH"},
        bake_anim=action is not None,
        bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False,
        bake_anim_force_startend_keying=True,
        add_leaf_bones=False,
        path_mode="COPY",
        embed_textures=embed_textures,
    )
    return {
        "ok": True,
        "detail": "ready",
        "source": str(source),
        "fbx_output": str(destination),
        "action": action.name if action is not None else None,
        "clean_control_bones": clean_report,
        "notes": notes,
        "warnings": warnings,
    }
