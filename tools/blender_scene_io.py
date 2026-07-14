"""Blender worker for scene inspection and normalization."""

from __future__ import annotations

# ruff: noqa: I001

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    # bpy must be imported before bmesh/mathutils in the managed bpy runtime.
    import bpy
    import mathutils
    import bmesh
except ImportError:
    bpy = None
    mathutils = None

    bmesh = None

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from blender_io.math_utils import *
from blender_io.bone_utils import *
from blender_io.pose_transfer import *

from blender_worker_args import parse_worker_args  # noqa: E402
from blender_view import (  # noqa: E402
    VIEW_ALIASES,
    VIEW_PRESET_NAMES,
    VIEW_PRESETS,
    WORLD_UP,
    WORLD_X,
    WORLD_Y,
    _apply_roll_to_view_config,
    _axis_angle_rotation,
    _build_explicit_view_config,
    _build_view_config_from_direction,
    _finalize_view_config,
    _normalize_vector,
    _project_projection_space_direction,
    _resolve_view_preset_name,
    _transform_point_to_projection_space,
    compute_projection_frame,
    get_scene_view_config,
    get_view_config,
    point_depth,
    project_point_ortho,
    project_points_to_uv,
)
from blender_orientation import (  # noqa: E402
    normalize_model_orientation as _orientation_normalize_model_orientation,
    posed_rig_forward_world as _orientation_posed_rig_forward_world,
    rig_forward_world as _orientation_rig_forward_world,
    structural_root_bone as _orientation_structural_root_bone,
)
from blender_json_io import (  # noqa: E402
    matrix3_from_json as _matrix3_from_json,
    matrix3_to_json as _matrix3_to_json,
    matrix4_to_json as _matrix4_to_json,
    quat_to_stable_json as _quat_to_stable_json,
    vector_from_json as _vector_from_json,
    vector_to_json as _vector_to_json,
    view_config_to_json as _view_config_to_json,
    weights_to_json as _weights_to_json,
)

# ============================================================================
# Constants
# ============================================================================

SEGMENT_EPSILON = 1e-8
TERMINAL_CHAIN_ROOT_RATIO = 0.6
TERMINAL_CHAIN_MAX_LENGTH_RATIO = 0.8
TERMINAL_CHAIN_PARENT_RATIO = 1.5
TERMINAL_CHAIN_MAX_SPAN = 6
VECTOR_EPSILON = 1e-8


def _set_scene_armatures_rest_pose(scene):
    """Temporarily force armatures to rest pose for bind/setup extraction."""
    previous = []
    for scene_obj in scene.objects:
        if scene_obj.type != "ARMATURE" or scene_obj.data is None:
            continue
        previous.append((scene_obj.data, scene_obj.data.pose_position))
        scene_obj.data.pose_position = "REST"
    if previous:
        bpy.context.view_layer.update()
    return previous


def _restore_scene_armature_pose_positions(previous):
    for armature_data, pose_position in previous:
        armature_data.pose_position = pose_position
    if previous:
        bpy.context.view_layer.update()


def _apply_auto_setup_pose(
    armature_obj, source_frame=None, use_rest_pose=False
) -> dict[str, object]:
    # The A-pose path is intentionally inactive. We rely on the scene already
    # being at the chosen render frame (via _select_sprite_render_frame /
    # _resolve_setup_frame) and don't override pose bones here.
    return {"mode": "rest_pose" if use_rest_pose else "frame", "posed_bone_count": 0}






def _local_bind_pose_signature(armature_obj, bone_names=None) -> dict[str, dict[str, object]]:
    if armature_obj is None:
        return {}
    include = set(bone_names or [])
    if not include:
        include = {bone.name for bone in armature_obj.data.bones}
    result = {}
    for bone in armature_obj.data.bones:
        if bone.name not in include:
            continue
        result[bone.name] = {
            "parent": bone.parent.name if bone.parent is not None else None,
            "local_rotation": _quat_to_stable_json(_rest_local_quat(bone)),
            "length": float(bone.length),
        }
    return result




























def _maybe_borrow_bind_from_animation(
    armature_obj,
    bind_from_animation,
    *,
    source_frame=None,
    use_rest_pose=False,
):
    """Pre-load an external animation and bind it to `armature_obj` so its
    first frame becomes the implicit bind/setup pose.

    Skipped silently only when:
      * `bind_from_animation` is empty / None
      * the file doesn't exist
      * `use_rest_pose=True` (caller asked for the rig's rest pose verbatim)
      * no armature was found

    Otherwise the borrow ALWAYS happens — we no longer try to detect
    whether the model's own action is "substantive" enough to be
    trusted as the bind source. Empirically that distinction is
    unreliable across FBX exporters (Mixamo's "Without Animation"
    download still ships a multi-frame T-pose action). Forcing the
    optimization animation's first frame as bind gives a consistent,
    predictable starting pose across all source models.

    Note: a positive `source_frame` is NOT a reason to skip the borrow.
    The C++ pipeline always passes a frame (defaults to 1) — that's not
    an explicit user override, it's just the frame index within whatever
    action is active. After the borrow, the borrowed action becomes the
    active one and the frame is interpreted against it.

    Otherwise: imports the file, finds the first imported armature's
    action, transfers that action to the target armature, snaps the
    scene to the action's first frame. Returns a small dict describing
    what was applied (or why it was skipped) so the caller can log it
    in the scene payload for diagnostics.
    """
    if not bind_from_animation:
        return {"applied": False, "reason": "no_path"}
    if use_rest_pose:
        return {"applied": False, "reason": "use_rest_pose"}
    if armature_obj is None:
        return {"applied": False, "reason": "no_armature"}

    # Always override the model's own action with the bind donor when
    # the caller has provided one. Detecting whether a model's
    # auto-generated action is "real" or "T-pose dummy" turns out to
    # be unreliable across FBX exporters; the caller's contract is that
    # bind_from_animation is the canonical setup-pose donor. Animations
    # the model brings remain available in `bpy.data.actions` so the
    # animation-extraction pass can still pick them up by name.

    bind_path = Path(str(bind_from_animation)).expanduser()
    if not bind_path.exists():
        return {"applied": False, "reason": f"missing_file:{bind_path}"}

    actions_before = {action.name for action in bpy.data.actions}
    objects_before = {obj.name for obj in bpy.data.objects}
    armatures_before = {obj.name for obj in bpy.data.objects if obj.type == "ARMATURE"}
    try:
        import_model(str(bind_path))
    except Exception as exc:
        return {"applied": False, "reason": f"import_failed:{exc}"}

    # Find the armature(s) the import added; we read pose rotations
    # from there, then delete it.
    imported_armatures = [
        obj
        for obj in bpy.data.objects
        if obj.type == "ARMATURE" and obj.name not in armatures_before
    ]
    if not imported_armatures:
        _purge_imported_objects(objects_before)
        return {"applied": False, "reason": "no_armature_in_animation_file"}
    imported_arm = imported_armatures[0]

    # Find an action to determine the first frame; fall back to the
    # scene's frame_start.
    new_actions = [a for a in bpy.data.actions if a.name not in actions_before]
    candidate_actions = [a for a in new_actions if is_pose_action(a)]
    if not candidate_actions:
        candidate_actions = [a for a in bpy.data.actions if is_pose_action(a)]
    action = candidate_actions[0] if candidate_actions else None
    try:
        start_frame = (
            int(round(float(action.frame_range[0])))
            if action is not None
            else int(bpy.context.scene.frame_start or 1)
        )
    except Exception:
        start_frame = int(bpy.context.scene.frame_start or 1)

    # Bind the action to the imported armature so its pose evaluates
    # correctly when we frame_set the scene.
    if action is not None:
        if imported_arm.animation_data is None:
            imported_arm.animation_data_create()
        imported_arm.animation_data.action = action

    bpy.context.scene.frame_set(start_frame)
    bpy.context.view_layer.update()

    # Copy the first-frame pose from imported bones to target bones with the
    # SAME routine the per-frame animation transfer uses
    # (`_extract_transferred_animation` -> `_copy_source_pose_to_target`).
    #
    # The bind/setup pose MUST be identical to the transferred animation
    # evaluated at the setup frame: the exported 2D rig composes
    # world = setup ∘ delta, so any divergence between the two copy semantics
    # becomes a constant per-bone orientation error across the WHOLE clip.
    # A previous version rest-compensated each bone here
    # (matrix_basis_tgt = rest_local_tgt^-1 @ rest_local_src @ matrix_basis_src,
    # see doc/WALKING_APOSE_RETARGET_INVESTIGATION.md #3/#4) while the
    # animation path copied raw local channels; on rigs whose rest differs
    # from the donor's (Mixamo auto-rigged piggy) that mismatch put the
    # legs/feet 90-180deg off through every animation frame (mirrored limbs).
    # Raw copy is also Mixamo's native playback semantic and preserves the
    # target rig's own rest stylization instead of forcing donor anatomy.
    bone_map = _stem_pose_bone_map(imported_arm, armature_obj)
    posed_bones = _copy_source_pose_to_target(imported_arm, armature_obj, bone_map)

    # CRITICAL: leave the target with NO active action so the world-rotation
    # pose just written survives the caller's next `scene.frame_set(...)`.
    #
    # `_copy_source_pose_to_target` writes each bone's world rotation into the
    # pose-bone basis. If we instead assigned the donor action to the target
    # (as a previous version did, "to keep the depsgraph connected"), the
    # caller's `frame_set(setup_frame)` re-evaluated that action and RE-POSED
    # the rig with the donor's RAW local channels — silently discarding the
    # world-rotation copy. The exported 2D rig is built from this bind pose,
    # while the animation path re-applies the world-rotation copy per frame,
    # so the bind ended up ~40° off the animation's frame 0 on roll-divergent
    # rigs (auto-rigged piggy): bind != frame0 -> knees/elbows bend the wrong
    # way. Clearing the action makes the bind, setup and per-frame poses all
    # share the one world-rotation convention. The model's own action (if any)
    # is also cleared here so it can't drive the bind either; the animation
    # extraction re-imports its source separately and never relies on an action
    # being live on the target.
    if armature_obj.animation_data is not None:
        armature_obj.animation_data.action = None
        for track in getattr(armature_obj.animation_data, "nla_tracks", []) or []:
            track.mute = True
    if action is not None:
        action.use_fake_user = True

    # Clean up the imported armature/mesh — we only needed its pose, and
    # leaving extra armatures around causes ambiguity for
    # `find_mesh_and_armature` in callers further down the pipeline.
    _purge_imported_objects(objects_before)
    bpy.context.view_layer.update()

    return {
        "applied": True,
        "action_name": action.name if action is not None else None,
        "frame": start_frame,
        "source": str(bind_path),
        "posed_bones": posed_bones,
    }


def _purge_imported_objects(names_before: set) -> None:
    """Delete every object added since `names_before` was snapshotted.

    Used to drop the throwaway armature/mesh that `_maybe_borrow_bind_from_animation`
    imports — we only wanted the action, not the auxiliary objects.
    Actions stay (they live in bpy.data.actions, not bpy.data.objects)
    so the borrowed pose remains evaluable on the target armature.
    """
    to_remove = [obj for obj in bpy.data.objects if obj.name not in names_before]
    for obj in to_remove:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            continue








def extract_2d_mesh(
    mesh_obj,
    view_cfg,
    projection_frame=None,
    source_frame=None,
    projection_inverse=None,
    use_rest_pose=False,
):
    """Extract the bind-pose mesh projected to 2D."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    if source_frame is None:
        source_frame = scene.frame_start
    scene.frame_set(source_frame)
    rest_pose_state = _set_scene_armatures_rest_pose(scene) if use_rest_pose else []
    depsgraph.update()

    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = None
    try:
        eval_mesh = eval_obj.to_mesh()
        world_mat = eval_obj.matrix_world

        # Extract vertices in world space
        vertices_3d = np.empty((len(eval_mesh.vertices), 3), dtype=np.float64)
        for index, vert in enumerate(eval_mesh.vertices):
            world_co = world_mat @ vert.co
            vertices_3d[index] = (world_co.x, world_co.y, world_co.z)

        # Project to 2D
        vertices_2d_list = []
        for i in range(len(vertices_3d)):
            pt = vertices_3d[i]
            projected = _transform_point_to_projection_space(
                pt, projection_inverse=projection_inverse
            )
            basis_2d = np.asarray(view_cfg["basis_2d"], dtype=np.float64)
            proj_2d = basis_2d @ projected
            vertices_2d_list.append([proj_2d[0], proj_2d[1]])
        vertices_2d = np.array(vertices_2d_list, dtype=np.float64)

        if projection_frame is None:
            projection_frame = compute_projection_frame(vertices_2d)
        fallback_uvs = project_points_to_uv(vertices_2d, projection_frame)

        # Compute depths
        depths = np.array(
            [
                point_depth(vertices_3d[i], view_cfg, projection_inverse=projection_inverse)
                for i in range(len(vertices_3d))
            ],
            dtype=np.float64,
        )

        bm = bmesh.new()
        try:
            bm.from_mesh(eval_mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            # Backface visibility is decided in world space: transform the
            # bmesh (it comes out of from_mesh() in object-local space) and
            # recompute normals so face normals and the world-space view
            # direction share one coordinate frame. Without this a rotated or
            # non-uniformly scaled object would test backfaces along the wrong
            # axis.
            bm.transform(world_mat)
            bm.normal_update()
            bm.faces.ensure_lookup_table()
            uv_layer = bm.loops.layers.uv.active

            view_dir_vec = mathutils.Vector(view_cfg["view_dir"]).normalized()

            output_vertices_2d = []
            output_vertices_3d = []
            output_uvs = []
            output_depths = []
            source_vertex_indices = []
            triangles = []
            triangle_keys = []
            triangle_visible = []
            vertex_remap = {}

            for face in bm.faces:
                if len(face.loops) != 3:
                    continue

                # Record BACKFACE visibility per triangle, never dropping
                # geometry: the full mesh must survive so the 3D preview renders
                # intact. Backface is geometry-intrinsic (independent of any
                # sprite split), so deciding it globally is correct, and it
                # removes the back-side surfaces that used to contaminate the
                # 2D sprites. OCCLUSION is intentionally NOT tested here: a
                # global occlusion cull would delete a limb the moment the torso
                # hides it in the donor pose, mutilating that limb's own sprite.
                # Self-occlusion is resolved per sprite downstream, where the
                # part split is known.
                face_visible = face.normal.dot(view_dir_vec) < -1e-5

                tri = []
                source_tri = []
                for loop in face.loops:
                    source_index = int(loop.vert.index)
                    if source_index < 0 or source_index >= len(vertices_2d):
                        continue
                    source_tri.append(source_index)
                    if uv_layer is not None:
                        loop_uv = loop[uv_layer].uv
                        source_uv = (float(loop_uv.x), float(loop_uv.y))
                    else:
                        source_uv = (
                            float(fallback_uvs[source_index][0]),
                            float(fallback_uvs[source_index][1]),
                        )
                    remap_key = (
                        source_index,
                        round(source_uv[0], 8),
                        round(source_uv[1], 8),
                    )
                    output_index = vertex_remap.get(remap_key)
                    if output_index is None:
                        output_index = len(output_vertices_2d)
                        vertex_remap[remap_key] = output_index
                        output_vertices_2d.append(vertices_2d[source_index].tolist())
                        output_vertices_3d.append(vertices_3d[source_index].tolist())
                        output_uvs.append([source_uv[0], source_uv[1]])
                        output_depths.append(float(depths[source_index]))
                        source_vertex_indices.append(source_index)
                    tri.append(output_index)
                if len(tri) != 3 or len(set(tri)) != 3:
                    continue
                triangles.append(tri)
                triangle_keys.append(tuple(sorted(source_tri)))
                triangle_visible.append(bool(face_visible))
        finally:
            bm.free()

        # Drop near-degenerate sliver triangles. A face that projects edge-on
        # (it spans front-to-back across a silhouette) collapses to a near-zero
        # area spike in 2D: no visible coverage, but it renders as the long thin
        # shards seen on the neck/torso. Shape quality = 4*sqrt(3)*area/sum(edge^2)
        # is 1 for equilateral and ~0 for a sliver; a healthy decimated triangle
        # sits well above the threshold, so this only removes the artifacts.
        min_quality = 0.01
        kept_triangles = []
        kept_keys = []
        kept_visible = []
        for tri, key, vis in zip(triangles, triangle_keys, triangle_visible):
            ax, ay = output_vertices_2d[tri[0]]
            bx, by = output_vertices_2d[tri[1]]
            cx, cy = output_vertices_2d[tri[2]]
            double_area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
            sum_sq_edges = (
                (ax - bx) ** 2 + (ay - by) ** 2
                + (bx - cx) ** 2 + (by - cy) ** 2
                + (cx - ax) ** 2 + (cy - ay) ** 2
            )
            quality = (
                (2.0 * math.sqrt(3.0) * double_area) / sum_sq_edges
                if sum_sq_edges > VECTOR_EPSILON
                else 0.0
            )
            if quality >= min_quality:
                kept_triangles.append(tri)
                kept_keys.append(key)
                kept_visible.append(vis)
        triangles = kept_triangles
        triangle_keys = kept_keys
        triangle_visible = kept_visible

        # Extract vertex groups (weights)
        vertex_groups = {}
        group_names = {group.index: group.name for group in mesh_obj.vertex_groups}

        for vert in mesh_obj.data.vertices:
            for group in vert.groups:
                group_name = group_names.get(group.group, f"group_{group.group}")
                vertex_groups.setdefault(group_name, []).append((vert.index, group.weight))

        return {
            "vertices_2d": output_vertices_2d,
            "vertices_3d": output_vertices_3d,
            "triangles": triangles,
            "triangle_keys": triangle_keys,
            "triangle_visible": triangle_visible,
            "uvs": output_uvs,
            "depths": output_depths,
            "vertex_groups": vertex_groups,
            "source_vertex_indices": source_vertex_indices,
            "projection_frame": projection_frame,
        }
    finally:
        if eval_mesh is not None:
            eval_obj.to_mesh_clear()
        _restore_scene_armature_pose_positions(rest_pose_state)


def parse_args():
    return parse_worker_args(VIEW_PRESETS.keys())


def _apply_fbx_light_import_fix() -> None:
    """Work around a bug in bpy 5.0.x's bundled io_scene_fbx importer.

    `blen_read_light` still assigns ``lamp.cycles.cast_shadow`` — an attribute
    removed from CyclesLightSettings in Blender 5.0 — so importing ANY FBX that
    contains a light (most Mixamo/DCC exports do) raises AttributeError and aborts
    the entire import. The importer exposes no option to skip lights/cameras, and
    bpy 5.0.1 (the latest on PyPI) still ships the bug. We wrap ``blen_read_light``
    so that this one failure is swallowed and the lamp that was already created and
    configured (only the trailing cast_shadow line fails) is returned, letting the
    import finish. The light is harmless downstream — only mesh + armature matter.
    Idempotent.
    """
    try:
        from io_scene_fbx import import_fbx
    except Exception:
        return
    if getattr(import_fbx, "_flatrig_light_fix", False):
        return
    _orig = import_fbx.blen_read_light

    def _safe(fbx_tmpl, fbx_obj, settings):
        before = set(bpy.data.lights.keys())
        try:
            return _orig(fbx_tmpl, fbx_obj, settings)
        except AttributeError as exc:
            if "cast_shadow" not in str(exc):
                raise
            created = [bpy.data.lights[k] for k in bpy.data.lights.keys() if k not in before]
            return created[-1] if created else None

    import_fbx.blen_read_light = _safe
    import_fbx._flatrig_light_fix = True


# Canonical world facing for imported rigs. Projection presets use fixed world
# axes, so imported rigs need one stable facing convention. The orientation
# detector below is intentionally topology-based: it uses the structural root
# bone and its children, never anatomical names or humanoid left/right joints.
NORMALIZE_TARGET_FORWARD = (0.0, -1.0, 0.0)

_OBJECT_TRANSFORM_FCURVE_PREFIXES = (
    "location",
    "rotation_euler",
    "rotation_quaternion",
    "rotation_axis_angle",
    "scale",
    "delta_location",
    "delta_rotation_euler",
    "delta_rotation_quaternion",
    "delta_scale",
)


def _strip_object_transform_animation(objects) -> int:
    """Remove object-level transform fcurves from imported armature actions.

    Some exporters (e.g. AI-generated models auto-rigged through Mixamo) key
    the armature OBJECT itself (location/rotation_euler/scale) alongside the
    pose-bone channels. Those keys re-evaluate on every depsgraph update and
    silently revert the object-level facing rotation applied by
    normalize_model_orientation — the model then faces sideways in previews
    and renders its back after generate. Character motion in this pipeline
    lives in pose bones (root motion included), so object-level transform
    keys on an armature are always unwanted. Returns the number of removed
    fcurves.
    """

    def _is_object_transform(fcurve):
        return fcurve.data_path.startswith(_OBJECT_TRANSFORM_FCURVE_PREFIXES)

    removed = 0
    for obj in objects:
        if obj.type != "ARMATURE":
            continue
        anim = obj.animation_data
        if anim is None or anim.action is None:
            continue
        action = anim.action
        layers = getattr(action, "layers", None)
        if layers:
            # bpy >= 4.4 layered actions: fcurves live in per-slot channelbags.
            for layer in layers:
                for strip in layer.strips:
                    for slot in action.slots:
                        bag = strip.channelbag(slot)
                        if bag is None:
                            continue
                        for fcurve in [fc for fc in bag.fcurves if _is_object_transform(fc)]:
                            bag.fcurves.remove(fcurve)
                            removed += 1
        else:
            for fcurve in [fc for fc in action.fcurves if _is_object_transform(fc)]:
                action.fcurves.remove(fcurve)
                removed += 1
    if removed:
        print(
            "[blender_scene_io] stripped "
            f"{removed} object-level transform fcurves from imported armature action(s)"
        )
    return removed




# --- FlatRig HML22 rig reduction (input-side) -------------------------------
# Collapse a BIPED HUMANOID rig to the FlatRig 22-bone canonical skeleton in
# place on the user's mesh. The topology mirrors HumanML3D/T2M, but the internal
# names are owned by FlatRig: core, spine.1/2/3, collar/shoulder/elbow/wrist,
# hip/knee/ankle/foot, neck, head. Unknown/non-humanoid rigs are exported
# untouched so the caller can use the generic/GMR path.






def _reduce_armature_to_canonical(armature_obj, meshes):
    data = armature_obj.data
    canon = {b.name: _canonical_target_for_bone_name(b.name) for b in data.bones}
    kept_original_by_canon = {c: name for name, c in canon.items() if c}

    survivor = {}
    dropped = []
    for bone in data.bones:
        if canon.get(bone.name):
            continue
        dropped.append(bone.name)
        parent = bone.parent
        surv = None
        while parent is not None:
            if canon.get(parent.name):
                surv = canon[parent.name]
                break
            parent = parent.parent
        survivor[bone.name] = surv

    # Merge dropped bones' skin weights into the survivor's vertex group, then drop
    # the empty groups and rename the kept groups to canonical.
    for mesh in meshes:
        vgs = mesh.vertex_groups
        for dropped_name in dropped:
            surv_canon = survivor.get(dropped_name)
            surv_original = kept_original_by_canon.get(surv_canon) if surv_canon else None
            src = vgs.get(dropped_name)
            dst = vgs.get(surv_original) if surv_original else None
            if src is None or dst is None:
                continue
            src_index = src.index
            for vert in mesh.data.vertices:
                for ge in vert.groups:
                    if ge.group == src_index and ge.weight > 0.0:
                        dst.add([vert.index], ge.weight, "ADD")
        for dropped_name in dropped:
            g = vgs.get(dropped_name)
            if g is not None:
                vgs.remove(g)
        for original, canon_name in canon.items():
            if canon_name:
                g = vgs.get(original)
                if g is not None and g.name != canon_name:
                    g.name = canon_name

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = data.edit_bones
    nearest_kept_original = {}
    for name, canon_name in canon.items():
        if not canon_name:
            continue
        anc = ebs[name].parent
        while anc is not None and not canon.get(anc.name):
            anc = anc.parent
        nearest_kept_original[name] = anc.name if anc is not None else None
    for name, parent_name in nearest_kept_original.items():
        # Disconnect FIRST so a connected bone does not snap onto the new parent's
        # tail (that yanks shoulders onto the chest -- "shoulders out of the neck").
        ebs[name].use_connect = False
        ebs[name].parent = ebs.get(parent_name) if parent_name else None
    for dropped_name in dropped:
        eb = ebs.get(dropped_name)
        if eb is not None:
            ebs.remove(eb)
    for name, canon_name in list(canon.items()):
        if canon_name and name in ebs and canon_name not in ebs:
            ebs[name].name = canon_name
    bpy.ops.object.mode_set(mode="OBJECT")
    return {"kept_count": len([c for c in canon.values() if c]), "dropped_count": len(dropped)}


def reduce_rig_to_canonical_cli(source_path, output_path, flat_output):
    """Reduce a humanoid rig to canonical and export to ``flat_output`` (.fbx).

    Non-humanoid rigs are exported unchanged with ``reduced=False`` so the caller
    can fall back to the existing cross-rig retarget path.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)
    mesh_obj, armature_obj = find_mesh_and_armature()
    if armature_obj is None:
        return {"ok": False, "detail": "No armature found in source."}
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and _mesh_uses_armature(o, armature_obj)]
    bone_names = [b.name for b in armature_obj.data.bones]
    before = len(armature_obj.data.bones)
    report = {
        "ok": True,
        "source": source_path,
        "canonical_format": "flatrig_hml22",
        "reduced": False,
        "bones_before": before,
    }
    if is_humanoid_biped(bone_names):
        report.update(_reduce_armature_to_canonical(armature_obj, meshes))
        report["reduced"] = True
        report["bones_after"] = len(armature_obj.data.bones)
        unweighted = 0
        for mesh in meshes:
            for v in mesh.data.vertices:
                if not any(g.weight > 0.0 for g in v.groups):
                    unweighted += 1
        report["unweighted_vertices"] = unweighted
    else:
        report["detail"] = "not_humanoid_biped"
    flat_path = Path(flat_output).expanduser().resolve()
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(flat_path), use_selection=False, object_types={"ARMATURE", "MESH"},
        add_leaf_bones=False, bake_anim=False,
        # Carry the source textures into the reduced FBX (the source packs them);
        # a plain export drops the packed image data and the build renders magenta.
        # The downstream extract-scene still writes a separate PNG -- nothing inline.
        path_mode="COPY", embed_textures=True, bake_space_transform=True,
    )
    report["flat_output"] = str(flat_path)
    return report


def import_model(filepath: str) -> None:
    extension = Path(filepath).suffix.lower()
    before = set(bpy.context.scene.objects)
    if extension == ".fbx":
        _apply_fbx_light_import_fix()
        bpy.ops.import_scene.fbx(filepath=filepath, use_custom_props=False)
    elif extension in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        raise ValueError(f"Unsupported format: {extension}. Use .fbx, .glb, or .gltf.")
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    _strip_object_transform_animation(imported)
    # (Removed) Mixamo is no longer canonicalized to mannequin; it maps directly to canonical later.
    normalize_model_orientation(imported)
    sanitize_imported_armature_terminal_geometry(imported)








def normalize_model_orientation(objects=None, target_forward=NORMALIZE_TARGET_FORWARD) -> float:
    return _orientation_normalize_model_orientation(
        objects,
        target_forward=target_forward,
        deform_bone_names_fn=_deform_bone_names,
    )


def _terminal_bone_length_limit(mesh_diagonal, bone_lengths):
    """Return a conservative upper bound for a bone segment inside the mesh."""
    positive_lengths = [
        float(length) for length in bone_lengths if float(length) > SEGMENT_EPSILON
    ]
    median_length = float(np.median(positive_lengths)) if positive_lengths else 0.0
    return max(float(mesh_diagonal) * 2.0, median_length * 20.0, SEGMENT_EPSILON)


def _mesh_world_bounds(mesh_objects):
    points = [
        mesh_obj.matrix_world @ mathutils.Vector(corner)
        for mesh_obj in mesh_objects
        for corner in mesh_obj.bound_box
    ]
    if not points:
        return None
    minimum = mathutils.Vector(
        tuple(min(float(point[axis]) for point in points) for axis in range(3))
    )
    maximum = mathutils.Vector(
        tuple(max(float(point[axis]) for point in points) for axis in range(3))
    )
    return minimum, maximum




# Per-process cache: the deform-bone classification scans every skin weight, so
# we compute it once per armature (each CLI call is its own process / one rig).




def _sanitize_armature_terminal_geometry(armature_obj, mesh_objects):
    """Remove or shorten corrupt terminal FBX bones without semantic names.

    Some FBX files encode terminal tail offsets before applying their unit
    scale. Blender then imports a one-unit character with leaf segments tens of
    units long. Bone tails do not affect skinning, but those lengths poison the
    projected hierarchy, retarget scale and optimizer geometry.
    """
    bounds = _mesh_world_bounds(mesh_objects)
    if bounds is None:
        return {"removed": [], "shortened": []}
    minimum, maximum = bounds
    mesh_diagonal = float((maximum - minimum).length)
    if mesh_diagonal <= SEGMENT_EPSILON:
        return {"removed": [], "shortened": []}

    weighted_names = _weighted_bone_names(mesh_objects)
    world_matrix = armature_obj.matrix_world.copy()
    previous_active = bpy.context.view_layer.objects.active
    previous_mode = armature_obj.mode
    removed = []
    shortened = []

    if previous_active is not None and previous_active.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        edit_bones = armature_obj.data.edit_bones

        def world_length(edit_bone):
            return float(
                (
                    world_matrix @ edit_bone.tail
                    - world_matrix @ edit_bone.head
                ).length
            )

        lengths = [world_length(bone) for bone in edit_bones]
        length_limit = _terminal_bone_length_limit(mesh_diagonal, lengths)
        center = (minimum + maximum) * 0.5
        head_distance_limit = mesh_diagonal * 2.0

        while True:
            removable = []
            for bone in edit_bones:
                if bone.children or bone.name in weighted_names:
                    continue
                head_world = world_matrix @ bone.head
                if (
                    world_length(bone) > length_limit
                    or float((head_world - center).length) > head_distance_limit
                ):
                    removable.append(bone)
            if not removable:
                break
            for bone in removable:
                removed.append(bone.name)
                edit_bones.remove(bone)

        remaining_lengths = [world_length(bone) for bone in edit_bones]
        typical_length = float(np.median(remaining_lengths)) if remaining_lengths else 0.0
        for bone in edit_bones:
            if bone.children or bone.name not in weighted_names:
                continue
            current_world_length = world_length(bone)
            if current_world_length <= length_limit or bone.length <= SEGMENT_EPSILON:
                continue
            parent_world_length = (
                world_length(bone.parent) if bone.parent is not None else typical_length
            )
            target_world_length = max(
                min(parent_world_length, length_limit),
                min(typical_length, length_limit),
                SEGMENT_EPSILON,
            )
            local_to_world_scale = current_world_length / float(bone.length)
            bone.length = target_world_length / max(
                local_to_world_scale,
                SEGMENT_EPSILON,
            )
            shortened.append(bone.name)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        armature_obj.select_set(False)
        if previous_active is not None:
            previous_active.select_set(True)
            bpy.context.view_layer.objects.active = previous_active
            if previous_mode != "OBJECT":
                bpy.ops.object.mode_set(mode=previous_mode)

    return {"removed": removed, "shortened": shortened}


def sanitize_imported_armature_terminal_geometry(objects=None):
    """Sanitize terminal geometry for every imported armature and its meshes."""
    imported = list(objects) if objects is not None else list(bpy.context.scene.objects)
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    reports = []
    for armature_obj in (obj for obj in imported if obj.type == "ARMATURE"):
        attached_meshes = [
            mesh_obj
            for mesh_obj in mesh_objects
            if _mesh_uses_armature(mesh_obj, armature_obj)
        ]
        if not attached_meshes and len(
            [obj for obj in imported if obj.type == "ARMATURE"]
        ) == 1:
            attached_meshes = mesh_objects
        if not attached_meshes:
            continue
        report = _sanitize_armature_terminal_geometry(
            armature_obj,
            attached_meshes,
        )
        if report["removed"] or report["shortened"]:
            print(
                "[flatrig] Sanitized terminal FBX bones: "
                f"removed={len(report['removed'])} "
                f"shortened={len(report['shortened'])}"
            )
        reports.append({"armature": armature_obj.name, **report})
    return reports


def align_animation_root_to_rest(armature_obj, setup_frame) -> float:
    """Remove a constant root-bone yaw the action carries vs the rest pose.

    Generic and rig-agnostic: it uses the structural root and its child branches
    (no feet, left/right labels, or anatomy). Some source clips are authored
    with the character pre-rotated in the action, so the animated pose faces a
    different direction than the already-normalized rest. Rotate the whole rig
    about world Z so both topology-derived forward vectors match. Returns the
    applied rotation in degrees.
    """
    rest_forward = _rig_forward_world(armature_obj)
    if rest_forward is None:
        return 0.0
    scene = bpy.context.scene
    previous = scene.frame_current
    previous_sub = getattr(scene, "frame_subframe", 0.0)
    scene.frame_set(int(setup_frame))
    bpy.context.view_layer.update()
    posed_forward = _posed_rig_forward_world(armature_obj)
    if posed_forward is None:
        scene.frame_set(previous, subframe=previous_sub)
        bpy.context.view_layer.update()
        return 0.0
    cross_z = rest_forward.x * posed_forward.y - rest_forward.y * posed_forward.x
    dot = max(
        -1.0,
        min(1.0, rest_forward.x * posed_forward.x + rest_forward.y * posed_forward.y),
    )
    delta = math.atan2(cross_z, dot)
    if abs(delta) < math.radians(5.0):
        scene.frame_set(previous, subframe=previous_sub)
        bpy.context.view_layer.update()
        return 0.0
    pivot = armature_obj.matrix_world.translation.copy()
    rotation = (
        mathutils.Matrix.Translation(pivot)
        @ mathutils.Matrix.Rotation(-delta, 4, "Z")
        @ mathutils.Matrix.Translation(-pivot)
    )
    armature_obj.matrix_world = rotation @ armature_obj.matrix_world
    bpy.context.view_layer.update()
    scene.frame_set(previous, subframe=previous_sub)
    bpy.context.view_layer.update()
    print(
        f"[blender_scene_io] aligned animation root yaw by "
        f"{math.degrees(-delta):.1f} deg (setup_frame={setup_frame})"
    )
    return math.degrees(-delta)


def _align_root_unless_bind_borrowed(armature_obj, setup_frame, bind_borrow_info) -> float:
    """Run the root-yaw alignment only when NO bind pose was borrowed.

    `align_animation_root_to_rest` infers the rig's facing from its root child
    branches and removes a constant yaw an action carries vs the rest pose.
    That inference assumes a roughly symmetric pose; a borrowed bind donor is
    typically a mid-stride WALK pose (one leg forward, one back), which skews
    the inferred forward and makes the alignment rotate the rig by a large
    spurious angle (~90°). It is also unnecessary in that case: the bind borrow
    already poses the rig with the donor's world rotations, and the donor was
    facing-normalized at import, so the facing is correct by construction.
    Skip the alignment whenever the bind borrow was applied.
    """
    if isinstance(bind_borrow_info, dict) and bind_borrow_info.get("applied"):
        return 0.0
    return align_animation_root_to_rest(armature_obj, setup_frame)


def purge_model_animations(armature_obj=None) -> int:
    """Delete every action in the scene before retargeting external motion.

    The imported target model almost always ships its own embedded action (a
    bind/idle/walk baked into the FBX). If it stays in ``bpy.data.actions`` it
    contaminates retargeting: it can drive the armature during the transfer,
    it perturbs Blender's action-name dedup (so the source action gets a
    ``.001`` suffix or, worse, the model's own action leaks into the output as
    a static junk clip), and it shows up in the optimizer + exported clip
    list. The rule is simple and robust: wipe ALL model animation before we
    apply any retargeted source motion. Call this right after importing the
    model and BEFORE importing/borrowing any animation source.
    """
    if armature_obj is not None and armature_obj.animation_data is not None:
        armature_obj.animation_data.action = None
    removed = 0
    for action in list(bpy.data.actions):
        try:
            bpy.data.actions.remove(action)
            removed += 1
        except Exception:
            pass
    return removed


def find_all_meshes_and_armature():
    """Return every mesh object in the scene plus the armature.

    The mesh list is ordered with the primary mesh (the one with the most
    vertices) first, followed by the remaining meshes sorted by descending
    vertex count then name for a stable order. A source FBX may carry several
    mesh objects skinned to a single shared armature (e.g. a character body
    plus a separate sword / accessory); every consumer that only handled the
    largest mesh silently dropped the rest. Callers iterate this list so each
    object becomes its own sprite/slot.
    """
    meshes = []
    armature_obj = None

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            meshes.append(obj)
        elif obj.type == "ARMATURE":
            armature_obj = obj

    meshes.sort(key=lambda obj: (-len(obj.data.vertices), obj.name))

    primary = meshes[0] if meshes else None

    if primary and primary.parent and primary.parent.type == "ARMATURE":
        armature_obj = primary.parent

    if primary and armature_obj is None:
        for modifier in primary.modifiers:
            if modifier.type == "ARMATURE" and modifier.object:
                armature_obj = modifier.object
                break

    return meshes, armature_obj


def find_mesh_and_armature():
    meshes, armature_obj = find_all_meshes_and_armature()
    return (meshes[0] if meshes else None), armature_obj


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
    all_meshes, armature_obj = find_all_meshes_and_armature()
    mesh_obj = all_meshes[0] if all_meshes else None

    payload: dict[str, object] = {
        "ok": True,
        "detail": "ready",
        "source": source_path,
        "format": Path(source_path).suffix.lower().lstrip("."),
        "source_space": "3d",
        "supports_character_build": True,
        "supports_animation_append": True,
        "mesh": None,
        "meshes": [],
        "armature": None,
        "actions": [],
        "normalized_format": "glb"
        if Path(source_path).suffix.lower() == ".fbx"
        else Path(source_path).suffix.lower().lstrip("."),
    }

    payload["meshes"] = [
        {
            "name": scene_mesh.name,
            "is_primary": index == 0,
            "vertex_count": int(len(scene_mesh.data.vertices)),
            "triangle_count": int(len(scene_mesh.data.polygons)),
        }
        for index, scene_mesh in enumerate(all_meshes)
    ]

    if mesh_obj is not None:
        payload["mesh"] = {
            "name": mesh_obj.name,
            "vertex_count": int(len(mesh_obj.data.vertices)),
            "triangle_count": int(len(mesh_obj.data.polygons)),
        }

    if armature_obj is not None:
        deform_bone_names = sorted(_deform_bone_names(armature_obj, all_meshes))
        bind_signature_bones = deform_bone_names or sorted(bone.name for bone in armature_obj.data.bones)
        payload["armature"] = {
            "name": armature_obj.name,
            "bone_count": int(len(armature_obj.data.bones)),
            "bone_names": sorted(bone.name for bone in armature_obj.data.bones),
            # Name-agnostic deform skeleton (skin-weighted bones + connectors +
            # tips). Lets the C++ "same rig" check compare deform skeletons and
            # treat a control-rig-wrapped rig (mesh2motion) as identical to the
            # bare same skeleton (quaternius) for direct retarget.
            "deform_bone_names": deform_bone_names,
            "local_bind_pose": _local_bind_pose_signature(armature_obj, bind_signature_bones),
            "canonicalized_mixamo_to_mannequin": False,
        }
        payload["actions"] = list_actions(armature_obj)

    return payload


def convert_source(source_path: str, output_path: str) -> dict[str, object]:
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
        "target_format": "glb",
        "inspection": inspected,
    }


def _drop_debris_islands(mesh_obj, min_dimension_fraction: float = 0.1):
    """Split loose parts and delete floating debris islands.

    Keeps every island whose world-space bounding size is at least
    ``min_dimension_fraction`` of the largest island, then rejoins the
    keepers. Returns ``(joined_mesh_obj, dropped_count)``.
    """
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")

    parts = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not parts:
        return mesh_obj, 0
    sizes = {part.name: max(part.dimensions) for part in parts}
    largest_size = max(sizes.values()) if sizes else 0.0
    keepers = [
        part
        for part in parts
        if largest_size <= 0.0 or sizes[part.name] >= min_dimension_fraction * largest_size
    ]
    dropped = [part for part in parts if part not in keepers]
    for part in dropped:
        mesh_data = part.data
        bpy.data.objects.remove(part, do_unlink=True)
        if mesh_data.users == 0:
            bpy.data.meshes.remove(mesh_data)

    bpy.ops.object.select_all(action="DESELECT")
    primary = max(keepers, key=lambda part: sizes[part.name])
    for part in keepers:
        part.select_set(True)
    bpy.context.view_layer.objects.active = primary
    if len(keepers) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active, len(dropped)


def _prepare_materials_for_fbx_export(mesh_obj, out_dir, stem: str) -> list[str]:
    """Make GLB-imported textures survive an FBX ``embed_textures`` export.

    Two Blender/glTF quirks otherwise silently drop the texture:

    1. The glTF importer wires the base-colour image through a *MIX* node
       (image x baseColorFactor), but the FBX exporter only recognises a
       texture wired *directly* to the Principled BSDF's Base Color -- so we
       relink the image node straight to Base Color.
    2. A GLB-imported image is packed in memory with no on-disk path, and
       ``embed_textures`` only embeds images that have a filepath -- so we
       save each to a PNG beside the FBX (which doubles as a sidecar texture)
       and repoint the image at it.

    Returns the written PNG paths.
    """
    written: list[str] = []
    seen = set()
    for material in (m for m in mesh_obj.data.materials if m):
        if not material.use_nodes:
            continue
        node_tree = material.node_tree
        bsdf = next((n for n in node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        tex = next(
            (n for n in node_tree.nodes if n.type == "TEX_IMAGE" and n.image is not None), None
        )
        if bsdf is not None and tex is not None:
            base_color = bsdf.inputs["Base Color"]
            for link in list(base_color.links):
                node_tree.links.remove(link)
            node_tree.links.new(tex.outputs["Color"], base_color)
        for node in node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            image = node.image
            if image.name in seen or not image.has_data:
                continue
            seen.add(image.name)
            suffix = f"_{len(written)}" if written else ""
            png_path = Path(out_dir) / f"{stem}_texture{suffix}.png"
            image.filepath_raw = str(png_path)
            image.file_format = "PNG"
            image.save()
            written.append(str(png_path))
    return written


def cleanup_generated_mesh(
    source_path: str,
    *,
    glb_output: str,
    target_triangles: int = 10000,
    voxel_remesh: bool = True,
    remove_loose: bool = True,
    fbx_output: str | None = None,
    orientation_fix: str = "none",
) -> dict[str, object]:
    """Prepare a raw image-to-3D mesh for auto-rigging.

    Import -> join mesh objects -> optional up-axis fix -> drop floating
    debris -> optional voxel remesh (closes holes; single manifold skin;
    note it discards UVs, so it should stay off for textured generators) ->
    decimate to the triangle budget -> GLB export. Armatures are neither
    expected nor preserved.

    ``glb_output`` is always written (the auto-riggers read it back as their
    mesh input). ``fbx_output`` is optional and, when set, exports the same
    cleaned mesh as FBX too -- used by the image-to-3D pipeline's *no-rig*
    path so the final saved asset is FBX like every rigged path, without a
    second Blender launch (the mesh is already prepared here).

    ``orientation_fix`` bakes an up-axis correction into the saved asset so
    it stands upright when opened directly (e.g. in Blender's Z-up world),
    not just inside FlatRig (whose extract-scene normalizes orientation on
    its own). ``"y_up_to_z_up"`` rotates +90 deg about X -- Y-up
    generators emit meshes that otherwise import lying on their back. ``"none"`` leaves
    orientation untouched (generators that already match, or where the
    convention isn't confirmed).
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        return {"ok": False, "detail": "Source has no mesh objects.", "source": source_path}

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    if len(mesh_objects) > 1:
        bpy.ops.object.join()
    mesh_obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    if orientation_fix == "y_up_to_z_up":
        # Stand a Y-up mesh (imported lying down in Blender's Z-up world)
        # upright. -90 deg about X is the empirically-correct sense: +90 puts
        # the figure head-down (verified by rendering a character at 0/+90/-90;
        # -90 is the only upright one). NB the glTF importer leaves objects in
        # QUATERNION rotation mode, so assigning rotation_euler is silently
        # ignored -- force XYZ first.
        bpy.ops.object.select_all(action="DESELECT")
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.rotation_mode = "XYZ"
        mesh_obj.rotation_euler = (math.radians(-90.0), 0.0, 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    triangles_before = _mesh_triangle_count(mesh_obj)

    # Weld position-duplicate vertices before any island/decimate step.
    # Generators that bake a UV atlas (e.g. via xatlas) ship the GLB with
    # vertices split along every UV seam, and the glTF importer keeps them
    # split -- so by edge connectivity each UV chart is its own "loose part".
    # Without this weld, separate(type="LOOSE") sees a closed body as
    # hundreds of floating charts and the debris filter tears real holes in
    # the surface (and the decimate modifier tears along seams the same way).
    # UVs live on face loops, so merging coincident vertices preserves seams.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-6)
    bpy.ops.object.mode_set(mode="OBJECT")

    dropped_islands = 0
    if remove_loose:
        mesh_obj, dropped_islands = _drop_debris_islands(mesh_obj)

    voxel_remesh_applied = False
    if voxel_remesh:
        diagonal = float(mesh_obj.dimensions.length)
        if diagonal > 1e-6:
            mesh_obj.data.remesh_voxel_size = max(diagonal / 120.0, 1e-4)
            bpy.context.view_layer.objects.active = mesh_obj
            bpy.ops.object.voxel_remesh()
            voxel_remesh_applied = True

    triangles_current = _mesh_triangle_count(mesh_obj)
    decimated = False
    if target_triangles > 0 and triangles_current > target_triangles:
        modifier = mesh_obj.modifiers.new(name="flatrig_cleanup_decimate", type="DECIMATE")
        modifier.ratio = max(0.01, float(target_triangles) / float(triangles_current))
        modifier.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        decimated = True

    export_path = Path(glb_output).expanduser().resolve()
    if export_path.suffix.lower() != ".glb":
        export_path = export_path.with_suffix(".glb")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.export_scene.gltf(
        filepath=str(export_path),
        export_format="GLB",
        export_yup=True,
        export_animations=False,
        export_skins=False,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        use_selection=True,
    )

    fbx_export_path = None
    if fbx_output:
        fbx_export_path = Path(fbx_output).expanduser().resolve()
        if fbx_export_path.suffix.lower() != ".fbx":
            fbx_export_path = fbx_export_path.with_suffix(".fbx")
        fbx_export_path.parent.mkdir(parents=True, exist_ok=True)
        # Relink base colour + write packed textures to disk so the FBX
        # export actually carries the texture (see helper for the two glTF/
        # FBX quirks that otherwise drop it).
        _prepare_materials_for_fbx_export(mesh_obj, fbx_export_path.parent, fbx_export_path.stem)
        bpy.ops.object.select_all(action="DESELECT")
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.export_scene.fbx(
            filepath=str(fbx_export_path),
            use_selection=True,
            add_leaf_bones=False,
            bake_anim=False,
            path_mode="COPY",
            embed_textures=True,
        )

    return {
        "ok": True,
        "detail": "cleaned",
        "source": source_path,
        "output": str(export_path),
        "fbx_output": str(fbx_export_path) if fbx_export_path else None,
        "triangles_before": int(triangles_before),
        "triangles_after": int(_mesh_triangle_count(mesh_obj)),
        "islands_dropped": int(dropped_islands),
        "voxel_remesh_applied": bool(voxel_remesh_applied),
        "decimated": bool(decimated),
    }


def _mesh_topology_mismatch(target_obj, source_obj) -> str | None:
    """Describe why two mesh datablocks cannot share loop-domain surface data."""
    target = target_obj.data
    source = source_obj.data
    if len(target.vertices) != len(source.vertices):
        return (
            "vertex count differs "
            f"(prediction={len(target.vertices)}, source={len(source.vertices)})"
        )
    if len(target.polygons) != len(source.polygons):
        return (
            "polygon count differs "
            f"(prediction={len(target.polygons)}, source={len(source.polygons)})"
        )
    if len(target.loops) != len(source.loops):
        return (
            "loop count differs "
            f"(prediction={len(target.loops)}, source={len(source.loops)})"
        )

    for index, (target_polygon, source_polygon) in enumerate(
        zip(target.polygons, source.polygons)
    ):
        if int(target_polygon.loop_start) != int(source_polygon.loop_start):
            return (
                f"polygon {index} loop start differs "
                f"(prediction={target_polygon.loop_start}, source={source_polygon.loop_start})"
            )
        if int(target_polygon.loop_total) != int(source_polygon.loop_total):
            return (
                f"polygon {index} loop count differs "
                f"(prediction={target_polygon.loop_total}, source={source_polygon.loop_total})"
            )
    for index, (target_loop, source_loop) in enumerate(zip(target.loops, source.loops)):
        if int(target_loop.vertex_index) != int(source_loop.vertex_index):
            return (
                f"loop {index} references a different vertex "
                f"(prediction={target_loop.vertex_index}, source={source_loop.vertex_index})"
            )
    return None


def _copy_mesh_surface_data_by_topology(target_obj, source_obj) -> dict[str, int]:
    """Copy loop UVs and material assignments between identical mesh topologies."""
    mismatch = _mesh_topology_mismatch(target_obj, source_obj)
    if mismatch is not None:
        raise ValueError(f"Cannot copy mesh surface data: {mismatch}.")

    target = target_obj.data
    source = source_obj.data
    for target_uv in list(target.uv_layers):
        target.uv_layers.remove(target_uv)
    for source_uv in source.uv_layers:
        target_uv = target.uv_layers.new(name=source_uv.name)
        if len(target_uv.data) != len(source_uv.data):
            raise ValueError(
                f"Cannot copy UV layer '{source_uv.name}': loop data length differs "
                f"(prediction={len(target_uv.data)}, source={len(source_uv.data)})."
            )
        for target_value, source_value in zip(target_uv.data, source_uv.data):
            target_value.uv = source_value.uv
        if hasattr(target_uv, "active_render") and hasattr(source_uv, "active_render"):
            target_uv.active_render = source_uv.active_render
    if source.uv_layers:
        target.uv_layers.active_index = source.uv_layers.active_index

    target.materials.clear()
    for material in source.materials:
        target.materials.append(material)
    for target_polygon, source_polygon in zip(target.polygons, source.polygons):
        target_polygon.material_index = source_polygon.material_index

    return {
        "uv_layer_count": len(target.uv_layers),
        "material_count": len(target.materials),
    }


def _fit_orientation_preserving_similarity(
    source_points, target_points, *, relative_rms_tolerance=1e-5, relative_max_tolerance=1e-4
) -> dict[str, object]:
    """Fit ``target = scale * rotation @ source + translation`` with proper rotation."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or target.shape != source.shape:
        raise ValueError(
            "Similarity fit requires equally-shaped (N, 3) source and target point arrays."
        )
    if len(source) < 4:
        raise ValueError("Similarity fit requires at least four corresponding vertices.")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Similarity fit received non-finite vertex coordinates.")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_variance <= 1e-16:
        raise ValueError("Similarity fit source vertices have no usable spatial extent.")
    if np.linalg.matrix_rank(source_centered) < 3 or np.linalg.matrix_rank(target_centered) < 3:
        raise ValueError("Similarity fit requires non-degenerate 3D vertex correspondences.")

    covariance = (target_centered.T @ source_centered) / float(len(source))
    left, singular_values, right_transposed = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(left @ right_transposed) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transposed
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"Similarity fit produced an invalid uniform scale ({scale!r}).")
    rotation_determinant = float(np.linalg.det(rotation))
    if rotation_determinant <= 0.0 or abs(rotation_determinant - 1.0) > 1e-5:
        raise ValueError(
            "Similarity fit did not produce an orientation-preserving rotation "
            f"(determinant={rotation_determinant:.8g})."
        )

    translation = target_center - scale * (rotation @ source_center)
    mapped = (scale * (rotation @ source.T)).T + translation
    errors = np.linalg.norm(mapped - target, axis=1)
    rms_error = float(np.sqrt(np.mean(errors * errors)))
    max_error = float(errors.max())
    target_diagonal = float(np.linalg.norm(target.max(axis=0) - target.min(axis=0)))
    reference_extent = max(target_diagonal, 1e-12)
    relative_rms_error = rms_error / reference_extent
    relative_max_error = max_error / reference_extent
    if (
        relative_rms_error > float(relative_rms_tolerance)
        or relative_max_error > float(relative_max_tolerance)
    ):
        raise ValueError(
            "Prediction/source vertices are not related by one orientation-preserving "
            "uniform similarity "
            f"(relative RMS={relative_rms_error:.6g}, "
            f"relative max={relative_max_error:.6g})."
        )

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return {
        "matrix": matrix,
        "scale": scale,
        "rotation_determinant": rotation_determinant,
        "rms_error": rms_error,
        "max_error": max_error,
        "relative_rms_error": relative_rms_error,
        "relative_max_error": relative_max_error,
    }


def bake_predicted_rig(npz_path: str, *, fbx_output: str, mesh_path: str | None = None) -> dict[str, object]:
    """Build a from-scratch armature for an externally predicted rig and export FBX.

    ``npz_path`` is not a 3D source file: it's a numpy ``.npz`` written by an
    external prediction step (currently the Make-It-Animatable rigger runner
    in ``flatrig_private``, which has no bpy dependency of its own) carrying:

    - ``vertices`` (V, 3) float32, ``triangles`` (M, 3) uint32 -- the exact
      mesh the weights below were predicted against (built directly via
      ``from_pydata`` rather than re-importing a GLB, so vertex order can
      never drift from what the predictor used).
    - ``bone_names`` (B,) str, ``parent_indices`` (B,) int32 (-1 = root),
      ``heads``/``tails`` (B, 3) float32 -- a skeleton with no template file
      behind it (no bpy import at prediction time either -- see
      ``mia_runner.py``'s module docstring for why: the upstream project's
      own hierarchy derivation needs a real Mixamo-derived ``bones.fbx``
      pulled from a gated HF dataset whose terms don't cover commercial
      redistribution).
    - ``joints_top4`` (V, 4) uint8, ``weights_top4`` (V, 4) float32 -- the
      predicted skin weights, already normalized to the 4 dominant bones per
      vertex (the standard glTF/FBX skinning convention).

    When ``mesh_path`` is provided, its imported mesh must retain this exact
    polygon/loop topology. Its UVs and materials are copied by loop index, and
    the unique proper uniform similarity from prediction coordinates to the
    imported mesh's world coordinates is applied to the armature parent. This
    preserves the source asset's coordinate frame without transforming the
    parented mesh twice or changing any vertex-to-weight correspondence.
    """
    data = np.load(npz_path, allow_pickle=True)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    triangles = np.asarray(data["triangles"], dtype=np.int64)
    bone_names = [str(name) for name in data["bone_names"]]
    parent_indices = np.asarray(data["parent_indices"], dtype=np.int64)
    heads = np.asarray(data["heads"], dtype=np.float64)
    tails = np.asarray(data["tails"], dtype=np.float64)
    joints_top4 = np.asarray(data["joints_top4"], dtype=np.int64)
    weights_top4 = np.asarray(data["weights_top4"], dtype=np.float64)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    mesh_data = bpy.data.meshes.new("mia_mesh")
    mesh_data.from_pydata(vertices.tolist(), [], triangles.tolist())
    mesh_data.update()
    mesh_obj = bpy.data.objects.new("mia_mesh", mesh_data)
    bpy.context.collection.objects.link(mesh_obj)

    armature_data = bpy.data.armatures.new("mia_armature")
    armature_obj = bpy.data.objects.new("mia_armature", armature_data)
    bpy.context.collection.objects.link(armature_obj)

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature_data.edit_bones
    created = []
    min_length = 1e-3
    for index, name in enumerate(bone_names):
        edit_bone = edit_bones.new(name)
        head = mathutils.Vector(heads[index].tolist())
        tail = mathutils.Vector(tails[index].tolist())
        if (tail - head).length < min_length:
            tail = head + mathutils.Vector((0.0, min_length, 0.0))
        edit_bone.head = head
        edit_bone.tail = tail
        created.append(edit_bone)
    for index, parent_index in enumerate(parent_indices.tolist()):
        if parent_index >= 0:
            created[index].parent = created[parent_index]
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh_obj.parent = armature_obj
    for name in bone_names:
        mesh_obj.vertex_groups.new(name=name)
    vertex_count = vertices.shape[0]
    for vertex_index in range(vertex_count):
        for slot in range(joints_top4.shape[1]):
            weight = float(weights_top4[vertex_index, slot])
            if weight <= 1e-4:
                continue
            bone_index = int(joints_top4[vertex_index, slot])
            mesh_obj.vertex_groups[bone_index].add([vertex_index], weight, "REPLACE")
    modifier = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature_obj

    surface_transfer = None
    source_alignment = None
    if mesh_path:
        resolved_mesh_path = Path(mesh_path).expanduser().resolve()
        if not resolved_mesh_path.is_file():
            raise ValueError(f"Original mesh path does not exist: {resolved_mesh_path}")
        objects_before = set(bpy.context.scene.objects)
        import_model(str(resolved_mesh_path))
        imported_objects = [
            obj for obj in bpy.context.scene.objects if obj not in objects_before
        ]
        imported_meshes = [obj for obj in imported_objects if obj.type == "MESH"]
        compatible_meshes = [
            obj
            for obj in imported_meshes
            if _mesh_topology_mismatch(mesh_obj, obj) is None
        ]
        if not compatible_meshes:
            mismatch_details = "; ".join(
                f"{obj.name}: {_mesh_topology_mismatch(mesh_obj, obj)}"
                for obj in imported_meshes
            )
            if not mismatch_details:
                mismatch_details = "the imported source contains no mesh objects"
            raise ValueError(
                "Original mesh cannot provide UV/material data because no imported mesh "
                f"matches the prediction topology ({mismatch_details})."
            )
        if len(compatible_meshes) > 1:
            names = ", ".join(obj.name for obj in compatible_meshes)
            raise ValueError(
                "Original mesh import is ambiguous: multiple objects match the prediction "
                f"topology ({names})."
            )
        original_mesh = compatible_meshes[0]
        surface_transfer = _copy_mesh_surface_data_by_topology(mesh_obj, original_mesh)

        original_world_vertices = np.asarray(
            [
                tuple(original_mesh.matrix_world @ vertex.co)
                for vertex in original_mesh.data.vertices
            ],
            dtype=np.float64,
        )
        source_alignment = _fit_orientation_preserving_similarity(
            vertices, original_world_vertices
        )
        # The mesh is already parented to the armature. Applying the similarity
        # only to that parent makes both mesh and bones inherit it exactly once;
        # transforming both objects would rotate/scale the child twice and break
        # the bind relationship.
        armature_obj.matrix_world = mathutils.Matrix(source_alignment["matrix"].tolist())

        for obj in imported_objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.context.view_layer.update()

    export_path = Path(fbx_output).expanduser().resolve()
    if export_path.suffix.lower() != ".fbx":
        export_path = export_path.with_suffix(".fbx")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.export_scene.fbx(
        filepath=str(export_path),
        use_selection=True,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode="COPY",
        embed_textures=True,
    )

    return {
        "ok": True,
        "detail": "Predicted rig baked and exported.",
        "source": npz_path,
        "output": str(export_path),
        "bone_count": len(bone_names),
        "vertex_count": int(vertex_count),
        "surface_transfer": surface_transfer,
        "source_alignment": (
            {
                key: value
                for key, value in source_alignment.items()
                if key != "matrix"
            }
            if source_alignment is not None
            else None
        ),
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
        [world_matrix[3][0], world_matrix[3][1], world_matrix[3][2], world_matrix[3][3]],
    ]


def _base_color_image(mesh_obj):
    """Return the first image driving a material's base color."""
    if mesh_obj is None:
        return None

    mesh = mesh_obj.data
    if not hasattr(mesh, "materials") or not mesh.materials:
        return None

    # Try to get the first material with a base color texture
    for slot in mesh.materials:
        if slot is None:
            continue

        material = slot
        # Handle material slots in Blender 5.0+
        if hasattr(slot, "material"):
            material = slot.material

        if material is None:
            continue

        # Check for Principled BSDF shader and its Base Color input
        if hasattr(material, "node_tree") and material.node_tree:
            nodes = material.node_tree.nodes
            for node in nodes:
                if node.type == "BSDF_PRINCIPLED":
                    # Try to get the Base Color input
                    base_color_input = node.inputs.get("Base Color")
                    if base_color_input and base_color_input.links:
                        link = base_color_input.links[0]
                        from_node = link.from_node

                        # Check if it's an image texture
                        if from_node and from_node.type == "TEX_IMAGE":
                            image = from_node.image
                            if image and image.size[0] > 0 and image.size[1] > 0:
                                return image

        # Fallback: try to find any image texture node
        if hasattr(material, "node_tree") and material.node_tree:
            nodes = material.node_tree.nodes
            for node in nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image and image.size[0] > 0 and image.size[1] > 0:
                        return image

    return None


def save_base_color_texture(mesh_obj, output_path: str) -> dict[str, Any] | None:
    """Save the model's base-color image as a full-resolution PNG."""
    image = _base_color_image(mesh_obj)
    if image is None:
        return None

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_copy = image.copy()
    try:
        image_copy.filepath_raw = str(destination)
        image_copy.file_format = "PNG"
        image_copy.save()
    finally:
        bpy.data.images.remove(image_copy, do_unlink=True)
    if not destination.exists():
        return None
    return {
        "path": str(destination),
        "width": int(image.size[0]),
        "height": int(image.size[1]),
        "channels": int(getattr(image, "channels", 4) or 4),
    }


def _mesh_triangle_count(mesh_obj) -> int:
    mesh = mesh_obj.data
    if hasattr(mesh, "calc_loop_triangles"):
        mesh.calc_loop_triangles()
    return int(len(getattr(mesh, "loop_triangles", []) or mesh.polygons))


# Decimation importance (weight-aware collapse). Vertices whose skin weight is
# split between several bones sit in deformation-sensitive blend zones (near
# joints, or where authoring left "problematic" mixed weights); preserving a
# little more geometry there while collapsing flat single-bone regions keeps the
# projected sprite clean where it matters. The signal is the *secondary*
# influence (1 - dominant normalized weight): a vertex fully owned by one bone
# contributes no bonus.
#
# The bias is deliberately *compensated*, not winner-take-all. Every vertex keeps
# a baseline FLOOR weight so rigid regions still survive decimation with most of
# their detail; blend zones only earn a bounded bonus on top. An earlier version
# scored rigid vertices at 0 (free to collapse in Blender's Collapse modifier),
# which dumped the whole vertex budget onto joints, gutted flat regions, and tend
# to trip the starvation guard into a uniform fallback — losing the prior
# entirely. The narrow FLOOR..MAX band keeps the two zones close while still
# protecting joints a touch more.
IMPORTANCE_GROUP_NAME = "FlatRig_DecimateImportance"
# Baseline weight every vertex receives, so rigid single-bone surfaces are never
# treated as free-to-collapse relative to joints.
DECIMATE_IMPORTANCE_FLOOR = 0.45
# Slope of the per-vertex bonus above the floor as the secondary influence grows.
DECIMATE_IMPORTANCE_GAIN = 1.6
# Upper bound on the total weight (floor + bonus). Keeps the rigid:flexible ratio
# gentle (MAX / FLOOR), so detail stays spread out instead of piling onto joints.
DECIMATE_IMPORTANCE_MAX = 0.8
# Blender Collapse decimate: higher group weight preserves detail. Flip only if a
# future bpy build inverts that mapping.
DECIMATE_IMPORTANCE_FACTOR = 1.0
DECIMATE_IMPORTANCE_INVERT = False


def _build_decimation_weight_importance(mesh_obj, enabled: bool = False):
    """Paint a temporary vertex group steering Collapse decimation toward joints.

    Returns the group name, or ``None`` when weight-aware decimation is disabled
    or the mesh carries no usable skin weights. The group is computed purely from
    numeric weights (no bone names), so it stays rig-agnostic and deterministic
    across extract/render passes. Callers MUST remove the group before extraction
    so it is never mistaken for a bone influence.

    The prior is intentionally compensated: every vertex keeps a baseline floor
    weight and blend regions earn only a bounded bonus on top, so a rigid torso
    or armor plate is never starved to feed nearby joints. The fallback guard in
    ``reduce_mesh_object`` still throws away a weighted pass if a sizable region
    collapsed far below its fair share. The path remains opt-in because rigid
    armored assets can deform worse even when the starvation guard does not fire.
    """
    if not enabled:
        return None
    mesh = mesh_obj.data
    if not mesh_obj.vertex_groups or len(mesh.vertices) == 0:
        return None

    # Seed every vertex with the baseline floor (rigid/unskinned surfaces keep
    # their fair share of geometry) and only raise blend-zone vertices above it.
    importance = [DECIMATE_IMPORTANCE_FLOOR] * len(mesh.vertices)
    any_bonus = False
    for vert in mesh.vertices:
        weights = [g.weight for g in vert.groups if g.weight > 0.0]
        if len(weights) < 2:
            continue  # single-bone/unskinned: keep the baseline floor only
        total = float(sum(weights))
        if total <= VECTOR_EPSILON:
            continue
        secondary = 1.0 - (max(weights) / total)
        if secondary <= 0.0:
            continue
        score = min(
            DECIMATE_IMPORTANCE_MAX,
            DECIMATE_IMPORTANCE_FLOOR + secondary * DECIMATE_IMPORTANCE_GAIN,
        )
        importance[vert.index] = score
        any_bonus = True

    # No blend zones means a flat floor everywhere == uniform; let the plain
    # uniform path handle it instead of painting a pointless group.
    if not any_bonus:
        return None

    group = mesh_obj.vertex_groups.new(name=IMPORTANCE_GROUP_NAME)
    for index, score in enumerate(importance):
        group.add([index], score, "REPLACE")
    return group.name


# Auto-fallback guard for weight-aware decimation. A region (dominant vertex
# group) that lost almost all of its geometry means the weight prior dumped the
# whole budget elsewhere, hollowing that region into a hole. When that happens we
# throw the weight-aware result away and redo the decimation uniformly.
#
# The guard is measured *relative to the global decimation ratio*, not against a
# fixed retention floor. Decimating to (say) 25 % of the source naturally brings
# every region to ~25 %, so an absolute 0.25 floor used to fire on essentially
# every aggressive run and force a needless uniform fallback. A region is only
# starved when it kept far less than its fair share of the global ratio.
DECIMATE_MIN_REGION_VERTS = 25
# A region must retain at least this fraction of its *fair share* (before_count *
# global_ratio). Below it, the prior genuinely hollowed the region.
DECIMATE_REGION_STARVE_FACTOR = 0.45


def _dominant_group_counts(mesh_obj) -> dict:
    """Count vertices per dominant vertex group (≈ per bone region)."""
    counts: dict = {}
    for vert in mesh_obj.data.vertices:
        best_group, best_weight = -1, 0.0
        for influence in vert.groups:
            if influence.weight > best_weight:
                best_weight, best_group = influence.weight, influence.group
        if best_group >= 0:
            counts[best_group] = counts.get(best_group, 0) + 1
    return counts


def _weight_aware_starved(before: dict, after: dict, global_ratio: float) -> tuple:
    """Return (starved, worst_group) if a region collapsed far below its fair share.

    ``global_ratio`` is the overall target/source vertex ratio. A region's fair
    share after decimation is ``before_count * global_ratio``; it is flagged as
    starved only when it kept less than ``DECIMATE_REGION_STARVE_FACTOR`` of that,
    i.e. it shrank much harder than the mesh as a whole (the prior diverted its
    budget) rather than merely because the target is aggressive.
    """
    floor_ratio = max(0.0, float(global_ratio)) * DECIMATE_REGION_STARVE_FACTOR
    for group, before_count in before.items():
        if before_count >= DECIMATE_MIN_REGION_VERTS:
            after_count = after.get(group, 0)
            if after_count < before_count * floor_ratio:
                return True, group
    return False, None


def _run_collapse_passes(mesh_obj, target_vertices, source_vertex_count, importance_group):
    """Apply Collapse decimation and return the output vertex count.

    With an importance group a single pass keeps the protected regions dense;
    without one it iterates toward the target.
    """
    current_vertices = source_vertex_count
    for pass_index in range(4):
        if current_vertices <= target_vertices:
            break
        ratio = max(0.01, min(1.0, float(target_vertices) / max(float(current_vertices), 1.0)))
        modifier = mesh_obj.modifiers.new(
            name=f"FlatRig_SourceMeshReduction_{pass_index + 1}",
            type="DECIMATE",
        )
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio
        if importance_group is not None and importance_group in mesh_obj.vertex_groups:
            modifier.vertex_group = importance_group
            if hasattr(modifier, "vertex_group_factor"):
                modifier.vertex_group_factor = DECIMATE_IMPORTANCE_FACTOR
            if hasattr(modifier, "invert_vertex_group"):
                modifier.invert_vertex_group = DECIMATE_IMPORTANCE_INVERT
        if hasattr(modifier, "use_collapse_triangulate"):
            modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.context.view_layer.update()
        current_vertices = int(len(mesh_obj.data.vertices))
        if importance_group is not None:
            # Single weight-aware pass: re-collapsing to chase the target would
            # hollow out the unprotected rigid surfaces.
            break
    return current_vertices


def reduce_mesh_object(
    mesh_obj,
    target_vertices=5000,
    enabled=True,
    weight_aware_decimation: bool = False,
) -> dict[str, object]:
    """Reduce the source mesh in Blender before extraction.

    Decimate runs inside Blender so UV layers and vertex-group weights remain on
    the mesh that the native pipeline receives. By default decimation is uniform;
    callers can opt into a temporary importance group that keeps detail near
    joints/blend zones and collapses flat regions harder.
    """
    source_vertex_count = int(len(mesh_obj.data.vertices)) if mesh_obj is not None else 0
    source_triangle_count = _mesh_triangle_count(mesh_obj) if mesh_obj is not None else 0
    target_vertices = int(target_vertices or 0)
    report = {
        "enabled": bool(enabled),
        "applied": False,
        "mode": "sidecar_blender_decimate",
        "target_vertices": target_vertices,
        "source_vertex_count": source_vertex_count,
        "source_triangle_count": source_triangle_count,
        "output_vertex_count": source_vertex_count,
        "output_triangle_count": source_triangle_count,
        "reason": "disabled" if not enabled else "not_run",
        "weight_aware_requested": bool(weight_aware_decimation),
    }

    if not enabled:
        return report
    if mesh_obj is None:
        report["reason"] = "no_mesh"
        return report
    if target_vertices <= 0:
        report["reason"] = "no_target"
        return report
    if source_vertex_count <= target_vertices:
        report["reason"] = "source_under_target"
        return report

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    try:
        importance_group = _build_decimation_weight_importance(
            mesh_obj,
            enabled=bool(weight_aware_decimation),
        )
    except Exception:
        importance_group = None

    weight_aware_attempted = importance_group is not None
    weight_aware_used = False
    fallback_applied = False
    try:
        if importance_group is not None:
            # Try weight-aware, but keep a copy so we can undo if it starves a
            # region. The check is the user's rule: measure the reduction per
            # vertex-group region; if the most aggressively reduced (flattest)
            # one collapses near a hole, switch to uniform automatically.
            before_counts = _dominant_group_counts(mesh_obj)
            backup_mesh = mesh_obj.data.copy()
            _run_collapse_passes(mesh_obj, target_vertices, source_vertex_count, importance_group)
            global_ratio = float(target_vertices) / max(float(source_vertex_count), 1.0)
            starved, _worst = _weight_aware_starved(
                before_counts, _dominant_group_counts(mesh_obj), global_ratio
            )
            if importance_group in mesh_obj.vertex_groups:
                mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[importance_group])
            importance_group = None
            if starved:
                hollowed = mesh_obj.data
                mesh_obj.data = backup_mesh
                bpy.data.meshes.remove(hollowed)
                if IMPORTANCE_GROUP_NAME in mesh_obj.vertex_groups:
                    mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[IMPORTANCE_GROUP_NAME])
                fallback_applied = True
                _run_collapse_passes(mesh_obj, target_vertices, source_vertex_count, None)
            else:
                weight_aware_used = True
                bpy.data.meshes.remove(backup_mesh)
        else:
            _run_collapse_passes(mesh_obj, target_vertices, source_vertex_count, None)
    except Exception as exc:
        raise RuntimeError(f"Source mesh reduction failed: {exc}") from exc
    finally:
        # The importance group must never reach extraction; otherwise it would be
        # read back as a spurious bone weight.
        if IMPORTANCE_GROUP_NAME in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[IMPORTANCE_GROUP_NAME])

    mesh_obj.data.update()
    output_vertex_count = int(len(mesh_obj.data.vertices))
    output_triangle_count = _mesh_triangle_count(mesh_obj)
    if fallback_applied:
        reason = "weight_aware_fallback_uniform"
    elif weight_aware_used:
        reason = "weight_aware_single_pass"
    else:
        reason = "target_reached" if output_vertex_count <= target_vertices else "best_effort"
    report.update(
        {
            "applied": output_vertex_count < source_vertex_count,
            "output_vertex_count": output_vertex_count,
            "output_triangle_count": output_triangle_count,
            "reason": reason,
            "weight_aware": weight_aware_used,
            "weight_aware_attempted": weight_aware_attempted,
            "weight_aware_fallback": fallback_applied,
        }
    )
    return report


# ============================================================================
# Projection Helpers
# ============================================================================








# ============================================================================
# Skeleton Helpers
# ============================================================================


















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


def _build_joint_children(joints):
    children = {int(joint["index"]): [] for joint in joints}
    for joint in joints:
        parent_index = int(joint.get("parent_index", -1))
        if parent_index >= 0:
            children.setdefault(parent_index, []).append(int(joint["index"]))
    return children


def _write_3d_joint_hierarchy(lines, joints, joint_children, joint_index, depth):
    joint = joints[joint_index]
    indent = "\t" * depth
    label = "ROOT" if int(joint.get("parent_index", -1)) < 0 else "JOINT"
    lines.append(f"{indent}{label} {joint['bvh_name']}")
    lines.append(f"{indent}{{")
    channel_indent = f"{indent}\t"
    offset = joint.get("offset") or [0.0, 0.0, 0.0]
    lines.append(f"{channel_indent}OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
    lines.append(
        f"{channel_indent}CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
    )

    children = joint_children.get(joint_index) or []
    if children:
        for child_index in children:
            _write_3d_joint_hierarchy(lines, joints, joint_children, child_index, depth + 1)
    else:
        tail_offset = joint.get("tail_offset") or [1.0, 0.0, 0.0]
        lines.append(f"{channel_indent}End Site")
        lines.append(f"{channel_indent}{{")
        lines.append(
            f"{channel_indent}\tOFFSET {tail_offset[0]:.6f} {tail_offset[1]:.6f} {tail_offset[2]:.6f}"
        )
        lines.append(f"{channel_indent}}}")

    lines.append(f"{indent}}}")


def _write_3d_bvh(output_path, joints, positions, rotations, fps):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not joints:
        raise ValueError("Cannot write BVH without joints.")

    joint_children = _build_joint_children(joints)
    lines = ["HIERARCHY"]
    _write_3d_joint_hierarchy(lines, joints, joint_children, 0, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {len(positions)}")
    lines.append(f"Frame Time: {1.0 / fps:.8f}")

    for frame_positions, frame_rotations in zip(positions, rotations, strict=True):
        values = []
        cursor = 0
        for _joint in joints:
            position_triplet = frame_positions[cursor : cursor + 3]
            rotation_triplet = frame_rotations[cursor : cursor + 3]
            cursor += 3
            values.extend(f"{value:.6f}" for value in position_triplet)
            values.extend(f"{value:.6f}" for value in rotation_triplet)
        lines.append(" ".join(values))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")






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


def _rest_3d_bvh_frames(layout, frame_count=2):
    frame_count = max(2, int(frame_count or 2))
    frame_positions = []
    frame_rotations = []
    for _joint in layout["joints"]:
        frame_positions.extend((0.0, 0.0, 0.0))
        frame_rotations.extend((0.0, 0.0, 0.0))
    return (
        [list(frame_positions) for _ in range(frame_count)],
        [list(frame_rotations) for _ in range(frame_count)],
    )


def _action_match_key(name, object_name=None):
    """Canonical key for matching an FBX take name across a bake/reimport roundtrip.

    A ``bake -> FBX export -> reimport`` roundtrip (e.g. the pseudo-2D
    pre-flatten) re-decorates take names: the exporter prefixes the armature
    object name and the importer prefixes it again, while the ``|Layer<N>``
    suffix is dropped — so ``Armature|mixamo.com|Layer0`` comes back as
    ``Armature|Armature|mixamo.com``. Stripping the trailing ``Layer<N>``
    tokens and any leading repeats of the object name collapses both forms to
    the same key, so clip-name lookups survive the roundtrip instead of failing
    and silently demoting the retarget onto the GMR backend.
    """
    tokens = [token for token in str(name).split("|") if token]
    while tokens and re.match(r"(?i)^layer\d+$", tokens[-1]):
        tokens.pop()
    if object_name:
        object_lower = str(object_name).lower()
        while len(tokens) > 1 and tokens[0].lower() == object_lower:
            tokens.pop(0)
    return "|".join(tokens).lower()


def _resolve_action_for_export(armature_obj, animation_names):
    requested = [str(name) for name in (animation_names or []) if str(name).strip()]
    actions = [action for action in bpy.data.actions if is_pose_action(action)]
    if requested:
        wanted = requested[0]
        for action in actions:
            if action.name == wanted:
                return action
        wanted_lower = wanted.lower()
        for action in actions:
            action_name_lower = str(action.name).lower()
            action_tokens = [token.lower() for token in str(action.name).split("|")]
            if wanted_lower == action_name_lower or wanted_lower in action_tokens:
                return action
            if action_name_lower.endswith("|" + wanted_lower) or wanted_lower in action_name_lower:
                return action
        # Decoration-tolerant match: survive a bake -> FBX -> reimport roundtrip
        # that re-decorates take names (see _action_match_key). Without this the
        # clip-name lookup hard-fails and the caller silently demotes onto GMR.
        object_name = getattr(armature_obj, "name", None)
        wanted_key = _action_match_key(wanted, object_name)
        if wanted_key:
            for action in actions:
                if _action_match_key(action.name, object_name) == wanted_key:
                    return action
        # Last resort: a single pose action is unambiguous, so prefer it over a
        # hard failure (which is exactly what knocks the retarget off the
        # direct backend). Warn so the mismatch stays visible.
        if len(actions) == 1:
            print(
                f"[blender_scene_io] Requested animation '{wanted}' not found; "
                f"using the only pose action '{actions[0].name}'.",
                file=sys.stderr,
            )
            return actions[0]
        available = ", ".join(sorted(action.name for action in actions)) or "<none>"
        raise ValueError(f"Animation '{wanted}' not found. Available animations: {available}")
    if armature_obj.animation_data and armature_obj.animation_data.action:
        return armature_obj.animation_data.action
    if actions:
        return sorted(actions, key=lambda action: str(action.name).lower())[0]
    return None


def _select_sprite_render_frame(armature_obj, source_frame=None) -> int:
    """Pick the frame used as bind pose for sprite/setup extraction.

    Policy (decided 2026-05-09):
      * Explicit positive source_frame wins.
      * Otherwise, return the **first frame** of the input model's action — not
        a percentage of the duration. The first frame is what the user can
        guarantee looks good (the previous 35%-of-duration heuristic landed on
        unpredictable poses and was the original "thin arms" bug).
      * If no action exists, fall back to scene.frame_start. The intended
        long-term fallback is the first frame of a curated single-clip
        optimization animation; that wiring is pending — see
        TODO(single-clip-optimization) below.
    """
    if source_frame is not None and int(source_frame) > 0:
        return int(source_frame)

    scene = bpy.context.scene
    if armature_obj is None:
        return int(scene.frame_start or 1)

    action = _resolve_action_for_export(armature_obj, [])
    if action is not None:
        start, _end = action.frame_range
        return int(round(float(start)))

    # TODO(single-clip-optimization): when the curated single optimization clip
    # is wired through, load its first frame here instead of falling back to
    # scene.frame_start. This is the path taken when the input model brings no
    # animation of its own.
    return int(scene.frame_start or 1)


def _resolve_setup_frame(
    armature_obj,
    source_frame=None,
    use_rest_pose=False,
    neutral_auto_pose=False,
) -> int:
    """Resolve the shared setup frame for mesh, bones, target BVH, and sprites.

    source_frame=-1 means "auto": choose a stable in-action pose when possible.
    Rest-pose extraction still evaluates at a concrete scene frame so evaluated
    meshes and projection metadata stay deterministic.
    """
    scene = bpy.context.scene
    if source_frame is not None:
        source_frame = int(source_frame)
        if source_frame > 0:
            return source_frame
        if source_frame < 0 and neutral_auto_pose:
            return int(scene.frame_start or 1)
        if source_frame < 0 and not use_rest_pose:
            return _select_sprite_render_frame(armature_obj)
    return int(scene.frame_start or 1)


def export_3d_animation_bvh_cli(
    source_path: str,
    output_path: str,
    bvh_output: str,
    animation_names: list = None,
    fps: float = 30.0,
    frame_start: int = None,
    frame_end: int = None,
) -> dict[str, object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)
    _mesh_obj, armature_obj = find_mesh_and_armature()
    if armature_obj is None:
        return {"ok": False, "detail": "No armature found in scene"}
    if fps <= 0.0:
        return {"ok": False, "detail": "fps must be > 0"}

    action = _resolve_action_for_export(armature_obj, animation_names)
    if action is None:
        return {"ok": False, "detail": "No pose animation actions found in scene"}
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    armature_obj.animation_data.action = action

    action_start = int(round(float(action.frame_range[0])))
    alignment_frame = frame_start if frame_start is not None else action_start
    animation_root_alignment = align_animation_root_to_rest(
        armature_obj, alignment_frame
    )

    layout = _build_3d_bvh_layout(armature_obj)
    sample_frames = _sample_action_frames(
        action,
        bpy.context.scene,
        fps,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    positions, rotations = _collect_3d_bvh_frames(armature_obj, layout, sample_frames, fps)
    _write_3d_bvh(bvh_output, layout["joints"], positions, rotations, fps)

    payload = {
        "ok": True,
        "detail": "exported",
        "source": source_path,
        "output": str(Path(output_path)),
        "bvh_output": str(Path(bvh_output)),
        "animation_name": action.name,
        "duration": round((len(sample_frames) - 1) / fps, 4),
        "fps": float(fps),
        "frame_count": len(sample_frames),
        "frame_time": 1.0 / fps,
        "positions_mode": "all",
        "animation_root_alignment_degrees": animation_root_alignment,
        **layout,
    }
    return payload


def export_3d_rest_bvh_cli(
    source_path: str,
    output_path: str,
    bvh_output: str,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
    source_frame: int = None,
    use_rest_pose: bool = False,
    projection_space: str = "world",
    fps: float = 30.0,
    frame_count: int = None,
    bind_from_animation: str = None,
) -> dict[str, object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)
    _mesh_obj, armature_obj = find_mesh_and_armature()
    if armature_obj is None:
        return {"ok": False, "detail": "No armature found in scene"}
    if fps <= 0.0:
        return {"ok": False, "detail": "fps must be > 0"}

    bind_borrow_info = _maybe_borrow_bind_from_animation(
        armature_obj,
        bind_from_animation,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
    )

    setup_frame = _resolve_setup_frame(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
        neutral_auto_pose=True,
    )
    bpy.context.scene.frame_set(setup_frame)
    # Same constant-root-yaw removal as extract_scene_cli so the exported
    # rest BVH faces the same way as the extracted scene (skipped when a bind
    # pose was borrowed — the donor pose already fixes the facing).
    if not use_rest_pose:
        _align_root_unless_bind_borrowed(armature_obj, setup_frame, bind_borrow_info)
    setup_pose = _apply_auto_setup_pose(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
    )
    view_cfg = get_scene_view_config(
        view_name=view_preset,
        view_dir=tuple(view_dir) if view_dir is not None else None,
        up_hint=tuple(view_up) if view_up is not None else None,
        roll_degrees=view_roll,
        armature_obj=armature_obj,
    )
    bones_2d = extract_setup_bone_hierarchy(
        armature_obj,
        view_cfg,
        source_frame=setup_frame,
        use_rest_pose=use_rest_pose,
        projection_space=projection_space,
        projection_reference_root=None,
        bind_borrow_info=bind_borrow_info,
    )
    # Capture the 3D bind orientation at the same donor setup frame used for
    # sprites, but never borrow donor head/tail positions as target morphology.
    bind_pose_3d = extract_setup_bone_hierarchy_3d(
        armature_obj,
        source_frame=setup_frame,
        use_rest_pose=use_rest_pose,
        bind_borrow_info=bind_borrow_info,
    )
    # The "rest" BVH that external matcher uses as the target reference must be
    # built from the actual 3D rest pose of the rig, NOT from the bind frame
    # the user picked for sprite rendering. The source BVH (input animation)
    # is also built in rest pose (default of `_build_3d_bvh_layout`) and external matcher
    # assumes both rigs share that convention. Building the target layout from
    # the bind frame (e.g. frame 1 of a Walk action) made external matcher interpret every
    # rotation against a posed reference, producing visibly broken retargets
    # whenever the bind frame wasn't the rest pose. Fixed 2026-05-09.
    layout = _build_3d_bvh_layout(
        armature_obj,
        source_frame=None,
        use_rest_pose=True,
    )
    positions, rotations = _rest_3d_bvh_frames(layout, frame_count=frame_count)
    _write_3d_bvh(bvh_output, layout["joints"], positions, rotations, fps)

    return {
        "ok": True,
        "detail": "exported",
        "source": source_path,
        "output": str(Path(output_path)),
        "bvh_output": str(Path(bvh_output)),
        "animation_name": "__target_rest__",
        "duration": round((len(positions) - 1) / fps, 4),
        "fps": float(fps),
        "frame_count": len(positions),
        "frame_time": 1.0 / fps,
        "positions_mode": "all",
        "projection_space": projection_space,
        "setup_frame": setup_frame,
        "setup_pose": setup_pose,
        "bind_borrow_info": bind_borrow_info,
        "use_rest_pose": bool(use_rest_pose),
        "retarget_use_rest_pose": bool(use_rest_pose),
        "view": _view_config_to_json(view_cfg),
        "bones_2d": bones_2d,
        "bind_pose_3d": bind_pose_3d,
        **layout,
    }


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


def _timeline_duration(timeline: dict[str, object]) -> float:
    duration = 0.0
    for key in ("rotate", "translate", "scale", "shear"):
        for record in timeline.get(key) or []:
            duration = max(duration, float(record.get("time", 0.0)))
    return duration


def _animation_duration(animation_payload: dict[str, object]) -> float:
    duration = 0.0
    for timeline in (animation_payload.get("bones") or {}).values():
        duration = max(duration, _timeline_duration(timeline or {}))
    return duration


def _serialize_animations(animations: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    records = []
    for animation_name, payload in animations.items():
        payload = payload or {}
        records.append(
            {
                "name": str(animation_name),
                "duration": round(_animation_duration(payload), 4),
                "bones": dict(payload.get("bones") or {}),
                "frame_filter": dict(payload.get("frame_filter") or {}),
                "source_authored_transform_keys": bool(
                    payload.get("source_authored_transform_keys", False)
                ),
                "transfer_mode": str(payload.get("transfer_mode") or ""),
            }
        )
    records.sort(key=lambda record: str(record.get("name", "")).lower())
    return records


def _extract_scene_mesh_payload(
    mesh_obj,
    view_cfg,
    setup_frame,
    use_rest_pose,
    bone_name_to_index,
    mesh_reduction,
    mesh_target_vertices,
    weight_aware_decimation,
    base_color_texture_output,
):
    """Reduce, project and weight-transfer a single mesh object.

    Returns ``(mesh_data, mesh_reduction_report, source_weights)`` where
    ``source_weights`` is a per-2D-vertex list of {bone_index: weight} dicts.
    Used once per mesh object in the scene so each object becomes its own
    sprite/slot. ``base_color_texture_output`` is the PNG path the diffuse
    texture is written to; the payload only ever carries a path reference,
    never inline pixels.
    """
    mesh_reduction_report = reduce_mesh_object(
        mesh_obj,
        target_vertices=mesh_target_vertices,
        enabled=mesh_reduction,
        weight_aware_decimation=weight_aware_decimation,
    )

    mesh_data = extract_2d_mesh(
        mesh_obj,
        view_cfg,
        source_frame=setup_frame,
        use_rest_pose=use_rest_pose,
    )
    mesh_reduction_report["output_vertex_count"] = len(mesh_data.get("vertices_2d") or [])
    mesh_reduction_report["output_triangle_count"] = len(mesh_data.get("triangles") or [])
    if (
        mesh_reduction_report.get("applied")
        and int(mesh_reduction_report.get("target_vertices") or 0) > 0
        and int(mesh_reduction_report["output_vertex_count"])
        > int(mesh_reduction_report["target_vertices"])
        and mesh_reduction_report.get("reason") == "target_reached"
    ):
        mesh_reduction_report["reason"] = "target_reached_before_uv_split"
    mesh_data["mesh_reduction"] = mesh_reduction_report

    # Always write the diffuse/base-color texture to a separate PNG and emit a
    # path reference — never embed pixels inline. A full-resolution inline RGBA
    # array bloats the scene JSON to hundreds of MB (a 4K texture is ~67M ints)
    # and is pure overhead for consumers that only need the bone hierarchy
    # (e.g. the joint-mapping rig inspection, which silently failed when it had
    # to parse the giant payload). Native consumers load the PNG on demand.
    if base_color_texture_output:
        texture_data = save_base_color_texture(mesh_obj, base_color_texture_output)
        if texture_data:
            mesh_data["base_color_texture_path"] = texture_data["path"]
            mesh_data["base_color_texture_width"] = texture_data["width"]
            mesh_data["base_color_texture_height"] = texture_data["height"]
            mesh_data["base_color_texture_channels"] = texture_data["channels"]

    try:
        from flatrig.mesh import transfer_3d_weights_to_2d

        base_source_weights = [{} for _ in range(len(mesh_data.get("vertices_2d") or []))]
        if bone_name_to_index:
            import bpy
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = mesh_obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            try:
                base_source_weights = transfer_3d_weights_to_2d(eval_obj, eval_mesh, bone_name_to_index)
            finally:
                eval_obj.to_mesh_clear()

        source_indices = mesh_data.get("source_vertex_indices") or list(
            range(len(base_source_weights))
        )
        source_weights = [
            base_source_weights[int(source_index)]
            if 0 <= int(source_index) < len(base_source_weights)
            else {}
            for source_index in source_indices
        ]
    except ImportError:
        source_weights = [{} for _ in range(len(mesh_data.get("vertices_2d") or []))]

    return mesh_data, mesh_reduction_report, source_weights


def extract_scene_cli(
    source_path: str,
    output_path: str,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
    source_frame: int = None,
    use_rest_pose: bool = False,
    projection_space: str = "world",
    mesh_reduction: bool = True,
    mesh_target_vertices: int = 5000,
    weight_aware_decimation: bool = False,
    bind_from_animation: str = None,
    base_color_texture_output: str = None,
) -> dict[str, object]:
    """CLI wrapper for scene extraction (mesh + bones + weights).

    This combines extract_2d_mesh and extract_bone_hierarchy with weight transfer.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    all_meshes, armature_obj = find_all_meshes_and_armature()
    mesh_obj = all_meshes[0] if all_meshes else None
    # Animation-only sources (e.g. Mixamo training clips exported without a
    # skinned mesh) still carry a full armature. They cannot be built as 2D
    # sprite characters, but they ARE valid rigs to inspect for joint mapping,
    # so fall through with an empty mesh payload whenever an armature exists.
    # Only a scene with neither mesh nor armature is truly uninspectable.
    if mesh_obj is None and armature_obj is None:
        return {"ok": False, "detail": "No mesh or armature found in scene"}

    # Borrow the canonical setup pose from the donor animation whenever the
    # caller provides one. The donor contributes retargeted joint rotations;
    # setup extraction below keeps the target rig's own offsets/lengths.
    bind_borrow_info = _maybe_borrow_bind_from_animation(
        armature_obj,
        bind_from_animation,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
    )

    setup_frame = _resolve_setup_frame(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
        neutral_auto_pose=True,
    )
    bpy.context.scene.frame_set(setup_frame)
    # When the rig uses its OWN action, it can carry a constant root yaw vs the
    # normalized rest (root-bone roll convention differs from the donor's), so
    # remove it at the object level. When a bind pose was BORROWED instead, the
    # donor's world rotations already set the facing and the donor walk pose is
    # asymmetric, so the yaw inference would misfire — skip it then.
    if armature_obj is not None and not use_rest_pose:
        _align_root_unless_bind_borrowed(armature_obj, setup_frame, bind_borrow_info)
    setup_pose = _apply_auto_setup_pose(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
    )

    # Build view configuration
    view_cfg = get_scene_view_config(
        view_name=view_preset,
        view_dir=tuple(view_dir) if view_dir is not None else None,
        up_hint=tuple(view_up) if view_up is not None else None,
        roll_degrees=view_roll,
        armature_obj=armature_obj,
    )

    # Extract bone hierarchy
    bones = []
    if armature_obj is not None:
        bones = extract_setup_bone_hierarchy(
            armature_obj,
            view_cfg,
            source_frame=setup_frame,
            use_rest_pose=use_rest_pose,
            projection_space=projection_space,
            projection_reference_root=None,
            bind_borrow_info=bind_borrow_info,
        )
    bones_3d = extract_setup_bone_hierarchy_3d(
        armature_obj,
        source_frame=setup_frame,
        use_rest_pose=use_rest_pose,
        bind_borrow_info=bind_borrow_info,
    )

    # Re-bind offset (EXPERIMENTAL, OFF by default — opt-in via env).
    # A naive (rest_rotation − bind_rotation) offset was tried to re-base the
    # external matcher tracks onto the borrowed bind setup. It fixed ch36's arms-out bug but
    # REGRESSED Walk.fbx: both rigs share the same T-pose rest, so the geometric
    # offset is ~identical (~65° on arms), yet only ch36 needs it (Walk's
    # retarget is already correct). A purely geometric offset therefore cannot
    # discriminate the two. Left here off-by-default for experimentation; the
    # real fix must calibrate against the retargeted bind-donor clip (the
    # retargeted "Standard Walk" should match the bind at its bind frame; the
    # mismatch = the per-bone correction, ~0 for Walk, ~65° for ch36).
    # See doc/CH36_ARMS_INVESTIGATION.md.
    import os as _os

    if (
        _os.environ.get("FLATRIG_REBIND_OFFSET") == "1"
        and bind_borrow_info.get("applied")
        and not use_rest_pose
        and armature_obj is not None
        and bones
    ):
        rest_bones = extract_bone_hierarchy(
            armature_obj,
            view_cfg,
            use_rest_pose=True,
            projection_space=projection_space,
            projection_reference_root=None,
        )
        rest_rot_by_name = {b["name"]: b.get("rotation", 0.0) for b in rest_bones}
        for bone in bones:
            rest_rot = rest_rot_by_name.get(bone["name"])
            if rest_rot is not None:
                bone["rebind_offset"] = round(float(rest_rot) - float(bone.get("rotation", 0.0)), 4)
        # Restore the bind frame disturbed by the rest-pose extraction above
        # so the mesh / weight extraction below still sees the borrowed pose.
        bpy.context.scene.frame_set(setup_frame)
        bpy.context.view_layer.update()

    # Build the bone-name → index map once; every mesh shares the armature.
    bone_name_to_index = {
        str(bone["name"]): int(bone["index"]) for bone in bones if bone.get("name") is not None
    }

    # Extract every mesh object in the scene. A source FBX may carry several
    # meshes skinned to the same armature (e.g. a character body plus a
    # separate sword); each becomes its own entry so the C++ pipeline can turn
    # it into a dedicated sprite/slot. The primary mesh (most vertices) is also
    # mirrored into the legacy "mesh"/"source_weights"/"mesh_reduction" keys so
    # older single-mesh consumers keep working.
    meshes_payload = []
    primary_mesh_data = None
    primary_source_weights = None
    primary_reduction = None
    # Every mesh writes its base-color texture to its own PNG beside the output
    # JSON (never inline). Honour an explicit --base-color-texture-output for the
    # primary mesh; derive the rest so multi-object scenes (body + sword) each
    # get a distinct file.
    _output_path = Path(output_path)
    _texture_stem = _output_path.parent / _output_path.stem
    for mesh_index, scene_mesh in enumerate(all_meshes):
        is_primary = mesh_index == 0
        if is_primary and base_color_texture_output:
            mesh_texture_output = base_color_texture_output
        else:
            mesh_texture_output = f"{_texture_stem}_basecolor_{mesh_index}.png"
        mesh_data, mesh_reduction_report, source_weights = _extract_scene_mesh_payload(
            scene_mesh,
            view_cfg,
            setup_frame,
            use_rest_pose,
            bone_name_to_index,
            mesh_reduction,
            mesh_target_vertices,
            weight_aware_decimation,
            mesh_texture_output,
        )
        meshes_payload.append(
            {
                "object_name": scene_mesh.name,
                "is_primary": is_primary,
                "mesh": mesh_data,
                "source_weights": _weights_to_json(source_weights),
            }
        )
        if is_primary:
            primary_mesh_data = mesh_data
            primary_source_weights = source_weights
            primary_reduction = mesh_reduction_report

    return {
        "ok": True,
        "detail": "extracted",
        "source": source_path,
        "setup_frame": setup_frame,
        "setup_pose": setup_pose,
        "bind_borrow": bind_borrow_info,
        "use_rest_pose": bool(use_rest_pose),
        "view_config": _view_config_to_json(view_cfg),
        # Meshless (animation-only) rigs leave the primary mesh None; emit an
        # empty object so native consumers that read mesh sub-keys don't choke
        # on a null while inspection still gets the populated bone hierarchy.
        "mesh": primary_mesh_data if primary_mesh_data is not None else {},
        "meshes": meshes_payload,
        "bones": bones,
        "bones_3d": bones_3d,
        "source_weights": _weights_to_json(primary_source_weights or []),
        "mesh_reduction": primary_reduction,
    }


def extract_animations_cli(
    source_path: str,
    output_path: str,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
    source_frame: int = None,
    projection_space: str = "world",
    animation_names: list = None,
    fps: float = 30.0,
    frame_start: int = None,
    frame_end: int = None,
    sample_substeps: int = 2,
    optimize_animation_keys: bool = True,
    force_loop_closing_keys: bool = False,
    pose_mode: str = "full",
    pose_blend: float = 1.0,
    rotation_flatten: float = 0.0,
    rotation_flatten_scope: str = "all",
    rotation_flatten_bones: str = "",
    connected_translation_scope: str = "none",
    connected_translation_bones: str = "",
    stretch_guard_enabled: bool = False,
    stretch_guard_max_scale: float = 1.75,
    stretch_guard_strength: float = 0.65,
    stretch_guard_bones: str = "all",
    ik_leaf_refine_enabled: bool = False,
    ik_leaf_strength: float = 0.35,
    ik_leaf_iterations: int = 6,
    ik_leaf_max_chain_length: int = 3,
    ik_leaf_preserve_scale: float = 0.65,
    drop_problematic_frames: bool = False,
    preserve_root_motion: bool = False,
    preserve_root_rotation: bool = False,
    bind_from_animation: str = None,
    animation_source: str = None,
    decouple_scale: bool = False,
) -> dict[str, object]:
    """CLI wrapper for animation extraction.

    This extracts animations using the bone hierarchy and animation functions.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    mesh_obj, armature_obj = find_mesh_and_armature()
    if armature_obj is None:
        return {"ok": False, "detail": "No armature found in scene"}

    # Retargeting external motion onto this model: wipe the model's own
    # embedded animation first so it can't drive the armature, perturb action
    # naming, or leak into the output as a junk clip. Only when we are about
    # to transfer an external source (bind_from_animation set); the source-
    # model extraction path below needs the model's own action and skips this.
    if bind_from_animation:
        purge_model_animations(armature_obj)

    bind_borrow_info = _maybe_borrow_bind_from_animation(
        armature_obj,
        bind_from_animation,
        source_frame=source_frame,
        use_rest_pose=False,
    )

    setup_frame = _resolve_setup_frame(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=False,
    )
    # Some clips are authored with a constant root-bone yaw in the action, so
    # the animated pose faces a different way than the normalized rest. Align
    # the animation's root yaw (at the setup frame) to the rest's so the clip
    # plays facing the canonical direction and the fixed-view projection reads
    # the correct side of the body. Rig-agnostic: topmost bone only. Skipped
    # when a bind pose was borrowed (the donor pose already fixes facing and
    # its asymmetric stride would mislead the yaw inference).
    _align_root_unless_bind_borrowed(armature_obj, setup_frame, bind_borrow_info)
    # Build view configuration
    view_cfg = get_scene_view_config(
        view_name=view_preset,
        view_dir=tuple(view_dir) if view_dir is not None else None,
        up_hint=tuple(view_up) if view_up is not None else None,
        roll_degrees=view_roll,
        armature_obj=armature_obj,
    )

    # Extract bone hierarchy
    bones = extract_setup_bone_hierarchy(
        armature_obj,
        view_cfg,
        source_frame=setup_frame,
        use_rest_pose=False,
        projection_space=projection_space,
        projection_reference_root=None,
        bind_borrow_info=bind_borrow_info,
    )

    # When bind_from_animation is set, the imported source armature was
    # deleted by _maybe_borrow_bind_from_animation after copying its first
    # frame. We need to re-import the animation source to get the full
    # animation data and apply it to the target armature frame by frame.
    # If animation_source is explicitly provided, use it as the source
    # armature (keeping bind_from_animation as the shared bind pose donor).
    source_arm_obj = None
    source_import_path = animation_source or bind_from_animation
    if source_import_path:
        objects_before = {obj.name for obj in bpy.data.objects}
        import_model(str(Path(source_import_path).expanduser()))
        imported = [
            obj
            for obj in bpy.data.objects
            if obj.type == "ARMATURE" and obj.name not in objects_before
        ]
        if imported:
            source_arm_obj = imported[0]

    # Get animation extraction function if available
    try:
        from flatrig.animation import extract_bone_animations

        if source_arm_obj is not None:
            # 3D retarget: apply source rotations to target armature at each
            # frame, then project target to 2D. This produces deltas in the
            # target's 2D space, consistent with extract-scene's setup.
            animations = _extract_transferred_animation(
                target_arm=armature_obj,
                source_arm=source_arm_obj,
                bones_setup=bones,
                view_cfg=view_cfg,
                setup_frame=setup_frame,
                fps=fps,
                frame_start=frame_start,
                frame_end=frame_end,
                sample_substeps=sample_substeps,
                optimize_animation_keys=optimize_animation_keys,
                force_loop_closing_keys=force_loop_closing_keys,
                projection_space=projection_space,
                pose_mode=pose_mode,
                pose_blend=pose_blend,
                rotation_flatten=rotation_flatten,
                rotation_flatten_scope=rotation_flatten_scope,
                rotation_flatten_bones=rotation_flatten_bones,
                connected_translation_scope=connected_translation_scope,
                connected_translation_bones=connected_translation_bones,
                stretch_guard_enabled=stretch_guard_enabled,
                stretch_guard_max_scale=stretch_guard_max_scale,
                stretch_guard_strength=stretch_guard_strength,
                stretch_guard_bones=stretch_guard_bones,
                ik_leaf_refine_enabled=ik_leaf_refine_enabled,
                ik_leaf_strength=ik_leaf_strength,
                ik_leaf_iterations=ik_leaf_iterations,
                ik_leaf_max_chain_length=ik_leaf_max_chain_length,
                ik_leaf_preserve_scale=ik_leaf_preserve_scale,
                drop_problematic_frames=drop_problematic_frames,
                preserve_root_motion=preserve_root_motion,
                preserve_root_rotation=preserve_root_rotation,
                animation_names=animation_names,
                decouple_scale=decouple_scale,
            )
        else:
            animations = extract_bone_animations(
                armature_obj,
                bones,
                view_cfg,
                fps=fps,
                frame_start=frame_start,
                frame_end=frame_end,
                sample_substeps=sample_substeps,
                optimize_animation_keys=optimize_animation_keys,
                force_loop_closing_keys=force_loop_closing_keys,
                projection_space=projection_space,
                pose_mode=pose_mode,
                pose_blend=pose_blend,
                rotation_flatten={
                    "amount": rotation_flatten,
                    "scope": rotation_flatten_scope,
                    "bones": rotation_flatten_bones,
                }
                if rotation_flatten > 0
                else None,
                connected_translation={
                    "scope": connected_translation_scope,
                    "bones": connected_translation_bones,
                }
                if connected_translation_scope != "none"
                else None,
                stretch_guard={
                    "enabled": True,
                    "max_scale": stretch_guard_max_scale,
                    "strength": stretch_guard_strength,
                    "bones": stretch_guard_bones,
                }
                if stretch_guard_enabled
                else None,
                leaf_ik_refine={
                    "enabled": True,
                    "strength": ik_leaf_strength,
                    "iterations": ik_leaf_iterations,
                    "max_chain_length": ik_leaf_max_chain_length,
                    "preserve_scale": ik_leaf_preserve_scale,
                }
                if ik_leaf_refine_enabled
                else None,
                problem_frame_filter={"enabled": True} if drop_problematic_frames else None,
                projection_reference_root=None,
                preserve_root_motion=preserve_root_motion,
                preserve_root_rotation=preserve_root_rotation,
                action_names=animation_names or [],
            )
    except ImportError as e:
        return {"ok": False, "detail": f"Animation extraction not available: {e}"}

    serialized_animations = _serialize_animations(animations)
    # Preview entries must follow the serialized animation order so native
    # callers can match them positionally as well as by name.
    preview_3d_animations = []
    for record in serialized_animations:
        payload = animations.get(record["name"]) or {}
        preview = payload.get("preview_3d")
        if preview:
            preview_3d_animations.append(preview)

    return {
        "ok": True,
        "detail": "extracted",
        "source": source_path,
        "setup_frame": setup_frame,
        "bones": bones,
        # Name-agnostic deform skeleton so rig-only consumers (the pseudo-2D
        # animation library) can drop control-rig / auxiliary joints from the
        # preview. Empty when the clip carries no skin weights.
        "deform_bone_names": sorted(
            _deform_bone_names(armature_obj, [mesh_obj] if mesh_obj is not None else None)
        ),
        "animations": serialized_animations,
        "preview_3d_animations": preview_3d_animations,
    }


def dump_rig_animation_cli(
    source_path: str,
    output_path: str,
    *,
    animation_names: list = None,
    frame_start: int = None,
    frame_end: int = None,
) -> dict[str, object]:
    """Dump rig topology + per-frame pose matrices for one action as JSON.

    Generic rig I/O for external animation processors: rest hierarchy
    (matrix_local, parent, length), the armature object's world matrix, and
    per sampled frame each pose bone's world matrix and local matrix_basis.
    No projection, no view, no interpretation — callers own the math and
    send the result back through bake-rig-animation.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    _mesh_obj, armature_obj = find_mesh_and_armature()
    if armature_obj is None:
        return {"ok": False, "detail": "No armature found in scene"}

    action = _resolve_action_for_export(armature_obj, animation_names)
    if action is None:
        return {"ok": False, "detail": "No animation found in source"}
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    armature_obj.animation_data.action = action

    start, end = action.frame_range
    first = int(frame_start) if frame_start is not None else int(round(float(start)))
    last = int(frame_end) if frame_end is not None else int(round(float(end)))
    if last < first:
        first, last = last, first

    # Same canonicalization as the extraction paths: clips authored with a
    # constant root yaw vs the normalized rest get re-aligned so the dump is
    # expressed in the canonical facing.
    align_animation_root_to_rest(armature_obj, first)

    bone_names = _topological_sort(armature_obj)
    data_bones = armature_obj.data.bones
    bones_payload = []
    for name in bone_names:
        bone = data_bones[name]
        bones_payload.append(
            {
                "name": name,
                "parent": bone.parent.name if bone.parent is not None else None,
                "matrix_local": _matrix4_to_json(bone.matrix_local),
                "length": float(bone.length),
            }
        )

    scene = bpy.context.scene
    pose_bones = armature_obj.pose.bones
    frames_payload = []
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        world_obj = armature_obj.matrix_world
        per_bone = {}
        for name in bone_names:
            pose_bone = pose_bones[name]
            per_bone[name] = {
                "world": _matrix4_to_json(world_obj @ pose_bone.matrix),
                "basis": _matrix4_to_json(pose_bone.matrix_basis),
            }
        frames_payload.append({"frame": frame, "bones": per_bone})

    fps_base = float(getattr(scene.render, "fps_base", 1.0) or 1.0)
    return {
        "ok": True,
        "detail": "dumped",
        "source": source_path,
        "animation": str(action.name),
        "frame_start": first,
        "frame_end": last,
        "fps": float(scene.render.fps) / fps_base,
        "armature_matrix_world": _matrix4_to_json(armature_obj.matrix_world),
        "bones": bones_payload,
        "frames": frames_payload,
    }


def bake_rig_animation_cli(
    source_path: str,
    output_path: str,
    *,
    bake_spec: str,
    flat_output: str,
) -> dict[str, object]:
    """Re-bake externally computed local pose transforms and export the rig.

    `bake_spec` is a JSON file with the action name, the sampled frames, and
    per bone columnar arrays of location / rotation_quaternion (w,x,y,z) /
    scale in matrix_basis space. The source model is re-imported, its own
    animation purged, the spec keyframed onto the skeleton, and the scene
    exported to `flat_output` (.fbx or .glb).
    """
    spec_path = Path(bake_spec).expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    frames = [int(value) for value in spec.get("frames") or []]
    if not frames:
        return {"ok": False, "detail": "bake spec has no frames"}
    spec_bones = spec.get("bones") or {}
    if not spec_bones:
        return {"ok": False, "detail": "bake spec has no bones"}

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    _mesh_obj, armature_obj = find_mesh_and_armature()
    if armature_obj is None:
        return {"ok": False, "detail": "No armature found in scene"}

    # Re-apply the same root-yaw canonicalization the dump used so the baked
    # local transforms land in the same object frame they were computed in.
    if armature_obj.animation_data is not None and armature_obj.animation_data.action:
        align_animation_root_to_rest(armature_obj, frames[0])
    purge_model_animations(armature_obj)

    pose_bones = armature_obj.pose.bones
    missing_bones = [name for name in spec_bones if name not in pose_bones]
    baked_names = [name for name in spec_bones if name in pose_bones]
    for name in baked_names:
        pose_bones[name].rotation_mode = "QUATERNION"

    for index, frame in enumerate(frames):
        for name in baked_names:
            tracks = spec_bones[name]
            pose_bone = pose_bones[name]
            pose_bone.location = tracks["location"][index]
            pose_bone.rotation_quaternion = tracks["rotation_quaternion"][index]
            pose_bone.scale = tracks["scale"][index]
            pose_bone.keyframe_insert("location", frame=frame)
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame)
            pose_bone.keyframe_insert("scale", frame=frame)

    new_action = armature_obj.animation_data.action if armature_obj.animation_data else None
    if new_action is not None and spec.get("animation"):
        # The FBX exporter writes one take per action named after the action,
        # and the importer decorates takes as "<Object>|<Take>|Layer<N>".
        # Strip those decorations so a re-imported bake yields the SAME
        # action name as the original file (clip-name lookups keep working).
        take_name = str(spec["animation"])
        object_prefix = f"{armature_obj.name}|"
        while take_name.startswith(object_prefix):
            take_name = take_name[len(object_prefix) :]
        take_name = re.sub(r"\|Layer\d+$", "", take_name)
        new_action.name = take_name or str(spec["animation"])

    scene = bpy.context.scene
    scene.frame_start = frames[0]
    scene.frame_end = frames[-1]

    # FBX exports the armature's CURRENT pose as the bind/rest transform while
    # also exporting actions as takes. Leaving the newly baked action active at
    # its first frame therefore bakes frame 1 into the FBX rest pose and the
    # re-imported action comes back nearly identity (the apparent "root/rest is
    # tilted" regression). Keep the action in bpy.data for all-actions export,
    # but detach it and reset the evaluated pose to the canonical rest before
    # writing the file.
    if armature_obj.animation_data is not None:
        armature_obj.animation_data.action = None
    for pose_bone in armature_obj.pose.bones:
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)
    scene.frame_set(frames[0])
    bpy.context.view_layer.update()

    flat_path = Path(flat_output).expanduser().resolve()
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = flat_path.suffix.lower()
    if suffix == ".fbx":
        bpy.ops.export_scene.fbx(
            filepath=str(flat_path),
            use_selection=False,
            object_types={"ARMATURE", "MESH"},
            add_leaf_bones=False,
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=False,
            bake_space_transform=True,
            # One take per action, named after the action. With the scene
            # bake (all_actions=False) the take is always called "Scene", so
            # the re-imported clip loses its name and downstream
            # animation-name lookups (e.g. retargeting the baked file by the
            # original clip name) stop resolving.
            bake_anim_use_all_actions=True,
            bake_anim_force_startend_keying=True,
            bake_anim_simplify_factor=0.0,
        )
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.export_scene.gltf(
            filepath=str(flat_path),
            export_format="GLB",
            export_yup=True,
            export_animations=True,
            export_skins=True,
        )
    else:
        return {
            "ok": False,
            "detail": f"Unsupported bake output format '{suffix}'. Use .fbx or .glb.",
        }

    return {
        "ok": True,
        "detail": "baked",
        "source": source_path,
        "flat_output": str(flat_path),
        "animation": str(new_action.name) if new_action is not None else None,
        "frame_start": frames[0],
        "frame_end": frames[-1],
        "baked_bone_count": len(baked_names),
        "missing_bones": missing_bones,
    }


def _extract_transferred_animation(
    target_arm,
    source_arm,
    bones_setup,
    view_cfg,
    setup_frame=None,
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
    rotation_flatten_scope=None,
    rotation_flatten_bones="",
    connected_translation_scope="none",
    connected_translation_bones="",
    stretch_guard_enabled=False,
    stretch_guard_max_scale=1.75,
    stretch_guard_strength=0.65,
    stretch_guard_bones="all",
    ik_leaf_refine_enabled=False,
    ik_leaf_strength=0.35,
    ik_leaf_iterations=6,
    ik_leaf_max_chain_length=3,
    ik_leaf_preserve_scale=0.65,
    drop_problematic_frames=False,
    preserve_root_motion=False,
    preserve_root_rotation=False,
    animation_names=None,
    decouple_scale=False,
):
    """Transfer 3D rotations from source armature to target armature frame
    by frame, then project the target to 2D. This ensures 2D rotation
    deltas are computed in the target's projection space, consistent
    with extract-scene's bone hierarchy setup.
    """
    import math as _math
    import numpy as np

    from flatrig.animation import (
        _compute_frame_local_bone_poses_2d,
        _sample_projected_bone_segments_2d,
        _unwrap_angle_near,
        _optimize_keyframes,
        _stabilize_frame_local_poses_2d,
        _collect_problem_frame_metrics,
        _evaluate_problem_frame_sample,
        _select_problem_frame_samples,
        _prepare_rotation_flatten,
        _prepare_leaf_ik_refine,
        _prepare_problem_frame_filter,
        _build_leaf_ik_chains,
        _build_local_rotation_reference,
        _camera_parallel_bone_names,
        _refine_frame_local_poses_with_leaf_ik,
        _is_rotation_pose_mode,
    )

    # Build bone name mapping (source → target) — the same flat stem map the
    # bind borrow uses, so bind pose and transferred frames can never pair
    # bones differently.
    bone_map = _stem_pose_bone_map(source_arm, target_arm)
    use_local_pose_transfer = _can_use_same_rig_local_pose_transfer(
        source_arm, target_arm, bone_map
    )
    if use_local_pose_transfer:
        print(
            "[blender_scene_io] direct same-rig local transfer enabled "
            f"({len(bone_map)} mapped pose bone(s)); preserving source local channels."
        )
    # The target topology + rest rotations are constant across frames; derive
    # them once so the per-frame world-rotation copy stays cheap.
    fk_cache = build_target_fk_cache(target_arm)

    # Get source action and frame range
    src_action = source_arm.animation_data.action if source_arm.animation_data else None
    if src_action is None:
        return {}

    scene = bpy.context.scene
    source_arm.animation_data.action = src_action

    action_start = int(round(src_action.frame_range[0]))
    action_end = int(round(src_action.frame_range[1]))
    if frame_start is None:
        frame_start = action_start
    if frame_end is None:
        frame_end = action_end
    fps = max(1.0, fps)
    total_duration = max(0.0, (frame_end - frame_start) / fps)
    rotation_flatten_config = _prepare_rotation_flatten(
        {
            "amount": rotation_flatten or 0.0,
            "scope": rotation_flatten_scope or "all",
            "bones": rotation_flatten_bones or "",
        }
    )
    leaf_ik_refine = _prepare_leaf_ik_refine(
        {
            "enabled": ik_leaf_refine_enabled,
            "strength": ik_leaf_strength,
            "iterations": ik_leaf_iterations,
            "max_chain_length": ik_leaf_max_chain_length,
            "preserve_scale": ik_leaf_preserve_scale,
        }
    )
    leaf_ik_chains = _build_leaf_ik_chains(bones_setup, leaf_ik_refine)

    # Build sample times
    sample_points = []
    substeps = max(1, sample_substeps)
    for frame in range(frame_start, frame_end):
        for substep in range(substeps):
            sample_points.append(
                (frame, substep / substeps, (frame - frame_start + substep / substeps) / fps)
            )
    if sample_points:
        sample_points.append((frame_end, 0.0, total_duration))

    # Compute bone setup poses
    scene.frame_set(setup_frame if setup_frame is not None else frame_start)
    bpy.context.view_layer.update()
    setup_segments = _sample_projected_bone_segments_2d(target_arm, bones_setup, view_cfg)
    local_rotation_reference = None
    if pose_mode == "local_rotation":
        local_rotation_reference = _build_local_rotation_reference(
            target_arm,
            bones_setup,
            view_cfg,
        )
    frozen_bone_names = _camera_parallel_bone_names(target_arm, bones_setup)
    setup_poses = _compute_frame_local_bone_poses_2d(
        target_arm,
        bones_setup,
        view_cfg,
        projected_segments=setup_segments,
        pose_mode=pose_mode,
        rotation_flatten=rotation_flatten_config,
        local_rotation_reference=local_rotation_reference,
        decouple_scale=decouple_scale,
        frozen_bone_names=frozen_bone_names,
    )

    # Build stretch guard config
    stretch_guard = (
        {
            "enabled": stretch_guard_enabled,
            "max_scale": stretch_guard_max_scale,
            "strength": stretch_guard_strength,
            "bones": stretch_guard_bones,
        }
        if stretch_guard_enabled
        else None
    )

    # Build problem frame filter config
    problem_frame_filter = _prepare_problem_frame_filter(
        {"enabled": True} if drop_problematic_frames else None
    )

    # Extract animation with stabilization and filtering
    bone_timelines = {}
    for bone_info in bones_setup:
        bone_timelines[bone_info["name"]] = {"rotate": [], "translate": [], "scale": []}

    previous_rotation = {}
    previous_stable_poses = None
    previous_leaf_ik_poses = None
    previous_sample_metrics = None
    sample_records = []

    for frame, subframe, time in sample_points:
        # Pose source armature at this frame
        scene.frame_set(frame, subframe=subframe)
        bpy.context.view_layer.update()

        if use_local_pose_transfer:
            _copy_source_local_pose_to_target(source_arm, target_arm, bone_map)
        else:
            # Cross-rig transfer: match source bone directions while preserving
            # the target roll/morphology. This intentionally does not copy
            # donor/source translations or scales as anatomy compensation.
            _copy_source_pose_to_target(
                source_arm,
                target_arm,
                bone_map,
                copy_root_location=preserve_root_motion,
                fk_cache=fk_cache,
            )

        bpy.context.view_layer.update()

        # Sample 2D from TARGET armature
        segments = _sample_projected_bone_segments_2d(target_arm, bones_setup, view_cfg)
        raw_poses = _compute_frame_local_bone_poses_2d(
            target_arm,
            bones_setup,
            view_cfg,
            projected_segments=segments,
            pose_mode=pose_mode,
            rotation_flatten=rotation_flatten_config,
            local_rotation_reference=local_rotation_reference,
            decouple_scale=decouple_scale,
            frozen_bone_names=frozen_bone_names,
        )

        # Stabilize frame poses (prevents short-bone flipping)
        if previous_stable_poses is not None:
            frame_poses = _stabilize_frame_local_poses_2d(
                raw_poses, previous_stable_poses, bones_setup, stretch_guard=stretch_guard
            )
        else:
            frame_poses = raw_poses
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
                segments,
                leaf_ik_chains,
                leaf_ik_refine,
            )
            previous_leaf_ik_poses = {name: pose.copy() for name, pose in frame_poses.items()}
        elif leaf_ik_refine.get("enabled", False):
            previous_leaf_ik_poses = {name: pose.copy() for name, pose in frame_poses.items()}

        # Collect problem frame metrics
        current_metrics = _collect_problem_frame_metrics(bones_setup, segments, frame_poses)
        frame_filter = _evaluate_problem_frame_sample(
            current_metrics,
            previous_sample_metrics,
            problem_frame_filter,
            time_value=time,
            fps=fps,
        )

        # World-space 3D bone snapshot for this sample (same fields as
        # extract_bone_hierarchy_3d). Consumed by the native optimizer's
        # skinning3d targets and the animation-aware draw order; without it
        # the direct-extraction path emitted no preview_3d_animations and
        # both consumers silently fell back (self-consistent targets /
        # static draw order).
        armature_world = target_arm.matrix_world
        bones_3d = []
        for bone_info in bones_setup:
            pose_bone = target_arm.pose.bones.get(bone_info["name"])
            if pose_bone is None:
                continue
            bones_3d.append(
                {
                    "name": bone_info["name"],
                    "head": _vector_to_json(armature_world @ pose_bone.head),
                    "tail": _vector_to_json(armature_world @ pose_bone.tail),
                    "rotation": _matrix3_to_json(
                        _world_rotation_3x3(armature_world @ pose_bone.matrix)
                    ),
                }
            )

        sample_records.append(
            {
                "time": time,
                "frame_poses": frame_poses,
                "root_motion_offset": (0.0, 0.0),
                "root_rotation_offset": 0.0,
                "frame_filter": frame_filter,
                "bones_3d": bones_3d,
            }
        )
        previous_sample_metrics = {"time": time, "bones": current_metrics}

    # Filter problem frames
    total_duration = max(0.0, (frame_end - frame_start) / fps)
    sample_records, _filter_summary = _select_problem_frame_samples(
        sample_records, total_duration, problem_frame_filter
    )

    # Build bone timelines from filtered samples
    for rec in sample_records:
        time = float(rec["time"])
        frame_poses = rec["frame_poses"]

        for bone_info in bones_setup:
            name = bone_info["name"]
            if name not in frame_poses or name not in setup_poses:
                continue
            current = frame_poses[name]
            setup = setup_poses[name]

            raw_rotation = _normalize_angle(current["rotation"] - setup["rotation"])
            rel_rotation = _unwrap_angle_near(raw_rotation, previous_rotation.get(name))
            previous_rotation[name] = rel_rotation

            bone_timelines[name]["rotate"].append(
                {
                    "time": round(time, 4),
                    "angle": round(rel_rotation, 2),
                    "value": round(rel_rotation, 2),
                }
            )
            bone_timelines[name]["translate"].append(
                {
                    "time": round(time, 4),
                    "x": round(current["x"] - setup.get("x", 0.0), 4),
                    "y": round(current["y"] - setup.get("y", 0.0), 4),
                }
            )
            # Emit scale.x as a FACTOR relative to the bone's setup scale_x,
            # matching the canonical extract_bone_animations path
            # (flatrig/animation.py: local_scale_x / base_scale_x). Emitting
            # the absolute local scale_x here double-counts the setup scale
            # whenever the rig bone's setup scale_x != 1.0, which is exactly
            # the kind of scale_x divergence vs the Python reference that
            # shows up as stretched limbs.
            base_scale_x = max(abs(float(bone_info.get("scale_x", 1.0))), 1e-8)
            base_scale_y = max(abs(float(bone_info.get("scale_y", 1.0))), 1e-8)
            bone_timelines[name]["scale"].append(
                {
                    "time": round(time, 4),
                    "x": round(current.get("scale_x", 1.0) / base_scale_x, 4),
                    "y": (
                        round(current.get("scale_y", 1.0) / base_scale_y, 4)
                        if decouple_scale
                        else 1.0
                    ),
                }
            )
            if decouple_scale:
                # shear_y cancels the parent's non-uniform skew so the chain
                # stays orthogonal (no "underwater" ripple). shear_x stays 0.
                base_shear_y = float(bone_info.get("shear_y", 0.0))
                bone_timelines[name].setdefault("shear", []).append(
                    {
                        "time": round(time, 4),
                        "x": 0.0,
                        "y": round(current.get("shear_y", 0.0) - base_shear_y, 4),
                    }
                )

    if optimize_animation_keys:
        for name, tl in list(bone_timelines.items()):
            bone_timelines[name] = _optimize_keyframes(tl)

    anim_name = animation_names[0] if animation_names else src_action.name
    # Per-sample 3D bone snapshots (post problem-frame filter) in the same
    # schema the retarget path emits (preview_3d_animations entries):
    # {"name", "frames": [{"time", "bones": [{"name","head","tail","rotation"}]}]}
    preview_frames = [
        {"time": round(float(rec["time"]), 4), "bones": rec.get("bones_3d") or []}
        for rec in sample_records
    ]
    return {
        anim_name: {
            "bones": bone_timelines,
            "frame_filter": {},
            "preview_3d": {"name": anim_name, "frames": preview_frames},
            "source_authored_transform_keys": bool(use_local_pose_transfer),
            "transfer_mode": "same_rig_local" if use_local_pose_transfer else "direction_retarget",
        }
    }


def _split_sprite_render_manifest(entries):
    """Separate the optional full-scene reference from legacy part entries.

    Part entries intentionally need no ``kind`` field so manifests produced by
    older FlatRig builds remain valid. A reference is returned separately and
    must never contribute to the ``renders`` list: native callers use that
    list's length as the part-count compatibility contract.
    """
    if not isinstance(entries, list):
        raise ValueError("Sprite render manifest must be a JSON array")

    parts = []
    reference = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Sprite render manifest entry {index} must be an object")
        kind = str(entry.get("kind") or "part")
        if kind == "reference":
            if reference is not None:
                raise ValueError("Sprite render manifest may contain only one reference entry")
            if not entry.get("output_path"):
                raise ValueError("Sprite reference entry requires output_path")
            frame = entry.get("projection_frame")
            if not isinstance(frame, dict):
                raise ValueError("Sprite reference entry requires projection_frame")
            missing = [key for key in ("center_x", "center_y", "span") if key not in frame]
            if missing:
                raise ValueError(
                    "Sprite reference projection_frame is missing: " + ", ".join(missing)
                )
            reference = entry
            continue
        if kind != "part":
            raise ValueError(f"Unknown sprite render manifest kind: {kind}")
        parts.append(entry)
    return parts, reference


def _resolve_reference_triangle_groups(reference, mesh_by_name, primary_mesh):
    """Resolve the optional exported-core filter without legacy regressions.

    A reference entry with neither filter field is the historical full-scene
    request and returns ``None``. New entries name the exact source-object /
    source-triangle union to render. Missing named objects are an error instead
    of falling back to the primary mesh, which could expose unrelated geometry.
    """
    filter_name = reference.get("triangle_filter")
    raw_groups = reference.get("triangle_groups")
    if filter_name is None and raw_groups is None:
        return None
    if filter_name != "exported_core_union":
        raise ValueError(f"Unsupported sprite reference triangle_filter: {filter_name!r}")
    if not isinstance(raw_groups, list):
        raise ValueError("Filtered sprite reference requires triangle_groups array")

    keys_by_name = {}
    for index, group in enumerate(raw_groups):
        if not isinstance(group, dict):
            raise ValueError(f"Sprite reference triangle group {index} must be an object")
        object_name = str(group.get("object_name") or "")
        raw_keys = group.get("triangle_keys")
        if not isinstance(raw_keys, list):
            raise ValueError(
                f"Sprite reference triangle group {index} requires triangle_keys array"
            )
        keys = keys_by_name.setdefault(object_name, set())
        for key_index, raw_key in enumerate(raw_keys):
            if (
                not isinstance(raw_key, (list, tuple))
                or len(raw_key) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_key)
            ):
                raise ValueError(
                    f"Sprite reference triangle group {index} key {key_index} "
                    "must contain three integer vertex indices"
                )
            keys.add(tuple(sorted(int(value) for value in raw_key)))

    resolved = []
    for object_name in sorted(keys_by_name):
        triangle_keys = sorted(keys_by_name[object_name])
        if not triangle_keys:
            continue
        source_obj = mesh_by_name.get(object_name) if object_name else primary_mesh
        if source_obj is None:
            label = object_name or "<primary mesh>"
            raise ValueError(f"Sprite reference object not found: {label}")
        resolved.append(
            {
                "object_name": object_name,
                "object": source_obj,
                "triangle_keys": triangle_keys,
            }
        )
    return resolved


def _applied_reference_triangle_filter(reference, triangle_groups):
    """Return the filter ACK only after resolution selected the filtered path."""
    if triangle_groups is None:
        return None
    return str(reference.get("triangle_filter"))


def render_sprites_cli(
    source_path: str,
    output_path: str,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
    source_frame: int = None,
    use_rest_pose: bool = False,
    projection_space: str = "world",
    parts_json: str = None,
    images_dir: str = None,
    resolution: int = 2048,
    bind_frame: int = 0,
    mesh_reduction: bool = True,
    mesh_target_vertices: int = 5000,
    weight_aware_decimation: bool = False,
    bind_from_animation: str = None,
) -> dict[str, object]:
    """CLI wrapper for sprite rendering.

    This renders part sprites using the projection and sprite functions.
    """
    import json as json_module

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    all_meshes, armature_obj = find_all_meshes_and_armature()
    mesh_obj = all_meshes[0] if all_meshes else None
    if mesh_obj is None:
        return {"ok": False, "detail": "No mesh found in scene"}

    bind_borrow_info = _maybe_borrow_bind_from_animation(
        armature_obj,
        bind_from_animation,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
    )

    # Reduce every mesh with the same parameters extract-scene used so each
    # object's triangle_keys line up with the geometry rendered here. Index the
    # meshes by object name so a part can be rendered against its owning mesh
    # (the primary body, a sword, etc.).
    mesh_by_name = {}
    mesh_reduction_report = None
    for mesh_index, scene_mesh in enumerate(all_meshes):
        report = reduce_mesh_object(
            scene_mesh,
            target_vertices=mesh_target_vertices,
            enabled=mesh_reduction,
            weight_aware_decimation=weight_aware_decimation,
        )
        mesh_by_name[scene_mesh.name] = scene_mesh
        if mesh_index == 0:
            mesh_reduction_report = report

    if not parts_json or not images_dir:
        return {"ok": False, "detail": "parts-json and images-dir are required"}

    parts_path = Path(parts_json).expanduser().resolve()
    output_dir = Path(images_dir).expanduser().resolve()

    if not parts_path.exists():
        return {"ok": False, "detail": f"parts-json not found: {parts_path}"}

    manifest = json_module.loads(parts_path.read_text(encoding="utf-8"))
    parts, reference_request = _split_sprite_render_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    if bind_frame > 0:
        render_frame = int(bind_frame)
        setup_pose = {"mode": "frame", "posed_bone_count": 0}
    else:
        render_frame = _resolve_setup_frame(
            armature_obj,
            source_frame=(source_frame if bind_frame == 0 else -1),
            use_rest_pose=use_rest_pose,
            neutral_auto_pose=True,
        )
        setup_pose = None
    bpy.context.scene.frame_set(render_frame)
    if setup_pose is None:
        setup_pose = _apply_auto_setup_pose(
            armature_obj,
            source_frame=(source_frame if bind_frame == 0 else -1),
            use_rest_pose=use_rest_pose,
        )
    # Same constant-root-yaw removal as extract_scene_cli so the rendered
    # sprites face the same way as the extracted bind skeleton (skipped when a
    # bind pose was borrowed — the donor pose already fixes the facing).
    if armature_obj is not None and not use_rest_pose:
        _align_root_unless_bind_borrowed(armature_obj, render_frame, bind_borrow_info)
    bpy.context.view_layer.update()

    # Build view configuration
    view_cfg = get_scene_view_config(
        view_name=view_preset,
        view_dir=tuple(view_dir) if view_dir is not None else None,
        up_hint=tuple(view_up) if view_up is not None else None,
        roll_degrees=view_roll,
        armature_obj=armature_obj,
    )

    # Get projection reference
    projection_reference_root = None
    if armature_obj:
        world_matrix = armature_obj.matrix_world
        projection_reference_root = [
            [world_matrix[0][0], world_matrix[0][1], world_matrix[0][2], world_matrix[0][3]],
            [world_matrix[1][0], world_matrix[1][1], world_matrix[1][2], world_matrix[1][3]],
            [world_matrix[2][0], world_matrix[2][1], world_matrix[2][2], world_matrix[2][3]],
            [world_matrix[3][0], world_matrix[3][1], world_matrix[3][2], world_matrix[3][3]],
        ]

    # Get render_part_sprite function if available
    try:
        from flatrig.projection import get_projection_reference_matrix
        from flatrig.texture import render_part_sprite, render_preview_sprite

        projection_matrix = (
            get_projection_reference_matrix(
                armature_obj,
                projection_space=projection_space,
                use_rest_pose=use_rest_pose,
                reference_root_matrix=projection_reference_root,
            )
            if armature_obj
            else None
        )

        reference_result = None
        if reference_request is not None:
            requested_path = Path(str(reference_request["output_path"])).expanduser()
            reference_output = (
                requested_path.resolve()
                if requested_path.is_absolute()
                else (output_dir / requested_path).resolve()
            )
            reference_output.parent.mkdir(parents=True, exist_ok=True)
            reference_triangle_groups = _resolve_reference_triangle_groups(
                reference_request,
                mesh_by_name,
                mesh_obj,
            )
            rest_pose_state = []
            try:
                if use_rest_pose:
                    rest_pose_state = _set_scene_armatures_rest_pose(bpy.context.scene)
                reference_ok = render_preview_sprite(
                    mesh_obj,
                    view_cfg,
                    dict(reference_request.get("projection_frame") or {}),
                    str(reference_output),
                    resolution=resolution,
                    depth_center=float(reference_request.get("depth_center", 0.0) or 0.0),
                    bind_frame=render_frame,
                    projection_matrix=projection_matrix,
                    triangle_groups=reference_triangle_groups,
                )
            finally:
                if rest_pose_state:
                    _restore_scene_armature_pose_positions(rest_pose_state)
            reference_result = {
                "name": str(
                    reference_request.get("name") or "__flatrig_sprite_reference__"
                ),
                "kind": "reference",
                "ok": bool(reference_ok),
                "detail": "rendered" if reference_ok else "render failed",
                "output_path": str(reference_output),
                "width": resolution,
                "height": resolution,
                # This is deliberately an applied-filter acknowledgement, not
                # an echo of the request.  Native callers use it to reject an
                # older sidecar that silently renders the historical full
                # scene while receiving a newer filtered manifest.
                "applied_triangle_filter": _applied_reference_triangle_filter(
                    reference_request,
                    reference_triangle_groups,
                ),
            }

        renders = []
        for part in parts:
            attachment_name = str(part.get("attachment_name") or part.get("name") or "part")
            part_output = output_dir / f"{attachment_name}.png"
            # Render each part against its owning mesh object; fall back to the
            # primary mesh when no object_name was supplied (legacy parts).
            part_mesh = mesh_by_name.get(str(part.get("object_name") or ""), mesh_obj)
            triangle_keys = [tuple(key) for key in (part.get("triangle_keys") or [])]
            # Core triangles come first; the remainder is the one-ring
            # dilation shared with adjacent parts that the renderer fades
            # out with a soft image-space mask.
            raw_core_count = part.get("core_triangle_count")
            core_triangle_keys = None
            if raw_core_count is not None:
                core_count = max(0, min(int(raw_core_count), len(triangle_keys)))
                if core_count < len(triangle_keys):
                    core_triangle_keys = triangle_keys[:core_count]
            ok = render_part_sprite(
                part_mesh,
                view_cfg,
                triangle_keys,
                dict(part.get("projection_frame") or {}),
                str(part_output),
                resolution=resolution,
                depth_center=float(part.get("mean_depth", 0.0) or 0.0),
                bind_frame=render_frame,
                use_rest_pose=use_rest_pose,
                projection_matrix=projection_matrix,
                core_triangle_keys=core_triangle_keys,
            )
            renders.append(
                {
                    "name": str(part.get("name") or attachment_name),
                    "attachment_name": attachment_name,
                    "ok": bool(ok),
                    "detail": "rendered" if ok else "render failed",
                    "output_path": str(part_output),
                    "width": resolution,
                    "height": resolution,
                }
            )

        payload = {
            "ok": True,
            "detail": "rendered",
            "source": source_path,
            "render_frame": render_frame,
            "setup_pose": setup_pose,
            "bind_borrow": bind_borrow_info,
            "use_rest_pose": bool(use_rest_pose),
            "mesh_reduction": mesh_reduction_report,
            "renders": renders,
        }
        if reference_result is not None:
            payload["reference"] = reference_result
        return payload
    except ImportError as e:
        return {"ok": False, "detail": f"Sprite rendering not available: {e}"}


def extract_mesh_targets_cli(
    source_path: str,
    output_path: str,
    *,
    view_preset: str = "front",
    view_dir=None,
    view_up=None,
    view_roll: float = 0.0,
    source_frame: int = None,
    use_rest_pose: bool = False,
    projection_space: str = "world",
    mesh_reduction: bool = True,
    mesh_target_vertices: int = 5000,
    weight_aware_decimation: bool = False,
    target_spec_path: str = None,
    bind_from_animation: str = None,
) -> dict[str, object]:
    """Evaluate the animated mesh per frame and project vertices to 2D.

    Produces Blender-skinned ground-truth targets for the C++ Spine global
    optimizer. The dedup map (source_vertex_indices) is rebuilt with the same
    extract_2d_mesh call used during extract-scene so vertex ordering matches.

    The target spec JSON has the form:
        {
          "fps": 30.0,
          "animations": [
            {"name": "Walk", "sample_times": [0.0, 0.0333, ...]}
          ]
        }
    """
    import json as json_module

    if not target_spec_path:
        return {"ok": False, "detail": "target-spec is required"}
    spec_path = Path(target_spec_path).expanduser().resolve()
    if not spec_path.exists():
        return {"ok": False, "detail": f"target-spec not found: {spec_path}"}
    try:
        spec = json_module.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "detail": f"target-spec parse failed: {exc}"}

    fps = float(spec.get("fps") or 30.0)
    if fps <= 0.0:
        return {"ok": False, "detail": "fps must be > 0"}
    requested_animations = list(spec.get("animations") or [])
    if not requested_animations:
        return {"ok": False, "detail": "target-spec has no animations"}

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(source_path)

    # Optimizer ground-truth targets are computed for the primary mesh only;
    # extra rigid accessory objects (e.g. a sword) are not weight-optimized.
    mesh_obj, armature_obj = find_mesh_and_armature()
    if mesh_obj is None:
        return {"ok": False, "detail": "No mesh found in scene"}
    if armature_obj is None:
        return {"ok": False, "detail": "No armature found in scene"}

    # Every clip here is an externally-sourced retargeted animation (each spec
    # entry carries its own source_path that _import_action_from_source loads
    # below). Wipe the model's own embedded action first so it can't be picked
    # as a target action, drive the mesh, or perturb action-name dedup — the
    # per-frame mesh targets must come only from the retargeted source motion.
    purge_model_animations(armature_obj)

    bind_borrow_info = _maybe_borrow_bind_from_animation(
        armature_obj,
        bind_from_animation,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
    )

    setup_frame = _resolve_setup_frame(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
        neutral_auto_pose=True,
    )
    bpy.context.scene.frame_set(setup_frame)
    # Same constant-root-yaw removal as extract_scene_cli so the per-frame
    # mesh targets live in the same world frame as the extracted scene (skipped
    # when a bind pose was borrowed — the donor pose already fixes the facing).
    if not use_rest_pose:
        _align_root_unless_bind_borrowed(armature_obj, setup_frame, bind_borrow_info)
    _apply_auto_setup_pose(
        armature_obj,
        source_frame=source_frame,
        use_rest_pose=use_rest_pose,
    )
    reduce_mesh_object(
        mesh_obj,
        target_vertices=mesh_target_vertices,
        enabled=mesh_reduction,
        weight_aware_decimation=weight_aware_decimation,
    )

    view_cfg = get_scene_view_config(
        view_name=view_preset,
        view_dir=tuple(view_dir) if view_dir is not None else None,
        up_hint=tuple(view_up) if view_up is not None else None,
        roll_degrees=view_roll,
        armature_obj=armature_obj,
    )

    # Rebuild dedup map using the same call extract-scene uses on the
    # bind frame so the vertex ordering matches the C++ optimizer's
    # mesh_data.vertices_2d index space.
    bind_mesh = extract_2d_mesh(
        mesh_obj,
        view_cfg,
        source_frame=setup_frame,
        use_rest_pose=use_rest_pose,
    )
    source_vertex_indices = [int(i) for i in (bind_mesh.get("source_vertex_indices") or [])]
    if not source_vertex_indices:
        return {"ok": False, "detail": "bind dedup map is empty"}

    basis_2d = np.asarray(view_cfg["basis_2d"], dtype=np.float64)
    # Triangles in the deduped vertex index space + the view direction, used
    # to compute per-frame front-facing visibility (see below). Without a
    # visibility mask the C++ optimizer fits BOTH front- and back-facing
    # vertices of every body part to the same 2D bone — a contradiction that
    # 2D-LBS cannot satisfy, so the bone oscillates frame to frame (the
    # whole-body "tremble"). Python's build_visibility_map returns this mask;
    # the CLI target extractor previously omitted it.
    visibility_triangles = np.asarray(bind_mesh.get("triangles") or [], dtype=np.int64).reshape(
        -1, 3
    )
    view_dir_np = np.asarray(view_cfg["view_dir"], dtype=np.float64)
    num_dedup_verts = len(source_vertex_indices)

    def _front_facing_visibility(proj_positions_3d):
        """Mark deduped vertices on any front-facing triangle visible.

        Mirrors flatrig.projection.compute_front_facing_vertex_visibility_from_
        triangles: a triangle faces the camera when its projection-space normal
        points against the view direction (normal . view_dir < 0).
        """
        visible = np.zeros(num_dedup_verts, dtype=bool)
        if visibility_triangles.size == 0:
            visible[:] = True
            return visible
        p0 = proj_positions_3d[visibility_triangles[:, 0]]
        p1 = proj_positions_3d[visibility_triangles[:, 1]]
        p2 = proj_positions_3d[visibility_triangles[:, 2]]
        normals = np.cross(p1 - p0, p2 - p0)
        normal_lengths = np.linalg.norm(normals, axis=1)
        eps = 1e-9
        facing = (normal_lengths > eps) & (np.einsum("ij,j->i", normals, view_dir_np) < -eps)
        if np.any(facing):
            visible[np.unique(visibility_triangles[facing].reshape(-1))] = True
        return visible

    actions = [action for action in bpy.data.actions if is_pose_action(action)]
    if not actions:
        return {"ok": False, "detail": "No pose animations in scene"}
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()

    output_animations: list[dict[str, object]] = []

    def _import_action_from_source(source_path: str):
        """Import an FBX action and bind it to the target armature.

        Returns the imported pose action (or None when import failed / the
        FBX brought no action). Cleans up the imported armature/mesh and
        leaves only the action in `bpy.data.actions` so the depsgraph
        evaluates the model armature against the borrowed motion.

        This is the path that lets `extract-mesh-targets` evaluate the
        per-frame mesh for a retargeted animation whose action does not
        live in the model FBX. Without it `_resolve_action_for_export`
        could only find actions named after whatever the model's own FBX
        ships with (typically a single default action), which is why the
        optimizer was getting mesh targets for at most one clip.
        """
        if not source_path:
            return None
        bind_path = Path(str(source_path)).expanduser()
        if not bind_path.exists():
            return None
        actions_before = {a.name for a in bpy.data.actions}
        objects_before = {obj.name for obj in bpy.data.objects}
        try:
            import_model(str(bind_path))
        except Exception:
            return None
        new_actions = [a for a in bpy.data.actions if a.name not in actions_before]
        candidates = [a for a in new_actions if is_pose_action(a)]
        action = candidates[0] if candidates else None
        if action is not None:
            action.use_fake_user = True
        _purge_imported_objects(objects_before)
        bpy.context.view_layer.update()
        return action

    for anim_spec in requested_animations:
        name = str(anim_spec.get("name") or "").strip()
        sample_times = [float(t) for t in (anim_spec.get("sample_times") or [])]
        source_path_override = str(anim_spec.get("source_path") or "").strip()
        if not name or not sample_times:
            continue

        action = None
        # When a source path is provided, prefer importing its action — that
        # is the action that drives the retargeted animation. Falling back
        # to name lookup against the model's own action list is only
        # correct for clips that happen to share the model's action name.
        if source_path_override:
            action = _import_action_from_source(source_path_override)
        if action is None:
            try:
                action = _resolve_action_for_export(armature_obj, [name])
            except ValueError as exc:
                output_animations.append(
                    {
                        "name": name,
                        "sample_times": sample_times,
                        "target_positions_2d": [],
                        "ok": False,
                        "detail": str(exc),
                    }
                )
                continue
        if action is None:
            output_animations.append(
                {
                    "name": name,
                    "sample_times": sample_times,
                    "target_positions_2d": [],
                    "ok": False,
                    "detail": "action not found",
                }
            )
            continue

        armature_obj.animation_data.action = action
        action_start = float(action.frame_range[0])

        frame_positions: list[list[list[float]]] = []
        frame_visibility: list[list[int]] = []
        for sample_time in sample_times:
            frame_float = action_start + sample_time * fps
            frame_int = int(math.floor(frame_float))
            subframe = float(frame_float - frame_int)
            bpy.context.scene.frame_set(frame_int, subframe=subframe)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            depsgraph.update()

            eval_obj = mesh_obj.evaluated_get(depsgraph)
            eval_mesh = None
            try:
                eval_mesh = eval_obj.to_mesh()
                world_mat = eval_obj.matrix_world
                eval_verts = eval_mesh.vertices

                vertex_positions: list[list[float]] = []
                proj_positions_3d = np.zeros((num_dedup_verts, 3), dtype=np.float64)
                for dedup_index, source_index in enumerate(source_vertex_indices):
                    if source_index < 0 or source_index >= len(eval_verts):
                        vertex_positions.append([0.0, 0.0])
                        continue
                    co_world = world_mat @ eval_verts[source_index].co
                    co_world_np = np.array((co_world.x, co_world.y, co_world.z), dtype=np.float64)
                    co_projected = _transform_point_to_projection_space(
                        co_world_np, projection_inverse=None
                    )
                    proj_positions_3d[dedup_index] = co_projected
                    co_2d = basis_2d @ co_projected
                    vertex_positions.append([float(co_2d[0]), float(co_2d[1])])
                frame_positions.append(vertex_positions)
                visible = _front_facing_visibility(proj_positions_3d)
                frame_visibility.append([int(v) for v in visible])
            finally:
                if eval_mesh is not None:
                    eval_obj.to_mesh_clear()

        output_animations.append(
            {
                "name": name,
                "sample_times": sample_times,
                "target_positions_2d": frame_positions,
                "visibility_mask": frame_visibility,
                "ok": True,
            }
        )

    return {
        "ok": True,
        "detail": "extracted",
        "source": source_path,
        "setup_frame": setup_frame,
        "vertex_count": len(source_vertex_indices),
        "fps": fps,
        "animations": output_animations,
    }


def main() -> None:
    args = parse_args()
    source_path = str(Path(args.source).expanduser().resolve())
    output_path = Path(args.output).expanduser().resolve()

    payload: dict[str, object]

    if args.command == "inspect":
        payload = inspect_source(source_path)
    elif args.command == "convert":
        payload = convert_source(source_path, str(output_path))
    elif args.command == "extract-scene":
        payload = extract_scene_cli(
            source_path,
            str(output_path),
            view_preset=args.view_preset,
            view_dir=args.view_dir,
            view_up=args.view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            use_rest_pose=args.use_rest_pose,
            projection_space=args.projection_space,
            mesh_reduction=args.mesh_reduction,
            mesh_target_vertices=args.mesh_target_vertices,
            weight_aware_decimation=args.weight_aware_decimation,
            bind_from_animation=getattr(args, "bind_from_animation", None),
            base_color_texture_output=getattr(args, "base_color_texture_output", None),
        )
    elif args.command == "extract-animations":
        payload = extract_animations_cli(
            source_path,
            str(output_path),
            view_preset=args.view_preset,
            view_dir=args.view_dir,
            view_up=args.view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            projection_space=args.projection_space,
            animation_names=args.animation_names,
            fps=args.fps,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            sample_substeps=args.sample_substeps,
            optimize_animation_keys=args.optimize_animation_keys,
            force_loop_closing_keys=args.force_loop_closing_keys,
            pose_mode=args.pose_mode,
            pose_blend=args.pose_blend,
            rotation_flatten=args.rotation_flatten,
            rotation_flatten_scope=args.rotation_flatten_scope,
            rotation_flatten_bones=args.rotation_flatten_bones,
            connected_translation_scope=args.connected_translation_scope,
            connected_translation_bones=args.connected_translation_bones,
            stretch_guard_enabled=args.stretch_guard_enabled,
            stretch_guard_max_scale=args.stretch_guard_max_scale,
            stretch_guard_strength=args.stretch_guard_strength,
            stretch_guard_bones=args.stretch_guard_bones,
            ik_leaf_refine_enabled=args.ik_leaf_refine_enabled,
            ik_leaf_strength=args.ik_leaf_strength,
            ik_leaf_iterations=args.ik_leaf_iterations,
            ik_leaf_max_chain_length=args.ik_leaf_max_chain_length,
            ik_leaf_preserve_scale=args.ik_leaf_preserve_scale,
            drop_problematic_frames=args.drop_problematic_frames,
            preserve_root_motion=args.preserve_root_motion,
            preserve_root_rotation=args.preserve_root_rotation,
            bind_from_animation=getattr(args, "bind_from_animation", None),
            animation_source=getattr(args, "animation_source", None),
            decouple_scale=getattr(args, "decouple_scale", False),
        )
    elif args.command == "export-3d-animation-bvh":
        if not args.bvh_output:
            raise ValueError("--bvh-output is required for export-3d-animation-bvh")
        payload = export_3d_animation_bvh_cli(
            source_path,
            str(output_path),
            bvh_output=args.bvh_output,
            animation_names=args.animation_names,
            fps=args.fps,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
    elif args.command == "export-3d-rest-bvh":
        if not args.bvh_output:
            raise ValueError("--bvh-output is required for export-3d-rest-bvh")
        payload = export_3d_rest_bvh_cli(
            source_path,
            str(output_path),
            bvh_output=args.bvh_output,
            view_preset=args.view_preset,
            view_dir=args.view_dir,
            view_up=args.view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            use_rest_pose=args.use_rest_pose,
            projection_space=args.projection_space,
            fps=args.fps,
            frame_count=args.frame_count,
            bind_from_animation=getattr(args, "bind_from_animation", None),
        )
    elif args.command == "dump-rig-animation":
        payload = dump_rig_animation_cli(
            source_path,
            str(output_path),
            animation_names=args.animation_names,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
    elif args.command == "bake-rig-animation":
        if not args.bake_spec:
            raise ValueError("--bake-spec is required for bake-rig-animation")
        if not args.flat_output:
            raise ValueError("--flat-output is required for bake-rig-animation")
        payload = bake_rig_animation_cli(
            source_path,
            str(output_path),
            bake_spec=args.bake_spec,
            flat_output=args.flat_output,
        )
    elif args.command == "reduce-rig-to-canonical":
        if not args.flat_output:
            raise ValueError("--flat-output is required for reduce-rig-to-canonical")
        payload = reduce_rig_to_canonical_cli(
            source_path,
            str(output_path),
            args.flat_output,
        )
    elif args.command == "cleanup-mesh":
        if not args.glb_output:
            raise ValueError("--glb-output is required for cleanup-mesh")
        payload = cleanup_generated_mesh(
            source_path,
            glb_output=args.glb_output,
            target_triangles=args.target_triangles,
            voxel_remesh=args.voxel_remesh,
            remove_loose=args.remove_loose,
            fbx_output=args.fbx_output,
            orientation_fix=args.orientation_fix,
        )
    elif args.command == "bake-predicted-rig":
        if not args.fbx_output:
            raise ValueError("--fbx-output is required for bake-predicted-rig")
        payload = bake_predicted_rig(source_path, fbx_output=args.fbx_output, mesh_path=args.mesh_path)
    elif args.command == "extract-mesh-targets":
        payload = extract_mesh_targets_cli(
            source_path,
            str(output_path),
            view_preset=args.view_preset,
            view_dir=args.view_dir,
            view_up=args.view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            use_rest_pose=args.use_rest_pose,
            projection_space=args.projection_space,
            mesh_reduction=args.mesh_reduction,
            mesh_target_vertices=args.mesh_target_vertices,
            weight_aware_decimation=args.weight_aware_decimation,
            target_spec_path=args.target_spec,
            bind_from_animation=getattr(args, "bind_from_animation", None),
        )
    elif args.command == "render-sprites":
        payload = render_sprites_cli(
            source_path,
            str(output_path),
            view_preset=args.view_preset,
            view_dir=args.view_dir,
            view_up=args.view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            use_rest_pose=args.use_rest_pose,
            projection_space=args.projection_space,
            parts_json=args.parts_json,
            images_dir=args.images_dir,
            resolution=args.resolution,
            bind_frame=args.bind_frame,
            mesh_reduction=args.mesh_reduction,
            mesh_target_vertices=args.mesh_target_vertices,
            weight_aware_decimation=args.weight_aware_decimation,
            bind_from_animation=getattr(args, "bind_from_animation", None),
        )
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
