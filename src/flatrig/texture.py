"""
Orthographic preview and part sprite rendering for the Blender worker.
"""

import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from flatrig.projection import (
    compose_projection_plane_point,
    transform_direction_from_projection_space,
    transform_point_from_projection_space,
)


def _pick_render_engine(scene):
    engine_property = scene.render.bl_rna.properties.get("engine")
    available = {
        item.identifier
        for item in (engine_property.enum_items if engine_property is not None else [])
    }
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH", "CYCLES"):
        if candidate in available:
            return candidate
    return scene.render.engine


def _frame_to_world_center(projection_frame, view_cfg, depth_center=0.0, projection_matrix=None):
    coords = compose_projection_plane_point(
        projection_frame["center_x"],
        projection_frame["center_y"],
        depth_center,
        view_cfg,
    )
    world_coords = transform_point_from_projection_space(
        coords,
        projection_matrix=projection_matrix,
    )
    return Vector(np.asarray(world_coords, dtype=np.float64))


def setup_orthographic_camera(
    view_cfg,
    projection_frame,
    depth_center=0.0,
    distance=10.0,
    camera_name="Sidecar_Camera",
    projection_matrix=None,
):
    """Create an orthographic camera that matches a projection frame."""
    center = _frame_to_world_center(
        projection_frame,
        view_cfg,
        depth_center=depth_center,
        projection_matrix=projection_matrix,
    )
    right_axis = Vector(
        transform_direction_from_projection_space(
            view_cfg["right_axis"],
            projection_matrix=projection_matrix,
        )
    ).normalized()
    up_axis = Vector(
        transform_direction_from_projection_space(
            view_cfg["up_axis"],
            projection_matrix=projection_matrix,
        )
    ).normalized()
    view_dir = Vector(
        transform_direction_from_projection_space(
            view_cfg["view_dir"],
            projection_matrix=projection_matrix,
        )
    ).normalized()
    camera_location = center - view_dir * distance

    z_axis = (-view_dir).normalized()
    camera_matrix = Matrix(
        (
            (right_axis.x, up_axis.x, z_axis.x, camera_location.x),
            (right_axis.y, up_axis.y, z_axis.y, camera_location.y),
            (right_axis.z, up_axis.z, z_axis.z, camera_location.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.name = camera_name
    camera.matrix_world = camera_matrix
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = projection_frame["span"]
    bpy.context.scene.camera = camera
    return camera


def render_projected_sprite(scene, output_path, resolution=2048):
    """Render the current scene from its active orthographic camera."""
    print(f"[sidecar_texture] Rendering sprite at {resolution}x{resolution}...")

    scene.render.engine = _pick_render_engine(scene)
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = output_path

    try:
        bpy.ops.render.render(write_still=True)
        print(f"[sidecar_texture] Sprite rendered to: {output_path}")
        return True
    except Exception as exc:
        print(f"[sidecar_texture] Render failed: {exc}")
        _create_placeholder_atlas(output_path, resolution)
        return True


def _build_unlit_material(original_material):
    """Create a temporary emission-only copy that preserves base-color textures."""
    material = original_material.copy()
    material.use_nodes = True
    if hasattr(material, "use_backface_culling"):
        material.use_backface_culling = False
    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output_node = next(
        (node for node in nodes if node.bl_idname == "ShaderNodeOutputMaterial"),
        None,
    )
    principled_node = next(
        (node for node in nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"),
        None,
    )

    if output_node is None:
        output_node = nodes.new("ShaderNodeOutputMaterial")

    emission_node = nodes.new("ShaderNodeEmission")
    emission_node.name = "_sidecar_emission"
    emission_node.inputs["Strength"].default_value = 1.0

    if principled_node is not None:
        base_input = principled_node.inputs["Base Color"]
        if base_input.links:
            source_socket = base_input.links[0].from_socket
            links.new(source_socket, emission_node.inputs["Color"])
        else:
            emission_node.inputs["Color"].default_value = base_input.default_value
    else:
        emission_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    for link in list(output_node.inputs["Surface"].links):
        links.remove(link)
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
    return material


def _apply_unlit_materials(objects):
    """Swap materials to temporary unlit copies and return a restore token."""
    restore_info = []
    created_materials = []

    for obj in objects:
        if obj.type != "MESH":
            continue

        original_materials = list(obj.data.materials)
        replacement_materials = []
        for material in original_materials:
            if material is None:
                replacement_materials.append(None)
                continue
            unlit_material = _build_unlit_material(material)
            replacement_materials.append(unlit_material)
            created_materials.append(unlit_material)

        obj.data.materials.clear()
        for material in replacement_materials:
            obj.data.materials.append(material)
        restore_info.append((obj, original_materials))

    return restore_info, created_materials


def _restore_materials(restore_info, created_materials):
    """Restore original materials and delete temporary unlit copies."""
    for obj, original_materials in restore_info:
        obj.data.materials.clear()
        for material in original_materials:
            obj.data.materials.append(material)

    for material in created_materials:
        bpy.data.materials.remove(material, do_unlink=True)


def _soft_cut_support_available():
    try:
        from PIL import Image, ImageFilter  # noqa: F401
    except ImportError:
        return False
    return True


def _apply_soft_ring_cut(scene, part_obj, core_obj, output_path, resolution=1024):
    """Fade the shared one-ring dilation out with a smooth image-space mask.

    The part mesh was rendered with one extra ring of triangles borrowed from
    its neighbours. A second render of the core-only object provides a
    coverage mask through its alpha channel (material-independent); blurring
    that ownership mask and remapping it with a smoothstep turns the
    triangle-edge cut into a smooth anti-aliased alpha edge that stays inside
    the dilated geometry. The adjacent part computes the complementary mask
    over the same corridor, so the two sprites cross-fade and the seam stays
    closed.
    """
    from PIL import Image, ImageFilter

    mask_path = str(output_path).rsplit(".", 1)[0] + "_ring_mask.png"

    previous_part_state = part_obj.hide_render
    previous_core_state = core_obj.hide_render
    part_obj.hide_render = True
    core_obj.hide_render = False
    try:
        rendered = render_projected_sprite(scene, mask_path, resolution=resolution)
    finally:
        part_obj.hide_render = previous_part_state
        core_obj.hide_render = previous_core_state

    if not rendered or not os.path.isfile(mask_path):
        return False

    try:
        color_image = Image.open(output_path).convert("RGBA")
        mask_image = Image.open(mask_path).convert("RGBA")
        if mask_image.size != color_image.size:
            mask_image = mask_image.resize(color_image.size)
        color = np.asarray(color_image, dtype=np.float32) / 255.0
        core_alpha = np.asarray(mask_image, dtype=np.float32)[..., 3] / 255.0
        coverage = color[..., 3]

        # Normalized matte: the blurred core coverage divided by the blurred
        # part coverage measures the local core fraction. Deep inside the core
        # (and along its outer silhouette) the ratio stays 1, inside the ring
        # it drops to 0, and across the seam it ramps smoothly — so the cut
        # fades without eroding outer silhouettes or ghosting ring edges.
        # Sized to straighten label-boundary sawtooth (measured up to ~±19 px
        # at 2048 regardless of tessellation); the two-triangle ring corridor
        # gives the level set room to deviate from the jagged mesh boundary.
        blur_radius = max(2.0, resolution * 0.008)

        def blur(values):
            image = Image.fromarray((values * 255.0).astype(np.uint8))
            image = image.filter(ImageFilter.GaussianBlur(blur_radius))
            return np.asarray(image, dtype=np.float32) / 255.0

        ratio = blur(core_alpha) / np.maximum(blur(coverage), 1e-4)
        ratio = np.clip(ratio, 0.0, 1.0)
        ramp = np.clip((ratio - 0.25) / 0.5, 0.0, 1.0)
        soft_mask = ramp * ramp * (3.0 - 2.0 * ramp)
        pixels = color * 255.0
        pixels[..., 3] *= soft_mask
        Image.fromarray(
            np.clip(pixels, 0.0, 255.0).astype(np.uint8)
        ).save(output_path)
        return True
    finally:
        try:
            os.remove(mask_path)
        except OSError:
            pass


def render_preview_sprite(
    obj,
    view_cfg,
    projection_frame,
    output_path,
    resolution=2048,
    depth_center=0.0,
    bind_frame=None,
    projection_matrix=None,
):
    """Render a full-body preview that matches the exported projection."""
    scene = bpy.context.scene
    if bind_frame is not None:
        scene.frame_set(bind_frame)
        bpy.context.view_layer.update()
    armatures = [scene_obj for scene_obj in scene.objects if scene_obj.type == "ARMATURE"]
    mesh_objects = [scene_obj for scene_obj in scene.objects if scene_obj.type == "MESH"]
    hidden_armatures = []

    for armature in armatures:
        hidden_armatures.append((armature, armature.hide_render))
        armature.hide_render = True

    camera = setup_orthographic_camera(
        view_cfg,
        projection_frame,
        depth_center=depth_center,
        camera_name="Sidecar_PreviewCamera",
        projection_matrix=projection_matrix,
    )
    restore_info, created_materials = _apply_unlit_materials(mesh_objects)

    try:
        return render_projected_sprite(scene, output_path, resolution=resolution)
    finally:
        _restore_materials(restore_info, created_materials)
        bpy.data.objects.remove(camera, do_unlink=True)
        for armature, previous_state in hidden_armatures:
            armature.hide_render = previous_state


def render_part_sprite(
    source_obj,
    view_cfg,
    triangle_keys,
    projection_frame,
    output_path,
    resolution=1024,
    depth_center=0.0,
    bind_frame=None,
    use_rest_pose=False,
    projection_matrix=None,
    core_triangle_keys=None,
):
    """Render a cropped sprite for one body part.

    The part is extracted from the evaluated bind-pose mesh, so the image and
    the exported mesh live in the same 2D setup pose.

    When ``core_triangle_keys`` is given, ``triangle_keys`` is expected to
    contain the core triangles plus a one-ring dilation shared with adjacent
    parts. The ring is rendered but faded out with a smooth image-space mask,
    so the visible cut is an anti-aliased alpha edge inside the dilated
    geometry instead of the jagged triangle boundary.
    """
    scene = bpy.context.scene
    if bind_frame is not None:
        scene.frame_set(bind_frame)
        bpy.context.view_layer.update()
    rest_pose_state = []
    if use_rest_pose:
        for scene_obj in scene.objects:
            if scene_obj.type == "ARMATURE" and scene_obj.data is not None:
                rest_pose_state.append((scene_obj.data, scene_obj.data.pose_position))
                scene_obj.data.pose_position = "REST"
        if rest_pose_state:
            bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    render_mesh = None
    try:
        eval_obj = source_obj.evaluated_get(depsgraph)
        render_mesh = bpy.data.meshes.new_from_object(
            eval_obj,
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
    finally:
        for armature_data, pose_position in rest_pose_state:
            armature_data.pose_position = pose_position
        if rest_pose_state:
            bpy.context.view_layer.update()

    bm = bmesh.new()
    bm.from_mesh(render_mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    wanted = {tuple(key) for key in triangle_keys}
    core_wanted = (
        {tuple(key) for key in core_triangle_keys}
        if core_triangle_keys is not None
        else None
    )
    if core_wanted is not None and not _soft_cut_support_available():
        # Without PIL the ring cannot be faded out; fall back to the exact
        # core cut (previous behaviour) instead of leaving a hard overlap.
        wanted = core_wanted
        core_wanted = None

    # Vertex indices are only reliable before bmesh.ops.delete, so core
    # membership has to be captured in the same pass that picks the faces
    # to delete. kept_core_flags follows the surviving-face order, which
    # deletion and bm.copy() both preserve.
    delete_faces = []
    kept_core_flags = [] if core_wanted is not None else None
    for face in bm.faces:
        tri = tuple(sorted(vert.index for vert in face.verts))
        if tri not in wanted:
            delete_faces.append(face)
            continue
        if kept_core_flags is not None:
            kept_core_flags.append(tri in core_wanted)

    if delete_faces:
        bmesh.ops.delete(bm, geom=delete_faces, context="FACES")

    if not bm.faces:
        bm.free()
        bpy.data.meshes.remove(render_mesh, do_unlink=True)
        return False

    # Core-only copy used to render the ownership mask for the soft cut.
    core_mesh = None
    if kept_core_flags is not None and not all(kept_core_flags):
        core_bm = bm.copy()
        core_bm.faces.ensure_lookup_table()
        ring_faces = [
            face
            for face, is_core in zip(core_bm.faces, kept_core_flags)
            if not is_core
        ]
        if ring_faces:
            bmesh.ops.delete(core_bm, geom=ring_faces, context="FACES")
        if core_bm.faces:
            core_mesh = bpy.data.meshes.new(f"{source_obj.name}_sidecar_core")
            core_bm.to_mesh(core_mesh)
        core_bm.free()

    bm.to_mesh(render_mesh)
    bm.free()

    # render_mesh.update()
    # render_mesh.calc_normals_split()
    # render_mesh.calc_loop_triangles()

    render_obj = bpy.data.objects.new(f"{source_obj.name}_sidecar_part", render_mesh)
    render_obj.matrix_world = source_obj.matrix_world.copy()
    scene.collection.objects.link(render_obj)

    core_obj = None
    if core_mesh is not None:
        core_obj = bpy.data.objects.new(f"{source_obj.name}_sidecar_core", core_mesh)
        core_obj.matrix_world = source_obj.matrix_world.copy()
        scene.collection.objects.link(core_obj)
        core_obj.hide_render = True

    restore_info, created_materials = _apply_unlit_materials([render_obj])

    camera = setup_orthographic_camera(
        view_cfg,
        projection_frame,
        depth_center=depth_center,
        camera_name="Sidecar_PartCamera",
        projection_matrix=projection_matrix,
    )

    hidden_objects = []
    for scene_obj in scene.objects:
        # core_obj is managed (and removed) explicitly by the soft-cut pass.
        if scene_obj in (render_obj, camera, core_obj):
            continue
        hidden_objects.append((scene_obj, scene_obj.hide_render))
        scene_obj.hide_render = True

    try:
        success = render_projected_sprite(scene, output_path, resolution=resolution)
        if success and core_obj is not None:
            _apply_soft_ring_cut(
                scene,
                render_obj,
                core_obj,
                output_path,
                resolution=resolution,
            )
        return success
    finally:
        _restore_materials(restore_info, created_materials)
        bpy.data.objects.remove(render_obj, do_unlink=True)
        bpy.data.meshes.remove(render_mesh, do_unlink=True)
        if core_obj is not None:
            bpy.data.objects.remove(core_obj, do_unlink=True)
        if core_mesh is not None:
            bpy.data.meshes.remove(core_mesh, do_unlink=True)
        bpy.data.objects.remove(camera, do_unlink=True)
        for scene_obj, previous_state in hidden_objects:
            scene_obj.hide_render = previous_state


def _create_placeholder_atlas(output_path, resolution=2048):
    """Create a plain placeholder image if rendering fails."""
    print("[sidecar_texture] Creating placeholder atlas...")
    image = bpy.data.images.new("placeholder", width=resolution, height=resolution)
    image.pixels = [1.0] * (resolution * resolution * 4)
    image.filepath_raw = output_path
    image.file_format = "PNG"
    image.save()
    print(f"[sidecar_texture] Placeholder saved to: {output_path}")
