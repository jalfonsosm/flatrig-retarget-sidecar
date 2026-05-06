"""
Orthographic preview and part sprite rendering for the Blender worker.
"""

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
        material.use_backface_culling = True
    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = False
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
):
    """Render a cropped sprite for one body part.

    The part is extracted from the evaluated bind-pose mesh, so the image and
    the exported mesh live in the same 2D setup pose.
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
    delete_faces = []
    for face in bm.faces:
        tri = tuple(sorted(vert.index for vert in face.verts))
        if tri not in wanted:
            delete_faces.append(face)

    if delete_faces:
        bmesh.ops.delete(bm, geom=delete_faces, context="FACES")

    fill_holes = False

    if not bm.faces:
        bm.free()
        bpy.data.meshes.remove(render_mesh, do_unlink=True)
        return False

    bm.to_mesh(render_mesh)
    bm.free()

    # render_mesh.update()
    # render_mesh.calc_normals_split()
    # render_mesh.calc_loop_triangles()

    render_obj = bpy.data.objects.new(f"{source_obj.name}_sidecar_part", render_mesh)
    render_obj.matrix_world = source_obj.matrix_world.copy()
    scene.collection.objects.link(render_obj)

    if fill_holes:
        # Ensure there's a material slot for the new hole faces
        hole_mat_index = len(render_obj.data.materials)
        hole_mat = bpy.data.materials.new(name="sidecar_hole_mask")
        render_obj.data.materials.append(hole_mat)

        # Assign the new material index to the filled polygons.
        for p in render_obj.data.polygons:
            if p.index >= len(render_obj.data.polygons) - len(hole_faces):
                p.material_index = hole_mat_index

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
        if scene_obj in (render_obj, camera):
            continue
        hidden_objects.append((scene_obj, scene_obj.hide_render))
        scene_obj.hide_render = True

    try:
        if fill_holes:
            # 1) Render the standard part sprite
            success = render_projected_sprite(scene, output_path, resolution=resolution)

            # 2) Render the hole mask
            mask_path = str(output_path).rsplit(".", 1)[0] + "_mask.png"

            # We need all materials except `hole_mat_index` to be completely transparent/black emission
            # And `hole_mat_index` to be completely white emission
            mask_materials = []
            for index, mat in enumerate(render_obj.data.materials):
                if mat is None:
                    mask_materials.append(None)
                    continue

                mask_mat = bpy.data.materials.new(name=f"sidecar_mask_mat_{index}")
                mask_mat.use_nodes = True
                if hasattr(mask_mat, "use_backface_culling"):
                    mask_mat.use_backface_culling = True
                if hasattr(mask_mat, "show_transparent_back"):
                    mask_mat.show_transparent_back = False
                nodes = mask_mat.node_tree.nodes
                links = mask_mat.node_tree.links
                nodes.clear()

                output_node = nodes.new("ShaderNodeOutputMaterial")
                emission_node = nodes.new("ShaderNodeEmission")
                emission_node.inputs["Strength"].default_value = 1.0

                if index == hole_mat_index:
                    emission_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # White hole
                else:
                    emission_node.inputs["Color"].default_value = (
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )  # Transparent rest

                links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

                mass_blend = getattr(mask_mat, "blend_method", None)
                if mass_blend is not None:
                    mask_mat.blend_method = "CLIP"  # Ensure transparency works in EEVEE if needed

                mask_materials.append(mask_mat)

            # Swap materials for mask rendering
            render_obj.data.materials.clear()
            for m in mask_materials:
                render_obj.data.materials.append(m)

            # For the mask, we want a black solid background, not transparent
            scene.render.film_transparent = False
            original_bg_color = None
            bg_node = None
            if scene.world is not None:
                scene.world.use_nodes = True
                bg_node = scene.world.node_tree.nodes.get("Background")
                if bg_node:
                    original_bg_color = tuple(bg_node.inputs["Color"].default_value)
                    bg_node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)  # Black background

            render_projected_sprite(scene, mask_path, resolution=resolution)

            # Restore transparent preference and background
            scene.render.film_transparent = True
            if bg_node and original_bg_color is not None:
                bg_node.inputs["Color"].default_value = original_bg_color

            # Cleanup mask materials
            for m in mask_materials:
                if m:
                    bpy.data.materials.remove(m, do_unlink=True)

            return success
        else:
            return render_projected_sprite(scene, output_path, resolution=resolution)
    finally:
        _restore_materials(restore_info, created_materials)
        bpy.data.objects.remove(render_obj, do_unlink=True)
        bpy.data.meshes.remove(render_mesh, do_unlink=True)
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
