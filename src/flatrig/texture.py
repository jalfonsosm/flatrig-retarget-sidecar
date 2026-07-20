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


def _filter_alpha(alpha, image_filter):
    """Apply a Pillow filter to a normalized alpha matte."""
    from PIL import Image

    pixels = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    filtered = Image.fromarray(pixels, mode="L").filter(image_filter)
    return np.asarray(filtered, dtype=np.float32) / 255.0


def _build_soft_ring_alpha(
    core_alpha,
    coverage_alpha,
    *,
    feather_radius=None,
    underlay_radius=None,
):
    """Build a seam-safe alpha matte for a core plus borrowed triangle ring.

    ``core_alpha`` is the canonical ownership matte and ``coverage_alpha`` is
    the render of the core plus its borrowed ring.  The core must remain an
    opaque underlay: cross-fading two independently rendered ownership mattes
    would otherwise turn two 0.5 edge samples into only 0.75 source-over
    coverage.  A small, coverage-clipped dilation protects that raster edge,
    while only the remainder of the borrowed ring is feathered.

    The borrowed geometry is an *underlay*, not a second complementary
    cross-fade.  Inside the seam band, any real source coverage therefore
    becomes opaque even when Blender rasterized that sample at alpha 0.5.  The
    outer boundary of ``coverage_alpha`` is detected separately and copied
    byte-for-byte, so hardening the internal seam cannot grow the character's
    silhouette or bridge a source hole.
    """
    from PIL import ImageFilter

    core = np.asarray(core_alpha, dtype=np.float32)
    coverage = np.asarray(coverage_alpha, dtype=np.float32)
    if core.ndim != 2 or coverage.ndim != 2:
        raise ValueError("soft-ring alpha mattes must be two-dimensional")
    if core.shape != coverage.shape:
        raise ValueError("core and coverage alpha mattes must have the same shape")
    if not core.size:
        return np.empty_like(core)

    core = np.clip(core, 0.0, 1.0)
    coverage = np.clip(coverage, 0.0, 1.0)
    image_extent = max(core.shape)
    if feather_radius is None:
        feather_radius = max(2.0, image_extent * 0.008)
    else:
        feather_radius = max(0.0, float(feather_radius))
    if underlay_radius is None:
        # At least one texel is required to cover the two complementary
        # antialias samples at a shared triangle edge.  Scaling the guard with
        # the image keeps it effective when neighbouring sprites use pages at
        # different pixel densities.
        underlay_radius = max(2, int(np.ceil(image_extent * 0.0015)))
    else:
        underlay_radius = max(0, int(np.ceil(float(underlay_radius))))

    def blur(values):
        if feather_radius <= 0.0:
            return values
        return _filter_alpha(values, ImageFilter.GaussianBlur(feather_radius))

    blurred_coverage = blur(coverage)
    ratio = blur(core) / np.maximum(blurred_coverage, 1e-4)
    ratio = np.clip(ratio, 0.0, 1.0)
    ramp = np.clip((ratio - 0.25) / 0.5, 0.0, 1.0)
    soft_mask = ramp * ramp * (3.0 - 2.0 * ramp)
    feathered_ring = coverage * soft_mask

    if underlay_radius > 0:
        filter_size = underlay_radius * 2 + 1
        expanded_core = _filter_alpha(core, ImageFilter.MaxFilter(filter_size))
        borrowed = (coverage > core + (1.0 / 255.0)).astype(np.float32)
        near_borrowed = _filter_alpha(borrowed, ImageFilter.MaxFilter(filter_size))
    else:
        expanded_core = core
        near_borrowed = (coverage > core + (1.0 / 255.0)).astype(np.float32)

    coverage_support = coverage > (1.0 / 255.0)
    # A one-pixel erosion distinguishes an internal cut from the external
    # silhouette (and from the boundary of a genuine coverage hole).  Pillow's
    # MinFilter keeps this dependency-free in the Blender sidecar runtime.
    coverage_interior = _filter_alpha(
        coverage_support.astype(np.float32), ImageFilter.MinFilter(3)
    ) > 0.5
    seam_underlay = (
        coverage_support
        & coverage_interior
        & (expanded_core > (1.0 / 255.0))
        & (near_borrowed > 0.5)
    )

    result = np.maximum(core, feathered_ring)
    result[seam_underlay] = 1.0

    # A boundary sample shared by core and expanded renders is the true outer
    # silhouette (or the antialiased edge of a real source hole).  By contrast,
    # the outer edge of the *borrowed* geometry must remain free to feather;
    # copying that edge at alpha 1 would merely move the hard seam outward.
    local_borrowed = coverage > core + (1.0 / 255.0)
    true_source_boundary = ~coverage_interior & ~local_borrowed
    # Samples with no nearby borrowed geometry are unrelated to a segmentation
    # seam and also remain byte-exact.
    preserve_source = true_source_boundary | (near_borrowed <= 0.5)
    result[preserve_source] = coverage[preserve_source]
    result[~coverage_support] = coverage[~coverage_support]
    return np.clip(result, 0.0, 1.0)


def _apply_soft_ring_cut(scene, part_obj, core_obj, output_path, resolution=1024):
    """Keep the core opaque and fade its borrowed ring in image space.

    The part mesh was rendered with one extra ring of triangles borrowed from
    its neighbours. A second render of the core-only object provides a
    coverage mask through its alpha channel (material-independent). The core
    remains the deterministic opaque underlay on its side of the cut; only the
    extra ring feathers over the adjacent part. This avoids the alpha loss of
    compositing two complementary translucent mattes with source-over.
    """
    from PIL import Image

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
            mask_image = mask_image.resize(color_image.size, Image.Resampling.BILINEAR)
        color = np.asarray(color_image, dtype=np.float32) / 255.0
        core_alpha = np.asarray(mask_image, dtype=np.float32)[..., 3] / 255.0
        coverage = color[..., 3]
        output_alpha = _build_soft_ring_alpha(core_alpha, coverage)

        # Blender renders premultiplied alpha (film_transparent=True), so the
        # RGB in `color` is already premultiplied by `coverage`. The soft-ring
        # cut replaces `coverage` with `output_alpha`, so the RGB must be
        # re-premultiplied to stay valid. Leaving the original premultiplied
        # RGB paired with a different alpha creates invalid pixels (RGB > alpha
        # where alpha was lowered, or RGB < alpha where it was raised), which
        # show up as white halos or dark fringes at sprite edges in runtimes
        # that expect premultiplied textures (PixiJS with alphaMode =
        # "premultiplied-alpha"). Un-premultiply by the old alpha, then
        # re-premultiply by the new alpha.
        safe_coverage = np.maximum(coverage, 1e-4)
        rgb_straight = color[..., :3] / safe_coverage[..., None]
        rgb_straight = np.clip(rgb_straight, 0.0, 1.0)
        rgb_repremultiplied = rgb_straight * output_alpha[..., None]

        pixels = np.empty_like(color)
        pixels[..., :3] = rgb_repremultiplied * 255.0
        pixels[..., 3] = output_alpha * 255.0
        Image.fromarray(np.rint(np.clip(pixels, 0.0, 255.0)).astype(np.uint8)).save(output_path)
        return True
    finally:
        try:
            os.remove(mask_path)
        except OSError:
            pass


def _build_triangle_filtered_render_object(source_obj, triangle_keys, depsgraph):
    """Copy one evaluated mesh while retaining only explicitly exported faces."""
    eval_obj = source_obj.evaluated_get(depsgraph)
    render_mesh = bpy.data.meshes.new_from_object(
        eval_obj,
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    bm = bmesh.new()
    has_faces = False
    try:
        bm.from_mesh(render_mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        wanted = {tuple(sorted(int(value) for value in key)) for key in triangle_keys}
        delete_faces = [
            face
            for face in bm.faces
            if tuple(sorted(vert.index for vert in face.verts)) not in wanted
        ]
        if delete_faces:
            bmesh.ops.delete(bm, geom=delete_faces, context="FACES")
        has_faces = bool(bm.faces)
        if has_faces:
            bm.to_mesh(render_mesh)
    finally:
        bm.free()
    if not has_faces:
        bpy.data.meshes.remove(render_mesh, do_unlink=True)
        return None

    render_obj = bpy.data.objects.new(f"{source_obj.name}_sidecar_reference", render_mesh)
    render_obj.matrix_world = source_obj.matrix_world.copy()
    bpy.context.scene.collection.objects.link(render_obj)
    return render_obj


def _render_filtered_preview_sprite(
    view_cfg,
    projection_frame,
    output_path,
    triangle_groups,
    *,
    resolution,
    depth_center,
    bind_frame,
    projection_matrix,
):
    """Render only the exported core-triangle union, preserving scene depth."""
    scene = bpy.context.scene
    if bind_frame is not None:
        scene.frame_set(bind_frame)
        bpy.context.view_layer.update()

    depsgraph = bpy.context.evaluated_depsgraph_get()
    render_objects = []
    camera = None
    hidden_objects = []
    restore_info = []
    created_materials = []
    try:
        for group in triangle_groups:
            render_obj = _build_triangle_filtered_render_object(
                group["object"],
                group["triangle_keys"],
                depsgraph,
            )
            if render_obj is not None:
                render_objects.append(render_obj)
        if not render_objects:
            return False

        camera = setup_orthographic_camera(
            view_cfg,
            projection_frame,
            depth_center=depth_center,
            camera_name="Sidecar_PreviewCamera",
            projection_matrix=projection_matrix,
        )
        render_set = set(render_objects)
        for scene_obj in scene.objects:
            if scene_obj in render_set or scene_obj == camera:
                continue
            hidden_objects.append((scene_obj, scene_obj.hide_render))
            scene_obj.hide_render = True

        restore_info, created_materials = _apply_unlit_materials(render_objects)
        return render_projected_sprite(scene, output_path, resolution=resolution)
    finally:
        if restore_info or created_materials:
            _restore_materials(restore_info, created_materials)
        for render_obj in render_objects:
            render_mesh = render_obj.data
            bpy.data.objects.remove(render_obj, do_unlink=True)
            bpy.data.meshes.remove(render_mesh, do_unlink=True)
        if camera is not None:
            bpy.data.objects.remove(camera, do_unlink=True)
        for scene_obj, previous_state in hidden_objects:
            scene_obj.hide_render = previous_state


def render_preview_sprite(
    obj,
    view_cfg,
    projection_frame,
    output_path,
    resolution=2048,
    depth_center=0.0,
    bind_frame=None,
    projection_matrix=None,
    triangle_groups=None,
):
    """Render an assembled preview that matches the exported projection.

    ``triangle_groups=None`` preserves the legacy full-scene behaviour. New
    manifests pass filtered source objects and core keys; their temporary mesh
    copies share one camera/render so Blender's depth buffer resolves overlap.
    """
    if triangle_groups is not None:
        return _render_filtered_preview_sprite(
            view_cfg,
            projection_frame,
            output_path,
            triangle_groups,
            resolution=resolution,
            depth_center=depth_center,
            bind_frame=bind_frame,
            projection_matrix=projection_matrix,
        )

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
