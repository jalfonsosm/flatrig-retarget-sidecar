"""
Orthographic preview and part sprite rendering for the Blender worker.
"""

import os
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from flatrig._blender_projection import (
    compose_projection_plane_point,
    transform_direction_from_projection_space,
    transform_point_from_projection_space,
)


# Sprites are rendered from emission-only materials with no lights, no shadows
# and no global illumination, so Eevee's temporal samples buy edge antialiasing
# and nothing else. That saturates far below Eevee's 64-sample default: measured
# against a 64-sample reference at 2048x2048, 16 samples renders ~11x faster for
# a mean absolute alpha error of 0.009/255 and a 99th-percentile error of 0.
# Anything below 16 starts to show on high-frequency silhouettes.
DEFAULT_SPRITE_RENDER_SAMPLES = 16


def _sprite_render_samples():
    try:
        samples = int(
            os.environ.get(
                "FLATRIG_SPRITE_RENDER_SAMPLES", str(DEFAULT_SPRITE_RENDER_SAMPLES)
            )
        )
    except ValueError:
        samples = DEFAULT_SPRITE_RENDER_SAMPLES
    return max(1, min(samples, 64))


def _engine_is_selectable(scene, identifier):
    """Whether ``identifier`` can actually be assigned as the render engine.

    ``render.engine``'s enum only lists the engines built into Blender itself.
    Add-on engines register a RenderEngine subclass instead and never appear
    there, so Cycles is reported as unavailable by an enum check even though
    the property accepts it. Probing by assignment is the only reliable test.
    """
    previous = scene.render.engine
    if previous == identifier:
        return True
    try:
        scene.render.engine = identifier
    except (TypeError, ValueError):
        return False
    selected = scene.render.engine
    scene.render.engine = previous
    return selected == identifier


def _pick_render_engine(scene):
    candidates = []
    requested = os.environ.get("FLATRIG_SPRITE_RENDER_ENGINE", "").strip()
    if requested:
        candidates.append(requested)
    elif cycles_gpu_devices():
        # Cycles first, but only when it can reach a real compute device.
        #
        # Eevee has no device selector: it renders on whichever GPU the
        # process' OpenGL context landed on, and on a hybrid-graphics laptop
        # that is the integrated one. Cycles enumerates compute devices and
        # can be pointed at the discrete card. Measured on a GTX 1070 Max-Q +
        # UHD 630 machine, a 12-sprite 2048x2048 batch took 116.4s through
        # Eevee on the iGPU versus 80.2s through Cycles on the dGPU, most of
        # the difference being Eevee's ~20s cold shader compilation.
        #
        # Without a GPU the ranking inverts (Cycles CPU at 16 samples measured
        # 4.6s per render against Eevee's 2.7s), hence the guard.
        candidates.append("CYCLES")
    # Eevee reproduces the emission materials and alpha-blended textures the
    # sprite path relies on. Workbench is a last-resort fallback for builds
    # where Eevee is unavailable.
    candidates.extend(("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH", "CYCLES"))
    for candidate in candidates:
        if candidate == "CYCLES":
            _ensure_cycles_addon()
        if _engine_is_selectable(scene, candidate):
            return candidate
    return scene.render.engine


def _ensure_cycles_addon():
    """Register Cycles so it can be selected. Best effort; safe if missing."""
    try:
        import addon_utils

        addon_utils.enable("cycles", default_set=False, persistent=True)
    except Exception:
        pass


# Order matters: the first backend with a usable device wins. CUDA is tried
# before OptiX because sprites are a single-bounce emission pass, where OptiX's
# ray-tracing advantage barely applies but its first-render kernel compile is
# still paid. On a GTX 1070 the two finished a 12-sprite batch within 0.1s of
# each other, while in isolation OptiX stalled ~6s compiling kernels on the
# first render.
_CYCLES_GPU_BACKENDS = ("METAL", "CUDA", "OPTIX", "HIP", "ONEAPI")


def cycles_gpu_devices():
    """``(backend, [device names])`` for the first usable Cycles GPU backend.

    Returns ``None`` when Cycles cannot reach any GPU. Enabling the devices is
    left to :func:`_configure_cycles_device`; this only answers "is there one",
    so engine selection can ask before committing to Cycles.
    """
    _ensure_cycles_addon()
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
    except (AttributeError, KeyError):
        return None
    for backend in _CYCLES_GPU_BACKENDS:
        try:
            preferences.compute_device_type = backend
        except (TypeError, ValueError):
            continue
        try:
            preferences.get_devices()
        except Exception:
            continue
        names = [device.name for device in preferences.devices if device.type == backend]
        if names:
            return backend, names
    return None


def _configure_cycles_device(scene):
    """Point Cycles at a discrete GPU when one is usable.

    Unlike Eevee -- whose device is whatever GPU the process' OpenGL context
    landed on, whether or not that is the fast one -- Cycles enumerates the
    real compute devices and lets us choose. On a hybrid-graphics laptop the
    OpenGL context defaults to the integrated GPU, so this is the only path
    that reaches the discrete card. Returns a human-readable device label.
    """
    cycles = getattr(scene, "cycles", None)
    if cycles is None:
        return "cycles settings unavailable"

    cycles.samples = _sprite_render_samples()
    for attr in ("use_denoising", "use_adaptive_sampling"):
        if hasattr(cycles, attr):
            setattr(cycles, attr, False)
    # Sprites are unlit emission: no bounce carries any signal, so every extra
    # bounce is pure cost. Transparent bounces stay put -- alpha-clipped
    # textures need them to composite correctly.
    for attr in (
        "max_bounces",
        "diffuse_bounces",
        "glossy_bounces",
        "transmission_bounces",
        "volume_bounces",
    ):
        if hasattr(cycles, attr):
            setattr(cycles, attr, 0)

    selected = cycles_gpu_devices()
    # Enumerating compute devices invalidates previously fetched RNA pointers,
    # so `cycles` captured above is stale from here on: re-fetch it rather than
    # reusing it, or the assignment below raises AttributeError.
    cycles = scene.cycles
    if selected is None:
        cycles.device = "CPU"
        return "CPU (no GPU compute device found)"

    backend, names = selected
    preferences = bpy.context.preferences.addons["cycles"].preferences
    # cycles_gpu_devices() already left compute_device_type on `backend`; now
    # actually enable those devices and disable everything else.
    for device in preferences.devices:
        device.use = device.type == backend
    scene.cycles.device = "GPU"
    return f"{backend}: {', '.join(names)}"


def _set_enum_if_available(owner, attr, value):
    prop = getattr(getattr(owner, "bl_rna", None), "properties", {}).get(attr)
    if prop is not None:
        allowed = {item.identifier for item in prop.enum_items}
        if value not in allowed:
            return False
    try:
        setattr(owner, attr, value)
        return True
    except Exception:
        return False


def _configure_sprite_colour_management(scene):
    """Render sprites through an identity view transform.

    Blender defaults to AgX, a filmic tone mapper. Two things go wrong when a
    sprite is rendered through it.

    The sprite is an unlit emission render of an albedo texture, so any tone
    curve is pure distortion: the point is to reproduce the source texture, not
    to grade a lit scene.

    Worse, `film_transparent` output is premultiplied, so a partially covered
    edge texel stores ``colour * coverage`` -- a very small number. A filmic
    curve lifts small values hard, so the stored RGB gets pushed up while alpha
    stays put, and the texel's effective straight colour ``rgb / alpha`` runs
    away toward white. That is a white rim one texel wide around every
    silhouette (measured on a teal character: fringe luma 240 against a body
    luma of 103). Standard leaves the fringe at the body's own colour.
    """
    view_settings = getattr(scene, "view_settings", None)
    if view_settings is None:
        return
    # `view_transform` and `look` are filled in from the active OCIO config, so
    # a headless bpy reports their enum as just ("NONE",) until it resolves.
    # _set_enum_if_available would read that as "unsupported" and silently skip
    # the assignment, so set these directly and let the exception be the test.
    for attr, value in (
        ("view_transform", "Standard"),
        ("look", "None"),
        ("exposure", 0.0),
        ("gamma", 1.0),
    ):
        if not hasattr(view_settings, attr):
            continue
        try:
            setattr(view_settings, attr, value)
        except Exception:
            pass


def _configure_sprite_render(scene, engine):
    """Keep orthographic sprite renders cheap: flat/textured, transparent PNGs."""
    scene.render.engine = engine
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    _configure_sprite_colour_management(scene)

    if engine == "CYCLES":
        device = _configure_cycles_device(scene)
        print(f"[sidecar_texture] Cycles device: {device}")
        return

    if engine == "BLENDER_WORKBENCH":
        shading = getattr(getattr(scene, "display", None), "shading", None)
        if shading is not None:
            _set_enum_if_available(shading, "light", "FLAT")
            if not _set_enum_if_available(shading, "color_type", "TEXTURE"):
                _set_enum_if_available(shading, "color_type", "MATERIAL")
            _set_enum_if_available(shading, "background_type", "TRANSPARENT")
            if hasattr(shading, "show_object_outline"):
                shading.show_object_outline = False
            if hasattr(shading, "show_cavity"):
                shading.show_cavity = False
        display = getattr(scene, "display", None)
        if display is not None:
            _set_enum_if_available(display, "render_aa", "8")
            _set_enum_if_available(display, "viewport_aa", "8")
        return

    if engine.startswith("BLENDER_EEVEE"):
        eevee = getattr(scene, "eevee", None)
        if eevee is None:
            return
        samples = _sprite_render_samples()
        for attr in ("taa_render_samples", "taa_samples"):
            if hasattr(eevee, attr):
                setattr(eevee, attr, samples)
        for attr in (
            "use_gtao",
            "use_bloom",
            "use_motion_blur",
            "use_shadows",
            "use_volumetric_shadows",
            "use_taa_reprojection",
        ):
            if hasattr(eevee, attr):
                setattr(eevee, attr, False)
        for attr in (
            "shadow_ray_count",
            "shadow_step_count",
            "volumetric_samples",
            "volumetric_shadow_samples",
        ):
            if hasattr(eevee, attr):
                setattr(eevee, attr, 1)


def sprite_render_device_label():
    """Which physical device the current render engine will actually use.

    Eevee has no device selector: it renders on whatever GPU the process'
    OpenGL context landed on, which on a hybrid-graphics laptop is the
    integrated one. Reporting it is the only way a build log can show that.
    """
    engine = str(bpy.context.scene.render.engine)
    if engine == "CYCLES":
        cycles = getattr(bpy.context.scene, "cycles", None)
        if cycles is None:
            return "unknown"
        if getattr(cycles, "device", "CPU") != "GPU":
            return "CPU"
        try:
            preferences = bpy.context.preferences.addons["cycles"].preferences
            enabled = [d.name for d in preferences.devices if d.use]
            return f"{preferences.compute_device_type}: {', '.join(enabled)}"
        except (AttributeError, KeyError):
            return "GPU"
    try:
        import gpu

        return gpu.platform.renderer_get()
    except Exception:
        return "unknown"


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
    engine = _pick_render_engine(scene)
    print(f"[sidecar_texture] Rendering sprite at {resolution}x{resolution} with {engine}...")

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.filepath = output_path
    _configure_sprite_render(scene, engine)

    try:
        started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        print(
            f"[sidecar_texture] Sprite rendered to: {output_path} "
            f"({time.perf_counter() - started:.3f}s)"
        )
        return True
    except Exception as exc:
        print(f"[sidecar_texture] Render failed: {exc}")
        _create_placeholder_atlas(output_path, resolution)
        return True


# --- sprite lighting ------------------------------------------------------
#
# The native side owns this look. `SpriteLighting` in
# flatRig/include/pipeline/splat_frame_render.hpp shades a Gaussian cloud with a
# wrapped Lambert term, optional cel banding and an additive Blinn-Phong
# highlight, and the viewer's 3D preview shades live with the same model. A mesh
# asset -- TRELLIS.2.cpp output, or a TripoSplat asset exported as geometry --
# has no cloud, so before this it rendered unlit whatever the light gizmo said.
#
# So the formula below is a transcription, not a design. Deliberately *not*
# Blender lamps and a Principled BSDF: those would give a physically-based
# result that looks nothing like the splat path under the same controls, and
# they would drag shadows and GI into a render that is currently 16 samples of
# flat emission. Evaluating the same arithmetic in a node graph and feeding it
# to an Emission keeps the render setup, the speed and the look identical.
#
#   lambert   = clamp((N.L + wrap) / (1 + wrap), 0, 1)
#   banded    = clamp(floor(lambert * bands) / (bands - 1), 0, 1)     [bands >= 2]
#   highlight = max(0, N.H) ** shininess                              [specular]
#   out       = albedo * (ambient + diffuse * lambert) + specular * highlight
#
# One honest difference: the native path multiplies the cloud's stored colour,
# while these nodes multiply the base-colour texture after Blender has decoded
# it to linear. The controls and the shape of the falloff match; the ramp
# through the midtones is slightly different. Rendering the two side by side is
# not a thing the product does -- an asset has a cloud or it does not -- so
# consistency of look and controls is what this buys, not pixel equality.


def _sprite_light_vectors(lighting):
    """``(to_light, half)`` unit vectors, matching ``light_vector`` in C++.

    The spec states where the light *travels*; a Lambert term wants the
    direction from the surface towards it, hence the negation. A zero or
    non-finite direction is not a light, so it falls back to the
    azimuth/elevation pair the same way the native code does.
    """
    import math

    def _normalise(vec):
        length = math.sqrt(sum(component * component for component in vec))
        if not math.isfinite(length) or length <= 1e-12:
            return None
        return [component / length for component in vec]

    to_light = None
    raw = lighting.get("direction")
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        try:
            to_light = _normalise([-float(component) for component in raw])
        except (TypeError, ValueError):
            to_light = None
    if to_light is None:
        azimuth = math.radians(float(lighting.get("azimuth", 135.0)))
        elevation = math.radians(float(lighting.get("elevation", 35.0)))
        to_light = [
            math.sin(azimuth) * math.cos(elevation),
            math.cos(azimuth) * math.cos(elevation),
            math.sin(elevation),
        ]

    # `to_eye` is the view's depth axis: it already points from the model back
    # at the camera. Light and eye exactly opposed leaves no half vector, and
    # no highlight to place -- any finite value does, the specular term is 0.
    to_eye = _normalise([float(component) for component in
                         lighting.get("to_eye", (0.0, 1.0, 0.0))]) or [0.0, 1.0, 0.0]
    half = _normalise([a + b for a, b in zip(to_light, to_eye)]) or list(to_eye)
    return to_light, half


def _sprite_light_term(lighting, name, default_colour, default_intensity):
    block = lighting.get(name)
    colour, intensity = default_colour, default_intensity
    if isinstance(block, dict):
        raw = block.get("color")
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            try:
                colour = tuple(float(component) for component in raw[:3])
            except (TypeError, ValueError):
                colour = default_colour
        if "intensity" in block:
            try:
                intensity = float(block["intensity"])
            except (TypeError, ValueError):
                intensity = default_intensity
    return tuple(component * intensity for component in colour), intensity


def sprite_lighting_enabled(lighting):
    return bool(isinstance(lighting, dict) and lighting.get("enabled"))


def _build_lit_material(original_material, lighting):
    """Emission-only copy whose colour is the shaded base colour.

    Still an emission surface: nothing here casts, receives or bounces light.
    The shading is arithmetic on the interpolated normal, which is what makes it
    reproducible and what makes it agree with the cloud shader.
    """
    material = original_material.copy()
    material.use_nodes = True
    if hasattr(material, "use_backface_culling"):
        material.use_backface_culling = False
    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output_node = next(
        (node for node in nodes if node.bl_idname == "ShaderNodeOutputMaterial"), None
    )
    principled_node = next(
        (node for node in nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"), None
    )
    if output_node is None:
        output_node = nodes.new("ShaderNodeOutputMaterial")

    # --- albedo, exactly as the unlit path resolves it ---
    albedo_socket = None
    albedo_value = (1.0, 1.0, 1.0, 1.0)
    if principled_node is not None:
        base_input = principled_node.inputs["Base Color"]
        if base_input.links:
            albedo_socket = base_input.links[0].from_socket
        else:
            albedo_value = base_input.default_value

    to_light, half = _sprite_light_vectors(lighting)
    ambient, _ = _sprite_light_term(lighting, "ambient", (0.45, 0.52, 0.68), 0.35)
    diffuse, _ = _sprite_light_term(lighting, "diffuse", (1.0, 0.98, 0.94), 1.0)
    specular, specular_intensity = _sprite_light_term(
        lighting, "specular", (1.0, 1.0, 1.0), 0.0
    )
    try:
        softness = min(1.0, max(0.0, float(lighting.get("softness", 0.35))))
    except (TypeError, ValueError):
        softness = 0.35
    try:
        bands = int(lighting.get("bands", 0) or 0)
    except (TypeError, ValueError):
        bands = 0
    bands = bands if bands >= 2 else 0
    try:
        shininess = max(1.0, float(lighting.get("shininess", 24.0)))
    except (TypeError, ValueError):
        shininess = 24.0

    def _new(idname, **attrs):
        node = nodes.new(idname)
        for key, value in attrs.items():
            setattr(node, key, value)
        return node

    def _vec(vector):
        node = _new("ShaderNodeCombineXYZ")
        for socket, component in zip("XYZ", vector):
            node.inputs[socket].default_value = float(component)
        return node.outputs["Vector"]

    # Geometry normal is already world space, which is the space the light
    # direction is expressed in (pipeline-normalised world: Z up, model -Y).
    geometry = _new("ShaderNodeNewGeometry")

    dot = _new("ShaderNodeVectorMath", operation="DOT_PRODUCT")
    links.new(geometry.outputs["Normal"], dot.inputs[0])
    links.new(_vec(to_light), dot.inputs[1])

    # clamp((N.L + wrap) / (1 + wrap), 0, 1)
    wrapped = _new("ShaderNodeMath", operation="ADD")
    links.new(dot.outputs["Value"], wrapped.inputs[0])
    wrapped.inputs[1].default_value = softness
    lambert = _new("ShaderNodeMath", operation="MULTIPLY", use_clamp=True)
    links.new(wrapped.outputs["Value"], lambert.inputs[0])
    lambert.inputs[1].default_value = 1.0 / (1.0 + softness)
    lambert_socket = lambert.outputs["Value"]

    if bands:
        scaled = _new("ShaderNodeMath", operation="MULTIPLY")
        links.new(lambert_socket, scaled.inputs[0])
        scaled.inputs[1].default_value = float(bands)
        floored = _new("ShaderNodeMath", operation="FLOOR")
        links.new(scaled.outputs["Value"], floored.inputs[0])
        stepped = _new("ShaderNodeMath", operation="DIVIDE", use_clamp=True)
        links.new(floored.outputs["Value"], stepped.inputs[0])
        stepped.inputs[1].default_value = float(bands - 1)
        lambert_socket = stepped.outputs["Value"]

    # albedo * (ambient + diffuse * lambert)
    diffuse_term = _new("ShaderNodeVectorMath", operation="SCALE")
    links.new(_vec(diffuse), diffuse_term.inputs[0])
    links.new(lambert_socket, diffuse_term.inputs["Scale"])
    lit = _new("ShaderNodeVectorMath", operation="ADD")
    links.new(diffuse_term.outputs["Vector"], lit.inputs[0])
    links.new(_vec(ambient), lit.inputs[1])

    shaded = _new("ShaderNodeVectorMath", operation="MULTIPLY")
    if albedo_socket is not None:
        links.new(albedo_socket, shaded.inputs[0])
    else:
        for index, component in enumerate(albedo_value[:3]):
            shaded.inputs[0].default_value[index] = float(component)
    links.new(lit.outputs["Vector"], shaded.inputs[1])
    colour_socket = shaded.outputs["Vector"]

    # + specular * max(0, N.H) ** shininess, additive and untinted by albedo:
    # a highlight is light bouncing off the surface, not light it absorbed.
    if specular_intensity > 0.0:
        n_dot_h = _new("ShaderNodeVectorMath", operation="DOT_PRODUCT")
        links.new(geometry.outputs["Normal"], n_dot_h.inputs[0])
        links.new(_vec(half), n_dot_h.inputs[1])
        clamped = _new("ShaderNodeMath", operation="MAXIMUM")
        links.new(n_dot_h.outputs["Value"], clamped.inputs[0])
        clamped.inputs[1].default_value = 0.0
        highlight = _new("ShaderNodeMath", operation="POWER")
        links.new(clamped.outputs["Value"], highlight.inputs[0])
        highlight.inputs[1].default_value = shininess
        highlight_socket = highlight.outputs["Value"]
        if bands:
            # A cel highlight is a hard shape, not a gradient falling off into
            # the band below it.
            hard = _new("ShaderNodeMath", operation="GREATER_THAN")
            links.new(highlight_socket, hard.inputs[0])
            hard.inputs[1].default_value = 0.5
            highlight_socket = hard.outputs["Value"]
        specular_term = _new("ShaderNodeVectorMath", operation="SCALE")
        links.new(_vec(specular), specular_term.inputs[0])
        links.new(highlight_socket, specular_term.inputs["Scale"])
        summed = _new("ShaderNodeVectorMath", operation="ADD")
        links.new(colour_socket, summed.inputs[0])
        links.new(specular_term.outputs["Vector"], summed.inputs[1])
        colour_socket = summed.outputs["Vector"]

    emission_node = _new("ShaderNodeEmission")
    emission_node.name = "_sidecar_emission"
    emission_node.inputs["Strength"].default_value = 1.0
    links.new(colour_socket, emission_node.inputs["Color"])

    for link in list(output_node.inputs["Surface"].links):
        links.remove(link)
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
    return material


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


def _apply_unlit_materials(objects, lighting=None):
    """Swap materials to temporary render copies and return a restore token.

    ``lighting`` omitted or disabled keeps the historical behaviour exactly:
    emission-only copies, so the sprite is the source texture reproduced. The
    native side promises the same thing on the cloud path -- "an unlit sprite is
    still the cloud's own baked colour, byte for byte" -- and that guarantee is
    why the disabled case routes to the untouched builder rather than to a lit
    one with neutral terms, which would not be a no-op.
    """
    lit = sprite_lighting_enabled(lighting)
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
            unlit_material = (
                _build_lit_material(material, lighting) if lit
                else _build_unlit_material(material)
            )
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


#: Coverage below which a saved texel's colour cannot be trusted. Blender's
#: render buffer is premultiplied but its 8-bit PNG writer saves *straight*
#: (unassociated) alpha, so it divides the covered colour by a very small,
#: rounding-limited alpha for the fringe texels and writes whatever comes out
#: -- and plain black wherever nothing was rasterized at all. Below this floor
#: the colour is taken from the neighbourhood instead.
_MIN_RELIABLE_COVERAGE = 12.0 / 255.0

#: Radius of the alpha-weighted colour bleed that fills the unreliable texels.
_COLOR_BLEED_RADIUS = 3.0


def _straight_rgb_with_bleed(straight_rgb, coverage):
    """Return the render's straight colour, bled inward where coverage is low.

    The saved PNG already carries straight (unassociated) RGB, so no
    un-premultiply is needed -- dividing it by coverage a second time is what
    painted a white rim around every silhouette (measured: 50-74% of a sprite's
    fringe texels clipped to white, fringe luma ~2x the interior). Only the
    texels under ``_MIN_RELIABLE_COVERAGE`` carry no usable colour of their own;
    those take a coverage-weighted average of their neighbourhood, so the
    fringe -- and any texel the soft ring cut turns opaque where the render was
    transparent -- inherits the colour of the geometry it belongs to.
    """
    from PIL import ImageFilter

    blurred_coverage = _filter_alpha(coverage, ImageFilter.GaussianBlur(_COLOR_BLEED_RADIUS))
    bled = np.empty_like(straight_rgb)
    for channel in range(straight_rgb.shape[-1]):
        blurred_channel = _filter_alpha(
            straight_rgb[..., channel] * coverage,
            ImageFilter.GaussianBlur(_COLOR_BLEED_RADIUS),
        )
        bled[..., channel] = blurred_channel / np.maximum(blurred_coverage, 1e-6)

    reliable = (coverage >= _MIN_RELIABLE_COVERAGE)[..., None]
    # Where even the neighbourhood carries no coverage there is no colour to
    # recover; those texels are fully transparent, so the value is irrelevant.
    recovered = np.where(reliable, straight_rgb, bled)
    return np.clip(recovered, 0.0, 1.0)


def _premultiply_saved_sprite(output_path):
    """Rewrite a finished sprite PNG with premultiplied RGB.

    Blender saves straight alpha, but the sprite atlas declares ``pma: true``
    and the viewer uploads the pages as already-premultiplied. Handing a
    straight-alpha page to a premultiplied blend adds the full fringe colour on
    top of the background instead of ``colour * alpha``, which is the white halo
    that outlines every sprite and its seams. Sprites that go through the soft
    ring cut are premultiplied there; this covers the rest -- a welded sprite, a
    lone accessory object, any part with no borrowed ring.
    """
    if not _soft_cut_support_available() or not os.path.isfile(output_path):
        return False
    from PIL import Image

    image = Image.open(output_path).convert("RGBA")
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    alpha = pixels[..., 3]
    premultiplied = np.empty_like(pixels)
    premultiplied[..., :3] = _straight_rgb_with_bleed(pixels[..., :3], alpha) * alpha[..., None]
    premultiplied[..., 3] = alpha
    Image.fromarray(
        np.rint(np.clip(premultiplied * 255.0, 0.0, 255.0)).astype(np.uint8)
    ).save(output_path)
    return True


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

        # Blender's PNG writer saves straight (unassociated) alpha, so `color`
        # holds the plain surface colour and the soft cut only has to pair it
        # with the new matte. Both consumers of the atlas -- the `pma: true`
        # header and the viewer's "premultiplied-alpha" upload -- expect
        # premultiplied pages, so premultiply by `output_alpha` here; leaving
        # the page straight is what outlines every sprite with a white halo.
        rgb_straight = _straight_rgb_with_bleed(color[..., :3], coverage)
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
    lighting=None,
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

        restore_info, created_materials = _apply_unlit_materials(
            render_objects, lighting
        )
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
    lighting=None,
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
            lighting=lighting,
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
    restore_info, created_materials = _apply_unlit_materials(mesh_objects, lighting)

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
    lighting=None,
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

    restore_info, created_materials = _apply_unlit_materials([render_obj], lighting)

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
        soft_cut = False
        if success and core_obj is not None:
            soft_cut = bool(
                _apply_soft_ring_cut(
                    scene,
                    render_obj,
                    core_obj,
                    output_path,
                    resolution=resolution,
                )
            )
        if success and not soft_cut:
            # No borrowed ring (a welded sprite, a lone accessory object) or no
            # soft-cut support: the render is still straight alpha and the atlas
            # promises premultiplied pages, so convert it here.
            _premultiply_saved_sprite(output_path)
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
