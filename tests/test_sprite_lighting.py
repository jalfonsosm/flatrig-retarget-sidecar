"""Sprite lighting for geometric meshes.

`SpriteLighting` in FlatRig's `flatRig/include/pipeline/splat_frame_render.hpp`
owns this look, and shades a Gaussian cloud with it. A mesh asset has no cloud,
so before this it rendered unlit whatever the light gizmo said. These tests pin
the two properties that make the mesh path a real substitute rather than merely
"also lit": the silhouette is untouched, and the shading is the native formula
rather than whatever a Blender lamp would produce.
"""

import bpy  # noqa: F401 -- importing bpy registers Blender's bmesh module
import numpy as np
import pytest
from PIL import Image

from flatrig import texture as tx

RESOLUTION = 256

VIEW_CFG = {
    "view_dir": (0.0, -1.0, 0.0),
    "right_axis": (1.0, 0.0, 0.0),
    "up_axis": (0.0, 0.0, 1.0),
    "depth_axis": (0.0, -1.0, 0.0),
}
PROJECTION_FRAME = {"center_x": 0.0, "center_y": 0.0, "span": 2.4}

ALBEDO = (0.8, 0.2, 0.2)
LIGHT = {
    "enabled": True,
    "direction": [-1.0, 0.0, -1.0],
    "ambient": {"color": [0.45, 0.52, 0.68], "intensity": 0.35},
    "diffuse": {"color": [1.0, 0.98, 0.94], "intensity": 1.0},
    "softness": 0.35,
    "bands": 0,
}


@pytest.fixture
def sphere():
    """A smooth-shaded sphere: every normal in the hemisphere, one flat albedo."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0, location=(0, 0, 0), segments=64, ring_count=32
    )
    bpy.ops.object.shade_smooth()
    obj = bpy.context.active_object
    material = bpy.data.materials.new("flat_albedo")
    material.use_nodes = True
    principled = next(
        node
        for node in material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    principled.inputs["Base Color"].default_value = (*ALBEDO, 1.0)
    obj.data.materials.append(material)
    return obj


def _render(obj, tmp_path, name, lighting):
    output = tmp_path / f"{name}.png"
    assert tx.render_preview_sprite(
        obj,
        VIEW_CFG,
        PROJECTION_FRAME,
        str(output),
        resolution=RESOLUTION,
        depth_center=0.0,
        bind_frame=None,
        lighting=lighting,
    )
    pixels = np.asarray(Image.open(output).convert("RGBA")).astype(np.float64)
    return pixels[..., :3], pixels[..., 3] > 250


def _to_srgb(linear):
    return np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055
    )


def test_lighting_shades_the_mesh_without_touching_the_silhouette(sphere, tmp_path):
    """The matte is geometry; a light must not move it.

    Sprites are composited by their alpha, so a light that changed coverage
    would shift the part's footprint in the atlas -- the sheet would no longer
    line up with the mesh it was baked from.
    """
    unlit_rgb, unlit_mask = _render(sphere, tmp_path, "unlit", None)
    lit_rgb, lit_mask = _render(sphere, tmp_path, "lit", LIGHT)

    assert unlit_mask.sum() == lit_mask.sum()
    assert np.array_equal(unlit_mask, lit_mask)

    # One channel, so this measures spatial variation and not the gap between
    # red and green in the albedo.
    assert unlit_rgb[..., 0][unlit_mask].std() < 1.0, "unlit sphere should be flat"
    assert lit_rgb[..., 0][lit_mask].std() > 15.0, "lit sphere should have a gradient"


def test_shading_matches_the_native_wrapped_lambert(sphere, tmp_path):
    """Spot-check the formula, not just that something changed.

    The centre pixel's normal faces the camera dead on, so N is the negated
    depth axis and every term is computable by hand:

        lambert = clamp((N.L + softness) / (1 + softness), 0, 1)
        out     = albedo * (ambient + diffuse * lambert)
    """
    lit_rgb, _ = _render(sphere, tmp_path, "lit", LIGHT)

    to_light = -np.array(LIGHT["direction"], dtype=float)
    to_light /= np.linalg.norm(to_light)
    normal = -np.array(VIEW_CFG["depth_axis"], dtype=float)

    softness = LIGHT["softness"]
    lambert = np.clip((normal @ to_light + softness) / (1.0 + softness), 0.0, 1.0)
    ambient = np.array(LIGHT["ambient"]["color"]) * LIGHT["ambient"]["intensity"]
    diffuse = np.array(LIGHT["diffuse"]["color"]) * LIGHT["diffuse"]["intensity"]
    expected = _to_srgb(np.clip(np.array(ALBEDO) * (ambient + diffuse * lambert), 0, 1))

    centre = lit_rgb[RESOLUTION // 2, RESOLUTION // 2]
    assert np.allclose(centre, expected * 255.0, atol=4.0), (
        f"expected {expected * 255.0}, rendered {centre}"
    )


def test_disabled_lighting_is_the_untouched_unlit_path(sphere, tmp_path):
    """Not "lit with neutral terms" -- the same bytes as before this existed.

    The native side makes the same promise on the cloud path ("an unlit sprite
    is still the cloud's own baked colour, byte for byte"), and neutral terms
    would not keep it: ambient defaults alone would tint the result.
    """
    omitted, _ = _render(sphere, tmp_path, "omitted", None)
    disabled, _ = _render(sphere, tmp_path, "disabled", {"enabled": False})

    assert np.array_equal(omitted, disabled)


def test_banding_quantises_the_falloff_into_steps(sphere, tmp_path):
    """Cel shading pushes the falloff onto a few plateaus.

    Measured as histogram mass, not as a distinct-value count: Cycles averages
    16 sub-samples per pixel and a sphere's normal turns appreciably within one
    pixel, so a banded render still contains many intermediate values along
    every step edge. What banding actually does is concentrate the *area* onto
    `bands` plateaus, and that is what this asserts. Measured on an M3 with
    bands=3: 76% of the interior on the top three levels, against 29% for the
    same light unbanded.
    """
    smooth, mask = _render(sphere, tmp_path, "smooth", LIGHT)
    banded, _ = _render(sphere, tmp_path, "banded", {**LIGHT, "bands": 3})

    # Silhouette texels blend with transparent background, so drop the border.
    interior = mask.copy()
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False

    def _top_three_share(image):
        counts = np.unique(image[..., 0][interior].round(), return_counts=True)[1]
        return np.sort(counts)[::-1][:3].sum() / counts.sum()

    assert _top_three_share(banded) > 0.60
    assert _top_three_share(smooth) < 0.40
