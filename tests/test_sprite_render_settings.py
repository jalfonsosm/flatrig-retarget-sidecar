"""Sprite renders must stay tuned for unlit emission geometry.

Eevee's stock 64 temporal samples only buy edge antialiasing here, because the
sprite path replaces every material with an emission shader and disables lights.
Regressing back to the stock settings costs roughly an order of magnitude in
wall-clock time per part, so the tuned values are asserted directly.
"""

import bpy

from flatrig import texture


def _eevee_scene():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    return scene


def _samples(scene):
    eevee = scene.eevee
    for attr in ("taa_render_samples", "taa_samples"):
        if hasattr(eevee, attr):
            return getattr(eevee, attr)
    raise AssertionError("no Eevee temporal sample property found")


def test_sprite_render_uses_tuned_sample_count(monkeypatch):
    monkeypatch.delenv("FLATRIG_SPRITE_RENDER_SAMPLES", raising=False)
    scene = _eevee_scene()
    texture._configure_sprite_render(scene, "BLENDER_EEVEE")

    assert _samples(scene) == texture.DEFAULT_SPRITE_RENDER_SAMPLES
    assert texture.DEFAULT_SPRITE_RENDER_SAMPLES < 64


def test_sprite_render_forces_transparent_rgba_png(monkeypatch):
    monkeypatch.delenv("FLATRIG_SPRITE_RENDER_SAMPLES", raising=False)
    scene = _eevee_scene()
    texture._configure_sprite_render(scene, "BLENDER_EEVEE")

    assert scene.render.film_transparent is True
    assert scene.render.image_settings.file_format == "PNG"
    assert scene.render.image_settings.color_mode == "RGBA"


def test_sprite_render_disables_effects_that_emission_cannot_use(monkeypatch):
    monkeypatch.delenv("FLATRIG_SPRITE_RENDER_SAMPLES", raising=False)
    scene = _eevee_scene()
    eevee = scene.eevee
    for attr in ("use_gtao", "use_bloom", "use_motion_blur", "use_shadows"):
        if hasattr(eevee, attr):
            setattr(eevee, attr, True)

    texture._configure_sprite_render(scene, "BLENDER_EEVEE")

    for attr in ("use_gtao", "use_bloom", "use_motion_blur", "use_shadows"):
        if hasattr(eevee, attr):
            assert getattr(eevee, attr) is False, f"{attr} should be off for sprite renders"


def test_default_engine_follows_gpu_availability(monkeypatch):
    """Cycles is the default only where it can reach a GPU.

    Eevee cannot choose a device, so on a hybrid-graphics machine it renders on
    the integrated GPU. Cycles can be pointed at the discrete card and measured
    faster there -- but slower than Eevee when it falls back to the CPU, so the
    default has to depend on what the machine actually has.
    """
    monkeypatch.delenv("FLATRIG_SPRITE_RENDER_ENGINE", raising=False)
    scene = bpy.context.scene

    monkeypatch.setattr(texture, "cycles_gpu_devices", lambda: ("CUDA", ["Test GPU"]))
    assert texture._pick_render_engine(scene) == "CYCLES"

    monkeypatch.setattr(texture, "cycles_gpu_devices", lambda: None)
    assert texture._pick_render_engine(scene).startswith("BLENDER_EEVEE")


def test_cycles_can_be_requested_even_though_it_is_not_in_the_engine_enum(monkeypatch):
    """Add-on engines never appear in render.engine's enum_items.

    An enum-membership check therefore reported Cycles as unavailable and
    silently fell back to Eevee, which made the env override a no-op.
    """
    scene = bpy.context.scene
    enum_property = scene.render.bl_rna.properties.get("engine")
    listed = {item.identifier for item in (enum_property.enum_items if enum_property else [])}
    assert "CYCLES" not in listed, "guard: Cycles is expected to be absent from the enum"

    monkeypatch.setenv("FLATRIG_SPRITE_RENDER_ENGINE", "CYCLES")
    assert texture._pick_render_engine(scene) == "CYCLES"

    # Probing must not leave the scene on the probed engine.
    monkeypatch.delenv("FLATRIG_SPRITE_RENDER_ENGINE", raising=False)
    scene.render.engine = "BLENDER_EEVEE"
    texture._pick_render_engine(scene)
    assert scene.render.engine == "BLENDER_EEVEE"


def test_unknown_engine_request_falls_back_instead_of_raising(monkeypatch):
    monkeypatch.setenv("FLATRIG_SPRITE_RENDER_ENGINE", "NOT_A_REAL_ENGINE")
    assert texture._pick_render_engine(bpy.context.scene).startswith("BLENDER_EEVEE")


def test_cycles_is_configured_for_an_unlit_emission_pass(monkeypatch):
    monkeypatch.delenv("FLATRIG_SPRITE_RENDER_SAMPLES", raising=False)
    scene = bpy.context.scene
    texture._ensure_cycles_addon()
    scene.render.engine = "CYCLES"
    texture._configure_sprite_render(scene, "CYCLES")

    assert scene.render.film_transparent is True
    assert scene.render.image_settings.color_mode == "RGBA"
    assert scene.cycles.samples == texture.DEFAULT_SPRITE_RENDER_SAMPLES
    # No bounce carries signal in an emission-only sprite render.
    for attr in ("max_bounces", "diffuse_bounces", "glossy_bounces"):
        if hasattr(scene.cycles, attr):
            assert getattr(scene.cycles, attr) == 0
    # The device label must name something concrete so build logs can prove
    # which GPU actually rendered.
    assert texture.sprite_render_device_label()
    scene.render.engine = "BLENDER_EEVEE"


def test_sample_override_is_clamped_to_a_usable_range(monkeypatch):
    monkeypatch.setenv("FLATRIG_SPRITE_RENDER_SAMPLES", "0")
    assert texture._sprite_render_samples() == 1

    monkeypatch.setenv("FLATRIG_SPRITE_RENDER_SAMPLES", "4096")
    assert texture._sprite_render_samples() == 64

    monkeypatch.setenv("FLATRIG_SPRITE_RENDER_SAMPLES", "not-a-number")
    assert texture._sprite_render_samples() == texture.DEFAULT_SPRITE_RENDER_SAMPLES

    monkeypatch.setenv("FLATRIG_SPRITE_RENDER_SAMPLES", "8")
    assert texture._sprite_render_samples() == 8
