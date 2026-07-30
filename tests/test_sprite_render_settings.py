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


def test_sample_override_is_clamped_to_a_usable_range(monkeypatch):
    monkeypatch.setenv("FLATRIG_SPRITE_RENDER_SAMPLES", "0")
    assert texture._sprite_render_samples() == 1

    monkeypatch.setenv("FLATRIG_SPRITE_RENDER_SAMPLES", "4096")
    assert texture._sprite_render_samples() == 64

    monkeypatch.setenv("FLATRIG_SPRITE_RENDER_SAMPLES", "not-a-number")
    assert texture._sprite_render_samples() == texture.DEFAULT_SPRITE_RENDER_SAMPLES

    monkeypatch.setenv("FLATRIG_SPRITE_RENDER_SAMPLES", "8")
    assert texture._sprite_render_samples() == 8
