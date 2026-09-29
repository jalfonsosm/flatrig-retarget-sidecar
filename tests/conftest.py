"""Shared fixtures for the sidecar test suite."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

#: Renders a few pixels through every engine the sprite path would choose on
#: this machine, in a *separate interpreter*.
#:
#: Blender does not raise when it cannot bring up a GPU context -- it calls
#: ``abort()``. That is a SIGABRT: no exception, no traceback, nothing a
#: ``try`` around ``bpy.ops.render.render`` can catch, and it takes the whole
#: pytest session with it. On the GPU-less ubuntu CI runner exactly that
#: happened, and the abort inside the first sprite-lighting test also threw
#: away the ~60 unrelated tests queued behind it (exit code 134).
#:
#: Running the probe out-of-process is the only way to survive the answer.
_RENDER_PROBE = """
import sys

import bpy

from flatrig import texture

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.camera_add()
scene = bpy.context.scene
scene.camera = bpy.context.object
bpy.ops.mesh.primitive_cube_add(location=(0, 0, -5))
scene.render.resolution_x = 4
scene.render.resolution_y = 4
scene.render.filepath = sys.argv[-1]

# The lit and unlit paths can resolve to different engines (Cycles vs Eevee),
# and only one of them may be able to start here, so both are tried.
engines = []
for lit in (False, True):
    texture._LIT_SHADING_ACTIVE[0] = lit
    texture.reset_render_engine_cache()
    engine = texture._pick_render_engine(scene)
    if engine not in engines:
        engines.append(engine)
texture._LIT_SHADING_ACTIVE[0] = False
texture.reset_render_engine_cache()

for engine in engines:
    scene.render.engine = engine
    texture._configure_sprite_render(scene, engine)
    bpy.ops.render.render(write_still=True)
print("render-ok " + ",".join(engines))
"""

#: Set to a truthy value where rendering is *expected* to work (a dev machine,
#: a GPU runner). The probe then fails the run instead of skipping, so a real
#: render regression cannot hide behind a skip.
_REQUIRE_ENV = "FLATRIG_REQUIRE_SPRITE_RENDER"


def _probe_render(tmp_dir: str) -> str | None:
    """None when Blender can render here, else why it cannot."""
    try:
        finished = subprocess.run(
            [sys.executable, "-c", _RENDER_PROBE, os.path.join(tmp_dir, "probe.png")],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return "the probe render did not finish within 600s"
    except OSError as exc:
        return f"the probe render could not be started ({exc})"
    if finished.returncode == 0:
        return None
    tail = (finished.stderr or finished.stdout or "").strip().splitlines()
    detail = tail[-1] if tail else "no output"
    if finished.returncode < 0:
        return f"Blender died on signal {-finished.returncode} while rendering: {detail}"
    return f"Blender could not render (exit {finished.returncode}): {detail}"


@pytest.fixture(scope="session")
def _render_probe_result(tmp_path_factory) -> str | None:
    return _probe_render(str(tmp_path_factory.mktemp("render-probe")))


@pytest.fixture
def blender_can_render(_render_probe_result: str | None) -> None:
    """Skip the test unless this machine can actually complete a render.

    Ask for this in any test that reaches `bpy.ops.render.render`. Tests that
    monkeypatch the render away do not need it.
    """
    if _render_probe_result is None:
        return
    if os.environ.get(_REQUIRE_ENV, "").strip().lower() not in {"", "0", "false", "no"}:
        pytest.fail(
            f"{_REQUIRE_ENV} is set, but this machine cannot render: {_render_probe_result}"
        )
    pytest.skip(f"no usable Blender render backend here: {_render_probe_result}")
