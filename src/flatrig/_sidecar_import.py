"""Load helpers from the sidecar's own Blender worker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SIDE_CAR_ROOT = Path(__file__).resolve().parents[2]
BLENDER_SCRIPT = SIDE_CAR_ROOT / "tools" / "blender_scene_io.py"
MODULE_NAME = "flatrig_sidecar_blender_scene_io"


def load_blender_scene_io_module() -> ModuleType:
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]

    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_module is not None and main_file:
        try:
            if Path(main_file).resolve() == BLENDER_SCRIPT.resolve():
                sys.modules[MODULE_NAME] = main_module
                return main_module
        except OSError:
            pass

    spec = importlib.util.spec_from_file_location(MODULE_NAME, BLENDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load blender_scene_io.py from {BLENDER_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def compute_projection_frame(points_2d, margin=0.06):
    return load_blender_scene_io_module().compute_projection_frame(points_2d, margin=margin)


def project_points_to_uv(points_2d, frame):
    return load_blender_scene_io_module().project_points_to_uv(points_2d, frame)


def orthonormalize_2x2(matrix, epsilon=None):
    return load_blender_scene_io_module().orthonormalize_2x2(matrix, epsilon=epsilon)


def safe_inverse_2x2(matrix, epsilon=None):
    return load_blender_scene_io_module().safe_inverse_2x2(matrix, epsilon=epsilon)


def orthonormalize_3x3(matrix):
    return load_blender_scene_io_module().orthonormalize_3x3(matrix)
