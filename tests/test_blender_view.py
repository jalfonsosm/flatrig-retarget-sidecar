import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import blender_view  # noqa: E402


def test_alias_view_config_is_normalized():
    config = blender_view.get_view_config("3q")

    assert config["preset"] == "three_quarter"
    assert np.isclose(np.linalg.norm(config["view_dir"]), 1.0)
    assert np.isclose(np.linalg.norm(config["right_axis"]), 1.0)
    assert np.isclose(np.linalg.norm(config["up_axis"]), 1.0)
    assert np.isclose(float(np.dot(config["view_dir"], config["right_axis"])), 0.0)
    assert np.isclose(float(np.dot(config["view_dir"], config["up_axis"])), 0.0)
    assert np.isclose(float(np.dot(config["right_axis"], config["up_axis"])), 0.0)


def test_roll_changes_2d_axes_without_changing_view_direction():
    base = blender_view.get_view_config("side")
    rolled = blender_view.get_view_config("side", roll_degrees=90)

    assert np.allclose(rolled["view_dir"], base["view_dir"])
    assert np.allclose(rolled["right_axis"], base["up_axis"])
    assert np.allclose(rolled["up_axis"], -base["right_axis"])


def test_empty_projection_frame_uses_unit_square():
    frame = blender_view.compute_projection_frame([])

    assert frame["span"] == 1.0
    assert frame["min_x"] == -0.5
    assert frame["max_x"] == 0.5
    assert frame["min_y"] == -0.5
    assert frame["max_y"] == 0.5


def test_invalid_custom_vector_fails_fast():
    with pytest.raises(ValueError, match="view_dir must be non-zero"):
        blender_view.get_view_config("custom", view_dir=(0.0, 0.0, 0.0))
