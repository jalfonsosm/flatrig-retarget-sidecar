import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import blender_json_io  # noqa: E402
import blender_view  # noqa: E402


def test_matrix3_to_json_shape_and_floats():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = blender_json_io.matrix3_to_json(matrix)
    assert result == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    assert all(isinstance(value, float) for row in result for value in row)


def test_matrix4_to_json_shape_and_floats():
    matrix = [[float(row * 4 + col) for col in range(4)] for row in range(4)]
    result = blender_json_io.matrix4_to_json(matrix)
    assert result == matrix
    assert all(isinstance(value, float) for row in result for value in row)


def test_vector_to_json_truncates_to_three_components():
    assert blender_json_io.vector_to_json((1, 2, 3)) == [1.0, 2.0, 3.0]


def test_weights_to_json_sorts_and_normalizes_types():
    weights = [{"2": 0.25, "1": 0.75}, None, {}]
    result = blender_json_io.weights_to_json(weights)
    assert result == [[[1, 0.75], [2, 0.25]], [], []]
    assert isinstance(result[0][0][0], int)
    assert isinstance(result[0][0][1], float)


def test_view_config_round_trips_through_json():
    config = blender_view.get_view_config("side")
    payload = blender_json_io.view_config_to_json(config)

    assert payload["preset"] == "side"
    assert len(payload["view_dir"]) == 3
    assert np.isclose(np.linalg.norm(payload["view_dir"]), 1.0)
    assert np.asarray(payload["basis_2d"], dtype=np.float64).shape == (2, 3)
    assert np.asarray(payload["basis_3d"], dtype=np.float64).shape == (3, 3)
    assert isinstance(payload["auto_lateral_flip"], bool)
