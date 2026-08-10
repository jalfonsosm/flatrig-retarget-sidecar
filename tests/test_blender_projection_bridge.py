from __future__ import annotations

import numpy as np

from flatrig._blender_projection import (
    compose_projection_plane_point,
    get_projection_reference_inverse,
    get_projection_reference_matrix,
    project_direction_ortho,
    project_point_ortho,
    transform_direction_from_projection_space,
    transform_direction_to_projection_space,
    transform_point_from_projection_space,
    transform_points_to_projection_space,
)


def test_projection_bridge_keeps_points_and_directions_distinct() -> None:
    projection_matrix = np.eye(4, dtype=np.float64)
    projection_matrix[:3, 3] = (4.0, 5.0, 6.0)

    point = transform_point_from_projection_space((1.0, 2.0, 3.0), projection_matrix)
    direction = transform_direction_from_projection_space(
        (1.0, 2.0, 3.0), projection_matrix
    )

    np.testing.assert_allclose(point, (5.0, 7.0, 9.0))
    np.testing.assert_allclose(direction, (1.0, 2.0, 3.0))


def test_projection_bridge_composes_camera_plane_point() -> None:
    view = {
        "right_axis": (1.0, 0.0, 0.0),
        "up_axis": (0.0, 0.0, 1.0),
        "depth_axis": (0.0, -1.0, 0.0),
    }

    point = compose_projection_plane_point(2.0, 3.0, 4.0, view)

    np.testing.assert_allclose(point, (2.0, -4.0, 3.0))


def test_world_projection_needs_no_private_runtime() -> None:
    assert get_projection_reference_matrix(None, projection_space="world") is None
    assert get_projection_reference_inverse(None, projection_space="world") is None


def test_projection_bridge_projects_points_and_directions() -> None:
    view = {
        "basis_2d": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    }
    projection_inverse = np.eye(4, dtype=np.float64)
    projection_inverse[:3, 3] = (10.0, 20.0, 30.0)

    np.testing.assert_allclose(
        transform_points_to_projection_space((1.0, 2.0, 3.0), projection_inverse),
        (11.0, 22.0, 33.0),
    )
    np.testing.assert_allclose(
        transform_direction_to_projection_space((1.0, 2.0, 3.0), projection_inverse),
        (1.0, 2.0, 3.0),
    )
    assert project_point_ortho((1.0, 2.0, 3.0), view, projection_inverse) == (
        11.0,
        33.0,
    )
    np.testing.assert_allclose(
        project_direction_ortho((1.0, 2.0, 3.0), view, projection_inverse),
        (1.0, 3.0),
    )
