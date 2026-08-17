"""Contract tests for the per-splat skinning table.

The table is the hand-off that lets FlatRig pose a Gaussian cloud without
Blender: it is written here and read by the native sprite renderer, so the byte
layout is a contract between two repositories. These tests pin it.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import splat_utils  # noqa: E402


def _read_table(path):
    """Parse a FRSKIN1 file the way the native reader does."""
    data = Path(path).read_bytes()
    assert data[:8] == b"FRSKIN1\n"
    splat_count, bone_count, influences = struct.unpack_from("<III", data, 8)
    offset = 8 + 12

    names = []
    for _ in range(bone_count):
        (length,) = struct.unpack_from("<I", data, offset)
        offset += 4
        names.append(data[offset : offset + length].decode("utf-8"))
        offset += length

    slots = splat_count * influences
    indices = np.frombuffer(data, dtype="<u2", count=slots, offset=offset).reshape(
        splat_count, influences
    )
    offset += slots * 2
    weights = np.frombuffer(data, dtype="<f4", count=slots, offset=offset).reshape(
        splat_count, influences
    )
    offset += slots * 4
    assert offset == len(data), "the table must not carry trailing bytes"
    return names, indices, weights


def test_skin_table_round_trips(tmp_path):
    indices = np.array([[1, 2, 0, 0], [3, 0, 0, 0]], dtype=np.int32)
    weights = np.array([[0.6, 0.4, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    path = splat_utils._write_skin_table(
        tmp_path / "cloud.skin", ["__identity__", "hip", "spine", "head"], indices, weights
    )

    names, read_indices, read_weights = _read_table(path)
    assert names == ["__identity__", "hip", "spine", "head"]
    assert np.array_equal(read_indices, indices.astype(np.uint16))
    assert np.array_equal(read_weights, weights.astype(np.float32))


def test_slot_zero_is_the_identity_bone(tmp_path):
    # `_vertex_influence_tables` sends unweighted splats to slot 0 and the
    # renderer applies whatever matrix that slot names. It must never be a real
    # bone, or those splats would ride along with it.
    path = splat_utils._write_skin_table(
        tmp_path / "cloud.skin",
        ["__identity__", "hip"],
        np.zeros((1, 4), dtype=np.int32),
        np.array([[1.0, 0.0, 0.0, 0.0]]),
    )
    names, indices, weights = _read_table(path)
    assert names[0] == "__identity__"
    assert indices[0][0] == 0
    assert weights[0][0] == 1.0


def test_bone_names_carry_utf8(tmp_path):
    path = splat_utils._write_skin_table(
        tmp_path / "cloud.skin",
        ["__identity__", "brazo_izquierdo"],
        np.array([[1, 0, 0, 0]], dtype=np.int32),
        np.array([[1.0, 0.0, 0.0, 0.0]]),
    )
    names, _, _ = _read_table(path)
    assert names[1] == "brazo_izquierdo"


def test_table_shape_follows_the_influence_count(tmp_path):
    # The native side derives its stride from the header, so a table with a
    # different influence budget still has to describe itself correctly.
    indices = np.array([[1, 2]], dtype=np.int32)
    weights = np.array([[0.5, 0.5]])
    path = splat_utils._write_skin_table(
        tmp_path / "cloud.skin", ["__identity__", "hip", "spine"], indices, weights
    )
    data = Path(path).read_bytes()
    splat_count, bone_count, influences = struct.unpack_from("<III", data, 8)
    assert (splat_count, bone_count, influences) == (1, 3, 2)


def test_influence_tables_normalise_and_fall_back_to_identity():
    # The renderer trusts these weights to sum to one; anything else scales the
    # blended matrix and makes the splat drift toward the origin.
    groups = [
        [(0, 3.0), (1, 1.0)],  # unnormalised, two bones
        [],  # no groups at all
        [(9, 1.0)],  # a group with no matching bone
    ]
    group_index_to_bone = {0: "hip", 1: "spine", 9: "ghost"}
    bone_index_by_name = {"hip": 1, "spine": 2}

    indices, weights = splat_utils._vertex_influence_tables(
        groups, group_index_to_bone, bone_index_by_name
    )

    assert weights[0].sum() == 1.0
    assert indices[0][0] == 1 and weights[0][0] == 0.75
    assert indices[0][1] == 2 and weights[0][1] == 0.25

    for row in (1, 2):
        assert indices[row][0] == 0
        assert weights[row][0] == 1.0


def test_barycentric_coordinates_recover_a_known_point():
    corners = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    point = np.array([[0.25, 0.5, 0.0]])
    bary = splat_utils._barycentric_coordinates(corners, point)
    assert np.allclose(bary, [[0.25, 0.25, 0.5]])


def test_barycentric_coordinates_survive_a_degenerate_triangle():
    # A collapsed face has no basis to solve against; it has to resolve to a
    # corner rather than divide by zero and poison the weight row.
    corners = np.zeros((1, 3, 3))
    bary = splat_utils._barycentric_coordinates(corners, np.zeros((1, 3)))
    assert np.allclose(bary, [[1.0, 0.0, 0.0]])
    assert np.isfinite(bary).all()


def test_influence_rows_ramp_across_a_face_instead_of_stepping():
    # The regression this whole path exists for: sampling the nearest *vertex*
    # makes the field jump from one bone to the other halfway along an edge, and
    # a splat cloud dense enough to straddle that jump tears there. Interpolated
    # corners have to hand back a ramp instead.
    vertex_bones = np.array([[1, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]], dtype=np.int32)
    vertex_weights = np.array(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    corners = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangle = np.array([[0, 1, 2]], dtype=np.int32)

    samples = 21
    points = np.stack(
        [np.linspace(0.0, 1.0, samples), np.zeros(samples), np.zeros(samples)], axis=1
    )
    corner_vertices = np.repeat(triangle, samples, axis=0)
    bary = splat_utils._barycentric_coordinates(corners[corner_vertices], points)
    indices, weights = splat_utils._blend_influence_rows(
        corner_vertices, bary, vertex_bones, vertex_weights, bone_count=3
    )

    # Weight of bone 1 along the edge, however the two slots happen to be ordered.
    bone_one = np.where(indices == 1, weights, 0.0).sum(axis=1)
    assert bone_one[0] == pytest.approx(1.0)
    assert bone_one[-1] == pytest.approx(0.0)
    assert np.all(np.diff(bone_one) <= 1e-9)
    # A step would show up as a single jump of 1.0 between two samples.
    assert np.abs(np.diff(bone_one)).max() < 0.1
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_influence_rows_merge_a_bone_two_corners_share():
    vertex_bones = np.array([[1, 2, 0, 0], [1, 3, 0, 0], [1, 0, 0, 0]], dtype=np.int32)
    vertex_weights = np.array(
        [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    corner_vertices = np.array([[0, 1, 2]], dtype=np.int32)
    bary = np.array([[1 / 3, 1 / 3, 1 / 3]])
    indices, weights = splat_utils._blend_influence_rows(
        corner_vertices, bary, vertex_bones, vertex_weights, bone_count=4
    )

    merged = {int(bone): float(w) for bone, w in zip(indices[0], weights[0]) if w > 0}
    assert merged[1] == pytest.approx(2 / 3)
    assert merged[2] == pytest.approx(1 / 6)
    assert merged[3] == pytest.approx(1 / 6)


def test_influence_rows_keep_only_the_strongest_four():
    # Three corners with four influences each offer twelve candidates; the table
    # the renderer reads has room for four, and they must be the dominant four.
    vertex_bones = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [1, 9, 10, 11]], dtype=np.int32
    )
    vertex_weights = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.7, 0.1, 0.1, 0.1],
            [0.7, 0.1, 0.1, 0.1],
        ]
    )
    corner_vertices = np.array([[0, 1, 2]], dtype=np.int32)
    bary = np.array([[1 / 3, 1 / 3, 1 / 3]])
    indices, weights = splat_utils._blend_influence_rows(
        corner_vertices, bary, vertex_bones, vertex_weights, bone_count=12
    )

    assert weights[0].sum() == pytest.approx(1.0)
    assert np.all(np.diff(weights[0]) <= 1e-12)  # sorted, strongest first
    # Bone 1 is dominant in two corners, so it has to survive the cut and lead.
    assert indices[0][0] == 1
    # Kept: bone 1 (two corners at 0.7/3), bone 5 (0.7/3), then two of the 0.1/3
    # tails; dropping the rest renormalises what is left, as the vertex rows do.
    kept = 2 * 0.7 / 3 + 0.7 / 3 + 2 * 0.1 / 3
    assert weights[0][0] == pytest.approx((2 * 0.7 / 3) / kept)


def test_influence_rows_fall_back_to_identity_when_no_corner_is_weighted():
    # `_vertex_influence_tables` parks unweighted vertices on slot 0 with weight
    # one. Interpolating three of those must not produce an empty row, which the
    # renderer would read as a zero matrix and collapse onto the origin.
    vertex_bones = np.zeros((3, 4), dtype=np.int32)
    vertex_weights = np.zeros((3, 4))
    corner_vertices = np.array([[0, 1, 2]], dtype=np.int32)
    bary = np.array([[1 / 3, 1 / 3, 1 / 3]])
    indices, weights = splat_utils._blend_influence_rows(
        corner_vertices, bary, vertex_bones, vertex_weights, bone_count=4
    )
    assert indices[0][0] == 0
    assert weights[0][0] == pytest.approx(1.0)


def test_surface_influence_tables_fall_back_without_triangles():
    # A cloud carried beside a mesh with no faces still has to get a table; the
    # nearest-vertex path is the only thing left to sample.
    rest_world = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vertex_bones = np.array([[1, 0, 0, 0], [2, 0, 0, 0]], dtype=np.int32)
    vertex_weights = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    query = np.array([[0.9, 0.0, 0.0]])

    calls = {}

    def fake_nearest(rest, points):
        calls["points"] = points
        return np.array([1], dtype=np.int32)

    original = splat_utils._nearest_rest_vertices
    splat_utils._nearest_rest_vertices = fake_nearest
    try:
        indices, weights = splat_utils._surface_influence_tables(
            rest_world,
            np.zeros((0, 3), dtype=np.int32),
            vertex_bones,
            vertex_weights,
            query,
            bone_count=3,
        )
    finally:
        splat_utils._nearest_rest_vertices = original

    assert calls["points"] is query
    assert indices[0][0] == 2
    assert weights[0][0] == pytest.approx(1.0)


def test_surface_influence_tables_interpolate_and_patch_misses(monkeypatch):
    # The whole path with the Blender lookup stubbed: two splats land on the
    # face and get interpolated rows, one is unresolvable and has to fall back
    # to the nearest vertex rather than come back empty.
    rest_world = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    vertex_bones = np.array([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]], dtype=np.int32)
    vertex_weights = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
    query = np.array([[0.5, 0.0, 0.0], [0.1, 0.1, 0.0], [9.0, 9.0, 9.0]])

    def fake_surface(rest, tris, points):
        return (
            np.array([0, 0, -1], dtype=np.int32),
            np.array([[0.5, 0.0, 0.0], [0.1, 0.1, 0.0], [0.0, 0.0, 0.0]]),
        )

    monkeypatch.setattr(splat_utils, "_nearest_rest_surface", fake_surface)
    monkeypatch.setattr(
        splat_utils, "_nearest_rest_vertices", lambda rest, points: np.array([2], dtype=np.int32)
    )

    indices, weights = splat_utils._surface_influence_tables(
        rest_world, triangles, vertex_bones, vertex_weights, query, bone_count=4
    )

    # Midpoint of the 0-1 edge: half bone 1, half bone 2, nothing from bone 3.
    midpoint = {int(b): float(w) for b, w in zip(indices[0], weights[0]) if w > 0}
    assert midpoint == pytest.approx({1: 0.5, 2: 0.5})
    # Interior point: all three corners contribute.
    interior = {int(b): float(w) for b, w in zip(indices[1], weights[1]) if w > 0}
    assert set(interior) == {1, 2, 3}
    assert interior[1] == pytest.approx(0.8)
    # The unresolved splat took the nearest vertex (index 2 -> bone 3).
    assert indices[2][0] == 3
    assert weights[2][0] == pytest.approx(1.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
