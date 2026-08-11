"""Contract tests for the per-splat skinning table.

The table is the hand-off that lets FlatRig pose a Gaussian cloud without
Blender: it is written here and read by the native sprite renderer, so the byte
layout is a contract between two repositories. These tests pin it.
"""

import struct
import sys
from pathlib import Path

import numpy as np

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
