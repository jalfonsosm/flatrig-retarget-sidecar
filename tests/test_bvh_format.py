import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from blender_io.bvh_format import (  # noqa: E402
    _build_joint_children,
    _rest_3d_bvh_frames,
    _write_3d_bvh,
)


def _two_joint_layout():
    joints = [
        {
            "index": 0,
            "bvh_name": "root__000",
            "parent_index": -1,
            "offset": [0.0, 0.0, 0.0],
            "tail_offset": [0.0, 1.0, 0.0],
        },
        {
            "index": 1,
            "bvh_name": "spine__001",
            "parent_index": 0,
            "offset": [0.0, 1.0, 0.0],
            "tail_offset": [0.0, 2.0, 0.0],
        },
    ]
    return {"joints": joints}


def test_build_joint_children_preserves_parent_order():
    assert _build_joint_children(_two_joint_layout()["joints"]) == {0: [1], 1: []}


def test_rest_3d_bvh_frames_returns_at_least_two_zero_frames():
    positions, rotations = _rest_3d_bvh_frames(_two_joint_layout(), frame_count=1)

    assert positions == [[0.0] * 6, [0.0] * 6]
    assert rotations == [[0.0] * 6, [0.0] * 6]
    assert positions[0] is not positions[1]


def test_write_3d_bvh_emits_hierarchy_and_motion(tmp_path):
    output = tmp_path / "clip.bvh"
    positions = [[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]]
    rotations = [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]

    _write_3d_bvh(output, _two_joint_layout()["joints"], positions, rotations, fps=30.0)

    text = output.read_text(encoding="utf-8")
    assert "ROOT root__000" in text
    assert "\tJOINT spine__001" in text
    assert "End Site" in text
    assert "Frames: 1" in text
    assert "Frame Time: 0.03333333" in text
    assert "0.000000 0.000000 0.000000 10.000000 20.000000 30.000000" in text
    assert "1.000000 2.000000 3.000000 40.000000 50.000000 60.000000" in text


def test_write_3d_bvh_rejects_empty_joint_list(tmp_path):
    with pytest.raises(ValueError, match="without joints"):
        _write_3d_bvh(tmp_path / "empty.bvh", [], [], [], fps=30.0)
