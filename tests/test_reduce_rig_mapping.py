"""Reading a joint mapping for the canonical reduction.

Biped humanoids reduce by bone name. Every other body plan cannot: an adaptive
rigger names its joints bone_0..N (UniRig's `Order.make_names` emits exactly
that as padding when no skeleton template applies), so the alias table resolves
nothing and the rig used to be exported untouched. The mapping the retarget and
the joint editor already exchange supplies the missing names instead.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def _read(tmp_path, payload):
    # Imported lazily: blender_scene_io needs bpy at import time in some
    # environments, and only this one pure-Python helper is under test.
    from blender_scene_io import _canonical_by_bone_from_mapping_file

    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return _canonical_by_bone_from_mapping_file(path)


pytest.importorskip("bpy", reason="the helper lives beside bpy-only module code")


def test_the_mapping_is_read_from_the_retarget_direction(tmp_path):
    """`source` is the carrier that drives, `target` is the rig that follows,
    so the reduction reads the pairs the other way round."""
    result = _read(
        tmp_path,
        {"mapping": [
            {"source": "Hips", "target": "bone_13"},
            {"source": "Front_Leg_Upper_L", "target": "bone_7"},
        ]},
    )
    assert result == {"bone_13": "Hips", "bone_7": "Front_Leg_Upper_L"}


def test_the_alternate_key_names_are_accepted(tmp_path):
    result = _read(
        tmp_path,
        {"pairs": [{"source_bone": "Head", "target_bone": "bone_2"}]},
    )
    assert result == {"bone_2": "Head"}


def test_the_first_pair_for_a_bone_wins(tmp_path):
    """Matches how the retarget reads the same file, so a mapping cannot mean
    one thing to the reduction and another to the animation."""
    result = _read(
        tmp_path,
        {"mapping": [
            {"source": "Spine_1", "target": "bone_5"},
            {"source": "Spine_2", "target": "bone_5"},
        ]},
    )
    assert result == {"bone_5": "Spine_1"}


@pytest.mark.parametrize(
    "payload",
    [
        {"mapping": []},
        {"mapping": [{"source": "", "target": "bone_1"}]},
        {"mapping": [{"source": "Hips", "target": ""}]},
        {"mapping": ["not a pair"]},
        {},
    ],
)
def test_unusable_entries_are_skipped_rather_than_raising(tmp_path, payload):
    assert _read(tmp_path, payload) == {}
