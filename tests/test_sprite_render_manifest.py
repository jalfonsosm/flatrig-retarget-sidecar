import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import blender_scene_io as scene_io  # noqa: E402


def _reference_entry():
    return {
        "kind": "reference",
        "name": "__flatrig_sprite_reference__",
        "output_path": "/tmp/sprite_reference.png",
        "projection_frame": {"center_x": 1.0, "center_y": 2.0, "span": 4.0},
    }


def test_reference_entry_is_not_counted_as_a_part_render():
    legacy_part = {"name": "body", "attachment_name": "body"}
    accessory_part = {
        "kind": "part",
        "name": "sword",
        "attachment_name": "sword",
        "object_name": "Sword",
    }

    parts, reference = scene_io._split_sprite_render_manifest(
        [legacy_part, _reference_entry(), accessory_part]
    )

    assert parts == [legacy_part, accessory_part]
    assert reference == _reference_entry()
    assert parts[1]["object_name"] == "Sword"


def test_reference_entry_is_optional_for_legacy_manifests():
    parts, reference = scene_io._split_sprite_render_manifest([{"name": "body"}])

    assert parts == [{"name": "body"}]
    assert reference is None


def test_reference_contract_rejects_ambiguous_or_incomplete_entries():
    with pytest.raises(ValueError, match="only one reference"):
        scene_io._split_sprite_render_manifest([_reference_entry(), _reference_entry()])

    incomplete = _reference_entry()
    incomplete["projection_frame"] = {"center_x": 0.0}
    with pytest.raises(ValueError, match="projection_frame is missing"):
        scene_io._split_sprite_render_manifest([incomplete])
