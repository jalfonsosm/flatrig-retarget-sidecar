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


def _filtered_reference_entry():
    entry = _reference_entry()
    entry["triangle_filter"] = "exported_core_union"
    entry["triangle_groups"] = [
        {"object_name": "Body", "triangle_keys": [[2, 0, 1], [4, 5, 6]]},
        {"object_name": "Body", "triangle_keys": [[1, 2, 0]]},
    ]
    return entry


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


def test_filtered_reference_resolves_only_exported_objects_and_deduplicated_keys():
    body = object()
    decoy = object()

    groups = scene_io._resolve_reference_triangle_groups(
        _filtered_reference_entry(),
        {"Body": body, "DecoyCape": decoy},
        body,
    )

    assert groups == [
        {
            "object_name": "Body",
            "object": body,
            "triangle_keys": [(0, 1, 2), (4, 5, 6)],
        }
    ]
    assert all(group["object"] is not decoy for group in groups)


def test_legacy_reference_resolves_to_full_scene_render():
    assert scene_io._resolve_reference_triangle_groups(_reference_entry(), {}, object()) is None


def test_reference_filter_ack_is_emitted_only_for_the_resolved_filtered_path():
    filtered = _filtered_reference_entry()

    assert (
        scene_io._applied_reference_triangle_filter(filtered, [{"triangle_keys": [(0, 1, 2)]}])
        == "exported_core_union"
    )
    assert scene_io._applied_reference_triangle_filter(_reference_entry(), None) is None


def test_reference_contract_rejects_ambiguous_or_incomplete_entries():
    with pytest.raises(ValueError, match="only one reference"):
        scene_io._split_sprite_render_manifest([_reference_entry(), _reference_entry()])

    incomplete = _reference_entry()
    incomplete["projection_frame"] = {"center_x": 0.0}
    with pytest.raises(ValueError, match="projection_frame is missing"):
        scene_io._split_sprite_render_manifest([incomplete])

    invalid_filter = _filtered_reference_entry()
    invalid_filter["triangle_filter"] = "all_scene_geometry"
    with pytest.raises(ValueError, match="Unsupported.*triangle_filter"):
        scene_io._resolve_reference_triangle_groups(invalid_filter, {}, object())

    missing_object = _filtered_reference_entry()
    with pytest.raises(ValueError, match="object not found: Body"):
        scene_io._resolve_reference_triangle_groups(missing_object, {}, object())
