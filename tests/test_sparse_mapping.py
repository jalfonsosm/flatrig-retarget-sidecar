from __future__ import annotations

from flatrig_retarget_sidecar.sparse_mapping import build_motion2motion_mapping_payload
from flatrig_retarget_sidecar.spine_import import build_spine_package
from tests.helpers import build_spine_payload


def test_sparse_mapping_includes_root_and_side_pairs() -> None:
    source = build_spine_package(build_spine_payload(arm_prefix="arm"), source_label="source.json")
    target = build_spine_package(build_spine_payload(arm_prefix="limb"), source_label="target.json")

    payload = build_motion2motion_mapping_payload(source, target)
    mapping = {(pair["source"], pair["target"]) for pair in payload["mapping"]}

    assert payload["root_joint"] == "root"
    assert ("root", "root") in mapping
    assert ("arm_l", "limb_l") in mapping
    assert ("arm_r", "limb_r") in mapping
