from __future__ import annotations

from flatrig_retarget_sidecar.spine_import import build_spine_package
from flatrig_retarget_sidecar.spine_motion2motion_bridge import (
    build_exported_motion2motion_mapping,
    build_sample_times,
)


def test_exported_mapping_prefers_semantic_biped_pairs() -> None:
    source = build_spine_package(
        {
            "bones": [
                {"name": "root"},
                {"name": "hip", "parent": "root"},
                {"name": "torso", "parent": "hip"},
                {"name": "neck", "parent": "torso"},
                {"name": "head", "parent": "neck"},
                {"name": "front-upper-arm", "parent": "torso"},
                {"name": "front-bracer", "parent": "front-upper-arm"},
                {"name": "front-fist", "parent": "front-bracer"},
                {"name": "rear-upper-arm", "parent": "torso"},
                {"name": "rear-bracer", "parent": "rear-upper-arm"},
                {"name": "front-thigh", "parent": "hip"},
                {"name": "front-shin", "parent": "front-thigh"},
                {"name": "front-foot", "parent": "front-shin"},
                {"name": "rear-thigh", "parent": "hip"},
                {"name": "rear-shin", "parent": "rear-thigh"},
                {"name": "rear-foot", "parent": "rear-shin"},
            ],
            "animations": {"walk": {"bones": {}}},
        },
        source_label="spineboy.json",
    )
    target = build_spine_package(
        {
            "bones": [
                {"name": "root"},
                {"name": "mixamorig:Hips", "parent": "root"},
                {"name": "mixamorig:Spine", "parent": "mixamorig:Hips"},
                {"name": "mixamorig:Spine2", "parent": "mixamorig:Spine"},
                {"name": "mixamorig:Neck", "parent": "mixamorig:Spine2"},
                {"name": "mixamorig:Head", "parent": "mixamorig:Neck"},
                {"name": "mixamorig:HeadTop_End", "parent": "mixamorig:Head"},
                {"name": "mixamorig:RightArm", "parent": "mixamorig:Spine2"},
                {"name": "mixamorig:RightForeArm", "parent": "mixamorig:RightArm"},
                {"name": "mixamorig:RightHand", "parent": "mixamorig:RightForeArm"},
                {"name": "mixamorig:LeftArm", "parent": "mixamorig:Spine2"},
                {"name": "mixamorig:LeftForeArm", "parent": "mixamorig:LeftArm"},
                {"name": "mixamorig:LeftHand", "parent": "mixamorig:LeftForeArm"},
                {"name": "mixamorig:RightUpLeg", "parent": "mixamorig:Hips"},
                {"name": "mixamorig:RightLeg", "parent": "mixamorig:RightUpLeg"},
                {"name": "mixamorig:RightFoot", "parent": "mixamorig:RightLeg"},
                {"name": "mixamorig:RightToe_End", "parent": "mixamorig:RightFoot"},
                {"name": "mixamorig:LeftUpLeg", "parent": "mixamorig:Hips"},
                {"name": "mixamorig:LeftLeg", "parent": "mixamorig:LeftUpLeg"},
                {"name": "mixamorig:LeftFoot", "parent": "mixamorig:LeftLeg"},
                {"name": "mixamorig:LeftToe_End", "parent": "mixamorig:LeftFoot"},
            ],
            "animations": {"walk": {"bones": {}}},
        },
        source_label="target.json",
    )

    payload = build_exported_motion2motion_mapping(source, target)
    mapping = {(pair["source"], pair["target"]) for pair in payload["mapping"]}

    assert ("head", "mixamorig_Head") in mapping
    assert ("front_foot", "mixamorig_RightFoot") in mapping
    assert ("rear_foot", "mixamorig_LeftFoot") in mapping
    assert ("front_fist", "mixamorig_RightHand") in mapping
    assert ("front_upper_arm", "mixamorig_RightArm") in mapping
    assert ("rear_bracer", "mixamorig_LeftForeArm") in mapping
    assert ("front_foot", "mixamorig_RightToe_End") not in mapping
    assert ("head", "mixamorig_HeadTop_End") not in mapping


def test_static_spine_animation_exports_enough_sample_frames_for_m2m() -> None:
    samples = build_sample_times(0.0, 30.0)
    assert len(samples) == 31
    assert samples[0] == 0.0
    assert samples[-1] == 1.0
