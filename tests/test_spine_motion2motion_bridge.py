from __future__ import annotations

from flatrig_retarget_sidecar.motion2motion_retarget import (
    _direct_spine_mapping_transfer,
    retarget_spine_animation,
)
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


def test_direct_spine_transfer_uses_world_delta_not_raw_local_copy() -> None:
    source = build_spine_package(
        {
            "bones": [
                {"name": "root"},
                {"name": "torso", "parent": "root"},
                {"name": "front-upper-arm", "parent": "torso"},
            ],
            "animations": {
                "walk": {
                    "bones": {
                        "torso": {
                            "rotate": [
                                {"time": 0.0, "angle": 0.0},
                                {"time": 1.0, "angle": 30.0},
                            ]
                        },
                        "front-upper-arm": {
                            "rotate": [
                                {"time": 0.0, "angle": 0.0},
                                {"time": 1.0, "angle": 10.0},
                            ]
                        },
                    }
                }
            },
        },
        source_label="source.json",
    )
    target = build_spine_package(
        {
            "bones": [
                {"name": "root"},
                {"name": "mixamorig:RightShoulder", "parent": "root"},
                {"name": "mixamorig:RightArm", "parent": "mixamorig:RightShoulder"},
            ],
            "animations": {"walk": {"bones": {}}},
        },
        source_label="target.json",
    )
    mapping = {
        "root_joint": "root",
        "mapping": [
            {"source": "front_upper_arm", "target": "mixamorig_RightArm"},
        ],
    }

    clip, diagnostics = _direct_spine_mapping_transfer(
        source,
        target,
        "walk",
        mapping,
        reason="test",
    )

    arm_keys = clip["bones"]["mixamorig:RightArm"]["rotate"]
    assert arm_keys[-1]["angle"] == 40.0
    assert diagnostics["transfer_mode"] == "target_space_world_delta"


def test_spine_retarget_uses_target_rest_pose_without_target_animations() -> None:
    source = build_spine_package(
        {
            "bones": [
                {"name": "root"},
                {"name": "front-upper-arm", "parent": "root"},
            ],
            "animations": {
                "walk": {
                    "bones": {
                        "front-upper-arm": {
                            "rotate": [
                                {"time": 0.0, "angle": 0.0},
                                {"time": 1.0, "angle": 20.0},
                            ]
                        },
                    }
                }
            },
        },
        source_label="source.json",
    )
    target = build_spine_package(
        {
            "bones": [
                {"name": "root"},
                {"name": "mixamorig:RightArm", "parent": "root"},
            ],
            "animations": {},
        },
        source_label="target.json",
    )

    result = retarget_spine_animation(source, target, "walk")

    assert result.diagnostics["backend_label"] == "RestPoseSpine2D"
    assert result.diagnostics["target_exemplar_mode"] == "rest_pose"
    assert result.animation["bones"]["mixamorig:RightArm"]["rotate"][-1]["angle"] == 20.0


def test_spine_retarget_accepts_static_pose_animation() -> None:
    source = build_spine_package(
        {
            "bones": [
                {"name": "root"},
                {"name": "front-upper-arm", "parent": "root"},
            ],
            "animations": {
                "aim": {
                    "bones": {
                        "front-upper-arm": {
                            "rotate": [
                                {"time": 0.0, "angle": 35.0},
                                {"time": 1.0, "angle": 35.0},
                            ]
                        },
                    }
                }
            },
        },
        source_label="source.json",
    )
    target = build_spine_package(
        {
            "bones": [
                {"name": "root"},
                {"name": "mixamorig:RightArm", "parent": "root"},
            ],
            "animations": {},
        },
        source_label="target.json",
    )

    result = retarget_spine_animation(source, target, "aim")

    assert result.diagnostics["backend_label"] == "RestPoseSpine2D"
    assert result.diagnostics["target_guide"]["result_has_motion"] is False
    assert result.diagnostics["target_guide"]["result_has_pose_or_motion"] is True
    assert result.animation["bones"]["mixamorig:RightArm"]["rotate"][-1]["angle"] == 35.0
