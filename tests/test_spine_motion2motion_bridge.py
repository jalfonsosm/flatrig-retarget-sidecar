from __future__ import annotations

from flatrig_retarget_sidecar.motion2motion_retarget import (
    _force_spine_clip_loop_closure,
    _resolve_target_package_with_exemplar,
    _source_animation_loop_closed,
)
from flatrig_retarget_sidecar.spine_import import build_spine_package
from flatrig_retarget_sidecar.spine_motion2motion_bridge import (
    build_exported_motion2motion_mapping,
    build_sample_times,
    export_spine_animation_to_bvh,
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


def test_target_bvh_export_can_be_resampled_to_source_duration(tmp_path) -> None:
    package = build_spine_package(
        {
            "bones": [
                {"name": "root"},
                {"name": "arm", "parent": "root", "length": 10},
            ],
            "animations": {
                "short_walk": {
                    "bones": {
                        "arm": {
                            "rotate": [
                                {"time": 0.0, "angle": 0.0},
                                {"time": 0.5, "angle": 30.0},
                            ]
                        }
                    }
                }
            },
        },
        source_label="target.json",
    )

    metadata = export_spine_animation_to_bvh(
        package,
        "short_walk",
        tmp_path / "target.bvh",
        sample_duration=1.0,
    )

    assert metadata.duration == 1.0
    assert metadata.frame_count == 31


def test_resolve_target_exemplar_prefers_matching_animation_over_arbitrary_idle() -> None:
    target = build_spine_package(
        {
            "bones": [
                {"name": "root"},
                {"name": "mixamorig:RightArm", "parent": "root"},
            ],
            "animations": {
                "idle": {"bones": {"mixamorig:RightArm": {"rotate": [{"time": 0, "angle": 0}]}}},
                "walk": {
                    "bones": {"mixamorig:RightArm": {"rotate": [{"time": 0, "angle": 10}]}}
                },
            },
        },
        source_label="target.json",
    )

    resolved, animation_name, synthesized, mode = _resolve_target_package_with_exemplar(
        target,
        source_animation_name="walk",
        preferred_animation_name=None,
        source_duration=1.0,
    )

    assert resolved is target
    assert animation_name == "walk"
    assert synthesized is False
    assert mode == "matched"


def test_resolve_target_exemplar_synthesizes_static_rest_when_target_has_no_animation() -> None:
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

    resolved, animation_name, synthesized, mode = _resolve_target_package_with_exemplar(
        target,
        source_animation_name="walk",
        preferred_animation_name=None,
        source_duration=1.0,
    )

    assert resolved is not target
    assert animation_name == "__sidecar_rest__"
    assert synthesized is True
    assert mode == "synthetic"
    assert animation_name in resolved.animations
    root_rotate = resolved.animations[animation_name]["bones"]["root"]["rotate"]
    assert len(root_rotate) == 31
    assert root_rotate[0]["time"] == 0.0
    assert root_rotate[-1]["time"] == 1.0


def test_loop_closed_source_closes_retarget_clip() -> None:
    source = build_spine_package(
        {
            "bones": [
                {"name": "root"},
                {"name": "front-upper-arm", "parent": "root", "rotation": 5.0},
            ],
            "animations": {
                "walk": {
                    "bones": {
                        "front-upper-arm": {
                            "rotate": [
                                {"time": 0.0, "angle": 10.0},
                                {"time": 0.5, "angle": -20.0},
                                {"time": 1.0, "angle": 10.0},
                            ]
                        },
                    }
                }
            },
        },
        source_label="source.json",
    )
    clip = {
        "bones": {
            "mixamorig:RightArm": {
                "rotate": [
                    {"time": 0.0, "angle": 25.0, "value": 25.0},
                    {"time": 0.5, "angle": -15.0, "value": -15.0},
                    {"time": 1.0, "angle": 20.0, "value": 20.0},
                ]
            }
        }
    }

    assert _source_animation_loop_closed(source, "walk", 1.0)
    _force_spine_clip_loop_closure(clip, 1.0)

    arm_keys = clip["bones"]["mixamorig:RightArm"]["rotate"]
    # Loop closure no longer overwrites a keyframe that already sits at the
    # end_time — that destroyed real retargeted motion data and produced a
    # visible jump on the very last frames of every loop. The retargeted
    # last frame is preserved verbatim; loop closure only ADDS a closing key
    # when one isn't already present at end_time.
    assert arm_keys[-1]["time"] == 1.0
    assert arm_keys[-1]["angle"] == 20.0  # original retargeted value, not overwritten


def test_loop_closure_appends_when_last_key_is_short() -> None:
    """When the retargeted clip ends BEFORE the source loop boundary, the
    closure adds a fresh key at end_time mirroring the start (regression
    coverage for the fix that stopped overwriting last frames)."""
    clip = {
        "bones": {
            "mixamorig:RightArm": {
                "rotate": [
                    {"time": 0.0, "angle": 25.0, "value": 25.0},
                    {"time": 0.5, "angle": -15.0, "value": -15.0},
                    {"time": 0.9, "angle": 12.0, "value": 12.0},
                ]
            }
        }
    }
    _force_spine_clip_loop_closure(clip, 1.0)
    arm_keys = clip["bones"]["mixamorig:RightArm"]["rotate"]
    assert arm_keys[-1]["time"] == 1.0
    assert arm_keys[-1]["angle"] == arm_keys[0]["angle"]
    assert len(arm_keys) == 4
    assert arm_keys[-2]["time"] == 0.9
    assert arm_keys[-2]["angle"] == 12.0  # the real last retargeted frame still sits at 0.9
