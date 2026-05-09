from __future__ import annotations

from flatrig_retarget_sidecar.rig_identity import (
    can_use_mixamo_direct_bypass,
    detect_rig_family,
)


def _mixamo_with_prefix() -> list[str]:
    return [
        "mixamorig:Hips",
        "mixamorig:Spine",
        "mixamorig:Spine1",
        "mixamorig:Spine2",
        "mixamorig:Neck",
        "mixamorig:Head",
        "mixamorig:LeftShoulder",
        "mixamorig:LeftArm",
        "mixamorig:LeftForeArm",
        "mixamorig:LeftHand",
        "mixamorig:RightShoulder",
        "mixamorig:RightArm",
        "mixamorig:RightForeArm",
        "mixamorig:RightHand",
        "mixamorig:LeftUpLeg",
        "mixamorig:LeftLeg",
        "mixamorig:LeftFoot",
        "mixamorig:RightUpLeg",
        "mixamorig:RightLeg",
        "mixamorig:RightFoot",
    ]


def _mixamo_without_prefix() -> list[str]:
    return [name.split(":", 1)[1] for name in _mixamo_with_prefix()]


def _smpl_skeleton() -> list[str]:
    return [
        "pelvis",
        "left_upper_leg",
        "right_upper_leg",
        "left_lower_leg",
        "right_lower_leg",
        "left_upper_arm",
        "right_upper_arm",
        "left_lower_arm",
        "right_lower_arm",
        "spine1",
        "head",
    ]


def _generic_skeleton() -> list[str]:
    return [
        "Bone",
        "Bone.001",
        "Bone.002",
        "Bone.003",
        "tail_a",
        "tail_b",
    ]


def test_detects_mixamo_with_prefix() -> None:
    assert detect_rig_family(_mixamo_with_prefix()) == "mixamo"


def test_detects_mixamo_without_prefix_via_token_match() -> None:
    # No "mixamorig:" prefix but the canonical bone vocabulary is preserved.
    assert detect_rig_family(_mixamo_without_prefix()) == "mixamo"


def test_detects_smpl() -> None:
    assert detect_rig_family(_smpl_skeleton()) == "smpl"


def test_detects_generic_for_unknown_rig() -> None:
    assert detect_rig_family(_generic_skeleton()) == "generic"


def test_handles_dict_payload_with_joints_key() -> None:
    payload = {
        "joints": [
            {"name": name, "index": idx} for idx, name in enumerate(_mixamo_with_prefix())
        ]
    }
    assert detect_rig_family(payload) == "mixamo"


def test_handles_dict_payload_with_bvh_name_key() -> None:
    payload = {
        "joints": [
            {"bvh_name": name, "index": idx}
            for idx, name in enumerate(_mixamo_without_prefix())
        ]
    }
    assert detect_rig_family(payload) == "mixamo"


def test_can_use_mixamo_direct_bypass_requires_both_sides() -> None:
    assert can_use_mixamo_direct_bypass(_mixamo_with_prefix(), _mixamo_with_prefix())
    assert not can_use_mixamo_direct_bypass(_mixamo_with_prefix(), _generic_skeleton())
    assert not can_use_mixamo_direct_bypass(_generic_skeleton(), _mixamo_with_prefix())
    assert not can_use_mixamo_direct_bypass(_generic_skeleton(), _generic_skeleton())


def test_empty_payload_returns_generic() -> None:
    assert detect_rig_family([]) == "generic"
    assert detect_rig_family(None) == "generic"
    assert detect_rig_family({}) == "generic"


def test_partial_mixamo_overlap_does_not_misclassify() -> None:
    # Five tokens overlap with Mixamo (under the 8-token threshold) → must not
    # be promoted to Mixamo. This is the conservative-detector contract.
    partial = ["Hips", "Spine", "Neck", "Head", "LeftHand", "Tail", "Wing", "BeakA", "BeakB"]
    assert detect_rig_family(partial) == "generic"
