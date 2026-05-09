"""Rig family detection used by the retarget pipeline.

The retarget backend can short-circuit Motion2Motion when both source and target
share a known canonical naming convention (Mixamo today). For unknown rigs we
fall back to the existing M2M + sparse mapping flow.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal

RigFamily = Literal["mixamo", "smpl", "vroid", "generic"]


_ALPHANUM = re.compile(r"[^a-z0-9]+")


def _normalize_token(name: str) -> str:
    return _ALPHANUM.sub("", str(name).split(":")[-1].lower())


def _joint_names_from_payload(payload: Any) -> list[str]:
    """Extract bone/joint name strings from any of the shapes flatrig uses.

    Accepts: list[str], list[dict], dict (skeleton metadata), or any object with
    `.names` / `.bones` / `.joints` attributes. Anything unrecognized yields an
    empty list — callers must treat detection as best-effort.
    """
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        for key in ("joints", "bones", "names"):
            if key in payload:
                return _joint_names_from_payload(payload[key])
        return []
    if isinstance(payload, Iterable):
        names: list[str] = []
        for item in payload:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                for key in ("name", "bvh_name", "matching_name", "original_name"):
                    value = item.get(key)
                    if isinstance(value, str) and value:
                        names.append(value)
                        break
        return names
    for attr in ("names", "bones", "joints"):
        value = getattr(payload, attr, None)
        if value is not None:
            return _joint_names_from_payload(value)
    return []


# Canonical Mixamo bone tokens (after stripping "mixamorig:" and non-alphanum).
# A rig that hits a strong subset of these is considered Mixamo even when the
# prefix has been stripped by the importer.
_MIXAMO_CORE_TOKENS = frozenset({
    "hips",
    "spine",
    "spine1",
    "spine2",
    "neck",
    "head",
    "leftshoulder",
    "rightshoulder",
    "leftarm",
    "rightarm",
    "leftforearm",
    "rightforearm",
    "lefthand",
    "righthand",
    "leftupleg",
    "rightupleg",
    "leftleg",
    "rightleg",
    "leftfoot",
    "rightfoot",
})

# Tokens unique to SMPL (SMPL/SMPL-X) skeletons.
_SMPL_CORE_TOKENS = frozenset({
    "pelvis",
    "leftupperleg",
    "rightupperleg",
    "leftlowerleg",
    "rightlowerleg",
    "leftupperarm",
    "rightupperarm",
    "leftlowerarm",
    "rightlowerarm",
})

# Tokens characteristic of VRoid / VRM rigs (Unity-style humanoid avatars often
# carry "j_bip" prefixes; here we match the bare canonical names too).
_VROID_CORE_TOKENS = frozenset({
    "jbipchest",
    "jbipupperchest",
    "jbiplupperarm",
    "jbiplhand",
    "secondarymasterjoint",
})


def detect_rig_family(payload: Any) -> RigFamily:
    """Best-effort classification of a rig into a known family.

    Returns "generic" when no signature matches strongly enough. The detector is
    intentionally conservative: a wrong "mixamo" label routes the caller to the
    direct-mapping fast path, which only produces good motion when both rigs
    share the Mixamo joint naming and orientation. False positives are worse
    than false negatives here.
    """
    names = _joint_names_from_payload(payload)
    if not names:
        return "generic"

    has_mixamo_prefix = any(str(n).lower().startswith("mixamorig:") for n in names)
    tokens = {_normalize_token(n) for n in names if n}

    if has_mixamo_prefix:
        return "mixamo"

    mixamo_hits = len(tokens & _MIXAMO_CORE_TOKENS)
    if mixamo_hits >= 8:
        # Strong match against the Mixamo bone vocabulary even without the prefix.
        return "mixamo"

    smpl_hits = len(tokens & _SMPL_CORE_TOKENS)
    if smpl_hits >= 6:
        return "smpl"

    vroid_hits = len(tokens & _VROID_CORE_TOKENS)
    if vroid_hits >= 3:
        return "vroid"

    return "generic"


def can_use_mixamo_direct_bypass(source_payload: Any, target_payload: Any) -> bool:
    """True when both rigs are Mixamo and direct-mapped retarget is safe.

    The fast path copies local rotations bone-for-bone via
    `direct_mapped_bvh_retarget`; that's only correct when the joint axes line
    up, which is the Mixamo guarantee.
    """
    return (
        detect_rig_family(source_payload) == "mixamo"
        and detect_rig_family(target_payload) == "mixamo"
    )
