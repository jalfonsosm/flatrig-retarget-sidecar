from __future__ import annotations

import json
from pathlib import Path


def build_spine_payload(
    *, arm_prefix: str = "arm", spine_version: str = "4.1"
) -> dict[str, object]:
    return {
        "skeleton": {"spine": spine_version},
        "bones": [
            {"name": "root"},
            {"name": "torso", "parent": "root", "x": 0, "y": 8, "length": 12},
            {"name": f"{arm_prefix}_l", "parent": "torso", "x": -6, "y": 6, "length": 9},
            {"name": f"{arm_prefix}_r", "parent": "torso", "x": 6, "y": 6, "length": 9},
        ],
        "slots": [
            {"name": "torso-slot", "bone": "torso"},
            {"name": "left-slot", "bone": f"{arm_prefix}_l"},
            {"name": "right-slot", "bone": f"{arm_prefix}_r"},
        ],
        "skins": [{"name": "default", "attachments": {}}],
        "animations": {
            "idle": {
                "bones": {
                    "torso": {
                        "rotate": [
                            {"time": 0.0, "angle": 0.0},
                            {"time": 0.5, "angle": 6.0},
                        ]
                    }
                }
            }
        },
    }


def write_spine_json(path: Path, *, arm_prefix: str = "arm") -> Path:
    payload = build_spine_payload(arm_prefix=arm_prefix)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
