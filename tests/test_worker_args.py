from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import blender_scene_io as scene_io  # noqa: E402


def test_parse_vec3_arg_accepts_three_numeric_components() -> None:
    assert scene_io._parse_vec3_arg("1, 2.5, -3") == (1.0, 2.5, -3.0)


def test_parse_vec3_arg_rejects_wrong_component_count() -> None:
    try:
        scene_io._parse_vec3_arg("1,2")
    except argparse.ArgumentTypeError as exc:
        assert "three comma-separated" in str(exc)
    else:
        raise AssertionError("expected ArgumentTypeError")


def test_parse_vec3_arg_rejects_non_numeric_components() -> None:
    try:
        scene_io._parse_vec3_arg("1,nope,3")
    except argparse.ArgumentTypeError as exc:
        assert "numeric" in str(exc)
    else:
        raise AssertionError("expected ArgumentTypeError")
