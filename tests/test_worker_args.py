from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import blender_worker_args as worker_args  # noqa: E402


def test_parse_vec3_arg_accepts_three_numeric_components() -> None:
    assert worker_args._parse_vec3_arg("1, 2.5, -3") == (1.0, 2.5, -3.0)


def test_parse_vec3_arg_rejects_wrong_component_count() -> None:
    try:
        worker_args._parse_vec3_arg("1,2")
    except argparse.ArgumentTypeError as exc:
        assert "three comma-separated" in str(exc)
    else:
        raise AssertionError("expected ArgumentTypeError")


def test_parse_vec3_arg_rejects_non_numeric_components() -> None:
    try:
        worker_args._parse_vec3_arg("1,nope,3")
    except argparse.ArgumentTypeError as exc:
        assert "numeric" in str(exc)
    else:
        raise AssertionError("expected ArgumentTypeError")


def test_parse_worker_args_reads_arguments_after_blender_separator() -> None:
    args = worker_args.parse_worker_args(
        ["front", "side"],
        [
            "blender",
            "--background",
            "--",
            "extract-scene",
            "source.fbx",
            "--output",
            "scene.json",
            "--view-preset",
            "side",
            "--view-dir",
            "1,0,0",
        ],
    )

    assert args.command == "extract-scene"
    assert args.source == "source.fbx"
    assert args.output == "scene.json"
    assert args.view_preset == "side"
    assert args.view_dir == (1.0, 0.0, 0.0)


def test_parse_worker_args_reads_negative_view_dir_with_equals() -> None:
    args = worker_args.parse_worker_args(
        ["front", "side", "three_quarter"],
        [
            "python",
            "blender_scene_io.py",
            "--",
            "extract-scene",
            "source.fbx",
            "--output",
            "scene.json",
            "--view-preset",
            "three_quarter",
            "--view-dir=-0.707107,-0.698325,0.111097",
            "--view-up=0,0.157115,0.98758",
        ],
    )

    assert args.command == "extract-scene"
    assert args.view_preset == "three_quarter"
    assert args.view_dir == (-0.707107, -0.698325, 0.111097)
    assert args.view_up == (0.0, 0.157115, 0.98758)


def test_parse_worker_args_reads_bake_predicted_rig_mesh_path() -> None:
    args = worker_args.parse_worker_args(
        ["front", "side"],
        [
            "blender",
            "--background",
            "--",
            "bake-predicted-rig",
            "prediction.npz",
            "--output",
            "report.json",
            "--fbx-output",
            "rigged.fbx",
            "--mesh-path",
            "source mesh.glb",
        ],
    )

    assert args.command == "bake-predicted-rig"
    assert args.fbx_output == "rigged.fbx"
    assert args.mesh_path == "source mesh.glb"
