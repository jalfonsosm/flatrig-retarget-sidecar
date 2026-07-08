"""Argument parser for the public Blender worker script."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence

WORKER_COMMANDS = (
    "inspect",
    "convert",
    "extract-scene",
    "extract-animations",
    "render-sprites",
    "export-3d-animation-bvh",
    "export-3d-rest-bvh",
    "extract-mesh-targets",
    "dump-rig-animation",
    "bake-rig-animation",
    "reduce-rig-to-canonical",
    "cleanup-mesh",
    "bake-predicted-rig",
)


def _parse_vec3_arg(raw: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values: x,y,z")
    try:
        x, y, z = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric comma-separated values: x,y,z") from exc
    return (x, y, z)


def _worker_script_args(argv: Sequence[str] | None = None) -> list[str]:
    source_argv = list(sys.argv if argv is None else argv)
    try:
        separator_index = source_argv.index("--")
    except ValueError:
        return []
    return source_argv[separator_index + 1 :]


def parse_worker_args(
    view_preset_names: Iterable[str],
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public 3D scene worker.")
    parser.add_argument("command", choices=WORKER_COMMANDS)
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--view-preset",
        default="front",
        choices=list(view_preset_names),
        help="View preset for projection (front, back, side, side_r, top, bottom)",
    )
    parser.add_argument(
        "--view-dir",
        type=_parse_vec3_arg,
        default=None,
        help="Custom view direction as 'x,y,z' tuple",
    )
    parser.add_argument(
        "--view-up",
        type=_parse_vec3_arg,
        default=None,
        help="Custom view up hint as 'x,y,z' tuple",
    )
    parser.add_argument("--view-roll", type=float, default=0.0, help="View roll in degrees")
    parser.add_argument(
        "--source-frame", type=int, default=None, help="Source frame for pose evaluation"
    )
    parser.add_argument(
        "--use-rest-pose",
        action="store_true",
        default=False,
        help="Evaluate setup mesh and bones in armature rest pose",
    )
    parser.add_argument(
        "--projection-space",
        default="world",
        choices=("world", "root"),
        help="Projection space used by the Python pipeline",
    )
    parser.add_argument(
        "--animation",
        dest="animation_names",
        action="append",
        default=[],
        help="Animation name (can be specified multiple times)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Target animation FPS")
    parser.add_argument("--frame-start", type=int, default=None, help="First frame to sample")
    parser.add_argument("--frame-end", type=int, default=None, help="Last frame to sample")
    parser.add_argument(
        "--frame-count", type=int, default=None, help="Frame count for rest BVH export"
    )
    parser.add_argument("--sample-substeps", type=int, default=2, help="Subsamples per frame")
    parser.add_argument(
        "--no-optimize-animation-keys",
        dest="optimize_animation_keys",
        action="store_false",
        default=True,
    )
    parser.add_argument("--force-loop-closing-keys", action="store_true", default=False)
    parser.add_argument(
        "--pose-mode",
        default="full",
        choices=("full", "rotation_only", "local_rotation", "blend"),
        help="Pose extraction mode",
    )
    parser.add_argument(
        "--pose-blend", type=float, default=1.0, help="Blend amount for pose-mode=blend"
    )
    parser.add_argument(
        "--rotation-flatten", type=float, default=0.0, help="Rotation flatten amount"
    )
    parser.add_argument("--rotation-flatten-scope", default="all", help="Rotation flatten scope")
    parser.add_argument("--rotation-flatten-bones", default="", help="Rotation flatten custom bones")
    parser.add_argument(
        "--connected-translation-scope",
        default="none",
        choices=("none", "terminal", "limbs", "all", "custom"),
        help="Connected-bone translation emission scope",
    )
    parser.add_argument(
        "--connected-translation-bones",
        default="",
        help="Custom bones for connected translation emission",
    )
    parser.add_argument("--stretch-guard-enabled", action="store_true", default=False)
    parser.add_argument("--stretch-guard-max-scale", type=float, default=1.75)
    parser.add_argument("--stretch-guard-strength", type=float, default=0.65)
    parser.add_argument(
        "--stretch-guard-bones",
        default="all",
        choices=("all", "terminal", "nonterminal"),
    )
    parser.add_argument("--ik-leaf-refine-enabled", action="store_true", default=False)
    parser.add_argument("--ik-leaf-strength", type=float, default=0.35)
    parser.add_argument("--ik-leaf-iterations", type=int, default=6)
    parser.add_argument("--ik-leaf-max-chain-length", type=int, default=3)
    parser.add_argument("--ik-leaf-preserve-scale", type=float, default=0.65)
    parser.add_argument("--drop-problematic-frames", action="store_true", default=False)
    parser.add_argument("--preserve-root-motion", action="store_true", default=False)
    parser.add_argument("--preserve-root-rotation", action="store_true", default=False)
    parser.add_argument("--bvh-output", help="Path where a BVH export should be written")
    parser.add_argument(
        "--parts-json", help="JSON file with part triangle keys and projection frames"
    )
    parser.add_argument("--images-dir", help="Directory where rendered part PNGs will be written")
    parser.add_argument(
        "--base-color-texture-output",
        help="Optional PNG path for the model's full-resolution base-color texture",
    )
    parser.add_argument(
        "--resolution", type=int, default=2048, help="Render resolution for each part image"
    )
    parser.add_argument(
        "--bind-frame", type=int, default=0, help="Frame to use for bind-pose sprite rendering"
    )
    parser.add_argument(
        "--mesh-target-vertices",
        type=int,
        default=5000,
        help="Target vertex count for source mesh reduction",
    )
    parser.add_argument(
        "--no-mesh-reduction",
        dest="mesh_reduction",
        action="store_false",
        default=True,
        help="Disable source mesh reduction",
    )
    parser.add_argument(
        "--weight-aware-decimation",
        dest="weight_aware_decimation",
        action="store_true",
        default=False,
        help="Bias mesh decimation toward blend/joint regions. Default is uniform decimation.",
    )
    parser.add_argument(
        "--no-weight-aware-decimation",
        dest="weight_aware_decimation",
        action="store_false",
        help="Use uniform mesh decimation.",
    )
    parser.add_argument(
        "--target-spec",
        default=None,
        help="JSON file describing per-animation sample_times for extract-mesh-targets",
    )
    parser.add_argument(
        "--bind-from-animation",
        default=None,
        help=(
            "Path to an external animation file (.fbx/.glb). When the source "
            "model has no actions of its own, pre-load this animation and "
            "use its first frame as the bind pose so the generated 2D rig "
            "inherits a natural starting pose. No-op when the model already "
            "carries its own action."
        ),
    )
    parser.add_argument(
        "--decouple-scale",
        action="store_true",
        default=False,
        help=(
            "Direct extraction only: force each bone's world basis orthogonal "
            "(rotation + foreshortening, scale_y=1) and emit the scale_y/shear_y "
            "that cancels a non-uniform parent's inherited skew, eliminating the "
            "'underwater' ripple."
        ),
    )
    parser.add_argument(
        "--animation-source",
        default=None,
        help=(
            "Path to an external animation file to use as the animation data "
            "source. When provided, this file's armature is imported separately "
            "and its bone rotations are transferred to the target model. Use "
            "together with --bind-from-animation to keep a consistent bind pose "
            "across multiple extractions."
        ),
    )
    parser.add_argument(
        "--bake-spec",
        default=None,
        help="JSON file with per-bone local transforms for bake-rig-animation",
    )
    parser.add_argument(
        "--flat-output",
        default=None,
        help="Animation file (.fbx/.glb) written by bake-rig-animation",
    )
    parser.add_argument(
        "--glb-output",
        default=None,
        help="Cleaned mesh GLB written by cleanup-mesh",
    )
    parser.add_argument(
        "--target-triangles",
        type=int,
        default=10000,
        help="Triangle budget for cleanup-mesh decimation (0 disables)",
    )
    parser.add_argument(
        "--no-voxel-remesh",
        dest="voxel_remesh",
        action="store_false",
        default=True,
        help="Skip the voxel remesh pass (keeps UVs; leaves holes as-is)",
    )
    parser.add_argument(
        "--no-remove-loose",
        dest="remove_loose",
        action="store_false",
        default=True,
        help="Keep floating debris islands",
    )
    parser.add_argument(
        "--fbx-output",
        default=None,
        help="Rigged FBX (bake-predicted-rig) or cleaned FBX (cleanup-mesh no-rig path)",
    )
    parser.add_argument(
        "--orientation-fix",
        default="none",
        choices=("none", "y_up_to_z_up"),
        help="Bake an up-axis correction into the cleaned mesh (TripoSR is Y-up)",
    )
    return parser.parse_args(_worker_script_args(argv))
