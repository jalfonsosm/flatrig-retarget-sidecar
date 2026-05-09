"""Stable CLI surface for the public retargeting sidecar."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from flatrig_retarget_sidecar import __version__
from flatrig_retarget_sidecar.motion2motion_retarget import (
    probe_motion2motion_backend,
    retarget_bvh_to_spine_animation,
    retarget_spine_animation,
)
from flatrig_retarget_sidecar.gmr_retarget import (
    retarget_3d_animations_to_model_gmr,
    retarget_bvh_to_spine_animation_gmr,
    retarget_spine_animation_gmr,
)
from flatrig_retarget_sidecar.retarget_3d import retarget_3d_animations_to_model
from flatrig_retarget_sidecar.scene_formats import (
    convert_3d_source,
    export_3d_animation_bvh,
    export_3d_rest_bvh,
    extract_animations,
    extract_scene,
    inspect_3d_source,
    probe_scene_backend,
    render_sprites,
)
from flatrig_retarget_sidecar.spine_import import load_spine_package


def _add_projection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--view", dest="view_name", default="side")
    parser.add_argument("--view-dir", default=None)
    parser.add_argument("--view-up", dest="view_up", default=None)
    parser.add_argument("--view-roll", dest="view_roll", type=float, default=0.0)
    parser.add_argument("--source-frame", type=int, default=None)
    parser.add_argument("--use-rest-pose", action="store_true", default=False)
    parser.add_argument("--projection-space", choices=("world", "root"), default="world")
    parser.add_argument("--animation", dest="animation_names", action="append", default=[])


def main() -> None:
    parser = argparse.ArgumentParser(description="Public retargeting sidecar backend.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("probe", help="probe the public sidecar backend")

    backend_choices = ("auto", "m2m", "mixamo", "gmr")

    spine_parser = subparsers.add_parser(
        "retarget-spine",
        help="retarget one Spine animation into a target Spine skeleton",
    )
    spine_parser.add_argument("source")
    spine_parser.add_argument("target")
    spine_parser.add_argument("--animation", required=True)
    spine_parser.add_argument("--target-animation", default=None)
    spine_parser.add_argument("--output", required=True)
    spine_parser.add_argument("--matching-alpha", type=float, default=None)
    spine_parser.add_argument("--mapping-file", default=None)
    spine_parser.add_argument("--backend", choices=backend_choices, default="auto")

    bvh_to_spine_parser = subparsers.add_parser(
        "retarget-bvh-to-spine",
        help="retarget one BVH animation into a target Spine skeleton",
    )
    bvh_to_spine_parser.add_argument("source_bvh")
    bvh_to_spine_parser.add_argument("target")
    bvh_to_spine_parser.add_argument("--animation-name", default=None)
    bvh_to_spine_parser.add_argument("--target-animation", default=None)
    bvh_to_spine_parser.add_argument("--output", required=True)
    bvh_to_spine_parser.add_argument("--matching-alpha", type=float, default=None)
    bvh_to_spine_parser.add_argument("--mapping-file", default=None)
    bvh_to_spine_parser.add_argument("--backend", choices=backend_choices, default="auto")

    retarget_3d_parser = subparsers.add_parser(
        "retarget-3d-animation-to-model",
        help="retarget 3D source animations onto a target 3D model rig via Motion2Motion",
    )
    retarget_3d_parser.add_argument("source")
    retarget_3d_parser.add_argument("target_model")
    retarget_3d_parser.add_argument("--output", required=True)
    _add_projection_args(retarget_3d_parser)
    retarget_3d_parser.add_argument("--fps", type=float, default=30.0)
    retarget_3d_parser.add_argument("--frame-start", type=int, default=None)
    retarget_3d_parser.add_argument("--frame-end", type=int, default=None)
    retarget_3d_parser.add_argument("--mapping-file", default=None)
    retarget_3d_parser.add_argument("--matching-alpha", type=float, default=None)
    retarget_3d_parser.add_argument("--mapping-quality-threshold", type=float, default=0.55)
    retarget_3d_parser.add_argument("--force-mapping-review", action="store_true", default=False)
    retarget_3d_parser.add_argument("--include-preview-3d", action="store_true", default=False)
    retarget_3d_parser.add_argument("--backend", choices=backend_choices, default="auto")

    spine_to_json_parser = subparsers.add_parser(
        "spine-to-json",
        help="load a Spine source (.json/.zip/.skel) and emit a normalized JSON skeleton payload",
    )
    spine_to_json_parser.add_argument("source")
    spine_to_json_parser.add_argument("--output", required=True)

    inspect_3d_parser = subparsers.add_parser(
        "inspect-3d-source",
        help="inspect a 3D source through the public Blender sidecar",
    )
    inspect_3d_parser.add_argument("source")
    inspect_3d_parser.add_argument("--output", required=True)

    convert_3d_parser = subparsers.add_parser(
        "convert-3d-source",
        help="normalize a 3D source through Blender into a format the native app can consume",
    )
    convert_3d_parser.add_argument("source")
    convert_3d_parser.add_argument("--output", required=True)

    extract_scene_parser = subparsers.add_parser(
        "extract-scene",
        help="extract mesh, skeleton, and weights from a 3D source as JSON",
    )
    extract_scene_parser.add_argument("source")
    extract_scene_parser.add_argument("--output", required=True)
    _add_projection_args(extract_scene_parser)
    extract_scene_parser.add_argument("--mesh-target-vertices", type=int, default=5000)
    extract_scene_parser.add_argument(
        "--no-mesh-reduction", dest="mesh_reduction", action="store_false", default=True
    )

    extract_animations_parser = subparsers.add_parser(
        "extract-animations",
        help="extract animations from a 3D source armature as JSON",
    )
    extract_animations_parser.add_argument("source")
    extract_animations_parser.add_argument("--output", required=True)
    _add_projection_args(extract_animations_parser)
    extract_animations_parser.add_argument("--fps", type=float, default=30.0)
    extract_animations_parser.add_argument("--frame-start", type=int, default=None)
    extract_animations_parser.add_argument("--frame-end", type=int, default=None)
    extract_animations_parser.add_argument("--sample-substeps", type=int, default=2)
    extract_animations_parser.add_argument(
        "--no-optimize-animation-keys", dest="optimize_animation_keys", action="store_false"
    )
    extract_animations_parser.set_defaults(optimize_animation_keys=True)
    extract_animations_parser.add_argument(
        "--force-loop-closing-keys", action="store_true", default=False
    )
    extract_animations_parser.add_argument(
        "--pose-mode",
        default="full",
        choices=("full", "rotation_only", "local_rotation", "blend"),
    )
    extract_animations_parser.add_argument("--pose-blend", type=float, default=1.0)
    extract_animations_parser.add_argument("--rotation-flatten", type=float, default=0.0)
    extract_animations_parser.add_argument("--rotation-flatten-scope", default="all")
    extract_animations_parser.add_argument(
        "--stretch-guard-enabled", action="store_true", default=False
    )
    extract_animations_parser.add_argument("--stretch-guard-max-scale", type=float, default=1.75)
    extract_animations_parser.add_argument("--stretch-guard-strength", type=float, default=0.65)
    extract_animations_parser.add_argument(
        "--ik-leaf-refine-enabled", action="store_true", default=False
    )
    extract_animations_parser.add_argument("--ik-leaf-strength", type=float, default=0.35)
    extract_animations_parser.add_argument("--ik-leaf-iterations", type=int, default=6)
    extract_animations_parser.add_argument("--ik-leaf-max-chain-length", type=int, default=3)
    extract_animations_parser.add_argument("--ik-leaf-preserve-scale", type=float, default=0.65)
    extract_animations_parser.add_argument(
        "--drop-problematic-frames", action="store_true", default=False
    )
    extract_animations_parser.add_argument(
        "--preserve-root-motion", action="store_true", default=False
    )
    extract_animations_parser.add_argument(
        "--preserve-root-rotation", action="store_true", default=False
    )

    export_3d_animation_bvh_parser = subparsers.add_parser(
        "export-3d-animation-bvh",
        help="export one 3D source animation as a Motion2Motion-friendly BVH",
    )
    export_3d_animation_bvh_parser.add_argument("source")
    export_3d_animation_bvh_parser.add_argument("--output", required=True)
    export_3d_animation_bvh_parser.add_argument("--bvh-output", required=True)
    export_3d_animation_bvh_parser.add_argument("--animation", default=None)
    export_3d_animation_bvh_parser.add_argument("--fps", type=float, default=30.0)
    export_3d_animation_bvh_parser.add_argument("--frame-start", type=int, default=None)
    export_3d_animation_bvh_parser.add_argument("--frame-end", type=int, default=None)

    export_3d_rest_bvh_parser = subparsers.add_parser(
        "export-3d-rest-bvh",
        help="export a target 3D model rig as a Motion2Motion target BVH",
    )
    export_3d_rest_bvh_parser.add_argument("source")
    export_3d_rest_bvh_parser.add_argument("--output", required=True)
    export_3d_rest_bvh_parser.add_argument("--bvh-output", required=True)
    _add_projection_args(export_3d_rest_bvh_parser)
    export_3d_rest_bvh_parser.add_argument("--fps", type=float, default=30.0)
    export_3d_rest_bvh_parser.add_argument("--frame-count", type=int, default=None)

    render_sprites_parser = subparsers.add_parser(
        "render-sprites",
        help="render sprites from a 3D source as PNG",
    )
    render_sprites_parser.add_argument("source")
    render_sprites_parser.add_argument("--output", required=True)
    _add_projection_args(render_sprites_parser)
    render_sprites_parser.add_argument("--parts-json", required=True)
    render_sprites_parser.add_argument("--images-dir", required=True)
    render_sprites_parser.add_argument("--resolution", type=int, default=2048)
    render_sprites_parser.add_argument("--bind-frame", type=int, default=0)
    render_sprites_parser.add_argument("--mesh-target-vertices", type=int, default=5000)
    render_sprites_parser.add_argument(
        "--no-mesh-reduction", dest="mesh_reduction", action="store_false", default=True
    )

    args = parser.parse_args()

    if args.command == "probe":
        probe = probe_motion2motion_backend()
        payload = asdict(probe)
        payload["backend"] = "motion2motion"
        payload["sidecar_version"] = __version__
        payload["scene_backend"] = probe_scene_backend()
        print(json.dumps(payload, indent=2))
        return

    if args.command == "retarget-spine":
        if args.backend == "gmr":
            payload = retarget_spine_animation_gmr()
            print(json.dumps(payload, indent=2))
            raise SystemExit(1)
        source = load_spine_package(args.source)
        target = load_spine_package(args.target)
        result = retarget_spine_animation(
            source,
            target,
            args.animation,
            target_animation_name=args.target_animation,
            matching_alpha=args.matching_alpha,
            mapping_file=args.mapping_file,
        )
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"animations": {result.animation_name: result.animation}}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(asdict(result), indent=2))
        return

    if args.command == "retarget-bvh-to-spine":
        if args.backend == "gmr":
            payload = retarget_bvh_to_spine_animation_gmr()
            print(json.dumps(payload, indent=2))
            raise SystemExit(1)
        target = load_spine_package(args.target)
        result = retarget_bvh_to_spine_animation(
            args.source_bvh,
            target,
            animation_name=args.animation_name,
            target_animation_name=args.target_animation,
            matching_alpha=args.matching_alpha,
            mapping_file=args.mapping_file,
        )
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"animations": {result.animation_name: result.animation}}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(asdict(result), indent=2))
        return

    if args.command == "retarget-3d-animation-to-model":
        if args.backend == "gmr":
            payload = retarget_3d_animations_to_model_gmr(
                args.source,
                args.target_model,
                output=args.output,
            )
            print(json.dumps(payload, indent=2))
            raise SystemExit(1)
        result = retarget_3d_animations_to_model(
            args.source,
            args.target_model,
            animation_names=args.animation_names,
            output=args.output,
            mapping_file=args.mapping_file,
            matching_alpha=args.matching_alpha,
            quality_threshold=args.mapping_quality_threshold,
            force_mapping_review=args.force_mapping_review,
            view_preset=args.view_name,
            view_dir=args.view_dir,
            view_up=args.view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            projection_space=args.projection_space,
            fps=args.fps,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            include_preview_3d=args.include_preview_3d,
        )
        print(json.dumps(result, indent=2))
        if not result.get("ok"):
            raise SystemExit(1)
        return

    if args.command == "spine-to-json":
        package = load_spine_package(args.source)
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(package.payload, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "ok": True,
                    "source": str(Path(args.source).expanduser().resolve()),
                    "output": str(output_path),
                    "bone_count": len(package.bones),
                    "animation_count": len(package.animations),
                    "slot_count": len(package.slots),
                },
                indent=2,
            )
        )
        return

    if args.command == "inspect-3d-source":
        result = inspect_3d_source(args.source, args.output)
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    if args.command == "convert-3d-source":
        result = convert_3d_source(args.source, args.output)
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    if args.command == "export-3d-animation-bvh":
        result = export_3d_animation_bvh(
            args.source,
            args.output,
            bvh_output=args.bvh_output,
            animation_name=args.animation,
            fps=args.fps,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    if args.command == "export-3d-rest-bvh":
        result = export_3d_rest_bvh(
            args.source,
            args.output,
            bvh_output=args.bvh_output,
            view_preset=args.view_name,
            view_dir=args.view_dir,
            view_up=args.view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            use_rest_pose=args.use_rest_pose,
            projection_space=args.projection_space,
            fps=args.fps,
            frame_count=args.frame_count,
        )
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    if args.command == "extract-scene":
        result = extract_scene(
            args.source,
            args.output,
            view_preset=args.view_name,
            view_dir=args.view_dir,
            view_up=args.view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            use_rest_pose=args.use_rest_pose,
            projection_space=args.projection_space,
            mesh_reduction=args.mesh_reduction,
            mesh_target_vertices=args.mesh_target_vertices,
        )
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    if args.command == "extract-animations":
        result = extract_animations(
            args.source,
            args.output,
            view_preset=args.view_name,
            view_dir=args.view_dir,
            view_up=args.view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            projection_space=args.projection_space,
            animation_names=args.animation_names,
            fps=args.fps,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            sample_substeps=args.sample_substeps,
            optimize_animation_keys=args.optimize_animation_keys,
            force_loop_closing_keys=args.force_loop_closing_keys,
            pose_mode=args.pose_mode,
            pose_blend=args.pose_blend,
            rotation_flatten=args.rotation_flatten,
            rotation_flatten_scope=args.rotation_flatten_scope,
            stretch_guard_enabled=args.stretch_guard_enabled,
            stretch_guard_max_scale=args.stretch_guard_max_scale,
            stretch_guard_strength=args.stretch_guard_strength,
            ik_leaf_refine_enabled=args.ik_leaf_refine_enabled,
            ik_leaf_strength=args.ik_leaf_strength,
            ik_leaf_iterations=args.ik_leaf_iterations,
            ik_leaf_max_chain_length=args.ik_leaf_max_chain_length,
            ik_leaf_preserve_scale=args.ik_leaf_preserve_scale,
            drop_problematic_frames=args.drop_problematic_frames,
            preserve_root_motion=args.preserve_root_motion,
            preserve_root_rotation=args.preserve_root_rotation,
        )
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    if args.command == "render-sprites":
        result = render_sprites(
            args.source,
            args.output,
            parts_json=args.parts_json,
            images_dir=args.images_dir,
            view_preset=args.view_name,
            view_dir=args.view_dir,
            view_up=args.view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            use_rest_pose=args.use_rest_pose,
            projection_space=args.projection_space,
            resolution=args.resolution,
            bind_frame=args.bind_frame,
            mesh_reduction=args.mesh_reduction,
            mesh_target_vertices=args.mesh_target_vertices,
        )
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
