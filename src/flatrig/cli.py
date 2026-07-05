"""Stable CLI surface for the public Blender worker sidecar.

Only Blender-bound (`bpy`) worker commands live here: scene/animation/mesh
extraction, rig dump/bake, sprite rendering, and BVH/format export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flatrig import __version__
from flatrig.scene_formats import (
    bake_predicted_rig,
    bake_rig_animation,
    cleanup_mesh,
    convert_3d_source,
    dump_rig_animation,
    export_3d_animation_bvh,
    export_3d_rest_bvh,
    extract_animations,
    extract_mesh_targets,
    extract_scene,
    inspect_3d_source,
    probe_scene_backend,
    reduce_rig_to_canonical,
    render_sprites,
)


def _add_projection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--view", dest="view_name", default="side")
    parser.add_argument("--view-dir", default=None)
    parser.add_argument("--view-up", dest="view_up", default=None)
    parser.add_argument("--view-roll", dest="view_roll", type=float, default=0.0)
    parser.add_argument("--source-frame", type=int, default=None)
    parser.add_argument("--use-rest-pose", action="store_true", default=False)
    parser.add_argument("--projection-space", choices=("world", "root"), default="world")
    parser.add_argument("--animation", dest="animation_names", action="append", default=[])


def _add_weight_aware_decimation_args(parser: argparse.ArgumentParser) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Public Blender worker sidecar.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("probe", help="probe the public Blender worker backend")

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
    _add_weight_aware_decimation_args(extract_scene_parser)
    extract_scene_parser.add_argument(
        "--bind-from-animation",
        default=None,
        help=(
            "Path to an external animation file (.fbx/.glb/...). When the "
            "source model has no actions of its own, the first frame of this "
            "animation is loaded and used as the bind/setup pose so the "
            "generated 2D rig inherits a natural starting pose (a slight "
            "walking step) instead of the bare T-pose. Required for clean "
            "cross-rig retargets onto T-pose mannequins."
        ),
    )
    extract_scene_parser.add_argument(
        "--base-color-texture-output",
        default=None,
        help="Write the model's full-resolution base-color texture to this PNG path.",
    )


    extract_animations_parser = subparsers.add_parser(
        "extract-animations",
        help="extract animations from a 3D source armature as JSON",
    )
    extract_animations_parser.add_argument("source")
    extract_animations_parser.add_argument("--output", required=True)
    _add_projection_args(extract_animations_parser)
    extract_animations_parser.add_argument("--bind-from-animation", default=None)
    extract_animations_parser.add_argument("--animation-source", default=None)
    extract_animations_parser.add_argument("--decouple-scale", action="store_true", default=False)
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


    dump_rig_parser = subparsers.add_parser(
        "dump-rig-animation",
        help=(
            "dump rig topology and per-frame pose matrices (world + local "
            "basis) for one action as JSON, for external animation processors"
        ),
    )
    dump_rig_parser.add_argument("source")
    dump_rig_parser.add_argument("--output", required=True)
    dump_rig_parser.add_argument(
        "--animation", dest="animation_names", action="append", default=[]
    )
    dump_rig_parser.add_argument("--frame-start", type=int, default=None)
    dump_rig_parser.add_argument("--frame-end", type=int, default=None)


    bake_rig_parser = subparsers.add_parser(
        "bake-rig-animation",
        help=(
            "bake externally computed local pose transforms (JSON spec) onto "
            "a rig and export the animated file (.fbx/.glb)"
        ),
    )
    bake_rig_parser.add_argument("source")
    bake_rig_parser.add_argument("--output", required=True)
    bake_rig_parser.add_argument("--bake-spec", required=True)
    bake_rig_parser.add_argument("--flat-output", required=True)

    reduce_rig_parser = subparsers.add_parser(
        "reduce-rig-to-canonical",
        help=(
            "reduce a biped-humanoid rig to the FlatRig HML22 canonical skeleton in place "
            "on the mesh (rename/drop/reparent bones, transfer weights) and export"
        ),
    )
    reduce_rig_parser.add_argument("source")
    reduce_rig_parser.add_argument("--output", required=True)
    reduce_rig_parser.add_argument("--flat-output", required=True)


    extract_mesh_targets_parser = subparsers.add_parser(
        "extract-mesh-targets",
        help="evaluate animated mesh per frame and project vertices to 2D",
    )
    extract_mesh_targets_parser.add_argument("source")
    extract_mesh_targets_parser.add_argument("--output", required=True)
    _add_projection_args(extract_mesh_targets_parser)
    extract_mesh_targets_parser.add_argument("--bind-from-animation", default=None)
    extract_mesh_targets_parser.add_argument("--target-spec", required=True)
    extract_mesh_targets_parser.add_argument("--mesh-target-vertices", type=int, default=5000)
    extract_mesh_targets_parser.add_argument(
        "--no-mesh-reduction",
        dest="mesh_reduction",
        action="store_false",
        default=True,
    )
    _add_weight_aware_decimation_args(extract_mesh_targets_parser)


    render_sprites_parser = subparsers.add_parser(
        "render-sprites",
        help="render sprites from a 3D source as PNG",
    )
    render_sprites_parser.add_argument("source")
    render_sprites_parser.add_argument("--output", required=True)
    _add_projection_args(render_sprites_parser)
    render_sprites_parser.add_argument("--bind-from-animation", default=None)
    render_sprites_parser.add_argument("--parts-json", required=True)
    render_sprites_parser.add_argument("--images-dir", required=True)
    render_sprites_parser.add_argument("--resolution", type=int, default=2048)
    render_sprites_parser.add_argument("--bind-frame", type=int, default=0)
    render_sprites_parser.add_argument("--mesh-target-vertices", type=int, default=5000)
    render_sprites_parser.add_argument(
        "--no-mesh-reduction", dest="mesh_reduction", action="store_false", default=True
    )
    _add_weight_aware_decimation_args(render_sprites_parser)


    export_anim_bvh_parser = subparsers.add_parser(
        "export-3d-animation-bvh",
        help="export one 3D action to BVH via the public Blender sidecar",
    )
    export_anim_bvh_parser.add_argument("source")
    export_anim_bvh_parser.add_argument("--output", required=True)
    export_anim_bvh_parser.add_argument("--bvh-output", required=True)
    export_anim_bvh_parser.add_argument("--animation-name", default=None)
    export_anim_bvh_parser.add_argument("--fps", type=float, default=30.0)
    export_anim_bvh_parser.add_argument("--frame-start", type=int, default=None)
    export_anim_bvh_parser.add_argument("--frame-end", type=int, default=None)

    cleanup_mesh_parser = subparsers.add_parser(
        "cleanup-mesh",
        help="clean a raw generated mesh (join, drop debris, remesh, decimate) to GLB",
    )
    cleanup_mesh_parser.add_argument("source")
    cleanup_mesh_parser.add_argument("--output", required=True)
    cleanup_mesh_parser.add_argument("--glb-output", required=True)
    cleanup_mesh_parser.add_argument(
        "--fbx-output",
        default=None,
        help="Also export the cleaned mesh as FBX (the no-rig path's final asset)",
    )
    cleanup_mesh_parser.add_argument(
        "--orientation-fix",
        default="none",
        choices=("none", "y_up_to_z_up"),
        help="Bake an up-axis correction into the cleaned mesh (TripoSR is Y-up)",
    )
    cleanup_mesh_parser.add_argument("--target-triangles", type=int, default=10000)
    cleanup_mesh_parser.add_argument(
        "--no-voxel-remesh", dest="voxel_remesh", action="store_false", default=True
    )
    cleanup_mesh_parser.add_argument(
        "--no-remove-loose", dest="remove_loose", action="store_false", default=True
    )

    bake_predicted_rig_parser = subparsers.add_parser(
        "bake-predicted-rig",
        help=(
            "build a from-scratch armature (no template file) for an "
            "externally predicted mesh/bones/weights .npz and export FBX"
        ),
    )
    bake_predicted_rig_parser.add_argument("source", help="Path to the prediction .npz")
    bake_predicted_rig_parser.add_argument("--output", required=True)
    bake_predicted_rig_parser.add_argument("--fbx-output", required=True)

    export_rest_bvh_parser = subparsers.add_parser(
        "export-3d-rest-bvh",
        help="export the rest/bind pose to BVH via the public Blender sidecar",
    )
    export_rest_bvh_parser.add_argument("source")
    export_rest_bvh_parser.add_argument("--output", required=True)
    export_rest_bvh_parser.add_argument("--bvh-output", required=True)
    _add_projection_args(export_rest_bvh_parser)
    export_rest_bvh_parser.add_argument("--fps", type=float, default=30.0)
    export_rest_bvh_parser.add_argument("--frame-count", type=int, default=None)
    export_rest_bvh_parser.add_argument("--bind-from-animation", default=None)

    args = parser.parse_args()

    if args.command == "probe":
        payload = {
            "backend": "blender_worker",
            "sidecar_version": __version__,
            "scene_backend": probe_scene_backend(),
        }
        print(json.dumps(payload, indent=2))
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
            weight_aware_decimation=args.weight_aware_decimation,
            bind_from_animation=getattr(args, "bind_from_animation", None),
            base_color_texture_output=getattr(
                args, "base_color_texture_output", None
            ),
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
            bind_from_animation=getattr(args, "bind_from_animation", None),
            animation_source=getattr(args, "animation_source", None),
            decouple_scale=getattr(args, "decouple_scale", False),
        )
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return


    if args.command == "dump-rig-animation":
        result = dump_rig_animation(
            args.source,
            args.output,
            animation_names=args.animation_names,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
        # The full dump (per-frame matrices) lives in the output file; keep
        # stdout to a light summary so callers can log it.
        summary = {
            key: value
            for key, value in result.payload.items()
            if key not in {"frames", "bones", "armature_matrix_world"}
        }
        summary.setdefault("output", args.output)
        print(json.dumps(summary, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return


    if args.command == "bake-rig-animation":
        result = bake_rig_animation(
            args.source,
            args.output,
            bake_spec=args.bake_spec,
            flat_output=args.flat_output,
        )
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    if args.command == "reduce-rig-to-canonical":
        result = reduce_rig_to_canonical(
            args.source,
            args.output,
            flat_output=args.flat_output,
        )
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    if args.command == "cleanup-mesh":
        result = cleanup_mesh(
            args.source,
            args.output,
            glb_output=args.glb_output,
            target_triangles=args.target_triangles,
            voxel_remesh=args.voxel_remesh,
            remove_loose=args.remove_loose,
            fbx_output=args.fbx_output,
            orientation_fix=args.orientation_fix,
        )
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return


    if args.command == "bake-predicted-rig":
        result = bake_predicted_rig(args.source, args.output, fbx_output=args.fbx_output)
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    if args.command == "extract-mesh-targets":
        result = extract_mesh_targets(
            args.source,
            args.output,
            target_spec=args.target_spec,
            view_preset=args.view_name,
            view_dir=args.view_dir,
            view_up=args.view_up,
            view_roll=args.view_roll,
            source_frame=args.source_frame,
            use_rest_pose=args.use_rest_pose,
            projection_space=args.projection_space,
            mesh_reduction=args.mesh_reduction,
            mesh_target_vertices=args.mesh_target_vertices,
            weight_aware_decimation=args.weight_aware_decimation,
            bind_from_animation=getattr(args, "bind_from_animation", None),
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
            weight_aware_decimation=args.weight_aware_decimation,
            bind_from_animation=getattr(args, "bind_from_animation", None),
        )
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return


    if args.command == "export-3d-animation-bvh":
        result = export_3d_animation_bvh(
            args.source,
            args.output,
            bvh_output=args.bvh_output,
            animation_name=args.animation_name,
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
            bind_from_animation=args.bind_from_animation,
        )
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
