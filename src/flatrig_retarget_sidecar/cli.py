"""Stable CLI surface for the public flatRig sidecar."""

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
from flatrig_retarget_sidecar.scene_formats import (
    convert_3d_source,
    inspect_3d_source,
    probe_scene_backend,
)
from flatrig_retarget_sidecar.spine_import import load_spine_package


def main() -> None:
    parser = argparse.ArgumentParser(description="flatRig public sidecar backend.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe", help="probe the public sidecar backend")
    probe_parser.add_argument("--backend", default="motion2motion")
    probe_parser.add_argument(
        "--include-scene-backend",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

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
    convert_3d_parser.add_argument("--target-format", default="glb")

    args = parser.parse_args()

    if args.command == "probe":
        probe = probe_motion2motion_backend()
        payload = asdict(probe)
        payload["backend"] = args.backend
        payload["sidecar_version"] = __version__
        if args.include_scene_backend:
            payload["scene_backend"] = probe_scene_backend()
        print(json.dumps(payload, indent=2))
        return

    if args.command == "retarget-spine":
        source = load_spine_package(args.source)
        target = load_spine_package(args.target)
        result = retarget_spine_animation(
            source,
            target,
            args.animation,
            target_animation_name=args.target_animation,
            matching_alpha=args.matching_alpha,
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
        target = load_spine_package(args.target)
        result = retarget_bvh_to_spine_animation(
            args.source_bvh,
            target,
            animation_name=args.animation_name,
            target_animation_name=args.target_animation,
            matching_alpha=args.matching_alpha,
        )
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"animations": {result.animation_name: result.animation}}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(asdict(result), indent=2))
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
        result = convert_3d_source(args.source, args.output, target_format=args.target_format)
        print(json.dumps(result.payload, indent=2))
        if not result.ok:
            raise SystemExit(1)
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
