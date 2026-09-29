"""Standalone Blender worker for offline asset curation commands.

This is intentionally separate from ``blender_scene_io.py``: curation has a
different contract (linked ``.blend`` sources, exact FBX export options, and a
declarative rest-pose edit) and should not enlarge the pipeline's primary scene
extraction worker.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from blender_worker_bootstrap import prepend_sidecar_src

prepend_sidecar_src(__file__)

from blender_curation_carrier import rewrite_armature_rest_pose
from blender_curation_export import export_curation_fbx, inspect_curation_source


COMMANDS = (
    "inspect-curation-source",
    "export-curation-fbx",
    "rewrite-armature-rest-pose",
)


def _worker_argv(argv: list[str] | None = None) -> list[str]:
    values = list(sys.argv if argv is None else argv)
    try:
        return values[values.index("--") + 1 :]
    except ValueError:
        return values[1:]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public Blender curation operations.")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("source")
    parser.add_argument("--output", required=True, help="JSON result report")
    parser.add_argument("--source-root", default=None, help="Third-party source catalog root")
    parser.add_argument("--fbx-output", default=None, help="Destination FBX for export commands")
    parser.add_argument("--action-name", default=None, help="Action to bake or name")
    parser.add_argument("--embed-textures", action="store_true", default=False)
    parser.add_argument("--clean-control-bones", action="store_true", default=False)
    parser.add_argument("--rest-spec", default=None, help="JSON edit-bone rest-pose specification")
    return parser.parse_args(_worker_argv(argv))


def _require(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required for this command")
    return value


def _run(args: argparse.Namespace) -> dict[str, object]:
    source = str(Path(args.source).expanduser().resolve())
    if args.command == "inspect-curation-source":
        return inspect_curation_source(source, _require(args.source_root, "--source-root"))
    if args.command == "export-curation-fbx":
        return export_curation_fbx(
            source,
            _require(args.source_root, "--source-root"),
            _require(args.fbx_output, "--fbx-output"),
            action_name=args.action_name,
            embed_textures=args.embed_textures,
            clean_control_bones=args.clean_control_bones,
        )
    if args.command == "rewrite-armature-rest-pose":
        return rewrite_armature_rest_pose(
            source,
            _require(args.fbx_output, "--fbx-output"),
            _require(args.rest_spec, "--rest-spec"),
            action_name=args.action_name,
        )
    raise AssertionError(f"Unhandled curation command: {args.command}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = _run(args)
    except Exception as exc:  # pragma: no cover - Blender runtime
        payload = {"ok": False, "detail": str(exc), "traceback": traceback.format_exc()}
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(1) from exc
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
