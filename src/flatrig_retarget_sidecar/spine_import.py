"""Import Spine JSON packages into a neutral flatRig representation."""

from __future__ import annotations

import functools
import json
import math
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
BONE_LENGTH_EPSILON = 1e-6
ROOT_DIR = Path(__file__).resolve().parents[2]
SPINE_BINARY_TOOL = ROOT_DIR / "tools" / "spine_binary_export.cjs"
SPINE_BINARY_RUNTIME_DIR = ROOT_DIR / "workflow" / ".spine_binary_runtime"
SPINE_BINARY_RUNTIME_MANIFEST = {
    "name": "flatrig-spine-binary-runtime",
    "private": True,
    "dependencies": {
        "@pixi-spine/runtime-3.8": "4.0.6",
        "@pixi-spine/runtime-4.0": "4.0.6",
        "@pixi-spine/runtime-4.1": "4.0.6",
        "@esotericsoftware/spine-core": "4.2.106",
    },
}


@dataclass(slots=True)
class SpineBoneRecord:
    index: int
    name: str
    parent: str | None
    x: float
    y: float
    rotation: float
    scale_x: float
    scale_y: float
    length: float
    inherit: str
    children: list[str] = field(default_factory=list)
    child_count: int = 0
    descendant_count: int = 0
    depth: int = 0
    slot_count: int = 0
    world_x: float = 0.0
    world_y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0
    world_rotation: float = 0.0
    world_scale_x: float = 1.0
    world_scale_y: float = 1.0
    world_basis: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 0.0),
        (0.0, 1.0),
    )


@dataclass(slots=True)
class SpinePackage:
    source_label: str
    payload: dict[str, Any]
    skeleton_info: dict[str, Any]
    bones: list[SpineBoneRecord]
    bones_by_name: dict[str, SpineBoneRecord]
    slots: list[dict[str, Any]]
    animations: dict[str, Any]
    skins: list[Any]
    attachment_types: dict[str, int]
    setup_bounds: dict[str, float]

    def summary(self) -> dict[str, Any]:
        return {
            "source_label": self.source_label,
            "spine_version": str(self.skeleton_info.get("spine") or "unknown"),
            "bone_count": len(self.bones),
            "slot_count": len(self.slots),
            "animation_count": len(self.animations),
            "skin_count": len(self.skins),
            "attachment_types": dict(self.attachment_types),
            "setup_bounds": dict(self.setup_bounds),
        }


def load_spine_package(source: str | Path) -> SpinePackage:
    source_label, payload = load_spine_payload(source)
    return build_spine_package(payload, source_label=source_label)


def load_spine_payload(source: str | Path) -> tuple[str, dict[str, Any]]:
    source_path = str(source)
    if "!/" in source_path:
        archive_path, inner_path = source_path.split("!/", 1)
        return _load_spine_payload_from_zip(Path(archive_path), inner_path)

    path = Path(source_path)
    if path.suffix.lower() == ".zip":
        return _load_single_spine_payload_from_zip(path)
    if path.suffix.lower() == ".skel":
        return str(path.resolve()), _load_spine_binary_payload(path)

    payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(payload, dict) or "bones" not in payload:
        raise ValueError(f"Not a Spine JSON skeleton: {path}")
    return str(path.resolve()), payload


def load_spine_json_payload(source: str | Path) -> tuple[str, dict[str, Any]]:
    return load_spine_payload(source)


def build_spine_package(payload: dict[str, Any], *, source_label: str = "<memory>") -> SpinePackage:
    if not isinstance(payload, dict) or "bones" not in payload:
        raise ValueError("Spine payload must be a JSON object with a 'bones' array.")

    bones, bones_by_name = _build_bone_records(payload)
    slots = list(payload.get("slots") or [])
    _apply_slot_counts(bones_by_name, slots)
    _compute_world_setup(bones, bones_by_name)
    attachment_types = _summarize_attachment_types(payload)
    setup_bounds = _compute_setup_bounds(bones)

    return SpinePackage(
        source_label=source_label,
        payload=payload,
        skeleton_info=dict(payload.get("skeleton") or {}),
        bones=bones,
        bones_by_name=bones_by_name,
        slots=slots,
        animations=dict(payload.get("animations") or {}),
        skins=list(payload.get("skins") or []),
        attachment_types=attachment_types,
        setup_bounds=setup_bounds,
    )


def _load_spine_payload_from_zip(path: Path, inner_path: str) -> tuple[str, dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        try:
            blob = archive.read(inner_path)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Missing Spine skeleton entry '{inner_path}' in {path}"
            ) from exc
    inner_suffix = _compound_suffix_from_name(PurePosixPath(inner_path).name)
    source_label = f"{path.resolve()}!/{inner_path}"
    if inner_suffix == ".skel":
        return source_label, _load_spine_binary_payload_from_blob(blob, source_label=source_label)

    payload = json.loads(blob.decode("utf-8", errors="ignore"))
    if not isinstance(payload, dict) or "bones" not in payload:
        raise ValueError(f"Not a Spine JSON skeleton: {path}!/{inner_path}")
    return source_label, payload


def _load_single_spine_payload_from_zip(path: Path) -> tuple[str, dict[str, Any]]:
    json_candidates: list[tuple[str, dict[str, Any]]] = []
    skel_candidates: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for entry_name in archive.namelist():
            suffix = _compound_suffix_from_name(PurePosixPath(entry_name).name)
            if suffix == ".skel":
                skel_candidates.append(entry_name)
                continue
            if suffix not in {".json", ".skel.json"}:
                continue
            payload = json.loads(archive.read(entry_name).decode("utf-8", errors="ignore"))
            if not isinstance(payload, dict) or "bones" not in payload:
                continue
            json_candidates.append((entry_name, payload))

    if len(json_candidates) == 1:
        entry_name, payload = json_candidates[0]
        return f"{path.resolve()}!/{entry_name}", payload
    if len(json_candidates) > 1:
        preview = ", ".join(name for name, _ in json_candidates[:6])
        if len(json_candidates) > 6:
            preview += ", ..."
        raise ValueError(
            f"Archive contains multiple Spine JSON skeletons. Use 'archive.zip!/path.json'. "
            f"Candidates: {preview}"
        )
    if len(skel_candidates) == 1:
        return _load_spine_payload_from_zip(path, skel_candidates[0])
    if len(skel_candidates) > 1:
        preview = ", ".join(skel_candidates[:6])
        if len(skel_candidates) > 6:
            preview += ", ..."
        raise ValueError(
            f"Archive contains multiple Spine binary skeletons. Use 'archive.zip!/path.skel'. "
            f"Candidates: {preview}"
        )
    raise ValueError(f"No Spine JSON or Spine binary skeleton found in archive: {path}")


def _build_bone_records(
    payload: dict[str, Any],
) -> tuple[list[SpineBoneRecord], dict[str, SpineBoneRecord]]:
    bones: list[SpineBoneRecord] = []
    bones_by_name: dict[str, SpineBoneRecord] = {}

    for index, raw_bone in enumerate(payload.get("bones") or []):
        bone = raw_bone or {}
        name = str(bone.get("name") or "").strip()
        if not name:
            raise ValueError(f"Unnamed bone at index {index} in {payload.get('skeleton', {})!r}")
        parent_name = bone.get("parent")
        record = SpineBoneRecord(
            index=index,
            name=name,
            parent=str(parent_name) if parent_name is not None else None,
            x=float(bone.get("x", 0.0) or 0.0),
            y=float(bone.get("y", 0.0) or 0.0),
            rotation=float(bone.get("rotation", 0.0) or 0.0),
            scale_x=float(bone.get("scaleX", 1.0) or 1.0),
            scale_y=float(bone.get("scaleY", 1.0) or 1.0),
            length=float(bone.get("length", 0.0) or 0.0),
            inherit=str(bone.get("inherit") or bone.get("transform") or "Normal"),
        )
        bones.append(record)
        bones_by_name[record.name] = record

    for bone in bones:
        if bone.parent and bone.parent in bones_by_name:
            bones_by_name[bone.parent].children.append(bone.name)

    for bone in bones:
        bone.child_count = len(bone.children)
        bone.depth = _compute_depth(bone.name, bones_by_name)

    for bone in bones:
        bone.descendant_count = _count_descendants(bone.name, bones_by_name)

    return bones, bones_by_name


def _apply_slot_counts(
    bones_by_name: dict[str, SpineBoneRecord],
    slots: list[dict[str, Any]],
) -> None:
    for slot in slots:
        bone_name = str((slot or {}).get("bone") or "").strip()
        if bone_name and bone_name in bones_by_name:
            bones_by_name[bone_name].slot_count += 1


def _compute_world_setup(
    bones: list[SpineBoneRecord],
    bones_by_name: dict[str, SpineBoneRecord],
) -> None:
    for bone in bones:
        local_basis = _build_basis_2d(bone.rotation, bone.scale_x, bone.scale_y)
        if bone.parent and bone.parent in bones_by_name:
            parent = bones_by_name[bone.parent]
            offset_x, offset_y = _transform_point(parent.world_basis, bone.x, bone.y)
            bone.world_x = parent.world_x + offset_x
            bone.world_y = parent.world_y + offset_y
            bone.world_basis = _multiply_basis_2d(parent.world_basis, local_basis)
            bone.world_rotation = _normalize_angle(parent.world_rotation + bone.rotation)
            bone.world_scale_x = parent.world_scale_x * bone.scale_x
            bone.world_scale_y = parent.world_scale_y * bone.scale_y
        else:
            bone.world_x = bone.x
            bone.world_y = bone.y
            bone.world_basis = local_basis
            bone.world_rotation = _normalize_angle(bone.rotation)
            bone.world_scale_x = bone.scale_x
            bone.world_scale_y = bone.scale_y

        end_offset_x, end_offset_y = _transform_point(bone.world_basis, bone.length, 0.0)
        bone.end_x = bone.world_x + end_offset_x
        bone.end_y = bone.world_y + end_offset_y


def _summarize_attachment_types(payload: dict[str, Any]) -> dict[str, int]:
    skins = payload.get("skins") or []
    if isinstance(skins, dict):
        skin_entries = [skins]
    elif isinstance(skins, list):
        skin_entries = [skin.get("attachments") or {} for skin in skins if isinstance(skin, dict)]
    else:
        skin_entries = []

    counts: dict[str, int] = {}
    for slot_map in skin_entries:
        for attachments in slot_map.values():
            if not isinstance(attachments, dict):
                continue
            for attachment in attachments.values():
                attachment_type = "region"
                if isinstance(attachment, dict):
                    attachment_type = str(attachment.get("type") or "region")
                counts[attachment_type] = counts.get(attachment_type, 0) + 1
    return dict(sorted(counts.items()))


def _compute_setup_bounds(bones: list[SpineBoneRecord]) -> dict[str, float]:
    points_x = [0.0]
    points_y = [0.0]
    for bone in bones:
        points_x.extend([bone.world_x, bone.end_x])
        points_y.extend([bone.world_y, bone.end_y])
    min_x = min(points_x)
    max_x = max(points_x)
    min_y = min(points_y)
    max_y = max(points_y)
    return {
        "min_x": round(min_x, 4),
        "max_x": round(max_x, 4),
        "min_y": round(min_y, 4),
        "max_y": round(max_y, 4),
        "width": round(max_x - min_x, 4),
        "height": round(max_y - min_y, 4),
    }


def path_has_runtime_package(path: str | Path) -> dict[str, Any]:
    source_path = str(path)
    if "!/" in source_path:
        archive_path, inner_path = source_path.split("!/", 1)
        with zipfile.ZipFile(archive_path) as archive:
            parent = PurePosixPath(inner_path).parent
            names = [PurePosixPath(name) for name in archive.namelist()]
        atlas_count = sum(
            1 for name in names if name.parent == parent and name.suffix.lower() == ".atlas"
        )
        image_count = sum(
            1 for name in names if name.parent == parent and name.suffix.lower() in IMAGE_SUFFIXES
        )
        return {"atlas_count": atlas_count, "image_count": image_count}

    direct_path = Path(source_path)
    parent = direct_path.parent
    atlas_count = sum(1 for child in parent.iterdir() if child.suffix.lower() == ".atlas")
    image_count = sum(1 for child in parent.iterdir() if child.suffix.lower() in IMAGE_SUFFIXES)
    return {"atlas_count": atlas_count, "image_count": image_count}


def _compound_suffix_from_name(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".skel.json"):
        return ".skel.json"
    return PurePosixPath(lowered).suffix


def _load_spine_binary_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Spine binary skeleton: {path}")
    return _run_spine_binary_export(path)


def _load_spine_binary_payload_from_blob(blob: bytes, *, source_label: str) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".skel", delete=False) as handle:
        handle.write(blob)
        temp_path = Path(handle.name)
    try:
        return _run_spine_binary_export(temp_path, source_label=source_label)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _run_spine_binary_export(path: Path, *, source_label: str | None = None) -> dict[str, Any]:
    runtime_dir = ensure_spine_binary_runtime()
    node_executable = shutil.which("node")
    if not node_executable:
        raise RuntimeError("Node.js is required to parse Spine binary files (.skel).")
    if not SPINE_BINARY_TOOL.exists():
        raise FileNotFoundError(f"Missing Spine binary helper: {SPINE_BINARY_TOOL}")

    command = [
        node_executable,
        str(SPINE_BINARY_TOOL),
        "--source",
        str(path),
        "--runtime-dir",
        str(runtime_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        label = source_label or str(path)
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Failed to parse Spine binary '{label}': {stderr}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Spine binary helper returned invalid JSON.") from exc
    if not isinstance(payload, dict) or "bones" not in payload:
        raise ValueError("Spine binary helper did not emit a valid skeleton payload.")
    return payload


@functools.lru_cache(maxsize=1)
def ensure_spine_binary_runtime() -> Path:
    npm_executable = shutil.which("npm")
    node_executable = shutil.which("node")
    if not node_executable or not npm_executable:
        raise RuntimeError("Node.js and npm are required to parse Spine binary files (.skel).")

    runtime_dir = SPINE_BINARY_RUNTIME_DIR
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime_dir / "package.json"
    desired_manifest = json.dumps(SPINE_BINARY_RUNTIME_MANIFEST, indent=2) + "\n"
    current_manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else None
    manifest_changed = current_manifest != desired_manifest
    if manifest_changed:
        manifest_path.write_text(desired_manifest, encoding="utf-8")

    if manifest_changed or not _runtime_packages_installed(runtime_dir):
        completed = subprocess.run(
            [
                npm_executable,
                "install",
                "--no-audit",
                "--no-fund",
                "--silent",
            ],
            cwd=runtime_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError("Failed to install Spine binary runtime dependencies: " + stderr)
    return runtime_dir


def _runtime_packages_installed(runtime_dir: Path) -> bool:
    required = [
        runtime_dir / "node_modules" / "@pixi-spine" / "runtime-3.8",
        runtime_dir / "node_modules" / "@pixi-spine" / "runtime-4.0",
        runtime_dir / "node_modules" / "@pixi-spine" / "runtime-4.1",
        runtime_dir / "node_modules" / "@esotericsoftware" / "spine-core",
    ]
    return all(path.exists() for path in required)


def _compute_depth(name: str, bones_by_name: dict[str, SpineBoneRecord]) -> int:
    depth = 0
    current = bones_by_name[name]
    visited = {name}
    while current.parent and current.parent in bones_by_name and current.parent not in visited:
        visited.add(current.parent)
        current = bones_by_name[current.parent]
        depth += 1
    return depth


def _count_descendants(name: str, bones_by_name: dict[str, SpineBoneRecord]) -> int:
    total = 0
    stack = list(bones_by_name[name].children)
    while stack:
        child_name = stack.pop()
        total += 1
        stack.extend(bones_by_name[child_name].children)
    return total


def _build_basis_2d(
    rotation_deg: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    radians = math.radians(rotation_deg)
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)
    return (
        (cos_value * scale_x, -sin_value * scale_y),
        (sin_value * scale_x, cos_value * scale_y),
    )


def _multiply_basis_2d(
    lhs: tuple[tuple[float, float], tuple[float, float]],
    rhs: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (
            lhs[0][0] * rhs[0][0] + lhs[0][1] * rhs[1][0],
            lhs[0][0] * rhs[0][1] + lhs[0][1] * rhs[1][1],
        ),
        (
            lhs[1][0] * rhs[0][0] + lhs[1][1] * rhs[1][0],
            lhs[1][0] * rhs[0][1] + lhs[1][1] * rhs[1][1],
        ),
    )


def _transform_point(
    basis: tuple[tuple[float, float], tuple[float, float]],
    x: float,
    y: float,
) -> tuple[float, float]:
    return (
        basis[0][0] * x + basis[0][1] * y,
        basis[1][0] * x + basis[1][1] * y,
    )


def _normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle
