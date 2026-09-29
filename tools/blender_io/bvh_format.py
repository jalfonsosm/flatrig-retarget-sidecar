"""Pure BVH formatting helpers for 3D retarget exchange."""

from __future__ import annotations

from pathlib import Path


def _build_joint_children(joints):
    children = {int(joint["index"]): [] for joint in joints}
    for joint in joints:
        parent_index = int(joint.get("parent_index", -1))
        if parent_index >= 0:
            children.setdefault(parent_index, []).append(int(joint["index"]))
    return children


def _write_3d_joint_hierarchy(lines, joints, joint_children, joint_index, depth):
    joint = joints[joint_index]
    indent = "\t" * depth
    label = "ROOT" if int(joint.get("parent_index", -1)) < 0 else "JOINT"
    lines.append(f"{indent}{label} {joint['bvh_name']}")
    lines.append(f"{indent}{{")
    channel_indent = f"{indent}\t"
    offset = joint.get("offset") or [0.0, 0.0, 0.0]
    lines.append(f"{channel_indent}OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
    lines.append(
        f"{channel_indent}CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
    )

    children = joint_children.get(joint_index) or []
    if children:
        for child_index in children:
            _write_3d_joint_hierarchy(lines, joints, joint_children, child_index, depth + 1)
    else:
        tail_offset = joint.get("tail_offset") or [1.0, 0.0, 0.0]
        lines.append(f"{channel_indent}End Site")
        lines.append(f"{channel_indent}{{")
        lines.append(
            f"{channel_indent}\tOFFSET {tail_offset[0]:.6f} {tail_offset[1]:.6f} {tail_offset[2]:.6f}"
        )
        lines.append(f"{channel_indent}}}")

    lines.append(f"{indent}}}")


def _write_3d_bvh(output_path, joints, positions, rotations, fps):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not joints:
        raise ValueError("Cannot write BVH without joints.")

    joint_children = _build_joint_children(joints)
    lines = ["HIERARCHY"]
    _write_3d_joint_hierarchy(lines, joints, joint_children, 0, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {len(positions)}")
    lines.append(f"Frame Time: {1.0 / fps:.8f}")

    for frame_positions, frame_rotations in zip(positions, rotations, strict=True):
        values = []
        cursor = 0
        for _joint in joints:
            position_triplet = frame_positions[cursor : cursor + 3]
            rotation_triplet = frame_rotations[cursor : cursor + 3]
            cursor += 3
            values.extend(f"{value:.6f}" for value in position_triplet)
            values.extend(f"{value:.6f}" for value in rotation_triplet)
        lines.append(" ".join(values))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rest_3d_bvh_frames(layout, frame_count=2):
    frame_count = max(2, int(frame_count or 2))
    frame_positions = []
    frame_rotations = []
    for _joint in layout["joints"]:
        frame_positions.extend((0.0, 0.0, 0.0))
        frame_rotations.extend((0.0, 0.0, 0.0))
    return (
        [list(frame_positions) for _ in range(frame_count)],
        [list(frame_rotations) for _ in range(frame_count)],
    )
