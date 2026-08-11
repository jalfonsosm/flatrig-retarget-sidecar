"""Carry a Gaussian-splat companion cloud into the extracted scene's space.

TripoSplat writes a ``.ply`` Gaussian cloud next to the mesh it reconstructs.
The cloud is expressed in the *source file's* frame -- i.e. Blender world space
at import time, BEFORE ``normalize_model_orientation`` turns the rig to face
canonical -Y, and before any setup/bind pose is applied.

Everything the native pipeline consumes (``vertices_3d``, ``bones_3d``, the
projection frame) is in *normalized world space, at the setup pose*. So the
cloud has to make the same two jumps before anything can project it:

1. the normalization rotation the importer applied to the objects, and
2. linear blend skinning from the rest pose to the setup pose, using the
   weights of the nearest rest-pose mesh vertex.

Both are done in world space, which is the only frame the splat file, the mesh
payload and the bone payload share: mesh *local* space is a scaled, rotated
frame of its own (the FBX object transform), and pose matrices live in armature
space, so mixing any of them silently produces garbage.
"""

import json
import math
import os

import numpy as np

try:
    import bpy
    import mathutils
except ImportError:  # pragma: no cover - import-time guard for non-Blender use
    bpy = None
    mathutils = None

# 3DGS PLY layout written by TripoSplat: x y z nx ny nz f_dc[3] opacity
# scale[3] rot[4] -- 17 float32 per point, rot stored WXYZ.
_PLY_STRIDE = 17
_ROT = slice(13, 17)

# Skinning influences kept per splat. Blender vertices routinely carry more
# groups than this with negligible tail weights; four matches the LBS budget
# every downstream runtime (Spine included) uses.
_MAX_INFLUENCES = 4

# Rows per vectorized chunk. The per-splat matrix gather is the memory peak
# (chunk * influences * 16 floats), so cap it instead of materializing all of
# it for a 260k-point cloud.
_CHUNK = 65536


def _write_skin_table(path, bone_names, indices, weights):
    """Write the per-splat skinning table the sprite renderer poses the cloud with.

    Binary, because a 262k-point cloud with four influences is a million index
    and weight pairs: as JSON that is tens of megabytes to write and parse for
    every render.

    Layout (little-endian, the only byte order this pipeline runs on)::

        "FRSKIN1\\n"                     8 bytes
        uint32 splat_count, bone_count, influences
        bone_count x (uint32 length, utf-8 name)
        uint16 indices[splat_count * influences]
        float32 weights[splat_count * influences]

    ``bone_names`` is the authoritative slot order: whoever renders this cloud
    must supply one pose matrix per entry, in this order. Slot 0 is the identity
    fallback carried by ``_vertex_influence_tables`` for splats with no usable
    weight, and it is named so the contract cannot be mistaken for a real bone.
    """
    influences = indices.shape[1]
    packed_indices = np.ascontiguousarray(indices, dtype=np.uint16)
    packed_weights = np.ascontiguousarray(weights, dtype=np.float32)
    with open(path, "wb") as stream:
        stream.write(b"FRSKIN1\n")
        stream.write(
            np.asarray(
                [len(indices), len(bone_names), influences], dtype="<u4"
            ).tobytes()
        )
        for name in bone_names:
            encoded = name.encode("utf-8")
            stream.write(np.asarray([len(encoded)], dtype="<u4").tobytes())
            stream.write(encoded)
        stream.write(packed_indices.tobytes())
        stream.write(packed_weights.tobytes())
    return path


def _read_ply(path):
    """Return ``(header_lines, points)`` for a binary little-endian 3DGS PLY."""
    with open(path, "rb") as stream:
        header = []
        while True:
            line = stream.readline().decode("utf-8")
            if not line:
                raise ValueError(f"Truncated PLY header in {path}")
            header.append(line)
            if line.strip() == "end_header":
                break
        data = stream.read()

    points = np.frombuffer(data, dtype=np.float32)
    if points.size % _PLY_STRIDE:
        raise ValueError(
            f"{path} is not a {_PLY_STRIDE}-float 3DGS cloud "
            f"({points.size} floats is not a multiple of {_PLY_STRIDE})"
        )
    return header, points.reshape(-1, _PLY_STRIDE).copy()


def _write_ply(path, header, points):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as stream:
        for line in header:
            if line.startswith("element vertex"):
                line = f"element vertex {len(points)}\n"
            stream.write(line.encode("utf-8"))
        stream.write(np.ascontiguousarray(points, dtype=np.float32).tobytes())


def _matrix_to_numpy(matrix):
    return np.array([[float(value) for value in row] for row in matrix], dtype=np.float64)


def _transform_points(points, matrix):
    """Apply a 4x4 to an (N, 3) array of positions."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _quaternion_multiply(left, right):
    """Hamilton product of two (N, 4) WXYZ quaternion arrays (broadcasting)."""
    lw, lx, ly, lz = left[..., 0], left[..., 1], left[..., 2], left[..., 3]
    rw, rx, ry, rz = right[..., 0], right[..., 1], right[..., 2], right[..., 3]
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _rotation_to_quaternion(rotations):
    """Convert an (N, 3, 3) array of rotation matrices to WXYZ quaternions.

    Branchy per-element formulas do not vectorize, so this evaluates all four
    trace variants and picks the numerically strongest one per matrix -- the
    standard Shepperd selection.
    """
    m00, m01, m02 = rotations[:, 0, 0], rotations[:, 0, 1], rotations[:, 0, 2]
    m10, m11, m12 = rotations[:, 1, 0], rotations[:, 1, 1], rotations[:, 1, 2]
    m20, m21, m22 = rotations[:, 2, 0], rotations[:, 2, 1], rotations[:, 2, 2]

    candidates = np.stack(
        (
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ),
        axis=1,
    )
    branch = np.argmax(candidates, axis=1)
    scale = 0.5 / np.sqrt(np.maximum(candidates[np.arange(len(branch)), branch], 1e-12))

    quaternions = np.empty((len(rotations), 4), dtype=np.float64)
    for index, components in enumerate(
        (
            lambda: (0.25 / scale, (m21 - m12) * scale, (m02 - m20) * scale, (m10 - m01) * scale),
            lambda: ((m21 - m12) * scale, 0.25 / scale, (m01 + m10) * scale, (m02 + m20) * scale),
            lambda: ((m02 - m20) * scale, (m01 + m10) * scale, 0.25 / scale, (m12 + m21) * scale),
            lambda: ((m10 - m01) * scale, (m02 + m20) * scale, (m12 + m21) * scale, 0.25 / scale),
        )
    ):
        mask = branch == index
        if not mask.any():
            continue
        for column, values in enumerate(components()):
            quaternions[mask, column] = values[mask]

    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions / np.maximum(norms, 1e-12)


def _orthonormalize(matrices):
    """Nearest rotation to each (N, 3, 3) linear part (polar decomposition).

    A blend of rigid bone matrices is not itself rigid, so the Gaussian's new
    orientation has to come from the rotation factor rather than the raw sum --
    otherwise the shear the blend introduces at every joint leaks into the
    splat ellipsoids.
    """
    u, _, vt = np.linalg.svd(matrices)
    rotations = u @ vt
    # Guard against the reflection SVD can return for degenerate blends.
    flipped = np.linalg.det(rotations) < 0.0
    if flipped.any():
        u[flipped, :, -1] *= -1.0
        rotations[flipped] = u[flipped] @ vt[flipped]
    return rotations


def _rest_pose_reference(mesh_obj, armature_obj):
    """Rest-pose world vertices plus their per-bone weights.

    The splat cloud was generated against the un-posed model, so the nearest
    neighbour search has to run against the rest shape even when the scene is
    currently sitting on a setup/donor frame.
    """
    previous_position = None
    if armature_obj is not None:
        previous_position = armature_obj.data.pose_position
        armature_obj.data.pose_position = "REST"
        bpy.context.view_layer.update()
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = mesh_obj.evaluated_get(depsgraph)
        rest_mesh = evaluated.to_mesh()
        try:
            local = np.empty((len(rest_mesh.vertices), 3), dtype=np.float64)
            rest_mesh.vertices.foreach_get("co", local.reshape(-1))
            groups = [[(g.group, g.weight) for g in v.groups] for v in rest_mesh.vertices]
        finally:
            evaluated.to_mesh_clear()
    finally:
        if previous_position is not None:
            armature_obj.data.pose_position = previous_position
            bpy.context.view_layer.update()

    world = _transform_points(local, _matrix_to_numpy(mesh_obj.matrix_world))
    return world, groups


def _vertex_influence_tables(groups, group_index_to_bone, bone_index_by_name):
    """Pack per-vertex weights into dense (V, _MAX_INFLUENCES) index/weight arrays.

    Bone slot 0 is the identity matrix, so a vertex with no usable weight gets
    full influence from it and leaves the splats that pick it exactly where they
    are. Leaving the row at zero instead would blend to a zero matrix and
    collapse every one of those splats onto the origin.
    """
    count = len(groups)
    indices = np.zeros((count, _MAX_INFLUENCES), dtype=np.int32)
    weights = np.zeros((count, _MAX_INFLUENCES), dtype=np.float64)
    weights[:, 0] = 1.0

    for vertex_index, entries in enumerate(groups):
        influences = []
        for group_index, weight in entries:
            if weight <= 0.0:
                continue
            bone_name = group_index_to_bone.get(group_index)
            bone_index = bone_index_by_name.get(bone_name)
            if bone_index is None:
                continue
            influences.append((weight, bone_index))
        if not influences:
            continue
        influences.sort(reverse=True)
        influences = influences[:_MAX_INFLUENCES]
        total = sum(weight for weight, _ in influences)
        if total <= 0.0:
            continue
        weights[vertex_index] = 0.0
        for slot, (weight, bone_index) in enumerate(influences):
            indices[vertex_index, slot] = bone_index
            weights[vertex_index, slot] = weight / total
    return indices, weights


def _world_skinning_matrices(armature_obj):
    """Per-bone world-space skinning matrices, plus the bone-name index map.

    Pose and rest bone matrices are both in armature space, so the armature's
    own world transform has to bracket the pair before the result can be
    applied to world-space splat positions.
    """
    to_world = _matrix_to_numpy(armature_obj.matrix_world)
    to_local = _matrix_to_numpy(armature_obj.matrix_world.inverted())

    names = []
    matrices = [np.eye(4)]  # slot 0 is the identity fallback for unweighted splats
    for pose_bone in armature_obj.pose.bones:
        skin = _matrix_to_numpy(pose_bone.matrix @ pose_bone.bone.matrix_local.inverted())
        matrices.append(to_world @ skin @ to_local)
        names.append(pose_bone.name)
    bone_index_by_name = {name: index + 1 for index, name in enumerate(names)}
    return np.stack(matrices), bone_index_by_name, names


def _nearest_rest_vertices(rest_world, query_points):
    """Index of the closest rest vertex for each query point."""
    tree = mathutils.kdtree.KDTree(len(rest_world))
    for index, position in enumerate(rest_world):
        tree.insert((float(position[0]), float(position[1]), float(position[2])), index)
    tree.balance()

    nearest = np.empty(len(query_points), dtype=np.int32)
    find = tree.find
    for index, point in enumerate(query_points):
        nearest[index] = find((float(point[0]), float(point[1]), float(point[2])))[1]
    return nearest


def carry_splat_to_world(splat_path, output_path, *, normalization_matrix=None):
    """Re-express a cloud in normalized world space, without skinning it.

    Used by the steps that re-export the model itself (the canonical rig
    reduction): they change the frame the model is stored in, so the cloud
    beside it has to move with it or the next consumer inherits a cloud and a
    mesh that no longer describe the same object.
    """
    return _write_carried_splat(splat_path, output_path, normalization_matrix, None, None)


def deform_splat_to_setup_pose(
    splat_path,
    output_path,
    mesh_obj,
    armature_obj,
    *,
    normalization_matrix=None,
):
    """Write ``splat_path`` into normalized world space at the current pose.

    Must be called with the scene already sitting on the setup frame and pose
    the mesh payload is extracted from, so the cloud and the mesh describe the
    same instant. Returns a summary dict, or ``None`` when there is no cloud.
    """
    return _write_carried_splat(splat_path, output_path, normalization_matrix, mesh_obj, armature_obj)


def _write_carried_splat(splat_path, output_path, normalization_matrix, mesh_obj, armature_obj):
    if not splat_path or not os.path.exists(splat_path):
        return None

    header, points = _read_ply(splat_path)
    print(f"[splat] carrying {len(points)} gaussians from {splat_path}")

    normalization = (
        np.eye(4)
        if normalization_matrix is None
        else _matrix_to_numpy(normalization_matrix)
    )
    world_positions = _transform_points(points[:, 0:3].astype(np.float64), normalization)
    world_rotations = _quaternion_multiply(
        _rotation_to_quaternion(normalization[np.newaxis, :3, :3])[0],
        points[:, _ROT].astype(np.float64),
    )

    dominant_bones = {}
    skin_table = None
    if mesh_obj is not None and armature_obj is not None and mesh_obj.vertex_groups:
        rest_world, rest_groups = _rest_pose_reference(mesh_obj, armature_obj)
        bone_matrices, bone_index_by_name, bone_names = _world_skinning_matrices(armature_obj)
        group_index_to_bone = {group.index: group.name for group in mesh_obj.vertex_groups}
        vertex_bones, vertex_weights = _vertex_influence_tables(
            rest_groups, group_index_to_bone, bone_index_by_name
        )

        nearest = _nearest_rest_vertices(rest_world, world_positions)
        splat_bones = vertex_bones[nearest]
        splat_weights = vertex_weights[nearest]

        for start in range(0, len(points), _CHUNK):
            stop = min(start + _CHUNK, len(points))
            blended = np.einsum(
                "nk,nkij->nij",
                splat_weights[start:stop],
                bone_matrices[splat_bones[start:stop]],
            )
            world_positions[start:stop] = np.einsum(
                "nij,nj->ni", blended[:, :3, :3], world_positions[start:stop]
            ) + blended[:, :3, 3]
            world_rotations[start:stop] = _quaternion_multiply(
                _rotation_to_quaternion(_orthonormalize(blended[:, :3, :3])),
                world_rotations[start:stop],
            )

        # Dominant bone per splat: the only per-splat provenance downstream
        # segmentation has today. Bone index 0 is the identity slot, not a bone,
        # so splats that landed on an unweighted vertex are left out entirely.
        rows = np.arange(len(points))
        dominant_indices = splat_bones[rows, np.argmax(splat_weights, axis=1)]
        weighted = dominant_indices > 0
        dominant_bones = {
            str(int(index)): bone_names[int(bone) - 1]
            for index, bone in zip(rows[weighted], dominant_indices[weighted])
        }

        # The full influence table, kept rather than reduced to the dominant
        # bone: the sprite renderer re-poses this cloud for every animation
        # frame, and a one-bone approximation would tear every joint it crosses.
        skin_table = _write_skin_table(
            os.path.splitext(output_path)[0] + ".skin",
            ["__identity__", *bone_names],
            splat_bones,
            splat_weights,
        )

    points[:, 0:3] = world_positions
    points[:, _ROT] = world_rotations
    _write_ply(output_path, header, points)

    weights_path = os.path.splitext(output_path)[0] + "_weights.json"
    with open(weights_path, "w", encoding="utf-8") as stream:
        json.dump(dominant_bones, stream)

    print(f"[splat] wrote {output_path} ({len(dominant_bones)} skinned)")
    return {
        "input": splat_path,
        "output": output_path,
        "weights": weights_path,
        "skin": skin_table,
        "count": int(len(points)),
        "skinned": int(len(dominant_bones)),
        "normalization_yaw_deg": round(
            math.degrees(math.atan2(normalization[1, 0], normalization[0, 0])), 4
        ),
    }
