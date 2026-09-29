import sys
from pathlib import Path
from types import SimpleNamespace

import bpy  # noqa: F401
import mathutils

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import blender_scene_io as scene_io  # noqa: E402
import blender_orientation as orientation  # noqa: E402


class _Bone:
    def __init__(self, name, matrix_local, *, length=1.0, parent=None):
        self.name = name
        self.matrix_local = matrix_local
        self.length = length
        self.parent = parent
        self.children = []
        self.head_local = matrix_local.translation
        if parent is not None:
            parent.children.append(self)


def _armature(*bones):
    return SimpleNamespace(
        data=SimpleNamespace(bones=list(bones)),
        matrix_world=mathutils.Matrix.Identity(4),
    )


def test_structural_root_uses_topology_not_bone_names():
    small_root = _Bone("hips_by_name_only", mathutils.Matrix.Identity(4), length=10.0)
    large_root = _Bone("node_17", mathutils.Matrix.Identity(4), length=1.0)
    child = _Bone("branch_a", mathutils.Matrix.Identity(4), parent=large_root)
    _Bone("branch_b", mathutils.Matrix.Identity(4), parent=child)

    selected = scene_io._structural_root_bone(
        _armature(small_root, large_root, child, *child.children)
    )

    assert selected is large_root


def test_vertical_root_uses_roll_axis_as_forward_without_anatomical_names():
    # Columns: local Y -> world +Z (upright), local Z -> world +X (forward).
    matrix = mathutils.Matrix(
        (
            (0.0, 0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    root = _Bone("root_0042", matrix)

    forward = scene_io._rig_forward_world(_armature(root))

    assert tuple(round(value, 6) for value in forward) == (1.0, 0.0, 0.0)


def test_vertical_root_uses_opposing_child_branches_without_anatomical_names():
    # local Y -> world +Z (upright), local Z -> world +Y (forward sign hint).
    matrix = mathutils.Matrix(
        (
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    root = _Bone("node_0", matrix)
    left_branch = _Bone(
        "node_1",
        mathutils.Matrix.Translation((1.0, 0.0, -1.0)),
        parent=root,
    )
    right_branch = _Bone(
        "node_2",
        mathutils.Matrix.Translation((-1.0, 0.0, -1.0)),
        parent=root,
    )

    forward = scene_io._rig_forward_world(_armature(root, left_branch, right_branch))

    assert tuple(round(value, 6) for value in forward) == (0.0, 1.0, 0.0)


def test_horizontal_root_uses_longitudinal_axis_for_chain_rigs():
    root = _Bone("spine_root_without_semantics", mathutils.Matrix.Identity(4))

    forward = scene_io._rig_forward_world(_armature(root))

    assert tuple(round(value, 6) for value in forward) == (0.0, 1.0, 0.0)


def test_terminal_bone_limit_rejects_unit_scale_outliers():
    limit = scene_io._terminal_bone_length_limit(
        1.4,
        [0.08, 0.1, 0.12, 0.15, 4.5, 12.2, 27.6],
    )

    assert limit == 3.0


class _FakeMesh:
    def __init__(self, points):
        vectors = [mathutils.Vector(point) for point in points]
        self.matrix_world = mathutils.Matrix.Identity(4)
        self.data = SimpleNamespace(
            vertices=[SimpleNamespace(co=vector) for vector in vectors]
        )
        low = mathutils.Vector(
            tuple(min(vector[axis] for vector in vectors) for axis in range(3))
        )
        high = mathutils.Vector(
            tuple(max(vector[axis] for vector in vectors) for axis in range(3))
        )
        self.bound_box = [
            (low.x if x else high.x, low.y if y else high.y, low.z if z else high.z)
            for x in (1, 0)
            for y in (1, 0)
            for z in (1, 0)
        ]


def _box_cloud(size, steps=6):
    half = [value / 2.0 for value in size]
    return [
        (
            -half[0] + 2 * half[0] * i / (steps - 1),
            -half[1] + 2 * half[1] * j / (steps - 1),
            -half[2] + 2 * half[2] * k / (steps - 1),
        )
        for i in range(steps)
        for j in range(steps)
        for k in range(steps)
    ]


def test_long_bodied_mesh_replaces_an_adaptive_root_narrow_axis_with_body_axis():
    mesh = _FakeMesh(_box_cloud((0.6, 4.0, 1.8)))

    result = orientation._disambiguate_forward_with_mesh(
        mathutils.Vector((1.0, 0.0, 0.0)), [mesh]
    )

    assert round(abs(result.y), 6) == 1.0
    assert round(result.x, 6) == 0.0


def test_upright_mesh_keeps_the_existing_narrow_axis_correction():
    mesh = _FakeMesh(_box_cloud((1.8, 0.4, 1.8)))

    result = orientation._disambiguate_forward_with_mesh(
        mathutils.Vector((1.0, 0.0, 0.0)), [mesh]
    )

    assert round(abs(result.y), 6) == 1.0
    assert round(result.x, 6) == 0.0
