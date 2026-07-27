from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import blender_scene_io as scene_io  # noqa: E402


def _mesh_object(name, vertices, faces):
    mesh = bpy.data.meshes.new(f"{name}_data")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def test_surface_data_copy_uses_identical_loop_topology() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    faces = [(0, 1, 2), (0, 2, 3)]
    source = _mesh_object("source", vertices, faces)
    target = _mesh_object("target", vertices, faces)

    source_uv = source.data.uv_layers.new(name="source_uv")
    expected_uvs = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
    for loop_uv, expected in zip(source_uv.data, expected_uvs):
        loop_uv.uv = expected
    target.data.uv_layers.new(name="stale")

    first_material = bpy.data.materials.new("first")
    second_material = bpy.data.materials.new("second")
    source.data.materials.append(first_material)
    source.data.materials.append(second_material)
    source.data.polygons[0].material_index = 1
    source.data.polygons[1].material_index = 0

    report = scene_io._copy_mesh_surface_data_by_topology(target, source)

    assert report == {"uv_layer_count": 1, "material_count": 2}
    assert list(target.data.uv_layers.keys()) == ["source_uv"]
    assert [tuple(value.uv) for value in target.data.uv_layers[0].data] == pytest.approx(
        expected_uvs
    )
    assert list(target.data.materials) == [first_material, second_material]
    assert [polygon.material_index for polygon in target.data.polygons] == [1, 0]


def test_surface_data_copy_rejects_different_loop_vertex_order() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    source = _mesh_object("source", vertices, [(0, 1, 2)])
    target = _mesh_object("target", vertices, [(0, 2, 1)])

    with pytest.raises(ValueError, match="references a different vertex"):
        scene_io._copy_mesh_surface_data_by_topology(target, source)


def test_similarity_fit_recovers_proper_uniform_transform() -> None:
    source = np.asarray(
        [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        ],
        dtype=np.float64,
    )
    x_angle, y_angle, z_angle = map(math.radians, (37.0, -18.0, 11.0))
    rotate_x = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(x_angle), -math.sin(x_angle)),
            (0.0, math.sin(x_angle), math.cos(x_angle)),
        )
    )
    rotate_y = np.asarray(
        (
            (math.cos(y_angle), 0.0, math.sin(y_angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(y_angle), 0.0, math.cos(y_angle)),
        )
    )
    rotate_z = np.asarray(
        (
            (math.cos(z_angle), -math.sin(z_angle), 0.0),
            (math.sin(z_angle), math.cos(z_angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    rotation = rotate_z @ rotate_y @ rotate_x
    scale = 2.75
    translation = np.asarray((4.0, -3.0, 1.25))
    target = (scale * (rotation @ source.T)).T + translation

    result = scene_io._fit_orientation_preserving_similarity(source, target)
    matrix = result["matrix"]
    mapped = (matrix[:3, :3] @ source.T).T + matrix[:3, 3]

    assert mapped == pytest.approx(target, abs=1e-10)
    assert result["scale"] == pytest.approx(scale, abs=1e-10)
    assert result["rotation_determinant"] == pytest.approx(1.0, abs=1e-10)
    assert result["relative_max_error"] < 1e-10


def test_similarity_fit_rejects_reflection() -> None:
    source = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 3.0),
            (1.0, 2.0, 3.0),
        ]
    )
    reflected = source.copy()
    reflected[:, 0] *= -1.0

    with pytest.raises(ValueError, match="orientation-preserving uniform similarity"):
        scene_io._fit_orientation_preserving_similarity(source, reflected)


def test_parent_similarity_is_inherited_by_mesh_once() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    predicted = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
    )
    rotation = np.asarray(mathutils.Matrix.Rotation(math.radians(90.0), 3, "X"))
    expected_world = (0.5 * (rotation @ predicted.T)).T + np.asarray((2.0, 3.0, 4.0))
    alignment = scene_io._fit_orientation_preserving_similarity(predicted, expected_world)

    mesh_obj = _mesh_object("predicted", predicted, [(0, 1, 2), (0, 3, 1)])
    armature_data = bpy.data.armatures.new("armature_data")
    armature_obj = bpy.data.objects.new("armature", armature_data)
    bpy.context.collection.objects.link(armature_obj)
    mesh_obj.parent = armature_obj
    armature_obj.matrix_world = mathutils.Matrix(alignment["matrix"].tolist())
    bpy.context.view_layer.update()

    actual_world = np.asarray(
        [tuple(mesh_obj.matrix_world @ vertex.co) for vertex in mesh_obj.data.vertices]
    )
    # mathutils matrices store float32 values even though the fitted matrix is
    # float64, so the object-level round trip is expected to lose a few ulps.
    assert actual_world == pytest.approx(expected_world, abs=1e-6)
    assert np.asarray(mesh_obj.matrix_local) == pytest.approx(np.eye(4), abs=1e-6)


def test_bake_predicted_rig_copies_surface_and_aligns_only_parent(
    monkeypatch, tmp_path
) -> None:
    predicted = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    triangles = np.asarray(
        [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)], dtype=np.uint32
    )
    rotation = np.asarray(mathutils.Matrix.Rotation(math.radians(90.0), 3, "X"))
    source_world = (0.5 * (rotation @ predicted.T)).T + np.asarray((2.0, 3.0, 4.0))
    expected_uvs = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
    ]

    prediction_path = tmp_path / "prediction.npz"
    np.savez(
        prediction_path,
        vertices=predicted,
        triangles=triangles,
        bone_names=np.asarray(["root"]),
        parent_indices=np.asarray([-1], dtype=np.int32),
        heads=np.asarray([(0.0, 0.0, 0.0)], dtype=np.float32),
        tails=np.asarray([(0.0, 1.0, 0.0)], dtype=np.float32),
        joints_top4=np.zeros((len(predicted), 4), dtype=np.uint8),
        weights_top4=np.asarray([(1.0, 0.0, 0.0, 0.0)] * len(predicted), dtype=np.float32),
    )
    original_mesh_path = tmp_path / "original.glb"
    original_mesh_path.touch()

    def import_original_mesh(_path) -> None:
        source_obj = _mesh_object("source", source_world, triangles)
        source_uv = source_obj.data.uv_layers.new(name="UVMap")
        for loop_uv, expected in zip(source_uv.data, expected_uvs):
            loop_uv.uv = expected
        source_obj.data.materials.append(bpy.data.materials.new("surface"))

    monkeypatch.setattr(scene_io, "import_model", import_original_mesh)
    output_path = tmp_path / "rigged.fbx"

    report = scene_io.bake_predicted_rig(
        str(prediction_path),
        fbx_output=str(output_path),
        mesh_path=str(original_mesh_path),
    )

    mesh_obj = next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")
    armature_obj = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    actual_world = np.asarray(
        [tuple(mesh_obj.matrix_world @ vertex.co) for vertex in mesh_obj.data.vertices]
    )
    actual_uvs = [tuple(value.uv) for value in mesh_obj.data.uv_layers[0].data]
    expected_parent_matrix = np.eye(4)
    expected_parent_matrix[:3, :3] = 0.5 * rotation
    expected_parent_matrix[:3, 3] = (2.0, 3.0, 4.0)

    assert output_path.is_file()
    assert report["surface_transfer"] == {"uv_layer_count": 1, "material_count": 1}
    assert report["source_alignment"]["scale"] == pytest.approx(0.5, abs=1e-6)
    assert actual_world == pytest.approx(source_world, abs=1e-6)
    assert actual_uvs == pytest.approx(expected_uvs)
    assert mesh_obj.parent is armature_obj
    assert np.asarray(mesh_obj.matrix_local) == pytest.approx(np.eye(4), abs=1e-6)
    assert np.asarray(armature_obj.matrix_world) == pytest.approx(
        expected_parent_matrix, abs=1e-6
    )
