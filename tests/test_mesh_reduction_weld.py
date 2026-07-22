import importlib
import sys
from pathlib import Path

import bpy

bmesh = importlib.import_module("bmesh")

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))
import blender_scene_io as scene_io  # noqa: E402


def _split_triangles(name: str, points, source_faces):
    vertices = []
    faces = []
    for source_face in source_faces:
        start = len(vertices)
        vertices.extend(points[index] for index in source_face)
        faces.append((start, start + 1, start + 2))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for index, loop in enumerate(uv_layer.data):
        loop.uv = ((index % 3) / 2.0, (index // 3) / 3.0)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _split_tetrahedron(name: str):
    points = [
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0),
    ]
    source_faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    return _split_triangles(name, points, source_faces)


def _split_octahedron(name: str):
    points = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    source_faces = [
        (4, 0, 2),
        (4, 2, 1),
        (4, 1, 3),
        (4, 3, 0),
        (5, 2, 0),
        (5, 1, 2),
        (5, 3, 1),
        (5, 0, 3),
    ]
    return _split_triangles(name, points, source_faces)


def _boundary_edge_count(mesh_obj) -> int:
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh_obj.data)
        return sum(1 for edge in bm.edges if len(edge.link_faces) == 1)
    finally:
        bm.free()


def _assert_closed_manifold(mesh_obj) -> None:
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh_obj.data)
        assert all(len(edge.link_faces) == 2 for edge in bm.edges)
        signatures = [tuple(sorted(vertex.index for vertex in face.verts)) for face in bm.faces]
        assert len(signatures) == len(set(signatures))
        assert all(face.calc_area() > 0.0 for face in bm.faces)
    finally:
        bm.free()


def setup_function():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def test_external_weld_merges_exact_duplicates_but_not_nearby_vertices():
    mesh = bpy.data.meshes.new("exact_weld_mesh")
    mesh.from_pydata([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (5e-7, 0.0, 0.0)], [], [])
    mesh.update()
    mesh_obj = bpy.data.objects.new("exact_weld", mesh)
    bpy.context.scene.collection.objects.link(mesh_obj)

    removed = scene_io._weld_exact_position_duplicates(mesh_obj)

    assert removed == 1
    assert len(mesh_obj.data.vertices) == 2
    positions = sorted(tuple(vertex.co) for vertex in mesh_obj.data.vertices)
    assert positions[0] == (0.0, 0.0, 0.0)
    assert abs(positions[1][0] - 5e-7) < 1e-12


def test_reduction_welds_uv_seams_before_budget_check():
    mesh_obj = _split_tetrahedron("split_tetra")
    assert len(mesh_obj.data.vertices) == 12
    assert _boundary_edge_count(mesh_obj) == 12
    original_uvs = [tuple(loop.uv) for loop in mesh_obj.data.uv_layers[0].data]

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)

    assert len(mesh_obj.data.vertices) == 4
    assert len(mesh_obj.data.loop_triangles) == 4
    assert _boundary_edge_count(mesh_obj) == 0
    assert [tuple(loop.uv) for loop in mesh_obj.data.uv_layers[0].data] == original_uvs
    assert report["source_vertex_count"] == 12
    assert report["welded_source_vertex_count"] == 4
    assert report["seam_vertices_welded"] == 8
    assert report["applied"] is True
    assert report["decimation_applied"] is False
    assert report["seam_weld_safe"] is True
    assert report["reason"] == "source_under_target_after_weld"


def test_disabled_reduction_keeps_split_topology_byte_for_byte():
    mesh_obj = _split_tetrahedron("disabled_split_tetra")

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=False)

    assert len(mesh_obj.data.vertices) == 12
    assert _boundary_edge_count(mesh_obj) == 12
    assert report["applied"] is False
    assert report["seam_vertices_welded"] == 0
    assert report["reason"] == "disabled"


def test_under_budget_reduction_does_not_weld_authored_coincident_surfaces():
    mesh_obj = _split_tetrahedron("under_budget_split_tetra")

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=12, enabled=True)

    assert len(mesh_obj.data.vertices) == 12
    assert _boundary_edge_count(mesh_obj) == 12
    assert report["applied"] is False
    assert report["seam_vertices_welded"] == 0
    assert report["reason"] == "source_under_target"


def test_reduction_skips_unsafe_weld_when_duplicate_weights_conflict():
    mesh_obj = _split_tetrahedron("weighted_split_tetra")
    duplicate_indices = [
        vertex.index for vertex in mesh_obj.data.vertices if tuple(vertex.co) == (1.0, 1.0, 1.0)
    ]
    group = mesh_obj.vertex_groups.new(name="Bone")
    group.add(duplicate_indices, 1.0, "REPLACE")
    group.add([duplicate_indices[-1]], 0.25, "REPLACE")

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)

    assert len(mesh_obj.data.vertices) == 12
    assert _boundary_edge_count(mesh_obj) == 12
    assert report["applied"] is False
    assert report["reduction_skipped"] is True
    assert report["seam_weld_safe"] is False
    assert report["seam_vertices_welded"] == 0
    assert report["reason"] == "unsafe_position_weld_skipped_reduction"
    assert any(
        issue.startswith("incompatible_vertex_group_weights:")
        for issue in report["seam_weld_issues"]
    )


def test_reduction_skips_unsafe_weld_when_shape_key_values_conflict():
    mesh_obj = _split_tetrahedron("shape_key_split_tetra")
    duplicate_indices = [
        vertex.index for vertex in mesh_obj.data.vertices if tuple(vertex.co) == (1.0, 1.0, 1.0)
    ]
    mesh_obj.shape_key_add(name="Basis")
    expression = mesh_obj.shape_key_add(name="Expression")
    expression.data[duplicate_indices[-1]].co.x += 0.25

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)

    assert len(mesh_obj.data.vertices) == 12
    assert report["reduction_skipped"] is True
    assert any(
        issue.startswith("incompatible_shape_key:Expression:")
        for issue in report["seam_weld_issues"]
    )


def test_reduction_skips_unsafe_weld_when_point_attribute_values_conflict():
    mesh_obj = _split_tetrahedron("attribute_split_tetra")
    duplicate_indices = [
        vertex.index for vertex in mesh_obj.data.vertices if tuple(vertex.co) == (1.0, 1.0, 1.0)
    ]
    attribute = mesh_obj.data.attributes.new("detail_mask", "FLOAT", "POINT")
    attribute.data[duplicate_indices[-1]].value = 1.0

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)

    assert len(mesh_obj.data.vertices) == 12
    assert report["reduction_skipped"] is True
    assert any(
        issue.startswith("incompatible_point_attribute:detail_mask:")
        for issue in report["seam_weld_issues"]
    )


def test_reduction_skips_weld_that_would_create_disconnected_face_fans():
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    ]
    tetra_faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    faces = tetra_faces + [tuple(index + 4 for index in face) for face in tetra_faces]
    mesh = bpy.data.meshes.new("touching_tetrahedra_mesh")
    mesh.from_pydata(points, [], faces)
    mesh.update()
    mesh_obj = bpy.data.objects.new("touching_tetrahedra", mesh)
    bpy.context.scene.collection.objects.link(mesh_obj)
    _assert_closed_manifold(mesh_obj)

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=7, enabled=True)

    assert len(mesh_obj.data.vertices) == 8
    _assert_closed_manifold(mesh_obj)
    assert report["reduction_skipped"] is True
    assert "topology_disconnected_face_fans" in report["seam_weld_issues"]


def test_safe_weld_then_runs_decimate_and_remains_manifold():
    mesh_obj = _split_octahedron("split_octahedron")

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)

    assert report["seam_vertices_welded"] == 18
    assert report["welded_source_vertex_count"] == 6
    assert report["decimation_applied"] is True
    assert report["reason"] == "target_reached"
    assert len(mesh_obj.data.vertices) <= 4
    assert mesh_obj.data.uv_layers.active is not None
    assert len(mesh_obj.data.uv_layers.active.data) == len(mesh_obj.data.loops)
    _assert_closed_manifold(mesh_obj)


def test_decimate_runs_before_existing_armature_modifier_and_preserves_it(capfd):
    mesh_obj = _split_octahedron("armature_stack_octahedron")
    armature_data = bpy.data.armatures.new("existing_armature_data")
    armature_obj = bpy.data.objects.new("existing_armature", armature_data)
    bpy.context.scene.collection.objects.link(armature_obj)
    armature_modifier = mesh_obj.modifiers.new(name="ExistingArmature", type="ARMATURE")
    armature_modifier.object = armature_obj
    assert [modifier.name for modifier in mesh_obj.modifiers] == ["ExistingArmature"]

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)
    captured = capfd.readouterr()

    assert report["decimation_applied"] is True
    assert [modifier.name for modifier in mesh_obj.modifiers] == ["ExistingArmature"]
    assert mesh_obj.modifiers["ExistingArmature"].object is armature_obj
    assert "Applied modifier was not first" not in captured.out
    assert "Applied modifier was not first" not in captured.err
    _assert_closed_manifold(mesh_obj)
