import importlib
import math
import sys
from pathlib import Path

import bpy

bmesh = importlib.import_module("bmesh")

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))
import blender_scene_io as scene_io  # noqa: E402
from blender_io import mesh_reduction as reduction_impl  # noqa: E402


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


def _open_torus_seam(name: str, major_segments: int = 12, minor_segments: int = 6):
    """Create a cylinder-like parameter sheet whose coincident ends form a torus."""
    vertices = []
    for major_index in range(major_segments + 1):
        major_angle = 2.0 * math.pi * major_index / major_segments
        for minor_index in range(minor_segments):
            minor_angle = 2.0 * math.pi * minor_index / minor_segments
            radius = 2.0 + 0.5 * math.cos(minor_angle)
            point = (
                radius * math.cos(major_angle),
                radius * math.sin(major_angle),
                0.5 * math.sin(minor_angle),
            )
            if major_index == major_segments:
                point = vertices[minor_index]
            vertices.append(point)
    faces = []
    for major_index in range(major_segments):
        row = major_index * minor_segments
        next_row = (major_index + 1) * minor_segments
        for minor_index in range(minor_segments):
            next_minor = (minor_index + 1) % minor_segments
            faces.append(
                (
                    row + minor_index,
                    next_row + minor_index,
                    next_row + next_minor,
                    row + next_minor,
                )
            )
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


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


def test_external_weld_rejects_coincident_seam_that_creates_a_handle():
    mesh_obj = _open_torus_seam("open_torus")
    before = reduction_impl.mesh_topology_metrics(mesh_obj)
    assert before["watertight"] is False

    assessment = reduction_impl._assess_exact_position_weld(mesh_obj)

    assert assessment["safe"] is False
    assert "topology_new_handles" in assessment["issues"]


def test_geometric_boundary_metrics_distinguish_paired_seam_from_open_edge():
    split = _split_tetrahedron("paired_boundary_tetra")
    paired = reduction_impl.geometric_boundary_pair_metrics(split)
    assert paired["boundary_edges"] == 12
    assert paired["unpaired_boundary_edges"] == 0
    assert paired["paired_fraction"] == 1.0

    mesh = bpy.data.meshes.new("open_triangle_mesh")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    mesh.update()
    open_obj = bpy.data.objects.new("open_triangle", mesh)
    bpy.context.scene.collection.objects.link(open_obj)
    unpaired = reduction_impl.geometric_boundary_pair_metrics(open_obj)
    assert unpaired["boundary_edges"] == 3
    assert unpaired["unpaired_boundary_edges"] == 3
    assert unpaired["paired_fraction"] == 0.0


def test_geometric_boundary_regression_rejects_new_crack():
    before = {
        "boundary_edges": 12,
        "paired_boundary_edges": 12,
        "unpaired_boundary_edges": 0,
        "paired_fraction": 1.0,
    }
    after = {
        "boundary_edges": 8,
        "paired_boundary_edges": 6,
        "unpaired_boundary_edges": 2,
        "paired_fraction": 0.75,
    }

    issues = reduction_impl.geometric_boundary_regression_issues(before, after)
    assert "topology_new_geometric_seam_gaps" in issues


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


def test_under_budget_invalid_mesh_is_not_reported_as_topology_validated():
    mesh = bpy.data.meshes.new("invalid_no_faces_mesh")
    mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
    mesh.update()
    mesh_obj = bpy.data.objects.new("invalid_no_faces", mesh)
    bpy.context.scene.collection.objects.link(mesh_obj)

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)

    assert report["reason"] == "source_under_target"
    assert report["mesh_valid_after"] is False
    assert report["topology_validated"] is False
    assert "topology_invalid_output" in report["topology_validation_issues"]


def test_reduction_contract_rejects_explicit_failure_but_allows_open_valid_mesh():
    report = {
        "reduced_vertex_count": 100,
        "budget_satisfied": True,
        "topology_validated": True,
        "mesh_valid_after": True,
        "watertight_after": False,
    }
    summary = scene_io._annotate_global_mesh_reduction([report], 100)
    assert scene_io._mesh_reduction_contract_issues(
        [report], summary, enabled=True, target_vertices=100
    ) == []

    report["budget_satisfied"] = False
    summary = scene_io._annotate_global_mesh_reduction([report], 100)
    issues = scene_io._mesh_reduction_contract_issues(
        [report], summary, enabled=True, target_vertices=100
    )
    assert any("budget" in issue for issue in issues)


def test_empty_scene_reduction_summary_is_unverified_not_failed():
    summary = scene_io._annotate_global_mesh_reduction([], 5000)

    assert summary["budget_satisfied"] is None
    assert summary["all_topology_validated"] is None
    assert summary["all_meshes_valid"] is None
    assert scene_io._mesh_reduction_contract_issues(
        [], summary, enabled=True, target_vertices=5000
    ) == []


def test_overfull_edge_mesh_is_invalid_even_without_boundary_edges():
    mesh = bpy.data.meshes.new("overfull_edge_mesh")
    mesh.from_pydata(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        [],
        [(0, 1, 2), (1, 0, 3), (0, 1, 4)],
    )
    mesh.update()
    mesh_obj = bpy.data.objects.new("overfull_edge", mesh)
    bpy.context.scene.collection.objects.link(mesh_obj)

    topology = reduction_impl.mesh_topology_metrics(mesh_obj)

    assert topology["overfull_edges"] == 1
    assert topology["valid"] is False


def test_unsafe_weight_weld_attempts_decimation_but_restores_new_seam_cracks():
    mesh_obj = _split_tetrahedron("weighted_split_tetra")
    duplicate_indices = [
        vertex.index for vertex in mesh_obj.data.vertices if tuple(vertex.co) == (1.0, 1.0, 1.0)
    ]
    group = mesh_obj.vertex_groups.new(name="Bone")
    group.add(duplicate_indices, 1.0, "REPLACE")
    group.add([duplicate_indices[-1]], 0.25, "REPLACE")

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)

    assert len(mesh_obj.data.vertices) == 12
    assert report["applied"] is False
    assert report["reduction_skipped"] is True
    assert report["seam_weld_safe"] is False
    assert report["seam_weld_skipped"] is True
    assert report["seam_vertices_welded"] == 0
    assert report["decimation_attempted"] is True
    assert report["decimation_applied"] is False
    assert report["topology_validated"] is False
    assert report["budget_satisfied"] is False
    assert report["reason"] == "decimation_topology_rejected"
    assert "topology_new_geometric_seam_gaps" in report["topology_validation_issues"]
    assert any(
        issue.startswith("incompatible_vertex_group_weights:")
        for issue in report["seam_weld_issues"]
    )


def test_reduction_restores_snapshot_when_unwelded_decimation_fails_on_shape_keys():
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
    assert report["seam_weld_skipped"] is True
    assert report["decimation_attempted"] is True
    assert report["decimation_applied"] is False
    assert report["reason"] == "decimation_failed_restored"
    assert report["budget_satisfied"] is False
    assert report["decimation_error"]
    assert any(
        issue.startswith("incompatible_shape_key:Expression:")
        for issue in report["seam_weld_issues"]
    )


def test_unsafe_attribute_weld_restores_decimation_that_opens_seam_cracks():
    mesh_obj = _split_tetrahedron("attribute_split_tetra")
    duplicate_indices = [
        vertex.index for vertex in mesh_obj.data.vertices if tuple(vertex.co) == (1.0, 1.0, 1.0)
    ]
    attribute = mesh_obj.data.attributes.new("detail_mask", "FLOAT", "POINT")
    attribute.data[duplicate_indices[-1]].value = 1.0

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)

    assert len(mesh_obj.data.vertices) == 12
    assert report["reduction_skipped"] is True
    assert report["seam_weld_skipped"] is True
    assert report["decimation_attempted"] is True
    assert report["topology_validated"] is False
    assert "topology_new_geometric_seam_gaps" in report["topology_validation_issues"]
    assert any(
        issue.startswith("incompatible_point_attribute:detail_mask:")
        for issue in report["seam_weld_issues"]
    )


def test_reduction_skips_unsafe_touching_shell_weld_without_corrupting_shells():
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
    assert report["seam_weld_skipped"] is True
    assert report["decimation_attempted"] is True
    assert report["budget_satisfied"] is False
    assert "topology_disconnected_face_fans" in report["seam_weld_issues"]


def test_safe_weld_then_runs_decimate_and_remains_manifold():
    mesh_obj = _split_octahedron("split_octahedron")

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)

    assert report["seam_vertices_welded"] == 18
    assert report["welded_source_vertex_count"] == 6
    assert report["decimation_applied"] is True
    assert report["reason"] == "target_reached"
    assert report["budget_satisfied"] is True
    assert report["topology_validated"] is True
    assert report["watertight_after"] is True
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


def test_decimation_that_opens_boundary_is_rejected_and_restored(monkeypatch):
    mesh_obj = _split_octahedron("transactional_octahedron")

    def open_one_face(obj, _target, _source, _importance):
        bm = bmesh.new()
        try:
            bm.from_mesh(obj.data)
            bm.faces.ensure_lookup_table()
            bm.faces.remove(bm.faces[0])
            bm.to_mesh(obj.data)
        finally:
            bm.free()
        obj.data.update()
        return len(obj.data.vertices), 1

    monkeypatch.setattr(reduction_impl, "_run_collapse_passes", open_one_face)

    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)

    # The safe seam weld remains, but the destructive decimation is rolled back.
    assert len(mesh_obj.data.vertices) == 6
    _assert_closed_manifold(mesh_obj)
    assert report["applied"] is True
    assert report["decimation_applied"] is False
    assert report["reason"] == "decimation_topology_rejected"
    assert report["topology_validated"] is False
    assert "topology_new_boundary_edges" in report["topology_validation_issues"]
    assert report["budget_satisfied"] is False


def test_decimation_exception_restores_unsafe_unwelded_source(monkeypatch):
    mesh_obj = _split_tetrahedron("transactional_unsafe_tetrahedron")
    duplicate_indices = [
        vertex.index for vertex in mesh_obj.data.vertices if tuple(vertex.co) == (1.0, 1.0, 1.0)
    ]
    group = mesh_obj.vertex_groups.new(name="Bone")
    group.add(duplicate_indices, 1.0, "REPLACE")
    group.add([duplicate_indices[-1]], 0.25, "REPLACE")

    def fail_after_mutation(obj, _target, _source, _importance):
        obj.data.vertices[0].co.x += 100.0
        raise RuntimeError("synthetic decimator failure")

    monkeypatch.setattr(reduction_impl, "_run_collapse_passes", fail_after_mutation)

    original_positions = [tuple(vertex.co) for vertex in mesh_obj.data.vertices]
    report = scene_io.reduce_mesh_object(mesh_obj, target_vertices=4, enabled=True)

    assert [tuple(vertex.co) for vertex in mesh_obj.data.vertices] == original_positions
    assert report["applied"] is False
    assert report["reduction_skipped"] is True
    assert report["reason"] == "decimation_failed_restored"
    assert "synthetic decimator failure" in report["decimation_error"]
    assert report["budget_satisfied"] is False


def test_weight_aware_reduction_retries_until_budget_without_opening_mesh():
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
    mesh_obj = bpy.context.object
    group_a = mesh_obj.vertex_groups.new(name="BoneA")
    group_b = mesh_obj.vertex_groups.new(name="BoneB")
    vertex_indices = [vertex.index for vertex in mesh_obj.data.vertices]
    group_a.add(vertex_indices, 0.6, "REPLACE")
    group_b.add(vertex_indices, 0.4, "REPLACE")

    report = scene_io.reduce_mesh_object(
        mesh_obj,
        target_vertices=100,
        enabled=True,
        weight_aware_decimation=True,
    )

    assert report["weight_aware"] is True
    assert report["decimation_passes"] >= 2
    assert report["decimation_passes"] <= reduction_impl.MAX_DECIMATION_PASSES
    assert report["budget_satisfied"] is True
    assert report["reduced_vertex_count"] <= 100
    assert report["topology_validated"] is True
    assert report["watertight_after"] is True
    _assert_closed_manifold(mesh_obj)


def test_weight_aware_best_effort_reports_unsatisfied_budget(monkeypatch):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8)
    mesh_obj = bpy.context.object
    group_a = mesh_obj.vertex_groups.new(name="BoneA")
    group_b = mesh_obj.vertex_groups.new(name="BoneB")
    vertex_indices = [vertex.index for vertex in mesh_obj.data.vertices]
    group_a.add(vertex_indices, 0.6, "REPLACE")
    group_b.add(vertex_indices, 0.4, "REPLACE")

    monkeypatch.setattr(
        reduction_impl,
        "_run_collapse_passes",
        lambda _obj, _target, source, _importance: (source, reduction_impl.MAX_DECIMATION_PASSES),
    )

    report = scene_io.reduce_mesh_object(
        mesh_obj,
        target_vertices=50,
        enabled=True,
        weight_aware_decimation=True,
    )

    assert report["weight_aware"] is True
    assert report["budget_satisfied"] is False
    assert report["reason"] == "weight_aware_budget_not_reached"
    assert report["decimation_passes"] == reduction_impl.MAX_DECIMATION_PASSES


def test_global_mesh_budget_preserves_small_accessory_and_caps_total():
    class _Mesh:
        def __init__(self, vertex_count):
            self.data = type("Data", (), {"vertices": range(vertex_count)})()

    walk_like = [_Mesh(8094), _Mesh(42)]
    balanced_pair = [_Mesh(4000), _Mesh(4000)]

    assert scene_io.allocate_mesh_vertex_budgets(walk_like, 5000) == [4958, 42]
    balanced = scene_io.allocate_mesh_vertex_budgets(balanced_pair, 5000)
    assert balanced == [2500, 2500]
    assert sum(balanced) == 5000
