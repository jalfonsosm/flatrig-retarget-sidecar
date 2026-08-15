import sys
from pathlib import Path

import bpy

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))
import blender_scene_io as scene_io  # noqa: E402
from blender_io import mesh_reduction as reduction_impl  # noqa: E402


def setup_function():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_synthetic_sphere(_source_path: str) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12)
    bpy.context.object.name = "synthetic_cleanup_sphere"


def test_cleanup_reports_budget_and_topology_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_io, "import_model", _import_synthetic_sphere)

    report = scene_io.cleanup_generated_mesh(
        "synthetic.glb",
        glb_output=str(tmp_path / "cleaned.glb"),
        target_triangles=200,
        voxel_remesh=False,
        remove_loose=False,
    )

    assert report["ok"] is True
    assert report["decimated"] is True
    assert report["decimation_rejected"] is False
    assert report["budget_satisfied"] is True
    assert report["triangles_after"] <= 200
    assert report["topology_validated"] is True
    assert report["boundary_edges_after"] <= report["boundary_edges_before"]
    assert report["overfull_edges_after"] <= report["overfull_edges_before"]
    assert report["watertight_after"] is True
    assert report["contract_ok"] is True
    assert Path(report["output"]).exists()


def test_cleanup_restores_decimation_rejected_by_topology_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_io, "import_model", _import_synthetic_sphere)
    monkeypatch.setattr(
        scene_io,
        "topology_regression_issues",
        lambda _before, _after: ["synthetic_topology_regression"],
    )

    report = scene_io.cleanup_generated_mesh(
        "synthetic.glb",
        glb_output=str(tmp_path / "restored.glb"),
        target_triangles=200,
        voxel_remesh=False,
        remove_loose=False,
    )

    assert report["ok"] is False
    assert report["decimated"] is False
    assert report["decimation_rejected"] is True
    assert report["budget_satisfied"] is False
    assert report["triangles_after"] == report["triangles_before"]
    assert report["topology_validated"] is False
    assert report["topology_validation_issues"] == ["synthetic_topology_regression"]
    assert report["watertight_after"] is True
    assert report["contract_ok"] is False
    assert report["output"] is None


def _import_synthetic_torus(_source_path: str) -> None:
    """A closed, single-piece, consistently wound surface with one handle."""
    bpy.ops.mesh.primitive_torus_add(major_segments=32, minor_segments=16)
    bpy.context.object.name = "synthetic_cleanup_torus"


def test_a_handle_fails_the_contract_unless_the_caller_allows_one(
    tmp_path, monkeypatch
):
    """The tolerance is the caller's to set, and 0 stays the default.

    Whether a handle is real character topology -- a hand resting on a hip --
    or a reconstruction artifact is a product judgement this tool cannot make,
    so it enforces the number it is given rather than choosing one. Nothing
    else moves: the surface still has to be closed and single-piece.
    """
    monkeypatch.setattr(scene_io, "import_model", _import_synthetic_torus)

    strict = scene_io.cleanup_generated_mesh(
        "synthetic.glb",
        glb_output=str(tmp_path / "strict.glb"),
        target_triangles=0,
        voxel_remesh=False,
        remove_loose=False,
    )

    assert strict["topology_after"]["handles"] == 1
    assert strict["handle_tolerance"] == 0
    assert strict["strict_topology_satisfied"] is False
    assert "closed_zero_handle_topology_not_satisfied" in strict["contract_issues"]
    assert strict["contract_ok"] is False
    # A refused contract writes no mesh at all, which is why the caller's
    # tolerance has to reach this far and not just the report it reads back.
    assert strict["output"] is None

    monkeypatch.setattr(scene_io, "import_model", _import_synthetic_torus)
    tolerant = scene_io.cleanup_generated_mesh(
        "synthetic.glb",
        glb_output=str(tmp_path / "tolerant.glb"),
        target_triangles=0,
        voxel_remesh=False,
        remove_loose=False,
        handle_tolerance=1,
    )

    assert tolerant["topology_after"]["handles"] == 1
    assert tolerant["handle_tolerance"] == 1
    assert tolerant["strict_topology_satisfied"] is True
    assert "closed_zero_handle_topology_not_satisfied" not in tolerant["contract_issues"]
    assert tolerant["contract_ok"] is True
    assert Path(tolerant["output"]).exists()


def test_open_surface_decimation_cannot_create_new_components():
    before = {
        "readable": True,
        "face_count": 10,
        "non_finite_vertices": 0,
        "boundary_edges": 4,
        "overfull_edges": 0,
        "duplicate_faces": 0,
        "degenerate_faces": 0,
        "components": 1,
        "watertight": False,
        "handles": None,
    }
    after = {**before, "components": 2}

    assert "topology_new_components" in reduction_impl.topology_regression_issues(
        before, after
    )
