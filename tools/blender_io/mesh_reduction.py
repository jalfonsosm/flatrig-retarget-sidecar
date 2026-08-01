"""Mesh welding, duplicate-vertex analysis and weight-aware decimation.

Two layers, kept together because the decimator depends on the welder:

* **Welding** — FBX/glTF importers split a vertex per UV/normal seam, so a mesh
  that looks watertight arrives as loose triangles. ``_weld_exact_position_duplicates``
  merges vertices that share an exact position, but only after
  ``_assess_exact_position_weld`` proves the merge is safe: the duplicates must
  agree on vertex-group weights and every point attribute, and welding must not
  create non-manifold fans. A mesh that fails the assessment remains unwelded;
  its reduction is still attempted transactionally and kept only when topology
  metrics do not regress.
* **Decimation** — ``reduce_mesh_object`` drives Blender's collapse decimator
  toward a vertex budget. With ``enabled`` it biases the collapse using a
  weight-importance vertex group (``_build_decimation_weight_importance``) so
  blend zones survive at the expense of rigid areas, and it watches for regions
  the bias starved (``_weight_aware_starved``) to fall back when the trade goes
  badly.

Extracted verbatim from ``blender_scene_io.py``; only this docstring is new.
The private helpers are re-exported from ``blender_scene_io`` for the callers
and tests that still reach for them by that name.
"""

from __future__ import annotations

# ruff: noqa: I001

import math

try:
    # bpy must be imported before bmesh in the managed bpy runtime.
    import bpy
    import bmesh
except ImportError:
    bpy = None
    bmesh = None

from blender_io.math_utils import VECTOR_EPSILON


def _mesh_triangle_count(mesh_obj) -> int:
    mesh = mesh_obj.data
    if hasattr(mesh, "calc_loop_triangles"):
        mesh.calc_loop_triangles()
    return int(len(getattr(mesh, "loop_triangles", []) or mesh.polygons))


def _weld_position_duplicates(mesh_obj, threshold: float = 1e-6) -> int:
    """Weld geometry split only for per-loop attributes such as UV seams.

    FBX/glTF exporters commonly duplicate a vertex at every UV chart boundary.
    The faces still occupy a closed surface, but Blender's Collapse modifier
    sees those duplicates as independent open boundaries and tears the surface
    during decimation. UVs are stored on face loops, so merging the coincident
    geometry vertices preserves the atlas while restoring manifold adjacency.
    """
    if mesh_obj is None or mesh_obj.type != "MESH":
        return 0
    before = int(len(mesh_obj.data.vertices))
    if before <= 0:
        return 0
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=max(0.0, float(threshold)))
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh_obj.data.update()
    return max(0, before - int(len(mesh_obj.data.vertices)))


def _exact_position_duplicate_clusters(mesh_obj) -> list[tuple[int, ...]]:
    """Return vertex-index clusters whose base-mesh coordinates match exactly."""
    by_position: dict[tuple[float, float, float], list[int]] = {}
    for vertex in mesh_obj.data.vertices:
        key = (float(vertex.co.x), float(vertex.co.y), float(vertex.co.z))
        by_position.setdefault(key, []).append(int(vertex.index))
    return [tuple(indices) for indices in by_position.values() if len(indices) > 1]


def _weld_exact_position_duplicates(mesh_obj, clusters: list[tuple[int, ...]] | None = None) -> int:
    """Weld only explicitly identified exact-position clusters.

    Blender clamps ``bpy.ops.mesh.remove_doubles`` to a minimum threshold of
    1e-6 even when the caller supplies zero.  Use an explicit bmesh target map
    instead so close authored layers can never be selected accidentally.
    """
    if mesh_obj is None or mesh_obj.type != "MESH":
        return 0
    if clusters is None:
        clusters = _exact_position_duplicate_clusters(mesh_obj)
    if not clusters:
        return 0

    before = int(len(mesh_obj.data.vertices))
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    target_map = {}
    for cluster in clusters:
        target = bm.verts[cluster[0]]
        for vertex_index in cluster[1:]:
            target_map[bm.verts[vertex_index]] = target
    bmesh.ops.weld_verts(bm, targetmap=target_map)
    bmesh.update_edit_mesh(mesh_obj.data, loop_triangles=True, destructive=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh_obj.data.update()
    return max(0, before - int(len(mesh_obj.data.vertices)))


def _weld_values_compatible(left, right, epsilon: float = 1e-6) -> bool:
    """Compare Blender attribute values without assuming a particular RNA type."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        left_value = float(left)
        right_value = float(right)
        return (
            math.isfinite(left_value)
            and math.isfinite(right_value)
            and abs(left_value - right_value) <= epsilon
        )
    if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)):
        return type(left) is type(right) and left == right
    try:
        left_items = tuple(left)
        right_items = tuple(right)
    except TypeError:
        return left == right
    return len(left_items) == len(right_items) and all(
        _weld_values_compatible(a, b, epsilon) for a, b in zip(left_items, right_items)
    )


def _vertex_group_weight_signature(mesh_obj, vertex_index: int) -> dict[int, float]:
    signature: dict[int, float] = {}
    for membership in mesh_obj.data.vertices[vertex_index].groups:
        weight = float(membership.weight)
        if abs(weight) > 1e-8:
            signature[int(membership.group)] = weight
    return signature


def _vertex_group_signatures_compatible(left: dict[int, float], right: dict[int, float]) -> bool:
    return left.keys() == right.keys() and all(
        _weld_values_compatible(left[group], right[group]) for group in left
    )


def _point_attribute_value(attribute, vertex_index: int):
    """Read one POINT-domain attribute value, or raise for an unknown data type."""
    if attribute.data_type in {"FLOAT_VECTOR", "FLOAT2"}:
        property_name = "vector"
    elif attribute.data_type in {"FLOAT_COLOR", "BYTE_COLOR"}:
        property_name = "color"
    else:
        property_name = "value"
    element = attribute.data[vertex_index]
    if not hasattr(element, property_name):
        raise TypeError(f"unsupported POINT attribute type {attribute.data_type}")
    return getattr(element, property_name)


def _duplicate_vertex_data_issues(mesh_obj, clusters: list[tuple[int, ...]]) -> list[str]:
    """Return reasons exact duplicate clusters cannot safely share one vertex."""
    issues: list[str] = []
    for cluster_index, cluster in enumerate(clusters):
        first = cluster[0]
        first_weights = _vertex_group_weight_signature(mesh_obj, first)
        if any(
            not _vertex_group_signatures_compatible(
                first_weights, _vertex_group_weight_signature(mesh_obj, vertex_index)
            )
            for vertex_index in cluster[1:]
        ):
            issues.append(f"incompatible_vertex_group_weights:{cluster_index}")

        shape_keys = getattr(mesh_obj.data, "shape_keys", None)
        for key_block in getattr(shape_keys, "key_blocks", []) or []:
            first_value = key_block.data[first].co
            if any(
                not _weld_values_compatible(first_value, key_block.data[index].co)
                for index in cluster[1:]
            ):
                issues.append(f"incompatible_shape_key:{key_block.name}:{cluster_index}")
                break

        for attribute in getattr(mesh_obj.data, "attributes", []) or []:
            if attribute.domain != "POINT" or bool(getattr(attribute, "is_internal", False)):
                continue
            try:
                first_value = _point_attribute_value(attribute, first)
                compatible = all(
                    _weld_values_compatible(
                        first_value, _point_attribute_value(attribute, vertex_index)
                    )
                    for vertex_index in cluster[1:]
                )
            except Exception:
                compatible = False
            if not compatible:
                issues.append(f"incompatible_point_attribute:{attribute.name}:{cluster_index}")

    return issues


def _bmesh_face_fan_count(vertex) -> int:
    """Count disconnected incident-face fans around a bmesh vertex."""
    remaining = set(vertex.link_faces)
    components = 0
    while remaining:
        components += 1
        pending = [remaining.pop()]
        while pending:
            face = pending.pop()
            for edge in face.edges:
                if vertex not in edge.verts:
                    continue
                for neighbor in edge.link_faces:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        pending.append(neighbor)
    return components


def _bmesh_weld_metrics(bm) -> dict[str, int]:
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()

    face_signatures: dict[tuple[int, ...], int] = {}
    degenerate_faces = 0
    for face in bm.faces:
        signature = tuple(sorted(int(vertex.index) for vertex in face.verts))
        face_signatures[signature] = face_signatures.get(signature, 0) + 1
        if len(signature) < 3 or len(set(signature)) < 3 or float(face.calc_area()) <= 0.0:
            degenerate_faces += 1
    return {
        "face_count": len(bm.faces),
        "duplicate_faces": sum(max(0, count - 1) for count in face_signatures.values()),
        "degenerate_faces": degenerate_faces,
        "boundary_edges": sum(1 for edge in bm.edges if len(edge.link_faces) == 1),
        "overfull_edges": sum(1 for edge in bm.edges if len(edge.link_faces) > 2),
    }


def _bmesh_surface_topology(bm) -> tuple[int, int]:
    """Return face-connected components and surface Euler characteristic."""
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    if not bm.faces:
        return 0, 0

    adjacency = [set() for _face in bm.faces]
    for edge in bm.edges:
        linked = [int(face.index) for face in edge.link_faces]
        for index in linked:
            adjacency[index].update(other for other in linked if other != index)
    remaining = set(range(len(bm.faces)))
    components = 0
    while remaining:
        components += 1
        pending = [remaining.pop()]
        while pending:
            face_index = pending.pop()
            neighbors = adjacency[face_index] & remaining
            remaining.difference_update(neighbors)
            pending.extend(neighbors)

    used_vertices = {
        int(vertex.index) for face in bm.faces for vertex in face.verts
    }
    used_edges = {int(edge.index) for face in bm.faces for edge in face.edges}
    euler = len(used_vertices) - len(used_edges) + len(bm.faces)
    return components, euler


def mesh_topology_metrics(mesh_obj) -> dict[str, object]:
    """Return stable topology evidence without mutating the mesh.

    Boundary and overfull edges are reported separately because an imported
    seam-split mesh can start with many boundaries while a genuinely
    non-manifold mesh contains edges shared by more than two faces.  Consumers
    also get their sum as ``non_manifold_edges`` and a conservative
    ``watertight`` boolean.
    """
    empty = {
        "readable": False,
        "vertex_count": 0,
        "edge_count": 0,
        "face_count": 0,
        "duplicate_faces": 0,
        "degenerate_faces": 0,
        "boundary_edges": 0,
        "overfull_edges": 0,
        "non_manifold_edges": 0,
        "non_manifold_vertices": 0,
        "non_finite_vertices": 0,
        "components": 0,
        "euler": 0,
        "handles": None,
        "valid": False,
        "watertight": False,
    }
    if mesh_obj is None or getattr(mesh_obj, "type", None) != "MESH":
        return empty

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh_obj.data)
        metrics = _bmesh_weld_metrics(bm)
        components, euler = _bmesh_surface_topology(bm)
        non_finite_vertices = sum(
            1
            for vertex in bm.verts
            if not all(math.isfinite(float(component)) for component in vertex.co)
        )
        non_manifold_vertices = sum(
            1 for vertex in bm.verts if _bmesh_face_fan_count(vertex) > 1
        )
        metrics.update(
            {
                "readable": True,
                "vertex_count": len(bm.verts),
                "edge_count": len(bm.edges),
                "non_manifold_edges": int(metrics["boundary_edges"])
                + int(metrics["overfull_edges"]),
                "non_finite_vertices": non_finite_vertices,
                "non_manifold_vertices": non_manifold_vertices,
                "components": components,
                "euler": euler,
            }
        )
        metrics["valid"] = bool(
            metrics["face_count"] > 0
            and metrics["non_finite_vertices"] == 0
            and metrics["overfull_edges"] == 0
            and metrics["degenerate_faces"] == 0
            and metrics["duplicate_faces"] == 0
        )
        metrics["watertight"] = bool(
            metrics["valid"] and metrics["boundary_edges"] == 0 and metrics["overfull_edges"] == 0
        )
        metrics["handles"] = (
            max(0, int(round(components - euler / 2.0)))
            if metrics["watertight"]
            else None
        )
        return metrics
    except Exception:
        return empty
    finally:
        bm.free()


def geometric_boundary_pair_metrics(mesh_obj) -> dict[str, object]:
    """Measure whether split boundary edges still occupy the same seam.

    UV/normal seams appear as pairs of boundary edges with identical endpoint
    positions. Counting only combinatorial boundary edges cannot detect two
    copies drifting apart during an unwelded decimation, so this metric groups
    edges by a scale-relative geometric signature.
    """
    empty = {
        "boundary_edges": 0,
        "paired_boundary_edges": 0,
        "unpaired_boundary_edges": 0,
        "paired_fraction": 1.0,
        "position_tolerance": 0.0,
    }
    if mesh_obj is None or getattr(mesh_obj, "type", None) != "MESH":
        return empty
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh_obj.data)
        boundary_edges = [edge for edge in bm.edges if len(edge.link_faces) == 1]
        if not boundary_edges:
            return empty
        coordinates = [
            tuple(float(component) for component in vertex.co)
            for edge in boundary_edges
            for vertex in edge.verts
        ]
        extent = max(
            max(point[axis] for point in coordinates)
            - min(point[axis] for point in coordinates)
            for axis in range(3)
        )
        tolerance = max(float(extent) * 1e-7, 1e-9)

        def point_key(vertex) -> tuple[int, int, int]:
            return tuple(
                int(round(float(component) / tolerance)) for component in vertex.co
            )

        signatures: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = {}
        for edge in boundary_edges:
            signature = tuple(sorted(point_key(vertex) for vertex in edge.verts))
            signatures[signature] = signatures.get(signature, 0) + 1
        total = len(boundary_edges)
        paired = sum(count - count % 2 for count in signatures.values())
        unpaired = total - paired
        return {
            "boundary_edges": total,
            "paired_boundary_edges": paired,
            "unpaired_boundary_edges": unpaired,
            "paired_fraction": float(paired) / float(total),
            "position_tolerance": tolerance,
        }
    finally:
        bm.free()


def geometric_boundary_regression_issues(
    before: dict[str, object], after: dict[str, object]
) -> list[str]:
    """Reject visible seam separation after an unwelded decimation."""
    before_total = int(before.get("boundary_edges") or 0)
    after_total = int(after.get("boundary_edges") or 0)
    if before_total <= 0 or after_total <= 0:
        return []
    issues: list[str] = []
    before_unpaired = int(before.get("unpaired_boundary_edges") or 0)
    after_unpaired = int(after.get("unpaired_boundary_edges") or 0)
    if after_unpaired > before_unpaired:
        issues.append("topology_new_geometric_seam_gaps")
    return issues


def topology_regression_issues(before: dict[str, object], after: dict[str, object]) -> list[str]:
    """Describe destructive topology changes introduced by a reduction pass."""
    issues: list[str] = []
    if not bool(after.get("readable")):
        issues.append("topology_unreadable")
        return issues
    if int(after.get("non_finite_vertices") or 0) > 0:
        issues.append("topology_non_finite_vertices")
    if int(before.get("face_count") or 0) > 0 and int(after.get("face_count") or 0) <= 0:
        issues.append("topology_all_faces_removed")
    if int(after.get("boundary_edges") or 0) > int(before.get("boundary_edges") or 0):
        issues.append("topology_new_boundary_edges")
    if int(after.get("overfull_edges") or 0) > int(before.get("overfull_edges") or 0):
        issues.append("topology_new_overfull_edges")
    if int(after.get("non_manifold_vertices") or 0) > int(
        before.get("non_manifold_vertices") or 0
    ):
        issues.append("topology_new_non_manifold_vertices")
    if int(after.get("duplicate_faces") or 0) > int(before.get("duplicate_faces") or 0):
        issues.append("topology_duplicate_faces")
    if int(after.get("degenerate_faces") or 0) > int(before.get("degenerate_faces") or 0):
        issues.append("topology_degenerate_faces")
    if int(after.get("components") or 0) > int(before.get("components") or 0):
        issues.append("topology_new_components")
    if bool(before.get("watertight")) and bool(after.get("watertight")):
        before_handles = before.get("handles")
        after_handles = after.get("handles")
        if (
            isinstance(before_handles, int)
            and isinstance(after_handles, int)
            and after_handles > before_handles
        ):
            issues.append("topology_new_handles")
    return issues


def _discard_mesh_snapshot(mesh_data) -> None:
    if mesh_data is not None and getattr(mesh_data, "users", 0) == 0:
        bpy.data.meshes.remove(mesh_data)


def _restore_mesh_snapshot(mesh_obj, snapshot) -> None:
    damaged = mesh_obj.data
    mesh_obj.data = snapshot
    if damaged is not snapshot and getattr(damaged, "users", 0) == 0:
        bpy.data.meshes.remove(damaged)
    mesh_obj.data.update()
    bpy.context.view_layer.update()


def _add_flat_topology_metrics(
    report: dict[str, object], prefix: str, metrics: dict[str, object]
) -> None:
    report[f"boundary_edges_{prefix}"] = int(metrics.get("boundary_edges") or 0)
    report[f"overfull_edges_{prefix}"] = int(metrics.get("overfull_edges") or 0)
    report[f"non_manifold_edges_{prefix}"] = int(metrics.get("non_manifold_edges") or 0)
    report[f"watertight_{prefix}"] = bool(metrics.get("watertight"))


def _finalize_reduction_report(report: dict[str, object], mesh_obj) -> dict[str, object]:
    if mesh_obj is not None:
        mesh_obj.data.update()
        output_vertex_count = int(len(mesh_obj.data.vertices))
        output_triangle_count = _mesh_triangle_count(mesh_obj)
        topology_after = mesh_topology_metrics(mesh_obj)
    else:
        output_vertex_count = 0
        output_triangle_count = 0
        topology_after = mesh_topology_metrics(None)

    target_vertices = int(report.get("target_vertices") or 0)
    topology_issues = list(report.get("topology_validation_issues") or [])
    mesh_valid_after = bool(topology_after.get("valid"))
    if not mesh_valid_after and "topology_invalid_output" not in topology_issues:
        topology_issues.append("topology_invalid_output")
    topology_validated = bool(
        report.get("topology_validated") is True and mesh_valid_after
    )
    report.update(
        {
            "output_vertex_count": output_vertex_count,
            "output_triangle_count": output_triangle_count,
            # This count is captured before extract_2d_mesh duplicates vertices
            # at loop/UV seams and is therefore the value the Blender reduction
            # budget actually controls.
            "reduced_vertex_count": output_vertex_count,
            "budget_satisfied": bool(
                target_vertices > 0 and output_vertex_count <= target_vertices
            ),
            "topology_after": topology_after,
            "mesh_valid_after": mesh_valid_after,
            "topology_validation_issues": topology_issues,
            "topology_validated": topology_validated,
        }
    )
    _add_flat_topology_metrics(report, "after", topology_after)
    return report


MESH_VERTEX_BUDGET_FLOOR = 64


def allocate_mesh_vertex_budgets(mesh_objects, total_target_vertices: int) -> list[int]:
    """Allocate one deterministic global vertex budget across scene meshes.

    Small accessories are kept intact up to a modest floor, then the remaining
    budget is distributed proportionally to each object's reducible excess.
    The allocations sum to the requested target whenever that target is
    feasible, preventing a scene with N meshes from silently receiving N times
    the configured budget.
    """
    counts = [max(0, int(len(mesh_obj.data.vertices))) for mesh_obj in mesh_objects]
    target = int(total_target_vertices or 0)
    if not counts:
        return []
    if target <= 0:
        return [0 for _ in counts]
    if sum(counts) <= target:
        return counts

    positive_count = sum(1 for count in counts if count > 0)
    viable_floor = min(
        MESH_VERTEX_BUDGET_FLOOR,
        max(1, target // max(positive_count * 10, 1)),
    )
    budgets = [min(count, viable_floor) for count in counts]

    # Extremely small targets can be lower than even one vertex per object.
    # Keep one vertex allocated to every non-empty object rather than turning a
    # positive target into reduce_mesh_object's special "no target" value.
    if sum(budgets) > target:
        budgets = [min(count, 1) for count in counts]
        if sum(budgets) >= target:
            return budgets

    remaining = max(0, target - sum(budgets))
    capacities = [max(0, count - budget) for count, budget in zip(counts, budgets)]
    capacity_total = sum(capacities)
    if remaining <= 0 or capacity_total <= 0:
        return budgets

    numerators = [remaining * capacity for capacity in capacities]
    additions = [numerator // capacity_total for numerator in numerators]
    for index, addition in enumerate(additions):
        budgets[index] += min(capacities[index], addition)

    leftover = target - sum(budgets)
    ranked = sorted(
        range(len(counts)),
        key=lambda index: (
            -(numerators[index] % capacity_total),
            -capacities[index],
            index,
        ),
    )
    for index in ranked:
        if leftover <= 0:
            break
        if budgets[index] >= counts[index]:
            continue
        budgets[index] += 1
        leftover -= 1
    return budgets


def _exact_weld_topology_issues(mesh_obj, clusters: list[tuple[int, ...]]) -> list[str]:
    """Simulate an exact weld and reject topology damage before mutating Blender data."""
    if not clusters:
        return []
    duplicate_positions = {
        tuple(float(component) for component in mesh_obj.data.vertices[cluster[0]].co)
        for cluster in clusters
    }
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh_obj.data)
        before = _bmesh_weld_metrics(bm)
        before_components, before_euler = _bmesh_surface_topology(bm)
        # Mirror the real mutation exactly. A distance-based operator can have
        # implementation-specific minimum tolerances; this explicit map only
        # joins the exact-position clusters found above.
        bm.verts.ensure_lookup_table()
        target_map = {}
        for cluster in clusters:
            target = bm.verts[cluster[0]]
            for vertex_index in cluster[1:]:
                target_map[bm.verts[vertex_index]] = target
        bmesh.ops.weld_verts(bm, targetmap=target_map)
        after = _bmesh_weld_metrics(bm)
        after_components, after_euler = _bmesh_surface_topology(bm)

        issues: list[str] = []
        if after["face_count"] != before["face_count"]:
            issues.append("topology_face_count_changed")
        if after["duplicate_faces"] > before["duplicate_faces"]:
            issues.append("topology_duplicate_faces")
        if after["degenerate_faces"] > before["degenerate_faces"]:
            issues.append("topology_degenerate_faces")
        if after["boundary_edges"] > before["boundary_edges"]:
            issues.append("topology_new_boundary_edges")
        if after["overfull_edges"] > before["overfull_edges"]:
            issues.append("topology_non_manifold_edges")
        if after_components > before_components:
            issues.append("topology_new_components")
        after_watertight = bool(
            after["face_count"] > 0
            and after["boundary_edges"] == 0
            and after["overfull_edges"] == 0
            and after["duplicate_faces"] == 0
            and after["degenerate_faces"] == 0
        )
        if after_watertight:
            after_handles = max(
                0, int(round(after_components - after_euler / 2.0))
            )
            before_watertight = bool(
                before["face_count"] > 0
                and before["boundary_edges"] == 0
                and before["overfull_edges"] == 0
                and before["duplicate_faces"] == 0
                and before["degenerate_faces"] == 0
            )
            before_handles = (
                max(0, int(round(before_components - before_euler / 2.0)))
                if before_watertight
                else 0
            )
            # Closing an exporter seam is safe only when it does not turn the
            # open seam into a genuine tunnel (for example, welding the two
            # coincident end rings of a cylinder into a torus).
            if after_handles > before_handles:
                issues.append("topology_new_handles")
        for vertex in bm.verts:
            position = tuple(float(component) for component in vertex.co)
            if position not in duplicate_positions:
                continue
            if any(len(edge.link_faces) > 2 for edge in vertex.link_edges):
                issues.append("topology_non_manifold_edges")
                break
            if sum(1 for edge in vertex.link_edges if len(edge.link_faces) == 1) > 2:
                issues.append("topology_non_manifold_vertex")
                break
            if _bmesh_face_fan_count(vertex) > 1:
                issues.append("topology_disconnected_face_fans")
                break
        return list(dict.fromkeys(issues))
    finally:
        bm.free()


def _assess_exact_position_weld(mesh_obj) -> dict[str, object]:
    """Preflight a seam weld used before reducing an externally loaded source."""
    clusters = _exact_position_duplicate_clusters(mesh_obj)
    issues = _duplicate_vertex_data_issues(mesh_obj, clusters)
    if not issues:
        issues.extend(_exact_weld_topology_issues(mesh_obj, clusters))
    return {
        "safe": not issues,
        "clusters": clusters,
        "duplicate_cluster_count": len(clusters),
        "duplicate_vertex_count": sum(len(cluster) - 1 for cluster in clusters),
        "issues": issues,
    }


# Decimation importance (weight-aware collapse). Vertices whose skin weight is
# split between several bones sit in deformation-sensitive blend zones (near
# joints, or where authoring left "problematic" mixed weights); preserving a
# little more geometry there while collapsing flat single-bone regions keeps the
# projected sprite clean where it matters. The signal is the *secondary*
# influence (1 - dominant normalized weight): a vertex fully owned by one bone
# contributes no bonus.
#
# The bias is deliberately *compensated*, not winner-take-all. Every vertex keeps
# a baseline FLOOR weight so rigid regions still survive decimation with most of
# their detail; blend zones only earn a bounded bonus on top. An earlier version
# scored rigid vertices at 0 (free to collapse in Blender's Collapse modifier),
# which dumped the whole vertex budget onto joints, gutted flat regions, and tend
# to trip the starvation guard into a uniform fallback — losing the prior
# entirely. The narrow FLOOR..MAX band keeps the two zones close while still
# protecting joints a touch more.
IMPORTANCE_GROUP_NAME = "FlatRig_DecimateImportance"
# Baseline weight every vertex receives, so rigid single-bone surfaces are never
# treated as free-to-collapse relative to joints.
DECIMATE_IMPORTANCE_FLOOR = 0.45
# Slope of the per-vertex bonus above the floor as the secondary influence grows.
DECIMATE_IMPORTANCE_GAIN = 1.6
# Upper bound on the total weight (floor + bonus). Keeps the rigid:flexible ratio
# gentle (MAX / FLOOR), so detail stays spread out instead of piling onto joints.
DECIMATE_IMPORTANCE_MAX = 0.8
# Blender Collapse decimate: higher group weight preserves detail. Flip only if a
# future bpy build inverts that mapping.
DECIMATE_IMPORTANCE_FACTOR = 1.0
DECIMATE_IMPORTANCE_INVERT = False


def _build_decimation_weight_importance(mesh_obj, enabled: bool = False):
    """Paint a temporary vertex group steering Collapse decimation toward joints.

    Returns the group name, or ``None`` when weight-aware decimation is disabled
    or the mesh carries no usable skin weights. The group is computed purely from
    numeric weights (no bone names), so it stays rig-agnostic and deterministic
    across extract/render passes. Callers MUST remove the group before extraction
    so it is never mistaken for a bone influence.

    The prior is intentionally compensated: every vertex keeps a baseline floor
    weight and blend regions earn only a bounded bonus on top, so a rigid torso
    or armor plate is never starved to feed nearby joints. The fallback guard in
    ``reduce_mesh_object`` still throws away a weighted pass if a sizable region
    collapsed far below its fair share. The path remains opt-in because rigid
    armored assets can deform worse even when the starvation guard does not fire.
    """
    if not enabled:
        return None
    mesh = mesh_obj.data
    if not mesh_obj.vertex_groups or len(mesh.vertices) == 0:
        return None

    # Seed every vertex with the baseline floor (rigid/unskinned surfaces keep
    # their fair share of geometry) and only raise blend-zone vertices above it.
    importance = [DECIMATE_IMPORTANCE_FLOOR] * len(mesh.vertices)
    any_bonus = False
    for vert in mesh.vertices:
        weights = [g.weight for g in vert.groups if g.weight > 0.0]
        if len(weights) < 2:
            continue  # single-bone/unskinned: keep the baseline floor only
        total = float(sum(weights))
        if total <= VECTOR_EPSILON:
            continue
        secondary = 1.0 - (max(weights) / total)
        if secondary <= 0.0:
            continue
        score = min(
            DECIMATE_IMPORTANCE_MAX,
            DECIMATE_IMPORTANCE_FLOOR + secondary * DECIMATE_IMPORTANCE_GAIN,
        )
        importance[vert.index] = score
        any_bonus = True

    # No blend zones means a flat floor everywhere == uniform; let the plain
    # uniform path handle it instead of painting a pointless group.
    if not any_bonus:
        return None

    group = mesh_obj.vertex_groups.new(name=IMPORTANCE_GROUP_NAME)
    for index, score in enumerate(importance):
        group.add([index], score, "REPLACE")
    return group.name


# Auto-fallback guard for weight-aware decimation. A region (dominant vertex
# group) that lost almost all of its geometry means the weight prior dumped the
# whole budget elsewhere, hollowing that region into a hole. When that happens we
# throw the weight-aware result away and redo the decimation uniformly.
#
# The guard is measured *relative to the global decimation ratio*, not against a
# fixed retention floor. Decimating to (say) 25 % of the source naturally brings
# every region to ~25 %, so an absolute 0.25 floor used to fire on essentially
# every aggressive run and force a needless uniform fallback. A region is only
# starved when it kept far less than its fair share of the global ratio.
DECIMATE_MIN_REGION_VERTS = 25
# A region must retain at least this fraction of its *fair share* (before_count *
# global_ratio). Below it, the prior genuinely hollowed the region.
DECIMATE_REGION_STARVE_FACTOR = 0.45
# A weighted modifier can preserve enough high-importance vertices that one
# ratio pass stops above budget. Recompute the ratio from the actual output and
# retry a bounded number of times; topology validation below still decides
# whether the result is safe to keep.
MAX_DECIMATION_PASSES = 4


def _dominant_group_counts(mesh_obj) -> dict:
    """Count vertices per dominant vertex group (≈ per bone region)."""
    counts: dict = {}
    for vert in mesh_obj.data.vertices:
        best_group, best_weight = -1, 0.0
        for influence in vert.groups:
            if influence.weight > best_weight:
                best_weight, best_group = influence.weight, influence.group
        if best_group >= 0:
            counts[best_group] = counts.get(best_group, 0) + 1
    return counts


def _weight_aware_starved(before: dict, after: dict, global_ratio: float) -> tuple:
    """Return (starved, worst_group) if a region collapsed far below its fair share.

    ``global_ratio`` is the overall target/source vertex ratio. A region's fair
    share after decimation is ``before_count * global_ratio``; it is flagged as
    starved only when it kept less than ``DECIMATE_REGION_STARVE_FACTOR`` of that,
    i.e. it shrank much harder than the mesh as a whole (the prior diverted its
    budget) rather than merely because the target is aggressive.
    """
    floor_ratio = max(0.0, float(global_ratio)) * DECIMATE_REGION_STARVE_FACTOR
    for group, before_count in before.items():
        if before_count >= DECIMATE_MIN_REGION_VERTS:
            after_count = after.get(group, 0)
            if after_count < before_count * floor_ratio:
                return True, group
    return False, None


def _run_collapse_passes(mesh_obj, target_vertices, source_vertex_count, importance_group):
    """Apply bounded Collapse passes and return ``(vertices, pass_count)``."""
    current_vertices = source_vertex_count
    pass_count = 0
    for pass_index in range(MAX_DECIMATION_PASSES):
        if current_vertices <= target_vertices:
            break
        ratio = max(0.01, min(1.0, float(target_vertices) / max(float(current_vertices), 1.0)))
        modifier = mesh_obj.modifiers.new(
            name=f"FlatRig_SourceMeshReduction_{pass_index + 1}",
            type="DECIMATE",
        )
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio
        if importance_group is not None and importance_group in mesh_obj.vertex_groups:
            modifier.vertex_group = importance_group
            if hasattr(modifier, "vertex_group_factor"):
                modifier.vertex_group_factor = DECIMATE_IMPORTANCE_FACTOR
            if hasattr(modifier, "invert_vertex_group"):
                modifier.invert_vertex_group = DECIMATE_IMPORTANCE_INVERT
        if hasattr(modifier, "use_collapse_triangulate"):
            modifier.use_collapse_triangulate = True
        # Source reduction must operate on the undeformed base mesh. Imported
        # rigged FBX files normally have an Armature modifier already at index
        # zero; applying a newly appended Decimate after it makes Blender bake
        # an out-of-order evaluated stack and emits "Applied modifier was not
        # first". Move only our temporary modifier to the front. Applying it
        # removes it again, leaving every authored modifier in its original
        # relative order and preserving the Armature modifier itself.
        modifier_index = mesh_obj.modifiers.find(modifier.name)
        if modifier_index > 0:
            mesh_obj.modifiers.move(modifier_index, 0)
        try:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        except Exception:
            # Applying normally removes the modifier. On failure Blender can
            # leave it behind; never let a rejected transactional pass leak a
            # live modifier into later extraction/rendering.
            if modifier.name in mesh_obj.modifiers:
                mesh_obj.modifiers.remove(mesh_obj.modifiers[modifier.name])
            raise
        bpy.context.view_layer.update()
        pass_count += 1
        next_vertices = int(len(mesh_obj.data.vertices))
        if next_vertices >= current_vertices:
            current_vertices = next_vertices
            break
        current_vertices = next_vertices
    return current_vertices, pass_count


def reduce_mesh_object(
    mesh_obj,
    target_vertices=5000,
    enabled=True,
    weight_aware_decimation: bool = False,
) -> dict[str, object]:
    """Reduce the source mesh in Blender before extraction.

    Decimate runs inside Blender so UV layers and vertex-group weights remain on
    the mesh that the native pipeline receives. By default decimation is uniform;
    callers can opt into a temporary importance group that keeps detail near
    joints/blend zones and collapses flat regions harder.
    """
    source_vertex_count = int(len(mesh_obj.data.vertices)) if mesh_obj is not None else 0
    source_triangle_count = _mesh_triangle_count(mesh_obj) if mesh_obj is not None else 0
    target_vertices = int(target_vertices or 0)
    topology_before = mesh_topology_metrics(mesh_obj)
    report = {
        "enabled": bool(enabled),
        "applied": False,
        "mode": "sidecar_blender_decimate",
        "target_vertices": target_vertices,
        "source_vertex_count": source_vertex_count,
        "source_triangle_count": source_triangle_count,
        "output_vertex_count": source_vertex_count,
        "output_triangle_count": source_triangle_count,
        "reason": "disabled" if not enabled else "not_run",
        "weight_aware_requested": bool(weight_aware_decimation),
        "seam_vertices_welded": 0,
        "welded_source_vertex_count": source_vertex_count,
        "seam_weld_checked": False,
        "seam_weld_safe": None,
        "seam_weld_issues": [],
        "seam_weld_skipped": False,
        "position_duplicate_cluster_count": 0,
        "reduction_skipped": False,
        "decimation_applied": False,
        "decimation_attempted": False,
        "decimation_passes": 0,
        "weight_aware_passes": 0,
        "uniform_passes": 0,
        "topology_before": topology_before,
        "topology_validation_issues": [],
        "topology_validated": True,
    }
    _add_flat_topology_metrics(report, "before", topology_before)

    if not enabled:
        return _finalize_reduction_report(report, mesh_obj)
    if mesh_obj is None:
        report["reason"] = "no_mesh"
        report["topology_validated"] = False
        return _finalize_reduction_report(report, mesh_obj)
    if target_vertices <= 0:
        report["reason"] = "no_target"
        return _finalize_reduction_report(report, mesh_obj)
    if source_vertex_count <= target_vertices:
        # No reduction is needed, so leave even exactly coincident authored
        # surfaces untouched.  The seam weld is only a prerequisite for an
        # actual collapse pass (or for discovering that seam duplicates alone
        # account for the apparent excess over the budget).
        report["reason"] = "source_under_target"
        return _finalize_reduction_report(report, mesh_obj)
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    try:
        weld_assessment = _assess_exact_position_weld(mesh_obj)
    except Exception as exc:
        weld_assessment = {
            "safe": False,
            "clusters": [],
            "duplicate_cluster_count": 0,
            "issues": [f"weld_preflight_failed:{type(exc).__name__}"],
        }
    report["seam_weld_checked"] = True
    report["seam_weld_safe"] = bool(weld_assessment["safe"])
    report["seam_weld_issues"] = list(weld_assessment["issues"])
    report["position_duplicate_cluster_count"] = int(weld_assessment["duplicate_cluster_count"])
    geometric_boundary_before = geometric_boundary_pair_metrics(mesh_obj)
    report["geometric_boundary_before"] = geometric_boundary_before
    seam_vertices_welded = 0
    if weld_assessment["safe"]:
        # Keep welding transactional too. A bpy/runtime failure must not strand
        # a partially welded source; restore it and continue with an unwelded,
        # topology-checked decimation attempt.
        weld_snapshot = mesh_obj.data.copy()
        try:
            # Exact equality is intentional here. Unlike generated-mesh cleanup,
            # external FBX input may contain authored layers that are merely close
            # together; only exporter-created vertices at the identical base-mesh
            # position are candidates, and the preflight above validates their data
            # and the resulting topology before this mutation is allowed.
            seam_vertices_welded = _weld_exact_position_duplicates(
                mesh_obj, clusters=weld_assessment["clusters"]
            )
        except Exception as exc:
            _restore_mesh_snapshot(mesh_obj, weld_snapshot)
            report["seam_weld_safe"] = False
            report["seam_weld_skipped"] = True
            report["seam_weld_issues"].append(f"weld_failed:{type(exc).__name__}")
        else:
            _discard_mesh_snapshot(weld_snapshot)
    else:
        # An unsafe weld can represent intentional coincident shells or
        # incompatible authored data. Preserve those vertices, but do not let
        # one unsafe cluster cancel reduction of the entire mesh. The collapse
        # itself runs from a snapshot and is accepted only if boundary and
        # overfull-edge counts do not regress.
        report["seam_weld_skipped"] = True

    reduction_source_vertex_count = int(len(mesh_obj.data.vertices))
    report["seam_vertices_welded"] = seam_vertices_welded
    report["welded_source_vertex_count"] = reduction_source_vertex_count
    if reduction_source_vertex_count <= target_vertices:
        report["applied"] = seam_vertices_welded > 0
        report["reason"] = (
            "source_under_target_after_weld" if seam_vertices_welded > 0 else "source_under_target"
        )
        return _finalize_reduction_report(report, mesh_obj)

    # Everything after this point is transactional. Safe welding, when it ran,
    # is the baseline we restore to; an unsafe weld leaves the original mesh as
    # that baseline. This makes it safe to try Blender's decimator on Walk-like
    # assets without accepting new holes or overfull edges.
    reduction_snapshot = mesh_obj.data.copy()
    decimation_topology_before = mesh_topology_metrics(mesh_obj)
    report["decimation_topology_before"] = decimation_topology_before
    report["decimation_attempted"] = True

    importance_group = None
    weight_aware_attempted = False
    weight_aware_used = False
    fallback_applied = False
    decimation_passes = 0
    weight_aware_passes = 0
    uniform_passes = 0
    decimation_error = None
    try:
        try:
            importance_group = _build_decimation_weight_importance(
                mesh_obj,
                enabled=bool(weight_aware_decimation),
            )
        except Exception:
            importance_group = None
        weight_aware_attempted = importance_group is not None

        if importance_group is not None:
            before_counts = _dominant_group_counts(mesh_obj)
            _, weighted_passes = _run_collapse_passes(
                mesh_obj,
                target_vertices,
                reduction_source_vertex_count,
                importance_group,
            )
            decimation_passes += weighted_passes
            weight_aware_passes += weighted_passes
            global_ratio = float(target_vertices) / max(float(reduction_source_vertex_count), 1.0)
            starved, _worst = _weight_aware_starved(
                before_counts, _dominant_group_counts(mesh_obj), global_ratio
            )
            if importance_group in mesh_obj.vertex_groups:
                mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[importance_group])
            importance_group = None
            if starved:
                # Restore an independent copy so the original transaction
                # snapshot remains available if the uniform retry also damages
                # topology or fails.
                _restore_mesh_snapshot(mesh_obj, reduction_snapshot.copy())
                if IMPORTANCE_GROUP_NAME in mesh_obj.vertex_groups:
                    mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[IMPORTANCE_GROUP_NAME])
                fallback_applied = True
                _, uniform_passes = _run_collapse_passes(
                    mesh_obj, target_vertices, reduction_source_vertex_count, None
                )
                decimation_passes += uniform_passes
            else:
                weight_aware_used = True
        else:
            _, uniform_passes = _run_collapse_passes(
                mesh_obj, target_vertices, reduction_source_vertex_count, None
            )
            decimation_passes += uniform_passes
    except Exception as exc:
        decimation_error = f"{type(exc).__name__}: {exc}"
    finally:
        # The importance group must never reach extraction; otherwise it would be
        # read back as a spurious bone weight.
        if IMPORTANCE_GROUP_NAME in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[IMPORTANCE_GROUP_NAME])

    report["decimation_passes"] = decimation_passes
    report["weight_aware_passes"] = weight_aware_passes
    report["uniform_passes"] = uniform_passes
    report["weight_aware"] = weight_aware_used
    report["weight_aware_attempted"] = weight_aware_attempted
    report["weight_aware_fallback"] = fallback_applied

    if decimation_error is not None:
        _restore_mesh_snapshot(mesh_obj, reduction_snapshot)
        report.update(
            {
                "reason": "decimation_failed_restored",
                "applied": seam_vertices_welded > 0,
                "reduction_skipped": seam_vertices_welded == 0,
                "decimation_error": decimation_error,
                "topology_validated": False,
            }
        )
        return _finalize_reduction_report(report, mesh_obj)

    mesh_obj.data.update()
    decimation_topology_after = mesh_topology_metrics(mesh_obj)
    topology_issues = topology_regression_issues(
        decimation_topology_before, decimation_topology_after
    )
    geometric_boundary_after = geometric_boundary_pair_metrics(mesh_obj)
    report["geometric_boundary_after"] = geometric_boundary_after
    if not bool(weld_assessment["safe"]):
        topology_issues.extend(
            geometric_boundary_regression_issues(
                geometric_boundary_before, geometric_boundary_after
            )
        )
        topology_issues = list(dict.fromkeys(topology_issues))
    report["decimation_topology_after"] = decimation_topology_after
    report["topology_validation_issues"] = topology_issues
    if topology_issues:
        _restore_mesh_snapshot(mesh_obj, reduction_snapshot)
        report.update(
            {
                "reason": "decimation_topology_rejected",
                "applied": seam_vertices_welded > 0,
                "reduction_skipped": seam_vertices_welded == 0,
                "topology_validated": False,
            }
        )
        return _finalize_reduction_report(report, mesh_obj)

    _discard_mesh_snapshot(reduction_snapshot)
    output_vertex_count = int(len(mesh_obj.data.vertices))
    budget_satisfied = output_vertex_count <= target_vertices
    if fallback_applied:
        reason = (
            "weight_aware_fallback_uniform"
            if budget_satisfied
            else "weight_aware_fallback_uniform_budget_not_reached"
        )
    elif weight_aware_used:
        reason = (
            "weight_aware_target_reached" if budget_satisfied else "weight_aware_budget_not_reached"
        )
    else:
        reason = "target_reached" if budget_satisfied else "best_effort_budget_not_reached"
    report.update(
        {
            "applied": output_vertex_count < source_vertex_count or seam_vertices_welded > 0,
            "decimation_applied": output_vertex_count < reduction_source_vertex_count,
            "reason": reason,
            "reduction_skipped": not (
                output_vertex_count < source_vertex_count or seam_vertices_welded > 0
            ),
        }
    )
    return _finalize_reduction_report(report, mesh_obj)
