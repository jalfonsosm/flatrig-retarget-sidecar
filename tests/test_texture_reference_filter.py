import bpy

from flatrig import texture


def _mesh_object(name, vertices, faces):
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _remove_mesh_object(obj):
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh, do_unlink=True)


def test_filtered_reference_removes_decoy_triangle_and_hides_decoy_object(tmp_path, monkeypatch):
    # Body has one exported triangle and one source triangle that segmentation
    # did not export. A separate cape object is also absent from the union.
    body = _mesh_object(
        "FilteredReferenceBody",
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        [(0, 1, 2), (0, 2, 3)],
    )
    decoy = _mesh_object(
        "FilteredReferenceDecoyCape",
        [(2, 0, 0), (3, 0, 0), (2, 0, 1)],
        [(0, 1, 2)],
    )
    captured = {}

    def fake_render(scene, _output_path, resolution):
        visible_meshes = [
            obj for obj in scene.objects if obj.type == "MESH" and not obj.hide_render
        ]
        captured["resolution"] = resolution
        captured["visible_meshes"] = visible_meshes
        assert len(visible_meshes) == 1
        assert visible_meshes[0].name.startswith("FilteredReferenceBody_sidecar_reference")
        assert len(visible_meshes[0].data.polygons) == 1
        assert tuple(sorted(visible_meshes[0].data.polygons[0].vertices)) == (0, 1, 2)
        assert body.hide_render
        assert decoy.hide_render
        return True

    monkeypatch.setattr(texture, "render_projected_sprite", fake_render)
    try:
        ok = texture.render_preview_sprite(
            body,
            {
                "right_axis": (1.0, 0.0, 0.0),
                "up_axis": (0.0, 0.0, 1.0),
                "view_dir": (0.0, -1.0, 0.0),
                "depth_axis": (0.0, 1.0, 0.0),
            },
            {"center_x": 0.5, "center_y": 0.5, "span": 2.0},
            str(tmp_path / "filtered_reference.png"),
            resolution=64,
            triangle_groups=[
                {
                    "object_name": "FilteredReferenceBody",
                    "object": body,
                    "triangle_keys": [(0, 1, 2)],
                }
            ],
        )
        assert ok
        assert captured["resolution"] == 64
        assert not body.hide_render
        assert not decoy.hide_render
    finally:
        if body.name in bpy.data.objects:
            _remove_mesh_object(body)
        if decoy.name in bpy.data.objects:
            _remove_mesh_object(decoy)
