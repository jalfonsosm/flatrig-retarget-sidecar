import sys
from pathlib import Path
from types import SimpleNamespace

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from blender_io import scene_pose  # noqa: E402


class _ViewLayer:
    def __init__(self):
        self.update_count = 0

    def update(self):
        self.update_count += 1


def test_scene_rest_pose_helpers_restore_only_armatures(monkeypatch):
    view_layer = _ViewLayer()
    monkeypatch.setattr(scene_pose, "bpy", SimpleNamespace(context=SimpleNamespace(view_layer=view_layer)))
    armature_data = SimpleNamespace(pose_position="POSE")
    scene = SimpleNamespace(
        objects=[
            SimpleNamespace(type="ARMATURE", data=armature_data),
            SimpleNamespace(type="MESH", data=SimpleNamespace(pose_position="POSE")),
            SimpleNamespace(type="ARMATURE", data=None),
        ]
    )

    previous = scene_pose._set_scene_armatures_rest_pose(scene)

    assert previous == [(armature_data, "POSE")]
    assert armature_data.pose_position == "REST"
    assert view_layer.update_count == 1

    scene_pose._restore_scene_armature_pose_positions(previous)

    assert armature_data.pose_position == "POSE"
    assert view_layer.update_count == 2
