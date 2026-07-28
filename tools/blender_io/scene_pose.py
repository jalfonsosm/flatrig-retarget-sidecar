"""Scene pose helpers shared by Blender extraction paths."""

from __future__ import annotations

try:
    import bpy
except ImportError:
    bpy = None


def _set_scene_armatures_rest_pose(scene):
    """Temporarily force armatures to rest pose for bind/setup extraction."""
    previous = []
    for scene_obj in scene.objects:
        if scene_obj.type != "ARMATURE" or scene_obj.data is None:
            continue
        previous.append((scene_obj.data, scene_obj.data.pose_position))
        scene_obj.data.pose_position = "REST"
    if previous:
        bpy.context.view_layer.update()
    return previous


def _restore_scene_armature_pose_positions(previous):
    for armature_data, pose_position in previous:
        armature_data.pose_position = pose_position
    if previous:
        bpy.context.view_layer.update()
