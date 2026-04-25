"""Minimal mesh helpers required by the public sidecar Blender worker."""

from __future__ import annotations


def _prune_weights(weights, max_bones_per_vertex=8, threshold=0.001):
    filtered = [
        (int(bone_index), float(weight_value))
        for bone_index, weight_value in (weights or {}).items()
        if float(weight_value) >= float(threshold)
    ]
    filtered.sort(key=lambda item: item[1], reverse=True)
    if max_bones_per_vertex > 0:
        filtered = filtered[: int(max_bones_per_vertex)]

    total = sum(weight_value for _, weight_value in filtered)
    if total <= 1e-8:
        return {}

    return {bone_index: weight_value / total for bone_index, weight_value in filtered}


def transfer_3d_weights_to_2d(obj, bone_name_to_index):
    """Transfer original 3D vertex-group weights directly into 2D bone indices."""
    group_names = {group.index: group.name for group in obj.vertex_groups}
    all_weights = []

    for vert in obj.data.vertices:
        weights = {}
        for group in vert.groups:
            group_name = group_names.get(group.group)
            if group_name in bone_name_to_index and group.weight > 0.001:
                weights[bone_name_to_index[group_name]] = float(group.weight)
        all_weights.append(_prune_weights(weights, 8, 0.001))

    return all_weights
