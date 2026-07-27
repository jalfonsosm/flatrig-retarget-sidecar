# FlatRig Retarget Sidecar

Public Blender/Python sidecar used by the private FlatRig C++ application.

This package contains the runtime pieces that must live outside the private
repository because they depend on Blender's `bpy` runtime. The private
application talks to this package through the `flatrig-retarget-sidecar` CLI and
through the narrow Python API exposed under `flatrig.scene_formats`.

## Main Responsibilities

- Inspect and normalize supported 3D sources.
- Extract projected 2D scene data from Blender.
- Extract and transfer 3D armature animations into FlatRig's 2D animation
  representation.
- Render sprite parts from a selected projection view.
- Provide mesh cleanup and mesh-target extraction commands used by the native
  optimizer.

## Development

Install in editable mode:

```bash
python -m pip install -e .[dev]
```

Run tests:

```bash
python -m pytest
```

The private FlatRig repository fetches this sidecar during CMake configure. When
developing both repositories side by side, edit the canonical sibling checkout,
not the generated copy under the private repository's build directory.
