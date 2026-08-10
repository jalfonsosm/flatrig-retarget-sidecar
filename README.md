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
- Provide mesh cleanup helpers used by the native pipeline.

## Development

Install in editable mode:

```bash
python -m pip install -e .[dev]
```

Run tests:

```bash
python -m pytest
```

The scene worker uses the installed `bpy` module when it can be loaded. On
Windows, automatic mode first looks for a separate Blender process so an
application-control policy never has to load an unsigned Python extension. It
discovers the official installer through the `App Paths` registry entry and the
standard `Program Files/Blender Foundation` layout. Portable installs can be
selected with `FLATRIG_RETARGET_BLENDER` or by putting `blender` on `PATH`; no
security-policy exception is required. If no Blender executable is present,
automatic mode retains the `bpy` module fallback. The Blender worker adds this
checkout's `src` directory itself, so the CLI fallback does not depend on
`PYTHONPATH` or Blender's `--python-use-system-env` option.

The private FlatRig repository fetches this sidecar during CMake configure. When
developing both repositories side by side, edit the canonical sibling checkout,
not the generated copy under the private repository's build directory.

## License

This Blender/`bpy` worker is licensed under GPL-3.0-or-later. See `LICENSE`.
FlatRig invokes it as a separate command-line process through a JSON/file
boundary; no sidecar Python module is linked into the private native
application.
