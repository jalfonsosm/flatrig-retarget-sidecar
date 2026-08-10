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

On Linux and macOS the scene worker first uses the installed `bpy` module and
keeps a separate Blender process as its fallback. Windows intentionally uses a
different policy: automatic mode requires the official Blender 4.2+ executable
and never imports the pip wheel's native `bpy.pyd`, which application-control
products can block before Python can recover. An explicit
`FLATRIG_RETARGET_SCENE_BACKEND=bpy` remains available for development only.
The normal Windows package does not install the `bpy` wheel at all; installing
the documented `.[dev]` extra adds it for sidecar tests and diagnostics.

Windows discovery checks `FLATRIG_RETARGET_BLENDER`, bundled/portable layouts,
`PATH`, the official installer's `App Paths` registration, standard
`Program Files/Blender Foundation` installs, and Blender installed in a Steam
library. `python -m flatrig.cli probe` reports the selected path, detected
version, and minimum version. Blender 5.x is recommended; 5.2 is the version
used for the current Windows integration validation. No security-policy
exception is required. The
Blender worker adds this checkout's `src` directory itself, so the CLI path does
not depend on `PYTHONPATH` or Blender's `--python-use-system-env` option.

The private FlatRig repository fetches this sidecar during CMake configure. When
developing both repositories side by side, edit the canonical sibling checkout,
not the generated copy under the private repository's build directory.

## License

This Blender/`bpy` worker is licensed under GPL-3.0-or-later. See `LICENSE`.
FlatRig invokes it as a separate command-line process through a JSON/file
boundary; no sidecar Python module is linked into the private native
application.
