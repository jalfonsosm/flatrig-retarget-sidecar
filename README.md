# flatrig-retarget-sidecar

Public Python sidecar for generic Spine, BVH, and 3D scene retargeting workflows.

This repository intentionally contains only the public integration layer:

- Motion2Motion-based retarget orchestration.
- Spine import and Spine/BVH bridge utilities.
- Scene inspection and format conversion through `bpy` first, Blender CLI second.
- A stable CLI contract for native or scripted clients.

## Runtime Model

The sidecar is designed around optional backends:

- Base package: public CLI, Spine parsing, sparse mapping, and scene-backend orchestration.
- Motion2Motion backend: optional. Installed through `tools/install_motion2motion_backend.py`.
- Torch runtime: provisioned separately through `tools/install_torch_runtime.py` because the correct wheel flavor depends on the host platform.

That separation keeps the package importable in CI without forcing heavyweight GPU runtimes.

## Install

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python tools/install_torch_runtime.py --python .venv/bin/python
python tools/install_motion2motion_backend.py --install-deps
pre-commit install
```

Optional extras:

```bash
python -m pip install -e .[motion2motion]
python -m pip install -e .[all]
```

`tools/install_torch_runtime.py` pins the sidecar to a single Torch version (`2.9.1`) while selecting the right wheel flavor for the platform:

- Apple Silicon: default wheels with `MPS`
- Windows/Linux: CUDA wheels by default
- CPU-only override: `FLATRIG_TORCH_FLAVOR=cpu`

`tools/install_motion2motion_backend.py` reuses the root `.venv` by default so host clients and Motion2Motion resolve the same Torch runtime. Pass `--dedicated-venv` only if you explicitly want a separate backend environment.

## CLI

```bash
python cli/flatrig_retarget.py probe
python cli/flatrig_retarget.py retarget-spine source.json target.json --animation walk --output out.json
python cli/flatrig_retarget.py retarget-bvh-to-spine source.bvh target.json --output out.json
python cli/flatrig_retarget.py spine-to-json source.skel --output skeleton.json
python cli/flatrig_retarget.py inspect-3d-source source.fbx --output inspect.json
python cli/flatrig_retarget.py convert-3d-source source.fbx --output normalized.glb
python cli/flatrig_retarget.py extract-scene source.fbx --output scene.json
python cli/flatrig_retarget.py extract-animations source.fbx --output animations.json
python cli/flatrig_retarget.py render-sprites source.fbx --output sprites.json --parts-json parts.json --images-dir sprites
```

## Quality Gates

Local checks:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=flatrig_retarget_sidecar --cov-report=term-missing
python -m pre_commit run --all-files
```

CI runs:

- `pre-commit` on the full repository
- unit tests on Python `3.10`, `3.11`, and `3.12`
- coverage reporting from the unit-test job

The default CI suite does not require Motion2Motion, `bpy`, Blender, CUDA, or Node.

## Scene Backends

Scene import/export is `bpy`-first. The sidecar falls back to a bundled or system Blender executable only when `bpy` is unavailable:

```bash
export FLATRIG_RETARGET_SCENE_BACKEND=bpy
export FLATRIG_RETARGET_BLENDER=/path/to/bundled/blender
```

Accepted values for `FLATRIG_RETARGET_SCENE_BACKEND`:

- `auto`
- `bpy`
- `blender`

## Motion2Motion Notes

Motion2Motion device policy is intentionally conservative:

- `cuda` is used automatically when available.
- Apple `mps` remains opt-in for Motion2Motion because the upstream runtime still hits CPU/GPU tensor mismatches on MPS in real retarget runs.
- To experiment with it anyway, set `FLATRIG_M2M_ALLOW_MPS=1` or force `FLATRIG_M2M_DEVICE=mps`.

You can also override the Torch bootstrap:

```bash
export FLATRIG_TORCH_VERSION=2.9.1
export FLATRIG_TORCH_FLAVOR=cu128
export FLATRIG_TORCH_INDEX_URL=
```

## Spine Binary Support

Spine `.skel` parsing uses `tools/spine_binary_export.cjs` plus a runtime bootstrapped under `workflow/.spine_binary_runtime`.

Requirements for `.skel` support:

- `node`
- `npm`

Plain Spine `.json` and `.zip` packages do not need Node.

## Status

- Motion2Motion path: supported and tested through the public CLI contract.
- Scene conversion path: supported through `bpy` or Blender CLI.
