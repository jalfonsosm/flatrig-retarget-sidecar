"""``flatrig.animation`` must not reference private helpers that do not exist.

The pure 2D math lives in ``flatrig_private.animation_math`` and reaches this
module through ``import *``, which skips underscore-prefixed names unless they
are listed in that module's ``__all__``. A helper that gets renamed or is never
migrated therefore fails only at *call* time, deep inside a Blender worker
subprocess, as a bare ``NameError`` in the worker's JSON payload - which is how
``_build_2d_basis``, ``_basis_inverse_for_inherit`` and ``_compose_world_matrix``
all went missing at once while the 2D pose extraction path silently died.

This resolves every name statically instead, so the same class of break is a
test failure rather than a runtime surprise.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import flatrig_private.animation_math as animation_math

ANIMATION_SOURCE = Path(__file__).resolve().parents[1] / "src" / "flatrig" / "animation.py"


def _locally_bound_names(tree: ast.AST) -> set[str]:
    """Every name the module binds itself: defs, assignments and imports."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def test_every_private_helper_used_is_actually_exported():
    tree = ast.parse(ANIMATION_SOURCE.read_text(encoding="utf-8"))
    bound = _locally_bound_names(tree)
    exported = set(getattr(animation_math, "__all__", ()))

    used = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    unresolved = sorted(
        name
        for name in used
        if name.startswith("_")
        and name not in bound
        and name not in exported
        and not hasattr(builtins, name)
    )

    assert not unresolved, (
        "flatrig/animation.py references private helpers that "
        f"flatrig_private.animation_math does not export: {unresolved}. "
        "Either add them to its __all__ or update the call sites."
    )


def test_exported_helpers_are_importable_by_name():
    # `import *` is what the module actually relies on; make sure __all__ is not
    # advertising names the module fails to provide.
    missing = sorted(
        name for name in getattr(animation_math, "__all__", ()) if not hasattr(animation_math, name)
    )
    assert not missing, f"__all__ lists names animation_math does not define: {missing}"


def test_animation_projection_uses_the_public_blender_bridge():
    tree = ast.parse(ANIMATION_SOURCE.read_text(encoding="utf-8"))
    imports = {
        node.module: {alias.name for alias in node.names}
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert imports["flatrig._blender_projection"] >= {
        "get_projection_reference_inverse",
        "project_direction_ortho",
        "project_point_ortho",
        "transform_direction_to_projection_space",
    }
    assert "flatrig.projection" not in imports
    assert "flatrig_private.projection_math" not in imports
