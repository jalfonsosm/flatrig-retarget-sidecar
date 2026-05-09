"""GMR (General Motion Retargeting) backend — stub.

This module is a placeholder so the C++ host and the CLI can already wire a
`--backend gmr` path through the factory before the real GMR integration
described in `doc/MOTION_RETARGET_ALTERNATIVES.md` lands. All entry points
return the same shape that the real implementation will use, with `ok=False`
and a clear `detail`. This lets us:

  * register `gmr` as a known backend in the C++ retarget factory,
  * exercise the dispatch path in `cli.py` end-to-end,
  * compare diagnostics shapes against the Motion2Motion backend in tests,

without taking on the full GMR runtime dependency yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

GMR_LABEL = "gmr-stub"


def _stub_payload(reason: str = "GMR backend not yet implemented") -> dict[str, Any]:
    return {
        "ok": False,
        "detail": reason,
        "backend_label": GMR_LABEL,
        "animations": [],
        "diagnostics": {
            "backend_label": GMR_LABEL,
            "stub": True,
            "reason": reason,
        },
    }


def retarget_3d_animations_to_model_gmr(
    source: str | Path,
    target_model: str | Path,
    *,
    output: str | Path | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Stub for the 3D retarget entry point."""
    payload = _stub_payload()
    payload["source"] = str(Path(source).expanduser().resolve())
    payload["target"] = str(Path(target_model).expanduser().resolve())
    if output is not None:
        try:
            output_path = Path(output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import json

            output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            payload["output"] = str(output_path)
        except Exception as exc:  # pragma: no cover — best-effort write for the stub
            payload["detail"] = f"{payload['detail']} (failed to write output: {exc})"
    return payload


def retarget_spine_animation_gmr(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Stub for the 2D Spine retarget entry point."""
    return _stub_payload()


def retarget_bvh_to_spine_animation_gmr(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Stub for the BVH-to-Spine retarget entry point."""
    return _stub_payload()


def probe_gmr_backend() -> dict[str, Any]:
    """Backend probe — always reports unavailable for the stub."""
    return {
        "available": False,
        "backend_label": GMR_LABEL,
        "detail": "GMR backend is not yet integrated; only the wiring exists.",
    }
