"""Import-path bootstrap shared by the standalone Blender worker."""

from __future__ import annotations

import sys
from pathlib import Path


def prepend_sidecar_src(
    worker_file: str | Path,
    search_path: list[str] | None = None,
) -> tuple[Path, Path]:
    """Put this checkout's public package ahead of Blender site-packages."""

    sidecar_root = Path(worker_file).resolve().parents[1]
    sidecar_src = sidecar_root / "src"
    target = sys.path if search_path is None else search_path
    src_text = str(sidecar_src)
    while src_text in target:
        target.remove(src_text)
    target.insert(0, src_text)
    return sidecar_root, sidecar_src
