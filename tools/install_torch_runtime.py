"""Install the pinned FlatRig Torch runtime with the right platform flavor."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys

DEFAULT_TORCH_VERSION = "2.9.1"
DEFAULT_CUDA_FLAVOR = "cu128"
INDEX_URLS = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu128": "https://download.pytorch.org/whl/cu128",
}


def detect_default_flavor() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "default"
    if system in {"linux", "windows"}:
        return DEFAULT_CUDA_FLAVOR
    return "default"


def resolve_index_url(flavor: str, override: str | None) -> str | None:
    if override:
        return override
    normalized = flavor.strip().lower()
    if normalized in {"default", "mps", "auto"}:
        return None
    return INDEX_URLS.get(normalized)


def run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}")


def pip_install_torch(python_executable: str, version: str, index_url: str | None) -> None:
    command = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        f"torch=={version}",
    ]
    if index_url:
        command.extend(["--index-url", index_url])
    run(command)


def probe_torch(python_executable: str) -> dict[str, object]:
    probe_script = """
import json
from pathlib import Path
import torch

payload = {
    "version": getattr(torch, "__version__", ""),
    "cuda_available": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
    "mps_available": bool((
        getattr(getattr(torch, "backends", None), "mps", None)
        and torch.backends.mps.is_available()
    ) or (getattr(torch, "mps", None) and torch.mps.is_available())),
    "cmake_dir": str((Path(torch.__file__).resolve().parent / "share" / "cmake" / "Torch")),
}
print(json.dumps(payload))
"""
    completed = subprocess.run(
        [python_executable, "-c", probe_script],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip())
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the pinned FlatRig torch runtime.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable whose environment will be updated.",
    )
    parser.add_argument(
        "--version",
        default=os.environ.get("FLATRIG_TORCH_VERSION", DEFAULT_TORCH_VERSION),
        help="Torch version to install.",
    )
    parser.add_argument(
        "--flavor",
        default=os.environ.get("FLATRIG_TORCH_FLAVOR", "auto"),
        help="Torch wheel flavor: auto, default, cpu, cu128.",
    )
    parser.add_argument(
        "--index-url",
        default=os.environ.get("FLATRIG_TORCH_INDEX_URL"),
        help="Optional explicit PyTorch wheel index URL.",
    )
    args = parser.parse_args()

    flavor = args.flavor.strip().lower()
    if flavor == "auto":
        flavor = detect_default_flavor()
    index_url = resolve_index_url(flavor, args.index_url)

    pip_install_torch(args.python, args.version, index_url)
    payload = probe_torch(args.python)
    payload["requested_version"] = args.version
    payload["requested_flavor"] = flavor
    payload["index_url"] = index_url
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Torch runtime installation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
