"""Clone and bootstrap a local Motion2Motion checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKOUT_DIR = ROOT_DIR / "workflow" / "external" / "Motion2Motion_codes"
M2M_REPO_URL = "https://github.com/LinghaoChan/Motion2Motion_codes.git"
TORCH_INSTALLER = ROOT_DIR / "tools" / "install_torch_runtime.py"
DEFAULT_SHARED_VENV_DIR = ROOT_DIR / ".venv"


def venv_python_path(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{stderr}")


def ensure_checkout(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", M2M_REPO_URL, str(path)])


def resolve_python(preferred: str | None) -> str:
    candidates = [preferred] if preferred else []
    candidates.extend(["python3.10", "python3.11", "python3.12", "python3.9", "python3"])
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("No suitable Python interpreter was found. Install python3.10-3.12.")


def resolve_venv_dir(checkout_dir: Path, python_executable: str, dedicated: bool) -> Path:
    if not dedicated:
        return DEFAULT_SHARED_VENV_DIR
    executable_name = Path(python_executable).name
    if "3.12" in executable_name:
        return checkout_dir / ".venv312"
    return checkout_dir / ".venv"


def ensure_venv(checkout_dir: Path, python_executable: str, dedicated: bool) -> Path:
    venv_dir = resolve_venv_dir(checkout_dir, python_executable, dedicated)
    python_path = venv_python_path(venv_dir)
    if python_path.exists():
        return python_path
    run([python_executable, "-m", "venv", str(venv_dir)])
    return python_path


def install_deps(python_path: Path, checkout_dir: Path) -> None:
    run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python_path), "-m", "pip", "install", "-r", str(checkout_dir / "requirements.txt")])
    if TORCH_INSTALLER.exists():
        run([str(python_path), str(TORCH_INSTALLER), "--python", str(python_path)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Motion2Motion.")
    parser.add_argument(
        "--checkout",
        default=str(DEFAULT_CHECKOUT_DIR),
        help="Directory where Motion2Motion will be cloned.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable to use for the dedicated venv.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install Motion2Motion requirements into the dedicated venv.",
    )
    parser.add_argument(
        "--dedicated-venv",
        action="store_true",
        help="Use a Motion2Motion-local virtual environment instead of the shared sidecar .venv.",
    )
    args = parser.parse_args()

    checkout_dir = Path(args.checkout).expanduser().resolve()
    python_executable = resolve_python(args.python)
    ensure_checkout(checkout_dir)
    venv_python = ensure_venv(checkout_dir, python_executable, args.dedicated_venv)

    if args.install_deps:
        install_deps(venv_python, checkout_dir)

    print(f"Motion2Motion checkout: {checkout_dir}")
    print(f"Motion2Motion venv python: {venv_python}")
    print(f"Set FLATRIG_M2M_DIR={checkout_dir}")
    print(f"Set FLATRIG_M2M_PYTHON={venv_python}")
    print()
    print("Notes:")
    print("- The default path reuses the sidecar root .venv so callers share one Torch runtime.")
    print(
        "- Pass --dedicated-venv if you explicitly want a Motion2Motion-local environment instead."
    )
    print(
        "- Device auto-selection now prefers CUDA, then MPS, then CPU based on the torch runtime in that venv."
    )
    print(
        "- Upstream recommends CPU execution for lightweight interactive runs, but the sidecar will use acceleration when torch exposes it."
    )
    print(
        "- If auto-detection is wrong for your machine, override it with FLATRIG_M2M_DEVICE=cuda, mps or cpu."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Motion2Motion bootstrap failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
