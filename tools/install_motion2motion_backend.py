"""Clone and bootstrap a local Motion2Motion checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
M2M_CHECKOUT_DIR = ROOT_DIR / "workflow" / "external" / "Motion2Motion_codes"
M2M_REPO_URL = "https://github.com/LinghaoChan/Motion2Motion_codes.git"
TORCH_INSTALLER = ROOT_DIR / "tools" / "install_torch_runtime.py"
SHARED_VENV_DIR = ROOT_DIR / ".venv"


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


def ensure_shared_venv(python_executable: str) -> Path:
    python_path = venv_python_path(SHARED_VENV_DIR)
    if python_path.exists():
        return python_path
    run([python_executable, "-m", "venv", str(SHARED_VENV_DIR)])
    return python_path


def install_deps(python_path: Path, checkout_dir: Path) -> None:
    run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python_path), "-m", "pip", "install", "-e", f"{ROOT_DIR}[motion2motion]"])
    run([str(python_path), "-m", "pip", "install", "-r", str(checkout_dir / "requirements.txt")])
    if TORCH_INSTALLER.exists():
        run([str(python_path), str(TORCH_INSTALLER)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Motion2Motion.")
    parser.add_argument(
        "--python",
        required=True,
        help="Python executable to use for creating the venv and installing dependencies.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install sidecar and Motion2Motion requirements into the shared .venv.",
    )
    args = parser.parse_args()

    checkout_dir = M2M_CHECKOUT_DIR.resolve()
    python_executable = args.python
    ensure_checkout(checkout_dir)
    venv_python = ensure_shared_venv(python_executable)

    if args.install_deps:
        install_deps(venv_python, checkout_dir)

    print(f"Motion2Motion checkout: {checkout_dir}")
    print(f"Shared venv: {SHARED_VENV_DIR.resolve()}")
    print(f"Shared Python: {venv_python}")
    print()
    print("Notes:")
    print("- The sidecar and host client use the sidecar root .venv as the single shared runtime.")
    print("- Device auto-selection prefers CUDA, then CPU based on the torch runtime in that venv.")
    print(
        "- Upstream recommends CPU execution for lightweight interactive runs, "
        "but the sidecar will use acceleration when torch exposes it."
    )
    print(
        "- If auto-detection is wrong for your machine, "
        "override it with FLATRIG_M2M_DEVICE=cuda, mps or cpu."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Motion2Motion bootstrap failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
