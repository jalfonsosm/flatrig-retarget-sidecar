"""Clone and bootstrap a local PAN checkout for flatRig."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKOUT_DIR = ROOT_DIR / "workflow" / "external" / "PAN"
PAN_REPO_URL = "https://github.com/hlcdyy/pan-motion-retargeting.git"


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
    run(["git", "clone", "--depth", "1", PAN_REPO_URL, str(path)])


def resolve_python(preferred: str | None) -> str:
    candidates = [preferred] if preferred else []
    candidates.extend(["python3.11", "python3.10", "python3.9", "python3.8", "python3"])
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("No suitable Python interpreter was found. Install python3.8+.")


def ensure_venv(checkout_dir: Path, python_executable: str) -> Path:
    venv_dir = checkout_dir / ".venv"
    python_path = venv_dir / "bin" / "python"
    if python_path.exists():
        return python_path
    run([python_executable, "-m", "venv", str(venv_dir)])
    return python_path


def install_deps(python_path: Path, checkout_dir: Path) -> None:
    run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python_path), "-m", "pip", "install", "-e", str(checkout_dir)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap PAN for flatRig.")
    parser.add_argument(
        "--checkout",
        default=str(DEFAULT_CHECKOUT_DIR),
        help="Directory where PAN will be cloned.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable to use for the dedicated venv.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install the PAN package and its Python dependencies into checkout/.venv.",
    )
    args = parser.parse_args()

    checkout_dir = Path(args.checkout).expanduser().resolve()
    python_executable = resolve_python(args.python)
    ensure_checkout(checkout_dir)
    venv_python = ensure_venv(checkout_dir, python_executable)

    if args.install_deps:
        install_deps(venv_python, checkout_dir)

    print(f"PAN checkout: {checkout_dir}")
    print(f"PAN venv python: {venv_python}")
    print(f"Set FLATRIG_PAN_DIR={checkout_dir}")
    print(f"Set FLATRIG_PAN_PYTHON={venv_python}")
    print()
    print("Important:")
    print("- Upstream PAN expects Python 3.8 and a CUDA-capable GPU for the official setup.")
    print(
        "- Upstream also expects PyTorch 1.10.0 to be installed manually inside that environment."
    )
    print(
        "- To run Mixamo retargeting you still need pretrained_mixamo/ and the preprocessed Mixamo assets"
    )
    print("  under data_preprocess/Mixamo/Mixamo/ (train/test lists, mean_var, std_bvhs).")
    print("- Those assets are not bundled by this installer.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PAN bootstrap failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
