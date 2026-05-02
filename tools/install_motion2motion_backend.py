"""Clone and bootstrap a local Motion2Motion checkout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
M2M_CHECKOUT_DIR = ROOT_DIR / "workflow" / "external" / "Motion2Motion_codes"
M2M_REPO_URL = "https://github.com/LinghaoChan/Motion2Motion_codes.git"
TORCH_INSTALLER = ROOT_DIR / "tools" / "install_torch_runtime.py"
SHARED_VENV_DIR = ROOT_DIR / ".venv"
REQUIRED_PYTHON = (3, 11)


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


def python_major_minor(python_executable: str | Path) -> tuple[int, int]:
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Could not query Python version for {python_executable}\n{stderr}")
    major, minor = completed.stdout.strip().split(".", 1)
    return int(major), int(minor)


def ensure_supported_python(python_executable: str | Path) -> None:
    version = python_major_minor(python_executable)
    if version != REQUIRED_PYTHON:
        required = ".".join(str(part) for part in REQUIRED_PYTHON)
        actual = ".".join(str(part) for part in version)
        raise RuntimeError(
            f"Motion2Motion sidecar requires Python {required}; got Python {actual} "
            f"from {python_executable}"
        )


def ensure_checkout(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", M2M_REPO_URL, str(path)])


def requirement_name(line: str) -> str:
    requirement = line.split("#", 1)[0].strip()
    if not requirement or requirement.startswith(("-", "git+", "http:", "https:")):
        return ""
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", ";"):
        requirement = requirement.split(separator, 1)[0]
    return requirement.strip().lower().replace("_", "-")


def filtered_m2m_requirements(source: Path) -> Path:
    blocked = {"numpy", "torch"}
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="flatrig-m2m-requirements-",
        suffix=".txt",
        delete=False,
    ) as handle:
        output_path = Path(handle.name)
        for line in source.read_text(encoding="utf-8").splitlines():
            if requirement_name(line) in blocked:
                continue
            handle.write(f"{line}\n")
    return output_path


def ensure_shared_venv(python_executable: str) -> Path:
    ensure_supported_python(python_executable)
    python_path = venv_python_path(SHARED_VENV_DIR)
    if python_path.exists():
        try:
            if python_major_minor(python_path) == REQUIRED_PYTHON:
                return python_path
        except RuntimeError:
            pass
        shutil.rmtree(SHARED_VENV_DIR)
    if python_path.exists():
        return python_path
    run([python_executable, "-m", "venv", str(SHARED_VENV_DIR)])
    return python_path


def install_deps(python_path: Path, checkout_dir: Path) -> None:
    print("[install_motion2motion_backend] Upgrading pip...")
    run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    print("[install_motion2motion_backend] Installing sidecar package...")
    run([
        str(python_path), "-m", "pip", "install", "-e",
        f"{ROOT_DIR}[motion2motion]"
    ])
    print("[install_motion2motion_backend] Installing M2M requirements...")
    filtered_requirements = filtered_m2m_requirements(checkout_dir / "requirements.txt")
    try:
        run([
            str(python_path), "-m", "pip", "install", "-r",
            str(filtered_requirements)
        ])
    finally:
        filtered_requirements.unlink(missing_ok=True)
    if TORCH_INSTALLER.exists():
        print("[install_motion2motion_backend] Installing torch runtime...")
        run([str(python_path), str(TORCH_INSTALLER)])
    
    # Install Spine binary runtime node_modules if Node.js is available
    spine_runtime_dir = ROOT_DIR / "workflow" / ".spine_binary_runtime"
    node_executable = shutil.which("node")
    npm_executable = shutil.which("npm")
    if node_executable and npm_executable:
        print("[install_motion2motion_backend] Installing Spine runtime")
        spine_runtime_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = spine_runtime_dir / "package.json"
        if not manifest_path.exists():
            manifest_content = json.dumps({
                "name": "flatrig-spine-binary-runtime",
                "private": True,
                "dependencies": {
                    "@pixi-spine/runtime-3.8": "4.0.6",
                    "@pixi-spine/runtime-4.0": "4.0.6",
                    "@pixi-spine/runtime-4.1": "4.0.6",
                    "@esotericsoftware/spine-core": "4.2.106",
                },
            }, indent=2) + "\n"
            manifest_path.write_text(manifest_content, encoding="utf-8")
        run([
            npm_executable, "install", "--no-audit", "--no-fund"
        ], cwd=spine_runtime_dir)
    else:
        print("[install_motion2motion_backend] WARNING: Node.js not found, "
              "Spine binary runtime will need npm install at runtime")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Motion2Motion.")
    parser.add_argument(
        "--python",
        required=True,
        help="Python executable to use for creating the venv.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install requirements into the shared .venv.",
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
    print("- The sidecar and host client share .venv as the single runtime.")
    print("- Device auto-selection prefers CUDA, then CPU.")
    print("    - Upstream recommends CPU for lightweight runs")
    print("- If auto-detection is wrong, set FLATRIG_M2M_DEVICE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Motion2Motion bootstrap failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
