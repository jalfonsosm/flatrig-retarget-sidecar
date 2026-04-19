"""PAN integration helpers for cross-rig retargeting."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flatrig_retarget_sidecar.spine_import import ROOT_DIR, SpinePackage

PAN_LABEL = "PAN"
DEFAULT_PAN_DIR = ROOT_DIR / "workflow" / "external" / "PAN"
PAN_PAIR_SCRIPT = "eval_single_pair.py"
ENV_PAN_DIR = "FLATRIG_PAN_DIR"
ENV_PAN_PYTHON = "FLATRIG_PAN_PYTHON"
ENV_PAN_DEVICE = "FLATRIG_PAN_DEVICE"
ENV_PAN_MODEL_DIR = "FLATRIG_PAN_MODEL_DIR"


@dataclass(slots=True)
class PanProbe:
    available: bool
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PanBvhRetargetResult:
    output_bvh: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def resolve_pan_dir() -> Path:
    raw = os.environ.get(ENV_PAN_DIR)
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_PAN_DIR.resolve()


def resolve_pan_python(pan_dir: Path | None = None) -> Path | None:
    raw = os.environ.get(ENV_PAN_PYTHON)
    if raw:
        candidate = Path(raw).expanduser().resolve()
        if candidate.exists():
            return candidate

    checkout_dir = pan_dir or resolve_pan_dir()
    venv_python = checkout_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python

    for candidate_name in ("python3.11", "python3.10", "python3.9", "python3.8", "python3"):
        executable = shutil.which(candidate_name)
        if executable:
            return Path(executable)
    return None


def _probe_torch_runtime(python_executable: Path, cwd: Path) -> dict[str, Any] | None:
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            (
                "import json, torch; "
                "payload = {"
                "'cuda': bool(getattr(torch, 'cuda', None) and torch.cuda.is_available()), "
                "'mps': bool(("
                "getattr(getattr(torch, 'backends', None), 'mps', None) "
                "and torch.backends.mps.is_available()"
                ") or (getattr(torch, 'mps', None) and torch.mps.is_available()))"
                "}; "
                "print(json.dumps(payload))"
            ),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def resolve_pan_device(
    pan_dir: Path | None = None,
    python_executable: Path | None = None,
) -> str:
    raw = os.environ.get(ENV_PAN_DEVICE)
    if raw:
        return str(raw).strip()

    checkout_dir = pan_dir or resolve_pan_dir()
    resolved_python = python_executable or resolve_pan_python(checkout_dir)
    if resolved_python is not None:
        runtime = _probe_torch_runtime(resolved_python, checkout_dir)
        if runtime:
            if runtime.get("cuda"):
                return "cuda:0"
            if runtime.get("mps"):
                return "mps"

    return "cuda:0" if shutil.which("nvidia-smi") else "cpu"


def resolve_pan_model_dir(pan_dir: Path | None = None) -> Path:
    raw = os.environ.get(ENV_PAN_MODEL_DIR)
    if raw:
        return Path(raw).expanduser().resolve()
    checkout_dir = pan_dir or resolve_pan_dir()
    return (checkout_dir / "pretrained_mixamo").resolve()


def _pan_pythonpath_entries(pan_dir: Path) -> list[str]:
    return [
        str(ROOT_DIR),
        str(pan_dir.resolve()),
    ]


def _required_mixamo_paths(pan_dir: Path, model_dir: Path) -> dict[str, Path]:
    mixamo_root = pan_dir / "data_preprocess" / "Mixamo" / "Mixamo"
    return {
        "pair_script": pan_dir / PAN_PAIR_SCRIPT,
        "setup_py": pan_dir / "setup.py",
        "model_para": model_dir / "para.txt",
        "model_models": model_dir / "models",
        "mixamo_test_list": mixamo_root / "test_list.txt",
        "mixamo_train_list": mixamo_root / "train_list.txt",
        "mixamo_mean_var": mixamo_root / "mean_var",
        "mixamo_std_bvhs": mixamo_root / "std_bvhs",
    }


def probe_pan_backend() -> PanProbe:
    pan_dir = resolve_pan_dir()
    if not pan_dir.exists():
        return PanProbe(
            available=False,
            detail=(
                f"PAN checkout not found at {pan_dir}. "
                "Run tools/install_pan_backend.py or set FLATRIG_PAN_DIR."
            ),
            metadata={"pan_dir": str(pan_dir)},
        )

    model_dir = resolve_pan_model_dir(pan_dir)
    required_paths = _required_mixamo_paths(pan_dir, model_dir)
    missing = [label for label, path in required_paths.items() if not path.exists()]

    python_executable = resolve_pan_python(pan_dir)
    if python_executable is None:
        return PanProbe(
            available=False,
            detail="No compatible Python executable was found for the PAN backend.",
            metadata={"pan_dir": str(pan_dir)},
        )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            *_pan_pythonpath_entries(pan_dir),
            *(entry for entry in existing_pythonpath.split(os.pathsep) if entry),
        ]
    )
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import torch, numpy, scipy, sklearn, tensorboard, torchsummary; print('ok')",
        ],
        cwd=pan_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        if "No module named 'torch'" in stderr:
            stderr = (
                f"{stderr}. Install PyTorch into the PAN environment first "
                "(upstream recommends PyTorch 1.10.0 on Python 3.8)."
            )
        return PanProbe(
            available=False,
            detail=(
                "PAN checkout was found, but runtime dependencies are missing. "
                f"Probe error: {stderr}"
            ),
            metadata={
                "pan_dir": str(pan_dir),
                "python": str(python_executable),
                "model_dir": str(model_dir),
            },
        )

    if missing:
        detail = (
            "PAN checkout and Python dependencies were detected, but the Mixamo runtime assets are incomplete. "
            "This backend needs the pretrained Mixamo checkpoint plus preprocessed Mixamo data "
            "(test/train lists, std_bvhs, mean_var). Missing: " + ", ".join(missing)
        )
        return PanProbe(
            available=False,
            detail=detail,
            metadata={
                "pan_dir": str(pan_dir),
                "python": str(python_executable),
                "model_dir": str(model_dir),
                "missing": missing,
                "device": resolve_pan_device(),
            },
        )

    return PanProbe(
        available=True,
        detail=(
            "PAN checkout, Mixamo runtime assets, and Python dependencies were detected. "
            "The backend is ready for Mixamo/BVH pair retargeting."
        ),
        metadata={
            "pan_dir": str(pan_dir),
            "python": str(python_executable),
            "model_dir": str(model_dir),
            "device": resolve_pan_device(),
        },
    )


def retarget_mixamo_bvh_pair(
    input_bvh: str | Path,
    target_bvh: str | Path,
    *,
    output_bvh: str | Path | None = None,
    test_type: str = "cross",
    epoch: int = 1000,
) -> PanBvhRetargetResult:
    probe = probe_pan_backend()
    if not probe.available:
        raise RuntimeError(f"{PAN_LABEL} backend unavailable: {probe.detail}")

    pan_dir = resolve_pan_dir()
    model_dir = resolve_pan_model_dir(pan_dir)
    python_executable = resolve_pan_python(pan_dir)
    if python_executable is None:
        raise RuntimeError(
            "PAN backend probe passed, but the Python executable could not be resolved."
        )

    input_path = Path(input_bvh).expanduser().resolve()
    target_path = Path(target_bvh).expanduser().resolve()
    if output_bvh is None:
        output_path = target_path.with_name(target_path.stem + "_pan_retarget.bvh")
    else:
        output_path = Path(output_bvh).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            *_pan_pythonpath_entries(pan_dir),
            *(entry for entry in existing_pythonpath.split(os.pathsep) if entry),
        ]
    )
    env[ENV_PAN_DIR] = str(pan_dir)
    env[ENV_PAN_PYTHON] = str(python_executable)
    env[ENV_PAN_DEVICE] = resolve_pan_device()
    env[ENV_PAN_MODEL_DIR] = str(model_dir)

    command = [
        str(python_executable),
        str((pan_dir / PAN_PAIR_SCRIPT).resolve()),
        f"--input_bvh={input_path}",
        f"--target_bvh={target_path}",
        f"--test_type={str(test_type or 'cross')}",
        f"--output_filename={output_path}",
        f"--model_dir={model_dir}",
        f"--epoch={int(epoch)}",
    ]
    completed = subprocess.run(
        command,
        cwd=pan_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"{PAN_LABEL} pair retarget failed: {stderr}")
    if not output_path.exists():
        raise RuntimeError(
            f"{PAN_LABEL} finished without producing the expected BVH output: {output_path}"
        )
    return PanBvhRetargetResult(
        output_bvh=str(output_path),
        diagnostics={
            "backend_label": PAN_LABEL,
            "input_bvh": str(input_path),
            "target_bvh": str(target_path),
            "output_bvh": str(output_path),
            "test_type": str(test_type or "cross"),
            "epoch": int(epoch),
            "model_dir": str(model_dir),
            "device": resolve_pan_device(),
        },
    )


def retarget_spine_animation(
    source: SpinePackage,
    target: SpinePackage,
    animation_name: str,
):
    del source, target, animation_name
    raise NotImplementedError(
        "PAN is now the selected cross-rig backend, but the current adapter only exists for "
        "Mixamo/BVH 3D pair retargeting. A Spine 2D -> PAN -> Spine 2D bridge is not wired yet."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the PAN backend.")
    parser.add_argument(
        "command",
        nargs="?",
        default="probe",
        choices=["probe"],
        help="Only backend probing is exposed from this CLI wrapper.",
    )
    args = parser.parse_args()
    if args.command == "probe":
        print(json.dumps(asdict(probe_pan_backend()), indent=2))


if __name__ == "__main__":
    main()
