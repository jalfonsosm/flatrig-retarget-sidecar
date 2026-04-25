from __future__ import annotations

import json
import sys

from flatrig_retarget_sidecar.cli import main
from flatrig_retarget_sidecar.motion2motion_retarget import Motion2MotionProbe
from tests.helpers import write_spine_json


def test_probe_command_includes_scene_backend(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "flatrig_retarget_sidecar.cli.probe_motion2motion_backend",
        lambda: Motion2MotionProbe(available=True, detail="ready", metadata={"device": "cpu"}),
    )
    monkeypatch.setattr(
        "flatrig_retarget_sidecar.cli.probe_scene_backend",
        lambda: {"available": True, "detail": "ready", "mode": "bpy_module"},
    )
    monkeypatch.setattr(sys, "argv", ["flatrig-retarget-sidecar", "probe"])

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["available"] is True
    assert payload["detail"] == "ready"
    assert payload["backend"] == "motion2motion"
    assert payload["scene_backend"]["mode"] == "bpy_module"


def test_spine_to_json_command_writes_normalized_payload(tmp_path, monkeypatch, capsys) -> None:
    source = write_spine_json(tmp_path / "hero.json")
    output = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["flatrig-retarget-sidecar", "spine-to-json", str(source), "--output", str(output)],
    )

    main()

    summary = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["bone_count"] == 4
    assert "bones" in written
