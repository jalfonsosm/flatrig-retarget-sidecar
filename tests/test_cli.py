from __future__ import annotations

import json
import sys

from flatrig.cli import main


def test_probe_command_reports_blender_worker(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "flatrig.cli.probe_scene_backend",
        lambda: {"available": True, "detail": "ready", "mode": "bpy_module"},
    )
    monkeypatch.setattr(sys, "argv", ["flatrig-retarget-sidecar", "probe"])

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "blender_worker"
    assert payload["scene_backend"]["mode"] == "bpy_module"
    assert payload["scene_backend"]["available"] is True
