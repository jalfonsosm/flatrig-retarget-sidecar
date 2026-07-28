import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from blender_io.bone_hierarchy import _annotate_bone_topology  # noqa: E402


def test_bone_topology_marks_main_and_terminal_chains():
    records = [
        {"name": "root", "parent": None, "length": 10.0},
        {"name": "spine", "parent": "root", "length": 9.0},
        {"name": "neck", "parent": "spine", "length": 8.0},
        {"name": "arm", "parent": "spine", "length": 0.5},
        {"name": "finger", "parent": "arm", "length": 0.4},
        {"name": "tip", "parent": "finger", "length": 0.3},
    ]

    _annotate_bone_topology(records)

    by_name = {record["name"]: record for record in records}
    assert {name for name, record in by_name.items() if record["main_chain"]} == {
        "root",
        "spine",
        "neck",
    }
    assert [by_name[name]["terminal_chain_order"] for name in ("arm", "finger", "tip")] == [
        0,
        1,
        2,
    ]
    assert by_name["arm"]["terminal_chain_root"] is True
    assert by_name["finger"]["inherit"] == "NoScale"
    assert by_name["neck"]["inherit"] == "Normal"
