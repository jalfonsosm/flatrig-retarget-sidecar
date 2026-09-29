import sys
from pathlib import Path
from types import SimpleNamespace

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from blender_io.bvh_scene import _sample_action_frames  # noqa: E402


def _scene(*, fps=24, fps_base=1.0, frame_start=1, frame_end=25):
    return SimpleNamespace(
        render=SimpleNamespace(fps=fps, fps_base=fps_base),
        frame_start=frame_start,
        frame_end=frame_end,
    )


def test_sample_action_frames_uses_scene_fps_and_requested_range():
    action = SimpleNamespace(frame_range=(10.0, 22.0))

    assert _sample_action_frames(action, _scene(fps=24), fps=12.0) == [
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        22.0,
    ]


def test_sample_action_frames_clamps_inverted_range_to_two_frames():
    action = SimpleNamespace(frame_range=(10.0, 20.0))

    assert _sample_action_frames(
        action,
        _scene(fps=30),
        fps=30.0,
        frame_start=12,
        frame_end=8,
    ) == [12.0, 13.0]
