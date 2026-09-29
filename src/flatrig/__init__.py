"""Public Blender (``bpy``) worker sidecar for FlatRig.

Holds the Blender-bound worker code (scene/animation/mesh extraction, rendering,
format export) and its CLI. FlatRig-specific algorithms live privately in the
``flatrig_private`` package and are re-exported by the worker modules.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
