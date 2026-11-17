"""Engine: owns the document, the evaluated-state cache, the layout bake and the inspector.

SPEC contract 1. Every mutation goes through ``Engine.apply(patch) -> PatchResult``;
there is no other write path. Implemented in Milestone 1.
"""
