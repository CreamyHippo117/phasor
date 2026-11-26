# Pending: a scene script names itself in `meta.modules` (contracts 2 and 3)

Status: **proposal, waiting for the owner's OK.** Not applied. Patch:
`pending/scene_script_in_modules.patch` (against 6909964, the commit before the one that added this file).

## The gap that is left after the worker fix

The owner's wishlist (reverb_la, 2026-11-25): a scene-local type draws from a module-level
registry that the scene script fills. Render workers now run the scene script first when
the render starts from it (`scene._worker_engine`), and every worker is checked against
the parent before its first frame (`core.record.census`). Both are built and need no
contract change: the worker only redoes what its parent process did.

What is still wrong is the document. `meta.modules` names the module that defines the type
(`progs.py`), not the script that fills its registry (`scene.py`). So the saved
`scene.phasor.json`, opened anywhere with `--allow-scripts` (one process or many, `phasor
inspect`, the MCP server, the app), loads a fresh `progs.py` with an empty registry, and
the object draws nothing, with no lint issue. Workers now fail loudly on it; one process
is silently wrong. Under contract 3, `state_at` depends only on the document, `t`, the
oracle and the files the document names. The script is code the drawing depends on that
the document does not name.

## Proposal

When `phasor.load` runs a script whose document already names scene code
(`meta.modules` is not empty), it adds the script itself to `meta.modules`, last. Opening
the saved document with `--allow-scripts` then loads `progs.py` and then runs `scene.py`,
which fills the registry (the script's own `Scene` is built and dropped, as it is today
when a type is defined in the script itself: `examples/open_contract.py`). It is listed
last because the script imports the type's module and registers the type again from the
same file, which `register` allows; listed first, `progs.py` would run after it and
replace the class with one whose registry is empty.

Scenes without code of their own are unchanged (no `meta.modules`, nothing added).

## Why it needs the owner

- Contract 2: saving a script scene now writes one more entry in `meta.modules` for every
  scene with its own types; existing saved documents change on their next save.
- Contract 3: it states that the script is one of "the code modules inside the scene's
  folder" a document names, which is what the drawing depends on.
- Cost: opening such a saved document runs the whole script (narration timing, any
  simulation set-up it does), not just the type's module. That is the price of being right.

## Before / after

Same scene as `tests/test_workers.py::registry_scene` (scratch copy), saved and then
rendered from the `.phasor.json` with `--allow-scripts --workers 1`, frame at t = 0:

- before: `pending/scene_script_in_modules_before.png`: the equation text only.
- after: `pending/scene_script_in_modules_after.png`: the two orange wedges as well.
