# ADR 0006: updaters defined interactively are saved as a copy

Status: **Proposed**, built on branch `claude/notebook-updaters` for the owner's review. It
changes no SPEC contract. The owner's idea (2026-11-24): "notebook cases can be saved as a
copy with the appropriate changes". The alternative the owner refused (session-only
references that open elsewhere as `missing_type`) is not built.

## Context

`obj.always(fn)` (ADR 0005) needs `fn` in a `.py` file in the scene's folder, which the
document names (`meta.modules`, `always: ["file.py:fn"]`). That is contract 3: the state
depends only on the document, `t` and files the document names inside the folder. A function
defined in a notebook cell or at a prompt has no such file, so `add` refused it, and the
notebook could not use `always` at all.

## Decision

**An interactive updater is held by the `Scene` builder, not by the document, until
`Scene.save_copy(path)` writes its source into a module beside the copy.**

1. **What counts as interactive.** A plain function whose `co_filename` names no file on disk
   (`<ipython-input-..>`, `/tmp/ipykernel_N/M.py` that only `linecache` knows, `<stdin>`,
   `exec`). A function from a real file stays under the old rule (in the folder: named;
   elsewhere: refused). Nothing else changes for scripts, the MCP server or the GUI.
2. **How the source is captured.** `inspect.getsource`, which reads `linecache`. IPython,
   Jupyter and the Python 3.13 prompt put each cell's text there, so it works; a plain
   `python` prompt keeps nothing, and `getsource` raises. That case is refused with the fix
   ("define it in an IPython or Jupyter cell, or write it in a .py file in the scene's
   folder"). Capture happens in `Scene.add`, so the source is the one at that moment (a cell
   edited and run again later does not change what was added) and a refusal comes at once,
   before the object is added.
3. **What may be captured (small and checked).** A top-level `def` (no lambda, decorator,
   method, nested function, closure, `global`, `nonlocal`) whose other names are builtins or
   one of: a module (`import numpy as np`), a function or class importable by name from a
   module that is not `__main__` (`from math import sqrt`), a constant of plain data (number,
   string, bytes, `None`, tuple or frozenset of those, written as the literal it has when
   captured). Everything else is refused, naming the thing and the fix: a list or array from
   the notebook, another notebook function, a class from `__main__`, a module that cannot
   be imported by name, an undefined name. Names are found with `symtable` (comprehensions,
   defaults and annotations included), not by guessing. The saved module is checked to parse
   and to hold only imports, constants and defs; it is never executed to check it.
4. **Where it is written.** Beside the copy's document, `name.updaters.py`; the document
   names it in `meta.modules` and the objects' `always` (`set_always`, a new patch op, applied
   to a private engine that runs no scene code, so contract 1 holds). A file is never
   overwritten: identical content is reused, different content gets the next free name
   (`name.updaters2.py`). Modules the document already names are copied into a new folder
   under the same rule; a differing file there stops the copy before anything is written.
   Files are bytes, UTF-8, `\n`. Everything is decided before the first write, so a refusal
   leaves nothing behind. The copy's `.phasor.json` is replaced like any save to that path.
5. **The open scene.** It does not change: its document has the object without the updater, it
   does not run it, and `add` warns (`HeldUpdaterWarning`) that the copy does. `save_copy`
   returns the copy, opened with scripts allowed (the person asked for it), and the notebook
   continues with that. `Scene.save` refuses while updaters are held, so a plain save cannot
   drop them silently (`Engine.save` on the live engine saves exactly the live document).
6. **Contract 3.** The copy's state depends on its document, `t` and `name.updaters.py`
   inside its folder, like any scene-local updater. `lint(strict=True)` checks purity
   (`impure_updater`) on the copy like any other.
7. **Errors, not lint.** There is no new lint code: the document never names a held updater,
   so there is nothing to lint. `UpdaterCaptureError` (a `ValueError`) names what is wrong and
   what to do, from `add` or `save_copy`.

## Consequences

- Good: a notebook can use `always`; the saved copy is reproducible and contract 3 clean; no
  session-only reference exists anywhere.
- Cost: the live scene does not show a held updater (the copy does); a notebook works with
  `s = s.save_copy(...)`. Module imports are by name, so the copy needs the same packages
  (and, for a module found on the notebook's own `sys.path`, the same path) where it runs.
- Not built: capturing helpers the updater calls, a plain-prompt source argument, writing
  into an existing user module.
- Untested on Windows, macOS and CUDA; real IPython/Jupyter is imitated by compiling under a
  cell name with the text in `linecache` (what both do), not run.
