# 13. The desktop app after Milestone 25: rebuild it and try every button on Windows 11

What the cloud box cannot do: build the Windows package, click through the real window on a
real screen, hear the sounds, and play the drafts in the Windows player. Short and exact;
write the results at the bottom (the template is there).

Branch: the Milestone 25 branch once merged into `claude/eloquent-mayer-jiqxs9`. Folder:
`C:\Users\Atlas\phasor-dev`.

## A. Tests first (PowerShell)

```powershell
cd C:\Users\Atlas\phasor-dev
git pull
uv sync --extra gl --extra encode --extra gui --extra mcp --extra gpu --extra boolean --extra svg --extra code --group package
uv run phasor doctor
uv run pytest -q tests/test_tweaks_block.py tests/test_render_jobs.py tests/test_app_code_first.py tests/test_app_window.py
uv run pytest -q tests/test_gui_controller.py tests/test_gui_window.py tests/test_desktop.py tests/test_m10_exit_demo.py
```

Expected: all pass (the Linux box: see `MILESTONE_25.md`). Write down any failure in full.

## B. Rebuild the packaged app

```powershell
$env:PHASOR_PACKAGE_GPU = "1"
uv run pyinstaller packaging/phasor.spec --noconfirm
$env:PHASOR_TEST_PACKAGE = "1"; uv run pytest -q tests/test_packaging.py; Remove-Item Env:PHASOR_TEST_PACKAGE
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Programs\phasor" -ErrorAction SilentlyContinue
Copy-Item -Recurse dist\phasor "$env:LOCALAPPDATA\Programs\phasor"
& "$env:LOCALAPPDATA\Programs\phasor\phasor.exe" --install-shortcut
```

Start phasor from the Start menu (not from the terminal) for everything below. A console
window must never appear or flash.

## C. The manual checklist (the packaged app)

Make a folder `Documents\phasor\m25` and copy into it, from the repo's `examples\assets\`:
`clip.mp4`, `narration_demo.wav`, `motif.mid`, and `examples\narrated_from_audio.py`.
Also save this as `Documents\phasor\m25\try.py`:

```python
from phasor import Dot, Scene, Text

s = Scene("m25 try")
s.add(Dot(id="pole", pos=(1, 0, 0), r=0.2))
s.add(Text(id="note", text="hello", pos=(0, 1, 0)))
s.marker("go", 1.0)
s.marker("end", 4.0)
```

1. **Trust asks once.** File > Open `try.py`. Expected: "Run this script?" appears; answer
   No: nothing opens. Open again, Yes: the window opens. Close it, open it a third time: no
   question.
2. **The layout.** The script on the left, the preview and timeline on the right, tabs
   Problems, Inspector, Looks, Narration, Sound and MIDI, Footage, Sliders, AI.
3. **An error line.** Add `s.add(Dot(id="bad", r=-1))` at the end and press Ctrl+Enter.
   Expected: line 8 marked red in the script, "not run: line 8 ..." under it, the error first
   in Problems, the preview unchanged. Delete the line, Ctrl+Enter: it runs.
4. **Select and line.** Click the dot in the preview: line 4 is marked. Click line 5 in the
   script: the text is selected (inspector shows `note`).
5. **A tweak goes into the script.** Select the dot, set `r` to 0.4 in the inspector.
   Expected: a block `# --- phasor tweaks ...` with `s.tweak('pole.r', 0.4)` appears at the
   end of the script, the title shows `•`. Ctrl+S: saved, `•` gone. Ctrl+Z: the line goes.
6. **An outside edit.** With nothing unsaved, open `try.py` in Notepad, change `"hello"` to
   `"from notepad"`, save. Expected within a second: the app reloads and the preview shows
   the new text, "reloaded" in the status bar.
7. **A conflict.** Type something in the app's script (do not save). Change the file in
   Notepad again and save. Expected: the app asks "The script changed on disk" with "Keep
   mine" / "Take the file's"; try both (once each). Nothing is merged.
8. **Draft preview.** Press "Draft preview" (Ctrl+Shift+P). Expected: a progress bar in the
   status bar, the window stays usable, `try.draft.mp4` plays in the Windows player when done.
   Write down the seconds it took and the file size.
9. **Cancel a draft.** Make the scene longer (`s.marker("end", 60.0)`, Ctrl+Enter), press
   Draft preview, then "Cancel render" half-way. Expected: the bar goes, "nothing was left
   behind"; no `try.draft.partial.mp4` in the folder; the earlier `try.draft.mp4` still plays.
10. **Render with a quality.** Scene > Render... Pick Final: the size shows the scene's
    (1920 x 1080) and 60 frames a second; pick Draft: 640 x 360, 15, and the name
    `try.draft.mp4`. Render Preview to `try.mp4`; then render Final to the same name: it asks
    to replace; say Yes, then Cancel half-way. Expected: `try.mp4` (the preview) still plays,
    no partial file.
11. **Footage.** Footage tab, "as the background" ticked, add `clip.mp4`. Expected: a line
    `s.background('clip.mp4', id='background')` in the script; the clip fills the frame behind
    the dot when you scrub. Add a PNG from outside the folder: it is copied into `media\`.
12. **Sound and MIDI.** Sound and MIDI tab: At = `go`, add `narration_demo.wav` as a sound
    clip; type `C4 0 0.5, E4 0.5 0.5, G4 1 1`, instrument bell, "Add the typed notes"; add
    `motif.mid`. Export the notes to `try.mid` (it asks before replacing). Draft preview:
    **listen**: the sound, the bell notes and the motif all start at 1 s (the marker `go`).
    Open `try.mid` in any MIDI player: the notes are there.
13. **Narration and snap.** Open `narrated_from_audio.py` (copy). Narration tab: "Speech and
    pauses" lists stretches and pauses; "Snap markers to narration: preview" lists moves (or
    says nothing would move, with reasons); if Apply is enabled, apply it: a line
    `s.snap_to_narration()` is added and runs. Draft preview: **listen**: each scene starts
    with its sentence.
14. **An AI on the same scene.** With `try.py` open: in a terminal
    `phasor mcp --attach --root Documents\phasor\m25` from Claude Code; ask it to make the
    dot bigger. Expected: the tweak appears in the script's block. Ask it to `render_start` a
    draft and `render_status`: progress is reported; `render_cancel` leaves nothing.
15. **Without OpenGL** (TESTING.md B1; only if you can: an RDP session or a VM): the preview
    says "No preview here ... the other panels still work"; the inspector and the script work.

16. **A draft open in the player** (audit S12). Play `try.draft.mp4` in the Windows player
    and leave it open; press Draft preview again. Expected: the render finishes as
    `try (2).draft.mp4` and the status bar says the old one could not be replaced.
17. **A runaway script** (audit S7). Add `while True: pass` at the end, Ctrl+Enter.
    Expected: the window stays usable, "Cancel run" appears; press it: "cancelled", the
    preview keeps the last scene. Remove the line.
18. **A broken tweaks block** (audit B1). Make a tweak and save; in Notepad delete the line
    `# --- end phasor tweaks ---` and save; make another tweak in the app and press Ctrl+S.
    Expected: "not saved: the tweaks block has a begin marker but no end marker (line N)" in
    Problems; the file is unchanged (check in Notepad). Put the line back: saving works.
19. **Save after an outside edit** (audit B3). Make a tweak (do not save), change the file in
    Notepad, then press Ctrl+S at once. Expected: the conflict question, never a silent
    overwrite.
20. **A read-only script** (audit S6). In Explorer: Properties > Read-only. Make a tweak,
    Ctrl+S. Expected: "Not saved: ... is read-only"; the app keeps working.
21. **Render outside the folder** (audit S1). Render... > Choose... a file in `Videos`.
    Expected: it renders there.

22. **Special characters** (re-audit B-A). In `try.py` change `"hello"` to `"10 km"` (a
    no-break space: Ctrl+Shift+Space in Word, or paste one) in Notepad and save; open it in the
    app: no `•` in the title; Ctrl+S; the file still has the no-break space (Notepad++: View >
    Show Symbol).
23. **A runaway that catches the cancel** (re-audit S-1). Add at the end:
    `while True:` / `    try:` / `        while True: pass` / `    except BaseException: pass`;
    Ctrl+Enter, then Cancel run. Expected: within about 5 s "the run did not stop ...", the
    file holds the code from before, the window and an attached AI work.
24. **The web export over a page open in a browser** (re-audit S-7): export a web page, open it
    in a browser, export again to the same name: saved as `name (2).html`, said.

## D. Results (fill in)

| # | Check | Expected | Measured / seen | Pass? |
| --- | --- | --- | --- | --- |
| A | focused tests | all pass | | |
| B | package built, `test_packaging.py` | passes, no console window | | |
| 1 | trust asks once | No opens nothing; asked once | | |
| 2 | layout | script left, preview right, 8 tabs | | |
| 3 | error line | line 8 red, preview unchanged | | |
| 4 | select and line | line 4 marked; line 5 selects note | | |
| 5 | tweak into the script | `s.tweak('pole.r', 0.4)` in the block; undo removes it | | |
| 6 | outside edit | reloads within a second | | |
| 7 | conflict | asks; both answers work | | |
| 8 | draft preview | plays when done; seconds: ___ size: ___ | | |
| 9 | cancel a draft | no partial file, old draft plays | | |
| 10 | render quality | sizes/fps right; cancelled final keeps the old file | | |
| 11 | footage | background line; copied into media | | |
| 12 | sound and MIDI | heard at the right times; `.mid` plays | | |
| 13 | narration and snap | snap line added; sentences in sync (heard) | | |
| 14 | AI attached | tweak in the block; job progress and cancel | | |
| 15 | without OpenGL | message; the rest works | | |
| 16 | draft open in a player | `try (2).draft.mp4`, said in the status bar | | |
| 17 | runaway script | window usable; Cancel run keeps the last scene | | |
| 18 | broken tweaks block | not saved, line named; file unchanged | | |
| 19 | save after an outside edit | asks; nothing overwritten | | |
| 20 | read-only script | "read-only", app keeps working | | |
| 21 | render outside the folder | rendered in Videos | | |
| 22 | special characters | opens clean; saved byte for byte | | |
| 23 | runaway catching cancel | let go in ~5 s; file restored | | |
| 24 | web export over an open page | `name (2).html`, said | | |

Also note: the Windows version, whether the app was started from the Start menu, and anything
that looked wrong or slow (with a screenshot if you can).
