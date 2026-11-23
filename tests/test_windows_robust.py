"""Windows robustness found on the owner's PC (pc_tasks/11, TESTING.md B2, 2026-11-23):
files saved with a BOM, an output a video player holds open, a non-English user name
(non-ASCII TEMP and cache paths for NVRTC and MinGW), and the OpenMP probe."""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from phasor.core.proc import ascii_path
from phasor.render.gpu import registry
from tests.test_kernels import CASES

BOM = b"\xef\xbb\xbf"
ROOT = Path(__file__).resolve().parent.parent
windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


def test_a_document_saved_with_a_bom_loads(tmp_path):
    from phasor.core.io import load_document

    plain = ROOT / "examples" / "moving_dot.phasor.json"
    bom = tmp_path / "dot.phasor.json"
    bom.write_bytes(BOM + plain.read_bytes())  # Notepad's "UTF-8 with BOM"
    assert load_document(bom) == load_document(plain)


def test_settings_saved_with_a_bom_are_kept(tmp_path, monkeypatch):
    from phasor.gui.app import Settings

    monkeypatch.setenv("PHASOR_CONFIG_DIR", str(tmp_path))
    raw = {"recent": [str(tmp_path / "a.phasor.json")], "math_engine": "latex"}
    (tmp_path / "settings.json").write_bytes(BOM + json.dumps(raw).encode())
    s = Settings.load()
    assert s.recent == raw["recent"] and s.math_engine == "latex"  # not reset to defaults


@windows
def test_a_sound_render_into_a_file_a_player_holds_says_so(tmp_path):
    pytest.importorskip("av")
    from phasor.render.encode import EncodeError, mux_audio, open_writer

    out = tmp_path / "v.mp4"
    try:
        w = open_writer(out, 64, 64, 10)
    except EncodeError as e:
        pytest.skip(f"no video encoder here: {e}")
    for _ in range(10):
        w.write(np.zeros((64, 64, 3), np.uint8))
    w.close()
    with open(out, "rb"), pytest.raises(PermissionError, match="video player"):
        mux_audio(out, np.zeros(4800, np.int16), 48000)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v.mp4"]  # no .mux.mp4 left


def test_ascii_paths_are_unchanged(tmp_path):
    assert ascii_path(tmp_path) == str(tmp_path)


@windows
def test_a_non_english_folder_gets_an_ascii_short_name(tmp_path):
    d = tmp_path / "Jürgen Ñoño Ψ"
    d.mkdir()
    short = ascii_path(d)
    if short == str(d):
        pytest.skip("this volume keeps no 8.3 short names")
    assert short.isascii() and Path(short).samefile(d)


def test_auto_falls_back_once_when_a_kernel_does_not_compile_for_cuda(monkeypatch):
    """Under a TEMP NVRTC cannot read, a fresh install could not build its kernels and the
    render stopped: ``auto`` now warns once and runs the kernel on the CPU (or NumPy)."""

    class CompileException(Exception):  # CuPy's class name; CuPy itself is not needed
        pass

    card = object()
    name = "vignette"
    real = registry.get(name)
    tries = []

    def launch(dev, tile, **params):
        if dev is card:
            tries.append(1)
            raise CompileException('Catastrophic error: cannot open source file "x.cu"')
        return real.launch(dev, tile, **params)

    monkeypatch.setitem(registry._KERNELS, name, dataclasses.replace(real, launch=launch))
    monkeypatch.setattr(registry, "_CUDA_FAILED", set())
    tile, params = CASES[name][0]
    monkeypatch.setattr(registry, "default_gpu", lambda: None)
    want = registry.run(name, tile, **params)  # what this machine gives without a card
    monkeypatch.setattr(registry, "default_gpu", lambda: card)
    with pytest.warns(RuntimeWarning, match="does not compile for CUDA"):
        got = registry.run(name, tile, **params)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(registry.run(name, tile, **params), want)
    assert len(tries) == 1  # remembered: no new compile per tile
    with pytest.raises(CompileException):  # asked for CUDA explicitly: the error, not a fallback
        registry.run(name, tile, backend="cuda", **params)


def test_a_compiler_that_fails_for_another_reason_is_not_remembered_as_no_openmp(
    tmp_path, monkeypatch
):
    """The probe once saved ``openmp: false`` for good when g++ could not write to a
    non-English TEMP: kernels stayed single-threaded after the path was fixed."""
    from phasor.render.cpu import backend as cpu

    class Done:
        returncode = 1
        stdout = stderr = "Fatal error: can't create ...: Invalid argument"

    monkeypatch.setenv("PHASOR_CPU_CACHE", str(tmp_path))
    monkeypatch.setattr(cpu.subprocess, "run", lambda *a, **k: Done())
    assert cpu._has_openmp("g++", "g++ test 1") is False
    assert not list(tmp_path.glob("openmp-*.json"))


@pytest.mark.skipif(shutil.which("g++") is None, reason="no g++")
def test_a_compiler_without_openmp_is_remembered(tmp_path, monkeypatch):
    from phasor.render.cpu import backend as cpu

    real = cpu.subprocess.run

    def run(cmd, *a, **k):
        if "-fopenmp" in cmd:
            return type("Done", (), {"returncode": 1, "stdout": "", "stderr": "no omp.h"})()
        return real(cmd, *a, **k)

    monkeypatch.setenv("PHASOR_CPU_CACHE", str(tmp_path))
    monkeypatch.setattr(cpu.subprocess, "run", run)
    assert cpu._has_openmp(shutil.which("g++"), "g++ test 2") is False
    (note,) = tmp_path.glob("openmp-*.json")
    assert json.loads(note.read_text(encoding="utf-8"))["openmp"] is False
