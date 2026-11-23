"""The scene scripts here are Manim scripts (``from manim import *``), run only through
``compare.py``: never collected or imported as tests (Windows CI has no Manim)."""

collect_ignore_glob = ["scenes/*.py", "record_manim.py"]
