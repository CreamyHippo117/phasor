"""The editor window (SPEC "GUI editor"), Qt (PySide6) widgets over ``controller.Editor``.

Nothing here decides anything: widgets draw what the ``Editor`` holds and forward clicks and
typing to it. A 30 Hz timer advances playback and notices edits made by an attached AI
(``phasor mcp --attach``), so both editors always show the same live scene.

Layout: the viewport (click to select, drag a selected object to move it) above the
timeline (markers, animated spans, the playhead: click or drag to scrub); on the right,
tabs for the inspector (parameters, style roles, what the inspector reads), lint (click an
issue to jump to it) and looks (switch look, pick among variants of a beat).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from phasor.gui.controller import Editor
from phasor.style.theme import ROLE_GROUPS
from phasor.workspace import ToolError

TICK_MS = 33
DRAG_PX = 4  # a press that moves less than this is a click


def to_qimage(img: np.ndarray) -> QImage:
    h, w = img.shape[:2]
    data = np.ascontiguousarray(img[..., :3], dtype=np.uint8)
    q = QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return q.copy()  # own the pixels: ``data`` goes away


class Viewport(QLabel):
    """The picture at the playhead. Click selects; dragging the selected object moves it."""

    picked = Signal(object)  # the id under a click, or None
    dragged = Signal(str, float, float, tuple)  # id, x, y, viewport size

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #050608; color: #8a93a8;")
        self._press: QPointF | None = None
        self._size = (0, 0)
        self.error: str | None = None

    def frame_size(self) -> tuple[int, int]:
        meta = self.ed.engine.document.meta
        w, h = max(self.width(), 2), max(self.height(), 2)
        scale = min(w / meta.width, h / meta.height)
        return max(2, int(meta.width * scale)), max(2, int(meta.height * scale))

    def refresh(self) -> None:
        size = self.frame_size()
        try:
            img = self.ed.frame(size)
        except RuntimeError as e:  # no OpenGL on this machine
            self.error = str(e)
            self.setText(f"No preview: {e}")
            return
        self.error = None
        self._size = size
        pix = QPixmap.fromImage(to_qimage(img))
        if self.ed.selected:
            self._outline(pix)
        self.setPixmap(pix)

    def _outline(self, pix: QPixmap) -> None:
        """A box around the selected object, from the inspector's own bounds."""
        try:
            with self.ed.lock:
                seen = self.ed.engine.inspect(self.ed.t, id=self.ed.selected)["objects"][0]
        except (ToolError, KeyError, IndexError):
            return
        box = (seen.get("bbox") or {}).get("norm")
        if not box:
            return
        x0, y0, x1, y1 = box  # normalized, origin top-left
        w, h = pix.width(), pix.height()
        p = QPainter(pix)
        p.setPen(QPen(QColor(255, 200, 90, 200), 1.5, Qt.PenStyle.DashLine))
        p.drawRect(QRectF(x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h))
        p.end()

    def _local(self, pos: QPointF) -> tuple[float, float]:
        """Widget point -> frame pixel (the frame is centered in the widget)."""
        fw, fh = self._size
        return pos.x() - (self.width() - fw) / 2, pos.y() - (self.height() - fh) / 2

    def mousePressEvent(self, ev) -> None:
        self._press = ev.position()

    def mouseReleaseEvent(self, ev) -> None:
        if self._press is None or not self._size[0]:
            return
        moved = (ev.position() - self._press).manhattanLength()
        u, v = self._local(ev.position())
        if moved >= DRAG_PX and self.ed.selected:
            self.dragged.emit(self.ed.selected, u, v, self._size)
        else:
            self.picked.emit(self.ed.select_at(*self._local(self._press), self._size))
        self._press = None

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if self._size != self.frame_size():
            self.refresh()


class Timeline(QWidget):
    """Markers, the animated spans of every parameter (one lane each) and the playhead."""

    scrubbed = Signal(float)

    LANE = 14
    LABELS = 130  # pixels of lane names on the left

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.setMinimumHeight(70)
        self.data: dict[str, Any] = {"duration": 0.0, "markers": {}, "lanes": {}}

    def refresh(self) -> None:
        self.data = self.ed.timeline()
        lanes = len(self.data["lanes"])
        self.setMinimumHeight(min(40 + self.LANE * max(lanes, 1), 320))
        self.update()

    def _x(self, t: float) -> float:
        d = max(self.data["duration"], 1e-6)
        return self.LABELS + (self.width() - self.LABELS - 8) * t / d

    def _t(self, x: float) -> float:
        d = max(self.data["duration"], 1e-6)
        return (x - self.LABELS) / max(self.width() - self.LABELS - 8, 1) * d

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0d12"))
        p.setFont(QFont("Sans", 8))
        for name, t in self.data["markers"].items():
            x = self._x(t)
            p.setPen(QPen(QColor("#5a6478"), 1))
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            p.setPen(QColor("#aab3c5"))
            p.drawText(QPointF(x + 3, 12), name)
        p.fillRect(QRectF(0, 0, self.LABELS - 4, self.height()), QColor("#07080b"))
        for k, (path, spans) in enumerate(self.data["lanes"].items()):
            y = 20 + k * self.LANE
            p.setPen(QColor("#6c7891"))
            p.drawText(QRectF(6, y, self.LABELS - 12, self.LANE), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(path, Qt.TextElideMode.ElideMiddle,
                                                  self.LABELS - 12))  # fmt: skip
            for t0, t1 in spans:
                p.fillRect(QRectF(self._x(t0), y + 2, max(self._x(t1) - self._x(t0), 2),
                                  self.LANE - 4), QColor(122, 162, 247, 150))  # fmt: skip
        x = self._x(self.ed.t)
        p.setPen(QPen(QColor("#ffb454"), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.end()

    def mousePressEvent(self, ev) -> None:
        self.scrubbed.emit(self._t(ev.position().x()))

    def mouseMoveEvent(self, ev) -> None:
        if ev.buttons():
            self.scrubbed.emit(self._t(ev.position().x()))


class Inspector(QWidget):
    """The selected object's parameters (type a value, press Enter), style roles and what
    the inspector reads from the picture."""

    edited = Signal(object)  # a patch result

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.form = QFormLayout()
        self.title = QLabel("Nothing selected")
        self.title.setStyleSheet("font-weight: bold;")
        self.semantic = QPlainTextEdit()
        self.semantic.setReadOnly(True)
        self.semantic.setMaximumHeight(160)
        lay = QVBoxLayout(self)
        lay.addWidget(self.title)
        holder = QWidget()
        holder.setLayout(self.form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        lay.addWidget(scroll)
        lay.addWidget(QLabel("What the inspector sees"))
        lay.addWidget(self.semantic)
        self.fields: dict[str, QLineEdit] = {}
        self.roles: dict[str, QComboBox] = {}

    def refresh(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.fields, self.roles = {}, {}
        panel = self.ed.inspector()
        if panel is None:
            self.title.setText("Nothing selected (click an object)")
            self.semantic.setPlainText("")
            return
        oid = panel["id"]
        self.title.setText(f"{oid}  ({panel['type']})")
        for name, value in panel["params"].items():
            field = QLineEdit(json.dumps(value, default=list))
            if any(p == f"{oid}.{name}" for p in panel["animated"]):
                field.setToolTip("animated: this is its value at the playhead")
            field.editingFinished.connect(
                lambda f=field, n=name: self.edited.emit(self.ed.set_param(oid, n, f.text()))
            )
            self.form.addRow(name, field)
            self.fields[name] = field
        for key, role in panel["style"].items():
            group = self.ed.engine.state_at(self.ed.t)[oid].impl.Style.get(key, (None,))[0]
            box = QComboBox()
            box.addItems(list(ROLE_GROUPS.get(group, (role,))))
            box.setCurrentText(role)
            box.currentTextChanged.connect(
                lambda r, k=key: self.edited.emit(self.ed.set_style(oid, k, r))
            )
            self.form.addRow(f"style: {key}", box)
            self.roles[key] = box
        for rule in panel["constraints"]:
            self.form.addRow("rule", QLabel(rule))
        self.semantic.setPlainText(json.dumps(panel["semantic"], indent=1, default=str))


class LintPanel(QListWidget):
    picked = Signal(dict)

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.itemClicked.connect(lambda item: self.picked.emit(item.data(Qt.ItemDataRole.UserRole)))

    def refresh(self) -> None:
        self.clear()
        items = self.ed.lint_items()
        if not items:
            self.addItem("No issues")
        for issue in items:
            when = f" @ {issue['times'][0]:g}s" if issue.get("times") else ""
            it = QListWidgetItem(f"[{issue['severity']}] {issue['code']}{when}: {issue['message']}")
            it.setData(Qt.ItemDataRole.UserRole, issue)
            self.addItem(it)


class LooksPanel(QWidget):
    """Switch the whole scene to a look; see variants of a beat and pick one."""

    changed = Signal(object)

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.look = QComboBox()
        self.look.addItems(editor.looks())
        apply = QPushButton("Apply look")
        apply.clicked.connect(
            lambda: self.changed.emit(self.ed.apply_look(self.look.currentText()))
        )
        self.beat = QComboBox()
        show = QPushButton("Show variants")
        show.clicked.connect(self.show_variants)
        self.picture = QLabel()
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.buttons = QHBoxLayout()
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(self.look)
        row.addWidget(apply)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(self.beat)
        row2.addWidget(show)
        lay.addLayout(row2)
        lay.addWidget(self.picture)
        lay.addLayout(self.buttons)
        lay.addStretch(1)

    def refresh(self) -> None:
        current = self.beat.currentText()
        self.beat.clear()
        self.beat.addItems(list(self.ed.timeline()["markers"]))
        if current:
            self.beat.setCurrentText(current)
        meta = self.ed.engine.document.meta
        if meta.look:
            self.look.setCurrentText(meta.look)

    def show_variants(self) -> None:
        if not self.beat.currentText():
            return
        r = self.ed.variants(self.beat.currentText())
        self.picture.setPixmap(QPixmap(r["image"]).scaledToWidth(
            max(self.width() - 20, 200), Qt.TransformationMode.SmoothTransformation))  # fmt: skip
        while self.buttons.count():
            w = self.buttons.takeAt(0).widget()
            if w:
                w.deleteLater()
        for v in r["variants"]:
            b = QPushButton(f"{v['name']} ({v['score']:g})")
            b.clicked.connect(lambda _=False, n=v["name"]: self.changed.emit(self.ed.choose(n)))
            self.buttons.addWidget(b)


class EditorWindow(QMainWindow):
    def __init__(self, editor: Editor, attach_hint: str = "") -> None:
        super().__init__()
        self.ed = editor
        self.playing = False
        self.setWindowTitle(f"phasor: {editor.engine.document.meta.title}")
        self.resize(1400, 860)
        self.viewport = Viewport(editor)
        self.timeline = Timeline(editor)
        self.inspector = Inspector(editor)
        self.lint = LintPanel(editor)
        self.looks = LooksPanel(editor)
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(self.viewport)
        left.addWidget(self.timeline)
        left.setStretchFactor(0, 4)
        left.setSizes([620, 180])
        tabs = QTabWidget()
        tabs.addTab(self.inspector, "Inspector")
        tabs.addTab(self.lint, "Lint")
        tabs.addTab(self.looks, "Looks")
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(tabs)
        split.setStretchFactor(0, 3)
        split.setSizes([1060, 340])
        self.setCentralWidget(split)
        self._actions()
        self.viewport.picked.connect(lambda _: self._after_selection())
        self.viewport.dragged.connect(self._drag)
        self.timeline.scrubbed.connect(self._scrub)
        self.inspector.edited.connect(self._report)
        self.lint.picked.connect(self._jump)
        self.looks.changed.connect(self._report)
        self.statusBar().showMessage(attach_hint or "Ready")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)
        self.refresh_all()

    def _actions(self) -> None:
        bar = self.addToolBar("edit")
        for text, key, fn in (
            ("Play", "Space", self.toggle_play),
            ("Undo", "Ctrl+Z", lambda: self._report(self.ed.undo())),
            ("Redo", "Ctrl+Shift+Z", lambda: self._report(self.ed.redo())),
            ("Save", "Ctrl+S", lambda: self._report(self.ed.save())),
        ):
            act = QAction(text, self)
            act.setShortcut(key)
            act.triggered.connect(fn)
            bar.addAction(act)

    # ------------------------------------------------------------ events

    def toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing and self.ed.t >= self.ed.engine.duration:
            self.ed.set_time(0.0)

    def tick(self) -> None:
        if self.playing:
            self.playing = self.ed.play_step(TICK_MS / 1000)
            self.viewport.refresh()
            self.timeline.update()
        if self.ed.changed():  # an edit here or by an attached AI
            self.refresh_all()

    def _scrub(self, t: float) -> None:
        self.ed.set_time(t)
        self.viewport.refresh()
        self.timeline.update()
        self.inspector.refresh()

    def _after_selection(self) -> None:
        self.viewport.refresh()
        self.inspector.refresh()

    def _drag(self, oid: str, u: float, v: float, size: tuple) -> None:
        try:
            self._report(self.ed.drag_to(oid, u, v, size))
        except ToolError as e:
            self.statusBar().showMessage(str(e), 6000)

    def _jump(self, issue: dict) -> None:
        self.ed.jump(issue)
        self.refresh_all()

    def _report(self, result: Any) -> None:
        if isinstance(result, dict) and result.get("ok") is False:
            errs = result.get("errors") or [{"message": "refused"}]
            self.statusBar().showMessage("Not applied: " + errs[0].get("message", ""), 8000)
        elif isinstance(result, dict) and result.get("saved"):
            self.statusBar().showMessage(f"Saved {result['saved']}", 4000)

    def refresh_all(self) -> None:
        self.viewport.refresh()
        self.timeline.refresh()
        self.inspector.refresh()
        self.lint.refresh()
        self.looks.refresh()

    def closeEvent(self, ev) -> None:
        self.timer.stop()
        super().closeEvent(ev)


DARK = """
QWidget { background: #0f1117; color: #d8dee9; }
QLineEdit, QPlainTextEdit, QComboBox, QListWidget {
    background: #161a23; border: 1px solid #262c3a;
}
QPushButton, QToolButton { background: #1b2030; border: 1px solid #2c3346; padding: 3px 8px; }
QPushButton:hover, QToolButton:hover { background: #242b40; }
QTabBar::tab { background: #141822; padding: 5px 12px; }
QTabBar::tab:selected { background: #1f2535; color: #ffb454; }
QStatusBar { color: #8a93a8; }
"""  # the house look is dark, always (STYLE.md); Qt's default is light


def open_editor(path: str | None, root: str | None = None, *, serve_ai: bool = True,
                show: bool = True):  # fmt: skip
    """Open the editor on ``path`` (a .phasor.json, or a .py builder script: the person at
    the keyboard may open code; an attached AI still may not). Returns (app, window, server)."""
    import threading

    from PySide6.QtWidgets import QApplication

    from phasor.workspace import Workspace
    from phasor.workspace.remote import RemoteServer

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(DARK)
    ws = Workspace(root or (Path(path).parent if path else Path.cwd()))
    lock = threading.RLock()
    if path:
        ws.allow_scripts = True  # only for this one call, made by the person
        try:
            ws.scene_open(str(Path(path).resolve()))
        finally:
            ws.allow_scripts = False
    else:
        ws.scene_new("untitled")
    editor = Editor(ws, lock)
    server, hint = None, ""
    if serve_ai:
        server = RemoteServer(ws, lock).start()
        server.write_handoff()
        hint = f"An AI can edit this scene: phasor mcp --attach --root {ws.root}"
    win = EditorWindow(editor, hint)
    if show:
        win.show()
    return app, win, server
