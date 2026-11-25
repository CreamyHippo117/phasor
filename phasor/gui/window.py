"""The editor window (SPEC "GUI editor"), Qt (PySide6) widgets over ``controller.Editor``.

Nothing here decides anything: widgets draw what the ``Editor`` holds and forward clicks and
typing to it. A 30 Hz timer advances playback, notices edits made by an attached AI
(``phasor mcp --attach``), watches the script file and follows render jobs, so every editor
always shows the same live scene.

Layout (the code-first window, APP_PLAN.md, Milestone 25): the script on the left (Run with
Ctrl+Enter; an error is marked at its line); on the right the viewport (click to select and
see its line, drag a selected object to move it), the timeline (markers, animated spans,
the playhead) and tabs: Problems (click to go to the object and its line), Inspector, Looks,
Narration, Sound and MIDI, Footage, Sliders, AI. Toolbar: Run, Draft preview, Render.

View > Explorer (Milestone 29) puts the Explorer's rows in the left pane instead of the
script (``ExplorerPanel``; its logic is ``gui.explorer.Explorer``) and sends the viewport's
presses, drags, wheel and keys to it; an explorer document opens there.
"""

from __future__ import annotations

import json
import keyword
import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextFormat,
    QTextOption,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from phasor.gui.controller import Editor, qt_view
from phasor.gui.explorer import CONVERT_WARNING, NEW, Explorer
from phasor.style import explicit
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
    turned = Signal(float, float, tuple, bool)  # pixels dx, dy, viewport size, pan
    zoomed = Signal(float)  # factor: > 1 moves in
    hovered = Signal(float, float, tuple)  # viewport point and size
    explored = Signal(object)  # what a click or drag did in the Explorer (its result)

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        # In the Explorer (Milestone 29) every press, drag, wheel turn and key goes to it.
        self.explorer: Explorer | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
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
        if self.explorer is not None:
            self._explorer_refresh(size)
            return
        try:
            img = self.ed.frame(size)
        except RuntimeError as e:  # no OpenGL on this machine
            self.error = str(e)
            self.setText(f"No preview here: {e}\n\nThe script, Problems, the inspector and the "
                         "other panels still work.")  # fmt: skip
            return
        self.error = None
        self._size = size
        pix = QPixmap.fromImage(to_qimage(img))
        if self.ed.selected:
            self._outline(pix)
        self.setPixmap(pix)

    def _explorer_refresh(self, size: tuple[int, int]) -> None:
        """The Explorer's picture: half size while the person interacts, shown at the
        display size (clicks map to the display size either way)."""
        try:
            img, _quality = self.explorer.frame(size)
        except RuntimeError as e:  # no OpenGL on this machine
            self.error = str(e)
            self._size = size  # clicks still work: they are mapped without the picture
            self.setText(f"No preview here: {e}\n\nThe rows, sliders, analysis and the AI "
                         "still work.")  # fmt: skip
            return
        self.error = None
        self._size = size
        pix = QPixmap.fromImage(to_qimage(img))
        if (pix.width(), pix.height()) != size:
            pix = pix.scaled(size[0], size[1])
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
        self._shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if self.explorer is not None and self._size[0]:
            self.setFocus()
            pan = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier) or (
                ev.button() == Qt.MouseButton.MiddleButton)  # fmt: skip
            self.explorer.press(*self._local(ev.position()), self._size, self._shift, pan)

    def mouseMoveEvent(self, ev) -> None:
        if self.explorer is not None and self._size[0]:
            if self.explorer.move(*self._local(ev.position()), self._size):
                self.refresh()
            elif not ev.buttons():  # hover: the coordinates in the status bar, as elsewhere
                self.hovered.emit(*self._local(ev.position()), self._size)
            return
        if self._size[0] and not ev.buttons():
            self.hovered.emit(*self._local(ev.position()), self._size)

    def mouseDoubleClickEvent(self, ev) -> None:
        if self.explorer is not None and self._size[0]:
            self.explored.emit(self.explorer.double_click(*self._local(ev.position()),
                                                          self._size))  # fmt: skip

    def keyPressEvent(self, ev) -> None:
        names = {Qt.Key.Key_Left: "left", Qt.Key.Key_Right: "right", Qt.Key.Key_R: "r",
                 Qt.Key.Key_Space: "space"}  # fmt: skip
        if self.explorer is not None and ev.key() in names:
            self.explorer.key(names[ev.key()])
            self.refresh()
            return
        super().keyPressEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        if self.explorer is not None:
            if self._size[0]:
                self.explored.emit(self.explorer.release(*self._local(ev.position()),
                                                         self._size))  # fmt: skip
            self._press = None
            return
        if self._press is None or not self._size[0]:
            return
        moved = (ev.position() - self._press).manhattanLength()
        u, v = self._local(ev.position())
        if moved >= DRAG_PX:
            on_selected = False
            if self.ed.selected:
                hit = self.ed.hit_ids(*self._local(self._press), self._size)
                on_selected = bool(hit) and hit[0] == self.ed.selected
            if on_selected:  # a drag that starts on the selected object moves it
                self.dragged.emit(self.ed.selected, u, v, self._size)
            else:  # a drag over anything else moves the camera (shift: slide it)
                d = ev.position() - self._press
                self.turned.emit(d.x(), d.y(), self._size, self._shift)
        else:
            self.picked.emit(self.ed.select_at(*self._local(self._press), self._size))
        self._press = None

    def wheelEvent(self, ev) -> None:
        if self.explorer is not None:
            self.explorer.wheel(1.1 ** (ev.angleDelta().y() / 120))
            self.refresh()
            return
        self.zoomed.emit(1.1 ** (ev.angleDelta().y() / 120))

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if self._size != self.frame_size():
            self.refresh()


def _extra_panels(window) -> list[tuple[str, QWidget]]:
    """Tabs other clients add (the ``phasor.panels`` entry point, e.g. Sketch): each entry
    is ``make(window) -> (title, widget)``. Looked up by name, so the editor never imports
    a sibling client; one that fails to load is reported and left out."""
    from importlib.metadata import entry_points

    out = []
    for ep in sorted(entry_points(group="phasor.panels"), key=lambda e: e.name):
        try:
            out.append(ep.load()(window))
        except Exception as e:  # a broken add-on must not stop the editor
            print(f"phasor gui: the {ep.name!r} tab did not load: {e}")
    return out


class Timeline(QWidget):
    """Narration (with its waveform), markers, the animated spans of every parameter (one
    lane each) and the playhead. Click or drag to scrub; drag a marker to move it; drag a span
    to slide it, or its ends to stretch it."""

    scrubbed = Signal(float)
    edited = Signal(object)  # a patch result

    LANE = 14
    LABELS = 130  # pixels of lane names on the left
    NARRATION = 34  # height of the narration lane, when there is narration
    EDGE = 5  # pixels: grab a span's end within this

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.setMinimumHeight(70)
        self.setMouseTracking(True)
        self.data: dict[str, Any] = {"duration": 0.0, "markers": {}, "lanes": {}, "narration": []}
        self._hits: list[tuple[QRectF, str, tuple]] = []
        self._drag: tuple | None = None  # (kind, key..., original, current)
        self._waves: dict[tuple, list[float]] = {}

    def refresh(self) -> None:
        self.data = self.ed.timeline()
        lanes = len(self.data["lanes"])
        self.setMinimumHeight(min(40 + self._top() + self.LANE * max(lanes, 1), 340))
        self.update()

    def _top(self) -> int:
        return self.NARRATION if self.data.get("narration") else 0

    def _x(self, t: float) -> float:
        d = max(self.data["duration"], 1e-6)
        return self.LABELS + (self.width() - self.LABELS - 8) * t / d

    def _t(self, x: float) -> float:
        d = max(self.data["duration"], 1e-6)
        return (x - self.LABELS) / max(self.width() - self.LABELS - 8, 1) * d

    def _wave(self, audio: str, bins: int, span=None) -> list[float]:
        key = (audio, bins, tuple(span) if span else None)
        if key not in self._waves:
            self._waves[key] = self.ed.waveform(audio, bins, span)
        return self._waves[key]

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0d12"))
        p.setFont(QFont("Sans", 8))
        self._hits = []
        top = self._top()
        for b in self.data.get("narration", []):  # spoken blocks, with their loudness
            x0, x1 = self._x(b["start"]), self._x(b["start"] + b["duration"])
            box = QRectF(x0, 16, max(x1 - x0, 3), self.NARRATION - 18)
            p.fillRect(box, QColor(80, 120, 90, 90))
            if b.get("audio"):
                bins = max(int(box.width() / 2), 2)
                wave = self._wave(b["audio"], bins, b.get("span"))
                p.setPen(QPen(QColor("#7fcf9a"), 1))
                for k, a in enumerate(wave):
                    h = a * box.height() / 2
                    x = box.left() + (k + 0.5) / len(wave) * box.width()
                    p.drawLine(QPointF(x, box.center().y() - h), QPointF(x, box.center().y() + h))
            p.setPen(QColor("#b8e0c4"))
            p.drawText(box.adjusted(2, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(b["text"], Qt.TextElideMode.ElideRight,
                                                  int(box.width()) - 4))  # fmt: skip
        p.fillRect(QRectF(0, 0, self.LABELS - 4, self.height()), QColor("#07080b"))
        for k, (path, spans) in enumerate(self.data["lanes"].items()):
            y = 20 + top + k * self.LANE
            p.setPen(QColor("#6c7891"))
            p.drawText(QRectF(6, y, self.LABELS - 12, self.LANE), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(path, Qt.TextElideMode.ElideMiddle,
                                                  self.LABELS - 12))  # fmt: skip
            for i, (t0, t1) in enumerate(spans):
                if self._drag and self._drag[0] == "span" and self._drag[1:3] == (path, i):
                    t0, t1 = self._drag[-1]
                r = QRectF(self._x(t0), y + 2, max(self._x(t1) - self._x(t0), 2), self.LANE - 4)
                p.fillRect(r, QColor(122, 162, 247, 150))
                self._hits.append((r, "span", (path, i, t0, t1)))
        for name, t in self.data["markers"].items():
            if self._drag and self._drag[0] == "marker" and self._drag[1] == name:
                t = self._drag[-1]
            x = self._x(t)
            p.setPen(QPen(QColor("#5a6478"), 1))
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            p.setPen(QColor("#aab3c5"))
            p.drawText(QPointF(x + 3, 12), name)
            self._hits.append((QRectF(x - self.EDGE, 0, 2 * self.EDGE, self.height()), "marker",
                               (name, t)))  # fmt: skip
        x = self._x(self.ed.t)
        p.setPen(QPen(QColor("#ffb454"), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.end()

    def _hit(self, pos: QPointF):
        for rect, kind, data in self._hits:  # markers first (thin, on top of spans)
            if kind == "marker" and rect.contains(pos):
                return kind, data
        for rect, kind, data in self._hits:
            if kind == "span" and rect.adjusted(-self.EDGE // 2, 0, self.EDGE // 2, 0).contains(
                pos
            ):
                return kind, data
        return None

    def mousePressEvent(self, ev) -> None:
        hit = self._hit(ev.position())
        if hit and hit[0] == "marker":
            name, t = hit[1]
            self._drag = ("marker", name, t, t)
        elif hit:
            path, i, t0, t1 = hit[1]
            x = ev.position().x()
            mode = ("start" if abs(x - self._x(t0)) <= self.EDGE
                    else "end" if abs(x - self._x(t1)) <= self.EDGE else "move")  # fmt: skip
            self._drag = ("span", path, i, mode, self._t(x), (t0, t1), (t0, t1))
        else:
            self._drag = None
            self.scrubbed.emit(self._t(ev.position().x()))

    def mouseMoveEvent(self, ev) -> None:
        d = self._drag
        if d is None:
            if ev.buttons():
                self.scrubbed.emit(self._t(ev.position().x()))
            else:
                hit = self._hit(ev.position())
                self.setCursor(Qt.CursorShape.SizeHorCursor if hit else Qt.CursorShape.ArrowCursor)
            return
        t = max(0.0, self._t(ev.position().x()))
        if d[0] == "marker":
            self._drag = ("marker", d[1], d[2], t)
        else:
            _, path, i, mode, grab, (o0, o1), _cur = d
            dt = t - grab
            new = ((max(0.0, o0 + dt), o1 + dt) if mode == "move"
                   else (min(t, o1 - 0.05), o1) if mode == "start"
                   else (o0, max(t, o0 + 0.05)))  # fmt: skip
            self._drag = ("span", path, i, mode, grab, (o0, o1), new)
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        d, self._drag = self._drag, None
        if d is None:
            return
        if d[0] == "marker" and abs(d[3] - d[2]) > 1e-3:
            self.edited.emit(self.ed.move_marker(d[1], d[3]))
        elif d[0] == "span" and max(abs(d[6][0] - d[5][0]), abs(d[6][1] - d[5][1])) > 1e-3:
            self.edited.emit(self.ed.retime_segment(d[1], d[2], *d[6]))
        self.update()


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
            # A value the scene typed (a color, a size) is shown as it is, first; picking a
            # role replaces it, and nothing replaces it unasked.
            typed = None if explicit.is_role(role) else role
            label = json.dumps(typed) if typed is not None else None
            if label is not None:
                box.addItem(label)
            box.addItems(list(ROLE_GROUPS.get(group, () if label else (role,))))
            box.setCurrentText(label if label is not None else role)
            box.currentTextChanged.connect(
                lambda r, k=key, keep=label: (
                    None if r == keep else self.edited.emit(self.ed.set_style(oid, k, r))
                )
            )
            self.form.addRow(f"style: {key}", box)
            self.roles[key] = box
        for rule in panel["constraints"]:
            self.form.addRow("rule", QLabel(rule))
        self.semantic.setPlainText(json.dumps(panel["semantic"], indent=1, default=str))


class LintPanel(QListWidget):
    """The Problems tab: the script's error (at its line), then lint. A click goes to the
    object, its time and its line in the script."""

    picked = Signal(dict)

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.itemClicked.connect(self._clicked)

    def _clicked(self, item) -> None:
        issue = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(issue, dict):
            self.picked.emit(issue)

    def refresh(self) -> None:
        self.clear()
        try:
            items = self.ed.problems()
        except (ToolError, RuntimeError) as e:  # lint needs what this machine lacks
            items = [{"severity": "info", "code": "lint", "message": str(e), "ids": []}]
        if not items:
            self.addItem("No problems")
        for issue in items:
            when = f" @ {issue['times'][0]:g}s" if issue.get("times") else ""
            line = f" (line {issue['line']})" if issue.get("line") else ""
            it = QListWidgetItem(f"[{issue['severity']}] {issue['code']}{when}{line}: "
                                 f"{issue['message']}")  # fmt: skip
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
        self.palette = QComboBox()
        self.palette.addItems(["theme's own", *editor.palettes()])
        recolor = QPushButton("Apply palette")
        recolor.clicked.connect(self.apply_palette)
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
        row_p = QHBoxLayout()
        row_p.addWidget(self.palette)
        row_p.addWidget(recolor)
        lay.addLayout(row_p)
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
        self.palette.setCurrentText(meta.palette or "theme's own")

    def apply_palette(self) -> None:
        name = self.palette.currentText()
        self.changed.emit(self.ed.set_palette(None if name == "theme's own" else name))

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


class ControlsPanel(QWidget):
    """The scene's sliders: moving one previews it in the viewport; Apply makes the values
    part of the scene (one undo step); Reset drops the preview."""

    moved = Signal()
    applied = Signal(object)

    RES = 1000

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.form = QFormLayout()
        holder = QWidget()
        holder.setLayout(self.form)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        apply = QPushButton("Apply to scene")
        apply.clicked.connect(lambda: self.applied.emit(self.ed.apply_controls()))
        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset)
        row = QHBoxLayout()
        row.addWidget(apply)
        row.addWidget(reset)
        lay = QVBoxLayout(self)
        lay.addWidget(self.note)
        lay.addWidget(holder)
        lay.addLayout(row)
        lay.addStretch(1)
        self.sliders: dict[str, QSlider] = {}
        self.reads: dict[str, QLabel] = {}

    def refresh(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.sliders, self.reads = {}, {}
        controls = self.ed.controls()
        self.note.setText("" if controls else "This scene has no sliders. Add one in code: "
                          "s.control('object.param', min=0, max=1).")  # fmt: skip
        for c in controls:
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, self.RES)
            s.setValue(round((c["value"] - c["min"]) / (c["max"] - c["min"]) * self.RES))
            read = QLabel(f"{c['value']:.4g}")
            s.valueChanged.connect(lambda v, c=c, read=read: self._moved(c, v, read))
            box = QHBoxLayout()
            box.addWidget(s, 1)
            box.addWidget(read)
            self.form.addRow(c["label"], box)
            self.sliders[c["key"]] = s
            self.reads[c["key"]] = read

    def _moved(self, c: dict, v: int, read: QLabel) -> None:
        value = self.ed.set_control(c["key"], c["min"] + v / self.RES * (c["max"] - c["min"]))
        read.setText(f"{value:.4g}")
        self.moved.emit()

    def _reset(self) -> None:
        self.ed.reset_controls()
        self.refresh()
        self.moved.emit()


class PythonHighlighter(QSyntaxHighlighter):
    """Colors for the code view: keywords, strings, numbers, comments, decorators and the
    names a scene script uses (Scene, expr, the object classes)."""

    def __init__(self, document) -> None:
        super().__init__(document)

        def fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            f.setFontItalic(italic)
            return f

        kw = "|".join(keyword.kwlist)
        self.rules = [
            (re.compile(rf"\b({kw})\b"), fmt("#c792ea", bold=True)),
            (
                re.compile(r"\b(Scene|expr|geo|cx|pde|calc|algebra|fourier|linalg|ac)\b"),
                fmt("#7aa2f7"),
            ),
            (re.compile(r"\b\d+(\.\d+)?([eE][-+]?\d+)?\b"), fmt("#f78c6c")),
            (re.compile(r"@\w+"), fmt("#ffcb6b")),
            (re.compile(r"(\"[^\"\\]*(\\.[^\"\\]*)*\")|('[^'\\]*(\\.[^'\\]*)*')"), fmt("#c3e88d")),
            (re.compile(r"#[^\n]*"), fmt("#5c6783", italic=True)),
        ]

    def highlightBlock(self, text: str) -> None:
        for pattern, f in self.rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), f)


class CodePanel(QWidget):
    """The script, on the left of the window: always editable. Run (Ctrl+Enter) builds the
    scene from it; a mistake is marked at its line and the preview keeps the last good scene.
    For a script scene it is the script itself, its tweaks block kept in step with the scene
    (ADR 0004); for a document it is code generated from it (Run makes it the scene,
    ``name.scene.py``). Clicking a line selects the object it makes."""

    ran = Signal(object)
    line_clicked = Signal(int)

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.text = QPlainTextEdit()
        self.text.setFont(QFont("Monospace", 10))
        self.text.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self.highlighter = PythonHighlighter(self.text.document())
        self.title = QLabel("")
        self.title.setStyleSheet("font-weight: bold;")
        self.status = QLabel("")
        self.status.setWordWrap(True)
        run = QPushButton("Run")
        run.setToolTip("Build the scene from this code (Ctrl+Enter; Ctrl+S runs and saves)")
        run.setShortcut("Ctrl+Return")
        run.clicked.connect(self.run)
        self.stop = QPushButton("Cancel run")
        self.stop.setToolTip("Stop a script that takes too long: the last scene that ran stays")
        self.stop.clicked.connect(self.cancel)
        self.stop.hide()
        self._typed = False  # a document's generated code was typed in
        self._unshowable = False  # the script holds a character the box cannot hold
        self._loading = False
        self._note = ""  # the result of the last Run, kept across refreshes
        self._line: int | None = None  # the line shown for the selection
        self.text.textChanged.connect(self._changed)
        self.text.viewport().installEventFilter(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        row.addWidget(self.title, 1)
        row.addWidget(self.stop)
        row.addWidget(run)
        lay.addLayout(row)
        lay.addWidget(self.text, 1)
        lay.addWidget(self.status)

    @property
    def edited(self) -> bool:
        """Edits in the box that are not saved (a script) or not run (generated code)."""
        return self.ed.dirty if self.ed.is_script else self._typed

    def _raw(self) -> str:
        """The box's text as it holds it (``toRawText``: a no-break space and a line
        separator stay as they are), paragraph separators read as new lines."""
        return self.text.document().toRawText().replace("\u2029", "\n")

    def _changed(self) -> None:
        if self._loading:
            return
        self.ed.set_view_text(self._raw())
        if not self.ed.is_script:
            self._typed = True

    def eventFilter(self, obj, ev) -> bool:  # a click on a line selects what it makes
        from PySide6.QtCore import QEvent

        released = ev.type() == QEvent.Type.MouseButtonRelease
        if obj is self.text.viewport() and released and not self.text.textCursor().hasSelection():
            self.line_clicked.emit(self.text.textCursor().blockNumber() + 1)
        return False

    def refresh(self, force: bool = False) -> None:
        """Show the editor's text (a tweak rewrote the block, the file was reloaded) where
        the cursor was, the error's line, and where the code is."""
        if self.ed.is_script:
            want = self.ed.buffer
        elif self._typed and not force:
            self.status.setText("edited: Run to use it")
            return
        else:
            want = self.ed.code()["text"]
            self.ed.set_buffer(want)
            self._typed = False
        if self._raw() != qt_view(want):
            self._replace_text(qt_view(want))
        self._unshowable = len(self._raw()) != len(qt_view(want))
        if self._unshowable:  # a character the box cannot hold: never written from the box
            self.text.setReadOnly(True)
            self.status.setText("this script holds a character the editor cannot show: edit "
                                "it in another editor (phasor reloads it)")  # fmt: skip
            return
        code = self.ed.code()
        self.title.setText(Path(code["path"]).name + ("" if code["is_script"] else " (generated)"))
        self._marks()
        if self.ed.error is not None:
            err = self.ed.error
            where = f"line {err['line']}: " if err.get("line") else ""
            self.status.setText(f"not run: {where}{err['message']}  (the preview shows the last "
                                "scene that ran)")  # fmt: skip
            return
        where = ("script: " if code["is_script"] else "generated: ") + code["path"]
        self.status.setText(f"{self._note}   ({where})" if self._note else where)

    def _replace_text(self, want: str) -> None:
        """Put ``want`` in the box by editing only the part that differs, as one step of
        the box's own undo (Ctrl+Z in the text): the person's typing history stays, and the
        cursor and scroll stay where they were."""
        old = self._raw()
        a = 0
        while a < min(len(old), len(want)) and old[a] == want[a]:
            a += 1
        b = 0
        while b < min(len(old), len(want)) - a and old[-1 - b] == want[-1 - b]:
            b += 1

        def qpos(text: str, i: int) -> int:  # Qt counts UTF-16 code units
            return len(text[:i].encode("utf-16-le")) // 2

        scroll = self.text.verticalScrollBar().value()
        cur = QTextCursor(self.text.document())
        cur.beginEditBlock()
        cur.setPosition(qpos(old, a))
        cur.setPosition(qpos(old, len(old) - b), QTextCursor.MoveMode.KeepAnchor)
        self._loading = True
        cur.insertText(want[a : len(want) - b])
        self._loading = False
        cur.endEditBlock()
        self.text.verticalScrollBar().setValue(scroll)

    def _marks(self) -> None:
        """The error's line in red, the selected object's line in amber."""
        marks = []
        for line, color in ((self._line, "#3a3220"),
                            ((self.ed.error or {}).get("line"), "#4a1f24")):  # fmt: skip
            block = self.text.document().findBlockByNumber((line or 0) - 1) if line else None
            if block is None or not block.isValid():
                continue
            sel = QTextEdit.ExtraSelection()
            sel.format.setBackground(QColor(color))
            sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            sel.cursor = QTextCursor(block)
            marks.append(sel)
        self.text.setExtraSelections(marks)

    def show_line(self, line: int | None) -> None:
        """Mark the selected object's line and scroll to it (None: no mark)."""
        self._line = line
        if line:
            block = self.text.document().findBlockByNumber(line - 1)
            if block.isValid():
                self.text.setTextCursor(QTextCursor(block))
                self.text.centerCursor()
        self._marks()

    QUICK = 2.0  # seconds Run waits before the window carries on (the run goes on behind)

    def run(self) -> None:
        """Run: on a worker thread, so a long or runaway script never freezes the window
        (Cancel run stops it); a quick one is shown at once."""
        r = self.ed.run_code(self.ed.buffer, wait=False)
        if r["ok"] is None:
            task = self.ed.running
            if task is not None and task.done.wait(self.QUICK):
                r = self.ed.poll_run() or r
        if r["ok"] is None:
            self.sync_state()
            self.status.setText("running... (Cancel run stops it; the preview shows the last "
                                "scene that ran)")  # fmt: skip
            return
        self.finish(r)

    def cancel(self) -> None:
        r = self.ed.cancel_run()
        if r is not None:
            self.finish(r)

    def sync_state(self) -> None:
        """Read-only while a script runs (what is typed meanwhile could not be part of what
        runs), with Cancel run shown."""
        running = self.ed.running is not None
        self.text.setReadOnly(running or self._unshowable)
        self.stop.setVisible(running)

    def finish(self, r: dict) -> None:
        """A run is over (here, or noticed by the window's timer)."""
        self.stop.hide()
        self.sync_state()
        if r["ok"]:
            self._note = "ran: the scene was rebuilt from this code"
            self._typed = False
            self.refresh(force=True)
        else:
            self._note = ""
            self.status.setText("not run: " + r["error"])
            self._marks()
            self.show_line(r.get("line"))
        self.ran.emit(r)


class NarrationPanel(QWidget):
    """Narration from a recording (a WAV, or the sound of an mp4): what speaks now, the
    speech and pauses found in it, and "Snap markers to narration" with a preview of what
    would move (nothing moves until Apply; one undoable step)."""

    changed = Signal(object)

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.now = QLabel("")
        self.now.setWordWrap(True)
        self.words = QPlainTextEdit()
        self.words.setPlaceholderText("What the recording says (optional): its sentences are "
                                      "found in it; [[name]] puts a marker on a word.")  # fmt: skip
        self.words.setMaximumHeight(70)
        pick = QPushButton("Choose a recording (wav, mp4, ...)...")
        pick.clicked.connect(lambda: self.choose())
        self.record = QPushButton("Record...")
        self.record.setEnabled(False)  # no audio input library in the gui extra
        self.record.setToolTip("Recording needs an audio input library the app does not ship; "
                               "record with any recorder and choose the file")  # fmt: skip
        listen = QPushButton("Speech and pauses")
        listen.clicked.connect(self.show_speech)
        preview = QPushButton("Snap markers to narration: preview")
        preview.clicked.connect(self.preview)
        self.apply = QPushButton("Apply the snap")
        self.apply.setEnabled(False)
        self.apply.clicked.connect(self.do_apply)
        self.report = QListWidget()
        lay = QVBoxLayout(self)
        lay.addWidget(self.now)
        lay.addWidget(self.words)
        row = QHBoxLayout()
        row.addWidget(pick, 1)
        row.addWidget(self.record)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(listen)
        row2.addWidget(preview)
        row2.addWidget(self.apply)
        lay.addLayout(row2)
        lay.addWidget(self.report, 1)

    def refresh(self) -> None:
        n = self.ed.narration()
        if n["voice"] is None:
            self.now.setText("No narration yet. Choose a recording, or write s.narrate(...) "
                             "in the script for a voice.")  # fmt: skip
        elif n["audio"]:
            self.now.setText(f"Narrated by the recording {n['audio']} ({n['blocks']} blocks)")
        else:
            self.now.setText(f"Narrated by the voice {n['voice']} ({n['blocks']} blocks)")

    def choose(self, path: str | None = None) -> None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Choose the narration's recording", "",
                                                  "Sound (*.wav *.mp4 *.m4a *.mp3 *.ogg *.flac "
                                                  "*.mov *.webm);;All files (*)")  # fmt: skip
            if not path:
                return
        try:
            r = self.ed.set_narration_audio(path, self.words.toPlainText().strip())
        except ToolError as e:
            r = {"ok": False, "errors": [{"message": str(e)}]}
        self.changed.emit(r)
        self.refresh()

    def _show(self, lines: list[str]) -> None:
        self.report.clear()
        self.report.addItems(lines or ["(nothing)"])

    def show_speech(self) -> None:
        r = self.ed.speech_preview()
        self._show([r["text"], *r["notes"]] if r["ok"] else [r["message"]])

    def preview(self) -> None:
        try:
            r = self.ed.snap_preview()
        except ToolError as e:
            self._show([str(e)])
            self.apply.setEnabled(False)
            return
        lines = [*(f"note: {n}" for n in r["notes"]), *r["rows"], *r["kept_rows"]]
        if not r["rows"]:
            lines.insert(0, "nothing would move")
        self._show(lines)
        self.apply.setEnabled(bool(r["can_apply"]))

    def do_apply(self) -> None:
        r = self.ed.snap_apply()
        self.apply.setEnabled(False)
        self.changed.emit(r)


class SoundPanel(QWidget):
    """Sound clips and MIDI notes at markers, and the notes exported to a .mid."""

    changed = Signal(object)

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.at = QComboBox()
        self.at.setEditable(True)
        self.at.setToolTip("A marker (the notes follow it), or seconds")
        self.instrument = QComboBox()
        self.instrument.addItems(["auto", "sine", "pluck", "organ", "bell", "pad", "lead"])
        self.roll = QCheckBox("show a piano roll")
        self.typed = QPlainTextEdit()
        self.typed.setPlaceholderText("Notes: pitch start length [velocity], e.g. "
                                      "C4 0 0.5, E4 0.5 0.5, G4 1 1")  # fmt: skip
        self.typed.setMaximumHeight(60)
        add_sound = QPushButton("Add a sound clip...")
        add_sound.clicked.connect(lambda: self.add_sound())
        add_mid = QPushButton("Add notes from a .mid...")
        add_mid.clicked.connect(lambda: self.add_midi())
        add_typed = QPushButton("Add the typed notes")
        add_typed.clicked.connect(self.add_typed)
        export = QPushButton("Export the notes to .mid...")
        export.clicked.connect(lambda: self.export())
        self.note = QLabel("")
        self.note.setWordWrap(True)
        form = QFormLayout()
        form.addRow("At", self.at)
        form.addRow("Instrument", self.instrument)
        form.addRow("", self.roll)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        row = QHBoxLayout()
        row.addWidget(add_sound)
        row.addWidget(add_mid)
        lay.addLayout(row)
        lay.addWidget(self.typed)
        row2 = QHBoxLayout()
        row2.addWidget(add_typed)
        row2.addWidget(export)
        lay.addLayout(row2)
        lay.addWidget(self.note)
        lay.addStretch(1)

    def refresh(self) -> None:
        current = self.at.currentText()
        self.at.clear()
        self.at.addItems(["0", *self.ed.timeline()["markers"]])
        self.at.setCurrentText(current or "0")

    def _done(self, r: dict) -> None:
        if r.get("copied"):
            self.note.setText(f"copied into the scene's folder: {r['copied']}")
        elif r.get("ok") is False:
            self.note.setText("not added: " + (r.get("errors") or [{}])[0].get("message", ""))
        else:
            self.note.setText(f"added {r.get('id', '')}")
        self.changed.emit(r)

    def _ask(self, title: str, filt: str) -> str:
        path, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        return path

    def add_sound(self, path: str | None = None) -> None:
        path = path or self._ask("Add a sound clip", "Sound (*.wav *.mp3 *.ogg *.flac *.m4a "
                                 "*.mp4);;All files (*)")  # fmt: skip
        if path:
            try:
                self._done(self.ed.add_sound(path, at=self.at.currentText()))
            except ToolError as e:
                self._done({"ok": False, "errors": [{"message": str(e)}]})

    def add_midi(self, path: str | None = None) -> None:
        path = path or self._ask("Add notes from a MIDI file", "MIDI (*.mid *.midi);;All (*)")
        if path:
            try:
                self._done(self.ed.add_notes(path=path, at=self.at.currentText(),
                                             instrument=self.instrument.currentText(),
                                             show=self.roll.isChecked()))  # fmt: skip
            except ToolError as e:
                self._done({"ok": False, "errors": [{"message": str(e)}]})

    def add_typed(self) -> None:
        try:
            self._done(self.ed.add_notes(typed=self.typed.toPlainText(), at=self.at.currentText(),
                                         instrument=self.instrument.currentText(),
                                         show=self.roll.isChecked()))  # fmt: skip
        except ToolError as e:
            self._done({"ok": False, "errors": [{"message": str(e)}]})

    def export(self, path: str | None = None, overwrite: bool = False) -> dict | None:
        if path is None:
            start = str(self.ed.default_output(".mid"))
            path, _ = QFileDialog.getSaveFileName(self, "Export the notes", start,
                                                  "MIDI (*.mid)")  # fmt: skip
            if not path:
                return None
            overwrite = True  # the save dialog already asked about replacing it
        try:
            r = self.ed.export_midi(path, overwrite=overwrite)
            self.note.setText(f"wrote {r['notes']} notes in {r['tracks']} tracks to {r['path']}")
        except ToolError as e:
            r = None
            self.note.setText(f"not exported: {e}")
        return r


class FootagePanel(QWidget):
    """Video clips, pictures and picture sequences: as the full-frame background, or placed
    like any object."""

    changed = Signal(object)

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor
        self.background = QCheckBox("as the background (fills the frame, behind everything)")
        self.background.setChecked(True)
        add = QPushButton("Add a video or picture...")
        add.clicked.connect(lambda: self.add())
        seq = QPushButton("Add a picture sequence (pick one frame)...")
        seq.clicked.connect(lambda: self.add_sequence())
        self.note = QLabel("")
        self.note.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.background)
        lay.addWidget(add)
        lay.addWidget(seq)
        lay.addWidget(self.note)
        lay.addStretch(1)

    def refresh(self) -> None:
        pass

    def _done(self, r: dict) -> None:
        if r.get("ok") is False:
            self.note.setText("not added: " + (r.get("errors") or [{}])[0].get("message", ""))
        else:
            extra = f" (copied into the scene's folder: {r['copied']})" if r.get("copied") else ""
            self.note.setText(f"added {r.get('id', '')}{extra}")
        self.changed.emit(r)

    def add(self, path: str | None = None) -> None:
        if path is None:
            kinds = "Video and pictures (*.mp4 *.mov *.m4v *.webm *.mkv *.png *.jpg *.jpeg *.gif)"
            path, _ = QFileDialog.getOpenFileName(self, "Add a video or picture", "",
                                                  f"{kinds};;All (*)")  # fmt: skip
            if not path:
                return
        try:
            self._done(self.ed.add_footage(path, background=self.background.isChecked()))
        except ToolError as e:
            self._done({"ok": False, "errors": [{"message": str(e)}]})

    def add_sequence(self, path: str | None = None) -> None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Pick one frame of the sequence", "", "Pictures (*.png *.jpg *.jpeg);;All (*)"
            )
            if not path:
                return
        try:
            self._done(self.ed.add_sequence(path, background=self.background.isChecked()))
        except ToolError as e:
            self._done({"ok": False, "errors": [{"message": str(e)}]})


class AIPanel(QWidget):
    """The AI tab: the chat inside the app comes in phase 4 (APP_PLAN.md). Until then an AI
    edits this same scene from outside, through ``phasor mcp --attach``."""

    def __init__(self, editor: Editor, hint: str = "") -> None:
        super().__init__()
        text = QLabel(
            "<b>The AI chat inside the app comes later (phase 4 of APP_PLAN.md).</b><p>Today an "
            "AI edits this same scene from outside: Claude Code or Claude Desktop with "
            "<code>phasor mcp</code> and the phasor skill. Its edits show up here at once (and "
            "in the script's tweaks block); edits to the script file reload it.</p>"
            + (f"<p>{hint}</p>" if hint else "")
        )
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay = QVBoxLayout(self)
        lay.addWidget(text)
        lay.addStretch(1)


class ActivityDock(QListWidget):
    """Who changed the scene, newest last: your edits, an attached AI's, the file's."""

    def __init__(self, editor: Editor) -> None:
        super().__init__()
        self.ed = editor

    def refresh(self) -> None:
        self.clear()
        for a in self.ed.activity():
            who = {"ai": "AI", "file": "file"}.get(a["who"], "you")
            item = QListWidgetItem(f"{who}: {a['what']}")
            item.setForeground(QColor("#7fb8ff" if who == "AI" else "#d8dee9"))
            self.addItem(item)
        self.scrollToBottom()


class RenderDialog(QDialog):
    """Render: what (a video or one picture), the quality (draft, preview, final, 4K; the
    size and frames a second it gives), where. It renders in the app on a thread (a render
    job), with progress; Cancel stops it and leaves nothing half-written (an earlier file of
    that name stays until the new one is done)."""

    KINDS: ClassVar[dict[str, str]] = {"video": "Video (MP4, or .mov for ProRes)",
                                       "picture": "Picture (PNG)"}  # fmt: skip

    def __init__(self, win: EditorWindow) -> None:
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Render")
        self.kind = QComboBox()
        self._add_kinds()
        self.out = QLineEdit()
        browse = QPushButton("Choose...")
        browse.clicked.connect(self.choose)
        where = QHBoxLayout()
        where.addWidget(self.out, 1)
        where.addWidget(browse)
        self.preset = QComboBox()
        for p, text in (("draft", "Draft (small, 15 fps, plain look: quickest)"),
                        ("preview", "Preview (960 x 540, 30 fps, quick)"),
                        ("final", "Final (the scene's size and frame rate, motion blur)"),
                        ("4k", "4K (3840 x 2160)")):  # fmt: skip
            self.preset.addItem(text, p)
        self.preset.setCurrentIndex(self.preset.findData("preview"))  # the default, as before
        self.size = QLabel("")
        self.fps = QDoubleSpinBox()
        self.fps.setRange(1.0, 240.0)
        self.fps.setDecimals(3)
        self.time = QDoubleSpinBox()
        self.time.setRange(0.0, max(win.ed.engine.duration, 0.0))
        self.time.setDecimals(2)
        self.time.setValue(win.ed.t)
        self.steps = QSpinBox()
        self.steps.setRange(2, 25)
        self.steps.setValue(5)
        form = QFormLayout()
        form.addRow("What", self.kind)
        form.addRow("Save to", where)
        form.addRow("Quality", self.preset)
        form.addRow("Size", self.size)
        form.addRow("Frames a second", self.fps)
        form.addRow("At time (s)", self.time)
        form.addRow("Steps per slider", self.steps)
        self.form = form
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.hide()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.buttons = QDialogButtonBox()
        self.go = self.buttons.addButton("Render", QDialogButtonBox.ButtonRole.AcceptRole)
        self.stop = self.buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self.show_file = self.buttons.addButton("Show File", QDialogButtonBox.ButtonRole.ActionRole)
        self.show_file.hide()
        self.go.clicked.connect(self.start)
        self.stop.clicked.connect(self.cancel)
        self.show_file.clicked.connect(self.reveal)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addWidget(self.buttons)
        self.job: str | None = None
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self._preset_changed()

    def _add_kinds(self) -> None:
        for k, label in self.KINDS.items():
            self.kind.addItem(label, k)

    def _kind(self) -> str:
        return self.kind.currentData()

    def _suffix(self, kind: str) -> str:
        return {"video": ".mp4", "picture": ".png", "web": ".html"}[kind]

    def _preset_changed(self) -> None:
        plan = self.win.ed.render_plan(self.preset.currentData())
        self.size.setText(f"{plan['size'][0]} x {plan['size'][1]}   ({plan['note']})")
        self.fps.setValue(float(plan["fps"]))
        self._kind_changed()

    def _kind_changed(self) -> None:
        k = self._kind()
        self.form.setRowVisible(self.preset, k != "web")
        self.form.setRowVisible(self.size, k != "web")
        self.form.setRowVisible(self.fps, k == "video")
        self.form.setRowVisible(self.time, k == "picture")
        self.form.setRowVisible(self.steps, k == "web")
        if k == "web" and not self.win.ed.engine.document.controls:
            self.status.setText("This scene has no sliders: the page will show only the picture.")
        else:
            self.status.setText("")
        base = Path(self.out.text()) if self.out.text() else self.win.ed.default_output("")
        # a draft never takes the real render's name (as `phasor render --preset draft`)
        draft = ".draft" if k != "web" and self.preset.currentData() == "draft" else ""
        self.out.setText(str(base.with_name(base.name.split(".")[0] + draft + self._suffix(k))))

    def choose(self) -> None:
        k = self._kind()
        suffix = self._suffix(k)
        path, _ = QFileDialog.getSaveFileName(self, "Render to", self.out.text(),
                                              f"{k} (*{suffix});;All files (*)")  # fmt: skip
        if path:
            self.out.setText(path)

    def confirm_replace(self, path: Path) -> bool:
        """Ask before replacing a file (it is replaced only when the new one is done)."""
        box = QMessageBox(self)
        box.setWindowTitle("Replace the file?")
        box.setText(f"{path.name} exists. Replace it when the new render is done?")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def start(self) -> None:
        ed = self.win.ed
        out = Path(self.out.text()).expanduser()
        if not out.name:
            self.status.setText("Choose where to save it.")
            return
        preset = self.preset.currentData()
        overwrite = None
        if out.exists():
            if not self.confirm_replace(out):
                return
            overwrite = True
        fps = self.fps.value() if self._kind() == "video" else None
        try:
            frame = self.time.value() if self._kind() == "picture" else None
            r = ed.start_render(preset, out=out, overwrite=overwrite, fps=fps, frame=frame)
        except ToolError as e:
            self.status.setText(f"Could not start: {e}")
            return
        self.job = r["job"]  # this dialog's own render (the Draft preview has its own)
        self.go.setEnabled(False)
        self.show_file.hide()
        self.bar.setRange(0, 0)
        self.bar.show()
        self.status.setText("Rendering... (you can keep editing; it renders the scene as it was "
                            "when you pressed Render)")  # fmt: skip

    def poll(self) -> None:
        """The window's timer: how far this dialog's own render is."""
        if self.job is None:
            return
        st = self.win.ed.job_status(self.job)
        if st["frames"]:
            self.bar.setRange(0, st["frames"])
            self.bar.setValue(st["frame"])
        if st["state"] in ("done", "failed", "cancelled"):
            self.bar.hide()
            self.go.setEnabled(True)
            self.job = None
            if st["state"] == "done":
                self.status.setText(f"Saved {st['path']}")
                self.show_file.show()
            elif st["state"] == "failed":
                self.status.setText(f"Render failed: {st.get('error', '')}")
            else:
                self.status.setText("Cancelled: nothing was left behind.")
            if st.get("note"):
                self.status.setText(self.status.text() + "  " + st["note"])

    def cancel(self) -> None:
        if self.job is not None:
            self.win.ed.cancel_render(self.job)  # this dialog's render, not the draft's
            self.bar.hide()
            self.go.setEnabled(True)
            self.job = None
            self.status.setText("Cancelled: nothing was left behind.")
        else:
            self.close()

    def reveal(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.out.text()).parent)))


class RowWidget(QWidget):
    """One row of the Explorer: visibility, the kind badge, the text as typed (its error in
    plain words under it), colour, delete; a variable adds its slider, value box, play and
    keep, and the range box (``>``). It forwards everything to the ``Explorer``."""

    RES = 1000

    def __init__(self, panel: ExplorerPanel, row: dict[str, Any]) -> None:
        super().__init__()
        self.panel, self.xp, self.key = panel, panel.xp, row["id"]
        self.row = row
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self.visible = QCheckBox()
        self.visible.setChecked(bool(row.get("visible", True)))
        self.visible.setToolTip("show or hide this row")
        self.visible.toggled.connect(lambda on: panel.done(self.xp.set_visible(self.key, on)))
        self.badge = QComboBox()
        self.badge.setToolTip("the kind this row is read as (choose one over the guess)")
        badge = row["badge"]
        self.badge.addItem(badge["kind"] if badge["override"] else f"{badge['kind']} (guess)", "")
        for k in badge["choices"]:
            self.badge.addItem(k, k)
        self.badge.setEnabled(bool(badge["choices"]) and row["id"] != NEW)
        self.badge.activated.connect(self._kind)
        self.text = QLineEdit(row.get("typing", row.get("source", "")))
        self.text.setPlaceholderText("type an equation: x' = y, rho = 28, (1, 2) ...")
        self.text.setReadOnly(bool(row.get("is_object")) or self.xp.read_only)
        self.text.textEdited.connect(lambda t: panel.typed(self.key, t))
        self.text.editingFinished.connect(lambda: panel.left(self.key))
        self.color = QComboBox()
        self.color.setToolTip("colour (a style role)")
        self.color.addItem("colour", "")
        for role in ROLE_GROUPS.get("emphasis", ()):
            self.color.addItem(role, role)
        self.color.activated.connect(self._color)
        self.remove = QPushButton("x")
        self.remove.setToolTip("delete this row")
        self.remove.setMaximumWidth(28)
        self.remove.clicked.connect(lambda: panel.done(self.xp.delete(self.key)))
        top.addWidget(self.visible)
        top.addWidget(self.badge)
        top.addWidget(self.text, 1)
        top.addWidget(self.color)
        top.addWidget(self.remove)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(top)
        self.error = QLabel(row.get("error") or "")
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #ff6b6b;")
        self.error.setVisible(bool(row.get("error")))
        lay.addWidget(self.error)
        self.slider: QSlider | None = None
        if row["kind"] == "variable" and row["id"] != NEW:
            self._slider(lay, row)
        if row.get("locked"):
            self.setToolTip("you are typing here: the AI cannot change this row until you leave it")

    def _slider(self, lay: QVBoxLayout, row: dict[str, Any]) -> None:
        line = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.RES)
        self.value = QLineEdit(f"{row['value']:.6g}")
        self.value.setMaximumWidth(90)
        self.value.returnPressed.connect(
            lambda: self.panel.done(self.xp.set_value(self.key, self.value.text()))
        )
        self.play = QPushButton("play")
        self.play.setCheckable(True)
        self.play.setChecked(bool(row.get("playing")))
        self.play.toggled.connect(lambda on: self.panel.done(self.xp.slider_play(self.key, on)))
        self.keep = QPushButton("keep")
        self.keep.setToolTip("make the playing slider a track on its value (it renders)")
        self.keep.clicked.connect(lambda: self.panel.done(self.xp.slider_keep(self.key)))
        more = QPushButton(">")
        more.setMaximumWidth(24)
        more.setToolTip("range and step")
        self.range = QWidget()
        rl = QHBoxLayout(self.range)
        rl.setContentsMargins(0, 0, 0, 0)
        self.lo, self.hi = QLineEdit(f"{row['min']:.6g}"), QLineEdit(f"{row['max']:.6g}")
        self.step = QLineEdit(f"{row['step']:.6g}" if row.get("step") else "")
        for label, w in (("min", self.lo), ("max", self.hi), ("step", self.step)):
            rl.addWidget(QLabel(label))
            rl.addWidget(w)
            w.returnPressed.connect(self._range)
        self.range.hide()
        more.clicked.connect(lambda: self.range.setVisible(not self.range.isVisible()))
        self.set_value(row)
        self.slider.sliderMoved.connect(self._moved)
        self.slider.sliderReleased.connect(
            lambda: self.panel.done(self.xp.slider_release(self.key))
        )
        line.addWidget(self.slider, 1)
        for w in (self.value, self.play, self.keep, more):
            line.addWidget(w)
        if row.get("animated"):
            self.slider.setEnabled(False)
            self.value.setToolTip("animated by a track")
        lay.addLayout(line)
        lay.addWidget(self.range)

    def set_value(self, row: dict[str, Any]) -> None:
        if self.slider is None or self.slider.isSliderDown():
            return
        span = (row["max"] - row["min"]) or 1.0
        self.slider.blockSignals(True)
        self.slider.setValue(round((row["value"] - row["min"]) / span * self.RES))
        self.slider.blockSignals(False)
        if not self.value.hasFocus():
            self.value.setText(f"{row['value']:.6g}")

    def _moved(self, v: int) -> None:
        r = self.row
        got = self.xp.slider_drag(self.key, r["min"] + v / self.RES * (r["max"] - r["min"]))
        if got.get("ok"):
            self.value.setText(f"{got['value']:.6g}")
        self.panel.preview.emit()

    def _range(self) -> None:
        self.panel.done(self.xp.slider_range(self.key, min=self.lo.text(), max=self.hi.text(),
                                             step=self.step.text()))  # fmt: skip

    def _kind(self, index: int) -> None:
        self.panel.done(self.xp.set_kind(self.key, self.badge.itemData(index) or ""))

    def _color(self, index: int) -> None:
        role = self.color.itemData(index)
        if role:
            self.panel.done(self.xp.set_color(self.key, role))


class ExportFlowDialog(QDialog):
    """Export a video of the explorer: "from tau 0 to 30 in 20 s", through the normal
    renderer (a render job on a fork; the document is not changed)."""

    def __init__(self, panel: ExplorerPanel) -> None:
        super().__init__(panel)
        self.panel, self.xp = panel, panel.xp
        self.setWindowTitle("Export video")
        self.tau0, self.tau1, self.seconds = QLineEdit("0"), QLineEdit("30"), QLineEdit("20")
        self.preset = QComboBox()
        for p in ("draft", "preview", "final", "4k"):
            self.preset.addItem(p, p)
        self.out = QLineEdit(str(panel.xp.export_path()))
        self.plan = QLabel("")
        self.plan.setWordWrap(True)
        form = QFormLayout()
        form.addRow("From flow time (tau)", self.tau0)
        form.addRow("To flow time (tau)", self.tau1)
        form.addRow("In seconds", self.seconds)
        form.addRow("Quality", self.preset)
        form.addRow("Save to", self.out)
        buttons = QDialogButtonBox()
        self.go = buttons.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        close = buttons.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        self.go.clicked.connect(self.start)
        close.clicked.connect(self.reject)
        for w in (self.tau0, self.tau1, self.seconds):
            w.textChanged.connect(self.check)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.plan)
        lay.addWidget(buttons)
        self.check()

    def check(self) -> dict[str, Any]:
        r = self.xp.export_plan(self.tau0.text(), self.tau1.text(), self.seconds.text())
        if r.get("ok"):
            self.plan.setText("  ".join([r["text"], *r["notes"]]))
        else:
            self.plan.setText("Cannot export: " + r["errors"][0]["message"])
        self.go.setEnabled(bool(r.get("ok")))
        return r

    def confirm_replace(self, path: Path) -> bool:
        """A file of that name exists: replaced only when the person says so (and only when
        the new one is done)."""
        box = QMessageBox(self)
        box.setWindowTitle("Replace the file?")
        box.setText(f"{path.name} exists. Replace it when the new video is done?")
        yes = box.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is yes

    def start(self) -> dict[str, Any]:
        out = Path(self.out.text())
        replace = self.xp.needs_replace(out) and self.confirm_replace(out)
        if self.xp.needs_replace(out) and not replace:
            return {"ok": False, "errors": [{"code": "exists", "message": "not replaced"}]}
        r = self.xp.export(out, self.tau0.text(), self.tau1.text(), self.seconds.text(),
                           self.preset.currentData(), overwrite=replace or None)  # fmt: skip
        self.panel.done(r)
        if r.get("ok"):
            self.plan.setText(f"Exporting {r['plan']['text']} to {r['path']} (the status bar "
                              "follows it)")  # fmt: skip
        return r


class ExplorerPanel(QWidget):
    """The Explorer (Milestone 29, View > Explorer): R2/R3, the ``+ row`` menu with the
    insert palette, presets, the rows with their sliders, the equilibrium card, the
    transport, record/mark/export, Convert to script, and tabs for the analysis and the
    AI's notes. All logic is ``gui.explorer.Explorer``; this only draws it and forwards
    input."""

    changed = Signal(object)  # a result to report (the window refreshes)
    preview = Signal()  # the picture should be drawn again (a slider mid-drag, the transport)
    convert_asked = Signal()

    def __init__(self, explorer: Explorer) -> None:
        super().__init__()
        self.xp = explorer
        self.r2, self.r3 = QPushButton("R2"), QPushButton("R3")
        for b, sp in ((self.r2, "R2"), (self.r3, "R3")):
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, sp=sp: self.switch(sp))
        self.add_button = QPushButton("+ row")
        self.add_menu = QMenu(self)
        self.add_button.setMenu(self.add_menu)
        self.add_menu.aboutToShow.connect(self.fill_add_menu)
        self.presets = QComboBox()
        self.presets.addItem("Presets...", "")
        for p in explorer.presets():
            self.presets.addItem(p["name"].replace("_", " "), p["name"])
            self.presets.setItemData(self.presets.count() - 1, p["note"],
                                     Qt.ItemDataRole.ToolTipRole)  # fmt: skip
        self.presets.activated.connect(self._preset)
        self.convert = QPushButton("Convert to script")
        self.convert.setToolTip(CONVERT_WARNING)
        self.convert.clicked.connect(self.convert_asked.emit)
        head = QHBoxLayout()
        for w in (self.r2, self.r3, self.add_button, self.presets):
            head.addWidget(w)
        head.addStretch(1)
        head.addWidget(self.convert)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.list = QVBoxLayout()
        self.list.setContentsMargins(0, 0, 0, 0)
        holder = QWidget()
        holder.setLayout(self.list)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        self.rows: dict[str, RowWidget] = {}
        self._shape: tuple = ()
        self._typing_key: str | None = None  # the row typed in at the last rebuild
        # the transport: session state, never in the document or its history
        self.back, self.play = QPushButton("<<"), QPushButton("play")
        self.fwd, self.reverse = QPushButton(">>"), QPushButton("reverse")
        self.restart, self.speed = QPushButton("restart"), QLineEdit("1")
        self.speed.setMaximumWidth(60)
        self.readout = QLabel("")
        self.play.setCheckable(True)
        self.reverse.setCheckable(True)
        self.back.clicked.connect(lambda: self._transport(lambda: self.xp.step(-1)))
        self.fwd.clicked.connect(lambda: self._transport(lambda: self.xp.step(1)))
        self.play.clicked.connect(lambda: self._transport(self.xp.toggle_play))
        self.reverse.clicked.connect(lambda: self._transport(self.xp.toggle_reverse))
        self.restart.clicked.connect(lambda: self._transport(self.xp.restart))
        self.speed.returnPressed.connect(
            lambda: self._transport(lambda: self.xp.set_speed(self.speed.text()))
        )
        transport = QHBoxLayout()
        for w in (self.back, self.play, self.fwd, self.reverse, self.restart, QLabel("speed"),
                  self.speed, self.readout):  # fmt: skip
            transport.addWidget(w)
        transport.addStretch(1)
        self.record, self.mark = QPushButton("record"), QPushButton("mark")
        self.record.setCheckable(True)
        self.record.setToolTip("record: the camera and slider values at the moments you mark "
                               "become the animation when you stop (eased between)")  # fmt: skip
        self.record.toggled.connect(lambda on: self.done(self.xp.record(on)))
        self.mark.clicked.connect(lambda: self.done(self.xp.mark()))
        self.export = QPushButton("Export video...")
        self.export.clicked.connect(self.open_export)
        send = QHBoxLayout()
        for w in (self.record, self.mark, self.export):
            send.addWidget(w)
        send.addStretch(1)
        self.card = QLabel("")
        self.card.setWordWrap(True)
        self.card.setStyleSheet("background: #161a23; padding: 6px;")
        self.zoom = QPushButton("zoom in")
        self.zoom.clicked.connect(lambda: self.done(self.xp.zoom_in()))
        card = QHBoxLayout()
        card.addWidget(self.card, 1)
        card.addWidget(self.zoom)
        self.tabs = QTabWidget()
        self.analysis = QListWidget()
        self.lyap = QPushButton("Lyapunov estimate")
        self.lyap.clicked.connect(lambda: self.done(self.xp.lyapunov()))
        self.sweep_name, self.sweep_lo = QLineEdit("rho"), QLineEdit("0")
        self.sweep_hi, self.sweep_n = QLineEdit("30"), QLineEdit("17")
        self.sweep_go = QPushButton("Sweep")
        self.sweep_go.clicked.connect(lambda: self.done(self.xp.sweep(
            self.sweep_name.text(), self.sweep_lo.text(), self.sweep_hi.text(),
            self.sweep_n.text())))  # fmt: skip
        an = QWidget()
        al = QVBoxLayout(an)
        tools = QHBoxLayout()
        tools.addWidget(self.lyap)
        for label, w in (("sweep", self.sweep_name), ("from", self.sweep_lo),
                         ("to", self.sweep_hi), ("n", self.sweep_n)):  # fmt: skip
            tools.addWidget(QLabel(label))
            tools.addWidget(w)
        tools.addWidget(self.sweep_go)
        al.addWidget(self.analysis, 1)
        al.addLayout(tools)
        self.ai_notes = QListWidget()
        self.tabs.addTab(an, "Analysis")
        self.tabs.addTab(self.ai_notes, "AI notes")
        lay = QVBoxLayout(self)
        lay.addLayout(head)
        lay.addWidget(self.note)
        lay.addWidget(scroll, 3)
        lay.addLayout(card)
        lay.addLayout(transport)
        lay.addLayout(send)
        lay.addWidget(self.tabs, 2)
        self.export_dialog: ExportFlowDialog | None = None

    # ---------------------------------------------------------- forwarding

    def done(self, r: Any) -> None:
        self.changed.emit(r)
        self.refresh()

    def typed(self, key: str, text: str) -> None:
        self.xp.type_row(key, text)

    def left(self, key: str) -> None:
        if key in self.xp.drafts:
            self.done(self.xp.leave(key))

    def switch(self, space: str) -> None:
        r = self.xp.switch_space(space)
        if r.get("needs_save"):
            self.xp.message = "Save this explorer first (File > Save): R2 and R3 are two files."
        self.done(r)

    def _preset(self, index: int) -> None:
        name = self.presets.itemData(index)
        self.presets.setCurrentIndex(0)
        if name:
            self.done(self.xp.load_preset(name))

    def fill_add_menu(self) -> None:
        self.add_menu.clear()
        for item in self.xp.add_items():
            a = self.add_menu.addAction(item["label"])
            a.setEnabled(item["why"] is None)
            if item["why"]:
                a.setToolTip(item["why"])
            a.triggered.connect(lambda _=False, k=item["key"]: self.done(self.xp.add(k)))
        insert = self.add_menu.addMenu("Insert")
        for p in self.xp.palette():
            a = insert.addAction(p["label"])
            a.setToolTip(p["about"])
            a.triggered.connect(lambda _=False, t=p["type"]: self.done(self.xp.place(t)))

    def _transport(self, fn) -> None:
        fn()
        self.show_transport()
        self.preview.emit()

    def open_export(self) -> ExportFlowDialog:
        self.export_dialog = ExportFlowDialog(self)
        self.export_dialog.show()
        return self.export_dialog

    # ---------------------------------------------------------- drawing

    def show_sliders(self) -> None:
        """Playing sliders: their handles and values follow (session values)."""
        for e in self.xp.sliders():
            w = self.rows.get(e["id"])
            if w is not None and w.slider is not None:
                w.row = {**w.row, **e}
                w.set_value(w.row)

    def show_transport(self) -> None:
        self.play.setChecked(self.xp.playing)
        self.reverse.setChecked(self.xp.reverse)
        perf = self.xp.perf_text()
        self.readout.setText(f"tau = {self.xp.flow_time():.3f}   {self.xp.speed_text()}"
                             + (f"   {perf}" if perf else ""))  # fmt: skip

    def refresh(self) -> None:
        """Draw the rows from the Explorer: rebuilt when the list's shape changed, else
        updated in place (the row being typed in keeps its cursor)."""
        xp = self.xp
        space = xp.space()
        self.r2.setChecked(space == "R2")
        self.r3.setChecked(space == "R3")
        for w in (self.convert, self.add_button, self.presets):
            w.setEnabled(not xp.read_only)
        self.record.setEnabled(not xp.read_only or xp.recording)
        self.mark.setEnabled(xp.recording)
        self.note.setText(xp.message if not xp.read_only else
                          "A script now: rows are code; sliders still move (tweaks).")  # fmt: skip
        rows = xp.rows()
        if NEW not in xp.drafts and not xp.read_only:
            rows.append({"id": NEW, "kind": "new", "source": "", "error": None, "visible": True,
                         "badge": {"kind": "new", "override": None, "choices": ()}})  # fmt: skip
        # what makes the list rebuild (a row typed in keeps its widget: only its badge and
        # error change while the person types)
        shape = tuple((e["id"],) if e["id"] == NEW else (
            e["id"], e["kind"] if "typing" not in e else "typing", e.get("animated"),
            e.get("source"), e.get("visible")) for e in rows)  # fmt: skip
        if shape != self._shape:
            self._shape = shape
            self._rebuild(rows)
        for e in rows:
            w = self.rows.get(e["id"])
            if w is None:
                continue
            w.row = e
            w.error.setText(e.get("error") or "")
            w.error.setVisible(bool(e.get("error")))
            if "typing" in e and not e["badge"]["override"]:
                w.badge.setItemText(0, f"{e['badge']['kind']} (guess)")
            if e["kind"] == "variable":
                w.set_value(e)
        c = xp.card
        self.card.setText("\n".join(c["lines"]) if c else "Click an equilibrium for its card; "
                          "double-click it to zoom in.")  # fmt: skip
        self.zoom.setEnabled(c is not None)
        self.show_transport()
        self.analysis.clear()
        for e in xp.equilibria_table() if xp._systems() else []:
            self.analysis.addItem("   ".join(e["lines"]))
        for name, a in xp.analysis().items():
            text = json.dumps(a["result"], default=str)[:400] if a["state"] == "done" else (
                a["error"] or a["state"])  # fmt: skip
            self.analysis.addItem(f"{name}{' (stale)' if a['stale'] else ''}: {text}")
        self.ai_notes.clear()
        for n in xp.ai_notes():
            self.ai_notes.addItem(f"AI: {n['what']}")
        if xp.ai_notes():
            self.ai_notes.addItem("")
        for n in reversed(xp.history_names()):
            self.ai_notes.addItem(n)

    def _rebuild(self, rows: list[dict[str, Any]]) -> None:
        """Rebuild the list. The row being typed in keeps its line edit (its cursor,
        selection and undo) when it is still the same kind of row (a commit changed its
        text or id); a row that changed kind (a new row became a variable) gets a new
        widget with the cursor where it was."""
        focus = self.xp.focus
        prev = self.rows.get(focus) if focus is not None else None
        if prev is None and focus is not None:
            prev = self.rows.get(self._typing_key) if self._typing_key else None
        keep, cursor, selection = None, None, None
        if prev is not None:
            cursor = prev.text.cursorPosition()
            if prev.text.hasSelectedText():
                selection = (prev.text.selectionStart(), len(prev.text.selectedText()))
            now = next((e for e in rows if e["id"] == focus), None)
            if (now is not None and now["kind"] == prev.row["kind"]
                    and bool(now.get("is_object")) == bool(prev.row.get("is_object"))):  # fmt: skip
                keep = prev
        while self.list.count():
            item = self.list.takeAt(0)
            w = item.widget()
            if w is not None and w is not keep:
                w.blockSignals(True)
                w.text.blockSignals(True)
                w.deleteLater()
        self.rows = {}
        for e in rows:
            if keep is not None and e["id"] == focus:
                w = keep
                w.key, w.row = focus, e
            else:
                w = RowWidget(self, e)
            self.rows[e["id"]] = w
            self.list.addWidget(w)
        self.list.addStretch(1)
        self._typing_key = focus
        if focus in self.rows:
            t = self.rows[focus].text
            t.setFocus()
            if t is not (keep.text if keep is not None else None):
                t.setCursorPosition(len(t.text()) if cursor is None else min(cursor, len(t.text())))
            if selection is not None and t is not (keep.text if keep else None):
                t.setSelection(*selection)


class EditorWindow(QMainWindow):
    """The code-first window (APP_PLAN.md): the script on the left; on the right the preview,
    the timeline and the tabs (Problems, Inspector, Looks, Narration, Sound and MIDI,
    Footage, Sliders, AI, and tabs other clients add). The toolbar: Run, Draft preview,
    Render, Play, Undo, Redo, Save. "Who changed what" is under View."""

    POLL_FILE_TICKS = 15  # look at the script file about twice a second

    def __init__(self, editor: Editor, attach_hint: str = "") -> None:
        super().__init__()
        self.ed = editor
        self.playing = False
        self.attach_hint = attach_hint
        self.setWindowTitle(f"phasor: {editor.engine.document.meta.title}")
        self.resize(1500, 900)
        self.viewport = Viewport(editor)
        self.timeline = Timeline(editor)
        self.inspector = Inspector(editor)
        self.lint = LintPanel(editor)  # the Problems tab
        self.looks = LooksPanel(editor)
        self.controls = ControlsPanel(editor)
        self.code = CodePanel(editor)
        self.narration = NarrationPanel(editor)
        self.sound = SoundPanel(editor)
        self.footage = FootagePanel(editor)
        self.ai = AIPanel(editor, attach_hint)
        self.activity = ActivityDock(editor)
        # The Explorer (Milestone 29): the left pane shows the script or the rows.
        self.xp = Explorer(editor)
        self.explorer = ExplorerPanel(self.xp)
        self.left = QStackedWidget()
        self.left.addWidget(self.code)
        self.left.addWidget(self.explorer)
        self.readout = QLabel("")
        self.job_bar = QProgressBar()
        self.job_bar.setMaximumWidth(220)
        self.job_bar.setFormat("render %p%")
        self.job_cancel = QPushButton("Cancel render")
        self.job_cancel.clicked.connect(self.cancel_job)
        for w in (self.job_bar, self.job_cancel):
            w.hide()
            self.statusBar().addPermanentWidget(w)
        self.statusBar().addPermanentWidget(self.readout)
        self.render_dialog: RenderDialog | None = None
        self._hover: tuple | None = None
        self._hover_done: tuple | None = None
        self._preview_dirty = False
        self._ticks = 0
        self._asked: dict | None = None  # the conflict last asked about (asked once)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.viewport)
        right.addWidget(self.timeline)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.lint, "Problems")
        self.tabs.addTab(self.inspector, "Inspector")
        self.tabs.addTab(self.looks, "Looks")
        self.tabs.addTab(self.narration, "Narration")
        self.tabs.addTab(self.sound, "Sound and MIDI")
        self.tabs.addTab(self.footage, "Footage")
        self.tabs.addTab(self.controls, "Sliders")
        self.tabs.addTab(self.ai, "AI")
        self.panels = _extra_panels(self)
        for title, widget in self.panels:  # e.g. Sketch (the Explorer is View > Explorer)
            self.tabs.addTab(widget, title)
        right.addWidget(self.tabs)
        right.setStretchFactor(0, 4)
        right.setSizes([480, 150, 260])
        self.activity_dock = QDockWidget("Who changed what", self)
        self.activity_dock.setWidget(self.activity)
        self.activity_dock.setObjectName("activity")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.activity_dock)
        self.activity_dock.hide()  # under View: hidden, not deleted (APP_PLAN)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.left)
        split.addWidget(right)
        split.setStretchFactor(1, 3)
        split.setSizes([620, 880])
        self.setCentralWidget(split)
        self._actions()
        self.viewport.picked.connect(lambda _: self._after_selection())
        self.viewport.dragged.connect(self._drag)
        self.viewport.turned.connect(self._turn)
        self.viewport.zoomed.connect(lambda f: self._report(self.ed.zoom_camera(f)))
        self.viewport.hovered.connect(lambda u, v, size: setattr(self, "_hover", (u, v, size)))
        self.timeline.scrubbed.connect(self._scrub)
        self.timeline.edited.connect(self._report)
        self.controls.moved.connect(lambda: setattr(self, "_preview_dirty", True))
        self.controls.applied.connect(self._report)
        self.code.ran.connect(self._ran)
        self.code.line_clicked.connect(self._line_clicked)
        self.inspector.edited.connect(self._report)
        self.lint.picked.connect(self._jump)
        self.looks.changed.connect(self._report)
        for panel in (self.narration, self.sound, self.footage):
            panel.changed.connect(self._panel_changed)
        self.viewport.explored.connect(self._explored)
        self.explorer.changed.connect(self._explored)
        self.explorer.preview.connect(self.viewport.refresh)
        self.explorer.convert_asked.connect(self.convert_to_script)
        self.statusBar().showMessage(attach_hint or "Ready")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)
        # an explorer document opens in the Explorer (it stays a .phasor.json: EXPLORER.md)
        self.show_explorer(self.xp.is_explorer_document())
        self.refresh_all()

    # ------------------------------------------------------------ the Explorer

    @property
    def explorer_on(self) -> bool:
        return self.left.currentWidget() is self.explorer

    def show_explorer(self, on: bool) -> None:
        """View > Explorer: the rows in the left pane and the viewport's input to the
        Explorer; off: the script and the editor's own viewport."""
        self.left.setCurrentWidget(self.explorer if on else self.code)
        self.viewport.explorer = self.xp if on else None
        act = self.actions.get("Explorer")
        if act is not None and act.isChecked() != on:
            act.setChecked(on)
        if on:
            self.explorer.refresh()
        self.viewport.refresh()

    def confirm_convert(self) -> bool:
        """Asked at the moment of conversion (one-way for structure)."""
        box = QMessageBox(self)
        box.setWindowTitle("Convert to script")
        box.setText(CONVERT_WARNING)
        go = box.addButton("Convert", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is go

    def convert_to_script(self) -> None:
        if self.xp.read_only or not self.confirm_convert():
            return
        self.xp.prepare_convert()
        self.code.refresh(force=True)
        self.code.run()  # on the worker thread, as Run: the window stays alive

    def _explored(self, result: Any) -> None:
        if isinstance(result, dict) and result.get("card"):
            self.statusBar().showMessage(result["card"]["lines"][0], 6000)
        elif self.xp.message:
            self.statusBar().showMessage(self.xp.message, 6000)
        self._report(result)
        if self.ed.changed():
            self.refresh_all()
        else:
            self.viewport.refresh()
            self.explorer.refresh()

    def _actions(self) -> None:
        """The toolbar. Its actions are kept in ``self.actions`` so a menu bar can show the
        same ones (one action per shortcut: Qt ignores a shortcut two actions share)."""
        bar = self.addToolBar("edit")
        bar.setObjectName("edit")
        self.actions: dict[str, QAction] = {}
        for text, key, fn, tip in (
            ("Run", None, self.code.run, "Build the scene from the script (Ctrl+Enter)"),
            ("Draft preview", "Ctrl+Shift+P", self.draft_preview,
             "A quick low-quality render (15 fps, small, no blur, bloom or haze), played "
             "when done"),
            ("Render...", "Ctrl+R", self.open_render, "Render a video or a picture: choose the "
             "quality"),
            ("Play", "Space", self.toggle_play, "Play the preview"),
            ("Undo", "Ctrl+Z", lambda: self._report(self.ed.undo()), ""),
            ("Redo", "Ctrl+Shift+Z", lambda: self._report(self.ed.redo()), ""),
            ("Save", "Ctrl+S", lambda: self.save_scene(), "Save (runs changed code first)"),
        ):  # fmt: skip
            act = QAction(text, self)
            if key:
                act.setShortcut(key)
            if tip:
                act.setToolTip(tip)
            act.triggered.connect(fn)
            bar.addAction(act)
            self.actions[text] = act
        self.actions["Who changed what"] = self.activity_dock.toggleViewAction()
        explorer = QAction("Explorer", self)
        explorer.setCheckable(True)
        explorer.setShortcut("Ctrl+Shift+E")
        explorer.setToolTip("The Explorer: rows, sliders and the flow (the left pane shows "
                            "the rows instead of the script)")  # fmt: skip
        explorer.toggled.connect(self.show_explorer)
        bar.addAction(explorer)
        self.actions["Explorer"] = explorer

    def save_scene(self) -> None:
        """Ctrl+S. The desktop app asks where when the scene has never been saved. When the
        script changed on disk meanwhile, nothing is written until the person chooses."""
        self.xp.settle()  # a row typed just now is saved with the rest
        try:
            if self.ed.save_needs_run() and self.ed.conflict is None:
                # Run on save, on the worker thread: a long script never freezes the window
                # (a Run that succeeds is saved; its outcome shows when it is over)
                self.code.run()
                if self.ed.running is None:
                    self._ask_pending_conflict(then_save=True)
                return
            r = self.ed.save()
            if isinstance(r, dict) and r.get("conflict"):
                self._ask_pending_conflict(then_save=True)
                return
            self._report(r)
        except (ToolError, OSError) as e:  # a read-only file, a full disk: said, not raised
            self.statusBar().showMessage(f"Not saved: {e}", 8000)
        self.code.refresh()

    # ------------------------------------------------------------ renders

    def draft_preview(self) -> None:
        """The low-quality preview render: in the background, progress in the status bar,
        played in the system's player when it is done."""
        try:
            self.ed.draft_preview()
        except ToolError as e:
            self.statusBar().showMessage(f"No draft: {e}", 8000)
            return
        self.job_bar.setRange(0, 0)
        self.job_bar.show()
        self.job_cancel.show()
        self.statusBar().showMessage("Rendering a draft...", 4000)

    def open_render(self) -> None:
        if self.render_dialog is None or not self.render_dialog.isVisible():
            self.render_dialog = RenderDialog(self)
        self.render_dialog.show()
        self.render_dialog.raise_()

    def cancel_job(self) -> None:
        r = self.ed.cancel_job()
        self.job_bar.hide()
        self.job_cancel.hide()
        if r is not None:
            self.statusBar().showMessage("Render cancelled: nothing was left behind.", 6000)

    def play_file(self, path: str) -> None:
        """Play a finished draft in the system's default player."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _poll_job(self) -> None:
        st = self.ed.poll_job()
        if st is None:
            return
        if st["frames"]:
            self.job_bar.setRange(0, st["frames"])
            self.job_bar.setValue(st["frame"])
        self.job_bar.show()
        self.job_cancel.show()
        if st["state"] in ("done", "failed", "cancelled"):
            self.job_bar.hide()
            self.job_cancel.hide()
            if st["state"] == "done":
                self.statusBar().showMessage(f"Rendered {st['path']}", 8000)
                if st.get("open"):
                    self.play_file(st["open"])
            elif st["state"] == "failed":
                self.statusBar().showMessage(f"Render failed: {st.get('error', '')}", 12000)
            if st.get("note"):
                self.statusBar().showMessage(st["note"], 12000)

    # ------------------------------------------------------------ the script file

    def ask_conflict(self) -> str:
        """The script changed on disk while there are unsaved edits here: which to keep."""
        box = QMessageBox(self)
        if (self.ed.conflict or {}).get("deleted"):
            box.setWindowTitle("The script was deleted")
            box.setText("The script's file was deleted (or moved) outside the app. Write your "
                        "text there again, or keep it here unsaved?")  # fmt: skip
            mine = box.addButton("Write it again", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Keep it unsaved", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            return "mine" if box.clickedButton() is mine else "file"
        box.setWindowTitle("The script changed on disk")
        box.setText("The script was changed outside the app (another editor, or an AI) while "
                    "you have unsaved edits here. Which version do you keep?")  # fmt: skip
        mine = box.addButton("Keep mine", QMessageBox.ButtonRole.RejectRole)
        box.addButton("Take the file's", QMessageBox.ButtonRole.AcceptRole)
        box.exec()
        return "mine" if box.clickedButton() is mine else "file"

    def _ask_pending_conflict(self, *, then_save: bool = False) -> None:
        """The file changed on disk while there are unsaved edits here (found by the file
        watch, or refused by Save, Run or a panel): ask once which version to keep."""
        conflict = self.ed.conflict
        if conflict is None or self._asked is conflict:
            return
        self._asked = conflict
        choice = self.ask_conflict()
        r = self.ed.resolve_conflict(choice, background=True)
        if choice == "mine":
            self.statusBar().showMessage("Kept your version: saving writes it over the "
                                         "file's", 8000)  # fmt: skip
            if then_save:
                self.save_scene()
        elif r.get("event") == "reloading":
            self.code.sync_state()
        self.refresh_all()

    def _poll_file(self) -> None:
        r = self.ed.poll_file(background=True)
        if r is None:
            return
        if r["event"] == "conflict":
            self._ask_pending_conflict()
            return
        if r["event"] == "reloading":
            self.code.sync_state()
            self.statusBar().showMessage("The script changed on disk: running it...", 4000)
            return
        self._reloaded(r)

    def _reloaded(self, r: dict) -> None:
        msg = ("The script changed on disk: reloaded." if r["ok"] else
               f"The script changed on disk and does not run: {r['error']}")  # fmt: skip
        self.statusBar().showMessage(msg, 8000)
        self._ran(r)

    def _poll_run(self) -> None:
        """A run on its worker thread is over: show it (a Run, or a reload)."""
        r = self.ed.poll_run()
        if r is None:
            return
        if r.get("event") == "reload":
            self.code.stop.hide()
            self._reloaded(r)
        else:
            self.code.finish(r)

    def wait_for_run(self, timeout: float = 60.0) -> None:
        """Tick until a run on its worker thread is over (tests, and closing the window)."""
        import time

        end = time.monotonic() + timeout
        while self.ed.running is not None and time.monotonic() < end:
            self.ed.running.done.wait(0.05)
            self._poll_run()

    # ------------------------------------------------------------ events

    def toggle_play(self) -> None:
        if self.explorer_on:  # the Explorer's transport (session state)
            self.xp.toggle_play()
            self.explorer.show_transport()
            return
        self.playing = not self.playing
        if self.playing and self.ed.t >= self.ed.engine.duration:
            self.ed.set_time(0.0)

    def _turn(self, dx: float, dy: float, size: tuple, pan: bool) -> None:
        self._report(self.ed.drag_camera(dx, dy, size, pan))

    def _ran(self, result: dict) -> None:
        self.setWindowTitle(f"phasor: {self.ed.engine.document.meta.title}")
        self.refresh_all()

    def _panel_changed(self, result: dict) -> None:
        self._report(result)
        self.refresh_all()

    def _line_clicked(self, line: int) -> None:
        if self.ed.select_line(line) is not None:
            self.viewport.refresh()
            self.inspector.refresh()

    def _show_readout(self) -> None:
        """The value under the pointer (f(z), escape count, coordinates) in the status bar."""
        if self._hover is None or self._hover == self._hover_done:
            return
        self._hover_done = self._hover
        try:
            self.readout.setText(self.ed.readout(*self._hover)["text"])
        except Exception:
            self.readout.setText("")

    def tick(self) -> None:
        self._ticks += 1
        if self.explorer_on:
            changed = self.xp.poll()
            moved = self.xp.advance(TICK_MS / 1000)
            if changed or moved:
                self.viewport.refresh()
                self.explorer.show_transport()
            if moved and self.xp.playing_sliders:
                self.explorer.show_sliders()
            if changed:
                self.explorer.refresh()
        self._show_readout()
        if self._preview_dirty:
            self._preview_dirty = False
            self.viewport.refresh()
        if self.playing and not self.explorer_on:
            self.playing = self.ed.play_step(TICK_MS / 1000)
            self.viewport.refresh()
            self.timeline.update()
        self.code.sync_state()
        if self.ed.running is not None:
            self._poll_run()
        elif self.ed.conflict is not None:  # refused by Save, Run or a panel: ask once
            self._ask_pending_conflict()
        elif self._ticks % self.POLL_FILE_TICKS == 0:
            self._poll_file()
        if self.ed.job is not None:
            self._poll_job()
        if self.xp.export_job is not None:  # the Explorer's export: its own job
            st = self.xp.export_status()
            if st is not None:
                msg = {"done": f"Exported {st.get('path', '')}",
                       "failed": f"Export failed: {st.get('error', '')}",
                       "cancelled": "Export cancelled: nothing was left behind."}.get(
                    st["state"], f"Exporting... {round(100 * st.get('progress', 0))}%")  # fmt: skip
                self.statusBar().showMessage(msg, 8000)
        if self.render_dialog is not None and self.render_dialog.job is not None:
            self.render_dialog.poll()
        if self.ed.changed():  # an edit here, by an attached AI, or the file reloaded
            self.refresh_all()

    def _scrub(self, t: float) -> None:
        self.ed.set_time(t)
        self.viewport.refresh()
        self.timeline.update()
        self.inspector.refresh()

    def _after_selection(self) -> None:
        self.viewport.refresh()
        self.inspector.refresh()
        self.code.show_line(self.ed.line_of())

    def _drag(self, oid: str, u: float, v: float, size: tuple) -> None:
        try:
            self._report(self.ed.drag_to(oid, u, v, size))
        except ToolError as e:
            self.statusBar().showMessage(str(e), 6000)

    def _jump(self, issue: dict) -> None:
        self.ed.jump(issue)
        self.refresh_all()
        self.code.show_line(issue.get("line") or self.ed.line_of())

    def _report(self, result: Any) -> None:
        if isinstance(result, dict) and result.get("ok") is False:
            errs = result.get("errors") or [{"message": "refused"}]
            self.statusBar().showMessage("Not applied: " + errs[0].get("message", ""), 8000)
        elif isinstance(result, dict) and result.get("saved"):
            self.statusBar().showMessage(f"Saved {result['saved']}", 4000)
        self.code.refresh()

    def refresh_all(self) -> None:
        self.viewport.refresh()
        self.timeline.refresh()
        self.inspector.refresh()
        self.lint.refresh()
        self.looks.refresh()
        self.controls.refresh()
        self.code.refresh()
        self.narration.refresh()
        self.sound.refresh()
        self.footage.refresh()
        self.activity.refresh()
        if self.explorer_on:
            self.explorer.refresh()

    def closeEvent(self, ev) -> None:
        self.timer.stop()
        self.xp.settle()  # a typed row and a wheel zoom at rest are committed, not dropped
        self.xp.cancel_export()
        self.xp.close()
        if self.ed.job is not None:
            self.ed.cancel_job()
        if self.ed.running is not None:
            self.ed.cancel_run()
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


def make_workspace(path: str | None, root: str | None = None):
    """A workspace with ``path`` open, or a new scene (``controller.open_workspace``: a .py
    the person opens runs, on the worker thread, with a progress dialog and Cancel after half
    a second, and the run timeout; a .phasor.json opens without running the modules it
    names)."""
    from phasor.gui.controller import open_workspace

    return open_workspace(path, root, wait=wait_with_dialog)


def wait_with_dialog(task) -> bool:
    """Wait for a script that is opening, the window alive: "Running scene.py..." with a
    Cancel button once it takes more than half a second."""
    from PySide6.QtWidgets import QApplication, QProgressDialog

    from phasor.gui.controller import wait_task

    if QApplication.instance() is None:  # no window yet (a script, a test): just wait
        return wait_task(task)
    dlg = QProgressDialog(f"Running {task.path.name}...", "Cancel", 0, 0)
    dlg.setWindowTitle("Opening a script")
    dlg.setMinimumDuration(500)
    try:
        return wait_task(task, canceled=dlg.wasCanceled, tick=QApplication.processEvents)
    finally:
        dlg.close()
        dlg.deleteLater()


def start_ai_server(ws, lock):
    """Let ``phasor mcp --attach --root <ws.root>`` edit this scene. Returns (server, hint)."""
    from phasor.workspace.remote import RemoteServer

    server = RemoteServer(ws, lock).start()
    server.write_handoff()
    return server, f"An AI can edit this scene: phasor mcp --attach --root {ws.root}"


def qt_app():
    """The one QApplication, in the house look."""
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from phasor.gui.app import ICON

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(DARK)
    app.setApplicationName("phasor")
    if ICON.exists():
        app.setWindowIcon(QIcon(str(ICON)))
    return app


def open_editor(path: str | None, root: str | None = None, *, serve_ai: bool = True,
                show: bool = True):  # fmt: skip
    """Open the editor on ``path`` (see ``make_workspace``). Returns (app, window, server)."""
    import threading

    app = qt_app()
    ws = make_workspace(path, root)
    lock = threading.RLock()
    editor = Editor(ws, lock)
    server, hint = None, ""
    if serve_ai:
        server, hint = start_ai_server(ws, lock)
    win = EditorWindow(editor, hint)
    if show:
        win.show()
    return app, win, server
