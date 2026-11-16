"""
Review Mode B pseudo labels into verified ground truth.

    .venv/bin/python tools/label_review.py [data/finetune/ipad_20261116]

Mouse: new box (class from the picker) = drag on empty space, or Shift+drag / right-drag anywhere, or any drag
in Draw mode (D); drag a box = move, drag a corner = resize, click = select. "+ Add box" (A) puts a box in the
middle of the frame to move / resize. Class picker: type to search (any part of the name), Enter or click picks;
the pick relabels the selected box AND becomes the class of new boxes (kept across frames). Esc leaves it.
Keys: Enter = save + next, S = skip frame (menu/unreadable), Delete/Backspace = delete box,
Left / Right = previous / next frame (no save), C = focus the class picker. Keys typed in the picker stay there.
Dashed red boxes are detections the eval dropped (card in neither deck): click one, pick the right class
and press "Use suggestion", or ignore it. Saved labels go to reviewed/; the pseudo labels are not changed.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shiboken6  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal  # noqa: E402
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QPen, QPixmap, QShortcut  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QCompleter, QFileDialog, QGraphicsItem,  # noqa: E402
                               QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView,
                               QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from backend_engine.perception.label_review import CROP_H, CROP_W, TOWERS, Box, LabelSet  # noqa: E402
from frontend_app.qt.theme import C, STYLE, cls, dark_palette  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET = ROOT / "data" / "finetune" / "ipad_20261116"
MODEL_DIR = ROOT / "pretrained_weights" / "sudhishneel_v8"
CLASS_MAP = ROOT / "configs" / "detector_class_map.json"
TARGET = 300
HANDLE = 7.0


class BoxItem(QGraphicsRectItem):
    """A label box: move by dragging inside, resize by dragging a corner."""

    def __init__(self, box: Box, name: str, color: str, suggestion: bool = False):
        super().__init__(QRectF(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0))
        self.cls, self.suggestion, self.color = box.cls, suggestion, color
        self.setFlags(QGraphicsItem.ItemIsSelectable | (QGraphicsItem.GraphicsItemFlag(0) if suggestion
                                                        else QGraphicsItem.ItemIsMovable))
        self.text = QGraphicsSimpleTextItem(self)
        self.text.setFont(QFont("Helvetica", 9, QFont.Bold))
        self._corner: Optional[int] = None
        self.set_class(box.cls, name)

    def set_class(self, c: int, name: str) -> None:
        self.cls = c
        self.text.setText(("? " if self.suggestion else "") + name)
        self._restyle()

    def _restyle(self) -> None:
        col = QColor(C["violet"] if self.isSelected() else self.color)
        pen = QPen(col, 2, Qt.DashLine if self.suggestion else Qt.SolidLine)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.text.setBrush(QBrush(col))
        r = self.rect()
        self.text.setPos(r.left(), r.top() - 13)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._restyle()
        return super().itemChange(change, value)

    def box(self) -> Box:
        r = self.mapRectToScene(self.rect())
        return Box(self.cls, r.left(), r.top(), r.right(), r.bottom()).clamped()

    def _corner_at(self, p: QPointF) -> Optional[int]:
        r = self.rect()
        for i, c in enumerate((r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight())):
            if abs(p.x() - c.x()) <= HANDLE and abs(p.y() - c.y()) <= HANDLE:
                return i
        return None

    def mousePressEvent(self, e):
        self._corner = None if self.suggestion else self._corner_at(e.pos())
        if self._corner is None:
            super().mousePressEvent(e)
        else:
            self.setSelected(True)
            e.accept()

    def mouseMoveEvent(self, e):
        if self._corner is None:
            super().mouseMoveEvent(e)
            return
        r, p = QRectF(self.rect()), e.pos()
        if self._corner in (0, 2):
            r.setLeft(p.x())
        else:
            r.setRight(p.x())
        if self._corner in (0, 1):
            r.setTop(p.y())
        else:
            r.setBottom(p.y())
        self.setRect(r.normalized())
        self._restyle()

    def mouseReleaseEvent(self, e):
        self._corner = None
        super().mouseReleaseEvent(e)


class Canvas(QGraphicsView):
    """Shows one 568x896 frame. A new box is drawn by: dragging on empty space, Shift+drag or right-drag
    anywhere (even over a box), or any left drag while draw mode is on."""
    drawn = Signal(QRectF)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(0, 0, CROP_W, CROP_H))
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(QColor(C["bg"]))
        self.setMinimumSize(240, 380)
        self.setFocusPolicy(Qt.StrongFocus)
        self.draw_mode = False
        self._start: Optional[QPointF] = None
        self._rubber: Optional[QGraphicsRectItem] = None

    def set_draw_mode(self, on: bool) -> None:
        self.draw_mode = on
        self.viewport().setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def _wants_draw(self, e, item) -> bool:
        if e.button() == Qt.RightButton:
            return True
        if e.button() != Qt.LeftButton:
            return False
        if self.draw_mode or e.modifiers() & Qt.ShiftModifier:
            return True
        return item is None or item.zValue() < 0          # empty space or the frame picture itself

    def contextMenuEvent(self, e):                        # right button is for drawing, no menu
        e.accept()

    def mousePressEvent(self, e):
        self.setFocus()
        p = self.mapToScene(e.position().toPoint())
        item = self.itemAt(e.position().toPoint())
        if self._wants_draw(e, item):
            self.scene().clearSelection()
            self._start = p
            pen = QPen(QColor(C["gold"]), 1, Qt.DashLine)
            pen.setCosmetic(True)
            self._rubber = self.scene().addRect(QRectF(p, p), pen)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._start is not None and self._rubber is not None:
            self._rubber.setRect(QRectF(self._start, self.mapToScene(e.position().toPoint())).normalized())
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._start is not None and self._rubber is not None:
            r = self._rubber.rect()
            self.scene().removeItem(self._rubber)
            self._start = self._rubber = None
            if r.width() >= 4 and r.height() >= 4:
                self.drawn.emit(r)
            return
        super().mouseReleaseEvent(e)


class ReviewWindow(QMainWindow):
    def __init__(self, labels: LabelSet):
        super().__init__()
        self.ls = labels
        self.queue: List[str] = labels.queue()
        self.pos = 0
        self.items: List[BoxItem] = []
        self.setWindowTitle("Label review")

        self.canvas = Canvas()
        self.canvas.drawn.connect(self._add_drawn)
        self.canvas.scene().selectionChanged.connect(self._selection_changed)

        side = QWidget()
        side.setFixedWidth(270)
        v = QVBoxLayout(side)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        self.title = cls(QLabel(), "h2")
        self.sub = cls(QLabel(), "sub")
        self.sub.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setRange(0, TARGET)
        self.bar.setTextVisible(True)
        v.addWidget(self.title)
        v.addWidget(self.sub)
        v.addWidget(self.bar)

        v.addWidget(cls(QLabel("Class for new / selected box"), "section"))
        self.new_class: Optional[int] = None     # class of the next new box; kept across frames
        self.picker = QComboBox()
        self.picker.setEditable(True)
        self.picker.setInsertPolicy(QComboBox.NoInsert)
        self.picker.lineEdit().setPlaceholderText("type to search, Enter to pick")
        self.picker.completer().setCompletionMode(QCompleter.PopupCompletion)
        self.picker.completer().setFilterMode(Qt.MatchContains)
        self.picker.completer().setCaseSensitivity(Qt.CaseInsensitive)
        self.picker.activated.connect(lambda i: self._choose(self.picker.itemData(i)))
        self.picker.completer().activated.connect(lambda text: self._choose(self._match_class(text)))
        self.picker.lineEdit().returnPressed.connect(self._pick_typed)
        self.picker.lineEdit().installEventFilter(self)
        v.addWidget(self.picker)
        self.new_label = cls(QLabel(), "sub")
        v.addWidget(self.new_label)

        tools = QHBoxLayout()
        self.b_draw = QPushButton("✎ Draw (D)")
        self.b_draw.setCheckable(True)
        self.b_draw.toggled.connect(self._draw_toggled)
        self.b_draw.setToolTip("Every left drag draws a new box (Shift+drag or right-drag also always draw)")
        self.b_center = QPushButton("+ Add box (A)")
        self.b_center.clicked.connect(self.add_center_box)
        self.b_center.setToolTip("Add a box at the centre of the frame, then move / resize it")
        tools.addWidget(self.b_draw)
        tools.addWidget(self.b_center)
        v.addLayout(tools)

        v.addWidget(cls(QLabel("Boxes"), "section"))
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._list_selected)
        self.list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self.list, 1)

        row = QHBoxLayout()
        self.b_del = QPushButton("Delete box")
        self.b_del.clicked.connect(self._delete_selected)
        self.b_use = QPushButton("Use suggestion")
        self.b_use.clicked.connect(self._use_suggestion)
        row.addWidget(self.b_del)
        row.addWidget(self.b_use)
        v.addLayout(row)

        self.b_save = cls(QPushButton("Save + next  (Enter)"), "primary")
        self.b_save.clicked.connect(self.save_next)
        self.b_skip = QPushButton("Skip frame  (S)")
        self.b_skip.clicked.connect(self.skip)
        nav = QHBoxLayout()
        b_prev, b_next = QPushButton("◀ Prev"), QPushButton("Next ▶")
        b_prev.clicked.connect(lambda: self.go(self.pos - 1))
        b_next.clicked.connect(lambda: self.go(self.pos + 1))
        nav.addWidget(b_prev)
        nav.addWidget(b_next)
        self.b_export = cls(QPushButton("Export training set…"), "ghost")
        self.b_export.clicked.connect(self.export)
        for w in (self.b_save, self.b_skip):
            v.addWidget(w)
        v.addLayout(nav)
        v.addWidget(self.b_export)
        help_ = cls(QLabel("New box: drag empty space, Shift+drag or right-drag anywhere, or Draw mode · "
                           "drag corner: resize · Del: delete · ←/→: browse · Esc: leave the class box"), "sub")
        help_.setWordWrap(True)
        v.addWidget(help_)

        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self.canvas, 1)
        h.addWidget(side)
        self.setCentralWidget(central)

        for keys, fn in ((Qt.Key_Return, self.save_next), (Qt.Key_Enter, self.save_next),
                         (Qt.Key_S, self.skip), (Qt.Key_Delete, self._delete_selected),
                         (Qt.Key_Backspace, self._delete_selected),
                         (Qt.Key_Left, lambda: self.go(self.pos - 1)), (Qt.Key_Right, lambda: self.go(self.pos + 1)),
                         (Qt.Key_C, self.focus_picker),
                         (Qt.Key_D, self.b_draw.toggle), (Qt.Key_A, self.add_center_box)):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(fn)
        self.go(0)
        self.canvas.setFocus()

    def eventFilter(self, obj, e):
        """While typing in the class box, keys belong to the text (no skip on 's', no save on Enter)."""
        if obj is self.picker.lineEdit():
            if e.type() == QEvent.ShortcutOverride:
                e.accept()
                return True
            if e.type() == QEvent.KeyPress and e.key() == Qt.Key_Escape:
                self.canvas.setFocus()
                return True
        return super().eventFilter(obj, e)

    # -- frame loading ------------------------------------------------------------
    @property
    def stem(self) -> Optional[str]:
        return self.queue[self.pos] if 0 <= self.pos < len(self.queue) else None

    def go(self, pos: int) -> None:
        if not self.queue:
            self.title.setText("Nothing left to review")
            self.sub.setText("Every frame in the queue is saved or skipped. Export the training set.")
            return
        self.pos = max(0, min(len(self.queue) - 1, pos))
        stem = self.stem
        sc = self.canvas.scene()
        self.items = []          # drop references first: clear() deletes the C++ items and emits selectionChanged
        sc.clear()
        bg = sc.addPixmap(QPixmap(str(self.ls.image_path(stem))))
        bg.setZValue(-1)
        for b in self.ls.boxes(stem):
            self._add_item(b)
        if not self.ls.is_reviewed(stem):           # once saved, the human decision replaces the suggestions
            for b in self.ls.suggestions(stem):
                self._add_item(b, suggestion=True)
        self.picker.blockSignals(True)
        self.picker.clear()
        for c in self.ls.class_choices(stem):
            tag = "★ " if self.ls.in_deck(stem, c) else ""
            self.picker.addItem(tag + self.ls.names[c], c)
        self.picker.blockSignals(False)
        if self.new_class is None:                  # first frame: the top deck card
            self.new_class = self.picker.itemData(0)
        self._show_picker(self.new_class)
        t = self.ls.time_ms(stem) / 1000
        match_no = self.ls.matches.index(self.ls.match_of(stem)) + 1
        status = {"done": "saved", "skipped": "skipped"}.get(self.ls.state.get(stem, ""), "not reviewed")
        self.title.setText(f"Match {match_no} · {int(t // 60)}:{t % 60:04.1f}")
        self.sub.setText(f"Frame {self.pos + 1} of {len(self.queue)} in queue · {status}\n"
                         f"★ = card in this match's decks")
        p = self.ls.progress()
        self.bar.setValue(min(TARGET, p["done"]))
        self.bar.setFormat(f"{p['done']} / {TARGET} reviewed")
        self._refresh_list()

    def _add_item(self, b: Box, suggestion: bool = False) -> BoxItem:
        name = self.ls.names[b.cls]
        color = C["loss"] if suggestion else C["muted"] if name in TOWERS else C["teal"]
        it = BoxItem(b, name, color, suggestion)
        self.canvas.scene().addItem(it)
        self.items.append(it)
        return it

    def _refresh_list(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for it in self.items:
            self.list.addItem(("? " if it.suggestion else "") + self.ls.names[it.cls])
        sel = [i for i, it in enumerate(self.items) if it.isSelected()]
        if sel:
            self.list.setCurrentRow(sel[0])
        self.list.blockSignals(False)

    # -- editing -------------------------------------------------------------------
    def _selected(self) -> Optional[BoxItem]:
        self.items = [it for it in self.items if shiboken6.isValid(it)]   # scene teardown deletes items
        return next((it for it in self.items if it.isSelected()), None)

    def _selection_changed(self) -> None:
        it = self._selected()
        # show the selected box's class; with nothing selected, show the class for new boxes
        self._show_picker(it.cls if it is not None else self.new_class)
        self._refresh_list()

    def _list_selected(self, row: int) -> None:
        if 0 <= row < len(self.items):
            self.canvas.scene().clearSelection()
            self.items[row].setSelected(True)

    # -- class picker ------------------------------------------------------------------
    def _show_picker(self, c: Optional[int]) -> None:
        """Display a class in the picker without treating it as a user choice."""
        i = self.picker.findData(c) if c is not None else -1
        if i >= 0:
            self.picker.blockSignals(True)
            self.picker.setCurrentIndex(i)
            self.picker.blockSignals(False)
        name = self.ls.names[self.new_class] if self.new_class is not None else "none"
        self.new_label.setText(f"New boxes: {name}")

    def _match_class(self, text: str) -> Optional[int]:
        """Typed text -> class: exact name first, else the first class containing it (deck cards first)."""
        t = text.replace("★", "").strip().lower()
        if not t:
            return None
        rows = [(self.picker.itemData(i), self.ls.names[self.picker.itemData(i)]) for i in range(self.picker.count())]
        exact = next((c for c, n in rows if n == t), None)
        if exact is not None:
            return exact
        return next((c for c, n in rows if t in n or t.replace(" ", "-") in n), None)

    def _pick_typed(self) -> None:
        c = self._match_class(self.picker.currentText())
        if c is None:
            self.new_label.setText(f"No class matches '{self.picker.currentText().strip()}'")
            return
        self._choose(c)
        self.canvas.setFocus()                      # Enter again now saves the frame

    def _choose(self, c: Optional[int]) -> None:
        """A user choice: becomes the class for new boxes AND relabels the selected box."""
        if c is None:
            return
        self.new_class = c
        it = self._selected()
        if it is not None and not it.suggestion:
            it.set_class(c, self.ls.names[c])
        self._show_picker(c)
        self._refresh_list()

    def focus_picker(self) -> None:
        self.picker.setFocus()
        self.picker.lineEdit().selectAll()

    def _draw_toggled(self, on: bool) -> None:
        self.canvas.set_draw_mode(on)
        self.b_draw.setText("✎ Drawing… (D)" if on else "✎ Draw (D)")

    # -- editing -------------------------------------------------------------------
    def _new_box(self, x0: float, y0: float, x1: float, y1: float) -> Optional[BoxItem]:
        if self.new_class is None or self.stem is None:
            self.new_label.setText("Pick a class first (C), then add the box.")
            return None
        it = self._add_item(Box(self.new_class, x0, y0, x1, y1).clamped())
        self.canvas.scene().clearSelection()
        it.setSelected(True)
        return it

    def _add_drawn(self, r: QRectF) -> None:
        self._new_box(r.left(), r.top(), r.right(), r.bottom())

    def add_center_box(self) -> None:
        """A 50x60 box in the middle of the frame (the view always shows the whole frame): move / resize it."""
        x, y = CROP_W / 2, CROP_H / 2
        self._new_box(x - 25, y - 30, x + 25, y + 30)
        self.canvas.setFocus()

    def _delete_selected(self) -> None:
        it = self._selected()
        if it is None:
            return
        self.items.remove(it)                       # first: removeItem emits selectionChanged
        self.canvas.scene().removeItem(it)
        self._refresh_list()

    def _use_suggestion(self) -> None:
        it = self._selected()
        if it is None or not it.suggestion:
            return
        b = it.box()
        shown = self.picker.currentData()          # what you picked, else the suggestion's own class
        c = shown if shown is not None else b.cls  # (read before deleting: that resets the picker)
        self._delete_selected()
        self.canvas.scene().clearSelection()
        self._add_item(Box(c, b.x0, b.y0, b.x1, b.y1)).setSelected(True)

    # -- review actions --------------------------------------------------------------
    def save_next(self) -> None:
        if self.stem is None:
            return
        self.ls.save(self.stem, [it.box() for it in self.items if not it.suggestion])
        self.go(self.pos + 1)

    def skip(self) -> None:
        if self.stem is None:
            return
        self.ls.skip(self.stem)
        self.go(self.pos + 1)

    def export(self) -> None:
        if self.ls.progress()["done"] == 0:
            QMessageBox.information(self, "Export", "Save at least one reviewed frame first.")
            return
        out = QFileDialog.getExistingDirectory(self, "Export training set to", str(self.ls.root.parent))
        if not out:
            return
        target = Path(out) / f"{self.ls.root.name}_reviewed"
        n = self.ls.export(target)
        zipped = shutil.make_archive(str(target), "zip", target)
        QMessageBox.information(self, "Export", f"Wrote {n['train']} train and {n['val']} validation frames "
                                                f"({n['boxes']} boxes). Validation = the last match.\n\n"
                                                f"Upload {Path(zipped).name} to "
                                                f"cloud_training/finetune_detector_colab.ipynb.")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SET
    app = QApplication(sys.argv)
    app.setPalette(dark_palette())
    app.setStyleSheet(STYLE)
    w = ReviewWindow(LabelSet.open(root, MODEL_DIR, CLASS_MAP))
    w.resize(860, 960)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
