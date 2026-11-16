"""Label review window (tools/label_review.py): drawing, class picking and keyboard safety, driven offscreen."""
import sys
from pathlib import Path

import pytest

from tests.test_label_review import make_set

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import label_review as lr  # noqa: E402

KNIGHT, ZAP, QUEEN = 5, 3, 1


@pytest.fixture
def win(tmp_path):
    app = QApplication.instance() or QApplication([])
    w = lr.ReviewWindow(make_set(tmp_path))
    w.resize(860, 960)
    w.show()
    app.processEvents()
    yield w
    w.close()


def view_pt(w, x, y) -> QPoint:
    """Scene (crop px) -> viewport point."""
    return w.canvas.mapFromScene(x, y)


def drag(w, a, b, button=Qt.LeftButton, mods=Qt.NoModifier):
    vp = w.canvas.viewport()
    QTest.mousePress(vp, button, mods, view_pt(w, *a))
    QTest.mouseMove(vp, view_pt(w, *b))
    QTest.mouseRelease(vp, button, mods, view_pt(w, *b))


def tower(w):
    return next(it for it in w.items if w.ls.names[it.cls] == "queen-tower")


def test_shift_and_right_drag_draw_over_a_box(win):
    t = tower(win)
    r = t.mapRectToScene(t.rect())
    before = (len(win.items), r)
    inside = (r.center().x(), r.center().y())
    end = (r.center().x() + 30, r.center().y() + 30)
    drag(win, inside, end, mods=Qt.ShiftModifier)
    assert len(win.items) == before[0] + 1
    assert t.mapRectToScene(t.rect()) == before[1]          # the tower box did not move
    drag(win, inside, end, button=Qt.RightButton)
    assert len(win.items) == before[0] + 2
    assert win.items[-1].isSelected() and win.items[-1].cls == win.new_class


def test_draw_mode_left_drag_inside_box(win):
    t = tower(win)
    t.setSelected(True)                                      # a selected box no longer blocks drawing
    QTest.keyClick(win.canvas, Qt.Key_D)
    assert win.canvas.draw_mode and win.b_draw.isChecked()
    c = t.mapRectToScene(t.rect()).center()
    n = len(win.items)
    drag(win, (c.x(), c.y()), (c.x() + 25, c.y() + 40))
    assert len(win.items) == n + 1
    win.b_draw.click()
    assert not win.canvas.draw_mode


def test_typed_class_sets_new_class_and_relabels_selected(win):
    t = tower(win)
    t.setSelected(True)
    le = win.picker.lineEdit()
    le.setFocus()
    le.selectAll()
    QTest.keyClicks(le, "kni")                               # partial, no ★ prefix
    QTest.keyClick(le, Qt.Key_Return)
    assert win.new_class == KNIGHT and t.cls == KNIGHT
    assert win.ls.progress()["done"] == 0                    # Enter in the class box did not save the frame
    win.canvas.scene().clearSelection()
    drag(win, (300, 300), (340, 350))
    assert win.items[-1].cls == KNIGHT
    assert "knight" in win.new_label.text()


def test_typing_s_and_backspace_in_picker_do_not_skip_or_delete(win):
    n, stem = len(win.items), win.stem
    tower(win).setSelected(True)
    le = win.picker.lineEdit()
    le.setFocus()
    le.selectAll()
    QTest.keyClicks(le, "skel")
    QTest.keyClick(le, Qt.Key_Backspace)
    assert win.stem == stem and win.ls.progress()["skipped"] == 0
    assert len(win.items) == n and le.text() == "ske"


def test_dropdown_choice_updates_selected_and_default(win):
    t = tower(win)
    t.setSelected(True)
    i = win.picker.findData(ZAP)
    win.picker.activated.emit(i)
    assert t.cls == ZAP and win.new_class == ZAP


def test_add_box_at_centre_and_class_persists_across_frames(win):
    win._choose(KNIGHT)
    n = len(win.items)
    win.b_center.click()
    assert len(win.items) == n + 1
    b = win.items[-1].box()
    assert b.cls == KNIGHT and b.valid and win.items[-1].isSelected()
    QTest.keyClick(win.canvas, Qt.Key_A)                     # shortcut
    assert len(win.items) == n + 2
    win.go(win.pos + 1)
    assert win.new_class == KNIGHT and win.picker.currentData() == KNIGHT


def test_use_suggestion_takes_the_picked_class(win):
    k = next(i for i, s in enumerate(win.queue) if win.ls.suggestions(s))
    win.go(k)
    sug = next(it for it in win.items if it.suggestion)
    sug.setSelected(True)
    win._choose(ZAP)
    assert sug.cls == KNIGHT                                 # suggestions are not relabelled in place
    win._use_suggestion()
    assert win.items[-1].cls == ZAP and not win.items[-1].suggestion
