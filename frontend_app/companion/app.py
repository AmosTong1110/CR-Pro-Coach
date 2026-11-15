"""
CR Bot Companion (Mode B) — always-on-top, read-only overlay for real games.

    python -m frontend_app.companion.app            # opens in Demo mode (simulated match)

Put it next to your phone mirror (QuickTime "New Movie Recording" with the iPhone as
camera, or scrcpy for Android), click Calibrate, load a detector model, press Start when
the battle begins. It only reads pixels; it never touches the game.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget, QMessageBox,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from backend_engine.perception.live.calibration import ArenaCalibration
from backend_engine.perception.live.capture import FolderSource, QtScreenSource
from backend_engine.perception.live.companion import OPP, LiveCompanion, SimDemoFeed
from backend_engine.perception.live.detector import CLASS_MAP_PATH, YoloOnnxDetector
from backend_engine.simulator.card_db import CardDB

FPS = 10
STYLE = """
QWidget { background:#15181d; color:#e8e8e8; font-size:12px; }
QPushButton { background:#2a3140; border:1px solid #3c4658; border-radius:5px; padding:4px 8px; }
QPushButton:hover { background:#34405a; }
QComboBox, QListWidget { background:#1d222b; border:1px solid #333c4c; }
"""


def pretty(card: str, names) -> str:
    return names.get(card, card.replace("_", " ").title())


class ElixirMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value: Optional[float] = None
        self.setMinimumHeight(18)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, v: Optional[float]) -> None:
        self.value = v
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        seg = self.width() / 10
        for i in range(10):
            r = QRectF(i * seg + 1, 2, seg - 2, self.height() - 4)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#3a1f45"))
            p.drawRoundedRect(r, 3, 3)
            if self.value is not None:
                fill = max(0.0, min(1.0, self.value - i))
                if fill > 0:
                    p.setBrush(QColor("#d43cf0") if fill >= 1 else QColor("#8e3aa6"))
                    p.drawRoundedRect(QRectF(r.x(), r.y(), r.width() * fill, r.height()), 3, 3)
        p.end()


class CalibrationDialog(QDialog):
    """Screenshot of the whole screen; click 4 points: mirror window top-left & bottom-right,
    then the arena's top-left corner (opponent's back-left) & bottom-right corner (your back-right)."""
    STEPS = ["Click the TOP-LEFT corner of the phone mirror window",
             "Click the BOTTOM-RIGHT corner of the phone mirror window",
             "Click the arena's TOP-LEFT corner (opponent side, left edge of the field)",
             "Click the arena's BOTTOM-RIGHT corner (your side, right edge of the field)"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate arena")
        screen = QGuiApplication.primaryScreen()
        self.shot = screen.grabWindow(0)
        self.dpr = self.shot.devicePixelRatio() or 1.0
        geo = screen.availableGeometry()
        self.scale = min(geo.width() * 0.8 / self.shot.width() * self.dpr,
                         geo.height() * 0.8 / self.shot.height() * self.dpr, 1.0)
        self.points = []
        self.result_cal: Optional[ArenaCalibration] = None
        self.info = QLabel(self.STEPS[0])
        self.info.setStyleSheet("font-size:14px; font-weight:bold; color:#ffd400;")
        self.canvas = QLabel()
        logical = QPixmap(self.shot)
        logical.setDevicePixelRatio(1.0)
        w = int(self.shot.width() / self.dpr * self.scale)
        self.canvas.setPixmap(logical.scaledToWidth(w, Qt.SmoothTransformation))
        self.canvas.mousePressEvent = self._click
        lay = QVBoxLayout(self)
        lay.addWidget(self.info)
        lay.addWidget(self.canvas)
        if self.shot.isNull() or self._looks_black():
            self.info.setText("Screen capture is black: allow Screen Recording for your terminal / Python in "
                              "System Settings → Privacy & Security, then restart the companion.")

    def _looks_black(self) -> bool:
        img = self.shot.toImage()
        return all(QColor(img.pixel(int(img.width() * fx), int(img.height() * fy))).value() < 5
                   for fx in (0.2, 0.5, 0.8) for fy in (0.2, 0.5, 0.8))

    def _click(self, e) -> None:
        # canvas pixels -> logical screen points
        x, y = e.position().x() / self.scale, e.position().y() / self.scale
        self.points.append((x, y))
        if len(self.points) < 4:
            self.info.setText(self.STEPS[len(self.points)])
            return
        (wx1, wy1), (wx2, wy2), (ax1, ay1), (ax2, ay2) = self.points
        region = (int(wx1), int(wy1), int(wx2 - wx1), int(wy2 - wy1))
        try:
            self.result_cal = ArenaCalibration.from_corners(
                region[2], region[3], (ax1 - wx1, ay1 - wy1), (ax2 - wx1, ay2 - wy1), screen_region=region)
        except ValueError as err:
            QMessageBox.warning(self, "Calibration", str(err))
            self.points = []
            self.info.setText(self.STEPS[0])
            return
        self.accept()


class CompanionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CR Bot Companion")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(STYLE)
        self.db = CardDB.load()
        self.names = {cid: c["name"] for cid, c in self.db.cards.items()}
        self.comp = LiveCompanion.create(db=self.db, calibration=ArenaCalibration.load())
        self.source = None
        self.demo: Optional[SimDemoFeed] = None
        self.mode = "demo"

        self.clock = QLabel("--:--")
        self.clock.setStyleSheet("font-size:26px; font-weight:bold;")
        self.phase = QLabel()
        self.phase.setStyleSheet("color:#d43cf0; font-weight:bold;")
        top = QHBoxLayout()
        top.addWidget(self.clock)
        top.addStretch(1)
        top.addWidget(self.phase)

        self.opp_value = QLabel("Opponent  –")
        self.opp_value.setStyleSheet("font-size:22px; font-weight:bold; color:#e7a6ff;")
        self.adv = QLabel()
        self.adv.setStyleSheet("font-size:14px; font-weight:bold;")
        row = QHBoxLayout()
        row.addWidget(self.opp_value)
        row.addStretch(1)
        row.addWidget(self.adv)
        self.meter = ElixirMeter()
        self.cycle = QLabel()
        self.cycle.setWordWrap(True)
        self.likely = QLabel()
        self.likely.setWordWrap(True)
        self.likely.setStyleSheet("color:#9ad;")
        self.plays = QListWidget()
        self.plays.setMaximumHeight(110)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#ffb347; font-size:11px;")

        self.start_btn = QPushButton("Start (S)")
        self.start_btn.clicked.connect(self.toggle_match)
        minus, plus = QPushButton("−1s"), QPushButton("+1s")
        minus.clicked.connect(lambda: self.comp.clock.nudge(-1000))
        plus.clicked.connect(lambda: self.comp.clock.nudge(1000))
        self.source_box = QComboBox()
        self.source_box.addItem("Demo (simulated match)", "demo")
        self.source_box.addItem("Screen (phone mirror)", "screen")
        self.source_box.addItem("Recorded frames folder…", "folder")
        self.source_box.currentIndexChanged.connect(self.change_source)
        cal_btn = QPushButton("Calibrate…")
        cal_btn.clicked.connect(self.calibrate)
        model_btn = QPushButton("Load model…")
        model_btn.clicked.connect(self.load_model)
        save_btn = QPushButton("Save log")
        save_btn.clicked.connect(self.save_log)
        self.manual_box = QComboBox()
        for cid in sorted(self.comp.costs):
            self.manual_box.addItem(f"{pretty(cid, self.names)} ({self.comp.costs[cid]})", cid)
        manual_btn = QPushButton("Opp played")
        manual_btn.clicked.connect(lambda: self.comp.manual_play(self.manual_box.currentData(), OPP))

        controls = QHBoxLayout()
        for w in (self.start_btn, minus, plus):
            controls.addWidget(w)
        controls2 = QHBoxLayout()
        controls2.addWidget(self.source_box, 1)
        controls2.addWidget(cal_btn)
        controls2.addWidget(model_btn)
        controls3 = QHBoxLayout()
        controls3.addWidget(self.manual_box, 1)
        controls3.addWidget(manual_btn)
        controls3.addWidget(save_btn)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(row)
        lay.addWidget(self.meter)
        lay.addWidget(self.cycle)
        lay.addWidget(self.likely)
        lay.addWidget(QLabel("Opponent plays"))
        lay.addWidget(self.plays)
        lay.addLayout(controls)
        lay.addLayout(controls2)
        lay.addLayout(controls3)
        lay.addWidget(self.status)
        note = QLabel("Read-only: reads pixels only, never sends input to the game.")
        note.setStyleSheet("color:#666; font-size:10px;")
        lay.addWidget(note)
        self.resize(360, 520)

        QShortcut(QKeySequence("S"), self, activated=self.toggle_match)
        self.timer = QTimer(self)
        self.timer.setInterval(1000 // FPS)
        self.timer.timeout.connect(self.on_frame)
        self.timer.start()
        self._try_default_model()
        self.change_source()

    # -- setup -------------------------------------------------------------
    def _try_default_model(self) -> None:
        onnx = sorted(Path(__file__).resolve().parents[2].joinpath("pretrained_weights").glob("**/*.onnx"))
        if onnx and CLASS_MAP_PATH.exists():
            try:
                self.comp.detector = YoloOnnxDetector(onnx[0], CLASS_MAP_PATH)
                self.comp.reset()   # rebuild the play tracker with the model's spawn / unit-count hints
                self.status.setText(f"Model: {onnx[0].parent.name}/{onnx[0].name}")
            except Exception as err:  # missing onnxruntime / bad model: stay usable
                self.status.setText(f"Model not loaded: {err}")

    def change_source(self) -> None:
        self.mode = self.source_box.currentData()
        self.comp.stop_match()
        self.start_btn.setText("Start (S)")
        self.demo, self.source = None, None
        if self.mode == "demo":
            self.demo = SimDemoFeed(seed=int(time.time()) % 10_000, db=self.db)
        elif self.mode == "screen":
            cal = self.comp.calibration
            if cal is None or not cal.screen_region:
                self.status.setText("Calibrate first: it tells the companion where the mirror window is.")
            else:
                self.source = QtScreenSource(cal.screen_region)
        elif self.mode == "folder":
            folder = QFileDialog.getExistingDirectory(self, "Folder with recorded frames (PNG/JPG)")
            if folder:
                self.source = FolderSource(folder, fps=FPS)

    def calibrate(self) -> None:
        self.hide()  # keep the overlay out of its own screenshot
        QApplication.processEvents()
        time.sleep(0.25)
        dlg = CalibrationDialog()
        self.show()
        if dlg.exec() and dlg.result_cal:
            dlg.result_cal.save()
            self.comp.calibration = dlg.result_cal
            self.status.setText(f"Calibrated: mirror window {dlg.result_cal.screen_region}")
            if self.mode == "screen":
                self.source = QtScreenSource(dlg.result_cal.screen_region)

    def load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "YOLOv8 ONNX model", "", "ONNX (*.onnx)")
        if not path:
            return
        cmap, _ = QFileDialog.getOpenFileName(self, "Class map JSON", str(CLASS_MAP_PATH.parent), "JSON (*.json)")
        if not cmap:
            return
        try:
            self.comp.detector = YoloOnnxDetector(path, cmap)
            self.status.setText(f"Model loaded: {Path(path).name}")
        except Exception as err:
            QMessageBox.warning(self, "Model", f"Could not load the model:\n{err}\n\nTry: pip install onnxruntime")

    def toggle_match(self) -> None:
        if self.comp.clock.running:
            self.comp.stop_match()
            self.start_btn.setText("Start (S)")
        else:
            wall = 0.0 if self.mode == "demo" else None
            if self.mode == "demo":
                self.demo = SimDemoFeed(seed=int(time.time()) % 10_000, db=self.db)
            self.comp.start_match(wall)
            self.start_btn.setText("Stop (S)")

    def save_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save play log", "match_log.json", "JSON (*.json)")
        if path:
            self.comp.save_log(path)

    # -- loop --------------------------------------------------------------
    def on_frame(self) -> None:
        wall = None
        if self.mode == "demo":
            if self.demo is not None and self.comp.clock.running:
                r = self.demo.next()
                if r is not None:
                    dets, wall = r
                    self.comp.process_tiles(dets, wall)
        elif self.source is not None:
            r = self.source.grab()
            if r is not None:
                frame, wall = r
                self.comp.process_frame(frame, wall)
        self.render(self.comp.state(wall))

    def render(self, st) -> None:
        self.clock.setText(st.clock_text)
        self.phase.setText(st.phase_text)
        if st.opp is None:
            self.opp_value.setText("Opponent  –")
            self.meter.set_value(None)
            self.adv.setText("")
            self.cycle.setText("")
            self.likely.setText("")
        else:
            e = st.opp
            self.opp_value.setText(f"Opponent  {e.point:.1f}")
            self.meter.set_value(e.point)
            adv = st.advantage or 0.0
            if abs(adv) < 0.5:
                self.adv.setText("even")
                self.adv.setStyleSheet("font-size:14px; font-weight:bold; color:#bbb;")
            else:
                self.adv.setText(f"you {adv:+.1f}")
                self.adv.setStyleSheet(f"font-size:14px; font-weight:bold; color:{'#4cd964' if adv > 0 else '#ff6b5e'};")
            hand = [pretty(c, self.names) for c in e.known_in_hand] + ["?"] * e.unknown_hand_slots
            back = [pretty(c, self.names) if c else "?" for c in e.queue]
            self.cycle.setText(f"In hand: {' · '.join(hand)}\nComes back next: ▶ {' → '.join(back)}")
            if st.guess and st.guess.card_probs:
                self.likely.setText("Likely also: " + " · ".join(
                    f"{pretty(c, self.names)} {p:.0%}" for c, p in st.guess.top(4)))
        shown = [f"{e.t_ms / 1000:5.1f}s  {pretty(e.card_id, self.names)} ({e.elixir_cost})"
                 for e in st.recent_plays if e.side == OPP]
        if [self.plays.item(i).text() for i in range(self.plays.count())] != shown:
            self.plays.clear()
            self.plays.addItems(shown)
            self.plays.scrollToBottom()
        src = {"demo": "demo", "screen": "screen", "folder": "frames"}[self.mode]
        bits = [f"source: {src}", f"{st.fps:.0f} fps", f"{st.last_detections} detections"]
        warn = [w for w in st.warnings if not (self.mode == "demo" and w in
                                              ("No detector model loaded", "Arena not calibrated"))]
        self.status.setText(" · ".join(bits) + ("\n" + "\n".join(warn) if warn else ""))


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = CompanionWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
