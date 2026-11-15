"""CRbot look: one palette and one stylesheet for the shell, the Fight page and dialogs.

Widgets opt into a style with a QSS class (`_cls(widget, "primary")` sets the `class` property), so layout code
never carries inline colours.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QWidget

ICONS = Path(__file__).resolve().parent / "icons"
_CHEVRON = (ICONS / "chevron_down.svg").as_posix()

# -- palette (own identity: charcoal + teal / violet) ------------------------------------------------------
C = {
    "bg": "#15171c", "panel": "#1d2027", "panel2": "#252932", "line": "#2f3440",
    "text": "#e6e8ec", "muted": "#8a909c", "teal": "#2ec4b6", "violet": "#8b5cf6",
    "win": "#3ddc97", "loss": "#ff6b6b", "draw": "#c9ced8",
    "gold": "#ffd400", "elixir": "#d43cf0", "you": "#5b9cff", "ai": "#ff7a70", "warn": "#ffb347",
}

STYLE = f"""
QWidget#ShellRoot, QStackedWidget#Pages, QWidget.page, QWidget#FightRoot {{ background:{C['bg']}; color:{C['text']}; }}
QWidget#Sidebar {{ background:{C['panel']}; border-right:1px solid {C['line']}; }}
QLabel#Brand {{ color:{C['text']}; font-size:22px; font-weight:800; padding:18px 16px 2px 18px; }}
QLabel#BrandSub {{ color:{C['teal']}; font-size:11px; padding:0 16px 14px 18px; letter-spacing:1px; }}
QLabel {{ color:{C['text']}; }}
QLabel.h1 {{ font-size:22px; font-weight:700; color:{C['text']}; }}
QLabel.h2 {{ font-size:15px; font-weight:700; color:{C['text']}; }}
QLabel.sub {{ font-size:12px; color:{C['muted']}; }}
QLabel.muted {{ font-size:11px; color:{C['muted']}; }}
QLabel.section {{ font-size:11px; font-weight:700; color:{C['muted']}; letter-spacing:1px; }}
QLabel.clock {{ font-size:32px; font-weight:800; color:{C['text']}; }}
QLabel.phase {{ font-size:13px; font-weight:700; color:{C['elixir']}; }}
QLabel.status {{ font-size:12px; color:{C['warn']}; }}
QLabel.stat {{ font-size:22px; font-weight:800; color:{C['text']}; }}
QLabel.chip {{ background:{C['panel2']}; border:1px solid {C['line']}; border-radius:6px; padding:3px 6px;
               font-size:11px; color:{C['text']}; }}
QLabel.chip_evo {{ background:{C['panel2']}; border:1px solid {C['violet']}; border-radius:6px; padding:3px 6px;
                   font-size:11px; color:{C['text']}; }}
QFrame.card {{ background:{C['panel']}; border:1px solid {C['line']}; border-radius:10px; }}
QFrame.card QLabel {{ background:transparent; border:none; }}
QFrame.card QLabel.chip, QFrame.card QLabel.chip_evo {{ background:{C['panel2']}; border:1px solid {C['line']}; }}
QFrame.card QLabel.chip_evo {{ border-color:{C['violet']}; }}

QPushButton {{ background:{C['panel2']}; color:{C['text']}; border:1px solid {C['line']}; border-radius:7px;
               padding:7px 12px; min-height:18px; }}
QPushButton:hover {{ border-color:{C['violet']}; }}
QPushButton:disabled {{ color:{C['muted']}; border-color:{C['panel2']}; }}
QPushButton.nav {{ text-align:left; padding:10px 16px; margin:2px 10px; border:none; border-radius:8px;
                   color:{C['muted']}; font-size:14px; background:transparent; }}
QPushButton.nav:hover {{ background:{C['panel2']}; color:{C['text']}; }}
QPushButton.nav:checked {{ background:{C['panel2']}; color:{C['text']}; border-left:3px solid {C['teal']}; }}
QPushButton.primary {{ background:{C['teal']}; color:#0b1f1d; border:none; border-radius:7px;
                       padding:8px 14px; font-weight:700; }}
QPushButton.primary:hover {{ background:#3fd6c8; }}
QPushButton.primary:disabled {{ background:{C['line']}; color:{C['muted']}; }}
QPushButton.big {{ background:{C['teal']}; color:#0b1f1d; border:none; border-radius:9px;
                   padding:12px 14px; font-size:16px; font-weight:800; min-height:22px; }}
QPushButton.big:hover {{ background:#3fd6c8; }}
QPushButton.big:disabled {{ background:{C['line']}; color:{C['muted']}; }}
QPushButton.stop {{ background:{C['loss']}; color:#2a0b0b; border:none; border-radius:9px; padding:11px 14px;
                    font-size:15px; font-weight:800; min-height:20px; }}
QPushButton.stop:hover {{ background:#ff8585; }}
QPushButton.ghost {{ background:transparent; color:{C['text']}; border:1px solid {C['line']}; border-radius:7px;
                     padding:7px 14px; }}
QPushButton.ghost:hover {{ border-color:{C['violet']}; }}
QPushButton.ghost:disabled {{ color:{C['muted']}; border-color:{C['panel2']}; }}
QPushButton.danger {{ background:transparent; color:{C['loss']}; border:1px solid {C['line']}; border-radius:7px;
                      padding:7px 14px; }}
QPushButton.danger:hover {{ border-color:{C['loss']}; }}
QPushButton.seg {{ background:{C['panel2']}; color:{C['muted']}; border:1px solid {C['line']}; border-radius:7px;
                   padding:6px 10px; }}
QPushButton.seg:checked {{ background:{C['teal']}; color:#0b1f1d; border-color:{C['teal']}; font-weight:700; }}

QComboBox {{ background:{C['panel2']}; color:{C['text']}; border:1px solid {C['line']}; border-radius:6px;
             padding:5px 8px; min-height:20px; }}
QComboBox:hover {{ border-color:{C['violet']}; }}
QComboBox::drop-down {{ border:none; width:22px; }}
QComboBox::down-arrow {{ image:url({_CHEVRON}); width:12px; height:8px; margin-right:8px; }}
QComboBox QAbstractItemView {{ background:{C['panel2']}; color:{C['text']}; border:1px solid {C['line']};
                               selection-background-color:{C['teal']}; selection-color:#0b1f1d; }}
QLineEdit {{ background:{C['panel2']}; color:{C['text']}; border:1px solid {C['line']}; border-radius:6px;
             padding:5px 8px; }}
QCheckBox, QRadioButton {{ color:{C['text']}; spacing:8px; background:transparent; }}
QCheckBox::indicator {{ width:14px; height:14px; border-radius:4px; border:2px solid {C['muted']}; background:{C['panel2']}; }}
QCheckBox::indicator:checked {{ background:{C['teal']}; border-color:{C['teal']}; }}
QRadioButton::indicator {{ width:12px; height:12px; border-radius:8px; border:2px solid {C['muted']}; background:{C['panel2']}; }}
QRadioButton::indicator:checked {{ background:{C['teal']}; border-color:{C['teal']}; }}
QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
QScrollBar::handle:vertical {{ background:{C['line']}; border-radius:3px; min-height:28px; }}
QScrollBar::handle:vertical:hover {{ background:{C['muted']}; }}
QScrollBar:horizontal {{ background:transparent; height:10px; margin:2px; }}
QScrollBar::handle:horizontal {{ background:{C['line']}; border-radius:3px; min-width:28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}
QGroupBox {{ color:{C['text']}; border:1px solid {C['line']}; border-radius:8px; margin-top:14px; padding-top:8px; }}
QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 4px; color:{C['muted']}; }}
QListWidget {{ background:{C['panel']}; color:{C['text']}; border:1px solid {C['line']}; border-radius:8px;
               outline:none; }}
QListWidget::item {{ padding:4px 6px; border-radius:6px; }}
QListWidget::item:selected {{ background:#2b3a4a; color:{C['text']}; }}
QListWidget::item:hover {{ background:{C['panel2']}; }}
QSlider::groove:horizontal {{ height:6px; background:{C['line']}; border-radius:3px; }}
QSlider::sub-page:horizontal {{ background:{C['teal']}; border-radius:3px; }}
QSlider::handle:horizontal {{ background:{C['text']}; width:16px; height:16px; margin:-5px 0; border-radius:8px; }}
QSlider::handle:horizontal:disabled {{ background:{C['muted']}; }}
QProgressBar {{ background:{C['panel2']}; border:none; border-radius:4px; max-height:8px; color:transparent; }}
QProgressBar::chunk {{ background:{C['teal']}; border-radius:4px; }}
QScrollArea {{ background:transparent; border:none; }}
QScrollArea > QWidget > QWidget {{ background:transparent; }}
QToolTip {{ background:{C['panel2']}; color:{C['text']}; border:1px solid {C['line']}; }}
QDialog {{ background:{C['bg']}; color:{C['text']}; }}
QMessageBox, QInputDialog {{ background:{C['bg']}; }}
"""


def dark_palette() -> QPalette:
    """CRbot palette for all standard widgets (Fight page, dialogs, lists), so nothing is dark-on-dark."""
    p = QPalette()
    roles = {
        QPalette.Window: C["bg"], QPalette.WindowText: C["text"], QPalette.Base: C["panel"],
        QPalette.AlternateBase: C["panel2"], QPalette.Text: C["text"], QPalette.Button: C["panel2"],
        QPalette.ButtonText: C["text"], QPalette.ToolTipBase: C["panel2"], QPalette.ToolTipText: C["text"],
        QPalette.Highlight: C["teal"], QPalette.HighlightedText: "#0b1f1d", QPalette.PlaceholderText: C["muted"],
        QPalette.Link: C["teal"], QPalette.BrightText: C["loss"],
    }
    for role, col in roles.items():
        p.setColor(role, QColor(col))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(C["muted"]))
    return p


def cls(w: QWidget, name: str) -> QWidget:
    """Attach a stylesheet class (QSS `.name`) to a widget and return it."""
    w.setProperty("class", name)
    return w
