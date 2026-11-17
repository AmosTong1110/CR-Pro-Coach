"""Tests for Clash Royale deck link import and export (clashroyale://copyDeck, RoyaleAPI, text)."""
import os
from unittest.mock import patch
import pytest

from backend_engine.meta.deck_link import (
    ParsedDeck,
    format_copy_deck_link,
    format_royaleapi_url,
    parse_deck_link,
)
from backend_engine.simulator.card_db import CardDB, default_card_db
from frontend_app.qt.session import AI, HUMAN, GameSession, SessionConfig


@pytest.fixture
def db():
    return default_card_db()


def test_parse_official_copy_deck_link(db):
    # Knight (26000000), Hog (26000021), Ice Spirit (26000030), Cannon (27000000),
    # Fireball (28000000), Log (28000011), Skeletons (26000010), Musketeer (26000014)
    link = "clashroyale://copyDeck?deck=26000000;26000021;26000030;27000000;28000000;28000011;26000010;26000014&slots=1;0;0;0;0;0;0;0"
    parsed = parse_deck_link(link, db)

    assert isinstance(parsed, ParsedDeck)
    assert parsed.playable is True
    assert len(parsed.cards) == 8
    assert "knight" in parsed.cards
    assert "hog_rider" in parsed.cards
    assert parsed.evo == ("knight",)
    assert parsed.missing == ()
    assert parsed.avg_elixir == pytest.approx(2.75, 0.01)
    assert "Hog" in parsed.archetype


def test_parse_web_deck_link(db):
    web_link = "https://link.clashroyale.com/deck/en?deck=26000000;26000021;26000030;27000000;28000000;28000011;26000010;26000014&slots=1;0;0;0;0;0;0;0"
    parsed = parse_deck_link(web_link, db)

    assert parsed.playable is True
    assert len(parsed.cards) == 8
    assert parsed.evo == ("knight",)


def test_parse_royaleapi_url(db):
    ra_url = "https://royaleapi.com/decks/stats/hog-rider,musketeer,cannon,fireball,the-log,skeletons,ice-spirit,ice-golem"
    parsed = parse_deck_link(ra_url, db)

    assert parsed.playable is True
    assert len(parsed.cards) == 8
    assert parsed.archetype == "Hog 2.6"
    assert parsed.avg_elixir == pytest.approx(2.625, 0.01)
    assert "hog_rider" in parsed.cards
    assert "the_log" in parsed.cards


def test_parse_plain_text(db):
    text = "Knight, Hog Rider, Ice Spirit; Cannon, Fireball, The Log\nSkeletons, Musketeer"
    parsed = parse_deck_link(text, db)

    assert parsed.playable is True
    assert len(parsed.cards) == 8
    assert "knight" in parsed.cards
    assert "musketeer" in parsed.cards


def test_roundtrip_format_and_parse(db):
    cards = ("knight", "hog_rider", "musketeer", "cannon", "fireball", "the_log", "skeletons", "ice_spirit")
    evo = ("knight",)

    # Official deep link roundtrip
    deep_link = format_copy_deck_link(cards, evo, db=db, web=False)
    assert deep_link.startswith("clashroyale://copyDeck?")
    parsed_deep = parse_deck_link(deep_link, db)
    assert parsed_deep.cards == cards
    assert parsed_deep.evo == evo
    assert parsed_deep.playable is True

    # Web link roundtrip
    web_link = format_copy_deck_link(cards, evo, db=db, web=True)
    assert web_link.startswith("https://link.clashroyale.com/deck/en?")
    parsed_web = parse_deck_link(web_link, db)
    assert parsed_web.cards == cards
    assert parsed_web.evo == evo
    assert parsed_web.playable is True


def test_format_royaleapi_url(db):
    cards = ("hog_rider", "musketeer", "the_log")
    ra_url = format_royaleapi_url(cards, db)
    assert ra_url == "https://royaleapi.com/decks/stats/hog-rider,musketeer,the-log"


def test_missing_unsimulated_cards(db):
    # 26000000 is Knight, 99999999 is unknown/unsimulated
    link = "clashroyale://copyDeck?deck=26000000;99999999&slots=0;0"
    parsed = parse_deck_link(link, db)

    assert parsed.playable is False
    assert "99999999" in parsed.missing
    assert "knight" in parsed.cards
    assert parsed.warning is not None
    assert "not simulated" in parsed.warning.lower() or "could not be resolved" in parsed.warning.lower()


def test_unsimulated_evo_fallback(db):
    # Cannon (27000000) does not have evolution simulated in this engine; Archers is 26000001
    link = "clashroyale://copyDeck?deck=27000000;26000021;26000030;28000000;28000011;26000010;26000014;26000001&slots=1;0;0;0;0;0;0;0"
    parsed = parse_deck_link(link, db)

    assert parsed.playable is True
    # Cannon should not be accepted as evo
    assert parsed.evo == ()
    assert parsed.warning is not None
    assert "Cannon" in parsed.warning


# ---------------------------------------------------------------------------
# Qt UI Integration Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_qt_deck_dialog_copy_and_import(qapp):
    from frontend_app.qt.app import DeckDialog

    session = GameSession(cfg=SessionConfig(seed=42))
    dlg = DeckDialog(session)

    # 1. Test Copy Link
    with patch("frontend_app.qt.app.QMessageBox.information") as mock_info:
        dlg._copy_link()
        clip_text = qapp.clipboard().text()
        assert clip_text.startswith("clashroyale://copyDeck?")
        assert mock_info.called

    # 2. Test Import Link (success)
    test_link = "clashroyale://copyDeck?deck=26000000;26000021;26000030;27000000;28000000;28000011;26000010;26000014&slots=1;0;0;0;0;0;0;0"
    with patch("frontend_app.qt.app.QInputDialog.getText", return_value=(test_link, True)):
        with patch("frontend_app.qt.app.QMessageBox.information"):
            dlg._import_link()

    chosen_cards, chosen_evo = dlg.choice[dlg.side]
    assert len(chosen_cards) == 8
    assert "knight" in chosen_cards
    assert "knight" in chosen_evo

    # 3. Test Import Link with invalid/unplayable deck
    invalid_link = "clashroyale://copyDeck?deck=26000000;99999999"
    with patch("frontend_app.qt.app.QInputDialog.getText", return_value=(invalid_link, True)):
        with patch("frontend_app.qt.app.QMessageBox.warning") as mock_warn:
            dlg._import_link()
            assert mock_warn.called
            assert len(dlg.choice[dlg.side][0]) == 8

    dlg.close()
    qapp.processEvents()


def test_qt_custom_deck_dialog_copy_and_import(qapp):
    from frontend_app.qt.app import CustomDeckDialog

    db = default_card_db()
    init_deck = ("knight", "archers", "zap", "fireball", "cannon", "skeletons", "musketeer", "hog_rider")
    dlg = CustomDeckDialog(db, init_deck, ("knight",), "Test Deck")

    # Copy link
    with patch("frontend_app.qt.app.QMessageBox.information") as mock_info:
        dlg._copy_link()
        assert qapp.clipboard().text().startswith("clashroyale://copyDeck?")
        assert mock_info.called

    # Import link
    new_link = "https://royaleapi.com/decks/stats/hog-rider,musketeer,cannon,fireball,the-log,skeletons,ice-spirit,ice-golem"
    with patch("frontend_app.qt.app.QInputDialog.getText", return_value=(new_link, True)):
        dlg._import_link()

    chosen = dlg.chosen()
    assert len(chosen) == 8
    assert "ice_golem" in chosen
    assert "the_log" in chosen

    dlg.close()
    qapp.processEvents()


def test_qt_decks_page_copy_and_import(qapp, tmp_path):
    from PySide6.QtWidgets import QMessageBox
    from frontend_app.qt.shell import CRbotWindow

    session = GameSession(cfg=SessionConfig(seed=42))
    win = CRbotWindow(session, data_dir=tmp_path)
    win.show()

    decks_page = win.decks

    # Select first row and copy link
    decks_page.table.selectRow(0)
    with patch("frontend_app.qt.shell.QMessageBox.information") as mock_info:
        decks_page._copy_link()
        assert qapp.clipboard().text().startswith("clashroyale://copyDeck?")
        assert mock_info.called

    # Import link
    imported_link = "clashroyale://copyDeck?deck=26000000;26000021;26000030;27000000;28000000;28000011;26000010;26000014&slots=1;0;0;0;0;0;0;0"
    with patch("frontend_app.qt.shell.QInputDialog.getText", return_value=(imported_link, True)):
        with patch.object(win, "fight_with") as mock_fight:
            created_boxes = []
            orig_exec = QMessageBox.exec
            def fake_exec(self):
                created_boxes.append(self)
                return 0

            with patch.object(QMessageBox, "exec", fake_exec):
                with patch.object(QMessageBox, "clickedButton", side_effect=lambda: created_boxes[-1].buttons()[0]):
                    decks_page._import_link()
                    assert mock_fight.called
                    args, kwargs = mock_fight.call_args
                    assert "knight" in args[0]
                    assert kwargs.get("as_ai") is False
                    assert kwargs.get("evo") == ("knight",)

    win.close()
    qapp.processEvents()
