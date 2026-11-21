"""Original board art + render-side animations: classification, effect diffing, read-only rendering, frame time."""
import os
import statistics
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from backend_engine.simulator.card_db import CardDB  # noqa: E402
from frontend_app.qt import art  # noqa: E402
from frontend_app.qt.fx import DURATION, FxTracker  # noqa: E402
from frontend_app.qt.session import AI, HUMAN, GameSession, SessionConfig  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


# ---------------------------------------------------------------- classification
def test_every_playable_card_gets_a_role(db):
    roles = {cid: art.role(c) for cid, c in db.cards.items()}
    assert set(roles.values()) <= set(art.ROLES)
    assert set(roles.values()) == set(art.ROLES)          # every glyph is in use


@pytest.mark.parametrize("card,expected", [
    ("knight", "melee"), ("hog_rider", "melee"), ("musketeer", "ranged"), ("archers", "ranged"),
    ("minions", "air"), ("baby_dragon", "air"), ("cannon", "building"), ("tesla", "building"),
    ("ice_spirit", "kamikaze"), ("fire_spirit", "kamikaze"), ("golden_knight", "champion"),
    ("fireball", "spell"), ("the_log", "spell"),
])
def test_role_examples(db, card, expected):
    assert art.role(db.cards[card]) == expected


def test_card_colour_is_stable_and_varied(app, db):
    assert art.card_color("knight").name() == art.card_color("knight").name()
    assert len({art.card_color(c).name() for c in db.cards}) > len(db.cards) // 2


def test_every_glyph_has_a_shape():
    for kind in (*art.ROLES, "crown"):
        assert not art.glyph(kind).isEmpty(), kind


# ---------------------------------------------------------------- effect tracker (fake match objects)
def U(uid, x, y, side=HUMAN, kind="troop", hp=100.0, cooldown=0, target=None, mech=(), layer="ground"):
    return SimpleNamespace(uid=uid, x=x, y=y, side=side, card_id="knight", kind=kind, layer=layer, radius=0.5,
                           hp=hp, cooldown_ms=cooldown, target_uid=target, mechanics=mech)


def M(t, units=(), projectiles=(), spells=()):
    return SimpleNamespace(t_ms=t, units={u.uid: u for u in units}, projectiles=list(projectiles),
                           spells=list(spells))


class Seq:
    """One mutable fake match whose state is replaced each step (the tracker keys on match identity)."""

    def __init__(self, **kw):
        self.m = M(0, **kw)

    def step(self, t, **kw):
        new = M(t, **kw)
        self.m.__dict__.update(new.__dict__)
        return self.m


def test_land_death_and_collapse():
    tr, s = FxTracker(), Seq(units=[U(1, 3, 3), U(9, 9, 3, kind="tower")])
    tr.observe(s.m, 0.0)
    assert tr.fx == []                                      # first sight: no effects
    tr.observe(s.step(50, units=[U(1, 3, 3), U(2, 5, 5), U(9, 9, 3, kind="tower")]), 0.05)
    assert [f.kind for f in tr.fx] == ["land"] and (tr.fx[0].x, tr.fx[0].y) == (5, 5)
    tr.observe(s.step(100, units=[U(2, 5, 5)]), 0.10)
    assert {f.kind for f in tr.fx} == {"land", "death", "collapse"}


def test_projectile_impact_and_spell_burst():
    pr = SimpleNamespace(uid=7, x=4, y=6, side=AI, splash=1.2)
    sp = SimpleNamespace(uid=8, x=9, y=20, side=HUMAN, radius=2.5, card_id="fireball")
    tr, s = FxTracker(), Seq(projectiles=[pr], spells=[sp])
    tr.observe(s.m, 0.0)
    tr.observe(s.step(50), 0.05)
    kinds = sorted(f.kind for f in tr.fx)
    assert kinds == ["burst", "burst", "impact"]
    assert any(f.kind == "burst" and f.size == 2.5 for f in tr.fx)


def test_melee_hit_and_hurt_flash():
    tr = FxTracker()
    s = Seq(units=[U(1, 3, 3, cooldown=0, target=2), U(2, 3, 4, side=AI, hp=100)])
    tr.observe(s.m, 0.0)
    tr.observe(s.step(50, units=[U(1, 3, 3, cooldown=1200, target=2), U(2, 3, 4, side=AI, hp=80)]), 0.05)
    assert [f.kind for f in tr.fx] == ["hit"]
    assert tr.hurt_level(2, 0.05) == 1.0 and tr.hurt_level(1, 0.05) == 0.0
    # a ranged attacker's hit is shown by its projectile, not by a melee spark
    tr2 = FxTracker()
    s2 = Seq(units=[U(1, 3, 3, target=2, mech=("projectile",)), U(2, 3, 4, side=AI)])
    tr2.observe(s2.m, 0.0)
    tr2.observe(s2.step(50, units=[U(1, 3, 3, cooldown=900, target=2, mech=("projectile",)),
                                   U(2, 3, 4, side=AI, hp=80)]), 0.05)
    assert tr2.fx == []


def test_seek_and_rewind_reset_without_effects():
    tr, s = FxTracker(), Seq(units=[U(1, 3, 3)])
    tr.observe(s.m, 0.0)
    tr.observe(s.step(5000, units=[U(2, 5, 5)]), 0.05)      # forward jump (seek)
    assert tr.fx == []
    tr.observe(s.step(4000, units=[]), 0.10)                 # rewind
    assert tr.fx == []
    tr.observe(M(4050, units=[U(3, 1, 1)]), 0.15)            # another match object (new match / fork)
    assert tr.fx == []


def test_effects_expire_and_motion_is_interpolated():
    tr, s = FxTracker(), Seq(units=[U(1, 3.0, 3.0)])
    tr.observe(s.m, 0.0)
    tr.observe(s.step(50, units=[U(1, 3.0, 4.0), U(2, 8, 8)]), 0.05)
    x, y = tr.pos(1, 3.0, 4.0, 0.05 + tr.step_s / 2)
    assert x == 3.0 and 3.4 < y < 3.6                       # halfway between the two simulation steps
    assert tr.pos(1, 3.0, 4.0, 1.0) == (3.0, 4.0)
    assert tr.animating(0.06)
    tr.prune(0.05 + max(DURATION.values()) + 0.01)
    assert tr.fx == [] and not tr.animating(5.0)             # idle: the 16 ms timer stops repainting


# ---------------------------------------------------------------- board rendering (real match)
def busy_session():
    s = GameSession(cfg=SessionConfig(seed=7))
    plays = [(HUMAN, "knight", 3.5, 8.5), (HUMAN, "musketeer", 4.5, 4.5), (HUMAN, "minions", 13.5, 12.5),
             (HUMAN, "cannon", 9.5, 9.5), (AI, "giant", 3.5, 23.5), (AI, "tesla", 9.5, 21.5),
             (AI, "bats", 14.5, 22.5), (AI, "fire_spirit", 2.5, 18.5), (HUMAN, "golden_knight", 14.5, 10.5)]
    for side, card, x, y in plays:
        if card not in s.db.cards:
            continue
        st = s.match.sides[side]
        if card not in st.hand:
            if card in st.queue:
                st.queue.remove(card)
            st.queue.insert(0, st.hand[0])
            st.hand[0] = card
        st.elixir = 10
        assert s.runner.human_play(card, x, y, side=side).ok, card
        s.runner.deliver_events()
        for _ in range(6):
            s.runner.tick()
    for _ in range(30):
        s.runner.tick()
    return s


@pytest.mark.parametrize("size", [(560, 820), (420, 620)])
def test_board_paints_without_touching_the_engine(app, size):
    from frontend_app.qt.board import BoardWidget
    s = busy_session()
    b = BoardWidget(s)
    b.resize(*size)
    b.set_selected(next(c for c in s.match.sides[HUMAN].hand if s.db.cards[c]["type"] != "spell"))
    b.hover = (14.5, 9.5)
    before = s.match.state_hash()
    clock = [10.0]
    b.clock = lambda: clock[0]
    times = []
    for _ in range(40):
        s.runner.tick()
        h = s.match.state_hash()
        for _ in range(3):
            t0 = time.perf_counter()
            assert not b.grab().isNull()
            times.append(time.perf_counter() - t0)
            clock[0] += 1 / 60
        assert s.match.state_hash() == h                    # painting never changes the simulation
    assert before != s.match.state_hash()
    # 60 fps budget is 16.7 ms per frame; offscreen paints take ~2 ms here, so this only catches big regressions
    assert statistics.median(times) < 0.012


def test_hand_shows_card_faces(app):
    from frontend_app.qt.app import MainWindow
    w = MainWindow(GameSession(cfg=SessionConfig(seed=4)))
    w.show()
    w.start_match()
    st = w.session.match.sides[HUMAN]
    st.elixir = 2.0
    w.refresh()
    for b, c in zip(w.hand.buttons, st.hand):
        card_id, name, cost, kind, afford, *_ = b.face
        assert card_id == c and kind == art.role(w.session.db.cards[c])
        assert afford == (cost <= 2.0)
    assert w.hand.next_card.face[0] == st.queue[0]
    assert not w.hand.grab().isNull()
    w.close()


def test_no_placement_zone_overlay_by_default(app):
    """No tinted area or dashed outline while placing a troop; the hover marker alone shows legal / illegal."""
    from frontend_app.qt.board import BoardWidget
    s = GameSession(cfg=SessionConfig(seed=3))
    b = BoardWidget(s)
    b.resize(450, 800)
    card = next(c for c in s.match.sides[HUMAN].hand if s.db.cards[c]["type"] != "spell")
    plain = b.grab().toImage()
    b.set_selected(card)
    assert b.legal and not b.show_zone
    assert b.grab().toImage() == plain                      # selecting a card (no hover) changes nothing


# ---------------------------------------------------------------- sprite art (CC0 3D renders)
def test_sprite_manifest_and_frames(app):
    from frontend_app.qt.sprites import SpriteBank
    bank = SpriteBank()
    assert bank.ok, "frontend_app/qt/assets/sprites/manifest.json missing: run tools/art/build_sprites.py"
    for card in ("knight", "archers", "skeletons"):
        name = bank.character(card)
        assert name is not None
        counts = bank.counts(name)
        assert {"idle", "walk", "attack", "death"} <= set(counts)
        for side in ("blue", "red"):
            for facing in ("up", "down"):
                pm = bank.frame(name, side, facing, "walk", 0, 20.0, 1.0)
                assert pm is not None and not pm.isNull()
    assert bank.character("hog_rider") is None                 # unmapped cards keep vector tokens
    for kind in ("tower", "castle"):
        pm, (gx, gy) = bank.tower(kind, "red", 20.0, 2.0)
        assert pm is not None and 0 < gx < 1 and 0 < gy < 1
    assert bank.arena(400, 600, 1.0) is not None


def test_missing_sprites_fall_back_to_vector(app, tmp_path):
    from frontend_app.qt.board import BoardWidget
    from frontend_app.qt.sprites import SpriteBank
    b = BoardWidget(GameSession(cfg=SessionConfig(seed=3)))
    b.bank = SpriteBank(tmp_path)                              # empty folder: no manifest
    b.set_art_mode("sprites")
    assert b.art_mode == "vector" and b.cos == 1.0 and b.margin == (0.0, 0.0)
    b.resize(450, 800)
    assert not b.grab().isNull()


@pytest.mark.parametrize("mode", ["vector", "sprites"])
def test_click_mapping_round_trips(app, mode):
    from frontend_app.qt.board import BoardWidget
    b = BoardWidget(GameSession(cfg=SessionConfig(seed=3)))
    b.set_art_mode(mode)
    b.resize(600, 760)
    for x, y in ((0.5, 0.5), (9.0, 16.0), (17.5, 31.5), (3.5, 14.5)):
        c = b.to_px(x, y)
        tx, ty = b.to_tile(c.x(), c.y())
        assert abs(tx - x) < 1e-9 and abs(ty - y) < 1e-9
    if mode == "sprites":
        assert b.cos < 0.9                                     # tilted: the ground is foreshortened


def test_pose_follows_engine_state():
    from frontend_app.qt.sprites import IMPACT_AT, pose
    counts = {"idle": 4, "walk": 8, "attack": 8, "death": 6}
    u = SimpleNamespace(uid=3, side=HUMAN, y=10.0, deploy_ms=0, target_uid=None, hit_speed_ms=1200, cooldown_ms=0)
    assert pose(u, (0.0, 0.1), None, 0, counts)[:2] == ("up", "walk")
    assert pose(u, (0.0, -0.1), None, 0, counts)[:2] == ("down", "walk")
    assert pose(u, (0.0, 0.0), None, 0, counts)[:2] == ("up", "idle")
    u.target_uid = 9
    u.cooldown_ms = 1200                                       # the blow just landed
    assert pose(u, (0.0, 0.0), (10.0, 8.0), 0, counts) == ("down", "attack", int(IMPACT_AT * 8))
    u.deploy_ms = 300
    assert pose(u, (0.0, 0.0), (10.0, 8.0), 0, counts)[1:] == ("idle", 0)


def test_sprite_board_draws_sprites_read_only(app):
    from frontend_app.qt.board import BoardWidget
    s = busy_session()
    b = BoardWidget(s)
    b.set_art_mode("sprites")
    b.resize(600, 760)
    knights = [u for u in s.match.units.values() if u.card_id == "knight"]
    assert knights
    h = s.match.state_hash()
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        assert not b.grab().isNull()
        times.append(time.perf_counter() - t0)
    assert s.match.state_hash() == h
    assert statistics.median(times) < 0.012                   # cached scaled frames: ~2-4 ms offscreen
    from PySide6.QtGui import QPainter, QImage
    img = QImage(600, 760, QImage.Format_ARGB32)
    p = QPainter(img)
    assert b._paint_sprite(p, knights[0], 20.0, (knights[0].x, knights[0].y), b.clock())
    p.end()
