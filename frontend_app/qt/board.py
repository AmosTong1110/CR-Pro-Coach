"""
竞技场棋盘渲染画布控件 (Arena Board Canvas Widget)
使用 PySide6 QPainter 绘制 18x32 战斗棋盘 (原创矢量美术，见 art.py)，
并负责处理鼠标悬停预览、下牌落点点击以及复盘最优解虚影 (Ghost Tiles) 渲染。

渲染只读引擎状态：动画 (投射物、命中闪光、法术爆开、死亡淡出、落地尘环) 由 fx.FxTracker 对比
相邻两帧状态得出，单位位置在两个 50 ms 模拟步之间平滑插值；16 ms 重绘定时器只在有动画时触发。
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath, QPen,
                           QRadialGradient)
from PySide6.QtWidgets import QWidget

from backend_engine.simulator import geometry as G
from frontend_app.qt import art
from frontend_app.qt.fx import FxTracker
from frontend_app.qt.session import AI, HUMAN, GameSession

# 阵营色彩：人类玩家为蓝，AI 为红 (与 art.BLUE / art.RED 一致)
SIDE_COLOR = {HUMAN: QColor(art.BLUE), AI: QColor(art.RED)}
BG = QColor("#15171c")
ANIM_MS = 16                                   # ~60 fps repaint while something animates

# 区域持续法术 (AreaEffect.kind) 的配色
EFFECT_COLOR = {"pull_spell": QColor(190, 215, 235), "dot_spell": QColor(170, 220, 60),
                "root_spell": QColor(90, 200, 90)}


def short_name(name: str) -> str:
    """提取卡牌名称缩写 (如 Hog Rider -> HR)，供无法显示图标的地方使用"""
    words = name.replace("The ", "").split()
    return "".join(w[0] for w in words[:2]) if len(words) > 1 else name[:2]


class BoardWidget(QWidget):
    """18x32 竞技场主画布控件"""
    tileClicked = Signal(float, float) # 玩家在棋盘上点击了某个网格 (x, y)
    cancelled = Signal()               # 玩家右键点击取消下牌

    def __init__(self, session: GameSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.selected_card: Optional[str] = None
        self.legal: Set[Tuple[float, float]] = set()
        self.hover: Optional[Tuple[float, float]] = None
        self.overlay_text: Optional[str] = None   # 倒计时遮罩大字 (如快退接管后的 3-2-1)
        # 复盘虚影：(card_id, x, y, "actual"实际打法 | "best"推荐最优解)
        self.ghosts: List[Tuple[str, float, float, str]] = []
        self.banner: Optional[str] = None         # 棋盘顶部的战术横幅提示文本
        self.setMouseTracking(True)
        self.setMinimumSize(int(G.W * 12), int(G.H * 12))
        # -- 渲染缓存与动画 (只读引擎状态) --
        self.clock = time.monotonic                # 可替换为假时钟 (测试)
        self.tracker = FxTracker()
        self.tokens = art.TokenCache()
        self._arena_key = None
        self._arena_pm = None
        self._zone_key = None
        self._zone_path = QPainterPath()
        self._roles: Dict[str, str] = {}
        # 可放置区域的半透明底色 + 虚线轮廓：用户不要 (2026-11-21)，默认关闭；悬停准星仍显示绿 / 红
        self.show_zone = False
        self._anim = QTimer(self)
        self._anim.setInterval(ANIM_MS)
        self._anim.timeout.connect(self._on_anim)
        self._anim.start()

    # -- 几何空间换算 (Geometry & Coordinates) -------------------------------
    def _geom(self) -> Tuple[float, float, float]:
        """计算保持 18:32 纵横比的单格像素尺寸 ts 及居中偏移量 (ox, oy)"""
        ts = min(self.width() / G.W, self.height() / G.H)
        return ts, (self.width() - G.W * ts) / 2, (self.height() - G.H * ts) / 2

    def to_px(self, x: float, y: float) -> QPointF:
        """竞技场网格坐标 -> 窗口屏幕像素坐标 (y 轴竖直翻转：y=0 在下方)"""
        ts, ox, oy = self._geom()
        return QPointF(ox + x * ts, oy + (G.H - y) * ts)

    def to_tile(self, px: float, py: float) -> Tuple[float, float]:
        """窗口屏幕像素坐标 -> 竞技场网格坐标"""
        ts, ox, oy = self._geom()
        return (px - ox) / ts, G.H - (py - oy) / ts

    def tile_rect(self, x0: float, y0: float, w: float, h: float) -> QRectF:
        """根据网格左下角 (x0, y0) 及宽高转换为屏幕矩形 QRectF"""
        ts, _, _ = self._geom()
        tl = self.to_px(x0, y0 + h)
        return QRectF(tl.x(), tl.y(), w * ts, h * ts)

    # -- 鼠标交互事件 --------------------------------------------------------
    def set_selected(self, card: Optional[str]) -> None:
        """设置当前激活选中的手牌，并高亮所有合法可放置的网格"""
        self.selected_card = card
        self.legal = set(self.session.legal_tiles(card)) if card else set()
        self.update()

    def refresh_legal(self) -> None:
        """刷新合法网格高亮"""
        if self.selected_card:
            self.legal = set(self.session.legal_tiles(self.selected_card))

    def mouseMoveEvent(self, e) -> None:
        """鼠标移动：更新悬停网格坐标与法术落点范围环"""
        x, y = self.to_tile(e.position().x(), e.position().y())
        self.hover = G.snap(x, y) if G.in_bounds(x, y) else None
        if self.selected_card:
            self.update()

    def leaveEvent(self, e) -> None:
        """鼠标离开画布区域"""
        self.hover = None
        self.update()

    def mousePressEvent(self, e) -> None:
        """鼠标点击：右键取消选中，左键发射下牌信号"""
        if e.button() == Qt.RightButton:
            self.cancelled.emit()
            return
        x, y = self.to_tile(e.position().x(), e.position().y())
        if G.in_bounds(x, y):
            self.tileClicked.emit(x, y)

    # -- 渲染管线 (Painting Pipeline) ----------------------------------------
    def _on_anim(self) -> None:
        """16 ms 定时器：只在可见且有动画 (插值、特效、受击闪白) 时重绘，静止时不耗 CPU"""
        if self.isVisible() and self.tracker.animating(self.clock()):
            self.update()

    def role_of(self, card_id: str) -> str:
        r = self._roles.get(card_id)
        if r is None:
            r = self._roles[card_id] = art.role(self.session.db.entry(card_id))
        return r

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        now = self.clock()
        m = self.session.match
        self.tracker.observe(m, now)
        self.tracker.prune(now)
        ts, _, _ = self._geom()
        # 1. 草地、河道与桥梁 (按尺寸缓存的背景图)
        self._paint_arena(p)
        # 2. 选牌时的可放置区域
        self._paint_zone(p)
        # 3. 被摧毁塔的废墟、持续区域法术、落地尘环
        self._paint_rubble(p, ts)
        self._paint_effects(p, ts)
        self._paint_fx(p, ts, now, ("land",))
        # 4. 场上实体：阴影 -> 地面 (按屏幕纵向排序) -> 死亡淡出 -> 空中
        units = list(m.units.values())
        pos = {u.uid: self.tracker.pos(u.uid, u.x, u.y, now) for u in units}
        self._paint_shadows(p, units, pos, ts)
        ground = sorted((u for u in units if u.layer != "air"), key=lambda u: (-pos[u.uid][1], u.uid))
        air = sorted((u for u in units if u.layer == "air"), key=lambda u: (-pos[u.uid][1], u.uid))
        for u in ground:
            if u.kind == "tower":
                self._paint_tower(p, u, ts)
            else:
                self._paint_unit(p, u, ts, pos[u.uid], now)
        self._paint_fx(p, ts, now, ("death",))
        for u in air:
            self._paint_unit(p, u, ts, pos[u.uid], now)
        # 5. 投射物、来袭法术、滚木；命中 / 爆开特效
        self._paint_spells(p, ts, now)
        self._paint_fx(p, ts, now, ("hit", "impact", "burst", "collapse"))
        # 6. 单位血条 (置顶，避免被遮挡)
        self._paint_unit_bars(p, units, pos, ts)
        # 7. 放置准星
        self._paint_placement(p, ts)
        # 8. 复盘虚影、结算、倒计时、横幅
        self._paint_ghosts(p, ts)
        if not self.ghosts:
            self._paint_result(p)
        self._paint_overlay(p)
        self._paint_banner(p)
        p.end()

    def _paint_ghosts(self, p: QPainter, ts: float) -> None:
        """绘制复盘对比虚影 (Ghost Indicators)
        
        红虚影 (actual): 玩家实际落子位置与范围。
        绿虚影 (best): AI 引擎建议的最佳落子位置与范围。
        两者标签纵向错开 (-1.7*ts 与 +0.8*ts)，确保在同一格子上下牌时文字绝不重叠。
        """
        styles = {"actual": (QColor(235, 70, 70, 120), QColor("#ff8a80"), "You"),
                  "best": (QColor(60, 220, 100, 120), QColor("#b9f6ca"), "Best")}
        for card_id, x, y, kind in self.ghosts:
            fill, edge, tag = styles[kind]
            card = self.session.db.cards[card_id]
            c = self.to_px(x, y)
            p.setBrush(fill)
            p.setPen(QPen(edge, 2, Qt.DashLine))
            spell = card.get("spell", {})
            if "radius_tiles" in spell:
                # 圆形范围法术 (如火球、万箭齐发)
                r = spell["radius_tiles"] * ts
                p.drawEllipse(c, r, r)
            elif "rolling_range_tiles" in spell:
                # 矩形滚木法术 (滚木滚过区域)
                w = spell["width_tiles"]
                p.drawRect(self.tile_rect(x - w / 2, y, w, spell["rolling_range_tiles"]))
            else:
                # 普通单位碰撞体积虚影
                r = max(card.get("collision_radius_tiles", 0.5), 0.5) * ts * 1.2
                p.drawEllipse(c, r, r)
            f = QFont()
            f.setPixelSize(max(10, int(ts * 0.6)))
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor("white"))
            # 避让逻辑：actual 往上浮动，best 往下浮动，防止文字相互遮挡
            dy = -1.7 * ts if kind == "actual" else 0.8 * ts
            label = QRectF(c.x() - 3 * ts, c.y() + dy, 6 * ts, ts)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(label.adjusted(ts, 0, -ts, 0), 4, 4)
            p.setPen(edge)
            p.drawText(label, Qt.AlignCenter, f"{tag} {card['name']}")

    def _paint_banner(self, p: QPainter) -> None:
        """绘制顶部提示横幅 (Banner Notification)，如接管模式提示或圣水问答反馈"""
        if not self.banner:
            return
        f = QFont()
        f.setBold(True)
        r = QRectF(8, 8, self.width() - 16, 26)
        size, text = 13, self.banner
        f.setPixelSize(size)
        while size > 10 and QFontMetrics(f).horizontalAdvance(text) > r.width() - 16:
            size -= 1
            f.setPixelSize(size)
        text = QFontMetrics(f).elidedText(text, Qt.ElideRight, int(r.width() - 16))
        p.setFont(f)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(r, 6, 6)
        p.setPen(QColor("#ffd400"))
        p.drawText(r, Qt.AlignCenter, text)

    def _paint_overlay(self, p: QPainter) -> None:
        """绘制居中大字覆盖层 (Overlay Text)，如开局 3, 2, 1 倒计时"""
        if not self.overlay_text:
            return
        p.fillRect(self.tile_rect(0, 0, G.W, G.H), QColor(0, 0, 0, 70))
        f = QFont()
        f.setPixelSize(max(48, self.width() // 4))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(0, 0, 0, 140))
        # 阴影字体
        p.drawText(self.rect().adjusted(4, 4, 4, 4), Qt.AlignCenter, self.overlay_text)
        # 前景亮黄字体
        p.setPen(QColor("#ffd400"))
        p.drawText(self.rect(), Qt.AlignCenter, self.overlay_text)

    def _paint_arena(self, p: QPainter) -> None:
        """草地、河道、两座木桥：按窗口尺寸绘制一次后缓存为位图，每帧只贴图"""
        key = (self.width(), self.height(), round(self.devicePixelRatioF(), 2))
        if self._arena_key != key:
            self._arena_pm = art.arena_pixmap(self, BG)
            self._arena_key = key
        p.drawPixmap(0, 0, self._arena_pm)

    def _paint_zone(self, p: QPainter) -> None:
        """选中卡牌时才显示可放置区域：合法格子合并成一块淡色区域 + 虚线轮廓 (无格子线)"""
        if not self.show_zone or not self.selected_card or not self.legal:
            return
        key = (hash(frozenset(self.legal)), self.width(), self.height())
        if self._zone_key != key:
            path = QPainterPath()
            for (x, y) in self.legal:
                path.addRect(self.tile_rect(x - 0.5, y - 0.5, 1, 1))
            self._zone_path = path.simplified()
            self._zone_key = key
        p.setBrush(QColor(255, 255, 255, 38))
        p.setPen(QPen(QColor(255, 255, 255, 150), 1.5, Qt.DashLine))
        p.drawPath(self._zone_path)

    def _paint_placement(self, p: QPainter, ts: float) -> None:
        """鼠标悬停准星：绿色 = 可放置，红色 = 不可放置；法术显示范围，远程单位显示射程虚线圈"""
        if not self.selected_card or self.hover is None:
            return
        ok = self.hover in self.legal
        card = self.session.db.cards[self.selected_card]
        c = self.to_px(*self.hover)
        col = QColor(120, 255, 160) if ok else QColor(255, 90, 90)
        fill = QColor(col.red(), col.green(), col.blue(), 70)
        spell = card.get("spell", {})
        p.setPen(QPen(col, 2))
        p.setBrush(fill)
        if "radius_tiles" in spell:
            r = spell["radius_tiles"] * ts
            p.drawEllipse(c, r, r)
            art.draw_glyph(p, "spell", c, min(r * 0.4, ts * 0.8), QColor(255, 255, 255, 200))
        elif "rolling_range_tiles" in spell:
            w = spell["width_tiles"]
            p.drawRoundedRect(self.tile_rect(self.hover[0] - w / 2, self.hover[1], w, spell["rolling_range_tiles"]),
                              ts * 0.4, ts * 0.4)
        else:
            rng = card.get("attack", {}).get("range_tiles", 0)
            if rng > 1.5:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(255, 255, 255, 120), 1, Qt.DashLine))
                p.drawEllipse(c, rng * ts, rng * ts)
            r = max(card.get("collision_radius_tiles", 0.5), 0.5) * ts * 1.3
            p.setBrush(fill)
            p.setPen(QPen(col, 2))
            p.drawEllipse(c, r, r)
            art.draw_glyph(p, self.role_of(self.selected_card), c, r * 0.55, QColor(255, 255, 255, 210))

    def _paint_rubble(self, p: QPainter, ts: float) -> None:
        """已被摧毁的塔留下废墟"""
        m = self.session.match
        for (side, role), uid in getattr(m, "tower_uid", {}).items():
            if uid in m.units:
                continue
            x, y, r = G.TOWER_LAYOUT[role]
            x, y = G.to_side(side, x, y)
            art.draw_rubble(p, self.to_px(x, y), r * ts, side)

    def _paint_effects(self, p: QPainter, ts: float) -> None:
        """持续区域法术 (旋风、毒雾、藤蔓等)：半透明圆 + 随已持续时间旋转的虚线环"""
        for ef in getattr(self.session.match, "effects", []):
            c = self.to_px(ef.x, ef.y)
            r = ef.radius * ts
            col = EFFECT_COLOR.get(ef.kind, QColor(180, 140, 255))
            left = 1.0 - ef.elapsed_ms / ef.duration_ms if ef.duration_ms else 1.0
            g = QRadialGradient(c, r)
            g.setColorAt(0, QColor(col.red(), col.green(), col.blue(), int(30 + 60 * left)))
            g.setColorAt(1, QColor(col.red(), col.green(), col.blue(), int(10 + 30 * left)))
            p.setBrush(g)
            pen = QPen(QColor(col.red(), col.green(), col.blue(), 200), max(1.5, ts * 0.1), Qt.DashLine)
            pen.setDashOffset(-ef.elapsed_ms / 60.0)
            p.setPen(pen)
            p.drawEllipse(c, r, r)

    def _paint_shadows(self, p: QPainter, units, pos, ts: float) -> None:
        """所有单位的地面阴影；空中单位阴影更淡更小 (本体画在阴影上方，显得浮起)"""
        p.setPen(Qt.NoPen)
        for u in units:
            if u.kind == "tower":
                continue
            r = self._unit_r(u, ts)
            c = self.to_px(*pos[u.uid])
            air = u.layer == "air"
            p.setBrush(QColor(0, 0, 0, 50 if air else 75))
            sr = r * (0.72 if air else 0.95)
            p.drawEllipse(QPointF(c.x(), c.y() + r * 0.55), sr, sr * 0.45)

    @staticmethod
    def _unit_r(u, ts: float) -> float:
        return max(u.radius, 0.42) * ts * 1.05

    def _unit_center(self, u, ts: float, xy) -> QPointF:
        c = self.to_px(*xy)
        if u.layer == "air":
            c = QPointF(c.x(), c.y() - self._unit_r(u, ts) * 0.9)
        return c

    def _hp_bar(self, p: QPainter, cx: float, top: float, width: float, frac: float, side: int) -> None:
        """通用血条 (阵营色、圆角、深色底槽)"""
        art.hp_bar(p, cx, top, width, frac, side)

    def _paint_tower(self, p: QPainter, u, ts: float) -> None:
        """立体石塔 (公主塔 / 国王塔)：正面暗面、垛口、阵营色塔顶、血条内嵌血量数字，国王未激活显示 z z"""
        c = self.to_px(u.x, u.y)
        art.draw_tower(p, c, u.radius * ts, ts, u.side, u.tower_role == "king", u.active, u.hp, u.max_hp,
                       self.tracker.hurt_level(u.uid, self.clock()))

    def _paint_unit(self, p: QPainter, u, ts: float, xy=None, now: Optional[float] = None) -> None:
        """单位与建筑：阵营色圆片 (建筑为圆角方块) + 类型图标 + 卡牌色小圆点；部署中半透明并显示倒计时环"""
        now = self.clock() if now is None else now
        xy = (u.x, u.y) if xy is None else xy
        kind = self.role_of(u.card_id)
        r = self._unit_r(u, ts)
        c = self._unit_center(u, ts, xy)
        ring = "champion" if kind == "champion" else ("evo" if u.evolved or u.variant == "evo" else "")
        pm = self.tokens.get(u.card_id, u.side, kind, r, self.devicePixelRatioF(), ring)
        half = art.token_extent(int(r * 4) / 4)
        hidden = bool(u.mstate.get("hidden"))
        opacity = 0.45 if u.deploy_ms > 0 else (0.4 if hidden else 1.0)
        p.setOpacity(opacity)
        p.drawPixmap(QPointF(c.x() - half, c.y() - half), pm)
        p.setOpacity(1.0)
        hurt = self.tracker.hurt_level(u.uid, now)
        if hurt > 0:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, int(150 * hurt)))
            p.drawEllipse(c, r, r)
        if "freeze" in u.status:
            p.setPen(QPen(QColor(220, 245, 255, 220), max(1.5, ts * 0.08)))
            p.setBrush(QColor(170, 225, 255, 110))
            p.drawEllipse(c, r * 1.05, r * 1.05)
        elif u.status:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 220, 80, 220), max(1.5, ts * 0.08), Qt.DotLine))
            p.drawEllipse(c, r * 1.15, r * 1.15)
        if u.mstate.get("shield"):
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(200, 230, 255, 200), max(1.5, ts * 0.1)))
            p.drawEllipse(c, r * 1.25, r * 1.25)
        if u.deploy_ms > 0:
            total = float(self.session.db.entry(u.card_id).get("deploy_time_s", 1.0)) * 1000.0 or 1000.0
            done = max(0.0, min(1.0, 1.0 - u.deploy_ms / total))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 220), max(1.5, ts * 0.08), Qt.SolidLine, Qt.RoundCap))
            p.drawArc(QRectF(c.x() - r * 1.3, c.y() - r * 1.3, r * 2.6, r * 2.6), 90 * 16, -int(360 * 16 * done))

    def _paint_unit_bars(self, p: QPainter, units, pos, ts: float) -> None:
        """头顶血条 (受伤后才显示)，不显示数字"""
        h = max(3.0, ts * 0.18)
        for u in units:
            if u.kind == "tower" or u.hp >= u.max_hp or u.deploy_ms > 0:
                continue
            r = self._unit_r(u, ts)
            c = self._unit_center(u, ts, pos[u.uid])
            art.hp_bar(p, c.x(), c.y() - r - h - 2, max(2 * r, ts * 0.9), u.hp / u.max_hp, u.side, h)

    def _paint_spells(self, p: QPainter, ts: float, now: Optional[float] = None) -> None:
        """投射物 (带拖尾的光点)、来袭范围法术 (目标圈 + 下落火球)、滚动中的滚木"""
        now = self.clock() if now is None else now
        m = self.session.match
        for pr in m.projectiles:
            x, y = self.tracker.pos(pr.uid, pr.x, pr.y, now, projectile=True)
            c = self.to_px(x, y)
            tgt = m.units.get(pr.target_uid)
            if tgt is not None:
                t = self.to_px(tgt.x, tgt.y)
                d = math.hypot(t.x() - c.x(), t.y() - c.y()) or 1.0
                ux, uy = (t.x() - c.x()) / d, (t.y() - c.y()) / d
            else:
                ux, uy = 0.0, -1.0
            tail = QPointF(c.x() - ux * ts * 1.3, c.y() - uy * ts * 1.3)
            warm = QColor(255, 236, 170) if pr.side == HUMAN else QColor(255, 205, 170)
            g = QLinearGradient(tail, c)
            g.setColorAt(0, QColor(warm.red(), warm.green(), warm.blue(), 0))
            g.setColorAt(1, QColor(warm.red(), warm.green(), warm.blue(), 230))
            p.setPen(QPen(QBrush(g), max(2.0, ts * 0.16), Qt.SolidLine, Qt.RoundCap))
            p.drawLine(tail, c)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#fff6d0"))
            rr = ts * (0.2 if pr.splash > 0 else 0.13)
            p.drawEllipse(c, rr, rr)
        for sp in m.spells:
            c = self.to_px(sp.x, sp.y)
            r = sp.radius * ts
            side = SIDE_COLOR[sp.side]
            p.setBrush(QColor(255, 150, 60, 35))
            p.setPen(QPen(QColor(side.red(), side.green(), side.blue(), 200), 2, Qt.DashLine))
            p.drawEllipse(c, r, r)
            h = min(1.0, sp.impact_in_ms / 1000.0) * ts * 4
            o = QPointF(c.x(), c.y() - h)
            g = QRadialGradient(o, ts * 0.7)
            g.setColorAt(0, QColor("#fff3b0"))
            g.setColorAt(0.5, QColor("#ff9a2e"))
            g.setColorAt(1, QColor(255, 80, 20, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(g)
            p.drawEllipse(o, ts * 0.7, ts * 0.7)
        for rs in m.rolling:
            rect = self.tile_rect(rs.x - rs.width / 2, rs.y - 0.45, rs.width, 0.9)
            g = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            g.setColorAt(0, QColor("#b07a43"))
            g.setColorAt(1, QColor("#6e4522"))
            p.setPen(QPen(QColor("#4a2e15"), 1.5))
            p.setBrush(g)
            p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
            p.setPen(QPen(QColor(74, 46, 21, 160), 1))
            for i in range(1, 4):                      # bark rings
                x = rect.left() + rect.width() * i / 4
                p.drawLine(QPointF(x, rect.top() + 2), QPointF(x, rect.bottom() - 2))

    def _paint_fx(self, p: QPainter, ts: float, now: float, kinds) -> None:
        """短时特效：落地尘环、死亡淡出、近战命中闪光、投射物命中、法术爆开、塔倒塌"""
        for f in self.tracker.fx:
            if f.kind not in kinds:
                continue
            a = f.age(now)
            c = self.to_px(f.x, f.y)
            if f.kind == "land":
                p.setBrush(Qt.NoBrush)
                k = 0.8 + 0.9 * a
                w = max(1.5, ts * 0.12) * (1 - a) + 0.5
                p.setPen(QPen(QColor(245, 235, 205, int(200 * (1 - a))), w))
                rr = max(f.size, 0.5) * ts * 1.4 * k
                p.drawEllipse(QPointF(c.x(), c.y() + ts * 0.2), rr, rr * 0.45)
            elif f.kind == "death":
                r = max(f.size, 0.42) * ts * 1.05
                if f.air:
                    c = QPointF(c.x(), c.y() - r * 0.9)
                kind = self.role_of(f.card_id) if f.card_id else "melee"
                pm = self.tokens.get(f.card_id, f.side, kind, r, self.devicePixelRatioF())
                half = art.token_extent(int(r * 4) / 4)
                p.save()
                p.setOpacity(max(0.0, 1.0 - a) * 0.85)
                p.translate(c)
                s = 1.0 - 0.35 * a
                p.scale(s, s)
                p.drawPixmap(QPointF(-half, -half), pm)
                p.restore()
                p.setPen(Qt.NoPen)
                for i in range(3):
                    p.setBrush(QColor(255, 255, 255, int((110 - i * 30) * (1 - a))))
                    p.drawEllipse(QPointF(c.x() + (i - 1) * ts * 0.25, c.y() - ts * (0.3 + i * 0.25 + a * 0.9)),
                                  ts * 0.16, ts * 0.16)
            elif f.kind == "hit":
                p.setPen(QPen(QColor(255, 255, 255, int(240 * (1 - a))), max(2.0, ts * 0.12), Qt.SolidLine,
                              Qt.RoundCap))
                grow = 0.6 + 0.6 * a
                for ang in range(0, 360, 45):
                    rr = ts * (0.55 if ang % 90 == 0 else 0.35) * grow
                    ar = math.radians(ang + 20)
                    p.drawLine(QPointF(c.x() + math.cos(ar) * ts * 0.12, c.y() + math.sin(ar) * ts * 0.12),
                               QPointF(c.x() + math.cos(ar) * rr, c.y() + math.sin(ar) * rr))
            elif f.kind == "impact":
                rr = ts * f.size * (0.6 + 0.8 * a)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(255, 245, 200, int(200 * (1 - a))))
                p.drawEllipse(c, rr, rr)
            elif f.kind in ("burst", "collapse"):
                big = f.kind == "collapse"
                rr = max(f.size, 0.6) * ts * (0.45 + 0.6 * a) * (1.4 if big else 1.0)
                fade = 1 - a
                inner = QColor(255, 230, 150) if not big else QColor(230, 225, 215)
                outer = QColor(255, 120, 40) if not big else QColor(140, 135, 130)
                g = QRadialGradient(c, rr)
                g.setColorAt(0, QColor(inner.red(), inner.green(), inner.blue(), int(170 * fade)))
                g.setColorAt(0.7, QColor(outer.red(), outer.green(), outer.blue(), int(100 * fade)))
                g.setColorAt(1, QColor(outer.red(), outer.green(), outer.blue(), 0))
                p.setPen(Qt.NoPen)
                p.setBrush(g)
                p.drawEllipse(c, rr, rr)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(255, 250, 225, int(220 * fade)), max(1.5, ts * 0.14 * fade)))
                p.drawEllipse(c, rr * 0.95, rr * 0.95)

    def _paint_result(self, p: QPainter) -> None:
        """绘制对局结算大屏：比分 (Crowns)、胜负判定文案与结算原因 (3皇冠、皇冠领先、破血比拼、训练通关)"""
        r = self.session.result
        if r is None:
            return
        p.fillRect(self.tile_rect(0, 0, G.W, G.H), QColor(0, 0, 0, 120))
        if getattr(self.session, "drill_runner", None) is not None:
            eval_res = self.session.drill_runner.evaluate()
            stars = "⭐" * eval_res.stars if eval_res.stars > 0 else "❌"
            text = f"{stars}  Cleared!" if eval_res.solved else "❌  Failed!"
            col = QColor("#ffd400") if eval_res.stars == 3 else (QColor("#5ad15a") if eval_res.solved else QColor("#ff6b5e"))
            sub_line = f"Tower damage taken: {eval_res.tower_damage_taken:.0f}"
        else:
            if r.winner is None:
                text, col = "Draw", QColor("#dddddd")
            elif r.winner == HUMAN:
                text, col = "Victory", QColor("#ffd400")
            else:
                text, col = "Defeat", QColor("#ff6b5e")
            reason_map = {"king_destroyed": "3 Crowns", "crowns": "Crown lead",
                          "tiebreaker": "Lowest tower HP", "draw": "",
                          "drill_cleared": "Drill Cleared", "drill_failed": "Drill Failed"}
            reason = reason_map.get(r.reason, "")
            sub_line = f"{r.crowns[0]} : {r.crowns[1]}   {reason}"
        f = QFont()
        f.setPixelSize(max(24, self.width() // 10))
        f.setBold(True)
        p.setFont(f)
        p.setPen(col)
        p.drawText(self.rect().adjusted(0, -40, 0, -40), Qt.AlignCenter, text)
        f.setPixelSize(max(14, self.width() // 26))
        p.setFont(f)
        p.setPen(QColor("white"))
        p.drawText(self.rect().adjusted(0, 40, 0, 40), Qt.AlignCenter, sub_line)
