"""
竞技场棋盘渲染画布控件 (Arena Board Canvas Widget)
使用 PySide6 QPainter 高性能绘制 18x32 皇室战争战斗棋盘，
并负责处理鼠标悬停预览、下牌落点点击以及复盘最优解虚影 (Ghost Tiles) 渲染。
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QWidget

from backend_engine.simulator import geometry as G
from frontend_app.qt.session import AI, HUMAN, GameSession

# 阵营色彩：人类玩家为皇家蓝，AI 为赤红
SIDE_COLOR = {HUMAN: QColor("#2f7de1"), AI: QColor("#e0453a")}
# 棋盘草地交错双色格
GRASS_A, GRASS_B = QColor("#73b04f"), QColor("#6aa748")
# 河道与桥梁色彩
RIVER, BRIDGE = QColor("#3b8fd9"), QColor("#a7814f")
# 下牌合法与非法高亮色彩
LEGAL, ILLEGAL = QColor(80, 255, 120, 70), QColor(255, 60, 60, 110)


def short_name(name: str) -> str:
    """提取卡牌名称缩写 (如 Hog Rider -> HR) 用于棋盘小圆形单位内的文字标识"""
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
        self.show_ai_vision = True
        self.debug_overlay = True   # Coach / Physics Debug Overlay (Grid, Bounds, Vectors, Countdown)
        self.overlay_text: Optional[str] = None   # 倒计时遮罩大字 (如快退接管后的 3-2-1)
        # 复盘虚影：(card_id, x, y, "actual"实际打法 | "best"推荐最优解)
        self.ghosts: List[Tuple[str, float, float, str]] = []
        self.banner: Optional[str] = None         # 棋盘顶部的战术横幅提示文本
        self.setMouseTracking(True)
        self.setMinimumSize(int(G.W * 12), int(G.H * 12))

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
    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#15171c"))
        ts, _, _ = self._geom()
        # 1. 绘制草地、河道与桥梁
        self._paint_arena(p)
        # 1b. 绘制物理调试网格与边界 (Coach / Physics Debug Overlay)
        if self.debug_overlay:
            self._paint_debug_grid(p, ts)
        # 2. 绘制下牌准星与合法区域高亮
        self._paint_placement(p, ts)
        m = self.session.match
        # 2b. 绘制索敌连线与寻路向量 (Coach / Physics Debug Overlay)
        if self.debug_overlay:
            self._paint_debug_vectors(p, m, ts)
        # 3. 绘制场上存活实体 (地面优先，空中居上)
        for u in sorted(m.units.values(), key=lambda u: (u.layer == "air", u.uid)):
            if u.kind == "tower":
                self._paint_tower(p, u, ts)
            else:
                self._paint_unit(p, u, ts)
        # 4. 绘制法术弹道与滚木
        self._paint_spells(p, ts)
        # 5. 绘制复盘红绿虚影
        self._paint_ghosts(p, ts)
        # 6. 局终结算蒙层或倒计时大字
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
        """绘制竞技场地表网格：双色交替草地、中间河道 (River) 以及左右两座桥梁 (Bridges)"""
        p.setPen(Qt.NoPen)
        # 棋盘格交替草地背景
        for y in range(int(G.H)):
            for x in range(int(G.W)):
                p.setBrush(GRASS_A if (x + y) % 2 == 0 else GRASS_B)
                p.drawRect(self.tile_rect(x, y, 1, 1))
        # 中部横向河道
        p.setBrush(RIVER)
        p.drawRect(self.tile_rect(0, G.RIVER_Y0, G.W, G.RIVER_Y1 - G.RIVER_Y0))
        # 左右两座石桥
        p.setBrush(BRIDGE)
        for bx in G.BRIDGE_X:
            p.drawRect(self.tile_rect(bx - G.BRIDGE_HALF_W, G.RIVER_Y0 - 0.2, 2 * G.BRIDGE_HALF_W,
                                      G.RIVER_Y1 - G.RIVER_Y0 + 0.4))

    def toggle_debug_overlay(self) -> bool:
        """切换物理/战术调试覆盖层 (网格、河道边界、路别中线、索敌向量与倒计时)"""
        self.debug_overlay = not self.debug_overlay
        self.update()
        return self.debug_overlay

    def _paint_debug_grid(self, p: QPainter, ts: float) -> None:
        """绘制物理网格线与河道/桥梁分界线 (Coach / Physics Debug Overlay)"""
        # 1. 浅色全图网格虚线
        p.setPen(QPen(QColor(255, 255, 255, 20), 1, Qt.DotLine))
        for x in range(1, int(G.W)):
            p.drawLine(self.to_px(x, 0), self.to_px(x, G.H))
        for y in range(1, int(G.H)):
            p.drawLine(self.to_px(0, y), self.to_px(G.W, y))
        # 2. 中路黄金分路中线 (x = 9.0)
        p.setPen(QPen(QColor(255, 215, 0, 80), 1.5, Qt.DashLine))
        p.drawLine(self.to_px(G.W / 2, 0), self.to_px(G.W / 2, G.H))
        # 3. 河道几何边界 (y = 15.0 & y = 17.0)
        p.setPen(QPen(QColor(0, 220, 255, 90), 1.2, Qt.DashLine))
        p.drawLine(self.to_px(0, G.RIVER_Y0), self.to_px(G.W, G.RIVER_Y0))
        p.drawLine(self.to_px(0, G.RIVER_Y1), self.to_px(G.W, G.RIVER_Y1))
        # 4. 左右桥梁外框线
        p.setPen(QPen(QColor(255, 255, 255, 120), 1.2))
        for bx in G.BRIDGE_X:
            r = self.tile_rect(bx - G.BRIDGE_HALF_W, G.RIVER_Y0, 2 * G.BRIDGE_HALF_W, G.RIVER_Y1 - G.RIVER_Y0)
            p.drawRect(r)

    def _paint_debug_vectors(self, p: QPainter, m, ts: float) -> None:
        """绘制当前场上单位的目标锁定寻路向量 (Coach / Physics Debug Overlay)"""
        for u in m.units.values():
            if u.hp <= 0 or u.kind != "troop" or u.target_uid is None:
                continue
            tgt = m.units.get(u.target_uid)
            if tgt is not None and tgt.hp > 0:
                col = QColor(80, 170, 255, 110) if u.side == 0 else QColor(255, 90, 90, 110)
                p.setPen(QPen(col, 1.2, Qt.DashLine))
                p.drawLine(self.to_px(u.x, u.y), self.to_px(tgt.x, tgt.y))

    def _paint_placement(self, p: QPainter, ts: float) -> None:
        """绘制选中卡牌时的下牌准星、合法下牌区域高亮与射程指示圈"""
        if not self.selected_card:
            return
        # 高亮当前卡牌的所有合法部署格子
        p.setPen(Qt.NoPen)
        p.setBrush(LEGAL)
        for (x, y) in self.legal:
            p.drawRect(self.tile_rect(x - 0.5, y - 0.5, 1, 1))
        if self.hover is None:
            return
        # 鼠标悬停位置：若在合法区域则显示绿色准星，否则显示红色违规指示
        ok = self.hover in self.legal
        card = self.session.db.cards[self.selected_card]
        color = QColor(60, 220, 100, 150) if ok else ILLEGAL
        p.setBrush(color)
        p.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
        c = self.to_px(*self.hover)
        spell = card.get("spell", {})
        if "radius_tiles" in spell:
            # 圆形法术影响半径
            r = spell["radius_tiles"] * ts
            p.drawEllipse(c, r, r)
        elif "rolling_range_tiles" in spell:
            # 矩形滚木滚动轨迹预览
            w = spell["width_tiles"]
            p.drawRect(self.tile_rect(self.hover[0] - w / 2, self.hover[1], w, spell["rolling_range_tiles"]))
        else:
            # 单位部署体积
            r = max(card.get("collision_radius_tiles", 0.5), 0.45) * ts
            p.drawEllipse(c, r, r)
            # 远程单位显示攻击距离虚线圆圈
            rng = card.get("attack", {}).get("range_tiles", 0)
            if rng > 1.5:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(255, 255, 255, 110), 1, Qt.DashLine))
                p.drawEllipse(c, rng * ts, rng * ts)

    def _hp_bar(self, p: QPainter, cx: float, top: float, width: float, frac: float, side: int) -> None:
        """绘制通用血条：底色深灰，前景色按阵营偏亮色显示当前血量比例"""
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 160))
        p.drawRect(QRectF(cx - width / 2, top, width, 4))
        p.setBrush(SIDE_COLOR[side].lighter(130))
        p.drawRect(QRectF(cx - width / 2, top, width * max(0.0, min(1.0, frac)), 4))

    def _paint_tower(self, p: QPainter, u, ts: float) -> None:
        """绘制皇家皇家塔 (公主塔 / 国王塔)，包括当前血量与国王未激活沉睡状态 (zzz)"""
        r = u.radius * ts
        c = self.to_px(u.x, u.y)
        base = SIDE_COLOR[u.side]
        p.setPen(QPen(base.darker(160), 2))
        p.setBrush(base if u.active else base.darker(140))
        p.drawRoundedRect(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), 4, 4)
        p.setPen(QColor("white"))
        f = QFont()
        f.setPixelSize(max(9, int(ts * 0.55)))
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), Qt.AlignCenter, f"{int(u.hp)}")
        if u.tower_role == "king" and not u.active:
            # 国王塔未受攻击且两座公主塔均在时处于沉睡未激活状态
            f.setPixelSize(max(8, int(ts * 0.4)))
            p.setFont(f)
            p.drawText(QRectF(c.x() - r, c.y() + r * 0.25, 2 * r, r * 0.7), Qt.AlignCenter, "zzz")
        self._hp_bar(p, c.x(), c.y() - r - 6, 2 * r, u.hp / u.max_hp, u.side)

    def _paint_unit(self, p: QPainter, u, ts: float) -> None:
        """绘制活体单位与建筑，支持部署半透明、空中飞行单位阴影、迷雾盲区虚线高亮"""
        r = max(u.radius, 0.4) * ts
        c = self.to_px(u.x, u.y)
        col = QColor(SIDE_COLOR[u.side])
        if u.deploy_ms > 0:
            # 部署生效延迟期 (半透明提示未完全登场)
            col.setAlpha(110)
        unseen = self.show_ai_vision and not self.session.ai_has_seen(u)
        if u.layer == "air":
            # 飞行空中单位 (在地面下方绘制椭圆暗影，增强空间立体感)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 60))
            p.drawEllipse(QPointF(c.x() + r * 0.3, c.y() + r * 0.5), r * 0.8, r * 0.5)
        p.setBrush(col)
        pen = QPen(QColor("#ffd400") if unseen else col.darker(170), 2.5 if unseen else 1.5)
        if unseen:
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        if u.kind == "building":
            # 建筑绘制为方形
            p.drawRect(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r))
        else:
            # 兵种单位绘制为圆盘
            p.drawEllipse(c, r, r)
        p.setPen(QColor("white"))
        f = QFont()
        f.setPixelSize(max(8, int(r * 0.85)))
        f.setBold(True)
        p.setFont(f)
        name = self.session.db.entry(u.card_id)["name"]
        p.drawText(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), Qt.AlignCenter, short_name(name))
        if u.hp < u.max_hp:
            self._hp_bar(p, c.x(), c.y() - r - 5, 2 * r, u.hp / u.max_hp, u.side)

        # Coach / Physics Debug Overlay: 部署倒计时与路别标签
        if self.debug_overlay:
            if u.deploy_ms > 0:
                df = QFont()
                df.setPixelSize(max(8, int(ts * 0.45)))
                df.setBold(True)
                p.setFont(df)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(0, 0, 0, 190))
                pill = QRectF(c.x() - r * 1.2, c.y() - r - 16, r * 2.4, 12)
                p.drawRoundedRect(pill, 3, 3)
                p.setPen(QColor("#ffd400"))
                p.drawText(pill, Qt.AlignCenter, f"{u.deploy_ms / 1000.0:.1f}s")
            if getattr(u, "lane", None):
                lf = QFont()
                lf.setPixelSize(max(7, int(ts * 0.35)))
                lf.setBold(True)
                p.setFont(lf)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(0, 0, 0, 160))
                lpill = QRectF(c.x() + r * 0.5, c.y() - r * 0.8, 10, 10)
                p.drawEllipse(lpill)
                p.setPen(QColor("#80d8ff") if u.lane == "left" else QColor("#ff8a80"))
                p.drawText(lpill, Qt.AlignCenter, "L" if u.lane == "left" else "R")

    def _paint_spells(self, p: QPainter, ts: float) -> None:
        """绘制飞行中的法术弹道光圈与滚动推进中的滚木 (The Log)"""
        m = self.session.match
        for sp in m.spells:
            c = self.to_px(sp.x, sp.y)
            p.setPen(QPen(QColor("#ff9b2f"), 2, Qt.DashLine))
            p.setBrush(QColor(255, 140, 40, 60))
            p.drawEllipse(c, sp.radius * ts, sp.radius * ts)
        for rs in m.rolling:
            p.setPen(QPen(QColor("#5a3b1c"), 2))
            p.setBrush(QColor("#8b5a2b"))
            p.drawRect(self.tile_rect(rs.x - rs.width / 2, rs.y - 0.4, rs.width, 0.8))

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
