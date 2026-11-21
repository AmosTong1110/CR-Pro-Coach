"""PPO self-play fine-tuning of crbot_il_v2 (Colab T4 / CPU). Needs torch (+ onnx, onnxruntime to export).

    cfg = PPOConfig(n_envs=8, updates=300)
    PPOTrainer(cfg, il_onnx="pretrained_weights/crbot_il_v2/crbot_il_v2.onnx",
               ckpt_dir="/content/drive/MyDrive/CRbot_Training/checkpoints/ppo", out_dir="...").train()

Policy = the IL network (backend_engine/brain/il_model.CRbotIL), initialised from crbot_il_v2.npz (plain arrays
saved by the IL notebook next to the .onnx; checked against the ONNX graph). Observations: obs_encoder (il_obs_v2),
so the trainer, the IL model and NeuralBot never depend on the size of the card pool.
Action at a decision point (every `decide_every_ms` of match time, only when a card is affordable, exactly when
NeuralBot decides): gate ~ Bernoulli(sigmoid(gate)); if play: slot ~ masked softmax(slot_logits), tile ~ softmax of
that slot's 576 tile logits masked by the card's legal tiles. Critic = 2 * value - 1 (the value head stays a win
probability, so the exported model keeps NeuralBot's contract). Champion / hero abilities use the scripted rules.

Rollouts: `n_envs` matches stepped in lockstep in one process (the engine + the unchanged MatchRunner). Neural
agents are *deferred*: at a decision tick they record the observation (fog of war through their own
HumanDelayBuffer, same encoder as training data and NeuralBot) and return nothing; the trainer runs one batched
forward per network for all waiting agents, and the chosen intent is submitted on the next tick (50 ms later),
still through the HumanDelayBuffer. No lookahead during training.

Reward (learner's view): +1 win / -1 loss / 0 draw at the end, plus potential-based shaping
F = gamma * Phi(s') - Phi(s) with Phi = w_tower * (own - enemy mean tower hp fraction) + w_crown * crown diff / 3
(Phi = 0 at the end, so the shaping can't change the optimal policy), minus `leak_penalty` per elixir leaked at
10. Loss: clipped PPO surrogate + value MSE - entropy bonus + kl_coef * KL(policy || frozen crbot_il_v2)
(analytic KL of the gate, the slot and the chosen slot's tile distribution), which keeps the human-like
placement grammar of the IL policy.

Opponents (fictitious self-play), per match: ScriptedBot with a random aggression (`opp_mix[0]`), frozen
crbot_il_v2 (`opp_mix[1]`), or a PPO snapshot from the pool (`opp_mix[2]`; the newest policy while the pool is
empty). Decks: random playable presets (with their evolution slots); a ScriptedBot only gets decks it can push
with. `learner_decks` (preset names) turns the learner into a deck specialist (e.g. ("Hog 2.6 Cycle",)): only
the learner's deck is restricted, opponents still get every preset. `time_budget_h` ends a run cleanly after that
many hours (e.g. 12 per day; the next run resumes from the checkpoint). Checkpoints (state + optimizer + snapshots + history) every `ckpt_every` updates, resumed automatically;
the best rolling score is exported as crbot_ppo_v2.onnx + .json (same format NeuralBot loads).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as TF

from backend_engine.brain import obs_encoder as F
from backend_engine.brain.il_model import CRbotIL, check_against_onnx, export_onnx, load_weights, model_meta, \
    save_weights
from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.contracts import ActionIntent, ActionKind, TilePos
from backend_engine.meta.presets import load_presets, playable, select_presets
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.match_runner import MatchRunner

NEG = -1e9
TOWER_ROLES = ("king", "princess_left", "princess_right")


@dataclass
class PPOConfig:
    n_envs: int = 8                    # matches per update (stepped in lockstep)
    updates: int = 300
    max_match_ms: Optional[int] = None  # cut matches short (smoke tests); None = full match
    level: int = 11
    decide_every_ms: int = 500
    gate_threshold: float = 0.45        # written to the exported meta (NeuralBot); training samples the gate
    lr: float = 2e-5
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 1024
    gamma: float = 0.995                # per decision (500 ms)
    lam: float = 0.95
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    kl_coef: float = 0.05
    max_grad_norm: float = 0.5
    w_tower: float = 0.5
    w_crown: float = 0.5
    leak_penalty: float = 0.01          # per elixir leaked at 10
    opp_mix: Tuple[float, float, float] = (0.3, 0.3, 0.4)   # scripted, frozen IL, PPO snapshots
    snapshot_every: int = 5
    pool_size: int = 10
    ckpt_every: int = 5
    score_window: int = 5               # rolling window (updates) for picking the best model
    device: str = "cpu"
    seed: int = 0
    learner_decks: Tuple[str, ...] = ()  # preset names the learner plays (a deck specialist); () = every playable preset
    time_budget_h: Optional[float] = None  # stop cleanly (checkpoint + export) after this many hours; resume next run
    export_name: str = "crbot_ppo_v2"     # <out_dir>/<export_name>.onnx (best) and <export_name>_last.onnx


# -- networks -------------------------------------------------------------------------------------------------
@dataclass
class Request:
    """A neural agent waiting for its action (filled at a decision tick)."""
    agent: "DeferredAgent"
    vec: np.ndarray
    smask: np.ndarray
    tmasks: np.ndarray                  # [4, 576]
    hand: List[str]


def sample_actions(model: CRbotIL, reqs: Sequence[Request], rng: np.random.Generator, device: str,
                   greedy: bool = False):
    """One batched forward; returns per request (play, slot, tile, logp, value_prob)."""
    if not reqs:
        return []
    obs = torch.from_numpy(np.stack([r.vec for r in reqs])).to(device)
    with torch.no_grad():
        gate, slots, tiles, value = (t.float().cpu().numpy() for t in model(obs))
    out = []
    for i, r in enumerate(reqs):
        p = 1.0 / (1.0 + math.exp(-float(gate[i])))
        play = (p >= 0.5) if greedy else (rng.random() < p)
        logp = math.log(max(p if play else 1.0 - p, 1e-12))
        slot = tile = -1
        if play:
            sp = _masked_softmax(slots[i], r.smask)
            slot = int(np.argmax(sp)) if greedy else int(rng.choice(4, p=sp))
            tp = _masked_softmax(tiles[i, slot], r.tmasks[slot])
            tile = int(np.argmax(tp)) if greedy else int(rng.choice(F.NUM_TILES, p=tp))
            logp += math.log(max(sp[slot], 1e-12)) + math.log(max(tp[tile], 1e-12))
        out.append((bool(play), slot, tile, logp, float(value[i])))
    return out


def _masked_softmax(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, x.astype(np.float64), -np.inf)
    z -= z[mask].max()
    e = np.where(mask, np.exp(z), 0.0)
    return e / e.sum()


class DeferredAgent:
    """MatchRunner agent driven by a network: records a Request at decision ticks, acts on the next tick."""

    def __init__(self, db: CardDB, side: int, brain_key: str, every_ms: int):
        self.db, self.side, self.brain_key, self.every = db, side, brain_key, every_ms
        self.rules = ScriptedBot(db, side)
        self.pending: Optional[List[ActionIntent]] = None
        self.request: Optional[Request] = None

    def decide(self, obs, visible_events, now_ms, buffer=None) -> List[ActionIntent]:
        if self.pending is not None:
            out, self.pending = self.pending, None
            return out
        ab = self.rules._ability(obs)
        if ab:
            return [ab]
        if now_ms % self.every:
            return []
        smask = F.slot_mask(obs, self.db)
        if not smask.any():
            return []
        hand = list(obs["hand"][:F.HAND_SLOTS])
        tm = np.zeros((F.HAND_SLOTS, F.NUM_TILES), dtype=bool)
        for k, c in enumerate(hand):
            if smask[k]:
                tm[k] = F.tile_mask(obs, c, self.db)
        vec = F.encode_observation(obs, obs.get("opp_estimate"), obs.get("opp_deck_guess"), self.db)
        self.request = Request(self, vec, smask, tm, hand)
        return []

    def answer(self, play: bool, slot: int, tile: int) -> None:
        if not play:
            self.pending = []
            return
        card = self.request.hand[slot]
        x, y = F.tile_center(self.side, tile)
        self.pending = [ActionIntent(ActionKind.DEPLOY, card, TilePos(x, y), self.db.cards[card]["elixir"],
                                     source="policy")]


# -- one match ------------------------------------------------------------------------------------------------
@dataclass
class Step:
    vec: np.ndarray
    smask: np.ndarray
    tmask: np.ndarray                  # chosen slot's legal tiles (zeros when waiting)
    play: bool
    slot: int
    tile: int
    logp: float
    value: float                       # critic (2p - 1)
    phi: float
    leaked: float
    reward: float = 0.0


@dataclass
class Episode:
    steps: List[Step] = field(default_factory=list)
    done: bool = False                 # ended by the game (not cut short)
    result: float = 0.0                # +1 / -1 / 0 from the learner's view
    opponent: str = ""
    learner_side: int = 0


class Env:
    def __init__(self, trainer: "PPOTrainer", rng: random.Random):
        t = trainer
        self.cfg, self.db = t.cfg, t.db
        self.learner_side = rng.randrange(2)
        L, O = self.learner_side, 1 - self.learner_side
        kind = rng.choices(("scripted", "il", "snapshot"), weights=t.cfg.opp_mix)[0]
        mine = rng.choice(t.learner_presets)
        theirs = rng.choice(t.pushers if kind == "scripted" else t.presets)
        decks, evos = [None, None], [None, None]
        decks[L], evos[L] = mine.cards, mine.evo
        decks[O], evos[O] = theirs.cards, theirs.evo
        self.match = Match(MatchConfig((decks[0], decks[1]), (t.cfg.level,) * 2, (evos[0], evos[1]),
                                       seed=rng.randrange(1 << 30)), self.db)
        self.learner = DeferredAgent(self.db, L, "learner", t.cfg.decide_every_ms)
        if kind == "scripted":
            opp = ScriptedBot(self.db, O, rng.uniform(0.3, 0.9))
        else:
            key = "il" if kind == "il" else t.pick_snapshot(rng)
            opp = DeferredAgent(self.db, O, key, t.cfg.decide_every_ms)
        self.opponent = opp
        self.runner = MatchRunner(self.match, {L: self.learner, O: opp})
        self.episode = Episode(opponent=kind if kind != "snapshot" else f"snapshot:{opp.brain_key}",
                               learner_side=L)
        self.end_ms = self.cfg.max_match_ms or (self.match.end_ms + 50)

    @property
    def finished(self) -> bool:
        return self.match.result is not None or self.match.t_ms >= self.end_ms

    def phi(self) -> float:
        m, L = self.match, self.learner_side
        def mean_hp(side):
            tot = 0.0
            for role in TOWER_ROLES:
                u = m.tower(side, role)
                tot += (u.hp / u.max_hp) if u is not None else 0.0
            return tot / 3.0
        crowns = (m.sides[L].crowns - m.sides[1 - L].crowns) / 3.0
        return self.cfg.w_tower * (mean_hp(L) - mean_hp(1 - L)) + self.cfg.w_crown * crowns

    def leaked(self) -> float:
        return float(self.match.sides[self.learner_side].leaked_elixir)

    def close_last(self, phi_next: float, leaked_now: float) -> None:
        """Finish the previous step's reward with the shaping up to now."""
        if not self.episode.steps:
            return
        s = self.episode.steps[-1]
        s.reward += self.cfg.gamma * phi_next - s.phi - self.cfg.leak_penalty * (leaked_now - s.leaked)

    def finish(self) -> Episode:
        ep = self.episode
        res = self.match.result
        if res is not None:
            ep.done = True
            ep.result = 0.0 if res.winner is None else (1.0 if res.winner == self.learner_side else -1.0)
            self.close_last(0.0, self.leaked())         # Phi(terminal) = 0
            if ep.steps:
                ep.steps[-1].reward += ep.result
        else:                                           # cut short: shaping up to here, value bootstraps
            self.close_last(self.phi(), self.leaked())
        return ep


# -- trainer --------------------------------------------------------------------------------------------------
class PPOTrainer:
    def __init__(self, cfg: PPOConfig, il_onnx: Path | str, ckpt_dir: Path | str, out_dir: Path | str,
                 il_meta: Optional[Path | str] = None, log=print):
        self.cfg, self.log = cfg, log
        self.db = CardDB.load()
        self.device = cfg.device
        self.il_onnx = Path(il_onnx)
        self.il_meta = json.loads(Path(il_meta or self.il_onnx.with_suffix(".json")).read_text())
        reason = F.check_model_meta(self.il_meta)
        if reason:
            raise ValueError(f"{self.il_onnx.name}: {reason}")
        weights = self.il_onnx.with_suffix(".npz")
        if not weights.exists():
            raise ValueError(f"{weights} is missing: the IL notebook saves it next to the .onnx")
        ref = load_weights(weights)
        check_against_onnx(ref, self.il_onnx)
        ref = ref.to(self.device)
        self.model = copy.deepcopy(ref)                 # the learner (copied before the reference is frozen)
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.train()
        self.ref = ref.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.presets = playable(self.db.cards, load_presets())
        self.pushers = [p for p in self.presets if p.ai_pushes] or self.presets
        self.learner_presets = select_presets(self.presets, cfg.learner_decks)
        self.ckpt_dir, self.out_dir = Path(ckpt_dir), Path(out_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.update = 0
        self.history: List[dict] = []
        self.best_score = -1e9
        self.snapshots: Dict[str, CRbotIL] = {}
        self.rng = random.Random(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self._resume()

    # -- pool / checkpoints ---------------------------------------------------------------------------------
    def pick_snapshot(self, rng: random.Random) -> str:
        return rng.choice(sorted(self.snapshots)) if self.snapshots else "latest"

    def _frozen(self, model: CRbotIL) -> CRbotIL:
        m = copy.deepcopy(model).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def _add_snapshot(self) -> None:
        key = f"snap_{self.update:05d}"
        self.snapshots[key] = self._frozen(self.model)
        torch.save(self.model.state_dict(), self.ckpt_dir / f"{key}.pt")
        for old in sorted(self.snapshots)[:-self.cfg.pool_size]:
            del self.snapshots[old]
            (self.ckpt_dir / f"{old}.pt").unlink(missing_ok=True)

    def save(self) -> Path:
        path = self.ckpt_dir / "ppo_state.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({"model": self.model.state_dict(), "optim": self.opt.state_dict(), "update": self.update,
                    "best_score": self.best_score, "history": json.dumps(self.history),
                    "snapshots": sorted(self.snapshots), "cfg": json.dumps(asdict(self.cfg)),
                    "rng": json.dumps(self.rng.getstate(), default=list)}, tmp)
        tmp.replace(path)                               # atomic on Drive too: never a half-written checkpoint
        return path

    def _resume(self) -> None:
        path = self.ckpt_dir / "ppo_state.pt"
        if not path.exists():
            return
        st = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["optim"])
        self.update, self.best_score = int(st["update"]), float(st["best_score"])
        self.history = json.loads(st["history"])
        for key in st["snapshots"]:
            f = self.ckpt_dir / f"{key}.pt"
            if f.exists():
                m = copy.deepcopy(self.ref)
                m.load_state_dict(torch.load(f, map_location=self.device, weights_only=True))
                self.snapshots[key] = m.eval()
        self.rng = random.Random(self.cfg.seed + self.update)
        self.log(f"resumed from update {self.update} ({len(self.snapshots)} snapshots)")

    # -- rollouts --------------------------------------------------------------------------------------------
    def collect(self) -> List[Episode]:
        envs = [Env(self, self.rng) for _ in range(self.cfg.n_envs)]
        latest = self._frozen(self.model)
        brains = {"learner": self.model, "il": self.ref, "latest": latest, **self.snapshots}
        live = list(envs)
        while live:
            for env in live:
                env.runner.tick()
            waiting: Dict[str, List[Request]] = {}
            for env in live:
                for ag in (env.learner, env.opponent):
                    if isinstance(ag, DeferredAgent) and ag.request is not None:
                        waiting.setdefault(ag.brain_key, []).append(ag.request)
            self.model.eval()
            for key, reqs in waiting.items():
                for r, (play, slot, tile, logp, vprob) in zip(reqs, sample_actions(brains[key], reqs, self.np_rng,
                                                                                    self.device)):
                    ag = r.agent
                    if key == "learner":
                        env = next(e for e in live if e.learner is ag)
                        phi, leaked = env.phi(), env.leaked()
                        env.close_last(phi, leaked)
                        tm = r.tmasks[slot] if play else np.zeros(F.NUM_TILES, dtype=bool)
                        env.episode.steps.append(Step(r.vec, r.smask, tm, play, slot, tile, logp, 2 * vprob - 1,
                                                      phi, leaked))
                    ag.answer(play, slot, tile)
                    ag.request = None
            self.model.train()
            live = [e for e in live if not e.finished]
        return [e.finish() for e in envs]

    # -- PPO ------------------------------------------------------------------------------------------------
    def _gae(self, ep: Episode) -> Tuple[np.ndarray, np.ndarray]:
        n = len(ep.steps)
        adv = np.zeros(n, dtype=np.float32)
        vals = np.array([s.value for s in ep.steps], dtype=np.float32)
        rews = np.array([s.reward for s in ep.steps], dtype=np.float32)
        last_v = 0.0 if ep.done else (float(vals[-1]) if n else 0.0)
        gae = 0.0
        for k in reversed(range(n)):
            nv = last_v if k == n - 1 else vals[k + 1]
            delta = rews[k] + self.cfg.gamma * nv - vals[k]
            gae = delta + self.cfg.gamma * self.cfg.lam * gae
            adv[k] = gae
        return adv, adv + vals

    def _policy_terms(self, model: CRbotIL, obs, smask, tmask, play, slot, tile):
        """log pi(a), entropy, and the distributions (for KL) of a batch of taken actions."""
        gate, slots, tiles, value = model(obs)
        lg_play, lg_wait = TF.logsigmoid(gate), TF.logsigmoid(-gate)
        lp = torch.where(play, lg_play, lg_wait)
        s_logp = torch.log_softmax(slots.masked_fill(~smask, NEG), -1)
        slot_c = slot.clamp(min=0)
        row = tiles.gather(1, slot_c[:, None, None].expand(-1, 1, F.NUM_TILES))[:, 0]
        safe_tmask = torch.where(play[:, None], tmask, torch.ones_like(tmask))
        t_logp = torch.log_softmax(row.masked_fill(~safe_tmask, NEG), -1)
        lp = lp + torch.where(play, s_logp.gather(1, slot_c[:, None])[:, 0] + t_logp.gather(1, tile.clamp(min=0)[:, None])[:, 0],
                              torch.zeros_like(lp))
        p_play = torch.sigmoid(gate)
        ent_gate = -(p_play * lg_play + (1 - p_play) * lg_wait)
        ent_slot = -(s_logp.exp() * s_logp.masked_fill(~smask, 0)).sum(-1)
        ent_tile = -(t_logp.exp() * t_logp.masked_fill(~safe_tmask, 0)).sum(-1)
        ent = ent_gate + torch.where(play, ent_slot + ent_tile, torch.zeros_like(ent_gate))
        return lp, ent, (lg_play, lg_wait, s_logp, t_logp), value

    def _kl(self, cur, ref, smask, tmask, play) -> torch.Tensor:
        (cp, cw, cs, ct), (rp, rw, rs, rt) = cur, ref
        kl_gate = cp.exp() * (cp - rp) + cw.exp() * (cw - rw)
        kl_slot = (cs.exp() * (cs - rs)).masked_fill(~smask, 0).sum(-1)
        safe = torch.where(play[:, None], tmask, torch.ones_like(tmask))
        kl_tile = (ct.exp() * (ct - rt)).masked_fill(~safe, 0).sum(-1)
        return kl_gate + torch.where(play, kl_slot + kl_tile, torch.zeros_like(kl_gate))

    def learn(self, episodes: Sequence[Episode]) -> dict:
        steps, advs, rets = [], [], []
        for ep in episodes:
            if not ep.steps:
                continue
            a, r = self._gae(ep)
            steps.extend(ep.steps)
            advs.append(a)
            rets.append(r)
        if not steps:
            return {"steps": 0}
        dev = self.device
        obs = torch.from_numpy(np.stack([s.vec for s in steps])).to(dev)
        smask = torch.from_numpy(np.stack([s.smask for s in steps])).to(dev)
        tmask = torch.from_numpy(np.stack([s.tmask for s in steps])).to(dev)
        play = torch.tensor([s.play for s in steps], device=dev)
        slot = torch.tensor([s.slot for s in steps], device=dev)
        tile = torch.tensor([s.tile for s in steps], device=dev)
        old_lp = torch.tensor([s.logp for s in steps], dtype=torch.float32, device=dev)
        adv = torch.from_numpy(np.concatenate(advs)).to(dev)
        ret = torch.from_numpy(np.concatenate(rets)).to(dev).clamp(-1.5, 1.5)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
        n = len(steps)
        stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl_ref": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
        batches = 0
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, self.cfg.minibatch):
                b = perm[i:i + self.cfg.minibatch]
                lp, ent, dists, value = self._policy_terms(self.model, obs[b], smask[b], tmask[b], play[b], slot[b],
                                                           tile[b])
                with torch.no_grad():
                    _, _, ref_d, _ = self._policy_terms(self.ref, obs[b], smask[b], tmask[b], play[b], slot[b],
                                                        tile[b])
                ratio = torch.exp(lp - old_lp[b])
                s1, s2 = ratio * adv[b], torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv[b]
                pg = -torch.min(s1, s2).mean()
                vf = ((2 * value - 1) - ret[b]).pow(2).mean()
                kl_ref = self._kl(dists, ref_d, smask[b], tmask[b], play[b]).mean()
                loss = pg + self.cfg.vf_coef * vf - self.cfg.ent_coef * ent.mean() + self.cfg.kl_coef * kl_ref
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["pg"] += pg.item()
                    stats["vf"] += vf.item()
                    stats["ent"] += ent.mean().item()
                    stats["kl_ref"] += kl_ref.item()
                    stats["approx_kl"] += (old_lp[b] - lp).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > self.cfg.clip).float().mean().item()
                batches += 1
        out = {k: v / max(batches, 1) for k, v in stats.items()}
        out["steps"] = n
        return out

    # -- main loop ------------------------------------------------------------------------------------------
    def score(self) -> float:
        recent = self.history[-self.cfg.score_window:]
        return float(np.mean([h["score"] for h in recent])) if recent else -1e9

    def train(self, updates: Optional[int] = None) -> dict:
        target = self.update + (updates if updates is not None else self.cfg.updates - self.update)
        deadline = time.time() + self.cfg.time_budget_h * 3600 if self.cfg.time_budget_h else None
        name = self.cfg.export_name
        while self.update < target:
            if deadline is not None and time.time() >= deadline:
                self.log(f"time budget of {self.cfg.time_budget_h} h used at update {self.update}: "
                         f"stopping (the next run resumes here)")
                break
            t0 = time.time()
            eps = self.collect()
            t1 = time.time()
            st = self.learn(eps)
            self.update += 1
            res = [e.result for e in eps]
            by_opp: Dict[str, List[float]] = {}
            for e in eps:
                by_opp.setdefault(e.opponent.split(":")[0], []).append(e.result)
            rec = {"update": self.update, "matches": len(eps), "score": float(np.mean([(r + 1) / 2 for r in res])),
                   "wins": sum(r > 0 for r in res), "losses": sum(r < 0 for r in res),
                   "by_opponent": {k: round(float(np.mean([(r + 1) / 2 for r in v])), 3) for k, v in by_opp.items()},
                   "plays_per_match": round(float(np.mean([sum(s.play for s in e.steps) for e in eps])), 1),
                   "rollout_s": round(t1 - t0, 1), "learn_s": round(time.time() - t1, 1),
                   **{k: round(v, 5) if isinstance(v, float) else v for k, v in st.items()}}
            self.history.append(rec)
            self.log(f"update {self.update:4d} | score {rec['score']:.2f} ({rec['wins']}-{rec['losses']}) "
                     f"{rec['by_opponent']} | steps {rec['steps']} | kl_ref {rec.get('kl_ref', 0):.4f} "
                     f"approx_kl {rec.get('approx_kl', 0):.4f} | {rec['rollout_s']}s + {rec['learn_s']}s")
            if self.update % self.cfg.snapshot_every == 0:
                self._add_snapshot()
            if len(self.history) >= min(self.cfg.score_window, 2) and self.score() > self.best_score:
                self.best_score = self.score()
                self.export(name)
                self.log(f"  new best rolling score {self.best_score:.3f}: exported {name}")
            if self.update % self.cfg.ckpt_every == 0:
                self.save()
        self.save()
        self.export(f"{name}_last")
        return {"updates": self.update, "best_score": self.best_score}

    def export(self, name: str) -> Tuple[Path, Path]:
        """<out_dir>/<name>.onnx + .npz + .json (the NeuralBot contract; gate threshold / cadence as configured)."""
        model = self._frozen(self.model).cpu()
        onnx_path = self.out_dir / f"{name}.onnx"
        diff = export_onnx(model, onnx_path, verify=True)
        save_weights(model, self.out_dir / f"{name}.npz")
        meta = model_meta(name)
        meta.update({"gate_threshold": self.cfg.gate_threshold,
                     "decide_every_ms": self.cfg.decide_every_ms, "init": "crbot_il_v2",
                     "ppo": {"update": self.update, "best_rolling_score": round(self.best_score, 4),
                             "cfg": asdict(self.cfg), "history": self.history[-50:]},
                     "onnx_check_max_abs_diff": diff,
                     "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest()})
        meta_path = self.out_dir / f"{name}.json"
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return onnx_path, meta_path
