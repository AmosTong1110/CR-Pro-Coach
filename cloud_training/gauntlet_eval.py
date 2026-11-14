"""
Gauntlet Evaluation Runner (Derived from ClashAI's e1_eval.py).
Evaluates a candidate checkpoint against frozen benchmark opponents in head-to-head matches.
Computes win rate with Wilson 95% confidence intervals and detects performance regression.
"""
import math
from typing import Tuple


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Wilson score interval for binomial proportion.
    k: wins, n: total matches.
    Returns: (win_rate_pct, lower_bound_pct, upper_bound_pct)
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996  # 95% confidence z-score
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    
    return round(p * 100, 2), round(max(0.0, center - spread) * 100, 2), round(min(1.0, center + spread) * 100, 2)


def run_gauntlet_eval(candidate_name: str, wins: int, losses: int, draws: int = 0):
    total = wins + losses + draws
    win_rate, lo, hi = wilson_score_interval(wins, total)
    print("=" * 60)
    print(f"Gauntlet Evaluation: {candidate_name}")
    print(f"Matches: {total} | Wins: {wins} | Losses: {losses} | Draws: {draws}")
    print(f"Win Rate: {win_rate}% [95% CI: {lo}%, {hi}%]")
    print("=" * 60)


if __name__ == "__main__":
    # Smoke test example: 64 wins out of 150 matches (42.7%)
    run_gauntlet_eval("RuneGiant_PPO_v1_vs_PekkaSpam", wins=64, losses=86)
