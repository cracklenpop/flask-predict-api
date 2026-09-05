"""Turning picks into a plan that targets doubling the bank on a match day.

This module exists because "only bet what you are sure about" and "double my
money today" pull in opposite directions, and the tension has to be solved
explicitly rather than wished away.

The arithmetic is unavoidable: a selection you are 92% sure of is priced near
1.10, and 1.10 does not double anything. To return 2x you need combined odds of
at least 2.00, which means either one genuinely uncertain bet, or several
confident ones multiplied together. Multiplying confident legs is the better
trade - four legs at 92% is a 71.6% chance of doubling, where a single bet at
2.00 that you are 55% sure of is a 55% chance - but it is still a chance, and
this module's job is to report that number honestly rather than dress it up.

Three plan shapes are searched:

  SINGLE   - one selection priced at 2.00 or better.
  PARLAY   - the accumulator with the highest win probability among those whose
             combined price clears the target.
  SPLIT    - the stake divided across several disjoint parlays, so any one of
             them landing doubles the bank. Each parlay must then clear a
             proportionally higher price, but the chances add up rather than
             multiply, which can beat a single parlay outright.

Legs are always drawn from different matches. Two selections on the same fixture
are correlated, and multiplying their probabilities as if they were independent
is the single fastest way to turn a 60% plan into a 40% one without noticing.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .conviction import Pick


@dataclass
class Plan:
    """A concrete staking plan for one match day."""

    kind: str                      # SINGLE / PARLAY / SPLIT
    legs: list[list[Pick]]         # one inner list per parlay
    stake_share: list[float]       # fraction of the day's stake on each parlay
    combined_odds: list[float]
    leg_probs: list[float]         # win probability of each parlay
    p_double: float                # P(returning at least the target multiple)
    ev: float                      # expected return per 1 staked
    target: float
    kelly_bankroll_fraction: float

    @property
    def n_legs(self) -> int:
        return sum(len(l) for l in self.legs)

    def describe(self, stake: float = 100.0, currency: str = "R") -> str:
        out = [f"PLAN: {self.kind}   target {self.target:.2f}x",
               f"  chance of hitting the target : {self.p_double*100:.1f}%",
               f"  expected return per 1 staked : {self.ev:.3f}  "
               f"({'+' if self.ev >= 1 else ''}{(self.ev-1)*100:.1f}%)",
               f"  suggested share of bankroll  : {self.kelly_bankroll_fraction*100:.1f}%"]
        for i, (legs, share, odds, p) in enumerate(
                zip(self.legs, self.stake_share, self.combined_odds, self.leg_probs), 1):
            tag = f"  Bet {i}" if len(self.legs) > 1 else "  Bet"
            out.append(f"{tag}: stake {currency}{stake*share:,.2f}  @ {odds:.2f}  "
                       f"-> returns {currency}{stake*share*odds:,.2f}   (win chance {p*100:.1f}%)")
            for lg in legs:
                out.append(f"      - {lg.home} v {lg.away}: {lg.selection}  "
                           f"@ {lg.price:.2f}  [{lg.tier} {lg.prob*100:.1f}%]")
        return "\n".join(out)


def _parlay_odds(legs: list[Pick]) -> float:
    o = 1.0
    for p in legs:
        o *= p.price
    return o


def _parlay_prob(legs: list[Pick]) -> float:
    """Joint probability, assuming independence across different fixtures.

    Independence is a real assumption, not a formality. Legs in the same league
    on the same day share weather, refereeing standards and scheduling effects,
    so the true joint probability is a little below this product. A haircut is
    applied per extra leg to keep the reported number on the pessimistic side.
    """
    p = 1.0
    for lg in legs:
        p *= lg.prob
    correlation_haircut = 0.995 ** max(0, len(legs) - 1)
    return p * correlation_haircut


def _kelly_for_plan(p: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (p * (b + 1.0) - 1.0) / b)


def _enumerate_parlays(picks: list[Pick], target: float, max_legs: int,
                       min_prob: float = 0.0) -> list[tuple[list[Pick], float, float]]:
    """All leg-combinations clearing `target`, as (legs, odds, probability).

    Only the shortest qualifying combinations matter: adding a leg to an
    already-qualifying parlay can only lower the win probability, so extensions
    are pruned.
    """
    out = []
    n = len(picks)
    for k in range(1, max_legs + 1):
        for combo in itertools.combinations(range(n), k):
            legs = [picks[i] for i in combo]
            if len({lg.match_id for lg in legs}) != len(legs):
                continue
            odds = _parlay_odds(legs)
            if odds < target:
                continue
            # Prune: if any proper subset already clears the target, this
            # combination is strictly worse.
            if k > 1 and any(_parlay_odds([legs[j] for j in range(k) if j != drop]) >= target
                             for drop in range(k)):
                continue
            p = _parlay_prob(legs)
            if p < min_prob:
                continue
            out.append((legs, odds, p))
    return out


def build_plan(picks: list[Pick], *, target: float = 2.0, max_legs: int = 6,
               max_candidates: int = 16, allow_split: bool = True,
               max_split: int = 3, kelly_fraction: float = 0.25) -> Plan | None:
    """Find the plan with the best honest chance of returning `target` x stake.

    `kelly_fraction` scales the recommended bankroll exposure. Quarter-Kelly is
    the default because full Kelly is brutally volatile and assumes the
    probabilities are exactly right, which they never are.
    """
    if not picks:
        return None

    # Work with the strongest candidates only; the combinatorics explode
    # otherwise, and weak legs never belong in a high-conviction parlay.
    cands = sorted(picks, key=lambda p: (-p.prob, -p.ev))[:max_candidates]

    parlays = _enumerate_parlays(cands, target, max_legs)
    if not parlays:
        return None

    best: Plan | None = None

    # ---- single parlay (covers the SINGLE case at k == 1) ----------------
    legs, odds, p = max(parlays, key=lambda t: (t[2], t[2] * t[1]))
    kind = "SINGLE" if len(legs) == 1 else "PARLAY"
    best = Plan(kind=kind, legs=[legs], stake_share=[1.0], combined_odds=[odds],
                leg_probs=[p], p_double=p, ev=p * odds, target=target,
                kelly_bankroll_fraction=kelly_fraction * _kelly_for_plan(p, odds))

    # ---- split across disjoint parlays ----------------------------------
    # With the stake divided m ways, a single winner only doubles the whole bank
    # if its own price clears m * target. The upside is that the chances add
    # instead of multiplying.
    if allow_split:
        for m in range(2, max_split + 1):
            pool = _enumerate_parlays(cands, target * m, max_legs)
            if len(pool) < m:
                continue
            pool.sort(key=lambda t: -t[2])
            chosen: list[tuple[list[Pick], float, float]] = []
            used: set[str] = set()
            for legs_i, odds_i, p_i in pool:
                ids = {lg.match_id for lg in legs_i}
                if ids & used:
                    continue
                chosen.append((legs_i, odds_i, p_i))
                used |= ids
                if len(chosen) == m:
                    break
            if len(chosen) < m:
                continue

            probs = [c[2] for c in chosen]
            p_any = 1.0 - math.prod(1.0 - q for q in probs)
            ev = sum(q * o / m for (_, o, q) in chosen)
            if p_any > best.p_double + 1e-9:
                eff_odds = sum(o / m for (_, o, _) in chosen) / max(len(chosen), 1)
                best = Plan(
                    kind="SPLIT", legs=[c[0] for c in chosen],
                    stake_share=[1.0 / m] * m,
                    combined_odds=[c[1] for c in chosen],
                    leg_probs=probs, p_double=p_any, ev=ev, target=target,
                    kelly_bankroll_fraction=kelly_fraction * _kelly_for_plan(p_any, max(eff_odds, 1.01)),
                )

    return best


def ladder(picks: list[Pick], targets=(1.5, 2.0, 3.0, 5.0), **kw) -> list[Plan]:
    """Plans at several target multiples.

    Seeing 1.5x at 84% next to 3x at 41% makes the trade-off concrete, and makes
    it obvious that the target is a choice about risk, not a setting that makes
    money appear.
    """
    out = []
    for t in targets:
        p = build_plan(picks, target=t, **kw)
        if p is not None:
            out.append(p)
    return out


def simulate_season(plans_probs: list[float], plans_evs: list[float],
                    *, stake_fraction: float = 0.25, n_sims: int = 20000,
                    seed: int = 11) -> dict:
    """Monte-Carlo a run of match days to show the spread of outcomes.

    A 70% chance of doubling sounds excellent until you notice it means a 30%
    chance of losing the stake, and that repeating it weekly makes a losing run
    close to inevitable. This puts a number on that.
    """
    rng = np.random.default_rng(seed)
    n_days = len(plans_probs)
    if n_days == 0:
        return {}
    banks = np.ones(n_sims)
    ruined = np.zeros(n_sims, dtype=bool)
    for d in range(n_days):
        p, ev = plans_probs[d], plans_evs[d]
        payout = ev / max(p, 1e-9)   # gross return multiple when the plan lands
        stake = banks * stake_fraction
        win = rng.random(n_sims) < p
        banks = banks - stake + np.where(win, stake * payout, 0.0)
        ruined |= banks < 0.10
    return {
        "days": n_days,
        "median_bank": float(np.median(banks)),
        "mean_bank": float(np.mean(banks)),
        "p_profit": float(np.mean(banks > 1.0)),
        "p_ruin": float(np.mean(ruined)),
        "p5": float(np.percentile(banks, 5)),
        "p95": float(np.percentile(banks, 95)),
    }
