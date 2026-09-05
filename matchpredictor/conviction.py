"""The conviction engine: the gate that keeps the bot quiet.

The brief for this project was a bot that only speaks when it is as close to
certain as the evidence allows - the machine version of the feeling a seasoned
watcher gets when they just know a result is coming. The difference is that a
human's certainty is unaudited, and this one is not.

A selection is only emitted if it survives EVERY gate below. Any single failure
and the pick is discarded silently, no matter how attractive it looked:

  1. CALIBRATED PROBABILITY - not the raw model number, the one corrected by
     what actually happened the last time the model said this.
  2. HISTORICAL EVIDENCE    - the probability band this pick sits in must have a
     real track record: enough past samples, and an actual hit rate that lived
     up to the claim.
  3. MODEL AGREEMENT        - the market, the ratings model and the learned
     model must broadly agree. Confidence built on one component shouting over
     the other two is exactly the confidence that gets punished.
  4. EDGE                   - the price must be better than fair. Being right at
     a bad price is a losing strategy; over enough bets it is indistinguishable
     from being wrong.
  5. PRICE SANITY           - no unbettably short prices, no long-shot lottery
     tickets dressed up as value.

What comes out the other side is small. That is the point. On most match days
the honest answer is "there are three of these, not thirty".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .calibration import CalibratorSet
from .config import ConvictionConfig, DEFAULT_CONVICTION_CONFIG
from .markets import label_for, market_family


@dataclass
class Pick:
    """One selection the engine is prepared to stand behind."""

    match_id: str
    date: str
    div: str
    home: str
    away: str
    market: str
    selection: str
    prob: float              # calibrated probability
    prob_raw: float          # before calibration
    price: float             # the price you can actually get
    fair_price: float        # 1 / calibrated probability
    edge: float              # calibrated prob - price-implied prob
    ev: float                # expected profit per 1 staked
    tier: str                # LOCK / STRONG / LEAN
    disagreement: float      # max gap between model components
    bin_n: int               # past samples in this probability band
    bin_rate: float          # what actually happened in that band
    kelly: float             # full-Kelly bankroll fraction
    price_is_estimate: bool  # True when no real book price was supplied

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        ev_pct = self.ev * 100
        return (f"[{self.tier}] {self.home} v {self.away} ({self.div})\n"
                f"    {self.selection}\n"
                f"    model {self.prob*100:.1f}%   price {self.price:.2f} "
                f"(fair {self.fair_price:.2f})   edge {self.edge*100:+.1f}pts   EV {ev_pct:+.1f}%\n"
                f"    track record in this band: {self.bin_rate*100:.1f}% over "
                f"{self.bin_n:,} past bets   |   model spread {self.disagreement*100:.1f}pts")


def kelly_fraction(p: float, price: float) -> float:
    """Full-Kelly stake as a fraction of bankroll. Negative means do not bet."""
    b = price - 1.0
    if b <= 0:
        return 0.0
    return (p * (b + 1.0) - 1.0) / b


def _tier_for(p: float, cfg: ConvictionConfig) -> str | None:
    for name in ("LOCK", "STRONG", "LEAN"):
        if p >= cfg.tiers[name]:
            return name
    return None


def select_picks(rows: pd.DataFrame,
                 probs_raw: dict[str, np.ndarray],
                 component_probs: dict[str, dict[str, np.ndarray]],
                 prices: dict[str, np.ndarray],
                 calibrators: CalibratorSet,
                 cfg: ConvictionConfig = DEFAULT_CONVICTION_CONFIG,
                 *,
                 market_probs: dict[str, np.ndarray] | None = None,
                 price_is_estimate: dict[str, bool] | None = None,
                 market_keys: list[str] | None = None) -> list[Pick]:
    """Run every candidate selection through every gate.

    `rows`             - match metadata, one row per match
    `probs_raw`        - {market_key: (n,) raw model probability}
    `component_probs`  - {component: {market_key: (n,)}} for the agreement check
    `prices`           - {market_key: (n,) decimal price}, NaN where unavailable
    """
    price_is_estimate = price_is_estimate or {}
    keys = market_keys if market_keys is not None else list(probs_raw)
    n = len(rows)
    if n == 0:
        return []

    meta_id = rows["match_id"].to_numpy()
    meta_date = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    meta_div = rows["div"].astype(str).to_numpy()
    meta_home = rows["home"].astype(str).to_numpy()
    meta_away = rows["away"].astype(str).to_numpy()
    season_mp_h = rows["season_mp_h"].to_numpy(float) if "season_mp_h" in rows else np.full(n, 99.0)
    season_mp_a = rows["season_mp_a"].to_numpy(float) if "season_mp_a" in rows else np.full(n, 99.0)

    picks: list[Pick] = []

    for key in keys:
        if key not in probs_raw:
            continue
        fam = market_family(key)
        p_raw = np.asarray(probs_raw[key], dtype=float)
        # The market probability is handed to the calibrator, not just used for
        # comparison: it is what stops the model being overconfident precisely
        # where it disagrees with the price.
        p_mkt = None
        if market_probs and key in market_probs:
            p_mkt = np.asarray(market_probs[key], dtype=float)
        elif "market" in component_probs and key in component_probs["market"]:
            p_mkt = np.asarray(component_probs["market"][key], dtype=float)
        p_cal = calibrators.transform(fam, p_raw, p_mkt)

        price = np.asarray(prices.get(key, np.full(n, np.nan)), dtype=float)
        est = bool(price_is_estimate.get(key, False))

        # --- Gate 3 prep: how far apart are the model's own components? ---
        comps = [np.asarray(c[key], dtype=float) for c in component_probs.values() if key in c]
        if len(comps) >= 2:
            stack = np.vstack(comps)
            disagreement = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        else:
            disagreement = np.zeros(n)

        # --- Vectorized gates -------------------------------------------
        tier_floor = min(cfg.tiers.values())
        ok = np.isfinite(p_cal) & (p_cal >= tier_floor)
        ok &= np.isfinite(disagreement) & (disagreement <= cfg.max_component_disagreement)
        ok &= (season_mp_h >= cfg.min_matches_played) & (season_mp_a >= cfg.min_matches_played)

        if cfg.require_market_price:
            ok &= np.isfinite(price)
        ok &= ~np.isfinite(price) | ((price >= cfg.min_odds) & (price <= cfg.max_odds))

        with np.errstate(divide="ignore", invalid="ignore"):
            implied = 1.0 / price
        edge = p_cal - implied
        ev = p_cal * price - 1.0

        is_lock = p_cal >= cfg.tiers["LOCK"]
        min_edge = np.where(is_lock, cfg.min_edge_lock, cfg.min_edge)
        ok &= np.isfinite(edge) & (edge >= min_edge)

        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue

        for i in idx:
            p = float(p_cal[i])
            tier = _tier_for(p, cfg)
            if tier is None:
                continue

            # --- Gate 2: does this probability band have a track record? ---
            bin_n, bin_rate = calibrators.evidence(fam, p)
            if bin_n < cfg.min_bin_samples:
                continue
            if not np.isfinite(bin_rate) or bin_rate < p - cfg.bin_hitrate_tolerance:
                continue

            k = kelly_fraction(p, float(price[i]))
            if k <= 0:
                continue

            picks.append(Pick(
                match_id=str(meta_id[i]), date=str(meta_date[i]), div=str(meta_div[i]),
                home=str(meta_home[i]), away=str(meta_away[i]),
                market=key, selection=label_for(key, str(meta_home[i]), str(meta_away[i])),
                prob=p, prob_raw=float(p_raw[i]),
                price=float(price[i]), fair_price=float(1.0 / max(p, 1e-9)),
                edge=float(edge[i]), ev=float(ev[i]), tier=tier,
                disagreement=float(disagreement[i]),
                bin_n=int(bin_n), bin_rate=float(bin_rate),
                kelly=float(k), price_is_estimate=est,
            ))

    tier_rank = {"LOCK": 0, "STRONG": 1, "LEAN": 2}
    picks.sort(key=lambda x: (tier_rank[x.tier], -x.prob, -x.ev))
    return picks


def dedupe_one_per_match(picks: list[Pick]) -> list[Pick]:
    """Keep only the single best selection per match.

    Two selections on the same match are not independent - "home win" and "home
    or draw" and "over 1.5" all move together. Treating them as separate bets
    would badly overstate diversification and, in an accumulator, would be
    correlated legs masquerading as independent ones.
    """
    best: dict[str, Pick] = {}
    rank = {"LOCK": 0, "STRONG": 1, "LEAN": 2}
    for p in picks:
        cur = best.get(p.match_id)
        if cur is None or (rank[p.tier], -p.prob, -p.ev) < (rank[cur.tier], -cur.prob, -cur.ev):
            best[p.match_id] = p
    out = list(best.values())
    out.sort(key=lambda x: (rank[x.tier], -x.prob, -x.ev))
    return out


@dataclass
class Watch:
    """A selection the model is confident in, but has no real price for.

    Most of the 62 markets the engine prices are not quoted in the free data
    feed. Inventing a price for them would be worse than useless: a fabricated
    price produces a fabricated edge, and the edge gate would either reject
    everything or wave through nonsense.

    So they come out here instead, with the one number that actually helps -
    the minimum price at which the bet becomes worth making. Look the selection
    up on Betway; if their price is at or above `min_price`, it clears the same
    edge bar every fully-gated pick had to clear.
    """

    match_id: str
    date: str
    div: str
    home: str
    away: str
    market: str
    selection: str
    prob: float
    fair_price: float
    min_price: float          # the lowest Betway price worth taking
    tier: str
    disagreement: float
    bin_n: int
    bin_rate: float

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        return (f"[{self.tier}] {self.home} v {self.away} ({self.div})\n"
                f"    {self.selection}\n"
                f"    model {self.prob*100:.1f}%   fair {self.fair_price:.2f}   "
                f"TAKE ONLY AT {self.min_price:.2f} OR BETTER\n"
                f"    track record in this band: {self.bin_rate*100:.1f}% over "
                f"{self.bin_n:,} past bets")


def build_watchlist(rows: pd.DataFrame,
                    probs_raw: dict[str, np.ndarray],
                    component_probs: dict[str, dict[str, np.ndarray]],
                    calibrators: CalibratorSet,
                    cfg: ConvictionConfig = DEFAULT_CONVICTION_CONFIG,
                    *,
                    market_probs: dict[str, np.ndarray] | None = None,
                    skip_keys: set[str] | None = None,
                    market_keys: list[str] | None = None) -> list[Watch]:
    """Apply every gate except the price-dependent ones, and report a target price.

    Same confidence, evidence and agreement standards as `select_picks`. The
    only difference is that the edge test is deferred to you, expressed as the
    minimum price that would satisfy it.
    """
    keys = market_keys if market_keys is not None else list(probs_raw)
    skip = skip_keys or set()
    n = len(rows)
    if n == 0:
        return []

    meta_id = rows["match_id"].to_numpy()
    meta_date = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    meta_div = rows["div"].astype(str).to_numpy()
    meta_home = rows["home"].astype(str).to_numpy()
    meta_away = rows["away"].astype(str).to_numpy()
    season_mp_h = rows["season_mp_h"].to_numpy(float) if "season_mp_h" in rows else np.full(n, 99.0)
    season_mp_a = rows["season_mp_a"].to_numpy(float) if "season_mp_a" in rows else np.full(n, 99.0)

    out: list[Watch] = []
    for key in keys:
        if key in skip or key not in probs_raw:
            continue
        fam = market_family(key)
        p_mkt = None
        if market_probs and key in market_probs:
            p_mkt = np.asarray(market_probs[key], dtype=float)
        elif "market" in component_probs and key in component_probs["market"]:
            p_mkt = np.asarray(component_probs["market"][key], dtype=float)
        p_cal = calibrators.transform(fam, np.asarray(probs_raw[key], dtype=float), p_mkt)

        comps = [np.asarray(c[key], dtype=float) for c in component_probs.values() if key in c]
        disagreement = (np.vstack(comps).max(axis=0) - np.vstack(comps).min(axis=0)
                        if len(comps) >= 2 else np.zeros(n))

        floor = min(cfg.tiers.values())
        ok = (np.isfinite(p_cal) & (p_cal >= floor)
              & np.isfinite(disagreement) & (disagreement <= cfg.max_component_disagreement)
              & (season_mp_h >= cfg.min_matches_played)
              & (season_mp_a >= cfg.min_matches_played))

        for i in np.flatnonzero(ok):
            p = float(p_cal[i])
            tier = _tier_for(p, cfg)
            if tier is None:
                continue
            bin_n, bin_rate = calibrators.evidence(fam, p)
            if bin_n < cfg.min_bin_samples:
                continue
            if not np.isfinite(bin_rate) or bin_rate < p - cfg.bin_hitrate_tolerance:
                continue

            edge_req = cfg.min_edge_lock if p >= cfg.tiers["LOCK"] else cfg.min_edge
            denom = p - edge_req
            if denom <= 0:
                continue
            min_price = 1.0 / denom
            if min_price > cfg.max_odds:
                continue

            out.append(Watch(
                match_id=str(meta_id[i]), date=str(meta_date[i]), div=str(meta_div[i]),
                home=str(meta_home[i]), away=str(meta_away[i]), market=key,
                selection=label_for(key, str(meta_home[i]), str(meta_away[i])),
                prob=p, fair_price=1.0 / max(p, 1e-9), min_price=min_price,
                tier=tier, disagreement=float(disagreement[i]),
                bin_n=int(bin_n), bin_rate=float(bin_rate),
            ))

    rank = {"LOCK": 0, "STRONG": 1, "LEAN": 2}
    out.sort(key=lambda w: (rank[w.tier], -w.prob))
    return out
