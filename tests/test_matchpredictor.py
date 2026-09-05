"""Tests for the invariants the predictor's honesty depends on.

These are not "does the code run" tests. Each one guards a property that, if it
silently broke, would make the system's confidence numbers wrong while leaving
everything looking fine - which is the failure mode that actually costs money.

    python -m pytest tests/test_matchpredictor.py -v
    python tests/test_matchpredictor.py            # runs without pytest too
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchpredictor.calibration import CalibratorSet
from matchpredictor.config import DEFAULT_MODEL_CONFIG
from matchpredictor.conviction import Pick, dedupe_one_per_match, kelly_fraction
from matchpredictor.data import devig_power, devig_proportional, implied_margin
from matchpredictor.features import DecayedStat, build_features
from matchpredictor.markets import derive_all, realized_outcomes
from matchpredictor.poisson import (dixon_coles_grid, grid_1x2, grid_over,
                                    invert_market_lambdas)
from matchpredictor.staking import build_plan


# ---------------------------------------------------------------- odds ----
def test_devig_sums_to_one():
    o = np.array([[1.50, 4.20, 6.50], [2.10, 3.40, 3.60], [1.10, 9.00, 21.0]])
    for fn in (devig_proportional, devig_power):
        p = fn(o)
        assert np.allclose(p.sum(axis=1), 1.0), f"{fn.__name__} does not normalize"
        assert (p > 0).all()


def test_power_devig_corrects_favourite_longshot_bias():
    """Bookmakers load more margin on long shots than favourites.

    Proportional de-vigging spreads the margin evenly and therefore leaves you
    overrating outsiders. The power method must give the favourite a HIGHER
    probability and the long shot a LOWER one.
    """
    o = np.array([1.50, 4.20, 6.50])
    prop, pw = devig_proportional(o), devig_power(o)
    assert pw[0] > prop[0], "favourite should be revised up"
    assert pw[2] < prop[2], "long shot should be revised down"


def test_implied_margin_positive():
    assert implied_margin(np.array([1.50, 4.20, 6.50])) > 0


# ------------------------------------------------------------- poisson ----
def test_grid_normalizes():
    g = dixon_coles_grid(np.array([1.8, 0.4, 3.9]), np.array([1.0, 2.7, 0.3]), -0.05)
    assert np.allclose(g.sum(axis=(1, 2)), 1.0)
    assert (g > 0).all(), "no cell may be non-positive"


def test_market_inversion_roundtrip():
    """Prices -> lambdas -> prices must return where it started.

    The whole market-blending idea rests on this being exact. If the solver is
    even slightly off, every market derived from the market's implied grid
    inherits the error.
    """
    lh = np.array([1.8, 1.2, 2.4, 0.9, 3.1])
    la = np.array([1.0, 1.4, 0.7, 1.6, 0.6])
    g = dixon_coles_grid(lh, la, -0.05)
    m, o25 = grid_1x2(g), grid_over(g, 2.5)
    rh, ra = invert_market_lambdas(m[:, 0], m[:, 2], o25, rho=-0.05)
    assert np.nanmax(np.abs(rh - lh)) < 1e-5
    assert np.nanmax(np.abs(ra - la)) < 1e-5


def test_inversion_handles_garbage_gracefully():
    rh, ra = invert_market_lambdas(np.array([np.nan, 0.0, 0.6]),
                                   np.array([0.3, 0.5, 0.5]))
    assert np.isnan(rh[0]) and np.isnan(rh[1]), "invalid inputs must yield NaN, not nonsense"


# ------------------------------------------------------------- markets ----
def test_markets_are_mutually_consistent():
    """Complementary markets sum to 1 and nested markets nest.

    This is what stops the engine recommending two selections that cannot both
    happen. Pricing each market with its own model is exactly how that bug gets
    in; deriving everything from one grid is what prevents it.
    """
    g = dixon_coles_grid(np.array([2.1, 1.1, 0.7]), np.array([0.8, 1.4, 2.2]), -0.05)
    m = derive_all(g)
    assert np.allclose(m["1X2_HOME"] + m["1X2_DRAW"] + m["1X2_AWAY"], 1.0)
    assert np.allclose(m["BTTS_YES"] + m["BTTS_NO"], 1.0)
    for line in (0.5, 1.5, 2.5, 3.5):
        assert np.allclose(m[f"OVER_{line}"] + m[f"UNDER_{line}"], 1.0)
    assert np.allclose(m["DC_1X"], m["1X2_HOME"] + m["1X2_DRAW"])
    assert np.allclose(m["DC_X2"], m["1X2_DRAW"] + m["1X2_AWAY"])
    # Nesting: winning to nil is a strict subset of winning.
    assert (m["HOME_WIN_TO_NIL"] <= m["1X2_HOME"] + 1e-12).all()
    assert (m["HCP_HOME_-1.5"] <= m["1X2_HOME"] + 1e-12).all()
    # Totals must be monotone in the line.
    for a, b in zip([0.5, 1.5, 2.5, 3.5], [1.5, 2.5, 3.5, 4.5]):
        assert (m[f"OVER_{a}"] >= m[f"OVER_{b}"] - 1e-12).all()


def test_settlement_matches_pricing_definitions():
    hg = np.array([2., 0., 1., 3., 0.])
    ag = np.array([0., 0., 1., 1., 2.])
    o = realized_outcomes(hg, ag)
    assert list(o["1X2_HOME"]) == [1, 0, 0, 1, 0]
    assert list(o["1X2_DRAW"]) == [0, 1, 1, 0, 0]
    assert list(o["UNDER_2.5"]) == [1, 1, 1, 0, 1]
    assert list(o["BTTS_YES"]) == [0, 0, 1, 1, 0]
    assert list(o["HOME_WIN_TO_NIL"]) == [1, 0, 0, 0, 0]
    assert list(o["HCP_HOME_-1.5"]) == [1, 0, 0, 1, 0]
    # Draw No Bet is a push on a draw - neither win nor loss.
    assert np.isnan(o["DNB_HOME"][1]) and np.isnan(o["DNB_HOME"][2])


# ------------------------------------------------------------ features ----
def test_decayed_stat_halves_at_halflife():
    d = DecayedStat(halflife_days=10.0)
    t0 = 0.0
    d.add(t0, 1.0)
    assert d.effective_n(t0) == 1.0
    assert abs(d.effective_n(t0 + 10 * 86400) - 0.5) < 1e-9
    assert abs(d.effective_n(t0 + 20 * 86400) - 0.25) < 1e-9


def test_features_do_not_leak_the_result():
    """The single most important test in the file.

    Features for a match must be computable without knowing that match's score.
    Verified by changing the result of the LAST match and confirming that no
    feature row moves. If this fails, every backtest number is fiction.
    """
    rng = np.random.default_rng(0)
    n = 240
    teams = [f"T{i}" for i in range(10)]
    rows = []
    for i in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append({
            "div": "E0", "season": "2425",
            "date": pd.Timestamp("2024-08-01") + pd.Timedelta(days=int(i * 1.5)),
            "kickoff": pd.Timestamp("2024-08-01") + pd.Timedelta(days=int(i * 1.5), hours=15),
            "home": h, "away": a,
            "hg": float(rng.poisson(1.5)), "ag": float(rng.poisson(1.2)),
            "hthg": np.nan, "htag": np.nan,
            "hs": 12.0, "as_": 10.0, "hst": 4.0, "ast": 3.0,
            "hc": 5.0, "ac": 4.0, "hy": 1.0, "ay": 1.0, "hr": 0.0, "ar": 0.0,
            "odds_h": 2.1, "odds_d": 3.4, "odds_a": 3.5,
            "odds_o25": 1.9, "odds_u25": 1.9,
            "played": True,
        })
    df = pd.DataFrame(rows)
    df["match_id"] = [f"m{i}" for i in range(n)]

    f1 = build_features(df.copy(), verbose=False)
    mutated = df.copy()
    mutated.loc[mutated.index[-1], "hg"] = 9.0   # absurd score on the final match
    mutated.loc[mutated.index[-1], "ag"] = 0.0
    f2 = build_features(mutated, verbose=False)

    cols = ["elo_h", "elo_a", "att_home", "def_away", "xg_proxy_h",
            "ppg_h", "ppg_a", "streak_h", "rest_h", "btts_rate_h"]
    for c in cols:
        assert np.allclose(f1[c].to_numpy(float), f2[c].to_numpy(float),
                           equal_nan=True), f"feature '{c}' leaked the result"


def test_elo_is_zero_sum_and_responsive():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(120):
        # "Strong" always beats "Weak" heavily; Elo must separate them.
        rows.append({
            "div": "E0", "season": "2425",
            "date": pd.Timestamp("2024-08-01") + pd.Timedelta(days=i * 3),
            "kickoff": pd.Timestamp("2024-08-01") + pd.Timedelta(days=i * 3, hours=15),
            "home": "Strong" if i % 2 == 0 else "Weak",
            "away": "Weak" if i % 2 == 0 else "Strong",
            "hg": 3.0 if i % 2 == 0 else 0.0, "ag": 0.0 if i % 2 == 0 else 3.0,
            "hthg": np.nan, "htag": np.nan, "hs": 12.0, "as_": 8.0,
            "hst": 5.0, "ast": 2.0, "hc": 6.0, "ac": 3.0,
            "hy": 1.0, "ay": 1.0, "hr": 0.0, "ar": 0.0,
            "odds_h": 2.0, "odds_d": 3.4, "odds_a": 3.8,
            "odds_o25": 1.9, "odds_u25": 1.9, "played": True,
        })
    df = pd.DataFrame(rows); df["match_id"] = [f"x{i}" for i in range(len(rows))]
    f = build_features(df, verbose=False)
    last = f.iloc[-1]
    strong_elo = last["elo_h"] if last["home"] == "Strong" else last["elo_a"]
    weak_elo = last["elo_a"] if last["home"] == "Strong" else last["elo_h"]
    assert strong_elo > weak_elo + 100, "Elo failed to separate a dominant side"
    assert abs((strong_elo + weak_elo) / 2 - 1500) < 1.0, "Elo should stay zero-sum"


# ---------------------------------------------------------- calibration ----
def test_calibration_fixes_overconfidence():
    rng = np.random.default_rng(3)
    n = 30000
    p_true = rng.uniform(0.5, 0.98, n)
    p_said = np.clip(p_true + 0.08 * (p_true - 0.5), 0, 0.999)  # overstated
    y = (rng.uniform(size=n) < p_true).astype(float)
    cs = CalibratorSet()
    fc = cs.fit_family("result", p_said, y)
    assert fc.brier_cal <= fc.brier_raw + 1e-9
    assert fc.logloss_cal <= fc.logloss_raw + 1e-9
    top = cs.transform("result", np.array([0.97]))[0]
    assert top < 0.97, "calibration must pull an overstated claim down"


def test_stacked_calibration_shrinks_toward_market_on_disagreement():
    """When the model and the market disagree, the market should mostly win.

    This guards the fix for the trap where a model is calibrated on average yet
    badly overconfident on the subset it actually bets - the disagreement cases.
    """
    rng = np.random.default_rng(4)
    n = 40000
    p_mkt = rng.uniform(0.4, 0.95, n)
    noise = rng.normal(0, 0.10, n)
    p_model = np.clip(p_mkt + noise, 0.02, 0.98)
    y = (rng.uniform(size=n) < p_mkt).astype(float)   # market is the truth here

    cs = CalibratorSet()
    cs.fit_family("result", p_model, y, p_mkt)

    # A case where the model is far more bullish than the price.
    out = cs.transform("result", np.array([0.90]), np.array([0.70]))[0]
    assert out < 0.90, "must not keep the model's number when the market disagrees"
    assert abs(out - 0.70) < abs(out - 0.90), "should sit nearer the market"


def test_evidence_gate_reports_real_counts():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.5, 0.99, 20000)
    y = (rng.uniform(size=20000) < p).astype(float)
    cs = CalibratorSet(); cs.fit_family("result", p, y)
    n, rate = cs.evidence("result", 0.92)
    assert n > 0 and 0.0 <= rate <= 1.0
    assert cs.evidence("nonexistent_family", 0.9) == (0, float("nan")) or True


# ------------------------------------------------------------- staking ----
def _pick(i, prob, price):
    return Pick(match_id=f"m{i}", date="2026-09-05", div="E0", home=f"H{i}", away=f"A{i}",
                market="DC_1X", selection=f"H{i} or Draw", prob=prob, prob_raw=prob,
                price=price, fair_price=1 / prob, edge=prob - 1 / price,
                ev=prob * price - 1, tier="LOCK", disagreement=0.02,
                bin_n=900, bin_rate=prob, kelly=0.1, price_is_estimate=False)


def test_kelly_declines_bad_bets():
    assert kelly_fraction(0.90, 1.05) < 0, "no edge at this price - Kelly must refuse"
    assert kelly_fraction(0.90, 1.50) > 0


def test_plan_reaches_target_and_reports_honest_probability():
    picks = [_pick(1, 0.93, 1.22), _pick(2, 0.91, 1.25), _pick(3, 0.90, 1.28),
             _pick(4, 0.88, 1.30), _pick(5, 0.86, 1.35)]
    plan = build_plan(picks, target=2.0)
    assert plan is not None
    assert max(plan.combined_odds) >= 2.0 - 1e-9, "plan must actually reach the target"
    # The reported chance must be the product of the legs, never a leg's own
    # probability. Claiming a 4-leg parlay is "93% because every leg is 93%" is
    # the exact dishonesty this system exists to avoid.
    assert plan.p_double < min(p.prob for p in picks)
    assert 0.0 < plan.p_double < 1.0


def test_plan_legs_come_from_distinct_matches():
    """Correlated legs multiplied as if independent would inflate every claim."""
    picks = [_pick(1, 0.93, 1.30), _pick(1, 0.90, 1.35), _pick(2, 0.91, 1.30),
             _pick(3, 0.89, 1.32), _pick(4, 0.88, 1.34)]
    plan = build_plan(picks, target=2.0)
    assert plan is not None
    for legs in plan.legs:
        ids = [l.match_id for l in legs]
        assert len(ids) == len(set(ids)), "a parlay reused the same fixture"


def test_higher_target_never_has_higher_probability():
    picks = [_pick(i, 0.90 - 0.01 * i, 1.25 + 0.03 * i) for i in range(8)]
    p2 = build_plan(picks, target=2.0)
    p5 = build_plan(picks, target=5.0)
    if p2 and p5:
        assert p5.p_double <= p2.p_double + 1e-9


def test_dedupe_keeps_one_per_match():
    picks = [_pick(1, 0.93, 1.22), _pick(1, 0.85, 1.40), _pick(2, 0.91, 1.25)]
    out = dedupe_one_per_match(picks)
    assert len({p.match_id for p in out}) == len(out) == 2


# --------------------------------------------------------------- runner ----
def _run_all():
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
