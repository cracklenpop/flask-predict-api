"""Walk-forward backtesting - the part that decides whether any of this is real.

The protocol is deliberately strict, because a football betting backtest that is
even slightly sloppy will show a fake edge:

  * EXPANDING WINDOW. To predict season S the model is fitted only on matches
    played before season S began. It is refitted from scratch for every season.
  * FEATURES ARE ALREADY LEAK-FREE. They were produced by a single forward pass
    that emits a match's features before folding in that match's result.
  * PRE-MATCH PRICES ONLY. The market signal is a pre-match price. Closing odds
    are never used as an input, because you cannot bet at a price that does not
    exist yet.
  * CALIBRATION IS ALSO OUT-OF-SAMPLE. Season S is calibrated using only the
    out-of-sample predictions from seasons before it. Fitting the calibrator on
    the season you are scoring is the subtlest and most common way to
    manufacture a confidence level that does not survive contact with reality.
  * SETTLED AT RECREATIONAL PRICES. Bets settle at a Bet365/average pre-match
    price, the bracket Betway sits in - not at best-price-across-all-books.

What this cannot capture: Betway will restrict or close an account that wins
consistently, and prices move. Treat the ROI figures as an upper bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import CalibratorSet
from .config import (ConvictionConfig, DEFAULT_CONVICTION_CONFIG,
                     DEFAULT_MODEL_CONFIG, ModelConfig)
from .conviction import dedupe_one_per_match, select_picks
from .markets import derive_all, market_family, realized_outcomes
from .model import MatchPredictor

# Markets tracked through the backtest. Every one of these is calibrated and
# scored; the subset with real historical prices is additionally used for P&L.
BACKTEST_KEYS = [
    "1X2_HOME", "1X2_DRAW", "1X2_AWAY",
    "DC_1X", "DC_12", "DC_X2", "DNB_HOME", "DNB_AWAY",
    "OVER_1.5", "OVER_2.5", "OVER_3.5", "UNDER_1.5", "UNDER_2.5", "UNDER_3.5",
    "BTTS_YES", "BTTS_NO",
    "HOME_OVER_0.5", "AWAY_OVER_0.5", "HOME_OVER_1.5", "AWAY_OVER_1.5",
    "HOME_CLEAN_SHEET", "AWAY_CLEAN_SHEET", "HOME_WIN_TO_NIL", "AWAY_WIN_TO_NIL",
    "HCP_HOME_-1.5", "HCP_HOME_+1.5", "HCP_AWAY_-1.5", "HCP_AWAY_+1.5",
    "HOME_AND_OVER_2.5", "AWAY_AND_OVER_2.5", "1X_AND_UNDER_3.5", "X2_AND_UNDER_3.5",
]

# Which markets we actually have a historical price for, and where it lives.
PRICE_COLUMNS = {
    "1X2_HOME": "bet_h",
    "1X2_DRAW": "bet_d",
    "1X2_AWAY": "bet_a",
    "OVER_2.5": "bet_o25",
    "UNDER_2.5": "bet_u25",
}

COMPONENTS = ("market", "ratings", "gbm")


def season_start_ts(season: str) -> pd.Timestamp:
    """'2425' -> 1 July 2024, the boundary the training window must respect."""
    return pd.Timestamp(year=2000 + int(season[:2]), month=7, day=1)


def run_walkforward(feat: pd.DataFrame,
                    test_seasons: list[str],
                    cfg: ModelConfig = DEFAULT_MODEL_CONFIG,
                    *, min_train: int = 8000, verbose: bool = True) -> pd.DataFrame:
    """Produce genuinely out-of-sample probabilities for every test season.

    Returns a long-format frame: one row per (match, market).
    """
    played = feat[feat["played"] == True].copy()  # noqa: E712
    chunks: list[pd.DataFrame] = []

    for season in test_seasons:
        cut = season_start_ts(season)
        train = played[played["kickoff"] < cut]
        test = played[played["season"] == season]
        if len(train) < min_train or len(test) == 0:
            if verbose:
                print(f"  {season}: skipped (train={len(train)}, test={len(test)})", flush=True)
            continue

        if verbose:
            print(f"  {season}: train={len(train):,}  test={len(test):,}", flush=True)

        model = MatchPredictor(cfg).fit(train, verbose=verbose)
        grids = model.component_grids(test)

        probs = derive_all(grids["blend"])
        comp = {c: derive_all(grids[c]) for c in COMPONENTS}
        outs = realized_outcomes(test["hg"].to_numpy(float), test["ag"].to_numpy(float))

        n = len(test)
        base = {
            "match_id": test["match_id"].to_numpy(),
            "season": season,
            "div": test["div"].astype(str).to_numpy(),
            "date": test["date"].to_numpy(),
            "kickoff": test["kickoff"].to_numpy(),
            "home": test["home"].astype(str).to_numpy(),
            "away": test["away"].astype(str).to_numpy(),
            "season_mp_h": test["season_mp_h"].to_numpy(float),
            "season_mp_a": test["season_mp_a"].to_numpy(float),
        }

        rows = []
        for key in BACKTEST_KEYS:
            d = dict(base)
            d["market"] = key
            d["family"] = market_family(key)
            d["p_raw"] = probs[key].astype(np.float32)
            for c in COMPONENTS:
                d[f"p_{c}"] = comp[c][key].astype(np.float32)
            d["outcome"] = outs[key].astype(np.float32)
            pcol = PRICE_COLUMNS.get(key)
            d["price"] = (test[pcol].to_numpy(float).astype(np.float32) if pcol
                          else np.full(n, np.nan, dtype=np.float32))
            rows.append(pd.DataFrame(d))
        chunks.append(pd.concat(rows, ignore_index=True))

    if not chunks:
        raise RuntimeError("Walk-forward produced no predictions.")
    return pd.concat(chunks, ignore_index=True)


def calibrate_progressively(oos: pd.DataFrame, test_seasons: list[str],
                            *, min_prior: int = 5000, verbose: bool = True
                            ) -> tuple[pd.DataFrame, dict[str, CalibratorSet]]:
    """Calibrate each season using ONLY the seasons that preceded it.

    Returns the frame with a `p_cal` column, plus the calibrator used for each
    season (the final one is what ships for live prediction).
    """
    oos = oos.copy()
    oos["p_cal"] = np.nan
    per_season: dict[str, CalibratorSet] = {}
    order = [s for s in test_seasons if s in set(oos["season"])]

    for i, season in enumerate(order):
        prior = oos[oos["season"].isin(order[:i])]
        mask = oos["season"] == season
        cs = CalibratorSet()
        if len(prior) >= min_prior:
            for fam, g in prior.groupby("family"):
                ok = np.isfinite(g["outcome"].to_numpy(float))
                cs.fit_family(fam, g["p_raw"].to_numpy(float)[ok],
                              g["outcome"].to_numpy(float)[ok],
                              g["p_market"].to_numpy(float)[ok])
        per_season[season] = cs

        # Seasons with no usable prior stay uncalibrated and are excluded from
        # P&L below rather than being scored with an untrained calibrator.
        sub = oos.loc[mask]
        if cs.families:
            vals = np.full(len(sub), np.nan)
            fams = sub["family"].to_numpy()
            praw = sub["p_raw"].to_numpy(float)
            pmkt = sub["p_market"].to_numpy(float)
            for fam in np.unique(fams):
                fm = fams == fam
                vals[fm] = cs.transform(fam, praw[fm], pmkt[fm])
            oos.loc[mask, "p_cal"] = vals
        if verbose:
            print(f"  calibrated {season} on {len(prior):,} prior rows "
                  f"({len(cs.families)} families)", flush=True)

    return oos, per_season


def evaluate_picks(oos: pd.DataFrame,
                   calibrators: dict[str, CalibratorSet],
                   cfg: ConvictionConfig = DEFAULT_CONVICTION_CONFIG,
                   *, one_per_match: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Run the conviction gates over the out-of-sample predictions and settle."""
    results = []
    for season, g in oos.groupby("season"):
        cs = calibrators.get(season)
        if cs is None or not cs.families:
            continue
        g = g[np.isfinite(g["price"].to_numpy(float))]
        if len(g) == 0:
            continue

        # Rebuild the per-match wide structure the conviction engine expects.
        keys = sorted(set(g["market"]))
        first = g[g["market"] == keys[0]]
        rows = first[["match_id", "date", "div", "home", "away",
                      "season_mp_h", "season_mp_a"]].reset_index(drop=True)
        index = {m: i for i, m in enumerate(rows["match_id"])}
        n = len(rows)

        probs_raw, prices, comp_probs = {}, {}, {c: {} for c in COMPONENTS}
        market_probs, outcome_map = {}, {}
        for key in keys:
            sub = g[g["market"] == key]
            pos = sub["match_id"].map(index).to_numpy()
            ok = np.isfinite(pos.astype(float))
            pos = pos[ok].astype(int)
            arr = np.full(n, np.nan)
            arr[pos] = sub["p_raw"].to_numpy(float)[ok]
            probs_raw[key] = arr
            pr = np.full(n, np.nan); pr[pos] = sub["price"].to_numpy(float)[ok]
            prices[key] = pr
            oc = np.full(n, np.nan); oc[pos] = sub["outcome"].to_numpy(float)[ok]
            outcome_map[key] = oc
            for c in COMPONENTS:
                ca = np.full(n, np.nan); ca[pos] = sub[f"p_{c}"].to_numpy(float)[ok]
                comp_probs[c][key] = ca
            market_probs[key] = comp_probs["market"][key]

        picks = select_picks(rows, probs_raw, comp_probs, prices, cs, cfg,
                             market_probs=market_probs, market_keys=keys)
        if one_per_match:
            picks = dedupe_one_per_match(picks)

        for p in picks:
            i = index.get(p.match_id)
            if i is None:
                continue
            y = outcome_map[p.market][i]
            if not np.isfinite(y):
                continue
            d = p.to_dict()
            d["season"] = season
            d["won"] = float(y)
            d["profit"] = float(y) * (p.price - 1.0) - (1.0 - float(y))
            results.append(d)

        if verbose:
            print(f"  {season}: {len(picks):,} picks passed the gates", flush=True)

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def tier_report(bets: pd.DataFrame) -> str:
    if len(bets) == 0:
        return "  No bets cleared the conviction gates."
    lines = ["  tier      bets      claimed   actual     gap    avg price     ROI      profit"]
    lines.append("  " + "-" * 76)
    for tier in ("LOCK", "STRONG", "LEAN"):
        b = bets[bets["tier"] == tier]
        if len(b) == 0:
            continue
        lines.append(
            f"  {tier:<8} {len(b):>6,}    {b['prob'].mean()*100:6.2f}%  "
            f"{b['won'].mean()*100:6.2f}%  {(b['won'].mean()-b['prob'].mean())*100:+6.2f}  "
            f"{b['price'].mean():8.2f}   {b['profit'].mean()*100:+6.2f}%  {b['profit'].sum():+9.1f}u")
    b = bets
    lines.append("  " + "-" * 76)
    lines.append(
        f"  {'ALL':<8} {len(b):>6,}    {b['prob'].mean()*100:6.2f}%  "
        f"{b['won'].mean()*100:6.2f}%  {(b['won'].mean()-b['prob'].mean())*100:+6.2f}  "
        f"{b['price'].mean():8.2f}   {b['profit'].mean()*100:+6.2f}%  {b['profit'].sum():+9.1f}u")
    return "\n".join(lines)


def season_report(bets: pd.DataFrame) -> str:
    if len(bets) == 0:
        return "  (none)"
    lines = ["  season    bets    hit rate      ROI       profit"]
    lines.append("  " + "-" * 50)
    for s, b in bets.groupby("season"):
        lines.append(f"  {s:<8} {len(b):>5,}    {b['won'].mean()*100:6.2f}%   "
                     f"{b['profit'].mean()*100:+6.2f}%   {b['profit'].sum():+8.1f}u")
    return "\n".join(lines)


def market_report(bets: pd.DataFrame) -> str:
    if len(bets) == 0:
        return "  (none)"
    lines = ["  market            bets   claimed   actual      ROI"]
    lines.append("  " + "-" * 52)
    for m, b in bets.groupby("market"):
        lines.append(f"  {m:<16} {len(b):>5,}   {b['prob'].mean()*100:6.2f}%  "
                     f"{b['won'].mean()*100:6.2f}%   {b['profit'].mean()*100:+6.2f}%")
    return "\n".join(lines)
