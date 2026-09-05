"""Top-level orchestration: update data, train, calibrate, predict today's card.

The live path is deliberately the same code as the backtest path. The model that
prices tomorrow's fixtures is the same class, fed by the same feature builder,
gated by the same conviction rules, and corrected by a calibrator fitted on
out-of-sample history. If the backtest says the LOCK tier hits 91%, that number
refers to this code, not to an idealised cousin of it.
"""

from __future__ import annotations

import datetime
import json
import os
import pickle

import numpy as np
import pandas as pd

from . import backtest as bt
from . import data as datamod
from .calibration import CalibratorSet
from .config import (CACHE_DIR, DEFAULT_CONVICTION_CONFIG, DEFAULT_LEAGUES,
                     DEFAULT_MODEL_CONFIG, ConvictionConfig, ModelConfig)
from .conviction import (Pick, Watch, build_watchlist, dedupe_one_per_match,
                          dedupe_watchlist_one_per_match, select_picks)
from .features import build_features
from .markets import ALL_MARKET_KEYS, derive_all, market_family
from .model import MatchPredictor

ARTIFACT_DIR = os.path.join(CACHE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
CALIB_PATH = os.path.join(ARTIFACT_DIR, "calibration.json")
META_PATH = os.path.join(ARTIFACT_DIR, "meta.json")
MATCHES_PATH = os.path.join(CACHE_DIR, "matches.parquet")
FEATURES_PATH = os.path.join(CACHE_DIR, "features.parquet")


# --------------------------------------------------------------------------
def update_data(leagues: list[str] | None = None, start_year: int = 2005,
                end_year: int = 2026, *, refresh: bool = True,
                verbose: bool = True) -> pd.DataFrame:
    """Pull results and upcoming fixtures, rebuild the feature table, cache it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if verbose:
        print("Downloading results...", flush=True)
    hist = datamod.load_matches(leagues or DEFAULT_LEAGUES, start_year, end_year,
                                refresh=refresh, verbose=verbose)
    if verbose:
        print("Downloading upcoming fixtures...", flush=True)
    fx = datamod.load_fixtures(refresh=refresh)

    allm = pd.concat([hist, fx], ignore_index=True).sort_values("kickoff").reset_index(drop=True)
    allm = allm.drop_duplicates(subset=["match_id"], keep="first").reset_index(drop=True)
    allm.to_parquet(MATCHES_PATH)

    if verbose:
        print(f"Building features for {len(allm):,} matches...", flush=True)
    feat = build_features(allm, verbose=verbose)
    feat.to_parquet(FEATURES_PATH)
    if verbose:
        print(f"Cached {len(feat):,} rows "
              f"({int(feat['played'].sum()):,} played, "
              f"{int((~feat['played']).sum()):,} upcoming)", flush=True)
    return feat


def load_features() -> pd.DataFrame:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("No cached features. Run `update` first.")
    return pd.read_parquet(FEATURES_PATH)


# --------------------------------------------------------------------------
def train(feat: pd.DataFrame | None = None,
          calib_seasons: list[str] | None = None,
          cfg: ModelConfig = DEFAULT_MODEL_CONFIG,
          *, verbose: bool = True) -> tuple[MatchPredictor, CalibratorSet]:
    """Fit the production model and the calibrator that gives it a conscience.

    The calibrator is built from walk-forward out-of-sample predictions over
    recent seasons, never from the model's own training fit - a model scoring
    its own training data looks perfectly calibrated and is worthless.
    """
    feat = load_features() if feat is None else feat
    played = feat[feat["played"] == True]  # noqa: E712

    if calib_seasons is None:
        seasons = sorted(played["season"].unique())
        calib_seasons = seasons[-6:]

    if verbose:
        print(f"Generating out-of-sample predictions for calibration "
              f"({', '.join(calib_seasons)})...", flush=True)
    oos = bt.run_walkforward(feat, calib_seasons, cfg, verbose=verbose)

    if verbose:
        print("Fitting calibration curves...", flush=True)
    cal = CalibratorSet()
    for fam, g in oos.groupby("family"):
        y = g["outcome"].to_numpy(float)
        ok = np.isfinite(y)
        cal.fit_family(fam, g["p_raw"].to_numpy(float)[ok], y[ok],
                       g["p_market"].to_numpy(float)[ok])

    if verbose:
        print(f"Fitting production model on {len(played):,} matches...", flush=True)
    model = MatchPredictor(cfg).fit(played, verbose=verbose)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)
    cal.save(CALIB_PATH)
    with open(META_PATH, "w") as fh:
        json.dump({
            "trained_at": pd.Timestamp.now().isoformat(),
            "n_train": int(len(played)),
            "calib_seasons": list(calib_seasons),
            "n_calib_rows": int(len(oos)),
            "rho": float(model.rho),
            "last_match": str(played["date"].max()),
        }, fh, indent=1)
    if verbose:
        print(f"Saved artifacts to {ARTIFACT_DIR}", flush=True)
    return model, cal


def load_artifacts() -> tuple[MatchPredictor, CalibratorSet, dict]:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(CALIB_PATH)):
        raise FileNotFoundError("No trained model. Run `train` first.")
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)
    cal = CalibratorSet.load(CALIB_PATH)
    meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    return model, cal, meta


# --------------------------------------------------------------------------
def price_fixtures(fixtures: pd.DataFrame, model: MatchPredictor,
                   cal: CalibratorSet) -> pd.DataFrame:
    """Price every market for every fixture. Long format, one row per selection."""
    if len(fixtures) == 0:
        return pd.DataFrame()

    grids = model.component_grids(fixtures)
    probs = derive_all(grids["blend"])
    comp = {c: derive_all(grids[c]) for c in bt.COMPONENTS}

    frames = []
    for key in ALL_MARKET_KEYS:
        if key not in probs:
            continue
        fam = market_family(key)
        p_raw = probs[key]
        d = {
            "match_id": fixtures["match_id"].to_numpy(),
            "date": pd.to_datetime(fixtures["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
            "div": fixtures["div"].astype(str).to_numpy(),
            "home": fixtures["home"].astype(str).to_numpy(),
            "away": fixtures["away"].astype(str).to_numpy(),
            "market": key,
            "family": fam,
            "p_raw": p_raw,
            "p_cal": cal.transform(fam, p_raw, comp["market"][key]),
        }
        for c in bt.COMPONENTS:
            d[f"p_{c}"] = comp[c][key]
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)


def _reference_prices(fixtures: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    """Prices we can read straight off the fixtures feed.

    Only 1X2 and over/under 2.5 come with a real quote. Everything else is
    marked as an estimate derived from the model's own fair price plus a typical
    recreational-book margin - useful for ranking, but you must type in the real
    Betway number before trusting the edge.
    """
    n = len(fixtures)
    out, est = {}, {}
    mapping = {"1X2_HOME": "bet_h", "1X2_DRAW": "bet_d", "1X2_AWAY": "bet_a",
               "OVER_2.5": "bet_o25", "UNDER_2.5": "bet_u25"}
    for key, col in mapping.items():
        if col in fixtures.columns:
            out[key] = fixtures[col].to_numpy(float)
        else:
            out[key] = np.full(n, np.nan)
        est[key] = False
    return out, est


def todays_picks(model: MatchPredictor, cal: CalibratorSet,
                 fixtures: pd.DataFrame,
                 betway_prices: dict[str, dict[str, float]] | None = None,
                 cfg: ConvictionConfig = DEFAULT_CONVICTION_CONFIG,
                 *, one_per_match: bool = True,
                 typical_margin: float = 0.06
                 ) -> tuple[list[Pick], pd.DataFrame, list[Watch]]:
    """Produce the conviction picks for a set of fixtures.

    `betway_prices` maps match_id -> {market_key: decimal odds}. Anything you
    supply overrides the reference price and is treated as real; anything you do
    not supply falls back to the feed where a quote exists, and otherwise to an
    estimated price, which is flagged so you know not to trust its edge.
    """
    priced = price_fixtures(fixtures, model, cal)
    if len(priced) == 0:
        return [], priced, []

    n = len(fixtures)
    index = {m: i for i, m in enumerate(fixtures["match_id"].to_numpy())}

    grids = model.component_grids(fixtures)
    probs = derive_all(grids["blend"])
    comp = {c: derive_all(grids[c]) for c in bt.COMPONENTS}

    ref, est = _reference_prices(fixtures)
    prices: dict[str, np.ndarray] = {}
    price_est: dict[str, bool] = {}
    for key in ALL_MARKET_KEYS:
        if key not in probs:
            continue
        if key in ref:
            prices[key] = ref[key].copy()
            price_est[key] = False
        else:
            # No real quote exists for this market. Deliberately left as NaN
            # rather than filled with a guess: a fabricated price yields a
            # fabricated edge, and the edge gate is the one thing standing
            # between "confident" and "profitable". These selections surface on
            # the watchlist instead, with a minimum price to look for.
            prices[key] = np.full(n, np.nan)
            price_est[key] = True

    # Overlay any real Betway prices the user has entered.
    if betway_prices:
        for mid, mk in betway_prices.items():
            # Keys beginning with "_" are human-readable scaffolding written by
            # the `prices` template command, not fixtures.
            if mid.startswith("_") or not isinstance(mk, dict):
                continue
            i = index.get(mid)
            if i is None:
                continue
            for key, odds in mk.items():
                if key.startswith("_") or key not in prices:
                    continue
                try:
                    val = float(odds)
                except (TypeError, ValueError):
                    continue  # left blank in the template, or mistyped
                if val > 1.01:
                    prices[key][i] = val

    row_cols = ["match_id", "date", "div", "home", "away",
                "season_mp_h", "season_mp_a"]
    if "kickoff_local" in fixtures.columns:
        row_cols.append("kickoff_local")
    elif "kickoff" in fixtures.columns:
        row_cols.append("kickoff")
    rows = fixtures[row_cols].reset_index(drop=True)
    keys = [k for k in ALL_MARKET_KEYS if k in probs]
    picks = select_picks(rows, probs, comp, prices, cal, cfg,
                         market_probs=comp["market"],
                         price_is_estimate=price_est,
                         market_keys=keys)
    if one_per_match:
        picks = dedupe_one_per_match(picks)

    # Everything the model is confident about but cannot price: reported with
    # the minimum Betway price that would make it a bet.
    priced_keys = {k for k in keys if np.isfinite(prices[k]).any()}
    watch = build_watchlist(rows, probs, comp, cal, cfg,
                            market_probs=comp["market"],
                            skip_keys=priced_keys, market_keys=keys)
    if one_per_match:
        seen = {p.match_id for p in picks}
        watch = dedupe_watchlist_one_per_match(
            [w for w in watch if w.match_id not in seen])
    return picks, priced, watch


# football-data.co.uk publishes kickoff times in UK local time (GMT in winter,
# BST in summer). The machine running this may be in any timezone, so the two
# must be reconciled explicitly rather than compared as naive timestamps.
FEED_TIMEZONE = "Europe/London"


def kickoff_utc(df: pd.DataFrame) -> pd.Series:
    """Kickoff times as timezone-aware UTC."""
    k = pd.to_datetime(df["kickoff"])
    if getattr(k.dt, "tz", None) is not None:
        return k.dt.tz_convert("UTC")
    return (k.dt.tz_localize(FEED_TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
             .dt.tz_convert("UTC"))


def upcoming(feat: pd.DataFrame, days: int = 3,
             min_lead_minutes: int = 10) -> pd.DataFrame:
    """Fixtures that have not kicked off yet and start within `days` days.

    Two things this has to get right, both of which are easy to get wrong and
    silently produce a card full of matches you cannot bet on:

    1. TIME OF DAY, not date. Filtering from midnight leaves a match that
       kicked off at 15:00 still showing as "upcoming" at 19:00. The comparison
       is against the current instant.
    2. TIMEZONE. The feed's kickoff times are UK local; the machine may not be.
       For a user two hours ahead, a naive comparison wrongly keeps matches that
       started two hours ago and drops ones about to start.

    `min_lead_minutes` additionally drops anything kicking off within the next
    few minutes - by the time you have looked up a price and placed the bet, the
    match has started.
    """
    up = feat[(feat["played"] == False)].copy()  # noqa: E712
    if len(up) == 0:
        return up.reset_index(drop=True)

    now = pd.Timestamp.now(tz="UTC")
    ku = kickoff_utc(up)
    keep = (ku >= now + pd.Timedelta(minutes=min_lead_minutes)) & \
           (ku <= now + pd.Timedelta(days=days)) & ku.notna()

    up = up[keep].copy()
    up["kickoff_utc"] = ku[keep]
    # A local-time label so the printed card matches the user's own clock.
    local_tz = datetime.datetime.now().astimezone().tzinfo
    up["kickoff_local"] = up["kickoff_utc"].dt.tz_convert(local_tz)
    return up.sort_values("kickoff_utc").reset_index(drop=True)
