"""Download, cache and normalize historical match data.

Source: football-data.co.uk. One CSV per league per season, free and keyless.
Column availability drifts a lot across seasons (Betfair Exchange columns only
appear recently, Ladbrokes disappears, and so on), so every accessor here is
written to degrade gracefully rather than explode on a missing column.
"""

from __future__ import annotations

import io
import os
import time
import warnings

import numpy as np
import pandas as pd
import requests

from .config import BASE_URL, CACHE_DIR, DEFAULT_LEAGUES, season_codes

warnings.filterwarnings("ignore", category=FutureWarning)

# Three distinct families of price, kept strictly separate. Conflating them is
# the most common way a football betting backtest quietly cheats.
#
#   SIGNAL  - the sharpest PRE-MATCH price, used as a model input. Pinnacle
#             first: a low-margin, high-limit book whose line is the best free
#             estimate of true probability that exists.
#   BET     - the price you could actually have taken PRE-MATCH at a soft,
#             recreational book. This is the Betway stand-in, and the only price
#             the backtest is allowed to settle bets at.
#   CLOSING - the price at kickoff. Known only AFTER you would have bet, so it
#             is never used as a feature. Kept for line-movement diagnostics.
#
# Note that no "C" (closing) column appears in the signal or bet lists.
_H_ODDS_PRIORITY = ["PSH", "AvgH", "BWH", "B365H", "MaxH"]
_D_ODDS_PRIORITY = ["PSD", "AvgD", "BWD", "B365D", "MaxD"]
_A_ODDS_PRIORITY = ["PSA", "AvgA", "BWA", "B365A", "MaxA"]

_O25_PRIORITY = ["P>2.5", "Avg>2.5", "B365>2.5", "Max>2.5"]
_U25_PRIORITY = ["P<2.5", "Avg<2.5", "B365<2.5", "Max<2.5"]

# The price a recreational customer actually gets. Betway sits in this bracket -
# comparable to Bet365 and the market average, and clearly below the best-price
# aggregator, so this is the conservative, realistic choice.
_BET_H = ["B365H", "BWH", "AvgH"]
_BET_D = ["B365D", "BWD", "AvgD"]
_BET_A = ["B365A", "BWA", "AvgA"]
_BET_O25 = ["B365>2.5", "Avg>2.5"]
_BET_U25 = ["B365<2.5", "Avg<2.5"]

# Closing prices - diagnostics only, never a feature.
_CLOSE_H = ["PSCH", "AvgCH", "B365CH"]
_CLOSE_D = ["PSCD", "AvgCD", "B365CD"]
_CLOSE_A = ["PSCA", "AvgCA", "B365CA"]

# Best price across all books. Optimistic (it assumes accounts everywhere), so
# it is reported for comparison but is not the default settlement price.
_BEST_H = ["MaxH", "B365H", "AvgH"]
_BEST_D = ["MaxD", "B365D", "AvgD"]
_BEST_A = ["MaxA", "B365A", "AvgA"]

CORE_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
]


def _cache_path(season: str, div: str) -> str:
    d = os.path.join(CACHE_DIR, season)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{div}.csv")


def fetch_season(season: str, div: str, *, refresh: bool = False,
                 max_age_hours: float = 12.0, timeout: int = 30) -> pd.DataFrame | None:
    """Fetch one league-season, using a local cache.

    The current season's file changes as matches are played, so the cache has a
    short TTL. Completed seasons never change and are cached forever.
    """
    path = _cache_path(season, div)
    if os.path.exists(path) and not refresh:
        age_h = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_h < max_age_hours or os.path.getsize(path) > 0 and _is_completed_season(season):
            try:
                return _read_csv(path)
            except Exception:
                pass

    url = f"{BASE_URL}/{season}/{div}.csv"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or len(resp.content) < 200:
            return _read_csv(path) if os.path.exists(path) else None
        with open(path, "wb") as fh:
            fh.write(resp.content)
        return _read_csv(io.BytesIO(resp.content))
    except Exception:
        return _read_csv(path) if os.path.exists(path) else None


def _is_completed_season(season: str) -> bool:
    """A season code like '2425' is complete once we are past the following August."""
    start_yy = int(season[:2])
    start_year = 2000 + start_yy
    now = pd.Timestamp.now()
    return now > pd.Timestamp(year=start_year + 1, month=8, day=1)


def _read_csv(src) -> pd.DataFrame:
    """Read a football-data CSV defensively.

    These files carry stray trailing commas, occasional malformed rows, a
    UTF-8 BOM on some seasons and latin-1 accented team names on others. Try
    utf-8-sig first (which strips the BOM), then fall back to latin-1, and
    scrub any BOM remnant out of the header either way.
    """
    if hasattr(src, "seek"):
        src.seek(0)
    try:
        df = pd.read_csv(src, encoding="utf-8-sig", on_bad_lines="skip")
    except (UnicodeDecodeError, LookupError):
        if hasattr(src, "seek"):
            src.seek(0)
        df = pd.read_csv(src, encoding="latin-1", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    """football-data mixes dd/mm/yy and dd/mm/yyyy, sometimes within one file."""
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    missing = d.isna()
    if missing.any():
        d2 = pd.to_datetime(s[missing], format="%d/%m/%y", errors="coerce")
        d.loc[missing] = d2
    missing = d.isna()
    if missing.any():
        d3 = pd.to_datetime(s[missing], errors="coerce", dayfirst=True)
        d.loc[missing] = d3
    return d


def _first_available(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Coalesce across a priority list of columns, taking the first valid value."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in candidates:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        # Odds below 1.01 are data errors, not prices.
        vals = vals.where(vals > 1.01)
        out = out.fillna(vals)
    return out


def normalize(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Reduce a raw league-season file to the tidy schema the rest of the code uses."""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Date" not in df.columns or "HomeTeam" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["div"] = df["Div"].astype(str).str.strip() if "Div" in df else np.nan
    out["season"] = season
    out["date"] = _parse_dates(df["Date"])

    # Kickoff time is only present in recent seasons; default to a mid-afternoon
    # kickoff so that same-day ordering is at least stable.
    if "Time" in df.columns:
        t = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.time
        out["kickoff"] = [
            pd.Timestamp.combine(d, tt) if pd.notna(d) and tt is not None and pd.notna(tt)
            else (d + pd.Timedelta(hours=15) if pd.notna(d) else pd.NaT)
            for d, tt in zip(out["date"], t)
        ]
    else:
        out["kickoff"] = out["date"] + pd.Timedelta(hours=15)

    out["home"] = df["HomeTeam"].astype(str).str.strip()
    out["away"] = df["AwayTeam"].astype(str).str.strip()

    for src, dst in [("FTHG", "hg"), ("FTAG", "ag"), ("HTHG", "hthg"), ("HTAG", "htag"),
                     ("HS", "hs"), ("AS", "as_"), ("HST", "hst"), ("AST", "ast"),
                     ("HC", "hc"), ("AC", "ac"), ("HY", "hy"), ("AY", "ay"),
                     ("HR", "hr"), ("AR", "ar")]:
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan

    # Sharp market prices (used as a model input).
    out["odds_h"] = _first_available(df, _H_ODDS_PRIORITY)
    out["odds_d"] = _first_available(df, _D_ODDS_PRIORITY)
    out["odds_a"] = _first_available(df, _A_ODDS_PRIORITY)

    # Best available prices (used to judge realistic returns in the backtest).
    out["best_h"] = _first_available(df, _BEST_H)
    out["best_d"] = _first_available(df, _BEST_D)
    out["best_a"] = _first_available(df, _BEST_A)
    for c, src in [("best_h", "odds_h"), ("best_d", "odds_d"), ("best_a", "odds_a")]:
        out[c] = out[c].fillna(out[src])

    out["odds_o25"] = _first_available(df, _O25_PRIORITY)
    out["odds_u25"] = _first_available(df, _U25_PRIORITY)

    # Settlement prices for the backtest (recreational-book, pre-match).
    out["bet_h"] = _first_available(df, _BET_H)
    out["bet_d"] = _first_available(df, _BET_D)
    out["bet_a"] = _first_available(df, _BET_A)
    out["bet_o25"] = _first_available(df, _BET_O25)
    out["bet_u25"] = _first_available(df, _BET_U25)

    # Closing prices, diagnostics only.
    out["close_h"] = _first_available(df, _CLOSE_H)
    out["close_d"] = _first_available(df, _CLOSE_D)
    out["close_a"] = _first_available(df, _CLOSE_A)

    # Asian handicap line and prices, where present.
    out["ah_line"] = _first_available(df, ["AHCh", "AHh", "AHCh"]).where(lambda s: s.abs() < 6)
    if "AHCh" in df.columns:
        out["ah_line"] = pd.to_numeric(df["AHCh"], errors="coerce")
    elif "AHh" in df.columns:
        out["ah_line"] = pd.to_numeric(df["AHh"], errors="coerce")
    else:
        out["ah_line"] = np.nan
    out["ah_home"] = _first_available(df, ["PCAHH", "PAHH", "AvgCAHH", "AvgAHH", "B365CAHH", "B365AHH"])
    out["ah_away"] = _first_available(df, ["PCAHA", "PAHA", "AvgCAHA", "AvgAHA", "B365CAHA", "B365AHA"])

    # Drop rows that are not real, completed matches.
    out = out[out["date"].notna() & out["home"].notna() & out["away"].notna()]
    out = out[(out["home"] != "") & (out["away"] != "") & (out["home"] != "nan")]
    out["played"] = out["hg"].notna() & out["ag"].notna()

    out["result"] = np.where(out["hg"] > out["ag"], "H",
                      np.where(out["hg"] < out["ag"], "A", "D"))
    out.loc[~out["played"], "result"] = np.nan

    out["total_goals"] = out["hg"] + out["ag"]
    out["match_id"] = (out["div"].astype(str) + "|" + out["season"].astype(str) + "|"
                       + out["date"].dt.strftime("%Y%m%d") + "|"
                       + out["home"] + "|" + out["away"])
    return out.reset_index(drop=True)


def load_matches(leagues: list[str] | None = None,
                 start_year: int = 2005, end_year: int = 2026,
                 *, refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Load and concatenate every requested league-season into one match table."""
    leagues = leagues or DEFAULT_LEAGUES
    seasons = season_codes(start_year, end_year)
    frames = []
    for season in seasons:
        for div in leagues:
            raw = fetch_season(season, div, refresh=refresh)
            if raw is None:
                continue
            norm = normalize(raw, season)
            if len(norm):
                frames.append(norm)
        if verbose:
            print(f"  season {season}: {sum(len(f) for f in frames):>6d} matches cumulative", flush=True)

    if not frames:
        raise RuntimeError("No data could be loaded. Check network access to football-data.co.uk.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["kickoff", "div", "home"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Odds helpers
# --------------------------------------------------------------------------
def devig_proportional(odds: np.ndarray) -> np.ndarray:
    """Remove the bookmaker margin by simple proportional scaling.

    Fast and stable, but it spreads the margin evenly, which slightly overstates
    long-shot probabilities. `devig_power` corrects for that.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / odds
    total = np.nansum(raw, axis=-1, keepdims=True)
    return raw / total


def devig_power(odds: np.ndarray, tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    """Power (Shin-like) de-vigging: solve for k where sum(p_i^k) == 1.

    Bookmakers load more margin onto long shots than favourites. Proportional
    de-vigging ignores that and leaves you systematically overvaluing outsiders.
    This solves for the exponent that removes the margin in a way that respects
    the favourite-longshot bias, which matters a lot when the whole system is
    built on short-priced high-confidence picks.
    """
    odds = np.asarray(odds, dtype=float)
    single = odds.ndim == 1
    if single:
        odds = odds[None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / odds
    out = np.full_like(raw, np.nan)

    for i in range(raw.shape[0]):
        r = raw[i]
        if not np.all(np.isfinite(r)) or r.sum() <= 0:
            continue
        lo, hi = 0.2, 3.0
        k = 1.0
        for _ in range(max_iter):
            k = 0.5 * (lo + hi)
            s = np.power(r, k).sum()
            if abs(s - 1.0) < tol:
                break
            # sum(p^k) decreases as k rises (since each p < 1)
            if s > 1.0:
                lo = k
            else:
                hi = k
        p = np.power(r, k)
        out[i] = p / p.sum()

    return out[0] if single else out


def implied_margin(odds: np.ndarray) -> np.ndarray:
    """Bookmaker overround, e.g. 0.05 means a 5% margin baked into the prices."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nansum(1.0 / np.asarray(odds, dtype=float), axis=-1) - 1.0


# --------------------------------------------------------------------------
# Upcoming fixtures (the live prediction path)
# --------------------------------------------------------------------------
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def load_fixtures(*, refresh: bool = True, timeout: int = 30) -> pd.DataFrame:
    """Fetch the upcoming-fixtures feed, with pre-match prices attached.

    This is what the engine actually predicts on. The file carries roughly the
    next week of matches across every covered division, together with opening
    prices from several books - including Max and Avg, which is what we compare
    a Betway price against to decide whether Betway is offering value.
    """
    path = os.path.join(CACHE_DIR, "fixtures.csv")
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = None
    if refresh:
        try:
            resp = requests.get(FIXTURES_URL, timeout=timeout)
            if resp.status_code == 200 and len(resp.content) > 200:
                with open(path, "wb") as fh:
                    fh.write(resp.content)
                raw = _read_csv(io.BytesIO(resp.content))
        except Exception:
            raw = None
    if raw is None and os.path.exists(path):
        raw = _read_csv(path)
    if raw is None:
        return pd.DataFrame()

    # The fixtures feed has no season column; infer it from the kickoff date.
    now = pd.Timestamp.now()
    yr = now.year if now.month >= 7 else now.year - 1
    season = f"{yr % 100:02d}{(yr + 1) % 100:02d}"

    out = normalize(raw, season)
    if len(out) == 0:
        return out
    # Anything without a result is a genuine upcoming fixture.
    out = out[~out["played"]].copy()
    out["played"] = False
    for c in ("hg", "ag", "result", "total_goals"):
        out[c] = np.nan
    return out.reset_index(drop=True)
