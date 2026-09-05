"""Turn a score grid into every market Betway actually offers.

One rule: every probability in here is a different summation over the *same*
joint score distribution. That is what keeps the book internally coherent. A
model that prices each market with its own separate classifier will happily tell
you "Over 2.5 is a lock" and "Home win to nil is a lock" in the same breath;
this one cannot, because both numbers come from the same grid.
"""

from __future__ import annotations

import numpy as np

OU_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
TEAM_TOTAL_LINES = [0.5, 1.5, 2.5]
HANDICAP_LINES = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]


def _axes(mg: int):
    x = np.arange(mg + 1)[:, None]
    y = np.arange(mg + 1)[None, :]
    return x, y


def derive_all(grid: np.ndarray) -> dict[str, np.ndarray]:
    """Return {market_key: probability array of shape (n,)} for one grid batch."""
    mg = grid.shape[1] - 1
    x, y = _axes(mg)
    tot = x + y
    diff = x - y
    out: dict[str, np.ndarray] = {}

    def s(mask) -> np.ndarray:
        return (grid * mask).sum(axis=(1, 2))

    # ---- Match result -------------------------------------------------
    home, draw, away = s(x > y), s(x == y), s(x < y)
    out["1X2_HOME"] = home
    out["1X2_DRAW"] = draw
    out["1X2_AWAY"] = away

    # ---- Double chance ------------------------------------------------
    # The workhorse of high-confidence betting: you get two of the three
    # outcomes, so the probability is high even when the price is short.
    out["DC_1X"] = home + draw
    out["DC_12"] = home + away
    out["DC_X2"] = draw + away

    # ---- Draw no bet --------------------------------------------------
    # Stake returned on a draw, so the effective probability is conditional.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["DNB_HOME"] = np.where(draw < 0.999, home / (1.0 - draw), np.nan)
        out["DNB_AWAY"] = np.where(draw < 0.999, away / (1.0 - draw), np.nan)

    # ---- Total goals --------------------------------------------------
    for ln in OU_LINES:
        out[f"OVER_{ln}"] = s(tot > ln)
        out[f"UNDER_{ln}"] = s(tot < ln)

    # ---- Both teams to score -----------------------------------------
    out["BTTS_YES"] = s((x > 0) & (y > 0))
    out["BTTS_NO"] = s((x == 0) | (y == 0))

    # ---- Team totals --------------------------------------------------
    for ln in TEAM_TOTAL_LINES:
        out[f"HOME_OVER_{ln}"] = s(x > ln)
        out[f"HOME_UNDER_{ln}"] = s(x < ln)
        out[f"AWAY_OVER_{ln}"] = s(y > ln)
        out[f"AWAY_UNDER_{ln}"] = s(y < ln)

    # ---- Clean sheets and shut-outs ----------------------------------
    out["HOME_CLEAN_SHEET"] = s(y == 0)
    out["AWAY_CLEAN_SHEET"] = s(x == 0)
    out["HOME_WIN_TO_NIL"] = s((x > y) & (y == 0))
    out["AWAY_WIN_TO_NIL"] = s((y > x) & (x == 0))

    # ---- Handicaps (half lines only, so there is never a push) -------
    # Line is applied to the home team: HANDICAP_HOME_-1.5 means home must win
    # by two or more.
    for ln in HANDICAP_LINES:
        out[f"HCP_HOME_{ln:+.1f}"] = s((diff + ln) > 0)
        out[f"HCP_AWAY_{ln:+.1f}"] = s((-diff + ln) > 0)

    # ---- Combined result / goals (Betway "Result & Both Teams to Score") ----
    out["HOME_AND_OVER_2.5"] = s((x > y) & (tot > 2.5))
    out["HOME_AND_UNDER_2.5"] = s((x > y) & (tot < 2.5))
    out["AWAY_AND_OVER_2.5"] = s((x < y) & (tot > 2.5))
    out["AWAY_AND_UNDER_2.5"] = s((x < y) & (tot < 2.5))
    out["HOME_AND_BTTS_YES"] = s((x > y) & (x > 0) & (y > 0))
    out["AWAY_AND_BTTS_YES"] = s((x < y) & (x > 0) & (y > 0))
    out["1X_AND_UNDER_3.5"] = s((x >= y) & (tot < 3.5))
    out["X2_AND_UNDER_3.5"] = s((x <= y) & (tot < 3.5))

    return out


def correct_scores(grid: np.ndarray, top_n: int = 5) -> list[list[tuple[str, float]]]:
    """Most likely exact scorelines per match, highest probability first."""
    n, k, _ = grid.shape
    flat = grid.reshape(n, -1)
    idx = np.argsort(-flat, axis=1)[:, :top_n]
    out = []
    for i in range(n):
        row = []
        for j in idx[i]:
            hx, ay = divmod(int(j), k)
            row.append((f"{hx}-{ay}", float(flat[i, j])))
        out.append(row)
    return out


# Human-readable descriptions, used when printing a slip.
MARKET_LABELS: dict[str, str] = {
    "1X2_HOME": "{home} to win",
    "1X2_DRAW": "Draw",
    "1X2_AWAY": "{away} to win",
    "DC_1X": "{home} or Draw (Double Chance)",
    "DC_12": "{home} or {away} (Draw No Bet both ways)",
    "DC_X2": "Draw or {away} (Double Chance)",
    "DNB_HOME": "{home} Draw No Bet",
    "DNB_AWAY": "{away} Draw No Bet",
    "BTTS_YES": "Both teams to score - Yes",
    "BTTS_NO": "Both teams to score - No",
    "HOME_CLEAN_SHEET": "{home} to keep a clean sheet",
    "AWAY_CLEAN_SHEET": "{away} to keep a clean sheet",
    "HOME_WIN_TO_NIL": "{home} to win to nil",
    "AWAY_WIN_TO_NIL": "{away} to win to nil",
    "HOME_AND_OVER_2.5": "{home} to win & Over 2.5 goals",
    "HOME_AND_UNDER_2.5": "{home} to win & Under 2.5 goals",
    "AWAY_AND_OVER_2.5": "{away} to win & Over 2.5 goals",
    "AWAY_AND_UNDER_2.5": "{away} to win & Under 2.5 goals",
    "HOME_AND_BTTS_YES": "{home} to win & Both teams to score",
    "AWAY_AND_BTTS_YES": "{away} to win & Both teams to score",
    "1X_AND_UNDER_3.5": "{home} or Draw & Under 3.5 goals",
    "X2_AND_UNDER_3.5": "Draw or {away} & Under 3.5 goals",
}


def label_for(key: str, home: str, away: str) -> str:
    if key in MARKET_LABELS:
        return MARKET_LABELS[key].format(home=home, away=away)
    if key.startswith("OVER_"):
        return f"Over {key.split('_')[1]} goals"
    if key.startswith("UNDER_"):
        return f"Under {key.split('_')[1]} goals"
    if key.startswith("HOME_OVER_"):
        return f"{home} over {key.rsplit('_', 1)[1]} goals"
    if key.startswith("HOME_UNDER_"):
        return f"{home} under {key.rsplit('_', 1)[1]} goals"
    if key.startswith("AWAY_OVER_"):
        return f"{away} over {key.rsplit('_', 1)[1]} goals"
    if key.startswith("AWAY_UNDER_"):
        return f"{away} under {key.rsplit('_', 1)[1]} goals"
    if key.startswith("HCP_HOME_"):
        return f"{home} {key.rsplit('_', 1)[1]} handicap"
    if key.startswith("HCP_AWAY_"):
        return f"{away} {key.rsplit('_', 1)[1]} handicap"
    return key


# Market families share a calibration curve - pooling related markets gives each
# curve enough samples to be trustworthy without blurring genuinely different
# prediction problems together.
def market_family(key: str) -> str:
    if key.startswith("1X2_"):
        return "result"
    if key.startswith("DC_") or key.startswith("DNB_"):
        return "double_chance"
    if key.startswith("OVER_") or key.startswith("UNDER_"):
        return "totals"
    if key.startswith("BTTS_"):
        return "btts"
    if "OVER_" in key or "UNDER_" in key:
        return "team_totals"
    if key.startswith("HCP_"):
        return "handicap"
    if "CLEAN_SHEET" in key or "WIN_TO_NIL" in key:
        return "clean_sheet"
    return "combo"


ALL_MARKET_KEYS: list[str] = (
    ["1X2_HOME", "1X2_DRAW", "1X2_AWAY", "DC_1X", "DC_12", "DC_X2", "DNB_HOME", "DNB_AWAY"]
    + [f"OVER_{l}" for l in OU_LINES] + [f"UNDER_{l}" for l in OU_LINES]
    + ["BTTS_YES", "BTTS_NO"]
    + [f"{s}_{d}_{l}" for s in ("HOME", "AWAY") for d in ("OVER", "UNDER") for l in TEAM_TOTAL_LINES]
    + ["HOME_CLEAN_SHEET", "AWAY_CLEAN_SHEET", "HOME_WIN_TO_NIL", "AWAY_WIN_TO_NIL"]
    + [f"HCP_{s}_{l:+.1f}" for s in ("HOME", "AWAY") for l in HANDICAP_LINES]
    + ["HOME_AND_OVER_2.5", "HOME_AND_UNDER_2.5", "AWAY_AND_OVER_2.5", "AWAY_AND_UNDER_2.5",
       "HOME_AND_BTTS_YES", "AWAY_AND_BTTS_YES", "1X_AND_UNDER_3.5", "X2_AND_UNDER_3.5"]
)


def realized_outcomes(hg: np.ndarray, ag: np.ndarray,
                      max_goals: int | None = None) -> dict[str, np.ndarray]:
    """Settle every market against the actual final score.

    Implemented by feeding a degenerate "grid" - all probability mass on the
    score that actually happened - through `derive_all`. That guarantees a
    market is settled by exactly the same predicate that priced it. Writing the
    settlement logic out by hand a second time is how backtests end up scoring
    "under 2.5" with a ">=" and reporting an edge that does not exist.

    Returns {market_key: array of 1.0 (won), 0.0 (lost), or NaN (push/unknown)}.
    """
    hg = np.asarray(hg, dtype=float)
    ag = np.asarray(ag, dtype=float)
    n = hg.shape[0]
    mg = max_goals if max_goals is not None else int(
        np.nanmax([np.nanmax(hg) if n else 0, np.nanmax(ag) if n else 0, 1]))
    mg = int(min(max(mg, 1), 30))

    ok = np.isfinite(hg) & np.isfinite(ag)
    hi = np.clip(np.nan_to_num(hg, nan=0.0).astype(int), 0, mg)
    ai = np.clip(np.nan_to_num(ag, nan=0.0).astype(int), 0, mg)

    grid = np.zeros((n, mg + 1, mg + 1))
    grid[np.arange(n), hi, ai] = 1.0

    out = derive_all(grid)

    # Draw No Bet is a push on a draw: stake returned, so it is neither a win
    # nor a loss and must be excluded rather than scored as zero.
    drawn = hi == ai
    for k in ("DNB_HOME", "DNB_AWAY"):
        v = np.asarray(out[k], dtype=float)
        v[drawn] = np.nan
        out[k] = v

    for k in out:
        out[k] = np.where(ok, out[k], np.nan)
    return out
