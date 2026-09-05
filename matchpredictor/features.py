"""Leak-free feature engineering.

Everything here runs as a single forward pass in kickoff order. For each match
we emit the feature row FIRST, using only state accumulated from earlier
matches, and only THEN fold that match's result into the state. That ordering is
what makes the backtest honest - it is impossible for a feature to contain
information from its own match or from the future.

The features try to capture what an experienced watcher actually notices:
who is genuinely stronger (ratings), who is in form right now (time-decayed
recent output), who is creating and conceding chances rather than just scoring
(shot-based proxies, which are far less noisy than goals), who is tired
(rest days, fixture congestion), and what the sharp money thinks (market).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import ModelConfig, DEFAULT_MODEL_CONFIG
from .data import devig_power


class DecayedStat:
    """A time-decayed running mean.

    Old matches should not count as much as recent ones, but a hard "last 5
    games" window throws away real information and jumps around. An exponential
    half-life decays smoothly: a match played `halflife` days ago carries half
    the weight of one played today.
    """

    __slots__ = ("halflife", "weight", "total", "last_ts")

    def __init__(self, halflife_days: float):
        self.halflife = halflife_days
        self.weight = 0.0
        self.total = 0.0
        self.last_ts: float | None = None

    def _decay_to(self, ts: float) -> None:
        if self.last_ts is None:
            self.last_ts = ts
            return
        dt_days = (ts - self.last_ts) / 86400.0
        if dt_days > 0:
            f = 0.5 ** (dt_days / self.halflife)
            self.weight *= f
            self.total *= f
        self.last_ts = ts

    def add(self, ts: float, value: float, w: float = 1.0) -> None:
        if value is None or not np.isfinite(value):
            return
        self._decay_to(ts)
        self.weight += w
        self.total += w * value

    def mean(self, ts: float, prior: float, prior_weight: float = 2.0) -> float:
        """Decayed mean, shrunk toward a prior so new teams are not wild."""
        if self.last_ts is None:
            return prior
        dt_days = (ts - self.last_ts) / 86400.0
        f = 0.5 ** (dt_days / self.halflife) if dt_days > 0 else 1.0
        w = self.weight * f
        t = self.total * f
        return (t + prior * prior_weight) / (w + prior_weight)

    def effective_n(self, ts: float) -> float:
        if self.last_ts is None:
            return 0.0
        dt_days = (ts - self.last_ts) / 86400.0
        f = 0.5 ** (dt_days / self.halflife) if dt_days > 0 else 1.0
        return self.weight * f


class TeamState:
    """Everything we track about one team."""

    def __init__(self, cfg: ModelConfig):
        hl = cfg.form_halflife_days
        self.elo = cfg.elo_start

        # Scoring / conceding, split by venue. Home and away form genuinely
        # differ - some sides are transformed at home - so they are tracked
        # separately as well as combined.
        self.gf_home, self.ga_home = DecayedStat(hl), DecayedStat(hl)
        self.gf_away, self.ga_away = DecayedStat(hl), DecayedStat(hl)
        self.gf_all, self.ga_all = DecayedStat(hl), DecayedStat(hl)

        # Shot volume and quality. Shots on target predict future goals better
        # than past goals do, because finishing is noisier than chance creation.
        self.sot_f, self.sot_a = DecayedStat(hl), DecayedStat(hl)
        self.sh_f, self.sh_a = DecayedStat(hl), DecayedStat(hl)
        self.corners_f, self.corners_a = DecayedStat(hl), DecayedStat(hl)

        # Discipline - red cards swing matches and cluster by team and referee.
        self.cards_f = DecayedStat(hl)

        # Results form
        self.points = DecayedStat(hl)
        self.btts = DecayedStat(hl)
        self.over25 = DecayedStat(hl)
        self.clean_sheets = DecayedStat(hl)
        self.failed_to_score = DecayedStat(hl)

        self.last_match_ts: float | None = None
        self.matches_played = 0
        self.season_matches = 0
        self.current_season: str | None = None
        self.recent_results: list[int] = []   # 1 win / 0 draw / -1 loss, newest last


class LeagueState:
    """League-level baselines: the average match in this division right now."""

    def __init__(self, cfg: ModelConfig):
        hl = cfg.form_halflife_days * 2  # league baselines move slowly
        self.home_goals = DecayedStat(hl)
        self.away_goals = DecayedStat(hl)
        self.sot_per_team = DecayedStat(hl)
        self.goals_per_sot = DecayedStat(hl)
        self.n = 0


def _elo_expected(elo_h: float, elo_a: float, home_adv: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_a - elo_h - home_adv) / 400.0))


def _mov_multiplier(goal_diff: int, elo_diff_adj: float) -> float:
    """Margin-of-victory multiplier (FiveThirtyEight style).

    A 4-0 win says more than a 1-0 win, but with diminishing returns, and we damp
    the effect when a strong favourite beats a weak side, which is expected and
    therefore not very informative.
    """
    gd = abs(goal_diff)
    if gd == 0:
        return 1.0
    return math.log(gd + 1.0) * (2.2 / (0.001 * abs(elo_diff_adj) + 2.2))


class FeatureBuilder:
    """Single forward pass over the match table, emitting pre-match features."""

    def __init__(self, cfg: ModelConfig = DEFAULT_MODEL_CONFIG):
        self.cfg = cfg
        self.teams: dict[tuple[str, str], TeamState] = {}
        self.leagues: dict[str, LeagueState] = {}

    def _team(self, div: str, name: str) -> TeamState:
        key = (div, name)
        if key not in self.teams:
            self.teams[key] = TeamState(self.cfg)
        return self.teams[key]

    def _league(self, div: str) -> LeagueState:
        if div not in self.leagues:
            self.leagues[div] = LeagueState(self.cfg)
        return self.leagues[div]

    # ------------------------------------------------------------------
    def _handle_season_change(self, team: TeamState, season: str, div: str) -> None:
        """Between seasons, regress ratings toward the mean.

        Squads change, managers change, promoted sides are usually worse than
        their rating suggests and relegated sides better. Regressing prevents the
        model from carrying stale certainty into a new campaign.
        """
        if team.current_season == season:
            return
        if team.current_season is not None:
            r = self.cfg.elo_season_regression
            team.elo = self.cfg.elo_start + (1 - r) * (team.elo - self.cfg.elo_start)
        team.current_season = season
        team.season_matches = 0

    # ------------------------------------------------------------------
    def build(self, df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
        """Return the original frame with pre-match feature columns attached."""
        df = df.sort_values("kickoff").reset_index(drop=True)
        # Unit-safe epoch seconds. Do NOT use .astype("int64")/1e9 here: the
        # underlying datetime resolution may be microseconds rather than
        # nanoseconds, which silently rescales every time delta by 1000x and
        # switches off the half-life decay entirely.
        ts_all = (df["kickoff"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(float)
        rows: list[dict] = []
        cfg = self.cfg

        # Pull every column out into a plain array up front. Row-wise .iloc
        # access costs ~100us a hit, which at 140k matches is minutes of pure
        # overhead; array indexing makes the same loop run in seconds.
        def col(name, dtype=float, default=np.nan):
            if name not in df.columns:
                return np.full(len(df), default, dtype=dtype)
            if dtype is float:
                return pd.to_numeric(df[name], errors="coerce").to_numpy(float)
            return df[name].to_numpy()

        c_div = df["div"].astype(str).to_numpy()
        c_season = df["season"].astype(str).to_numpy()
        c_home = df["home"].astype(str).to_numpy()
        c_away = df["away"].astype(str).to_numpy()
        c_played = df["played"].to_numpy(bool)
        c_hg, c_ag = col("hg"), col("ag")
        c_hst, c_ast = col("hst"), col("ast")
        c_hs, c_as = col("hs"), col("as_")
        c_hc, c_ac = col("hc"), col("ac")
        c_hy, c_ay = col("hy"), col("ay")
        c_hr, c_ar = col("hr"), col("ar")
        c_oh, c_od, c_oa = col("odds_h"), col("odds_d"), col("odds_a")
        c_o25, c_u25 = col("odds_o25"), col("odds_u25")

        for i in range(len(df)):
            ts = ts_all[i]
            div, season = c_div[i], c_season[i]
            lg = self._league(div)
            th, ta = self._team(div, c_home[i]), self._team(div, c_away[i])
            self._handle_season_change(th, season, div)
            self._handle_season_change(ta, season, div)

            # League baselines (with sane priors before enough data exists).
            lg_hg = lg.home_goals.mean(ts, prior=1.50, prior_weight=20)
            lg_ag = lg.away_goals.mean(ts, prior=1.15, prior_weight=20)
            lg_sot = lg.sot_per_team.mean(ts, prior=4.5, prior_weight=20)
            lg_conv = lg.goals_per_sot.mean(ts, prior=cfg.sot_conversion_prior, prior_weight=40)
            lg_conv = float(np.clip(lg_conv, 0.20, 0.45))

            f: dict = {"idx": i}

            # ---------------- ratings ----------------
            f["elo_h"], f["elo_a"] = th.elo, ta.elo
            f["elo_diff"] = th.elo - ta.elo
            f["elo_exp_h"] = _elo_expected(th.elo, ta.elo, cfg.elo_home_advantage)

            # ---------------- goal-rate strength ----------------
            # Attack strength = this team's scoring rate relative to the league
            # average for that venue. 1.0 is average, 1.3 is 30% better.
            gf_h = th.gf_home.mean(ts, prior=lg_hg, prior_weight=3)
            ga_h = th.ga_home.mean(ts, prior=lg_ag, prior_weight=3)
            gf_a = ta.gf_away.mean(ts, prior=lg_ag, prior_weight=3)
            ga_a = ta.ga_away.mean(ts, prior=lg_hg, prior_weight=3)

            f["att_home"] = gf_h / max(lg_hg, 0.2)
            f["def_home"] = ga_h / max(lg_ag, 0.2)
            f["att_away"] = gf_a / max(lg_ag, 0.2)
            f["def_away"] = ga_a / max(lg_hg, 0.2)

            # Overall (venue-agnostic) form, useful when venue samples are thin.
            f["att_home_all"] = th.gf_all.mean(ts, prior=(lg_hg + lg_ag) / 2, prior_weight=3)
            f["def_home_all"] = th.ga_all.mean(ts, prior=(lg_hg + lg_ag) / 2, prior_weight=3)
            f["att_away_all"] = ta.gf_all.mean(ts, prior=(lg_hg + lg_ag) / 2, prior_weight=3)
            f["def_away_all"] = ta.ga_all.mean(ts, prior=(lg_hg + lg_ag) / 2, prior_weight=3)

            # ---------------- shot-based expectation ----------------
            # Goals lie in small samples; shots on target lie less. This is the
            # closest thing to xG available from a free feed.
            sot_h = th.sot_f.mean(ts, prior=lg_sot, prior_weight=3)
            sot_ca_h = th.sot_a.mean(ts, prior=lg_sot, prior_weight=3)
            sot_a = ta.sot_f.mean(ts, prior=lg_sot, prior_weight=3)
            sot_ca_a = ta.sot_a.mean(ts, prior=lg_sot, prior_weight=3)
            f["xg_proxy_h"] = sot_h * lg_conv
            f["xg_proxy_a"] = sot_a * lg_conv
            f["xg_conceded_h"] = sot_ca_h * lg_conv
            f["xg_conceded_a"] = sot_ca_a * lg_conv

            # Finishing luck: scoring far above shot quality tends to regress.
            # A side "in red-hot form" on the back of unsustainable finishing is
            # exactly where naive models get burned.
            f["finishing_luck_h"] = th.gf_all.mean(ts, prior=1.3, prior_weight=3) - sot_h * lg_conv
            f["finishing_luck_a"] = ta.gf_all.mean(ts, prior=1.3, prior_weight=3) - sot_a * lg_conv

            f["shots_h"] = th.sh_f.mean(ts, prior=12.0, prior_weight=3)
            f["shots_a"] = ta.sh_f.mean(ts, prior=12.0, prior_weight=3)
            f["corners_h"] = th.corners_f.mean(ts, prior=5.0, prior_weight=3)
            f["corners_a"] = ta.corners_f.mean(ts, prior=5.0, prior_weight=3)
            f["cards_h"] = th.cards_f.mean(ts, prior=1.8, prior_weight=3)
            f["cards_a"] = ta.cards_f.mean(ts, prior=1.8, prior_weight=3)

            # ---------------- form and match-state tendencies ----------------
            f["ppg_h"] = th.points.mean(ts, prior=1.35, prior_weight=3)
            f["ppg_a"] = ta.points.mean(ts, prior=1.35, prior_weight=3)
            f["btts_rate_h"] = th.btts.mean(ts, prior=0.50, prior_weight=4)
            f["btts_rate_a"] = ta.btts.mean(ts, prior=0.50, prior_weight=4)
            f["over25_rate_h"] = th.over25.mean(ts, prior=0.52, prior_weight=4)
            f["over25_rate_a"] = ta.over25.mean(ts, prior=0.52, prior_weight=4)
            f["cs_rate_h"] = th.clean_sheets.mean(ts, prior=0.28, prior_weight=4)
            f["cs_rate_a"] = ta.clean_sheets.mean(ts, prior=0.28, prior_weight=4)
            f["fts_rate_h"] = th.failed_to_score.mean(ts, prior=0.26, prior_weight=4)
            f["fts_rate_a"] = ta.failed_to_score.mean(ts, prior=0.26, prior_weight=4)

            # Short-horizon streak: the thing a human would call "momentum".
            f["streak_h"] = sum(th.recent_results[-5:])
            f["streak_a"] = sum(ta.recent_results[-5:])

            # ---------------- fatigue and experience ----------------
            f["rest_h"] = (ts - th.last_match_ts) / 86400.0 if th.last_match_ts else 14.0
            f["rest_a"] = (ts - ta.last_match_ts) / 86400.0 if ta.last_match_ts else 14.0
            f["rest_h"] = min(f["rest_h"], 60.0)
            f["rest_a"] = min(f["rest_a"], 60.0)
            f["rest_diff"] = f["rest_h"] - f["rest_a"]
            # A short turnaround (under 4 days) measurably degrades performance.
            f["congested_h"] = 1.0 if f["rest_h"] < 4.0 else 0.0
            f["congested_a"] = 1.0 if f["rest_a"] < 4.0 else 0.0

            f["mp_h"], f["mp_a"] = th.matches_played, ta.matches_played
            f["season_mp_h"], f["season_mp_a"] = th.season_matches, ta.season_matches
            f["exp_n_h"] = th.gf_all.effective_n(ts)
            f["exp_n_a"] = ta.gf_all.effective_n(ts)

            f["lg_home_goals"], f["lg_away_goals"] = lg_hg, lg_ag
            f["lg_conv"] = lg_conv

            # ---------------- market ----------------
            oh, od, oa = c_oh[i], c_od[i], c_oa[i]
            if np.isfinite(oh) and np.isfinite(od) and np.isfinite(oa) and min(oh, od, oa) > 1.01:
                p = devig_power(np.array([oh, od, oa], dtype=float))
                f["mkt_h"], f["mkt_d"], f["mkt_a"] = float(p[0]), float(p[1]), float(p[2])
                f["mkt_margin"] = float(1 / oh + 1 / od + 1 / oa - 1)
                f["has_market"] = 1.0
            else:
                f["mkt_h"] = f["mkt_d"] = f["mkt_a"] = np.nan
                f["mkt_margin"] = np.nan
                f["has_market"] = 0.0

            o25, u25 = c_o25[i], c_u25[i]
            if np.isfinite(o25) and np.isfinite(u25) and min(o25, u25) > 1.01:
                p = devig_power(np.array([o25, u25], dtype=float))
                f["mkt_over25"] = float(p[0])
                f["has_ou_market"] = 1.0
            else:
                f["mkt_over25"] = np.nan
                f["has_ou_market"] = 0.0

            rows.append(f)

            # ==========================================================
            # STATE UPDATE - only after the feature row above is frozen.
            # ==========================================================
            if not c_played[i]:
                continue

            hg, ag = c_hg[i], c_ag[i]
            if not (np.isfinite(hg) and np.isfinite(ag)):
                continue
            gd = int(hg - ag)

            # Elo
            exp_h = f["elo_exp_h"]
            actual_h = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
            elo_diff_adj = (th.elo + cfg.elo_home_advantage - ta.elo) * (1 if gd > 0 else -1)
            mult = _mov_multiplier(gd, elo_diff_adj) * cfg.elo_mov_factor
            delta = cfg.elo_k * mult * (actual_h - exp_h)
            th.elo += delta
            ta.elo -= delta

            # Goal rates
            th.gf_home.add(ts, hg); th.ga_home.add(ts, ag)
            ta.gf_away.add(ts, ag); ta.ga_away.add(ts, hg)
            th.gf_all.add(ts, hg); th.ga_all.add(ts, ag)
            ta.gf_all.add(ts, ag); ta.ga_all.add(ts, hg)

            # Shots / corners / cards, when the feed carries them
            hst, ast_ = c_hst[i], c_ast[i]
            if np.isfinite(hst) and np.isfinite(ast_):
                th.sot_f.add(ts, hst); th.sot_a.add(ts, ast_)
                ta.sot_f.add(ts, ast_); ta.sot_a.add(ts, hst)
                tot_sot = hst + ast_
                if tot_sot > 0:
                    lg.goals_per_sot.add(ts, (hg + ag) / tot_sot)
                lg.sot_per_team.add(ts, tot_sot / 2.0)
            hs, as_ = c_hs[i], c_as[i]
            if np.isfinite(hs) and np.isfinite(as_):
                th.sh_f.add(ts, hs); th.sh_a.add(ts, as_)
                ta.sh_f.add(ts, as_); ta.sh_a.add(ts, hs)
            hc, ac = c_hc[i], c_ac[i]
            if np.isfinite(hc) and np.isfinite(ac):
                th.corners_f.add(ts, hc); th.corners_a.add(ts, ac)
                ta.corners_f.add(ts, ac); ta.corners_a.add(ts, hc)
            hy, ay, hr_, ar_ = c_hy[i], c_ay[i], c_hr[i], c_ar[i]
            red_h = hr_ if np.isfinite(hr_) else 0.0
            red_a = ar_ if np.isfinite(ar_) else 0.0
            # A red card is weighted as two yellows: it is far more damaging.
            if np.isfinite(hy):
                th.cards_f.add(ts, hy + 2.0 * red_h)
            if np.isfinite(ay):
                ta.cards_f.add(ts, ay + 2.0 * red_a)

            # Results-derived form
            th.points.add(ts, 3.0 if gd > 0 else (1.0 if gd == 0 else 0.0))
            ta.points.add(ts, 3.0 if gd < 0 else (1.0 if gd == 0 else 0.0))
            btts = 1.0 if (hg > 0 and ag > 0) else 0.0
            th.btts.add(ts, btts); ta.btts.add(ts, btts)
            o25f = 1.0 if (hg + ag) > 2.5 else 0.0
            th.over25.add(ts, o25f); ta.over25.add(ts, o25f)
            th.clean_sheets.add(ts, 1.0 if ag == 0 else 0.0)
            ta.clean_sheets.add(ts, 1.0 if hg == 0 else 0.0)
            th.failed_to_score.add(ts, 1.0 if hg == 0 else 0.0)
            ta.failed_to_score.add(ts, 1.0 if ag == 0 else 0.0)

            th.recent_results.append(1 if gd > 0 else (0 if gd == 0 else -1))
            ta.recent_results.append(1 if gd < 0 else (0 if gd == 0 else -1))
            th.recent_results = th.recent_results[-10:]
            ta.recent_results = ta.recent_results[-10:]

            th.last_match_ts = ta.last_match_ts = ts
            th.matches_played += 1; ta.matches_played += 1
            th.season_matches += 1; ta.season_matches += 1

            lg.home_goals.add(ts, hg); lg.away_goals.add(ts, ag)
            lg.n += 1

            if verbose and i % 20000 == 0 and i:
                print(f"    features: {i}/{len(df)}", flush=True)

        feats = pd.DataFrame(rows).set_index("idx")
        out = pd.concat([df, feats], axis=1)
        return out


FEATURE_COLUMNS = [
    "elo_diff", "elo_exp_h", "elo_h", "elo_a",
    "att_home", "def_home", "att_away", "def_away",
    "att_home_all", "def_home_all", "att_away_all", "def_away_all",
    "xg_proxy_h", "xg_proxy_a", "xg_conceded_h", "xg_conceded_a",
    "finishing_luck_h", "finishing_luck_a",
    "shots_h", "shots_a", "corners_h", "corners_a", "cards_h", "cards_a",
    "ppg_h", "ppg_a", "btts_rate_h", "btts_rate_a",
    "over25_rate_h", "over25_rate_a", "cs_rate_h", "cs_rate_a",
    "fts_rate_h", "fts_rate_a", "streak_h", "streak_a",
    "rest_h", "rest_a", "rest_diff", "congested_h", "congested_a",
    "season_mp_h", "season_mp_a", "exp_n_h", "exp_n_a",
    "lg_home_goals", "lg_away_goals", "lg_conv",
    "mkt_h", "mkt_d", "mkt_a", "mkt_over25", "mkt_margin",
]


def build_features(df: pd.DataFrame, cfg: ModelConfig = DEFAULT_MODEL_CONFIG,
                   *, verbose: bool = True) -> pd.DataFrame:
    return FeatureBuilder(cfg).build(df, verbose=verbose)
