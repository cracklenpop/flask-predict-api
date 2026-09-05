"""The ensemble that turns features into expected goals, and expected goals into
a full score distribution.

Three independent opinions are formed for every match, then blended:

1. RATINGS   - a classical Poisson strength model: league average scoring rate,
               scaled by the home side's attack and the away side's defence.
               Transparent, stable, and completely independent of the market.
2. MARKET    - the goal expectations implied by de-vigged closing prices. This is
               the sharpest single signal available and it is very hard to beat.
3. LEARNED   - gradient-boosted Poisson regressions over the full feature set,
               which pick up the non-linear interactions the first two miss
               (fatigue crossed with squad strength, finishing regression,
               congestion, discipline, venue effects).

Blending matters more than any one component. The market keeps us honest, the
ratings model stops us blindly following a mispriced line, and the learned model
finds the residual edge. Where they *agree strongly* is exactly where the
conviction engine is allowed to speak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import ModelConfig, DEFAULT_MODEL_CONFIG
from .features import FEATURE_COLUMNS
from .poisson import (DEFAULT_MAX_GOALS, dixon_coles_grid, estimate_rho,
                      grid_1x2, invert_market_lambdas)


def ratings_lambdas(df: pd.DataFrame, cfg: ModelConfig = DEFAULT_MODEL_CONFIG
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Classical Poisson strength model, with a shot-based correction.

    lambda_home = league_home_average * home_attack * away_defence

    The shot proxy is then mixed in, because a side scoring well above its chance
    creation is usually about to stop doing that, and a side creating chances
    without scoring is usually about to start.
    """
    lg_h = df["lg_home_goals"].to_numpy(float)
    lg_a = df["lg_away_goals"].to_numpy(float)

    att_h = np.clip(df["att_home"].to_numpy(float), 0.25, 3.0)
    def_a = np.clip(df["def_away"].to_numpy(float), 0.25, 3.0)
    att_a = np.clip(df["att_away"].to_numpy(float), 0.25, 3.0)
    def_h = np.clip(df["def_home"].to_numpy(float), 0.25, 3.0)

    lam_h = lg_h * att_h * def_a
    lam_a = lg_a * att_a * def_h

    # Shot-based expectation for the same match-up, on the same scale.
    xg_h = np.clip(df["xg_proxy_h"].to_numpy(float), 0.1, 5.0)
    xgc_a = np.clip(df["xg_conceded_a"].to_numpy(float), 0.1, 5.0)
    xg_a = np.clip(df["xg_proxy_a"].to_numpy(float), 0.1, 5.0)
    xgc_h = np.clip(df["xg_conceded_h"].to_numpy(float), 0.1, 5.0)
    shot_h = np.sqrt(xg_h * xgc_a)
    shot_a = np.sqrt(xg_a * xgc_h)

    w = cfg.shot_weight
    lam_h = (1 - w) * lam_h + w * shot_h
    lam_a = (1 - w) * lam_a + w * shot_a

    # Elo supremacy nudge: ratings carry cross-season information that the
    # decayed goal rates alone lose, especially early in a campaign.
    elo_sup = np.clip(df["elo_diff"].to_numpy(float) / 400.0, -1.2, 1.2)
    lam_h *= np.exp(0.10 * elo_sup)
    lam_a *= np.exp(-0.10 * elo_sup)

    return np.clip(lam_h, 0.15, 6.0), np.clip(lam_a, 0.15, 6.0)


def market_lambdas(df: pd.DataFrame, rho: float = -0.05) -> tuple[np.ndarray, np.ndarray]:
    """Goal expectations implied by the de-vigged market prices."""
    return invert_market_lambdas(
        df["mkt_h"].to_numpy(float),
        df["mkt_a"].to_numpy(float),
        df["mkt_over25"].to_numpy(float) if "mkt_over25" in df else None,
        rho=rho,
    )


GBM_EXTRA_COLUMNS = ["rat_lam_h", "rat_lam_a", "mkt_lam_h", "mkt_lam_a",
                     "rat_sup", "mkt_sup", "rat_tot", "mkt_tot"]


def _augment(df: pd.DataFrame, cfg: ModelConfig, rho: float) -> pd.DataFrame:
    """Attach the ratings and market lambdas as extra model inputs."""
    out = df.copy()
    rl_h, rl_a = ratings_lambdas(out, cfg)
    ml_h, ml_a = market_lambdas(out, rho)
    out["rat_lam_h"], out["rat_lam_a"] = rl_h, rl_a
    out["mkt_lam_h"], out["mkt_lam_a"] = ml_h, ml_a
    out["rat_sup"], out["rat_tot"] = rl_h - rl_a, rl_h + rl_a
    out["mkt_sup"], out["mkt_tot"] = ml_h - ml_a, ml_h + ml_a
    return out


class MatchPredictor:
    """Fits on history, predicts a score grid for any set of matches."""

    def __init__(self, cfg: ModelConfig = DEFAULT_MODEL_CONFIG,
                 max_goals: int = DEFAULT_MAX_GOALS):
        self.cfg = cfg
        self.max_goals = max_goals
        self.rho = -0.05
        self.gbm_h: HistGradientBoostingRegressor | None = None
        self.gbm_a: HistGradientBoostingRegressor | None = None
        self.feature_cols = FEATURE_COLUMNS + GBM_EXTRA_COLUMNS
        self.feature_cols_used: list[str] = list(self.feature_cols)
        self.weights = {"market": cfg.w_market, "gbm": cfg.w_dixon_coles, "ratings": cfg.w_ratings}
        self.fitted = False

    # ------------------------------------------------------------------
    def fit(self, train: pd.DataFrame, *, verbose: bool = True) -> "MatchPredictor":
        train = train[train["played"] == True].copy()  # noqa: E712
        if len(train) < 500:
            raise ValueError(f"Not enough training data: {len(train)} matches")

        # Estimate the low-score dependency parameter from the ratings model.
        rl_h, rl_a = ratings_lambdas(train, self.cfg)
        self.rho = estimate_rho(train["hg"].to_numpy(float), train["ag"].to_numpy(float), rl_h, rl_a)
        if verbose:
            print(f"    rho = {self.rho:+.4f}", flush=True)

        aug = _augment(train, self.cfg, self.rho)

        # Some inputs simply do not exist in older data - the over/under 2.5
        # market, for instance, is absent from the feed before roughly 2012. A
        # column that is entirely missing (or constant) in the training window
        # carries no information and makes the histogram binner fail outright,
        # so the usable subset is chosen per fit and recorded for prediction.
        usable = []
        for c in self.feature_cols:
            v = aug[c].to_numpy(float)
            finite = v[np.isfinite(v)]
            if finite.size >= 50 and np.unique(finite).size > 1:
                usable.append(c)
        self.feature_cols_used = usable
        if verbose:
            dropped = [c for c in self.feature_cols if c not in usable]
            if dropped:
                print(f"    dropped {len(dropped)} uninformative feature(s): "
                      f"{', '.join(dropped)}", flush=True)

        X = aug[self.feature_cols_used].to_numpy(float)
        yh = aug["hg"].to_numpy(float)
        ya = aug["ag"].to_numpy(float)

        # Recent matches matter more than a decade-old ones, but old data still
        # helps the model learn stable structure, so weight rather than discard.
        age_days = (aug["kickoff"].max() - aug["kickoff"]).dt.total_seconds().to_numpy() / 86400.0
        w = 0.5 ** (age_days / (self.cfg.form_halflife_days * 6))
        w = np.clip(w, 0.02, 1.0)

        params = dict(loss="poisson", max_iter=400, learning_rate=0.05,
                      max_leaf_nodes=31, min_samples_leaf=60,
                      l2_regularization=1.0, early_stopping=True,
                      validation_fraction=0.12, n_iter_no_change=25,
                      random_state=7)
        self.gbm_h = HistGradientBoostingRegressor(**params).fit(X, yh, sample_weight=w)
        self.gbm_a = HistGradientBoostingRegressor(**params).fit(X, ya, sample_weight=w)
        if verbose:
            print(f"    gbm fitted on {len(train)} matches "
                  f"({self.gbm_h.n_iter_}/{self.gbm_a.n_iter_} trees)", flush=True)

        self.fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_lambdas(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Blend the three opinions into one pair of goal expectations.

        Also returns the components, because the *spread* between them is the
        disagreement signal the conviction engine uses to stay quiet.
        """
        aug = _augment(df, self.cfg, self.rho)
        rl_h = aug["rat_lam_h"].to_numpy(float)
        rl_a = aug["rat_lam_a"].to_numpy(float)
        ml_h = aug["mkt_lam_h"].to_numpy(float)
        ml_a = aug["mkt_lam_a"].to_numpy(float)

        if self.fitted:
            X = aug[self.feature_cols_used].to_numpy(float)
            gl_h = np.clip(self.gbm_h.predict(X), 0.15, 6.0)
            gl_a = np.clip(self.gbm_a.predict(X), 0.15, 6.0)
        else:
            gl_h, gl_a = rl_h.copy(), rl_a.copy()

        # Blend in log space so the components combine multiplicatively, which
        # is the natural scale for rates, and renormalize the weights over
        # whichever components are actually available for this match.
        w_mkt = np.where(np.isfinite(ml_h) & np.isfinite(ml_a), self.weights["market"], 0.0)
        w_gbm = np.full(len(aug), self.weights["gbm"])
        w_rat = np.full(len(aug), self.weights["ratings"])

        # Early in a team's life we have little of our own signal, so lean harder
        # on the market rather than pretending to know better.
        thin = (aug["exp_n_h"].to_numpy(float) < self.cfg.form_min_matches) | \
               (aug["exp_n_a"].to_numpy(float) < self.cfg.form_min_matches)
        boost = np.where(thin & (w_mkt > 0), 0.25, 0.0)
        w_mkt = w_mkt + boost
        w_rat = np.maximum(w_rat - boost, 0.02)

        tot = w_mkt + w_gbm + w_rat
        w_mkt, w_gbm, w_rat = w_mkt / tot, w_gbm / tot, w_rat / tot

        safe_ml_h = np.where(np.isfinite(ml_h), ml_h, 1.0)
        safe_ml_a = np.where(np.isfinite(ml_a), ml_a, 1.0)

        log_h = (w_mkt * np.log(np.clip(safe_ml_h, 0.05, 8)) +
                 w_gbm * np.log(np.clip(gl_h, 0.05, 8)) +
                 w_rat * np.log(np.clip(rl_h, 0.05, 8)))
        log_a = (w_mkt * np.log(np.clip(safe_ml_a, 0.05, 8)) +
                 w_gbm * np.log(np.clip(gl_a, 0.05, 8)) +
                 w_rat * np.log(np.clip(rl_a, 0.05, 8)))

        return {
            "lam_h": np.clip(np.exp(log_h), 0.12, 6.5),
            "lam_a": np.clip(np.exp(log_a), 0.12, 6.5),
            "rat_lam_h": rl_h, "rat_lam_a": rl_a,
            "mkt_lam_h": ml_h, "mkt_lam_a": ml_a,
            "gbm_lam_h": gl_h, "gbm_lam_a": gl_a,
        }

    # ------------------------------------------------------------------
    def predict_grid(self, df: pd.DataFrame, chunk: int = 20000) -> np.ndarray:
        lam = self.predict_lambdas(df)
        return self._grid(lam["lam_h"], lam["lam_a"], chunk)

    def _grid(self, lh: np.ndarray, la: np.ndarray, chunk: int = 20000) -> np.ndarray:
        parts = []
        for i in range(0, len(lh), chunk):
            parts.append(dixon_coles_grid(lh[i:i + chunk], la[i:i + chunk],
                                          self.rho, self.max_goals))
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, self.max_goals + 1, self.max_goals + 1))

    def component_grids(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Score grids for each component separately, for the agreement check."""
        lam = self.predict_lambdas(df)
        out = {"blend": self._grid(lam["lam_h"], lam["lam_a"])}
        out["ratings"] = self._grid(lam["rat_lam_h"], lam["rat_lam_a"])
        out["gbm"] = self._grid(lam["gbm_lam_h"], lam["gbm_lam_a"])
        mh = np.where(np.isfinite(lam["mkt_lam_h"]), lam["mkt_lam_h"], lam["lam_h"])
        ma = np.where(np.isfinite(lam["mkt_lam_a"]), lam["mkt_lam_a"], lam["lam_a"])
        out["market"] = self._grid(mh, ma)
        out["_market_available"] = np.isfinite(lam["mkt_lam_h"]) & np.isfinite(lam["mkt_lam_a"])
        return out
