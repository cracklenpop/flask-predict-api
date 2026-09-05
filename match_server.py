"""HTTP API for the match predictor.

Kept separate from server.py (the PPO scalper) so the two have no shared
dependencies - you can run this without torch installed, and that one without
the football data cached.

    python match_server.py            # serves on 0.0.0.0:10001

Endpoints
    GET  /health                      service + model status
    GET  /fixtures?days=2             upcoming fixtures the engine can see
    GET  /markets/<match_id>          every market priced for one fixture
    GET  /slip?days=2&target=2.0      conviction picks + staking plan
    POST /slip                        same, with your real Betway prices
    GET  /calibration                 reliability tables behind the claims
"""

from __future__ import annotations

import os
import traceback

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from matchpredictor import pipeline
from matchpredictor.config import ConvictionConfig
from matchpredictor.markets import correct_scores, label_for
from matchpredictor.model import MatchPredictor
from matchpredictor.staking import build_plan, ladder

app = Flask(__name__)

_STATE: dict = {"model": None, "cal": None, "meta": {}, "feat": None}


def _ensure_loaded(reload: bool = False):
    if reload or _STATE["model"] is None:
        _STATE["model"], _STATE["cal"], _STATE["meta"] = pipeline.load_artifacts()
    if reload or _STATE["feat"] is None:
        _STATE["feat"] = pipeline.load_features()
    return _STATE["model"], _STATE["cal"], _STATE["feat"]


def _err(e: Exception):
    return jsonify(error=str(e), type=type(e).__name__,
                   trace=traceback.format_exc().splitlines()[-3:]), 500


@app.route("/health")
def health():
    try:
        model, cal, feat = _ensure_loaded()
        up = pipeline.upcoming(feat, days=7)
        return jsonify(
            status="ok",
            model_trained_at=_STATE["meta"].get("trained_at"),
            matches_in_history=int(feat["played"].sum()),
            upcoming_7d=len(up),
            calibrated_families=sorted(cal.families),
            rho=_STATE["meta"].get("rho"),
        )
    except Exception as e:
        return jsonify(status="not_ready", error=str(e)), 503


@app.route("/fixtures")
def fixtures():
    try:
        _, _, feat = _ensure_loaded()
        days = int(request.args.get("days", 2))
        up = pipeline.upcoming(feat, days=days)
        if request.args.get("league"):
            up = up[up["div"].isin(request.args["league"].split(","))]
        cols = ["match_id", "date", "div", "home", "away", "odds_h", "odds_d", "odds_a"]
        out = up[cols].copy()
        out["kickoff"] = up["kickoff"].dt.strftime("%Y-%m-%d %H:%M")
        return jsonify(count=len(out), fixtures=out.replace({np.nan: None}).to_dict("records"))
    except Exception as e:
        return _err(e)


@app.route("/markets/<path:match_id>")
def markets_for(match_id: str):
    """Every market priced for a single fixture, plus the likeliest scorelines."""
    try:
        model, cal, feat = _ensure_loaded()
        up = pipeline.upcoming(feat, days=14)
        row = up[up["match_id"] == match_id]
        if len(row) == 0:
            return jsonify(error="fixture not found", match_id=match_id), 404

        priced = pipeline.price_fixtures(row, model, cal)
        home, away = row["home"].iloc[0], row["away"].iloc[0]
        grid = model.predict_grid(row)
        lam = model.predict_lambdas(row)

        out = []
        for _, r in priced.sort_values("p_cal", ascending=False).iterrows():
            out.append({
                "market": r["market"],
                "selection": label_for(r["market"], home, away),
                "probability": round(float(r["p_cal"]), 4),
                "fair_odds": round(1.0 / max(float(r["p_cal"]), 1e-6), 3),
                "model_raw": round(float(r["p_raw"]), 4),
                "market_view": round(float(r["p_market"]), 4),
            })
        return jsonify(
            match_id=match_id, home=home, away=away, div=row["div"].iloc[0],
            date=str(row["date"].iloc[0])[:10],
            expected_goals={"home": round(float(lam["lam_h"][0]), 3),
                            "away": round(float(lam["lam_a"][0]), 3)},
            likely_scores=[{"score": s, "probability": round(p, 4)}
                           for s, p in correct_scores(grid, 6)[0]],
            markets=out,
        )
    except Exception as e:
        return _err(e)


def _slip_payload(days: int, target: float, bankroll: float, max_legs: int,
                  league: str | None, betway: dict | None,
                  min_prob: float | None, real_only: bool):
    model, cal, feat = _ensure_loaded()
    up = pipeline.upcoming(feat, days=days)
    if league:
        up = up[up["div"].isin(league.split(","))]
    if len(up) == 0:
        return {"fixtures_scanned": 0, "picks": [], "watchlist": [], "plan": None,
                "message": "No upcoming fixtures in this window."}

    cfg = ConvictionConfig()
    if min_prob is not None:
        cfg.tiers = {k: max(v, min_prob) for k, v in cfg.tiers.items()}

    picks, _, watch = pipeline.todays_picks(model, cal, up, betway, cfg)
    if real_only:
        watch = []

    plan = build_plan(picks, target=target, max_legs=max_legs) if picks else None
    lad = ladder(picks, max_legs=max_legs) if picks else []

    return {
        "fixtures_scanned": len(up),
        "picks": [p.to_dict() for p in picks],
        "watchlist": [w.to_dict() for w in watch],
        "plan": None if plan is None else {
            "kind": plan.kind,
            "target": plan.target,
            "probability_of_hitting_target": round(plan.p_double, 4),
            "expected_return_per_1_staked": round(plan.ev, 4),
            "suggested_bankroll_fraction": round(plan.kelly_bankroll_fraction, 4),
            "bets": [
                {"stake": round(bankroll * share, 2), "combined_odds": round(odds, 3),
                 "returns": round(bankroll * share * odds, 2), "win_probability": round(pp, 4),
                 "legs": [{"match": f"{l.home} v {l.away}", "selection": l.selection,
                           "price": l.price, "tier": l.tier, "probability": round(l.prob, 4)}
                          for l in legs]}
                for legs, share, odds, pp in zip(plan.legs, plan.stake_share,
                                                 plan.combined_odds, plan.leg_probs)
            ],
        },
        "ladder": [{"target": pl.target, "kind": pl.kind, "legs": pl.n_legs,
                    "probability": round(pl.p_double, 4),
                    "expected_return": round(pl.ev, 4)} for pl in lad],
    }


@app.route("/slip", methods=["GET", "POST"])
def slip():
    """Conviction picks and a staking plan.

    POST body (all optional):
        {"days": 2, "target": 2.0, "bankroll": 1000, "league": "E0,SP1",
         "prices": {"<match_id>": {"DC_1X": 1.24, "1X2_HOME": 1.55}}}

    Supplying `prices` is strongly recommended: without your real Betway
    numbers, any market the feed does not quote is priced from an estimate and
    its edge should not be trusted.
    """
    try:
        body = request.get_json(silent=True) or {}
        g = request.args

        def pick(name, cast, default):
            if name in body:
                return cast(body[name])
            if name in g:
                return cast(g[name])
            return default

        payload = _slip_payload(
            days=pick("days", int, 2),
            target=pick("target", float, 2.0),
            bankroll=pick("bankroll", float, 1000.0),
            max_legs=pick("max_legs", int, 6),
            league=pick("league", str, None),
            betway=body.get("prices"),
            min_prob=pick("min_prob", float, None),
            real_only=bool(pick("real_prices_only", int, 0)),
        )
        return jsonify(payload)
    except Exception as e:
        return _err(e)


@app.route("/calibration")
def calibration():
    try:
        _, cal, _ = _ensure_loaded()
        out = {}
        for fam, fc in cal.families.items():
            bands = []
            from matchpredictor.calibration import EVIDENCE_BINS
            for i in range(len(EVIDENCE_BINS) - 1):
                n = int(fc.bin_counts[i])
                if n == 0:
                    continue
                bands.append({
                    "band": f"{EVIDENCE_BINS[i]:.2f}-{min(EVIDENCE_BINS[i+1],1.0):.2f}",
                    "n": n, "actual_hit_rate": round(float(fc.bin_hits[i] / n), 4),
                })
            out[fam] = {"n_train": fc.n_train,
                        "brier_raw": fc.brier_raw, "brier_calibrated": fc.brier_cal,
                        "logloss_raw": fc.logloss_raw, "logloss_calibrated": fc.logloss_cal,
                        "bands": bands}
        return jsonify(out)
    except Exception as e:
        return _err(e)


@app.route("/reload", methods=["POST"])
def reload_artifacts():
    try:
        _ensure_loaded(reload=True)
        return jsonify(status="reloaded", trained_at=_STATE["meta"].get("trained_at"))
    except Exception as e:
        return _err(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10001)))
