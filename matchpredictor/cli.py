"""Command line interface.

    python -m matchpredictor update              # refresh results + fixtures
    python -m matchpredictor train               # fit model + calibration
    python -m matchpredictor backtest            # walk-forward validation
    python -m matchpredictor slip                # today's conviction picks
    python -m matchpredictor slip --target 2.0 --bankroll 1000
    python -m matchpredictor calibration         # show the reliability tables
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import backtest as bt
from . import pipeline
from .config import (DEFAULT_CONVICTION_CONFIG, DEFAULT_LEAGUES,
                     DEFAULT_MODEL_CONFIG, ConvictionConfig)
from .staking import build_plan, ladder


def _hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
def cmd_update(args) -> int:
    _hr("UPDATING DATA")
    leagues = args.leagues.split(",") if args.leagues else DEFAULT_LEAGUES
    feat = pipeline.update_data(leagues, args.start_year, args.end_year,
                                refresh=not args.no_refresh)
    up = pipeline.upcoming(feat, days=7)
    print(f"\n{len(up)} fixtures in the next 7 days.")
    return 0


def cmd_train(args) -> int:
    _hr("TRAINING")
    feat = pipeline.load_features()
    seasons = args.calib_seasons.split(",") if args.calib_seasons else None
    model, cal = pipeline.train(feat, seasons)
    print("\nCalibration summary:")
    for fam, fc in sorted(cal.families.items()):
        print(f"  {fam:<16} n={fc.n_train:>8,}  "
              f"Brier {fc.brier_raw:.4f} -> {fc.brier_cal:.4f}")
    return 0


def cmd_calibration(args) -> int:
    _hr("CALIBRATION / RELIABILITY")
    _, cal, meta = pipeline.load_artifacts()
    if meta:
        print(f"trained {meta.get('trained_at','?')} on {meta.get('n_train',0):,} matches")
    print(cal.reliability_report())
    return 0


def cmd_backtest(args) -> int:
    _hr("WALK-FORWARD BACKTEST")
    feat = pipeline.load_features()
    played = feat[feat["played"] == True]  # noqa: E712
    seasons = (args.seasons.split(",") if args.seasons
               else sorted(played["season"].unique())[-args.n_seasons:])
    print(f"test seasons: {', '.join(seasons)}\n")

    oos = bt.run_walkforward(feat, seasons, DEFAULT_MODEL_CONFIG)
    print()
    oos, cals = bt.calibrate_progressively(oos, seasons)

    cfg = ConvictionConfig()
    if args.min_prob is not None:
        cfg.tiers = {k: max(v, args.min_prob) for k, v in cfg.tiers.items()}
    if args.min_edge is not None:
        cfg.min_edge = args.min_edge
        cfg.min_edge_lock = args.min_edge

    print()
    bets = bt.evaluate_picks(oos, cals, cfg)

    _hr("RESULTS BY CONVICTION TIER")
    print(bt.tier_report(bets))
    _hr("RESULTS BY SEASON")
    print(bt.season_report(bets))
    _hr("RESULTS BY MARKET")
    print(bt.market_report(bets))

    if args.save:
        oos.to_parquet(args.save)
        print(f"\nsaved out-of-sample predictions to {args.save}")
    return 0


def cmd_slip(args) -> int:
    _hr("CONVICTION SLIP")
    model, cal, meta = pipeline.load_artifacts()
    feat = pipeline.load_features()
    fixtures = pipeline.upcoming(feat, days=args.days)
    if args.league:
        fixtures = fixtures[fixtures["div"].isin(args.league.split(","))]
    if len(fixtures) == 0:
        print("No upcoming fixtures in the window. Run `update` first.")
        return 1

    betway = None
    if args.prices:
        with open(args.prices) as fh:
            betway = json.load(fh)

    cfg = ConvictionConfig()
    if args.min_prob is not None:
        cfg.tiers = {k: max(v, args.min_prob) for k, v in cfg.tiers.items()}
    if args.real_prices_only:
        cfg.require_market_price = True

    picks, priced, watch = pipeline.todays_picks(model, cal, fixtures, betway, cfg)

    print(f"{len(fixtures)} fixtures scanned over the next {args.days} day(s).")
    print(f"{len(picks)} selection(s) cleared every conviction gate "
          f"at a real quoted price.")
    if watch and not args.real_prices_only:
        print(f"{len(watch)} more cleared every gate except price "
              f"(no quote in the free feed - check these on Betway).")
    print()

    for p in picks:
        print(p.describe()); print()

    if watch and not args.real_prices_only:
        _hr("WATCHLIST - confident, but you must check the Betway price")
        print("These passed the confidence, evidence and agreement gates. The only")
        print("thing not verified is the price, because the free feed does not quote")
        print("these markets. Look each one up: if Betway's price is at or above")
        print("the stated minimum, it clears the same edge bar as the picks above.\n")
        top = {round(w.prob, 3) for w in watch[:args.max_watch]}
        if len(top) == 1:
            print("Note: these all show the same probability because calibration")
            print("caps out there - the historical record cannot reliably tell")
            print("apart anything above about 92%, so it declines to pretend it")
            print("can. Their ordering among themselves is not meaningful.\n")
        for w in watch[:args.max_watch]:
            print(w.describe()); print()
        if len(watch) > args.max_watch:
            print(f"    ... and {len(watch)-args.max_watch} more "
                  f"(raise --max-watch to see them)\n")

        # Build the target plan the watchlist could support IF the prices are
        # there. Stated conditionally on purpose: these are not quotes.
        from .conviction import Pick as _Pick
        proxy = [_Pick(match_id=w.match_id, date=w.date, div=w.div, home=w.home,
                       away=w.away, market=w.market, selection=w.selection,
                       prob=w.prob, prob_raw=w.prob, price=w.min_price,
                       fair_price=w.fair_price, edge=0.0, ev=w.prob * w.min_price - 1,
                       tier=w.tier, disagreement=w.disagreement, bin_n=w.bin_n,
                       bin_rate=w.bin_rate, kelly=0.0, price_is_estimate=True)
                 for w in watch]
        cond = build_plan(proxy, target=args.target, max_legs=args.max_legs)
        _hr(f"IF YOU CAN GET THOSE PRICES - route to {args.target:.2f}x")
        if cond is None:
            import math as _math
            med = float(np.median([w.min_price for w in watch])) if watch else 0.0
            need = int(_math.ceil(_math.log(args.target) / _math.log(med))) if med > 1.001 else 0
            print(f"No combination of at most {args.max_legs} legs reaches "
                  f"{args.target:.2f}x, even at the minimum acceptable prices.")
            if need:
                p_all = 0.92 ** need
                print(f"At a typical {med:.2f} per leg you would need about {need} legs,")
                print(f"which lands roughly {p_all*100:.0f}% of the time before allowing")
                print(f"for the model running optimistic. Raise --max-legs to {need} to")
                print(f"see that plan, but understand what you are buying: each extra")
                print(f"leg multiplies in another chance to lose the whole stake.")
        else:
            print("Conditional on Betway actually offering at least the stated")
            print("minimum on every leg. These are requirements, not quotes.\n")
            for legs, share, odds, pp in zip(cond.legs, cond.stake_share,
                                             cond.combined_odds, cond.leg_probs):
                print(f"  Stake {args.currency}{args.bankroll*share:,.2f} "
                      f"@ {odds:.2f} -> {args.currency}{args.bankroll*share*odds:,.2f}")
                for lg in legs:
                    print(f"    - {lg.home} v {lg.away}: {lg.selection} "
                          f"(need {lg.price:.2f}+)")
            print(f"\n  Chance all legs land: {cond.p_double*100:.1f}%")
            print(f"  Chance you lose the stake: {(1-cond.p_double)*100:.1f}%")
            print("\n  Backtest note: selections like these landed about 4.8 points")
            print("  below their stated probability, so treat the figure above as")
            print("  optimistic. See MATCH_PREDICTOR.md for the measured numbers.")
        print()

    if not picks:
        print("No fully-priced selection qualifies today. That is a result, not a")
        print("failure - the gates exist so that a thin card produces silence")
        print("rather than a manufactured opinion.")
        if watch:
            print("Enter real Betway prices with --prices to turn watchlist entries")
            print("into gated picks with a staking plan.")
        return 0

    _hr(f"STAKING PLAN - target {args.target:.2f}x")
    plan = build_plan(picks, target=args.target, max_legs=args.max_legs)
    if plan is None:
        print(f"No combination of today's picks reaches {args.target:.2f}x.")
        print("Either the prices are too short or too few selections qualified.")
    else:
        print(plan.describe(stake=args.bankroll, currency=args.currency))

    _hr("TARGET LADDER")
    print("  target   shape    legs   chance of hitting   expected return")
    print("  " + "-" * 62)
    for pl in ladder(picks, max_legs=args.max_legs):
        print(f"  {pl.target:>5.1f}x   {pl.kind:<7} {pl.n_legs:>4}   "
              f"{pl.p_double*100:>15.1f}%   {(pl.ev-1)*100:>+13.1f}%")
    return 0


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="matchpredictor",
                                 description="Calibrated high-conviction football match predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("update", help="download results and fixtures, rebuild features")
    p.add_argument("--leagues", default=None, help="comma-separated division codes")
    p.add_argument("--start-year", type=int, default=2005)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--no-refresh", action="store_true", help="use cache only")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("train", help="fit the model and calibration curves")
    p.add_argument("--calib-seasons", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("calibration", help="print reliability tables")
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser("backtest", help="walk-forward validation")
    p.add_argument("--seasons", default=None, help="comma-separated season codes")
    p.add_argument("--n-seasons", type=int, default=8)
    p.add_argument("--min-prob", type=float, default=None)
    p.add_argument("--min-edge", type=float, default=None)
    p.add_argument("--save", default=None)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("slip", help="today's conviction picks and staking plan")
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--league", default=None)
    p.add_argument("--target", type=float, default=2.0)
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--currency", default="R")
    p.add_argument("--max-legs", type=int, default=8)
    p.add_argument("--min-prob", type=float, default=None)
    p.add_argument("--prices", default=None,
                   help="JSON file: {match_id: {market_key: betway_decimal_odds}}")
    p.add_argument("--real-prices-only", action="store_true",
                   help="hide the watchlist; show only fully-priced picks")
    p.add_argument("--max-watch", type=int, default=15,
                   help="how many watchlist entries to print")
    p.set_defaults(func=cmd_slip)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
