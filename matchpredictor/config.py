"""Central configuration: leagues, seasons, model hyper-parameters, thresholds."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Data source
# --------------------------------------------------------------------------
# football-data.co.uk publishes one CSV per league per season, free, no API key.
# Each row carries the result, detailed match stats (shots, shots on target,
# corners, cards, half-time score) and closing odds from several bookmakers
# including Pinnacle - which is the sharpest publicly available price and the
# single most useful input this system has.
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Division code -> human readable name.
LEAGUES: dict[str, str] = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "E2": "England League One",
    "E3": "England League Two",
    "SC0": "Scotland Premiership",
    "D1": "Germany Bundesliga",
    "D2": "Germany 2. Bundesliga",
    "I1": "Italy Serie A",
    "I2": "Italy Serie B",
    "SP1": "Spain La Liga",
    "SP2": "Spain Segunda",
    "F1": "France Ligue 1",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium Pro League",
    "P1": "Portugal Liga",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}

# The leagues we model by default. Deep, liquid, well-covered by the data feed.
DEFAULT_LEAGUES: list[str] = [
    "E0", "E1", "E2", "E3", "SC0",
    "D1", "D2", "I1", "I2", "SP1", "SP2",
    "F1", "F2", "N1", "B1", "P1", "T1", "G1",
]


def season_codes(start_year: int, end_year: int) -> list[str]:
    """Build football-data season codes, e.g. 2024/25 -> '2425'."""
    out = []
    for y in range(start_year, end_year + 1):
        out.append(f"{y % 100:02d}{(y + 1) % 100:02d}")
    return out


# Seasons to pull. 2005 onwards keeps us in the era of reliable odds + shot data.
DEFAULT_SEASON_START = 2005
DEFAULT_SEASON_END = 2026  # 2026/27, the season currently in play

CACHE_DIR = os.environ.get(
    "MATCHPREDICTOR_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mpcache"),
)


# --------------------------------------------------------------------------
# Model configuration
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Hyper-parameters for the rating / goal-expectation models."""

    # Elo
    elo_start: float = 1500.0
    elo_k: float = 20.0
    elo_home_advantage: float = 65.0
    elo_mov_factor: float = 1.0          # margin-of-victory scaling
    elo_season_regression: float = 0.25  # pull toward league mean between seasons

    # Exponentially weighted team strength
    form_halflife_days: float = 180.0    # how fast old matches stop mattering
    form_min_matches: int = 4            # below this, lean on priors/market

    # Dixon-Coles
    dc_halflife_days: float = 180.0
    dc_max_goals: int = 12               # score grid dimension
    dc_refit_every_days: int = 14
    dc_min_history: int = 300            # matches needed before DC is trusted

    # Ensemble blending weights for expected goals.
    # The market is genuinely hard to beat, so it gets the largest single share.
    # These are starting values; `blend.py` can fit them on out-of-sample data.
    w_market: float = 0.60
    w_dixon_coles: float = 0.25
    w_ratings: float = 0.15

    # Shot-based expected goals proxy
    sot_conversion_prior: float = 0.315  # league-average goals per shot on target
    shot_weight: float = 0.35            # how much the shot proxy pulls the goal rate


@dataclass
class ConvictionConfig:
    """Thresholds that decide whether the engine is allowed to speak at all.

    The whole point of this project is silence by default. A pick is only
    emitted when every one of these gates is passed.
    """

    # Calibrated probability floors for each conviction tier.
    tiers: dict[str, float] = field(default_factory=lambda: {
        "LOCK": 0.90,       # "this is going to happen"
        "STRONG": 0.80,
        "LEAN": 0.70,
    })

    # A pick must also be better than the price implies, otherwise you are just
    # paying the bookmaker for the privilege of being right.
    min_edge: float = 0.02               # model prob - de-vigged fair prob
    min_edge_lock: float = 0.015         # locks may run on a thinner edge

    # Historical evidence gate: in backtest, the calibration bin this pick falls
    # into must have contained at least this many past bets, and must have hit at
    # least `bin_hitrate_tolerance` below the claimed probability.
    min_bin_samples: int = 150
    bin_hitrate_tolerance: float = 0.04

    # Model agreement gate: the components (market, Dixon-Coles, ratings) must
    # not disagree wildly. Measured as max pairwise probability gap.
    max_component_disagreement: float = 0.18

    # Never bet a price so short that the risk/reward is absurd, and never chase
    # a long shot no matter how much "value" the model claims.
    min_odds: float = 1.10
    max_odds: float = 6.00

    # Liquidity / data quality gates
    min_matches_played: int = 4          # both teams need some season history
    require_market_price: bool = True    # no price, no bet


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_CONVICTION_CONFIG = ConvictionConfig()
