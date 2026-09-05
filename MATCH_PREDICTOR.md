# Match Predictor

A calibrated, high-conviction football match prediction engine, built to bet on
Betway.

The goal it was built for: *only produce a selection when the evidence is
overwhelming, and give an honest number for how overwhelming it actually is.*

---

## The idea

An experienced bettor sometimes just knows a result is coming. They have watched
the teams, they know who is missing, they know one side is quietly better than
its league position. That instinct is real, and it is built from evidence — but
it is unaudited. Nobody writes down how often the feeling was right.

This engine tries to reproduce that instinct from evidence rather than feel, and
then does the thing a human cannot: it checks itself. Every probability it
reports has been measured against what actually happened the last time it said
the same thing.

There is one thing it will not do, and it is worth being direct about it: it
will not label anything a 100% guarantee. Not because of caution, but because
the label would destroy the tool's usefulness. If everything is stamped
"guaranteed", you cannot tell a genuine 93% from a wishful 60% — and the whole
point of this system is that it *can* tell them apart, and stays silent on the
second one. What it gives you instead is a tier (`LOCK` / `STRONG` / `LEAN`), a
calibrated probability, and the historical hit rate of every previous bet that
looked like this one.

---

## How it forms an opinion

Everything is priced from a single object: the joint distribution over the final
score, `P(home goals = x, away goals = y)`. Every market — match result, double
chance, over/under, both teams to score, handicaps, correct score, and the
combination markets — is a different summation over that same grid. That is what
makes the outputs mutually consistent. The engine physically cannot tell you
"Over 2.5 is a lock" and "0-0 is likely" in the same breath.

The grid comes from expected goals for each side, and those come from blending
three independent opinions:

| Component | What it is | Why it is there |
|---|---|---|
| **Ratings** | Classical Poisson strength model — league scoring rate × home attack × away defence, corrected by a shots-on-target proxy and Elo supremacy | Transparent, stable, completely independent of the market |
| **Market** | Goal expectations recovered from de-vigged pre-match prices by numerically inverting the score grid | The sharpest single signal that exists; very hard to beat |
| **Learned** | Two gradient-boosted Poisson regressions over ~60 features | Picks up the non-linear interactions the other two miss |

The market component deserves a note. A 1X2 price only tells you about three
outcomes. By solving for the goal expectations that *reproduce* those prices,
the engine recovers the market's entire implied score distribution — and can
then price markets the bookmaker never quoted, and compare like with like.

### What the features actually look at

Team strength split by venue; time-decayed attack and defence rates (a match six
months old counts half as much as one today); a shots-on-target expected-goals
proxy; **finishing luck** — a side scoring far above its chance creation is
usually about to stop, which is exactly where naive "hot form" models get
burned; rest days and fixture congestion; discipline; short-horizon streaks;
Elo with margin-of-victory scaling and between-season regression toward the
mean; and the de-vigged market price.

---

## How it decides to stay quiet

A selection is emitted only if it survives **every** gate. Any single failure
and it is dropped, however attractive it looked.

1. **Calibrated probability** clears the tier floor — not the raw model number,
   the one corrected by what actually happened last time the model said this.
2. **Historical evidence** — the probability band this pick sits in must have a
   real track record: at least 150 past samples, and an actual hit rate that
   lived up to the claim.
3. **Model agreement** — market, ratings and learned models must not diverge by
   more than 18 points. Confidence built on one component shouting over the
   other two is the confidence that gets punished.
4. **Edge** — the price must beat fair value. Being right at a bad price is a
   losing strategy; over enough bets it is indistinguishable from being wrong.
5. **Price sanity** — nothing shorter than 1.10, nothing longer than 6.00.

On most match days this produces a handful of selections, sometimes none. That
is the design working, not failing.

### The trap this system is built around

A model can be perfectly calibrated *on average* and still be badly
overconfident on exactly the subset it bets. Bets get selected where the model
and the market disagree most — and disagreement is precisely where the model is
most likely to be the one that is wrong. Ordinary calibration cannot see this,
because those cases are a thin slice averaged in with everything else.

Measured on this system, that effect was worth **4.6 percentage points** of
phantom confidence and turned a positive-looking edge negative.

The fix is *stacked calibration*: a logistic regression over
`[logit(model), logit(market)]`, so the market is an explicit input and the
output shrinks toward the price exactly when the two diverge. What it learned is
worth reading directly:

| Market family | Model weight | Market weight | Reading |
|---|---|---|---|
| Match result (1X2) | **−0.36** | **+1.36** | The model's deviations from the price are *counterproductive*. The market wins outright. |
| Double chance | −0.29 | +1.31 | Same. |
| Totals (over/under) | +0.23 | +0.77 | Market-dominant, model adds a little. |
| Both teams to score | **+1.38** | **−0.38** | Here the **model genuinely beats the market**. |

That last row is the interesting one. BTTS is a derived, less efficiently priced
market, and the shot-based features carry real information the price does not.
The engine is not equally good everywhere, and it now knows where it is good.

---

## Install

```bash
pip install -r requirements.txt
```

Only `flask numpy pandas scipy scikit-learn requests pyarrow` are needed for the
predictor. (`torch` and `stable-baselines3` belong to the unrelated PPO scalper
in `server.py`.)

## Use

```bash
# 1. Download results + upcoming fixtures, build features  (~2 min first run)
python -m matchpredictor update

# 2. Fit the model and the calibration curves               (~10 min)
python -m matchpredictor train

# 3. Today's conviction picks and a staking plan
python -m matchpredictor slip --days 2 --target 2.0 --bankroll 1000 --currency R

# See the receipts behind every claim
python -m matchpredictor calibration

# Re-run the validation yourself
python -m matchpredictor backtest --n-seasons 10
```

### Picks vs. the watchlist

The output has two sections, and the split matters.

**Picks** cleared every gate *including* the edge test, because a real quoted
price existed. These come with a staking plan.

**Watchlist** entries cleared every gate *except* price — the free data feed
only quotes match result and over/under 2.5, so for the other ~57 markets there
is no price to test against. Rather than invent one (a fabricated price gives a
fabricated edge, which is worse than no answer), each entry reports the number
you actually need:

```
[LOCK] Bayern Munich v Heidenheim (D1)
    Bayern Munich or Draw (Double Chance)
    model 94.2%   fair 1.06   TAKE ONLY AT 1.08 OR BETTER
    track record in this band: 94.6% over 1,240 past bets
```

Look it up on Betway. At or above `min_price`, it clears the same edge bar every
pick had to clear. Below it, walk away — being right at a bad price is still a
losing strategy.

### Feeding it your real Betway prices

Entering real prices promotes watchlist entries into fully-gated picks and lets
them into the staking plan:

```json
{
  "E0|2627|20260912|Arsenal|Chelsea": { "DC_1X": 1.24, "1X2_HOME": 1.55 }
}
```

```bash
python -m matchpredictor slip --prices my_betway_odds.json --real-prices-only
```

Market keys are listed in `matchpredictor/markets.py` (`ALL_MARKET_KEYS`).

### HTTP API

```bash
python match_server.py     # 0.0.0.0:10001
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | model status, history size, upcoming count |
| `GET /fixtures?days=2` | fixtures the engine can see |
| `GET /markets/<match_id>` | all 62 markets priced for one fixture + likeliest scorelines |
| `GET /slip?days=2&target=2.0` | conviction picks + watchlist + staking plan |
| `POST /slip` | same, with your real Betway prices in the body |
| `GET /calibration` | reliability tables |

---

## About doubling your money

The arithmetic is unavoidable and worth stating plainly: a selection you are 92%
sure of is priced around 1.10, and 1.10 does not double anything. To return 2×
you need combined odds of at least 2.00 — which means either one genuinely
uncertain bet, or several confident ones multiplied together.

Multiplying confident legs is the better trade, but it is still a chance, and
the engine's job is to report that chance rather than dress it up. It searches
three plan shapes — a single, the highest-probability accumulator that clears
the target, and a stake split across disjoint parlays so any one landing doubles
the bank — and always reports the honest number:

```
  target   shape    legs   chance of hitting   expected return
  --------------------------------------------------------------
    1.5x   PARLAY      2              84.2%           +28.4%
    2.0x   PARLAY      3              72.9%           +48.0%
    3.0x   PARLAY      5              56.5%           +93.6%
    5.0x   PARLAY      6              41.6%          +128.0%
```

A 72.9% chance of doubling is a genuinely strong position. It is also a **27.1%
chance of losing the stake**, and repeating it every week makes a losing run
close to certain at some point. `staking.simulate_season()` will show you that
distribution. Legs are always drawn from different fixtures — two selections on
the same match are correlated, and multiplying them as if they were independent
is the fastest way to turn a 60% plan into a 40% one without noticing.

---

## Validation

Backtesting football is easy to get wrong in ways that manufacture an edge. This
protocol is deliberately strict:

- **Expanding window.** To predict season *S*, the model is fitted only on
  matches played before season *S* began, and refitted from scratch each season.
- **Leak-free features.** Built in a single forward pass that emits a match's
  features *before* folding in that match's result. There is a test
  (`test_features_do_not_leak_the_result`) that changes the last match's score
  and asserts no feature moves.
- **Pre-match prices only.** Closing odds are never an input — you cannot bet at
  a price that does not exist yet.
- **Out-of-sample calibration.** Season *S* is calibrated using only seasons
  before it. Fitting the calibrator on the season being scored is the subtlest
  way to manufacture confidence that does not survive contact with reality.
- **Settled at recreational prices.** Bets settle at a Bet365/average pre-match
  price — the bracket Betway sits in — not at best-price-across-all-books.

Run `python tests/test_matchpredictor.py` to check the invariants (19 tests, no
pytest required).

### What the backtest cannot capture

**Betway will restrict or close an account that wins consistently.** This is the
single biggest gap between a backtest and reality, and no model fixes it.

Prices also move between when the feed captured them and when you would bet, the
data feed carries no team-news or injury information, and the historical results
above are a sample — a real edge of a few percent is indistinguishable from
noise over a few hundred bets. **Treat every ROI figure as an upper bound.**

---

## Layout

```
matchpredictor/
  config.py       leagues, hyper-parameters, conviction thresholds
  data.py         download/cache/normalize; signal vs bet vs closing prices
  features.py     leak-free forward pass: Elo, decayed form, xG proxy, market
  poisson.py      Dixon-Coles score grids; market-price -> lambda inversion
  markets.py      62 markets derived from one grid; settlement
  model.py        the three-component ensemble
  calibration.py  stacked + isotonic calibration, evidence bands
  conviction.py   the gates
  staking.py      the 2x target optimizer
  backtest.py     walk-forward validation
  pipeline.py     orchestration
  cli.py          command line
match_server.py   HTTP API
tests/            invariant tests
```

Data: [football-data.co.uk](https://www.football-data.co.uk) — free, no API key,
18 divisions, results + match stats + multi-bookmaker odds back to 2005.
