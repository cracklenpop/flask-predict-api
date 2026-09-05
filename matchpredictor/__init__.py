"""matchpredictor - a calibrated, high-conviction football match prediction engine.

The design goal is simple: replicate the thing a sharp human bettor feels when
they *know* a result is coming, but do it from evidence instead of feel, and
attach a number to it that has been checked against history.

Nothing in here will tell you a bet is 100% certain, because the backtester
would catch the lie immediately. What it does instead is refuse to speak at all
unless the evidence clears a high bar - and tell you exactly how often bets that
looked like this one have actually landed.
"""

__version__ = "1.0.0"
