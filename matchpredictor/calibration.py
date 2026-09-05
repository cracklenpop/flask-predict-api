"""Calibration - the difference between a model that sounds confident and a
model whose confidence means something.

A raw model output of 0.90 is just a number. Calibration asks the only question
that matters: historically, when this system said 0.90, how often did it
actually happen? If the answer is 0.78, the model is overconfident and every
"lock" it emits is a slow leak of money.

Two things are produced here:

1. An isotonic mapping per market family that corrects systematic over- or
   under-confidence. Isotonic is used rather than Platt scaling because the
   distortion is not a neat sigmoid - models are typically well behaved in the
   middle and badly overconfident in the tails, which is precisely the region
   this whole system lives in.

2. Bin evidence: for each probability band, how many past bets landed in it and
   what fraction won. The conviction engine refuses to emit a pick whose band
   has too little history or a track record worse than its claim. That is the
   mechanism that stops the bot inventing certainty.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Bands used for the evidence gate. Fine-grained where it matters (the high end).
EVIDENCE_BINS = np.array([0.0, 0.5, 0.6, 0.7, 0.75, 0.80, 0.84, 0.87, 0.90,
                          0.92, 0.94, 0.96, 0.98, 1.01])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class FamilyCalibration:
    """Calibration state for one market family.

    Two stages, and the first one matters more than it looks.

    STAGE 1 - STACKING. A logistic regression over [logit(model), logit(market)]
    learns how much the model should be trusted *relative to the sharp price*.
    This exists because of a trap that quietly destroys value-betting systems: a
    model can be perfectly calibrated on average and still be badly overconfident
    on exactly the subset it bets. Bets are selected where model and market
    disagree most, and disagreement is precisely where the model is most likely
    to be the one that is wrong. Marginal calibration cannot see this, because
    the disagreement cases are a small slice averaged in with everything else.
    Stacking fixes it at the root by making the market an explicit input, so the
    output shrinks toward the price exactly when the two diverge.

    STAGE 2 - ISOTONIC. A monotone map applied on top, catching whatever
    systematic over- or under-confidence survives stage 1.
    """

    family: str
    iso: IsotonicRegression | None = None
    stacker: LogisticRegression | None = None
    n_train: int = 0
    bin_counts: np.ndarray = field(default_factory=lambda: np.zeros(len(EVIDENCE_BINS) - 1))
    bin_hits: np.ndarray = field(default_factory=lambda: np.zeros(len(EVIDENCE_BINS) - 1))
    brier_raw: float = float("nan")
    brier_cal: float = float("nan")
    logloss_raw: float = float("nan")
    logloss_cal: float = float("nan")

    def transform(self, p: np.ndarray, p_mkt: np.ndarray | None = None) -> np.ndarray:
        p_in = np.asarray(p, dtype=float)
        scalar = p_in.ndim == 0
        p_in = np.atleast_1d(p_in)

        # Missing predictions must stay missing. np.clip leaves NaN as NaN, and
        # a NaN reaching the stacker raises rather than returning something
        # sensible - so finite rows are transformed and the rest pass straight
        # through. Callers rely on NaN meaning "this market does not apply to
        # this match", never "probability zero".
        finite = np.isfinite(p_in)
        out = np.full(p_in.shape, np.nan)
        if not finite.any() or self.n_train < 200:
            res = np.where(finite, np.clip(p_in, 1e-6, 1 - 1e-6), np.nan)
            return float(res[0]) if scalar else res

        q = np.clip(p_in[finite], 1e-6, 1 - 1e-6)

        if self.stacker is not None and p_mkt is not None:
            pm = np.atleast_1d(np.asarray(p_mkt, dtype=float))[finite]
            # Where no market price exists, fall back to the model's own number
            # so the stacker sees a consistent pair rather than a hole.
            pm = np.clip(np.where(np.isfinite(pm), pm, q), 1e-6, 1 - 1e-6)
            X = np.column_stack([_logit(q), _logit(pm)])
            q = np.clip(self.stacker.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)

        if self.iso is not None:
            q = self.iso.predict(q)

        out[finite] = np.clip(q, 1e-4, 1 - 1e-4)
        return float(out[0]) if scalar else out

    def evidence(self, p: float) -> tuple[int, float]:
        """(sample count, empirical hit rate) for the band this probability falls in."""
        i = int(np.clip(np.digitize([p], EVIDENCE_BINS)[0] - 1, 0, len(self.bin_counts) - 1))
        n = int(self.bin_counts[i])
        rate = float(self.bin_hits[i] / n) if n > 0 else float("nan")
        return n, rate

    def to_dict(self) -> dict:
        d = {
            "family": self.family, "n_train": self.n_train,
            "bin_counts": self.bin_counts.tolist(), "bin_hits": self.bin_hits.tolist(),
            "brier_raw": self.brier_raw, "brier_cal": self.brier_cal,
            "logloss_raw": self.logloss_raw, "logloss_cal": self.logloss_cal,
        }
        if self.iso is not None:
            d["iso_x"] = np.asarray(self.iso.X_thresholds_).tolist()
            d["iso_y"] = np.asarray(self.iso.y_thresholds_).tolist()
        if self.stacker is not None:
            d["stack_coef"] = np.asarray(self.stacker.coef_).tolist()
            d["stack_intercept"] = np.asarray(self.stacker.intercept_).tolist()
        return d

    @staticmethod
    def from_dict(d: dict) -> "FamilyCalibration":
        fc = FamilyCalibration(family=d["family"], n_train=int(d["n_train"]))
        fc.bin_counts = np.asarray(d["bin_counts"], dtype=float)
        fc.bin_hits = np.asarray(d["bin_hits"], dtype=float)
        for k in ("brier_raw", "brier_cal", "logloss_raw", "logloss_cal"):
            setattr(fc, k, float(d.get(k, float("nan"))))
        if "iso_x" in d:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(np.asarray(d["iso_x"], dtype=float), np.asarray(d["iso_y"], dtype=float))
            fc.iso = iso
        if "stack_coef" in d:
            lr = LogisticRegression()
            lr.coef_ = np.asarray(d["stack_coef"], dtype=float)
            lr.intercept_ = np.asarray(d["stack_intercept"], dtype=float)
            lr.classes_ = np.array([0, 1])
            fc.stacker = lr
        return fc


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class CalibratorSet:
    """Isotonic calibration for every market family, plus the evidence tables."""

    def __init__(self):
        self.families: dict[str, FamilyCalibration] = {}

    def fit_family(self, family: str, p: np.ndarray, y: np.ndarray,
                   p_mkt: np.ndarray | None = None) -> FamilyCalibration:
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        pm = np.asarray(p_mkt, dtype=float) if p_mkt is not None else None

        ok = np.isfinite(p) & np.isfinite(y)
        if pm is not None:
            ok &= np.isfinite(pm)
        p, y = p[ok], y[ok]
        pm = pm[ok] if pm is not None else None

        fc = FamilyCalibration(family=family, n_train=len(p))
        p_raw_for_metrics = p.copy()
        if len(p) >= 200:
            if pm is not None:
                lr = LogisticRegression(C=1.0, max_iter=1000)
                lr.fit(np.column_stack([_logit(p), _logit(pm)]), (y > 0.5).astype(int))
                fc.stacker = lr
                stage1 = np.clip(lr.predict_proba(
                    np.column_stack([_logit(p), _logit(pm)]))[:, 1], 1e-6, 1 - 1e-6)
            else:
                stage1 = p

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(stage1, y)
            fc.iso = iso
            pc = fc.transform(p, pm)
            fc.brier_raw, fc.brier_cal = _brier(p_raw_for_metrics, y), _brier(pc, y)
            fc.logloss_raw, fc.logloss_cal = _logloss(p_raw_for_metrics, y), _logloss(pc, y)
        else:
            pc = p

        # Evidence table is built on CALIBRATED probabilities, because that is
        # what the conviction engine will be checking against.
        idx = np.clip(np.digitize(pc, EVIDENCE_BINS) - 1, 0, len(fc.bin_counts) - 1)
        for i in range(len(fc.bin_counts)):
            m = idx == i
            fc.bin_counts[i] = int(m.sum())
            fc.bin_hits[i] = float(y[m].sum())

        self.families[family] = fc
        return fc

    def transform(self, family: str, p: np.ndarray,
                  p_mkt: np.ndarray | None = None) -> np.ndarray:
        fc = self.families.get(family)
        if fc is None:
            return np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        return fc.transform(p, p_mkt)

    def evidence(self, family: str, p: float) -> tuple[int, float]:
        fc = self.families.get(family)
        if fc is None:
            return 0, float("nan")
        return fc.evidence(p)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({k: v.to_dict() for k, v in self.families.items()}, fh, indent=1)

    @staticmethod
    def load(path: str) -> "CalibratorSet":
        cs = CalibratorSet()
        with open(path) as fh:
            raw = json.load(fh)
        cs.families = {k: FamilyCalibration.from_dict(v) for k, v in raw.items()}
        return cs

    # ------------------------------------------------------------------
    def reliability_report(self) -> str:
        """Human-readable calibration table - the receipt for every claim made."""
        lines = []
        for fam in sorted(self.families):
            fc = self.families[fam]
            lines.append(f"\n  {fam}  (n={fc.n_train:,}   "
                         f"Brier {fc.brier_raw:.4f} -> {fc.brier_cal:.4f}   "
                         f"LogLoss {fc.logloss_raw:.4f} -> {fc.logloss_cal:.4f})")
            lines.append("    band          n      predicted   actual    gap")
            for i in range(len(EVIDENCE_BINS) - 1):
                n = int(fc.bin_counts[i])
                if n == 0:
                    continue
                lo, hi = EVIDENCE_BINS[i], EVIDENCE_BINS[i + 1]
                actual = fc.bin_hits[i] / n
                mid = (lo + min(hi, 1.0)) / 2
                lines.append(f"    {lo:.2f}-{min(hi,1.0):.2f}  {n:8,d}      "
                             f"{mid:6.3f}   {actual:6.3f}  {actual-mid:+6.3f}")
        return "\n".join(lines)
