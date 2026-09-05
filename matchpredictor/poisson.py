"""Score-grid mathematics.

Every market this system prices comes from one object: the joint probability
distribution over the final score, P(home goals = x, away goals = y). Deriving
1X2, over/under, BTTS, handicaps and correct score from a single grid guarantees
they are mutually consistent - you can never end up recommending "over 2.5" and
"0-0" at the same time, which is the classic failure of models that price each
market independently.

The base is a double Poisson with the Dixon-Coles low-score correction, which
fixes the known flaw that independent Poissons underestimate 0-0, 1-1 and
1-0/0-1 relative to reality.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

DEFAULT_MAX_GOALS = 10


def poisson_pmf_matrix(lam: np.ndarray, max_goals: int = DEFAULT_MAX_GOALS) -> np.ndarray:
    """P(K = k) for k in 0..max_goals, vectorized over an array of rates.

    Returns shape (n, max_goals+1). The top cell absorbs the tail so each row
    sums to exactly 1 - important, because a leaked 0.3% of probability mass
    would quietly bias every derived market.
    """
    lam = np.asarray(lam, dtype=float).reshape(-1)
    lam = np.clip(lam, 1e-6, 12.0)
    k = np.arange(max_goals + 1)
    logp = -lam[:, None] + k[None, :] * np.log(lam[:, None]) - gammaln(k + 1)[None, :]
    p = np.exp(logp)
    p[:, -1] += np.clip(1.0 - p.sum(axis=1), 0.0, None)
    return p / p.sum(axis=1, keepdims=True)


def dixon_coles_grid(lam_h: np.ndarray, lam_a: np.ndarray, rho: np.ndarray | float = -0.05,
                     max_goals: int = DEFAULT_MAX_GOALS) -> np.ndarray:
    """Joint score distribution with the Dixon-Coles dependency correction.

    Independent Poissons get low scores wrong: real football produces more 0-0
    and 1-1 draws than independence implies, because teams shut up shop. The tau
    correction reweights exactly the four cells where that matters.

    Returns shape (n, max_goals+1, max_goals+1), indexed [match, home, away].
    """
    lam_h = np.asarray(lam_h, dtype=float).reshape(-1)
    lam_a = np.asarray(lam_a, dtype=float).reshape(-1)
    n = lam_h.shape[0]
    rho = np.full(n, float(rho)) if np.isscalar(rho) else np.asarray(rho, dtype=float).reshape(-1)

    ph = poisson_pmf_matrix(lam_h, max_goals)
    pa = poisson_pmf_matrix(lam_a, max_goals)
    grid = ph[:, :, None] * pa[:, None, :]

    # tau correction on the 2x2 low-score block.
    lh, la = np.clip(lam_h, 1e-6, 12.0), np.clip(lam_a, 1e-6, 12.0)
    # Keep every multiplier positive; rho outside this range makes the pmf invalid.
    safe = np.minimum.reduce([
        np.where(lh * la > 0, 1.0 / np.maximum(lh * la, 1e-9), 10.0),
        np.where(lh > 0, 1.0 / np.maximum(lh, 1e-9), 10.0),
        np.where(la > 0, 1.0 / np.maximum(la, 1e-9), 10.0),
        np.ones(n),
    ])
    r = np.clip(rho, -safe * 0.98, 0.98)

    grid[:, 0, 0] *= (1.0 - lh * la * r)
    grid[:, 0, 1] *= (1.0 + lh * r)
    grid[:, 1, 0] *= (1.0 + la * r)
    grid[:, 1, 1] *= (1.0 - r)

    grid = np.clip(grid, 1e-15, None)
    return grid / grid.sum(axis=(1, 2), keepdims=True)


# --------------------------------------------------------------------------
# Outcome masks, built once and reused.
# --------------------------------------------------------------------------
def _outcome_masks(max_goals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(max_goals + 1)[:, None]
    y = np.arange(max_goals + 1)[None, :]
    return (x > y), (x == y), (x < y)


def grid_1x2(grid: np.ndarray) -> np.ndarray:
    """Collapse a score grid into [P(home), P(draw), P(away)]."""
    mg = grid.shape[1] - 1
    mh, md, ma = _outcome_masks(mg)
    return np.stack([
        (grid * mh).sum(axis=(1, 2)),
        (grid * md).sum(axis=(1, 2)),
        (grid * ma).sum(axis=(1, 2)),
    ], axis=1)


def grid_over(grid: np.ndarray, line: float) -> np.ndarray:
    """P(total goals > line). Only half-lines are used, so no push handling."""
    mg = grid.shape[1] - 1
    tot = np.arange(mg + 1)[:, None] + np.arange(mg + 1)[None, :]
    return (grid * (tot > line)).sum(axis=(1, 2))


# --------------------------------------------------------------------------
# Market inversion: prices -> implied goal expectations
# --------------------------------------------------------------------------
def invert_market_lambdas(p_h: np.ndarray, p_a: np.ndarray,
                          p_over25: np.ndarray | None = None,
                          rho: float = -0.05,
                          max_goals: int = DEFAULT_MAX_GOALS,
                          iters: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Recover the (lambda_home, lambda_away) that reproduce the market's prices.

    This is the single highest-value trick in the whole system. The de-vigged
    closing price is the sharpest free estimate of a match outcome in existence -
    it aggregates every professional opinion and every piece of team news. But a
    1X2 price only tells you about three outcomes. By solving for the goal
    expectations that generate those prices, we recover the market's *entire*
    implied score distribution, and can then price markets the market never
    quoted, and compare like with like.

    Solved by damped Newton iteration on (log lambda_h, log lambda_a),
    vectorized across all matches at once.
    """
    p_h = np.asarray(p_h, dtype=float).reshape(-1)
    p_a = np.asarray(p_a, dtype=float).reshape(-1)
    n = p_h.shape[0]

    valid = np.isfinite(p_h) & np.isfinite(p_a) & (p_h > 1e-4) & (p_a > 1e-4) & (p_h + p_a < 0.999)
    lh = np.full(n, np.nan)
    la = np.full(n, np.nan)
    if not valid.any():
        return lh, la

    ph_v, pa_v = p_h[valid], p_a[valid]
    use_ou = p_over25 is not None
    if use_ou:
        po = np.asarray(p_over25, dtype=float).reshape(-1)[valid]
        ou_ok = np.isfinite(po)
    else:
        po = None
        ou_ok = np.zeros(ph_v.shape[0], dtype=bool)

    # Warm start: total goals from the over/under price where available,
    # supremacy from the 1X2 balance. A good start halves the iteration count.
    total0 = np.full(ph_v.shape[0], 2.6)
    if use_ou:
        # Monotone map from P(over 2.5) to a plausible total-goals expectation.
        total0 = np.where(ou_ok, 2.55 + 3.2 * (np.clip(po, 0.05, 0.95) - 0.5), total0)
    total0 = np.clip(total0, 1.4, 5.2)
    sup0 = np.clip(1.1 * (ph_v - pa_v), -1.6, 1.6)
    u = np.log(np.clip(np.stack([(total0 + sup0) / 2, (total0 - sup0) / 2], axis=1), 0.12, 6.0))

    eps = 1e-4

    def residual(uu: np.ndarray) -> np.ndarray:
        g = dixon_coles_grid(np.exp(uu[:, 0]), np.exp(uu[:, 1]), rho, max_goals)
        m = grid_1x2(g)
        r = np.stack([m[:, 0] - ph_v, m[:, 2] - pa_v], axis=1)
        if use_ou:
            ro = np.where(ou_ok, grid_over(g, 2.5) - np.nan_to_num(po), 0.0)
            # Fold the over/under constraint in as a soft pull on the total.
            r = r + 0.35 * np.stack([ro, ro], axis=1)
        return r

    for _ in range(iters):
        r0 = residual(u)
        if np.nanmax(np.abs(r0)) < 1e-6:
            break
        # Numerical Jacobian (2x2 per match) - cheap because everything is batched.
        du0 = u.copy(); du0[:, 0] += eps
        du1 = u.copy(); du1[:, 1] += eps
        j0 = (residual(du0) - r0) / eps
        j1 = (residual(du1) - r0) / eps
        a, b = j0[:, 0], j1[:, 0]
        c, d = j0[:, 1], j1[:, 1]
        det = a * d - b * c
        det = np.where(np.abs(det) < 1e-9, np.sign(det) * 1e-9 + 1e-12, det)
        step0 = (d * r0[:, 0] - b * r0[:, 1]) / det
        step1 = (-c * r0[:, 0] + a * r0[:, 1]) / det
        step = np.clip(np.stack([step0, step1], axis=1), -0.6, 0.6)  # damping
        u = np.clip(u - step, np.log(0.06), np.log(7.0))

    lh[valid] = np.exp(u[:, 0])
    la[valid] = np.exp(u[:, 1])
    return lh, la


def estimate_rho(hg: np.ndarray, ag: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray,
                 max_goals: int = DEFAULT_MAX_GOALS,
                 grid_pts: int = 41) -> float:
    """Pick the Dixon-Coles rho that best explains observed low scores.

    A simple 1-D likelihood scan; rho is a single global nuisance parameter so
    there is no need for anything fancier.
    """
    ok = np.isfinite(hg) & np.isfinite(ag) & np.isfinite(lam_h) & np.isfinite(lam_a)
    if ok.sum() < 200:
        return -0.05
    hg_i = np.clip(hg[ok].astype(int), 0, max_goals)
    ag_i = np.clip(ag[ok].astype(int), 0, max_goals)
    lh, la = lam_h[ok], lam_a[ok]

    best_rho, best_ll = -0.05, -np.inf
    for rho in np.linspace(-0.25, 0.15, grid_pts):
        g = dixon_coles_grid(lh, la, rho, max_goals)
        ll = np.log(np.clip(g[np.arange(len(hg_i)), hg_i, ag_i], 1e-15, None)).sum()
        if ll > best_ll:
            best_ll, best_rho = ll, float(rho)
    return best_rho
