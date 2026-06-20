"""Label-free change-point baselines on the embedding stream.

Two methods, both run on a 1-D projection of the per-clip embedding series
(e.g. PC1, or the norm of the difference between consecutive embeddings):

  BOCPD  (Adams & MacKay 2007)  online, real-time onset alarm. Self-contained
         Gaussian implementation below (no external dependency).
  BEAST  (Zhao 2019; Kuluwan 2023)  offline Bayesian decomposition. Thin wrapper
         over the `Rbeast` package: pip install Rbeast.

These need NO event labels, so they de-risk the project under label scarcity and
directly support the thesis that change detection beats a fixed threshold.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# BOCPD: self-contained Gaussian (Normal-Inverse-Gamma) implementation.
# --------------------------------------------------------------------------- #
def bocpd_gaussian(x: np.ndarray, hazard_lambda: float = 240.0,
                   mu0: float = 0.0, kappa0: float = 1.0,
                   alpha0: float = 1.0, beta0: float = 1.0) -> dict:
    """Run BOCPD with a Gaussian (Normal-Inverse-Gamma) observation model.

    Returns a dict with, for each time step:
      expected_run_length  E[r_t] ; drops sharply at a change.
      map_run_length        argmax_r P(r_t) ; resets to ~0 at a change.
      p_r0                  P(r_t = 0) ; kept for reference, but note it pins to
                            the hazard rate when mass sits on one run length, so
                            prefer the run-length signal for detection.

    x: 1-D stream (e.g. PC1 of the embedding series). hazard_lambda is the
    expected segment length in samples (e.g. ~10 days in hours -> 240).
    Use changepoints_from_runlength(result['map_run_length']) to get indices.
    """
    n = len(x)
    H = 1.0 / hazard_lambda
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1.0
    mu = np.array([mu0]); kappa = np.array([kappa0])
    alpha = np.array([alpha0]); beta = np.array([beta0])
    p_r0 = np.zeros(n)
    erl = np.zeros(n)
    map_rl = np.zeros(n, dtype=int)

    for t in range(1, n + 1):
        xt = x[t - 1]
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        df = 2 * alpha
        z = (xt - mu) / scale
        pred = np.exp(_student_logpdf(z, df)) / scale

        growth = R[:t, t - 1] * pred * (1 - H)
        cp = np.sum(R[:t, t - 1] * pred * H)
        R[1:t + 1, t] = growth
        R[0, t] = cp
        R[:, t] /= R[:, t].sum() + 1e-300

        post = R[:t + 1, t]
        p_r0[t - 1] = post[0]
        erl[t - 1] = float(np.sum(np.arange(t + 1) * post))
        map_rl[t - 1] = int(post.argmax())

        # NIG sufficient-statistic update. Compute *_new from the OLD arrays,
        # then prepend the r=0 prior. Order matters: beta_new uses old mu/kappa.
        mu_new = (kappa * mu + xt) / (kappa + 1)
        beta_new = beta + (kappa * (xt - mu) ** 2) / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa + 1])
        alpha = np.concatenate([[alpha0], alpha + 0.5])
        beta = np.concatenate([[beta0], beta_new])

    return {"expected_run_length": erl, "map_run_length": map_rl, "p_r0": p_r0}


def changepoints_from_runlength(map_run_length: np.ndarray, min_drop: int = 24,
                                min_sep: int = 48) -> list[int]:
    """Return indices where the MAP run length drops by at least `min_drop`
    (a reset), enforcing a minimum separation of `min_sep` samples."""
    drops = []
    last = -10 ** 9
    for t in range(1, len(map_run_length)):
        if map_run_length[t - 1] - map_run_length[t] >= min_drop and (t - last) >= min_sep:
            drops.append(t)
            last = t
    return drops


from math import lgamma, pi

_gammaln = np.vectorize(lgamma)   # math.lgamma is scalar-only; vectorize for arrays


def _student_logpdf(z, df):
    # log pdf of standard Student-t with df degrees of freedom, evaluated at z.
    c = _gammaln((df + 1) / 2) - _gammaln(df / 2) - 0.5 * np.log(df * pi)
    return c - (df + 1) / 2 * np.log(1 + z ** 2 / df)


def beast_changepoints(x: np.ndarray, season_period: int = 0):
    """Offline BEAST decomposition. Returns the trend-changepoint probability.

    Requires: pip install Rbeast
    """
    import Rbeast as rb  # noqa: N813
    o = rb.beast(x, season="none" if season_period == 0 else "harmonic",
                 period=season_period if season_period else None)
    return o  # inspect o.trend.cpOccPr for per-time change probability


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(4, 1, 300)])
    res = bocpd_gaussian(x, hazard_lambda=200)
    cps = changepoints_from_runlength(res["map_run_length"], min_drop=50, min_sep=48)
    print("detected change-points:", cps, "(true = 300)")
