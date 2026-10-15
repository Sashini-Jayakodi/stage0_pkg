"""Stage 6 engine: Hoeffding-Bentkus p-value, per-group track-level certifier, accept/escalate policy (plan sections 1, 6.3, Stage 6).

Pure numpy/scipy. No torch, no file reading, no knowledge of GTSRB: it takes arrays in and gives decisions out, so it can be
proved on synthetic data (src/certify_synth.py) BEFORE any real data is touched.

The promise (plan section 1): in each group k, averaged over tracks, the error rate among accepted frames is <= alpha,
with probability >= 1 - delta over the Cert draw. Here is how that is tested.

  per-track loss    l_t = (1/n_tk) * sum_{i in track t, group k} ( 1[accepted and wrong] - alpha * 1[accepted] )
  shifted loss      L_t = l_t + alpha, in [0, 1]. H0: E[L] >= alpha   (equivalently E[accepted-error share] >= alpha * E[accepted share])
  p-value (HB)      p = min( exp(-n h1(min(mu_hat, alpha), alpha)),  e * P(Bin(n, alpha) <= ceil(n * mu_hat)) ),   n = tracks with >= 1 frame in k
                    h1(a, b) = a log(a/b) + (1-a) log((1-a)/(1-b))
  fixed sequence    thresholds are tested in a FIT-derived order (plan 6.3: start at the Fit argmin of estimated E[L], then looser);
                    stop at the first p > delta_k. The certified threshold is the last one that passed before that.
  levels            delta_k = delta / K (Bonferroni over the K groups), so P(any wrong certificate) <= delta.

Naive frame-level certification (the negative control) is the SAME code with every frame given its own track id.
"""
import math

import numpy as np
from scipy.special import xlogy
from scipy.stats import binom


# ------------------------------------------------------------------ p-value
def h1(a, b):
    """Binary KL divergence kl(a || b) with 0 log 0 = 0 (xlogy), vectorised in a."""
    a = np.asarray(a, dtype=float)
    return xlogy(a, a / b) + xlogy(1.0 - a, (1.0 - a) / (1.0 - b))


def hb_pvalue(mu_hat, n, alpha):
    """Hoeffding-Bentkus p-value for H0: E[L] >= alpha, L in [0,1] i.i.d., n samples, sample mean mu_hat (array ok).
    The binomial tail uses ceil(n * mu_hat); n * mu_hat is rounded to 9 decimals first so float noise such as 3.0000000000000004
    does not push the ceiling up by one (the effect on the p-value is below 1e-9 relative and the plan's hand values are exact)."""
    mu = np.asarray(mu_hat, dtype=float)
    hoeffding = np.exp(-n * h1(np.minimum(mu, alpha), alpha))
    k = np.ceil(np.round(n * mu, 9))
    bentkus = math.e * binom.cdf(k, n, alpha)
    return np.minimum(1.0, np.minimum(hoeffding, bentkus))


def floor_alpha(n, delta_k):
    """Smallest alpha that can ever be certified with n tracks at level delta_k: zero errors and everything accepted gives
    mu_hat = 0 and p = (1 - alpha)^n, so alpha >= 1 - delta_k^(1/n). n = 392, delta_k = 0.05 gives 0.76%."""
    n = np.asarray(n, dtype=float)
    return np.where(n > 0, 1.0 - delta_k ** (1.0 / np.maximum(n, 1.0)), np.nan)


# ------------------------------------------------------------------ losses
def track_sums(track, accepted, wrong):
    """Per track (compact ids 0..n-1): frames in the group, accepted frames, accepted-and-wrong frames."""
    _, inv = np.unique(track, return_inverse=True)
    n_tk = np.bincount(inv).astype(float)
    acc = np.bincount(inv, weights=accepted.astype(float), minlength=len(n_tk))
    err = np.bincount(inv, weights=(accepted & wrong).astype(float), minlength=len(n_tk))
    return n_tk, acc, err


def shifted_losses(n_tk, acc, err, alpha):
    """L_t = err/n_tk + alpha * (not accepted)/n_tk. Written this way (not l_t + alpha) so that 'nothing wrong, everything accepted'
    gives exactly 0.0, which the zero-error floor depends on."""
    return err / n_tk + alpha * (n_tk - acc) / n_tk


# ------------------------------------------------------------------ certifier for one group
def certify_sequence(track, margin, wrong, grid, order, alpha, delta_k):
    """Fixed-sequence test of one group.
    track, margin, wrong: one entry per gated-in Cert frame of the group (track ids may be any integers).
    grid:  margin thresholds (ascending; index 0 = loosest). Accept rule: margin >= threshold.
    order: grid indices in the order they are tested (from Fit: start at the Fit argmin of E[L], then looser).
    Returns a dict; 'certified' is False when even the first threshold fails (the group is then escalated in full)."""
    track, margin, wrong = np.asarray(track), np.asarray(margin), np.asarray(wrong, dtype=bool)
    out = {"certified": False, "grid_index": None, "threshold": None, "n_tracks": 0, "n_frames": int(len(track)),
           "p_at_certified": None, "tests": []}
    if len(track) == 0:
        return out
    n_tk0, _, _ = track_sums(track, np.zeros(len(track), bool), wrong)
    n = len(n_tk0)
    out["n_tracks"] = int(n)
    for j in order:
        accepted = margin >= grid[j]
        n_tk, acc, err = track_sums(track, accepted, wrong)
        mu = float(shifted_losses(n_tk, acc, err, alpha).mean())
        p = float(hb_pvalue(mu, n, alpha))
        out["tests"].append({"grid_index": int(j), "threshold": float(grid[j]), "mu_hat": mu, "p": p})
        if p > delta_k:
            break                                              # stop at the first failure
        out.update(certified=True, grid_index=int(j), threshold=float(grid[j]), p_at_certified=p)
    return out


# ------------------------------------------------------------------ evaluation of a decision on any frames
def risk_stats(track, accepted, wrong, alpha=None):
    """What a certified threshold did on some frames (Cert or Test) of one group.
    cert_risk = (mean over tracks of accepted-error share) / (mean over tracks of accepted share): the quantity the certificate bounds.
    frame_selective_error = plain share of wrong frames among accepted frames. mean_L > alpha means the track-averaged promise is broken
    on these tracks (on a finite Test set this is noisy evidence, not a proof either way)."""
    track, accepted, wrong = np.asarray(track), np.asarray(accepted, dtype=bool), np.asarray(wrong, dtype=bool)
    out = {"n_tracks": 0, "frames": int(len(track)), "accepted": 0, "coverage": float("nan"), "errors_accepted": 0,
           "frame_selective_error": float("nan"), "cert_risk": float("nan"), "mean_L": float("nan"), "violates": None}
    if len(track) == 0:
        return out
    n_tk, acc, err = track_sums(track, accepted, wrong)
    e_share, a_share = float((err / n_tk).mean()), float((acc / n_tk).mean())
    out.update(n_tracks=int(len(n_tk)), accepted=int(acc.sum()), coverage=float(acc.sum() / len(track)), errors_accepted=int(err.sum()),
               frame_selective_error=float(err.sum() / acc.sum()) if acc.sum() else float("nan"),
               cert_risk=e_share / a_share if a_share > 0 else float("nan"))
    if alpha is not None:
        out["mean_L"] = e_share + alpha * (1.0 - a_share)
        out["violates"] = bool(out["mean_L"] > alpha + 1e-12)
    return out


# ------------------------------------------------------------------ policy
def accept_mask(group, margin, gated, results):
    """Gate -> group -> certified threshold -> accept or escalate. A frame is accepted only if it passed the gate, its group was
    certified, and its margin is >= that group's certified threshold. Every other frame is escalated."""
    acc = np.zeros(len(margin), dtype=bool)
    for g, r in results.items():
        if r["certified"]:
            acc |= gated & (group == g) & (margin >= r["threshold"])
    return acc
