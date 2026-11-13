"""POC-2 (R8) Stage 6 methods: B5 / B5c (Ringel et al. 2024) and B6 (Boursinos and Koutsoukos 2022).
LOG_r8_stage6_entry.md Part A (b)-(g) and Amendment 1. Pure numpy/scipy; no file reading.

Track data for Ringel (one entry per track, frames in frame order):
  pi[i]   : score per frame in [0, 1], -1 where there is no evidence yet (cannot stop there)
  pred[i] : V1 top class per frame, -1 where there is no evidence
  y[i], yfull[i] = pred[i][-1] (the full-sequence prediction; -1 if the track never had evidence)
A track stops at the first t with pi_t >= lambda_t, or at its own last frame (Ringel eq. 6): it never escalates.
"""
import numpy as np
from scipy.stats import binom

GRID = np.round(np.arange(0, 101) / 100.0, 10)                       # Lambda = {0, 0.01, ..., 1}; inf handled apart
INF = np.inf


# ------------------------------------------------------------------ Ringel: halting and the gap loss
def halt_index(pi_track, lam_vec):
    """0-based stop frame: first t with pi_t >= lam_vec[t], else the last frame. lam_vec has one entry per frame index."""
    hit = np.flatnonzero(pi_track >= lam_vec[: len(pi_track)])
    return int(hit[0]) if len(hit) else len(pi_track) - 1


def gap_loss(y, yfull, ystop):
    """Ringel eq. 3: (1[Y = Y_full] - 1[Y = Y_early])_+ ; a track without any prediction has loss 0."""
    return int((y == yfull) and (ystop != y))


def halts_scalar(pis, lam):
    """Stop frame of every track for one scalar threshold (marginal method), via the running max of pi."""
    out = np.empty(len(pis), int)
    for i, p in enumerate(pis):
        rm = np.maximum.accumulate(p)
        j = int(np.searchsorted(rm, lam, side="left")) if np.isfinite(lam) else len(p)
        out[i] = min(j, len(p) - 1)
    return out


def losses_at(halts, preds, y, yfull):
    ystop = np.array([preds[i][h] for i, h in enumerate(halts)])
    return np.array([gap_loss(y[i], yfull[i], ystop[i]) for i in range(len(halts))]), ystop


def binom_p(k, n, alpha):
    """Ringel §3: p = CDF_bin(n * R_hat; n, alpha) with n * R_hat = the number of gap errors k."""
    return float(binom.cdf(k, n, alpha))


# ------------------------------------------------------------------ B5: Ringel marginal method (Algorithm A.3)
def marginal_calibrate(pis, preds, y, yfull, alpha, delta, grid=GRID):
    """Fixed-sequence test from the largest lambda down; lambda_hat = smallest lambda that passed, inf if none."""
    lam_hat = INF
    n = len(pis)
    for lam in grid[::-1]:
        L, _ = losses_at(halts_scalar(pis, lam), preds, y, yfull)
        if binom_p(int(L.sum()), n, alpha) > delta:
            break
        lam_hat = float(lam)
    return lam_hat


# ------------------------------------------------------------------ B5c: Ringel conditional method (Algorithms 1, 2)
def halts_vector(pis, lam_vec):
    return np.array([halt_index(p, lam_vec) for p in pis])


def stage1_screen(pis, preds, y, yfull, alpha, tmax, grid=GRID):
    """Algorithm 1 (candidate screening) on D_cal-1, computed incrementally. At step t the thresholds after t are
    infinite, so the tracks halted by t are those halted before t (fixed losses) plus the still-running tracks with
    pi_t >= lambda or whose last frame is t. Same result as recomputing every halt (checked against a literal
    transcription of Algorithm 1 in the tests). t is 1-based in the paper; column t-1 here."""
    n = len(pis)
    last = np.array([len(p) - 1 for p in pis])
    PI = np.full((n, tmax), -1.0)
    LOSS = np.zeros((n, tmax), int)
    for i in range(n):
        PI[i, : len(pis[i])] = pis[i]
        LOSS[i, : len(pis[i])] = [gap_loss(y[i], yfull[i], p) for p in preds[i]]
    lam = np.full(tmax, INF)
    halted = np.zeros(n, bool)
    loss_sum, count = 0, 0
    for t in range(tmax):
        running = ~halted
        forced = running & (last == t)
        chosen = INF
        for g in grid:
            new = forced | (running & (PI[:, t] >= g))
            I = count + int(new.sum())
            if I == 0:
                chosen = INF                                         # cannot calculate the empirical risk
                break
            if (loss_sum + int(LOSS[new, t].sum())) / I <= alpha:
                chosen = float(g)
                break
        lam[t] = chosen
        new = forced | (running & (PI[:, t] >= chosen))
        loss_sum += int(LOSS[new, t].sum())
        count += int(new.sum())
        halted |= new
    return lam


def stage2_test(pis, preds, y, yfull, cand, alpha, delta, tmax):
    """Algorithm 2 (testing) on D_cal-2: configurations lambda^t from t = tmax down to 1, finer nulls t' >= t, max-p."""
    lam_hat = np.full(tmax, INF)
    for t in range(tmax, 0, -1):
        conf = np.full(tmax, INF)
        conf[t - 1:] = cand[t - 1:]
        h = halts_vector(pis, conf)
        for tp in range(t, tmax + 1):
            I = np.flatnonzero(h <= tp - 1)
            if not len(I):
                return lam_hat                                       # no evidence to reject: stop testing
            L, _ = losses_at(h[I], [preds[i] for i in I], y[I], yfull[I])
            if binom_p(int(L.sum()), len(I), alpha) > delta:
                return lam_hat                                       # failed to reject: stop testing
        lam_hat = conf                                               # H0 for lambda^t rejected
    return lam_hat


def conditional_calibrate(pis, preds, y, yfull, d1, d2, alpha, delta, tmax=30):
    sub = lambda idx: ([pis[i] for i in idx], [preds[i] for i in idx], y[idx], yfull[idx])
    cand = stage1_screen(*sub(d1), alpha, tmax)
    return stage2_test(*sub(d2), cand, alpha, delta, tmax), cand


# ------------------------------------------------------------------ Ringel evaluation on Test
def ringel_test(pis, preds, y, yfull, lam_vec, sizes):
    """Coverage, error among committed, accuracy gap (marginal, 20% and 50% earliest), median stop frame and size."""
    h = halts_vector(pis, lam_vec)
    L, ystop = losses_at(h, preds, y, yfull)
    com = ystop >= 0
    order = np.argsort(h, kind="stable")
    n = len(h)
    early = lambda q: float(L[order[: int(np.ceil(q * n))]].mean())
    return {"coverage": float(com.mean()), "committed": int(com.sum()),
            "error_committed": float((ystop[com] != y[com]).mean()) if com.any() else None,
            "gap_marginal": float(L.mean()), "gap_20pct_earliest": early(0.2), "gap_50pct_earliest": early(0.5),
            "median_commit_frame": float(np.median(h[com] + 1)) if com.any() else None,
            "median_size": float(np.median([sizes[i][h[i]] for i in np.flatnonzero(com)])) if com.any() else None}


# ------------------------------------------------------------------ B6: ICP p-values, windows, combination
def nonconformity(D2):
    """[BK] eq. 9: d_j / min_{k != j} d_k, d = sqrt(squared Mahalanobis distance). (frames, classes)."""
    d = np.sqrt(np.maximum(D2, 0.0))
    part = np.partition(d, 1, axis=1)
    first, second = part[:, 0], part[:, 1]
    other = np.where(d == first[:, None], second[:, None], first[:, None])
    return d / other


def icp_pvalues(nc, cal_sorted):
    """[BK] eq. 10: share of calibration scores >= the frame's score (no +1)."""
    return (len(cal_sorted) - np.searchsorted(cal_sorted, nc, side="left")) / len(cal_sorted)


def track_windows(passed_rows, K):
    """Windows of the last K passing frames, one per passing frame from the K-th on: list of (end_row, rows)."""
    return [(passed_rows[i], passed_rows[i - K + 1:i + 1]) for i in range(K - 1, len(passed_rows))]


def combine(Pw, combo):
    return Pw.sum(0) if combo == "sum" else Pw.min(0)


def ecdf(sorted_vals, x):
    return np.searchsorted(sorted_vals, x, side="right") / len(sorted_vals)


def cred_conf(pcomb):
    s = np.sort(pcomb, axis=-1)
    return s[..., -1], 1.0 - s[..., -2], np.argmax(pcomb, axis=-1)


def aurc(k, correct):
    order = np.argsort(-k, kind="stable")
    return float(np.mean(np.cumsum(~correct[order]) / np.arange(1, len(k) + 1)))


AB = np.round(np.arange(-1.0, 1.0001, 0.1), 10)


def best_ab(cred, conf, correct):
    best = None
    for a in AB:
        for b in AB:
            v = aurc(a * cred + b * conf, correct)
            if best is None or v < best[0] - 1e-15:
                best = (v, float(a), float(b))
    return best


def theta_table(k_seqs, lab_seqs, y):
    """Track-level risk-coverage over every candidate theta (commit at the first window with k > theta).
    Returns candidates (ascending), committed counts and wrong counts."""
    cands = np.unique(np.concatenate([[-np.inf]] + [k for k in k_seqs if len(k)]))
    com, wr = np.zeros(len(cands)), np.zeros(len(cands))
    for ks, ls, yy in zip(k_seqs, lab_seqs, y):
        if not len(ks):
            continue
        rm = np.maximum.accumulate(ks)
        idx = np.searchsorted(rm, cands, side="right")              # first window with k > theta
        c = idx < len(ks)
        com += c
        wr += c & (ls[np.minimum(idx, len(ks) - 1)] != yy)
    return cands, com, wr


def choose_theta(cands, com, wr, r):
    """Smallest theta (maximum coverage) with track-level selective error <= r; None if none."""
    ok = (com > 0) & (wr <= r * com + 1e-12)
    return float(cands[np.flatnonzero(ok)[0]]) if ok.any() else None


def b6_parts(cert_idx, seed, salt=20261118):
    """Amendment 1: Cert permuted, split into thirds (261/261/262 for 784): ICP calibration, ECDF, selection."""
    perm = np.asarray(cert_idx)[np.random.default_rng([salt, seed]).permutation(len(cert_idx))]
    n = len(perm)
    a, b = n // 3, 2 * (n // 3)
    return {"icp": np.sort(perm[:a]), "ecdf": np.sort(perm[a:b]), "select": np.sort(perm[b:])}


def b6_fit(nc, passed_rows, y, parts, K, combo, alphas):
    """Everything B6 learns, from the three Cert parts only. Returns the fitted model."""
    cal = np.sort(np.concatenate([nc[passed_rows[i], y[i]] for i in parts["icp"]]))
    P = lambda rows: icp_pvalues(nc[rows], cal)
    ec = np.sort(np.array([combine(P(w), combo)[y[i]] for i in parts["ecdf"] for _, w in track_windows(passed_rows[i], K)]))
    seqs = []
    for i in parts["select"]:
        W = track_windows(passed_rows[i], K)
        if not W:
            seqs.append((np.array([]), np.array([]), np.array([], int)))
            continue
        pc = np.stack([ecdf(ec, combine(P(w), combo)) for _, w in W])
        seqs.append(cred_conf(pc))
    cr = np.concatenate([s[0] for s in seqs if len(s[0])])
    cf = np.concatenate([s[1] for s in seqs if len(s[0])])
    cor = np.concatenate([s[2] == y[i] for s, i in zip(seqs, parts["select"]) if len(s[0])])
    v, a, b = best_ab(cr, cf, cor)
    cands, com, wr = theta_table([a * s[0] + b * s[1] for s in seqs], [s[2] for s in seqs], y[parts["select"]])
    return {"cal": cal, "ecdf": ec, "a": a, "b": b, "aurc": v, "K": K, "combo": combo,
            "theta": {al: choose_theta(cands, com, wr, al) for al in alphas}}


def b6_sequence(model, nc, passed_rows_i):
    """Per full window of one track: score k, label, end row (computed once, reused for every alpha)."""
    W = track_windows(passed_rows_i, model["K"])
    if not W:
        return np.array([]), np.array([], int), np.array([], int)
    pc = np.stack([ecdf(model["ecdf"], combine(icp_pvalues(nc[w], model["cal"]), model["combo"])) for _, w in W])
    c1, c2, lab = cred_conf(pc)
    return model["a"] * c1 + model["b"] * c2, lab, np.array([e for e, _ in W])


def b6_decide(seq, theta):
    """Commit at the first full window with k > theta. Returns (label or -1, end row or -1)."""
    k, lab, ends = seq
    if theta is None or not len(k):
        return -1, -1
    hit = np.flatnonzero(k > theta)
    return (int(lab[hit[0]]), int(ends[hit[0]])) if len(hit) else (-1, -1)


# ------------------------------------------------------------------ ECE ([BK] eq. 24)
EPS = np.arange(1, 201) / 1000.0


def ece(P, y):
    """P: (n, classes) p-values, y: true classes. Prediction set = {j : p_j > eps}."""
    py = P[np.arange(len(y)), y]
    acc = np.array([(py > e).mean() for e in EPS])
    return float(np.mean(np.abs(acc + EPS - 1.0)))
