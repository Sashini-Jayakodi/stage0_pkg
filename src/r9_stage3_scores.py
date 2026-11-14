"""POC-3 (R9) Stage 3: the four wrapped scores as per-frame sequences, the S2 design halves, grids, design statistics,
starts and the uncertified reference (LOG_r9_stage3_entry.md Part A (d) to (i)). Pure numpy; no file reading, so the
same functions serve the design script and the Cert/Test runner.

A sequence is (conf, lab) over the frames of one track in frame order: conf = -inf where the score has no value yet.
A track commits at the first frame with a value and conf >= lambda (plan 5.1).
"""
import numpy as np

from src import r8_stage1 as R1
from src import r8_stage6_methods as M
from src import r9_certify as C
from src.head import mahal_sq
from src.r8_stage3_b1design import frame_margins

N_CERT = 784                               # Cert tracks per re-split (POC-2 Stage 4)
N_POINTS, F_LOW = 30, 0.30                 # (f): 30 points, design commit fraction about 100% down to about 30%
HALF_SALT = (20261114, 3)                  # (e): halves seed, per class
K_WINDOW = 5                               # (d): S2 window, POC-2 pre-registered K
SCORES = ("S1", "S2-sum", "S2-min", "S3", "S4")
PRIMARY_S2 = "S2-sum"                      # (d): author's choice before any Cert number
S2_COMBO = {"S2-sum": "sum", "S2-min": "min"}
ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05)
RULES = C.RULES                            # binom (primary), hb (primary HB start), hb_a1, hb_bs


# ------------------------------------------------------------------ per-frame inputs
def frame_inputs(head, emb, passed, track, frame):
    """Everything the scores need per frame. Rows of each track in frame order (Stage 1 track_rows)."""
    rows = R1.track_rows(track, frame)
    D2 = mahal_sq(head, emb)
    m, top = frame_margins(head, emb, passed)
    return {"rows": rows, "D2": D2, "nc": M.nonconformity(D2), "m": m, "top": top, "passed": passed,
            "passed_rows": [r[passed[r]] for r in rows], "log_prior": np.log(head["priors"])}


# ------------------------------------------------------------------ sequences, one per score and track
def seq_s1(F, i):
    """POC-2 V1: summed log-likelihood plus log prior; margin top minus runner-up (Stage 1 code, -0.5 * D2 = ll)."""
    r = F["rows"][i]
    S, _ = R1.running_scores(-0.5 * F["D2"][r], F["passed"][r], F["log_prior"], "V1")
    return R1.margins(S)


def seq_s3(F, i, K=R1.K):
    """Running majority vote over passing frames; margin = votes(label) - votes(runner-up); ties to the smaller id."""
    r = F["rows"][i]
    conf, lab = np.full(len(r), -np.inf), np.full(len(r), -1)
    votes = np.zeros(K, int)
    for j, row in enumerate(r):
        if F["passed"][row]:
            votes[F["top"][row]] += 1
        if votes.any():                                          # a value exists from the first passing frame on
            l = int(np.argmax(votes))                            # first maximum = smallest class id
            conf[j], lab[j] = votes[l] - np.partition(votes, -2)[-2], l
    return conf, lab


def seq_s4(F, i):
    """Current-frame margin only; a value at passing frames (frame_margins gives -inf at gate-failing frames)."""
    r = F["rows"][i]
    ok = F["passed"][r]
    return F["m"][r].astype(float), np.where(ok, F["top"][r], -1)


def seq_s2(F, i, model):
    """[BK] score from the POC-2 B6 code; a value at the end frame of every full window of K passing frames."""
    r = F["rows"][i]
    conf, lab = np.full(len(r), -np.inf), np.full(len(r), -1)
    k, l, ends = M.b6_sequence(model, F["nc"], F["passed_rows"][i])
    if len(k):
        pos = np.flatnonzero(np.isin(r, ends))                  # both in frame order, so positions line up with ends
        assert len(pos) == len(ends) and (r[pos] == ends).all()
        conf[pos], lab[pos] = k, l
    return conf, lab


def sequences(F, models):
    """All scores for all tracks: {score: [(conf, lab), ...]}."""
    n = len(F["rows"])
    out = {"S1": [seq_s1(F, i) for i in range(n)], "S3": [seq_s3(F, i) for i in range(n)],
           "S4": [seq_s4(F, i) for i in range(n)]}
    for s, model in models.items():
        out[s] = [seq_s2(F, i, model) for i in range(n)]
    return out


# ------------------------------------------------------------------ S2 calibration on design halves (e)
def halves(y_track, salt=HALF_SALT):
    """Within each class, permute its tracks and deal A, B, A, B, ... Returns sorted positions of A and B."""
    A, B = [], []
    for k in np.unique(y_track):
        idx = np.flatnonzero(y_track == k)
        idx = idx[np.random.default_rng([*salt, int(k)]).permutation(len(idx))]
        A += idx[0::2].tolist()
        B += idx[1::2].tolist()
    return np.sort(np.array(A, int)), np.sort(np.array(B, int))


def fit_s2(F, y_track, A, B, combo, K=K_WINDOW):
    """ICP calibration on A; ECDF and (a, b) on B (b6_fit with select = B; its theta values are not used)."""
    return M.b6_fit(F["nc"], F["passed_rows"], y_track, {"icp": A, "ecdf": B, "select": B}, K, combo, ())


# ------------------------------------------------------------------ commit table per lambda (hard rule 4)
def first_commit(conf, grid):
    """Per lambda (ascending grid): first frame with a value and conf >= lambda; len(conf) if none.
    The max with the first valued frame makes lambda = -inf mean 'commit at the first frame with a value'."""
    has = np.flatnonzero(np.isfinite(conf))
    first = has[0] if len(has) else len(conf)
    return np.maximum(np.searchsorted(np.maximum.accumulate(conf), grid, side="left"), first)


def track_table(seqs, y_track, grid):
    """Tracks-by-lambdas: committed, wrong, and 1-based commit position (0 if not committed)."""
    n, G = len(seqs), len(grid)
    com, wrong, pos = np.zeros((n, G), bool), np.zeros((n, G), bool), np.zeros((n, G), int)
    for i, (conf, lab) in enumerate(seqs):
        idx = first_commit(conf, grid)
        c = idx < len(conf)
        com[i], pos[i, c] = c, idx[c] + 1
        wrong[i, c] = lab[idx[c]] != y_track[i]
    return {"committed": com, "wrong": wrong, "pos": pos}


# ------------------------------------------------------------------ grid (f)
def max_conf(seqs):
    return np.array([conf.max() if len(conf) else -np.inf for conf, _ in seqs])


def make_grid(cmax, n_points=N_POINTS, f_low=F_LOW):
    """lambda_0 = -inf; lambda_j = the m_j-th largest C_i, m_j = ceil(f_j * N), f_j = 1 - (1 - f_low) * j / (n-1).
    Repeated values removed (ascending = loosest first)."""
    N = len(cmax)
    desc = np.sort(cmax)[::-1]
    g = [-np.inf]
    for j in range(1, n_points):
        f = 1.0 - (1.0 - f_low) * j / (n_points - 1)
        g.append(desc[int(np.ceil(f * N - 1e-9)) - 1])
    return np.unique(np.array(g, float))


def design_stats(tab):
    """Per lambda: committed count, wrong count, f_d over ALL design tracks, e_d among committed (NaN if none),
    median commit position among committed."""
    com, w = tab["committed"].sum(0), (tab["committed"] & tab["wrong"]).sum(0)
    n = tab["committed"].shape[0]
    med = np.array([np.median(tab["pos"][tab["committed"][:, j], j]) if com[j] else np.nan
                    for j in range(len(com))])
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(com > 0, w / np.maximum(com, 1), np.nan)
    return {"n": int(n), "committed": com, "wrong": w, "f_d": com / n, "e_d": e, "median_pos": med}


# ------------------------------------------------------------------ starts (g) and reference (i)
def starts(f_d, n_design, n=N_CERT, alphas=ALPHAS, rules=RULES):
    """Start index per (alpha, rule) from the frozen engine; -1 = no start possible."""
    return {a: {r: int(C.start_for(r, f_d, n, a, n_design=n_design)) for r in rules} for a in alphas}


def reference(stats, alphas=ALPHAS):
    """Uncertified reference: loosest grid index with design committed error <= alpha; -1 if none."""
    out = {}
    for a in alphas:
        ok = np.flatnonzero((stats["committed"] > 0) & (stats["wrong"] <= a * stats["committed"] + 1e-12))
        out[a] = int(ok[0]) if len(ok) else -1
    return out
