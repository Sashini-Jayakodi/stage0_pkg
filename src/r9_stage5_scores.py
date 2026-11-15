"""POC-3 (R9) Stage 5: BelgiumTSC scores as per-view sequences, S2 halves with per-length ECDFs, grids, design
statistics, Rule 3, re-splits, within-sign correlation and the baseline design fits (LOG_r9_stage5_entry.md Part A
(c) to (k)). Pure numpy/scipy; no file reading, so the same functions serve the design step and Stage 6.

A unit is (pole, class); its views are in ascending view number (YYYYY). A sequence is (conf, lab) over the views of
one unit: a unit commits at the first view with conf >= lambda (plan 5.1). There is no gate: every view has a value.
"""
import numpy as np
from scipy.stats import beta

from src import r8_stage6_methods as M
from src import r9_certify as C
from src import r9_stage3_scores as SC3
from src import r9_stage4_data as D4
from src.head import mahal_sq
from src.stage5 import icc_anova

N_POINTS, F_LOW = 30, 0.30                 # (f): as Stage 3
MAX_WINDOW = 3                             # (d), Q2: S2 combines at most the last 3 views
HALF_SEED = 20261115                       # (e): S2 halves
SCORES = ("S1", "S2-sum", "S2-min", "S3", "S4")
PRIMARY_S2 = "S2-sum"
S2_COMBO = {"S2-sum": "sum", "S2-min": "min"}
ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05)
FD1_ALPHAS = (0.01, 0.02, 0.03, 0.05)      # (i): F-D1 looks at these
ALPHA_EXTRA = 0.10                         # (i): added only if F-D1 fails
RULES = C.RULES
RULE3_F_MIN = (70, 100)                    # (i): design commit fraction >= 70%, compared in integers
RULE3_UCB_Q = 0.90                         # (i): upper one-sided 90% Clopper-Pearson bound
SEEDS = (0, 1, 2, 3, 4)
TEST_FRACTION = (1, 4)                     # (g): 25% of pool poles to Test
BOOT_DRAWS = 2000
BOOT_SALT = {"unit": (20261115, 5, 1), "pole": (20261115, 5, 2)}
RQ1_DRAWS, RQ1_SALT = 2000, (20261115, 6)  # (k): used in Stage 6, frozen here
B2_SALT, B2_DESIGN_KEY = (20261115, 7), 100
B5C_SALT, B5C_TMAX = 20261116, 3


# ------------------------------------------------------------------ units and per-view inputs
def unit_rows(pole, key, view):
    """Units sorted by (pole, key); rows of each unit in ascending view number. Returns (units, rows)."""
    pole, key, view = (np.asarray(a, dtype=np.int64) for a in (pole, key, view))
    order = np.lexsort((view, key, pole))
    units, rows = [], []
    for i in order:
        u = (int(pole[i]), int(key[i]))
        if not units or units[-1] != u:
            units.append(u)
            rows.append([])
        rows[-1].append(int(i))
    return units, [np.array(r, dtype=np.int64) for r in rows]


def view_inputs(head, emb):
    """Per view: log-likelihood, single-view margin and top class (head posterior), nonconformity ([BK] eq. 9)."""
    D2 = mahal_sq(head, emb)
    ll = -0.5 * D2
    lp = np.log(head["priors"])[None, :] + ll
    part = np.partition(lp, -2, axis=1)
    return {"D2": D2, "ll": ll, "log_prior": np.log(head["priors"]), "m": part[:, -1] - part[:, -2],
            "top": np.argmax(lp, axis=1), "nc": M.nonconformity(D2)}


# ------------------------------------------------------------------ sequences (d)
def _margin_top(S):
    part = np.partition(S, -2, axis=1)
    return part[:, -1] - part[:, -2], np.argmax(S, axis=1)


def seq_s1(V, r):
    """POC-2 V1: log prior plus summed log-likelihood over views 1..v; margin top minus runner-up."""
    return _margin_top(V["log_prior"][None, :] + np.cumsum(V["ll"][r], axis=0))


def seq_s3(V, r):
    """Running majority vote over per-view top classes (all views); ties to the smaller class id."""
    K = V["ll"].shape[1]
    votes = np.zeros(K, int)
    conf, lab = np.empty(len(r)), np.empty(len(r), int)
    for j, row in enumerate(r):
        votes[V["top"][row]] += 1
        l = int(np.argmax(votes))
        conf[j], lab[j] = votes[l] - np.partition(votes, -2)[-2], l
    return conf, lab


def seq_s4(V, r):
    """Current view alone."""
    return V["m"][r].astype(float), V["top"][r].astype(int)


def windows(n_views, L=MAX_WINDOW):
    """Q2: window of view v = views max(1, v - L + 1)..v (0-based positions)."""
    return [np.arange(max(0, v - L + 1), v + 1) for v in range(n_views)]


def pcomb_seq(P, ecdfs, combo):
    """Combined, ECDF-transformed p-values per view; each window uses the ECDF of its own length."""
    return np.stack([M.ecdf(ecdfs[len(w)], M.combine(P[w], combo)) for w in windows(len(P))])


def seq_s2(V, r, model):
    P = M.icp_pvalues(V["nc"][r], model["cal"])
    c1, c2, lab = M.cred_conf(pcomb_seq(P, model["ecdf"], model["combo"]))
    return model["a"] * c1 + model["b"] * c2, lab.astype(int)


def sequences(V, rows, models):
    out = {"S1": [seq_s1(V, r) for r in rows], "S3": [seq_s3(V, r) for r in rows],
           "S4": [seq_s4(V, r) for r in rows]}
    for s, model in models.items():
        out[s] = [seq_s2(V, r, model) for r in rows]
    return out


# ------------------------------------------------------------------ S2 calibration (e)
def halves(units):
    """Halves by pole with the Stage 4 rule: B = split_by_pole(units, 1/2, HALF_SEED); A = the other poles.
    Returns unit positions (A, B) and the pole lists."""
    B_poles = set(D4.split_by_pole(units, (1, 2), HALF_SEED))
    A = np.array([i for i, (p, _) in enumerate(units) if p not in B_poles], int)
    B = np.array([i for i, (p, _) in enumerate(units) if p in B_poles], int)
    A_poles = sorted({units[i][0] for i in A})
    return A, B, A_poles, sorted(B_poles)


def fit_s2(V, rows, y_unit, A, B, combo):
    """ICP scores on half A; one ECDF per window length and (a, b) on half B (POC-2 AURC rule)."""
    cal = np.sort(np.concatenate([V["nc"][rows[i], y_unit[i]] for i in A]))
    P = {i: M.icp_pvalues(V["nc"][rows[i]], cal) for i in B}
    ec = {L: [] for L in range(1, MAX_WINDOW + 1)}
    for i in B:
        for w in windows(len(rows[i])):
            ec[len(w)].append(M.combine(P[i][w], combo)[y_unit[i]])
    for L, v in ec.items():
        assert len(v), f"no half-B window of length {L}; stop (Part A (e) needs all three ECDFs)"
    ec = {L: np.sort(np.array(v)) for L, v in ec.items()}
    cr, cf, cor = [], [], []
    for i in B:
        c1, c2, lab = M.cred_conf(pcomb_seq(P[i], ec, combo))
        cr.append(c1)
        cf.append(c2)
        cor.append(lab == y_unit[i])
    v, a, b = M.best_ab(np.concatenate(cr), np.concatenate(cf), np.concatenate(cor))
    return {"cal": cal, "ecdf": ec, "a": a, "b": b, "aurc": v, "combo": combo, "max_window": MAX_WINDOW}


# ------------------------------------------------------------------ grids and statistics (f): Stage 3 functions
make_grid = SC3.make_grid
track_table = SC3.track_table
design_stats = SC3.design_stats
reference = SC3.reference


def max_conf(seqs):
    return SC3.max_conf(seqs)


# ------------------------------------------------------------------ re-splits (g)
def resplits(pool_units, seeds=SEEDS):
    """Per seed: Test poles by the Stage 4 rule (25%), Cert = the other pool poles; unit counts per side."""
    poles = sorted({p for p, _ in pool_units})
    out = {}
    for s in seeds:
        test = D4.split_by_pole(pool_units, TEST_FRACTION, s)
        ts = set(test)
        cert = [p for p in poles if p not in ts]
        out[s] = {"test_poles": test, "cert_poles": cert,
                  "n_cert_units": sum(1 for p, _ in pool_units if p not in ts),
                  "n_test_units": sum(1 for p, _ in pool_units if p in ts)}
    return out


# ------------------------------------------------------------------ starts (h)
def starts(f_d, n_by_seed, n_design, alphas, rules=RULES):
    """Start index per (seed, alpha, rule) from the frozen engine with that seed's Cert size; -1 = no start."""
    return {s: {a: {r: int(C.start_for(r, f_d, n, a, n_design=n_design)) for r in rules} for a in alphas}
            for s, n in n_by_seed.items()}


# ------------------------------------------------------------------ Rule 3 (i)
def ucb90(wrong, committed, q=RULE3_UCB_Q):
    """Upper one-sided Clopper-Pearson bound of wrong / committed (1 if all wrong)."""
    if wrong >= committed:
        return 1.0
    return float(beta.ppf(q, wrong + 1, committed - wrong))


def rule3(stats, n_min, alpha):
    """Predicted certifiable if some grid point with f_d >= 70% has UCB90 * n_c <= k_max(n_c, alpha),
    n_c = floor(f_d * n_min) (computed in integers: committed * n_min // n_design).
    Deciding point (reported): the loosest point that passes; if none passes, the point with f_d >= 70% where
    UCB90 * n_c - k_max is smallest (k_max = -1 when not certifiable even with 0 errors)."""
    nd = stats["n"]
    pts = []
    for j, (c, w) in enumerate(zip(stats["committed"], stats["wrong"])):
        c, w = int(c), int(w)
        if c * RULE3_F_MIN[1] < RULE3_F_MIN[0] * nd or c == 0:
            continue
        nc = c * n_min // nd
        km = C.max_errors(nc, alpha) if nc > 0 else None
        u = ucb90(w, c)
        pts.append({"j": j, "committed": c, "wrong": w, "f_d": c / nd, "e_d": w / c, "ucb90": u, "n_c": nc,
                    "k_max": km, "ucb90_x_n_c": u * nc, "pass": km is not None and u * nc <= km})
    passed = [p for p in pts if p["pass"]]
    if passed:
        dec = passed[0]
    elif pts:
        dec = min(pts, key=lambda p: p["ucb90_x_n_c"] - (p["k_max"] if p["k_max"] is not None else -1))
    else:
        dec = None
    return {"predicted": bool(passed), "deciding": dec, "points": pts}


def primary_alpha(r3, alphas):
    """Smallest alpha predicted certifiable for at least one score; None if none."""
    ok = [a for a in alphas if any(r3[s][a]["predicted"] for s in r3)]
    return min(ok) if ok else None


# ------------------------------------------------------------------ correlation (j)
def pole_boot_weights(cluster_pole, n_draws, salt):
    """Pole-cluster bootstrap: draw all poles with replacement; a cluster's weight = how often its pole was drawn."""
    poles = np.unique(cluster_pole)
    idx = np.searchsorted(poles, cluster_pole)
    rng = np.random.default_rng(list(salt))
    draws = rng.integers(0, len(poles), size=(n_draws, len(poles)))
    counts = np.stack([np.bincount(d, minlength=len(poles)) for d in draws])
    return counts[:, idx]


def rho_ci(n, e, cluster_pole, salt, n_draws=BOOT_DRAWS):
    """ANOVA ICC (POC-1 icc_anova) with a 95% percentile interval from the pole-cluster bootstrap."""
    n, e = np.asarray(n, float), np.asarray(e, float)
    est = float(icc_anova(np.ones((1, len(n))), n, e)[0])
    bs = icc_anova(pole_boot_weights(np.asarray(cluster_pole), n_draws, salt), n, e)
    ok = np.isfinite(bs)
    lo, hi = (np.percentile(bs[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan))
    return {"rho": est, "ci95": [float(lo), float(hi)], "draws": int(n_draws), "draws_undefined": int((~ok).sum()),
            "clusters": int(len(n)), "items": int(n.sum()), "errors": int(e.sum())}


def design_effect(rho, m):
    return 1.0 + (m - 1.0) * rho


# ------------------------------------------------------------------ baselines (k)
def b2_draw(n_views, rng):
    """One view per unit, uniform over its views, units in the given order."""
    return np.array([int(rng.integers(n)) for n in n_views], int)


def seq_b2(V, r, k):
    conf, lab = np.full(len(r), -np.inf), np.full(len(r), -1)
    conf[k], lab[k] = V["m"][r[k]], V["top"][r[k]]
    return conf, lab


def b6_theta(seqs_B, y_B, alphas):
    """B6 threshold per alpha on half B: smallest theta with unit-level selective error <= r = alpha (commit at the
    first view with k > theta, POC-2 rule)."""
    cands, com, wr = M.theta_table([s[0] for s in seqs_B], [s[1] for s in seqs_B], y_B)
    return {a: M.choose_theta(cands, com, wr, a) for a in alphas}
