#!/usr/bin/env python3
# CCC FORWARD FAIR RUN — Gupta & Samaras 2026 (MNRAS 552, arXiv:2608.11575)
# run FORWARD in CPQR's footing, same 175 SPARC galaxies, same mask,
# same error model, same M/L, same scoring as the fair run.
#
#   python3 ccc_forward_fair_run_2026-11-19.py
#
# THE LAW (paper eqs 15, 18, 19 — read from ccc_sparc_2026.txt):
#   The dynamically inferred density rho_o and the gravitating baryonic
#   density rho_bX are related by rho_o = rho_bX * nu(y), y = rho_bX/rho_t.
#   Sharp law (eq 15): rho_o = sqrt(rho_t * rho_bX) for rho_bX < rho_t,
#   rho_o = rho_bX above it. Smooth completion (paper default, no new
#   parameter), transcribed MOND forms into density space:
#     standard (eq 18): nu_std(y) = 1 / (1 - exp(-sqrt(y)))
#     simple   (eq 19): nu_simple(y) = 1/2 + sqrt(1/4 + 1/y),
#                       rho_o = rho_bX/2 + sqrt((rho_bX/2)^2 + rho_bX*rho_t)
#   Both reduce to the sharp law deep (rho_o -> sqrt(rho_t*rho_bX)) and
#   to Newtonian (rho_o -> rho_bX) in the core. Table 3 headline is the
#   STANDARD form (mean chi2_nu 2.58 vs simple 2.59); standard is PRIMARY
#   here, simple is the check variant.
#
# FORWARD CONSTRUCTION (this script — the paper fits INVERSE):
#   Vbar (M/L 0.5/0.7) -> Mb(R) = R*Vbar^2/G -> PCHIP interpolant of Mb
#   -> shell-mean rho_b = dMb/dV on a fine log grid -> rho_o via nu
#   -> M(R) = sum nu*dMb over shells (+ inner mass inside R_min from
#   the shell-mean density) -> Vpred = sqrt(G*M/R) at the data radii.
#   Key identity: dM = nu * dMb. Shells with dMb < 0 (SPARC Vbar tables
#   can be jagged, e.g. UGC01281) are carried at nu = 1, so the
#   Newtonian limit M(R) = Mb(R) is EXACT for every galaxy.
#
# FAIRNESS (the whole point — the Grim Rogers loss-function attack):
#   EVERY fitted model is fitted by the SAME loss, mean |resid| %,
#   which is the scored metric. chi2_red is reported as a secondary
#   metric only. The chi2-fit NFW/MOND numbers are reproduced in the
#   same code as GATES (they must match the fair-run scoreboard), then
#   NFW and MOND are REFIT by meanabs for the fair comparison.
#   Comparison set, 1 knob: CCC rho_t per galaxy (as in their paper)
#   vs CPQR tilt (row 586 optimal direction) vs MOND ups_disk vs NFW 2p.
#   Comparison set, 0 knobs: CCC with UNIVERSAL a_t = 2.14e-10 m/s^2
#   (Appendix B median; rho_t per galaxy from R_t = Vflat^2/a_t and the
#   defining equation rho_t = rho_b(R_t)*nu(rho_b(R_t)/rho_t)) vs CPQR
#   frozen vs MOND frozen. The 0-knob round fits NOTHING anywhere.
#
# PRE-REGISTRATION: this is a measurement, not a pass/fail on CPQR.
#   Report mean |resid| %, chi2_red, per-radius-bin means and
#   head-to-head win fractions for both rounds, and state plainly
#   whether CCC beats, ties or loses to CPQR tilt (1 knob) and CPQR
#   frozen (0 knobs), with the margins. No verdict spin.
#
# AMBIGUITIES (documented, not silently resolved):
#   (1) The paper fits INVERSE (Vb from Vobs). Forward is our
#       construction from its closed-form law; the paper validates a
#       forward implementation only for MOND on 2 galaxies (App. B).
#   (2) Appendix B converts universal a_t to rho_t "through its
#       reconstructed profile" without giving the conversion formula.
#       We solve the defining equation rho_t = rho_o(R_t) on the
#       FORWARD baryonic profile. For the smooth laws this gives
#       rho_t = rho_b(R_t)/y* with y* ~ 0.51 (std) / 0.50 (simple);
#       the sharp law would give rho_t = rho_b(R_t) exactly.
#       We use the OBSERVED Vflat (median Vobs over R > 0.7*Rmax),
#       not CPQR's baryonic V_flat, so the 0-knob CCC number is
#       independent of CPQR machinery. Sensitivity to the CPQR
#       V_flat choice is reported in the JSON.
#   (3) The turn-off radius R_t = Vflat^2/a_t can fall outside the
#       measured radial range; rho_b is then held at the edge value
#       (counted and reported).

import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
except Exception:  # pragma: no cover
    PchipInterpolator = None
    brentq = None

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = HERE
G_KPC = 4.3009e-6            # kpc (km/s)^2 / Msun, as in the fair run
KPC_M = 3.08567758e19        # m per kpc
MSUN_KPC3_TO_G_CM3 = 6.768e-32  # 1 Msun/kpc^3 in g/cm^3
AT_UNIVERSAL = 2.14e-10      # m/s^2, Appendix B median of resolved a_t
PI = math.pi

# Paper Table 2 (standard smooth nu, inverse-mode chi2 fits) for cross-check:
# fitted rho_t in 1e-24 g/cm^3. Forward meanabs fits should land within
# a factor ~2-3 of these, not identical (different direction + objective).
PAPER_RHOT_1E24 = {
    "D631-7": 2.13, "NGC1090": 1.28, "NGC3198": 1.62, "NGC3521": 3.67,
    "NGC6503": 4.19, "NGC7814": 6.86, "UGC09133": 1.56,
}
PAPER_RHOT_MEDIAN_1E24 = 2.9   # full-sample median, standard smooth nu
PAPER_RHOT_SCATTER_DEX = 0.82


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FR = _load_module("fair_run_galaxy",
                  os.path.join(THEORY, "sparc175_fair_run_2026-11-19_galaxy.py"))
OK = _load_module("one_knob",
                  os.path.join(THEORY, "sparc175_1knob_fair_run_2026-11-19.py"))


# ---------------------------------------------------------------- CCC core

def rho_o_std(rho_b, rho_t):
    """rho_o = rho_b * nu_std(rho_b/rho_t), nu_std = 1/(1-exp(-sqrt(y))).
    Stable via expm1; rho_b <= 0 -> 0."""
    rho_b = np.asarray(rho_b, dtype=float)
    out = np.zeros_like(rho_b)
    pos = rho_b > 0
    if np.any(pos):
        s = np.sqrt(rho_b[pos] / rho_t)
        out[pos] = rho_b[pos] / (-np.expm1(-s))
    return out


def rho_o_simple(rho_b, rho_t):
    """rho_o = rho_b/2 + sqrt((rho_b/2)^2 + rho_b*rho_t) (eq 19)."""
    rho_b = np.asarray(rho_b, dtype=float)
    return 0.5 * rho_b + np.sqrt((0.5 * rho_b) ** 2 + rho_b * rho_t)


RHO_O = {"std": rho_o_std, "simple": rho_o_simple}


def nu_std(y):
    """Standard smooth transition (eq 18): nu = 1/(1-exp(-sqrt(y)))."""
    y = np.asarray(y, dtype=float)
    return 1.0 / (-np.expm1(-np.sqrt(np.maximum(y, 0.0))))


def nu_simple(y):
    """Simple smooth transition (eq 19): nu = 1/2 + sqrt(1/4 + 1/y)."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.inf)
    pos = y > 0
    out[pos] = 0.5 + np.sqrt(0.25 + 1.0 / y[pos])
    return out


NU = {"std": nu_std, "simple": nu_simple}


def ccc_profile(R, Vd, Vb, Vg, n_fine=400):
    """Baryonic mass profile + fine-grid shell densities for one galaxy.
    Mb(R) = R*Vbar^2/G is interpolated with a shape-preserving PCHIP;
    the fine grid carries SHELL-MEAN densities rho_b = dMb/dV, so the
    Newtonian limit is exact by construction (see ccc_predict below).
    SPARC Vbar tables can be jagged (e.g. UGC01281: Vbar dips 6.4 ->
    3.9 km/s), making Mb locally non-monotone; shell means handle that
    without inventing mass, and negative-dMb shells are carried at nu=1
    in the predictor (documented there)."""
    Vd_ml = Vd * math.sqrt(0.5)
    Vb_ml = Vb * math.sqrt(0.7)
    Vb2 = np.abs(Vg) ** 2 + Vd_ml ** 2 + Vb_ml ** 2
    Mb = R * Vb2 / G_KPC
    pchip = PchipInterpolator(R, Mb, extrapolate=True)
    Rf = np.logspace(math.log10(R[0]), math.log10(R[-1]), n_fine)
    Mb_f = pchip(Rf)
    dMb = np.diff(Mb_f)
    dV = (4.0 * PI / 3.0) * np.diff(Rf ** 3)
    rho_b_shell = dMb / dV
    Rm = 0.5 * (Rf[1:] + Rf[:-1])
    rho_b_inner = Mb[0] / ((4.0 * PI / 3.0) * R[0] ** 3)
    return {"R": R, "pchip": pchip, "Rf": Rf, "Mb_f": Mb_f,
            "dMb": dMb, "dV": dV, "rho_b_shell": rho_b_shell, "Rm": Rm,
            "rho_b_inner": float(max(rho_b_inner, 0.0)), "Mb": Mb}


def ccc_density_at(prof, Rq):
    """Baryonic density at radius Rq (kpc): the fine-grid shell-mean
    density of the interval containing Rq. If that shell is non-positive
    (jagged-Vbar artifact), take the nearest positive shell-mean.
    Returns (rho_b, extrapolated_flag)."""
    R = prof["R"]
    clipped = bool(Rq < R[0] or Rq > R[-1])
    Rq_c = float(np.clip(Rq, R[0], R[-1]))
    Rm, rho_b = prof["Rm"], prof["rho_b_shell"]
    j = int(np.searchsorted(Rm, Rq_c))
    j = min(max(j, 0), len(Rm) - 1)
    if rho_b[j] <= 0:
        for jj in np.argsort(np.abs(Rm - Rq_c)):
            if rho_b[jj] > 0:
                j = int(jj)
                break
        else:
            return max(prof["rho_b_inner"], 1e-30), clipped
    return float(rho_b[j]), clipped


def ccc_predict_from_profile(prof, rho_t, form):
    """Vpred at the data radii for turn-off density rho_t (Msun/kpc^3).
    dM = nu*dMb per fine shell; shells with dMb < 0 (jagged-Vbar
    artifact) are carried at nu = 1, i.e. dM = dMb. In the Newtonian
    limit (rho_t -> 0, nu -> 1) this gives M(R) = Mb(R) EXACTLY for
    every galaxy, which the validation asserts."""
    f = RHO_O[form]
    rho_b = np.maximum(prof["rho_b_shell"], 0.0)
    rho_o = f(rho_b, rho_t)
    dM = np.where(prof["dMb"] >= 0, rho_o * prof["dV"], prof["dMb"])
    R0 = prof["R"][0]
    M_inner = (4.0 * PI / 3.0) * R0 ** 3 * float(f(prof["rho_b_inner"], rho_t))
    M_nodes = M_inner + np.concatenate([[0.0], np.cumsum(dM)])
    M_data = np.interp(prof["R"], prof["Rf"], M_nodes)
    return np.sqrt(G_KPC * M_data / np.maximum(prof["R"], 1e-9))


def ccc_predict(R, Vd, Vb, Vg, rho_t, form="std", n_fine=400):
    return ccc_predict_from_profile(ccc_profile(R, Vd, Vb, Vg, n_fine),
                                    rho_t, form)


def ccc_fit_rhot(gal, prof, form):
    """1-knob CCC: fit log10(rho_t) by mean |resid| % (the scored metric).
    Coarse grid + golden-section refine."""
    R, Vo = gal["R"], gal["Vo"]
    Vd, Vb, Vg = gal["Vd"], gal["Vb"], gal["Vg"]
    mask = (Vo > 1.0) & (gal["Ve"] > 0)

    def score(logrt):
        Vp = ccc_predict_from_profile(prof, 10.0 ** logrt, form)
        if not np.all(np.isfinite(Vp)):
            return 1e9
        return float(np.mean(np.abs(Vp[mask] - Vo[mask]) / Vo[mask])) * 100.0

    grid = np.arange(4.0, 11.01, 0.25)
    vals = [score(g) for g in grid]
    i = int(np.argmin(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    best_logrt = FR.golden_min(score, lo, hi, tol=1e-4)
    return best_logrt, score(best_logrt)


def ccc_rhot_from_at(gal, prof, form, a_t=AT_UNIVERSAL, use_cpqr_vflat=False):
    """0-knob CCC: universal a_t -> per-galaxy rho_t.
    R_t = Vflat^2/a_t; rho_t solves rho_t = rho_b(R_t)*nu(rho_b(R_t)/rho_t).
    Returns (rho_t, R_t_kpc, rho_b_at_Rt, Vflat_kms, extrapolated_flag,
    implied y* = rho_b(R_t)/rho_t)."""
    R, Vo = gal["R"], gal["Vo"]
    if use_cpqr_vflat:
        _, _, Vflat, _, _, _ = FR.cpqr_compute_v7(
            R, gal["Vd"], gal["Vb"], gal["Vg"], 1.0, 1.0)
        Vflat = float(Vflat)
    else:
        outer = Vo[R > 0.7 * R.max()]
        if len(outer) < 3:
            outer = Vo[-3:]
        Vflat = float(np.median(outer))
    Rt_kpc = (Vflat * 1000.0) ** 2 / a_t / KPC_M
    rho_b_at, clipped = ccc_density_at(prof, Rt_kpc)
    f = RHO_O[form]

    def g(logrt):
        rt = 10.0 ** logrt
        return float(f(rho_b_at, rt) - rt)

    lo, hi = math.log10(max(rho_b_at, 1e-30)) - 3.0, math.log10(max(rho_b_at, 1e-30)) + 3.0
    root = brentq(g, lo, hi, xtol=1e-10)
    rho_t = 10.0 ** root
    return rho_t, Rt_kpc, rho_b_at, Vflat, clipped, rho_b_at / rho_t


# ------------------------------------------------------- fair refits (meanabs)

def fit_mond_meanabs(gal, C):
    """MOND 1-param by mean |resid| % (same a0_local, same bounds as the
    chi2 fit — only the loss changes)."""
    R, Vo = gal["R"], gal["Vo"]
    Vd, Vb, Vg = gal["Vd"], gal["Vb"], gal["Vg"]
    mask = (Vo > 1.0) & (gal["Ve"] > 0)

    def obj(u):
        p = FR.mond_predict(R, Vd, Vb, Vg, u, C["a0_local"])
        if not np.all(np.isfinite(p)):
            return 1e12
        return float(np.mean(np.abs(p[mask] - Vo[mask]) / Vo[mask])) * 100.0

    u = FR.golden_min(obj, *FR.UPS_BOUNDS)
    return float(u), obj(u)


def fit_nfw_meanabs(gal, C):
    """NFW 2-param by mean |resid| % (same grid, bounds, Nelder-Mead as
    the chi2 fit — only the loss changes)."""
    R, Vo = gal["R"], gal["Vo"]
    Vd, Vb, Vg = gal["Vd"], gal["Vb"], gal["Vg"]
    mask = (Vo > 1.0) & (gal["Ve"] > 0)

    def obj(theta):
        logM, logc = theta
        if not (8.0 <= logM <= 14.5 and -0.5 <= logc <= 2.0):
            return 1e12
        try:
            p = FR.nfw_predict(R, Vd, Vb, Vg, 10.0 ** logM, 10.0 ** logc, C)
        except Exception:
            return 1e12
        if not np.all(np.isfinite(p)):
            return 1e12
        return float(np.mean(np.abs(p[mask] - Vo[mask]) / Vo[mask])) * 100.0

    best = (None, 1e18)
    for logM in np.arange(9.0, 13.75, 0.25):
        for logc in np.arange(0.3, 1.5, 0.2):
            v = obj((logM, logc))
            if v < best[1]:
                best = ((logM, logc), v)
    try:
        theta, v = FR.nelder_mead(obj, np.array(best[0]),
                                  np.array([0.2, 0.15]))
        logM, logc = float(theta[0]), float(theta[1])
        ok_fit = math.isfinite(v) and v < 1e11
    except Exception:
        ok_fit = False
        (logM, logc), v = best[0], best[1]
    if not ok_fit:
        return None
    return {"M200": float(10.0 ** logM), "conc": float(10.0 ** logc),
            "logM": float(logM), "score": float(v)}


# ---------------------------------------------------------------- validations

def run_validations(galaxies):
    """Cheap checks that must pass before the expensive loop runs.
    Returns dict of results; raises SystemExit on failure."""
    out = {}
    # (V1) Newtonian limit: rho_t -> 0 must reproduce Vbar (dense core:
    # y = rho_b/rho_t >> 1, nu -> 1, rho_o -> rho_b). NOTE: the limit is
    # rho_t -> 0, NOT infinity — huge rho_t puts everything in the deep
    # regime rho_o = sqrt(rho_t*rho_b).
    names = sorted(galaxies)
    worst = 0.0
    for name in names[:25]:
        gal = galaxies[name]
        R, Vd, Vb, Vg = gal["R"], gal["Vd"], gal["Vb"], gal["Vg"]
        Vd_ml = Vd * math.sqrt(0.5)
        Vb_ml = Vb * math.sqrt(0.7)
        Vbar = np.sqrt(np.abs(Vg) ** 2 + Vd_ml ** 2 + Vb_ml ** 2)
        Vp = ccc_predict(R, Vd, Vb, Vg, 1e-30, "std")
        err = float(np.max(np.abs(Vp - Vbar) / np.maximum(Vbar, 1e-9))) * 100.0
        worst = max(worst, err)
    out["V1_newtonian_limit_max_pct_err_25gal"] = worst
    assert worst < 0.5, f"V1 FAIL: Newtonian limit error {worst:.3f}%"

    # (V2) Deep regime: rho_o -> sqrt(rho_t * rho_b) for y << 1.
    rho_b = np.logspace(-2, 2, 9)
    rho_t = 1e6
    got = rho_o_std(rho_b, rho_t)
    want = np.sqrt(rho_t * rho_b)
    rel = float(np.max(np.abs(got - want) / want))
    out["V2_deep_regime_max_rel_err"] = rel
    assert rel < 0.01, f"V2 FAIL: deep-regime error {rel:.4f}"
    got_s = rho_o_simple(rho_b, rho_t)
    rel_s = float(np.max(np.abs(got_s - want) / want))
    out["V2b_deep_regime_simple_max_rel_err"] = rel_s
    assert rel_s < 0.01, f"V2b FAIL: deep-regime (simple) error {rel_s:.4f}"

    # (V3) Grid convergence: n_fine 200 vs 800 changes a fitted score by
    # < 0.05 pp on a test galaxy (NGC3198, a Table 2 benchmark).
    gal = galaxies["NGC3198"]
    prof200 = ccc_profile(gal["R"], gal["Vd"], gal["Vb"], gal["Vg"], 200)
    prof800 = ccc_profile(gal["R"], gal["Vd"], gal["Vb"], gal["Vg"], 800)
    _, s200 = ccc_fit_rhot(gal, prof200, "std")
    _, s800 = ccc_fit_rhot(gal, prof800, "std")
    out["V3_grid_convergence_NGC3198_pp"] = abs(s200 - s800)
    assert abs(s200 - s800) < 0.05, \
        f"V3 FAIL: grid convergence {abs(s200 - s800):.4f} pp"

    # (V4) Table 2 spot check: forward meanabs rho_t within factor 3 of
    # the paper's inverse chi2 rho_t on the 7 benchmark galaxies.
    ratios = {}
    for name, paper in PAPER_RHOT_1E24.items():
        gal = galaxies[name]
        prof = ccc_profile(gal["R"], gal["Vd"], gal["Vb"], gal["Vg"])
        logrt, _ = ccc_fit_rhot(gal, prof, "std")
        mine = (10.0 ** logrt) * MSUN_KPC3_TO_G_CM3 / 1e-24
        ratios[name] = float(mine / paper)
    out["V4_table2_ratios_forward_over_paper"] = ratios
    worst_ratio = max(max(ratios.values()), 1.0 / min(ratios.values()))
    out["V4_worst_factor"] = float(worst_ratio)
    assert worst_ratio < 3.0, f"V4 FAIL: worst Table 2 factor {worst_ratio:.2f}"
    return out


# ---------------------------------------------------------------- main

MODELS = ["cpqr_2p", "cpqr_1p_tilt", "cpqr_0p", "nfw_2p_chi2", "mond_1p_chi2",
          "mond_0p", "nfw_2p_meanabs", "mond_1p_meanabs",
          "ccc_1p_std", "ccc_1p_simple", "ccc_0p_std", "ccc_0p_simple"]
KNOBS = {"cpqr_2p": 2, "cpqr_1p_tilt": 1, "cpqr_0p": 0,
         "nfw_2p_chi2": 2, "mond_1p_chi2": 1, "mond_0p": 0,
         "nfw_2p_meanabs": 2, "mond_1p_meanabs": 1,
         "ccc_1p_std": 1, "ccc_1p_simple": 1, "ccc_0p_std": 0, "ccc_0p_simple": 0}
GATES = {"cpqr_2p": 5.7263, "cpqr_1p_tilt": 8.8648, "cpqr_0p": 20.6027,
         "nfw_2p_chi2": 8.89, "mond_1p_chi2": 12.47, "mond_0p": 23.38}


def main():
    t0 = time.time()
    print("CCC FORWARD FAIR RUN — Gupta & Samaras 2026, forward in CPQR footing")
    print("same 175 galaxies, same mask, same errors, same M/L, same scoring.")
    print("Every fitted model fitted by mean |resid| % (the scored metric);")
    print("chi2-fit NFW/MOND reproduced as gates, then refit by meanabs.")
    print()
    C = {"rho_crit_gal": 126.07441634091694,
         "a0_local": 1.1375295473059159e-10}
    galaxies = FR.load_sparc(os.path.join(THEORY, "sparc_data"), verbose=True)
    names = sorted(galaxies)
    print(f"Loaded {len(names)} galaxies.")
    print()
    print("VALIDATIONS ...", flush=True)
    val = run_validations(galaxies)
    for k, v in val.items():
        print(f"  {k}: {v}")
    print("  all validations PASS")
    print()

    per_gal = {m: [] for m in MODELS}
    bin_edges = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 1e9]
    bin_lab = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1", "1-1.5",
               "1.5-2", "2-3", ">3"]
    rad_models = ["cpqr_2p", "nfw_2p_meanabs", "mond_1p_meanabs",
                  "ccc_1p_std", "ccc_0p_std", "cpqr_0p"]
    rad_pts = {m: [[] for _ in bin_edges[:-1]] for m in rad_models}
    nfw_failures = {"chi2": [], "meanabs": []}
    rhot_rows = []
    at_extrap = {"std": 0, "simple": 0}
    at_vflat_alt = []  # sensitivity: 0-knob score with CPQR V_flat

    for i, name in enumerate(names):
        gal = galaxies[name]
        R, Vo, Ve = gal["R"], gal["Vo"], gal["Ve"]
        Vd, Vb, Vg = gal["Vd"], gal["Vb"], gal["Vg"]
        mask = (Vo > 1.0) & (Ve > 0)
        if mask.sum() < 4:
            continue
        prof = ccc_profile(R, Vd, Vb, Vg)

        # CPQR 2-param (meanabs fit, row 583)
        alpha, r_scale, _ = FR.cpqr_fit_alpha_rscale(gal)
        Vp, _, _, _, r_t, _ = FR.cpqr_compute_v7(R, Vd, Vb, Vg, alpha, r_scale)
        per_gal["cpqr_2p"].append(
            dict(FR.metrics_from_pred(Vp, gal, mask, 2), galaxy=name))
        # CPQR 1-knob tilt (row 586 optimal direction)
        _, Vp_t = OK.cpqr_one_knob_tilt(gal)
        per_gal["cpqr_1p_tilt"].append(
            dict(FR.metrics_from_pred(Vp_t, gal, mask, 1), galaxy=name))
        # CPQR frozen (0 knobs)
        Vp_0 = FR.cpqr_frozen(gal)
        per_gal["cpqr_0p"].append(
            dict(FR.metrics_from_pred(Vp_0, gal, mask, 0), galaxy=name))
        # NFW + MOND by chi2 (gates — verbatim post-script fits)
        r = FR.fit_mond_nfw(gal, C)
        Vp_n = None
        if r["nfw"].get("failed"):
            nfw_failures["chi2"].append(name)
        else:
            Vp_n = FR.nfw_predict(R, Vd, Vb, Vg, r["nfw"]["M200"],
                                  r["nfw"]["conc"], C)
            per_gal["nfw_2p_chi2"].append(
                dict(FR.metrics_from_pred(Vp_n, gal, mask, 2), galaxy=name))
        Vp_m = FR.mond_predict(R, Vd, Vb, Vg, r["mond"]["ups_disk"],
                               C["a0_local"])
        per_gal["mond_1p_chi2"].append(
            dict(FR.metrics_from_pred(Vp_m, gal, mask, 1), galaxy=name))
        # MOND frozen (0 knobs)
        Vp_m0 = FR.mond_predict(R, Vd, Vb, Vg, 0.5, 1.2e-10)
        per_gal["mond_0p"].append(
            dict(FR.metrics_from_pred(Vp_m0, gal, mask, 0), galaxy=name))
        # NFW + MOND REFIT by mean |resid| % (the fair comparison)
        nfw_ma = fit_nfw_meanabs(gal, C)
        Vp_nma = None
        if nfw_ma is None:
            nfw_failures["meanabs"].append(name)
        else:
            Vp_nma = FR.nfw_predict(R, Vd, Vb, Vg, nfw_ma["M200"],
                                    nfw_ma["conc"], C)
            per_gal["nfw_2p_meanabs"].append(
                dict(FR.metrics_from_pred(Vp_nma, gal, mask, 2), galaxy=name))
        ups_ma, _ = fit_mond_meanabs(gal, C)
        Vp_mma = FR.mond_predict(R, Vd, Vb, Vg, ups_ma, C["a0_local"])
        per_gal["mond_1p_meanabs"].append(
            dict(FR.metrics_from_pred(Vp_mma, gal, mask, 1), galaxy=name))
        # CCC 1-knob: fit rho_t by mean |resid| % (both smooth forms)
        logrt_std, _ = ccc_fit_rhot(gal, prof, "std")
        Vp_c1 = ccc_predict_from_profile(prof, 10.0 ** logrt_std, "std")
        per_gal["ccc_1p_std"].append(
            dict(FR.metrics_from_pred(Vp_c1, gal, mask, 1), galaxy=name))
        logrt_sim, _ = ccc_fit_rhot(gal, prof, "simple")
        Vp_c1s = ccc_predict_from_profile(prof, 10.0 ** logrt_sim, "simple")
        per_gal["ccc_1p_simple"].append(
            dict(FR.metrics_from_pred(Vp_c1s, gal, mask, 1), galaxy=name))
        # CCC 0-knob: universal a_t -> rho_t (both forms)
        rt_std, Rt_std, rhob_std, Vf_std, clip_std, ystar_std = \
            ccc_rhot_from_at(gal, prof, "std")
        at_extrap["std"] += int(clip_std)
        Vp_c0 = ccc_predict_from_profile(prof, rt_std, "std")
        per_gal["ccc_0p_std"].append(
            dict(FR.metrics_from_pred(Vp_c0, gal, mask, 0), galaxy=name))
        rt_sim, Rt_sim, rhob_sim, Vf_sim, clip_sim, ystar_sim = \
            ccc_rhot_from_at(gal, prof, "simple")
        at_extrap["simple"] += int(clip_sim)
        Vp_c0s = ccc_predict_from_profile(prof, rt_sim, "simple")
        per_gal["ccc_0p_simple"].append(
            dict(FR.metrics_from_pred(Vp_c0s, gal, mask, 0), galaxy=name))
        # sensitivity: 0-knob std with CPQR V_flat instead of observed
        rt_alt, _, _, _, _, _ = ccc_rhot_from_at(gal, prof, "std",
                                                 use_cpqr_vflat=True)
        Vp_alt = ccc_predict_from_profile(prof, rt_alt, "std")
        at_vflat_alt.append(float(np.mean(
            np.abs(Vp_alt[mask] - Vo[mask]) / Vo[mask])) * 100.0)

        rhot_rows.append({
            "galaxy": name,
            "log10_rhot_std": float(logrt_std),
            "log10_rhot_simple": float(logrt_sim),
            "rhot_1p_std_1e24": float((10.0 ** logrt_std)
                                      * MSUN_KPC3_TO_G_CM3 / 1e-24),
            "rhot_1p_simple_1e24": float((10.0 ** logrt_sim)
                                         * MSUN_KPC3_TO_G_CM3 / 1e-24),
            "rhot_0p_std_1e24": float(rt_std * MSUN_KPC3_TO_G_CM3 / 1e-24),
            "rhot_0p_simple_1e24": float(rt_sim * MSUN_KPC3_TO_G_CM3 / 1e-24),
            "Rt_kpc_std": float(Rt_std),
            "Vflat_obs_kms": float(Vf_std),
            "ystar_std": float(ystar_std),
            "ystar_simple": float(ystar_sim),
            "Rt_extrapolated": bool(clip_std or clip_sim),
        })
        # radius bins (CPQR 2p r_t, as in the fair run)
        preds = {"cpqr_2p": Vp, "nfw_2p_meanabs": Vp_nma,
                 "mond_1p_meanabs": Vp_mma, "ccc_1p_std": Vp_c1,
                 "ccc_0p_std": Vp_c0, "cpqr_0p": Vp_0}
        rr = R[mask] / r_t
        for m in rad_models:
            if preds[m] is None:
                continue
            ap = np.abs(preds[m][mask] - Vo[mask]) / Vo[mask] * 100.0
            for b in range(len(bin_edges) - 1):
                sel = (rr >= bin_edges[b]) & (rr < bin_edges[b + 1])
                rad_pts[m][b].extend(ap[sel].tolist())
        if (i + 1) % 25 == 0 or i + 1 == len(names):
            print(f"  fitted {i + 1}/{len(names)} galaxies ...", flush=True)

    summary = {m: FR.summarize(per_gal[m]) for m in MODELS}
    by_gal = {}
    for m in MODELS:
        for row in per_gal[m]:
            by_gal.setdefault(row["galaxy"], {})[m] = row

    print()
    print("GATES — CPQR-side scoreboard must reproduce the fair run")
    gate_ok = True
    gate_res = {}
    for m, expect in GATES.items():
        got = summary[m]["mean_mean_abs_pct"]
        d = got - expect
        ok_g = abs(d) <= 0.05
        gate_ok &= ok_g
        gate_res[m] = {"got": got, "expected": expect, "diff_pp": d,
                       "pass": bool(ok_g)}
        print(f"  {m:16s} got {got:7.4f}%  expected {expect:7.4f}%  "
              f"diff {d:+.4f} pp  {'PASS' if ok_g else 'FAIL'}")
    print(f"  NFW fit failures chi2/meanabs: "
          f"{nfw_failures['chi2'] or 'none'} / "
          f"{nfw_failures['meanabs'] or 'none'}")
    if not gate_ok:
        print("GATES FAILED — stopping before writing results. Diagnose.")
        with open(os.path.join(HERE, "ccc_forward_fair_run_2026-11-19.json"),
                  "w") as f:
            json.dump({"gates": gate_res, "gate_ok": False,
                       "validations": val}, f, indent=2)
        sys.exit(1)

    # ---- scoreboard, win fractions, radius bins
    def win_frac(a, b):
        wins = sum(1 for g in by_gal
                   if a in by_gal[g] and b in by_gal[g]
                   and by_gal[g][a]["mean_abs_pct"] < by_gal[g][b]["mean_abs_pct"])
        n = sum(1 for g in by_gal if a in by_gal[g] and b in by_gal[g])
        return wins, n, (100.0 * wins / n if n else float("nan"))

    pairs = [("ccc_1p_std", "cpqr_1p_tilt"), ("ccc_1p_std", "mond_1p_meanabs"),
             ("ccc_1p_std", "nfw_2p_meanabs"), ("ccc_1p_std", "cpqr_2p"),
             ("ccc_1p_simple", "cpqr_1p_tilt"),
             ("ccc_0p_std", "cpqr_0p"), ("ccc_0p_std", "mond_0p"),
             ("ccc_0p_simple", "cpqr_0p"),
             ("cpqr_1p_tilt", "nfw_2p_meanabs"),
             ("cpqr_1p_tilt", "mond_1p_meanabs"),
             ("nfw_2p_meanabs", "nfw_2p_chi2"),
             ("mond_1p_meanabs", "mond_1p_chi2")]
    wins = {}
    for a, b in pairs:
        w, n, pct = win_frac(a, b)
        wins[f"{a}_vs_{b}"] = {"wins": w, "n": n, "pct": pct}

    rad = {}
    for m in rad_models:
        rad[m] = [float(np.mean(rad_pts[m][b])) if rad_pts[m][b] else float("nan")
                  for b in range(len(bin_edges) - 1)]

    logrt_std = np.array([r["log10_rhot_std"] for r in rhot_rows])
    logrt_sim = np.array([r["log10_rhot_simple"] for r in rhot_rows])
    rhot0_std = np.array([r["rhot_0p_std_1e24"] for r in rhot_rows])
    rhot0_sim = np.array([r["rhot_0p_simple_1e24"] for r in rhot_rows])

    print()
    print("SCOREBOARD — mean |resid| %, same 175 galaxies, same scoring")
    print("  model                    knobs  mean|err|  chi2_red  RMS    acc")
    for m in MODELS:
        s = summary[m]
        print(f"  {m:24s} {KNOBS[m]:>3d}   {s['mean_mean_abs_pct']:7.4f}%  "
              f"{s['mean_chi2_red']:7.3f}  {s['mean_rms']:6.2f}  "
              f"{s['mean_acc']:6.2f}%   (n={s['n']})")
    print()
    print("WIN FRACTIONS (mean |resid| %, per galaxy)")
    for k, v in wins.items():
        print(f"  {k}: {v['wins']}/{v['n']} = {v['pct']:.1f}%")
    print()
    print("RADIUS BINS — mean |resid| % by R/r_t (CPQR 2p r_t)")
    print("  bins: " + "  ".join(bin_lab))
    for m in rad_models:
        print(f"  {m:18s} " + "  ".join(f"{x:6.2f}" for x in rad[m]))
    print()
    print("CCC FITTED rho_t (1-knob, standard nu): "
          f"median {float(np.median(10.0 ** logrt_std) * MSUN_KPC3_TO_G_CM3 / 1e-24):.2f}e-24 "
          f"g/cm^3 (paper inverse median 2.9), scatter "
          f"{float(np.std(logrt_std)):.2f} dex (paper 0.82)")
    print("CCC FITTED rho_t (1-knob, simple nu): "
          f"median {float(np.median(10.0 ** logrt_sim) * MSUN_KPC3_TO_G_CM3 / 1e-24):.2f}e-24 "
          f"g/cm^3, scatter {float(np.std(logrt_sim)):.2f} dex")
    print("CCC 0-knob rho_t (universal a_t, std): median "
          f"{float(np.median(rhot0_std)):.2f}e-24 g/cm^3, scatter "
          f"{float(np.std(np.log10(rhot0_std))):.2f} dex")
    print(f"R_t extrapolated outside measured range: "
          f"std {at_extrap['std']}, simple {at_extrap['simple']} galaxies")
    print(f"0-knob std score with CPQR V_flat instead of observed Vflat: "
          f"{float(np.mean(at_vflat_alt)):.4f}% (primary: "
          f"{summary['ccc_0p_std']['mean_mean_abs_pct']:.4f}%)")
    print(f"elapsed {time.time() - t0:.0f} s")

    result = {
        "date": "2026-11-19",
        "paper": "Gupta & Samaras 2026, MNRAS 552, arXiv:2608.11575",
        "footing": {
            "galaxies": 175, "mask": "Vo>1 & Verr>0, n>=4",
            "errors": "sqrt(Verr^2 + (0.05*Vobs)^2)",
            "M/L": "disk 0.5 / bulge 0.7",
            "fit_loss": "mean |resid| % for EVERY fitted model",
            "chi2": "secondary metric only",
            "ccc_law": "rho_o = rho_b*nu(rho_b/rho_t); std nu=1/(1-exp(-sqrt(y))), "
                       "simple nu=1/2+sqrt(1/4+1/y)",
            "ccc_0knob": "universal a_t = 2.14e-10 m/s^2 (App. B median); "
                         "R_t = Vflat_obs^2/a_t; rho_t solves "
                         "rho_t = rho_b(R_t)*nu(rho_b(R_t)/rho_t)",
        },
        "validations": val,
        "gates": gate_res,
        "gate_ok": True,
        "scoreboard": {m: {
            "knobs": KNOBS[m],
            "mean_abs_pct": summary[m]["mean_mean_abs_pct"],
            "median_mean_abs_pct": summary[m]["median_mean_abs_pct"],
            "mean_chi2_red": summary[m]["mean_chi2_red"],
            "median_chi2_red": summary[m]["median_chi2_red"],
            "mean_rms": summary[m]["mean_rms"],
            "mean_acc": summary[m]["mean_acc"],
            "n": summary[m]["n"],
        } for m in MODELS},
        "win_fractions": wins,
        "radius_bins": {"labels": bin_lab, "means": rad},
        "ccc_rhot_1p_std_median_1e24": float(
            np.median(10.0 ** logrt_std) * MSUN_KPC3_TO_G_CM3 / 1e-24),
        "ccc_rhot_1p_std_scatter_dex": float(np.std(logrt_std)),
        "ccc_rhot_1p_simple_median_1e24": float(
            np.median(10.0 ** logrt_sim) * MSUN_KPC3_TO_G_CM3 / 1e-24),
        "ccc_rhot_1p_simple_scatter_dex": float(np.std(logrt_sim)),
        "ccc_rhot_0p_std_median_1e24": float(np.median(rhot0_std)),
        "ccc_rhot_0p_std_scatter_dex": float(np.std(np.log10(rhot0_std))),
        "ccc_rhot_0p_simple_median_1e24": float(np.median(rhot0_sim)),
        "ccc_rhot_0p_simple_scatter_dex": float(np.std(np.log10(rhot0_sim))),
        "Rt_extrapolated_counts": at_extrap,
        "ccc_0p_std_with_cpqr_vflat_mean_abs_pct": float(np.mean(at_vflat_alt)),
        "nfw_failures": nfw_failures,
        "per_galaxy": [{**r,
                        **{m: by_gal[r["galaxy"]][m]["mean_abs_pct"]
                           for m in MODELS if m in by_gal[r["galaxy"]]}}
                       for r in rhot_rows],
        "elapsed_s": float(time.time() - t0),
    }
    with open(os.path.join(HERE, "ccc_forward_fair_run_2026-11-19.json"),
              "w") as f:
        json.dump(result, f, indent=2)
    print("wrote ccc_forward_fair_run_2026-11-19.json")


if __name__ == "__main__":
    main()
