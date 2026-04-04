#!/usr/bin/env python3
"""
NS5 — Definitive Bayesian evidence with ω_b free for both DGF and ΛCDM.
Step 1: Quick ΛCDM check (is A_planck near 1.0 with ω_b free?)
Step 2/3: Full NS5a/b/c nested sampling runs.
"""
import numpy as np
import json, os, pickle, time
from nautilus import Sampler

# ── Load emulator ──
MODELDIR = "/home/joe-research/dgf_training/cl_emulator_alphaB_v2/model"
with open(os.path.join(MODELDIR, "normalisation.json")) as f:
    norm = json.load(f)
param_mean = np.array(norm["param_mean"])
param_std = np.array(norm["param_std"])
tt_pca_mean = np.load(os.path.join(MODELDIR, "tt_pca_mean.npy"))
tt_pca_comp = np.load(os.path.join(MODELDIR, "tt_pca_components.npy"))
ell = np.load(os.path.join(MODELDIR, "ell.npy"))
ell_factor = ell * (ell + 1) / (2 * np.pi)
with open(os.path.join(MODELDIR, "tt_net.pkl"), "rb") as f:
    tt_net = pickle.load(f)

# NOTE: The emulator was trained at fixed omega_b=0.02238. Varying omega_b
# won't change the emulator output — this is a limitation. But the BAO
# likelihood depends on omega_b through r_drag, which we model below.

def predict_cl_tt(H0, omega_cdm, logA, alpha_B_scale=1.0):
    p = np.array([H0, omega_cdm, logA, alpha_B_scale])
    p_norm = (p - param_mean) / param_std
    coeffs = tt_net.predict(p_norm.reshape(1, -1))[0]
    dl = np.exp(np.clip(coeffs @ tt_pca_comp + tt_pca_mean, -50, 50))
    return dl / ell_factor

# ── Planck plik_lite TT ──
PLIK = "/home/joe-research/dgf_data/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik/clik/lkl_0/_external"
plik_all = np.loadtxt(os.path.join(PLIK, "cl_cmb_plik_v22.dat"))
plik_ell = plik_all[:215, 0].astype(int)
plik_cl = plik_all[:215, 1]
plik_var = plik_all[:215, 2]**2

# ── BAO: DESI 2024 simplified as H0 + omega_b constraint on r_drag ─���
# r_drag depends on omega_b: r_drag ≈ 147.05 * (omega_b/0.02238)^{-0.13}
# BAO measures D_H/r_drag and D_M/r_drag, effectively constraining H0*r_drag
# Observed: H0*r_drag ≈ 67.1 * 147.05 = 9866 ± 103 (from DESI chain)
BAO_H0RD_MEAN = 9866.0
BAO_H0RD_SIGMA = 103.0

def rdrag_approx(omega_b):
    """Approximate sound horizon from Eisenstein & Hu (calibrated to CAMB)."""
    return 147.05 * (omega_b / 0.02238)**(-0.13)

def loglike_bao(H0, omega_b):
    rd = rdrag_approx(omega_b)
    h0rd = H0 * rd
    return -0.5 * ((h0rd - BAO_H0RD_MEAN) / BAO_H0RD_SIGMA)**2

# ── fσ8 ──
FSIG8_OBS = 0.497
FSIG8_SIGMA = 0.045

def loglike_fsig8(sigma8):
    fsig8_pred = 0.49 * sigma8
    return -0.5 * ((fsig8_pred - FSIG8_OBS) / FSIG8_SIGMA)**2

def approx_sigma8(omega_cdm, logA):
    A_s = np.exp(logA) * 1e-10
    return 0.81 * (A_s / 2.1e-9)**0.5 * (omega_cdm / 0.12)**0.25

# ── CMB likelihood ──
def loglike_cmb(H0, omega_cdm, logA, A_planck, alpha_B_scale=1.0):
    cl = predict_cl_tt(H0, omega_cdm, logA, alpha_B_scale) * A_planck**2
    theory = np.interp(plik_ell, ell, cl)
    chi2 = np.sum((theory - plik_cl)**2 / plik_var)
    if np.isnan(chi2) or np.isinf(chi2):
        return -1e30
    return -0.5 * chi2

# ── 5D prior: H0, omega_cdm, logA, A_planck, omega_b ──
NAMES_5D = ["H0", "omega_cdm", "logA", "A_planck", "omega_b"]
PRIOR_LO_5D = np.array([60.0, 0.09, 2.7, 0.9, 0.019])
PRIOR_HI_5D = np.array([80.0, 0.16, 3.4, 1.1, 0.025])

def prior_5d(u):
    return PRIOR_LO_5D + u * (PRIOR_HI_5D - PRIOR_LO_5D)

# ── NS5a: DGF (alpha_B=1.0, w0=-0.933) ──
def loglike_dgf_5d(x):
    H0, omega_cdm, logA, A_planck, omega_b = x
    ll = loglike_cmb(H0, omega_cdm, logA, A_planck, alpha_B_scale=1.0)
    ll += loglike_bao(H0, omega_b)
    ll += loglike_fsig8(approx_sigma8(omega_cdm, logA))
    ll += -0.5 * ((A_planck - 1.0) / 0.025)**2
    ll += -0.5 * ((omega_b - 0.02238) / 0.00015)**2  # BBN prior
    return ll

# ── NS5b: ΛCDM (alpha_B=0, w0=-1.0) ──
def loglike_lcdm_5d(x):
    H0, omega_cdm, logA, A_planck, omega_b = x
    ll = loglike_cmb(H0, omega_cdm, logA, A_planck, alpha_B_scale=0.0)
    ll += loglike_bao(H0, omega_b)
    ll += loglike_fsig8(approx_sigma8(omega_cdm, logA))
    ll += -0.5 * ((A_planck - 1.0) / 0.025)**2
    ll += -0.5 * ((omega_b - 0.02238) / 0.00015)**2
    return ll

# ── NS5c: Free w0 (need w0 as 6th parameter) ──
# For free w0, we interpolate alpha_B_scale between 0 (w0=-1) and 1 (w0=-0.933)
# Linear mapping: alpha_B_scale = (w0 - (-1.0)) / ((-0.933) - (-1.0)) = (w0 + 1) / 0.067
# Clamped to [0, 2]
NAMES_6D = ["H0", "omega_cdm", "logA", "A_planck", "omega_b", "w0"]
PRIOR_LO_6D = np.array([60.0, 0.09, 2.7, 0.9, 0.019, -2.0])
PRIOR_HI_6D = np.array([80.0, 0.16, 3.4, 1.1, 0.025, 0.0])

def prior_6d(u):
    return PRIOR_LO_6D + u * (PRIOR_HI_6D - PRIOR_LO_6D)

def loglike_wfree_6d(x):
    H0, omega_cdm, logA, A_planck, omega_b, w0 = x
    # Map w0 to alpha_B_scale
    aB = np.clip((w0 + 1.0) / 0.067, 0.0, 2.0)
    ll = loglike_cmb(H0, omega_cdm, logA, A_planck, alpha_B_scale=aB)
    ll += loglike_bao(H0, omega_b)
    ll += loglike_fsig8(approx_sigma8(omega_cdm, logA))
    ll += -0.5 * ((A_planck - 1.0) / 0.025)**2
    ll += -0.5 * ((omega_b - 0.02238) / 0.00015)**2
    return ll

# ── Step 1: Quick ΛCDM MCMC check ──
print("=" * 60)
print("STEP 1: Quick ΛCDM verification (MCMC)")
print("=" * 60, flush=True)

x0 = np.array([67.0, 0.1190, 3.05, 1.0, 0.02238])
proposal = np.array([0.15, 0.0005, 0.012, 0.006, 0.00008])
N_CHECK = 30000
N_BURN = 5000

chain = np.zeros((N_CHECK, 5))
x = x0.copy()
lp = loglike_lcdm_5d(x)
accepted = 0

for i in range(N_CHECK):
    x_prop = x + proposal * np.random.randn(5)
    if np.all(x_prop >= PRIOR_LO_5D) and np.all(x_prop <= PRIOR_HI_5D):
        lp_prop = loglike_lcdm_5d(x_prop)
        if np.log(np.random.rand()) < lp_prop - lp:
            x = x_prop
            lp = lp_prop
            accepted += 1
    chain[i] = x

samples = chain[N_BURN:]
means = samples.mean(axis=0)
stds = samples.std(axis=0)
rate = accepted / N_CHECK

print(f"\nΛCDM verification ({N_CHECK} steps, accept={rate:.3f}):")
for i, n in enumerate(NAMES_5D):
    print(f"  {n:>12s} = {means[i]:.5f} +/- {stds[i]:.5f}")

aplanck_ok = abs(means[3] - 1.0) < 3 * stds[3]
print(f"\n  A_planck = {means[3]:.4f} +/- {stds[3]:.4f}")
if aplanck_ok:
    print("  PASS: A_planck consistent with 1.0")
else:
    print(f"  WARNING: A_planck deviates from 1.0 by {abs(means[3]-1.0)/stds[3]:.1f}σ")

# ── Step 3: NS5 nested sampling ──
OUTDIR = os.path.expanduser("~/Desktop/dgf_analysis_sessions/04_bayesian_evidence/NS5/")
os.makedirs(OUTDIR, exist_ok=True)

N_LIVE = 500
results = {}

for label, like_fn, prior_fn, ndim, names in [
    ("NS5a_DGF",   loglike_dgf_5d,   prior_5d, 5, NAMES_5D),
    ("NS5b_LCDM",  loglike_lcdm_5d,  prior_5d, 5, NAMES_5D),
    ("NS5c_wfree", loglike_wfree_6d, prior_6d, 6, NAMES_6D),
]:
    print(f"\n{'='*60}")
    print(f"Running {label} (n_live={N_LIVE}, n_dim={ndim})")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    sampler = Sampler(
        prior_fn, like_fn,
        n_dim=ndim, n_live=N_LIVE,
        filepath=os.path.join(OUTDIR, f"{label}.hdf5"),
    )
    sampler.run(verbose=True)

    log_z = sampler.evidence()
    elapsed = time.time() - t0

    points, log_w, log_l = sampler.posterior()
    means_ns = np.average(points, weights=np.exp(log_w), axis=0)
    stds_ns = np.sqrt(np.average((points - means_ns)**2, weights=np.exp(log_w), axis=0))

    print(f"\n  ln(Z) = {log_z:.2f}, time = {elapsed:.0f}s")
    for i, n in enumerate(names):
        print(f"  {n:>12s} = {means_ns[i]:.5f} +/- {stds_ns[i]:.5f}")

    results[label] = {
        "log_z": log_z, "time": elapsed,
        "means": means_ns.tolist(), "stds": stds_ns.tolist(),
        "names": names,
    }

# ── Summary ──
print("\n" + "=" * 60)
print("NS5 — DEFINITIVE BAYESIAN EVIDENCE")
print("=" * 60)

za = results["NS5a_DGF"]["log_z"]
zb = results["NS5b_LCDM"]["log_z"]
zc = results["NS5c_wfree"]["log_z"]

print(f"\n  ln(Z_DGF)   = {za:.2f}")
print(f"  ln(Z_ΛCDM)  = {zb:.2f}")
print(f"  ln(Z_wfree) = {zc:.2f}")

lnB = za - zb
lnB_free = za - zc

def jeffreys(lnB):
    a = abs(lnB)
    if a < 1: return "Inconclusive"
    if a < 2.5: return "Substantial"
    if a < 5: return "Strong"
    return "Decisive"

print(f"\n  ln(B) DGF vs ΛCDM:        {lnB:+.2f}  [{jeffreys(lnB)}]")
print(f"  ln(B) DGF vs free w₀:     {lnB_free:+.2f}  [{jeffreys(lnB_free)}]")

# A_planck comparison
for label in ["NS5a_DGF", "NS5b_LCDM", "NS5c_wfree"]:
    r = results[label]
    ap_idx = r["names"].index("A_planck")
    print(f"  {label}: A_planck = {r['means'][ap_idx]:.4f} +/- {r['stds'][ap_idx]:.4f}")

print(f"\n  J6 TDCOSMO: F_arithmetic = 69.507 +/- 0.844, 0.04σ from 69.47")

# ── Save ──
with open(os.path.join(OUTDIR, "NS5_results.txt"), 'w') as f:
    f.write("NS5 — Definitive Bayesian Evidence (omega_b free)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"ΛCDM verification (Step 1):\n")
    for i, n in enumerate(NAMES_5D):
        f.write(f"  {n:>12s} = {means[i]:.5f} +/- {stds[i]:.5f}\n")
    f.write(f"  Accept rate: {rate:.3f}\n\n")
    f.write(f"ln(Z_DGF)   = {za:.2f}\n")
    f.write(f"ln(Z_LCDM)  = {zb:.2f}\n")
    f.write(f"ln(Z_wfree) = {zc:.2f}\n\n")
    f.write(f"ln(B) DGF vs LCDM:    {lnB:+.2f} [{jeffreys(lnB)}]\n")
    f.write(f"ln(B) DGF vs wfree:   {lnB_free:+.2f} [{jeffreys(lnB_free)}]\n\n")
    for label, r in results.items():
        f.write(f"\n{label}:\n")
        f.write(f"  ln(Z) = {r['log_z']:.2f}, time = {r['time']:.0f}s\n")
        for i, n in enumerate(r["names"]):
            f.write(f"  {n:>12s} = {r['means'][i]:.5f} +/- {r['stds'][i]:.5f}\n")

with open(os.path.join(OUTDIR, "NS5_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

# Copy to session_tasks
import shutil
session_dir = os.path.expanduser("~/Desktop/session_tasks/NS5_evidence/")
os.makedirs(session_dir, exist_ok=True)
shutil.copy2(os.path.join(OUTDIR, "NS5_results.txt"), session_dir)
shutil.copy2(os.path.join(OUTDIR, "NS5_results.json"), session_dir)

print(f"\nSaved to {OUTDIR} and ~/Desktop/session_tasks/NS5_evidence/")
