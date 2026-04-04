#!/usr/bin/env python3
"""
NS4 — Final Bayesian evidence calculation.
Nautilus nested sampling with PCA emulator + plik_lite TT + DESI BAO.

NS4a: DGF (w0=-0.933 fixed)
NS4b: LCDM (w0=-1.0 fixed)
NS4c: DGF + Flash Field B3 constraint
NS4d: DGF + B3 + w0 tension constraint
"""
import numpy as np
import json, os, pickle, time
from nautilus import Sampler

# ── Load emulator (the merged v2 with α_B) ──
# We use it at α_B_scale=1.0 for DGF fiducial
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

def predict_cl_tt(H0, omega_cdm, logA, alpha_B_scale=1.0):
    p = np.array([H0, omega_cdm, logA, alpha_B_scale])
    p_norm = (p - param_mean) / param_std
    coeffs = tt_net.predict(p_norm.reshape(1, -1))[0]
    dl = np.exp(coeffs @ tt_pca_comp + tt_pca_mean)
    return dl / ell_factor  # return C_l

# ── Load Planck plik_lite TT ──
PLIK = "/home/joe-research/dgf_data/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik/clik/lkl_0/_external"
plik_all = np.loadtxt(os.path.join(PLIK, "cl_cmb_plik_v22.dat"))
N_TT = 215
plik_ell = plik_all[:N_TT, 0].astype(int)
plik_cl = plik_all[:N_TT, 1]
plik_var = plik_all[:N_TT, 2]**2

# ── DESI BAO data (simplified: DM/rd and DH/rd at effective redshifts) ──
# From DESI 2024: key constraints on DM(z)/rd and DH(z)/rd
# We use a simplified BAO chi2 based on H0 and omega_cdm
# DGF rdrag = 147.056 Mpc (from hi_class)
RDRAG_DGF = 147.056

# DESI measured DH/rd and DM/rd — simplified to H0 constraint
# The BAO effectively constrains H0*rd and Omega_m*h^2
# For a simplified likelihood: BAO prefers H0 ~ 67.1 +/- 0.7 (from our DESI chain)
BAO_H0_MEAN = 67.1
BAO_H0_SIGMA = 0.7

def loglike_bao(H0):
    return -0.5 * ((H0 - BAO_H0_MEAN) / BAO_H0_SIGMA)**2

# ── fσ8 data (simplified) ──
# From DESI+BOSS: fσ8(z=0.38) = 0.497 +/- 0.045
FSIG8_OBS = 0.497
FSIG8_SIGMA = 0.045

def loglike_fsig8(sigma8, Omega_m):
    # f(z=0.38) ≈ Omega_m(z)^0.55, σ8(z) ≈ σ8 * D(z)/D(0)
    # Simplified: fσ8 ��� 0.5 * sigma8 for DGF-like cosmologies
    fsig8_pred = 0.49 * sigma8  # calibrated to match J2 chi2_fsig8 = 1.4
    return -0.5 * ((fsig8_pred - FSIG8_OBS) / FSIG8_SIGMA)**2

# ── CMB likelihood ──
def loglike_cmb(H0, omega_cdm, logA, A_planck, alpha_B_scale=1.0):
    cl = predict_cl_tt(H0, omega_cdm, logA, alpha_B_scale) * A_planck**2
    theory = np.interp(plik_ell, ell, cl)
    chi2 = np.sum((theory - plik_cl)**2 / plik_var)
    if np.isnan(chi2) or np.isinf(chi2):
        return -1e30
    return -0.5 * chi2

# ── Approximate sigma8 from logA and omega_cdm ──
def approx_sigma8(omega_cdm, logA):
    # Calibrated linear relation from chain results
    A_s = np.exp(logA) * 1e-10
    return 0.81 * (A_s / 2.1e-9)**0.5 * (omega_cdm / 0.12)**0.25

# ── Flash Field constraint ──
H0_MATTER_MEAN = 72.45  # from J6
F_TARGET = 69.47
F_SIGMA = 0.76  # from J6 posterior

def loglike_flash_field(H0):
    F_chain = (H0 + H0_MATTER_MEAN) / 2.0
    return -0.5 * ((F_chain - F_TARGET) / F_SIGMA)**2

# ── w0 tension constraint ──
W0_OBS = -0.879   # from J4
W0_SIGMA = 0.029

def loglike_w0(w0_model):
    return -0.5 * ((w0_model - W0_OBS) / W0_SIGMA)**2

# ── Prior bounds ──
PRIOR_LO = np.array([60.0, 0.09, 2.7, 0.9])
PRIOR_HI = np.array([80.0, 0.16, 3.4, 1.1])
NAMES = ["H0", "omega_cdm", "logA", "A_planck"]

def prior_transform(u):
    """Map unit cube to parameter space."""
    x = PRIOR_LO + u * (PRIOR_HI - PRIOR_LO)
    return x

# ── NS4a: DGF baseline ──
def loglike_dgf(x):
    H0, omega_cdm, logA, A_planck = x
    ll = loglike_cmb(H0, omega_cdm, logA, A_planck, alpha_B_scale=1.0)
    ll += loglike_bao(H0)
    s8 = approx_sigma8(omega_cdm, logA)
    Om = (0.02238 + omega_cdm) / (H0/100)**2
    ll += loglike_fsig8(s8, Om)
    ll += -0.5 * ((A_planck - 1.0) / 0.025)**2  # A_planck prior
    return ll

# ── NS4b: LCDM (use α_B_scale=0 for no braiding) ���─
def loglike_lcdm(x):
    H0, omega_cdm, logA, A_planck = x
    ll = loglike_cmb(H0, omega_cdm, logA, A_planck, alpha_B_scale=0.0)
    ll += loglike_bao(H0)
    s8 = approx_sigma8(omega_cdm, logA)
    Om = (0.02238 + omega_cdm) / (H0/100)**2
    ll += loglike_fsig8(s8, Om)
    ll += -0.5 * ((A_planck - 1.0) / 0.025)**2
    return ll

# ── NS4c: DGF + Flash Field ──
def loglike_dgf_b3(x):
    ll = loglike_dgf(x)
    ll += loglike_flash_field(x[0])
    return ll

# ── NS4d: DGF + Flash Field + w0 constraint ──
def loglike_dgf_full(x):
    ll = loglike_dgf_b3(x)
    ll += loglike_w0(-0.933)  # DGF w0 is fixed, evaluate tension term
    return ll

# ── Run all four ──
OUTDIR = os.path.expanduser("~/Desktop/dgf_analysis_sessions/04_bayesian_evidence/NS4/")
os.makedirs(OUTDIR, exist_ok=True)

N_LIVE = 500
results = {}

for label, like_fn in [("NS4a_DGF", loglike_dgf),
                         ("NS4b_LCDM", loglike_lcdm),
                         ("NS4c_DGF_B3", loglike_dgf_b3),
                         ("NS4d_DGF_full", loglike_dgf_full)]:
    print(f"\n{'='*60}")
    print(f"Running {label} (n_live={N_LIVE})")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    sampler = Sampler(
        prior_transform, like_fn,
        n_dim=4, n_live=N_LIVE,
        filepath=os.path.join(OUTDIR, f"{label}.hdf5"),
    )
    sampler.run(verbose=True)

    log_z = sampler.evidence()
    elapsed = time.time() - t0

    # Get posterior samples
    points, log_w, log_l = sampler.posterior()
    means = np.average(points, weights=np.exp(log_w), axis=0)
    stds = np.sqrt(np.average((points - means)**2, weights=np.exp(log_w), axis=0))

    print(f"\n  ln(Z) = {log_z:.2f}")
    print(f"  Time: {elapsed:.0f}s")
    for i, n in enumerate(NAMES):
        print(f"  {n:>12s} = {means[i]:.4f} +/- {stds[i]:.4f}")

    results[label] = {"log_z": log_z, "time": elapsed, "means": means.tolist(), "stds": stds.tolist()}

# ── Summary ──
print("\n" + "=" * 60)
print("NS4 — FINAL BAYESIAN EVIDENCE SUMMARY")
print("=" * 60)

za = results["NS4a_DGF"]["log_z"]
zb = results["NS4b_LCDM"]["log_z"]
zc = results["NS4c_DGF_B3"]["log_z"]
zd = results["NS4d_DGF_full"]["log_z"]

print(f"\n  ln(Z_DGF)       = {za:.2f}  (NS4a)")
print(f"  ln(Z_LCDM)      = {zb:.2f}  (NS4b)")
print(f"  ln(Z_DGF+B3)    = {zc:.2f}  (NS4c)")
print(f"  ln(Z_DGF_full)  = {zd:.2f}  (NS4d)")

lnB_base = za - zb
lnB_b3 = zc - zb
lnB_full = zd - zb

def jeffreys(lnB):
    a = abs(lnB)
    if a < 1: return "Inconclusive"
    if a < 2.5: return "Substantial"
    if a < 5: return "Strong"
    return "Decisive"

print(f"\n  ln(B) DGF vs LCDM (baseline):     {lnB_base:+.2f}  [{jeffreys(lnB_base)}]")
print(f"  ln(B) DGF+B3 vs LCDM:             {lnB_b3:+.2f}  [{jeffreys(lnB_b3)}]")
print(f"  ln(B) DGF_full vs LCDM:            {lnB_full:+.2f}  [{jeffreys(lnB_full)}]")
print(f"\n  Improvement from Flash Field:      {lnB_b3 - lnB_base:+.2f}")
print(f"  Improvement from w0 constraint:    {lnB_full - lnB_b3:+.2f}")
print(f"  Total improvement over baseline:   {lnB_full - lnB_base:+.2f}")

print(f"\n  J6 TDCOSMO confirmation: F_arithmetic = 69.507 +/- 0.844")
print(f"  Distance from 69.47: 0.04sigma")

# Save
with open(os.path.join(OUTDIR, "NS4_results.txt"), 'w') as f:
    f.write("NS4 �� Final Bayesian Evidence\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"ln(Z_DGF)       = {za:.2f}\n")
    f.write(f"ln(Z_LCDM)      = {zb:.2f}\n")
    f.write(f"ln(Z_DGF+B3)    = {zc:.2f}\n")
    f.write(f"ln(Z_DGF_full)  = {zd:.2f}\n\n")
    f.write(f"ln(B) baseline:  {lnB_base:+.2f} [{jeffreys(lnB_base)}]\n")
    f.write(f"ln(B) +B3:       {lnB_b3:+.2f} [{jeffreys(lnB_b3)}]\n")
    f.write(f"ln(B) full:      {lnB_full:+.2f} [{jeffreys(lnB_full)}]\n\n")
    for label, r in results.items():
        f.write(f"\n{label}:\n")
        f.write(f"  ln(Z) = {r['log_z']:.2f}, time = {r['time']:.0f}s\n")
        for i, n in enumerate(NAMES):
            f.write(f"  {n:>12s} = {r['means'][i]:.4f} +/- {r['stds'][i]:.4f}\n")

with open(os.path.join(OUTDIR, "NS4_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUTDIR}")

# Copy to session_tasks
import shutil
session_dir = os.path.expanduser("~/Desktop/session_tasks/NS4_evidence/")
os.makedirs(session_dir, exist_ok=True)
shutil.copy2(os.path.join(OUTDIR, "NS4_results.txt"), session_dir)
shutil.copy2(os.path.join(OUTDIR, "NS4_results.json"), session_dir)
print(f"Copied to {session_dir}")
