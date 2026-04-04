#!/usr/bin/env python3
"""
GPU Planck chain with α_B as free parameter.
Uses PCA+NN emulator trained on hi_class C_l with varying alpha_B.
Extracts α = dH₀/dα_B from the posterior covariance matrix.
"""
import numpy as np
import json, os, time, shutil

# ── Load emulator ──
MODELDIR = "/home/joe-research/dgf_training/cl_emulator_alphaB/model"

print("Loading emulator...")
with open(os.path.join(MODELDIR, "normalisation.json")) as f:
    norm = json.load(f)

param_mean = np.array(norm["param_mean"])
param_std = np.array(norm["param_std"])
N_PCA = norm["n_pca"]

tt_pca_mean = np.load(os.path.join(MODELDIR, "tt_pca_mean.npy"))
tt_pca_comp = np.load(os.path.join(MODELDIR, "tt_pca_components.npy"))
ee_pca_mean = np.load(os.path.join(MODELDIR, "ee_pca_mean.npy"))
ee_pca_comp = np.load(os.path.join(MODELDIR, "ee_pca_components.npy"))
te_pca_mean = np.load(os.path.join(MODELDIR, "te_pca_mean.npy"))
te_pca_comp = np.load(os.path.join(MODELDIR, "te_pca_components.npy"))
ell = np.load(os.path.join(MODELDIR, "ell.npy"))

# Try torch first, fall back to sklearn
USE_TORCH = False
try:
    import torch
    import torch.nn as nn

    class PCANet(nn.Module):
        def __init__(self, n_in, n_out, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, n_out),
            )
        def forward(self, x):
            return self.net(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_params = norm["n_params"]

    tt_net = PCANet(n_params, N_PCA).to(device)
    tt_net.load_state_dict(torch.load(os.path.join(MODELDIR, "tt_net.pt"), map_location=device))
    tt_net.eval()

    ee_net = PCANet(n_params, N_PCA).to(device)
    ee_net.load_state_dict(torch.load(os.path.join(MODELDIR, "ee_net.pt"), map_location=device))
    ee_net.eval()

    te_net = PCANet(n_params, N_PCA).to(device)
    te_net.load_state_dict(torch.load(os.path.join(MODELDIR, "te_net.pt"), map_location=device))
    te_net.eval()

    tt_coeff_mean = np.array(norm["tt_coeff_mean"])
    tt_coeff_std = np.array(norm["tt_coeff_std"])
    ee_coeff_mean = np.array(norm["ee_coeff_mean"])
    ee_coeff_std = np.array(norm["ee_coeff_std"])
    te_coeff_mean = np.array(norm["te_coeff_mean"])
    te_coeff_std = np.array(norm["te_coeff_std"])

    USE_TORCH = True
    print(f"  PyTorch emulator on {device}")

except Exception as e:
    print(f"  PyTorch unavailable ({e}), trying sklearn...")
    import pickle
    with open(os.path.join(MODELDIR, "tt_net.pkl"), 'rb') as f:
        tt_net = pickle.load(f)
    with open(os.path.join(MODELDIR, "ee_net.pkl"), 'rb') as f:
        ee_net = pickle.load(f)
    with open(os.path.join(MODELDIR, "te_net.pkl"), 'rb') as f:
        te_net = pickle.load(f)
    print("  sklearn emulator loaded")


def predict_cl(H0, omega_cdm, logA, alpha_B_scale):
    """Predict C_l^TT, EE, TE for given parameters."""
    p = np.array([H0, omega_cdm, logA, alpha_B_scale])
    p_norm = (p - param_mean) / param_std

    if USE_TORCH:
        with torch.no_grad():
            x = torch.tensor(p_norm, dtype=torch.float32).unsqueeze(0).to(device)
            tt_coeffs = tt_net(x).cpu().numpy()[0] * tt_coeff_std + tt_coeff_mean
            ee_coeffs = ee_net(x).cpu().numpy()[0] * ee_coeff_std + ee_coeff_mean
            te_coeffs = te_net(x).cpu().numpy()[0] * te_coeff_std + te_coeff_mean
    else:
        tt_coeffs = tt_net.predict(p_norm.reshape(1, -1))[0]
        ee_coeffs = ee_net.predict(p_norm.reshape(1, -1))[0]
        te_coeffs = te_net.predict(p_norm.reshape(1, -1))[0]

    cl_tt = np.exp(tt_coeffs @ tt_pca_comp + tt_pca_mean)
    cl_ee = np.exp(ee_coeffs @ ee_pca_comp + ee_pca_mean)
    cl_te = te_coeffs @ te_pca_comp + te_pca_mean

    return cl_tt, cl_ee, cl_te


# ── Load Planck plik_lite data ──
print("Loading Planck plik_lite TTTEEE...")
PLIK = "/home/joe-research/dgf_data/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik/clik/lkl_0/_external"

# Binned data: columns = (l_center, C_l, sigma)
# First 215 bins are TT, then EE, then TE
plik_all = np.loadtxt(os.path.join(PLIK, "cl_cmb_plik_v22.dat"))
# Use only TT bins (0:215)
N_TT = 215
plik_data = plik_all[:N_TT]
plik_ell = plik_data[:, 0].astype(int)
plik_cl = plik_data[:, 1]     # Observed C_l (TT)
plik_sigma = plik_data[:, 2]  # 1-sigma errors
plik_var = plik_sigma**2

print(f"  {N_TT} TT bins (l={plik_ell[0]}..{plik_ell[-1]}), diagonal errors")


# Convert factor: plik stores C_l, emulator outputs D_l = l(l+1)/(2pi)*C_l
# So C_l = D_l * 2pi / (l*(l+1))
ell_factor = ell * (ell + 1) / (2 * np.pi)  # for emulator ell array

def planck_loglike(H0, omega_cdm, logA, alpha_B_scale, A_planck):
    """Compute Planck plik_lite TT log-likelihood (binned)."""
    dl_tt, dl_ee, dl_te = predict_cl(H0, omega_cdm, logA, alpha_B_scale)

    # Convert D_l -> C_l and apply A_planck calibration
    cl_tt = dl_tt / ell_factor * A_planck**2

    # Interpolate emulator C_l to plik bin centers
    theory_tt = np.interp(plik_ell, ell, cl_tt)

    # Residual against observed C_l
    residual = theory_tt - plik_cl

    # Diagonal chi2
    chi2 = np.sum(residual**2 / plik_var)
    if np.isnan(chi2) or np.isinf(chi2):
        return -1e30

    return -0.5 * chi2


# ── MCMC sampler (Metropolis-Hastings) ──
print("\nRunning MCMC...")

# Parameters: H0, omega_cdm, logA, alpha_B_scale, A_planck
param_names = ["H0", "omega_cdm", "logA", "alpha_B_scale", "A_planck"]
n_par = len(param_names)

# Starting point (from first run best-fit)
x0 = np.array([65.75, 0.1192, 3.086, 1.35, 0.979])

# Proposal widths (tuned from first run)
proposal = np.array([0.15, 0.0005, 0.015, 0.07, 0.007])

# Priors
prior_lo = np.array([60.0, 0.09, 2.7, 0.0, 0.9])
prior_hi = np.array([80.0, 0.16, 3.4, 2.0, 1.1])
# Gaussian prior on A_planck: N(1.0, 0.025)
A_planck_prior_mean = 1.0
A_planck_prior_sigma = 0.025

def log_prior(x):
    if np.any(x < prior_lo) or np.any(x > prior_hi):
        return -np.inf
    # Gaussian prior on A_planck
    lp = -0.5 * ((x[4] - A_planck_prior_mean) / A_planck_prior_sigma)**2
    return lp

N_SAMPLES = 100000
N_BURNIN = 10000

chain = np.zeros((N_SAMPLES, n_par))
logpost = np.zeros(N_SAMPLES)
accepted = 0

x = x0.copy()
lp = log_prior(x) + planck_loglike(*x)

t0 = time.time()

for i in range(N_SAMPLES):
    # Propose
    x_prop = x + proposal * np.random.randn(n_par)
    lp_prop = log_prior(x_prop)

    if lp_prop > -np.inf:
        lp_prop += planck_loglike(*x_prop)

    # Accept/reject
    if np.log(np.random.rand()) < lp_prop - lp:
        x = x_prop
        lp = lp_prop
        accepted += 1

    chain[i] = x
    logpost[i] = lp

    if (i + 1) % 5000 == 0:
        rate = accepted / (i + 1)
        elapsed = time.time() - t0
        print(f"  Step {i+1}/{N_SAMPLES}: accept rate {rate:.3f}, "
              f"elapsed {elapsed:.0f}s, logpost = {lp:.1f}")

elapsed = time.time() - t0
print(f"\nMCMC complete: {N_SAMPLES} steps in {elapsed:.1f}s ({elapsed/N_SAMPLES*1000:.1f} ms/step)")
print(f"  Acceptance rate: {accepted/N_SAMPLES:.3f}")

# ── Post-processing ──
print("\nPost-processing...")
samples = chain[N_BURNIN:]
n_eff = len(samples)

means = samples.mean(axis=0)
stds = samples.std(axis=0)

print(f"\nPosteriors ({n_eff} post-burnin samples):")
for i, name in enumerate(param_names):
    print(f"  {name:>15s} = {means[i]:.4f} ± {stds[i]:.4f}")

# Covariance matrix
cov_post = np.cov(samples.T)

# α = dH₀/dα_B from the posterior covariance
# Cov(H0, alpha_B) / Var(alpha_B)
idx_H0 = 0
idx_aB = 3
alpha = cov_post[idx_H0, idx_aB] / cov_post[idx_aB, idx_aB]
alpha_err = np.sqrt(cov_post[idx_H0, idx_H0] / cov_post[idx_aB, idx_aB]) / np.sqrt(n_eff)

# Correlation
r_H0_aB = cov_post[idx_H0, idx_aB] / np.sqrt(cov_post[idx_H0, idx_H0] * cov_post[idx_aB, idx_aB])

print(f"\n{'='*60}")
print(f"α = dH₀/dα_B = {alpha:.3f} ± {alpha_err:.3f}")
print(f"Correlation r(H₀, α_B) = {r_H0_aB:.4f}")
print(f"{'='*60}")

# Full correlation matrix
print("\nCorrelation matrix:")
corr = np.corrcoef(samples.T)
header = "              " + "".join(f"{n:>12s}" for n in param_names)
print(header)
for i, name in enumerate(param_names):
    row = f"  {name:>12s}" + "".join(f"{corr[i,j]:12.4f}" for j in range(n_par))
    print(row)

# ── Save outputs ──
outdir = os.path.expanduser("~/Desktop/dgf_master_findings/alpha_measurement/")
os.makedirs(outdir, exist_ok=True)

# Posteriors
with open(os.path.join(outdir, "posteriors.txt"), 'w') as f:
    f.write("α_B GPU Chain — Transfer Coefficient Measurement\n")
    f.write("=" * 60 + "\n")
    f.write(f"Emulator: PCA+NN on hi_class C_l with varying α_B\n")
    f.write(f"Likelihood: Planck plik_lite TTTEEE (simplified)\n")
    f.write(f"Samples: {n_eff} post-burnin\n\n")
    for i, name in enumerate(param_names):
        f.write(f"  {name:>15s} = {means[i]:.4f} ± {stds[i]:.4f}\n")
    f.write(f"\n  α = dH₀/dα_B = {alpha:.3f} ± {alpha_err:.3f}\n")
    f.write(f"  r(H₀, α_B) = {r_H0_aB:.4f}\n")
    f.write(f"\n  Runtime: {elapsed:.0f}s\n")

# Chain
np.savetxt(os.path.join(outdir, "chain.txt"), chain,
           header=" ".join(param_names))

# Config
shutil.copy2("/home/joe-research/chains/dgf_J6_flash_field.yaml",
             os.path.join(outdir, "config.yaml"))

# Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# H0 posterior
ax = axes[0, 0]
ax.hist(samples[:, 0], bins=60, density=True, color='steelblue', alpha=0.7)
ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
ax.set_title(f'$H_0$ = {means[0]:.2f} ± {stds[0]:.2f}')

# alpha_B posterior
ax = axes[0, 1]
ax.hist(samples[:, 3], bins=60, density=True, color='darkorange', alpha=0.7)
ax.axvline(1.0, color='red', ls='--', lw=2, label='DGF fiducial')
ax.set_xlabel(r'$\alpha_B$ scale')
ax.set_title(f'$\\alpha_B$ = {means[3]:.3f} ± {stds[3]:.3f}')
ax.legend()

# H0 vs alpha_B scatter
ax = axes[0, 2]
ax.scatter(samples[::3, 3], samples[::3, 0], s=1, alpha=0.3, c='steelblue')
ax.set_xlabel(r'$\alpha_B$ scale')
ax.set_ylabel(r'$H_0$ [km/s/Mpc]')
ax.set_title(f'$\\alpha$ = dH₀/dα_B = {alpha:.3f}, r = {r_H0_aB:.3f}')

# omega_cdm
ax = axes[1, 0]
ax.hist(samples[:, 1], bins=60, density=True, color='green', alpha=0.7)
ax.set_xlabel(r'$\omega_{\rm cdm}$')
ax.set_title(f'$\\omega_{{cdm}}$ = {means[1]:.4f} ± {stds[1]:.4f}')

# logA
ax = axes[1, 1]
ax.hist(samples[:, 2], bins=60, density=True, color='purple', alpha=0.7)
ax.set_xlabel(r'$\log A$')
ax.set_title(f'logA = {means[2]:.3f} ± {stds[2]:.3f}')

# A_planck
ax = axes[1, 2]
ax.hist(samples[:, 4], bins=60, density=True, color='crimson', alpha=0.7)
ax.set_xlabel(r'$A_{\rm planck}$')
ax.set_title(f'$A_{{planck}}$ = {means[4]:.4f} ± {stds[4]:.4f}')

plt.suptitle(f'α_B GPU Chain — α = dH₀/dα_B = {alpha:.3f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(outdir, "alpha_posteriors.png"), dpi=150, bbox_inches='tight')
print(f"\nSaved: {outdir}alpha_posteriors.png")
print(f"Saved: {outdir}posteriors.txt")
print(f"Saved: {outdir}chain.txt")
print("\nDONE.")
