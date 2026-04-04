#!/usr/bin/env python3
"""
Task 1 part 2: Retrain emulator on merged data and run GPU chain.
Run this AFTER generate_alphaB_fixed.py completes.
"""
import numpy as np
import json, os, time, shutil, pickle

DATADIR = "/home/joe-research/dgf_training/cl_emulator_alphaB_v2/merged"
MODELDIR = "/home/joe-research/dgf_training/cl_emulator_alphaB_v2/model"
os.makedirs(MODELDIR, exist_ok=True)

# ── Load merged training data ──
print("Loading merged training data...")
params = np.load(os.path.join(DATADIR, "params.npy"))
cl_tt = np.load(os.path.join(DATADIR, "cl_tt.npy"))
ell = np.load(os.path.join(DATADIR, "ell.npy"))

n_samples, n_ell = cl_tt.shape
n_params = params.shape[1]
print(f"  {n_samples} samples, {n_ell} multipoles, {n_params} parameters")

# ── Normalise ──
param_mean = params.mean(axis=0)
param_std = params.std(axis=0)
params_norm = (params - param_mean) / param_std

# Clip tiny/negative values before log
cl_tt = np.clip(cl_tt, 1e-10, None)
log_cl_tt = np.log(cl_tt)

# ── PCA ──
N_PCA = 30
mean_tt = log_cl_tt.mean(axis=0)
X_c = log_cl_tt - mean_tt
U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
components = Vt[:N_PCA]
coeffs = X_c @ components.T
X_recon = coeffs @ components + mean_tt
rel_err = np.abs(log_cl_tt - X_recon) / (np.abs(log_cl_tt) + 1e-30)
print(f"  PCA {N_PCA}: mean rel err = {rel_err.mean():.6f}")

# ── Train sklearn NN ──
from sklearn.neural_network import MLPRegressor

n_val = max(1, n_samples // 10)
idx = np.random.RandomState(42).permutation(n_samples)
train_idx, val_idx = idx[n_val:], idx[:n_val]

print("  Training NN...")
model = MLPRegressor(hidden_layer_sizes=(256, 256, 256),
                     activation='relu', max_iter=5000,
                     early_stopping=True, validation_fraction=0.1,
                     random_state=42, verbose=False)
model.fit(params_norm[train_idx], coeffs[train_idx])

# Validate
pred_coeffs = model.predict(params_norm[val_idx])
pred_log = pred_coeffs @ components + mean_tt
pred_cl = np.exp(pred_log)
true_cl = cl_tt[val_idx]
rel_err_nn = np.abs(pred_cl - true_cl) / (np.abs(true_cl) + 1e-30)
print(f"  NN validation: mean rel err = {rel_err_nn.mean():.4%}, max = {rel_err_nn.max():.4%}")

# Save
with open(os.path.join(MODELDIR, "tt_net.pkl"), 'wb') as f:
    pickle.dump(model, f)
np.save(os.path.join(MODELDIR, "tt_pca_mean.npy"), mean_tt)
np.save(os.path.join(MODELDIR, "tt_pca_components.npy"), components)
np.save(os.path.join(MODELDIR, "ell.npy"), ell)
with open(os.path.join(MODELDIR, "normalisation.json"), 'w') as f:
    json.dump({"param_mean": param_mean.tolist(), "param_std": param_std.tolist(),
               "n_pca": N_PCA, "n_params": n_params,
               "param_names": ["H0", "omega_cdm", "logA", "alpha_B_scale"]}, f, indent=2)
print(f"  Model saved to {MODELDIR}/")

# ── GPU Chain ──
print("\n" + "=" * 60)
print("Running α_B GPU chain (v2, merged training data)")
print("=" * 60)

# Load plik_lite TT
PLIK = "/home/joe-research/dgf_data/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik/clik/lkl_0/_external"
plik_all = np.loadtxt(os.path.join(PLIK, "cl_cmb_plik_v22.dat"))
N_TT = 215
plik_ell = plik_all[:N_TT, 0].astype(int)
plik_cl = plik_all[:N_TT, 1]
plik_var = plik_all[:N_TT, 2]**2

ell_factor = ell * (ell + 1) / (2 * np.pi)

def predict_cl_tt(p):
    p_norm = (p - param_mean) / param_std
    c = model.predict(p_norm.reshape(1, -1))[0]
    dl = np.exp(c @ components + mean_tt)
    return dl / ell_factor

def loglike(x):
    H0, omega_cdm, logA, alpha_B_scale, A_planck = x
    cl = predict_cl_tt(np.array([H0, omega_cdm, logA, alpha_B_scale])) * A_planck**2
    theory = np.interp(plik_ell, ell, cl)
    chi2 = np.sum((theory - plik_cl)**2 / plik_var)
    if np.isnan(chi2) or np.isinf(chi2):
        return -1e30
    return -0.5 * chi2

def logprior(x):
    lo = np.array([60.0, 0.09, 2.7, 0.1, 0.9])
    hi = np.array([80.0, 0.16, 3.4, 2.0, 1.1])
    if np.any(x < lo) or np.any(x > hi):
        return -np.inf
    return -0.5 * ((x[4] - 1.0) / 0.025)**2

param_names = ["H0", "omega_cdm", "logA", "alpha_B_scale", "A_planck"]
x0 = np.array([65.75, 0.1192, 3.086, 1.0, 0.979])
proposal = np.array([0.15, 0.0005, 0.015, 0.07, 0.007])

N_SAMPLES = 100000
N_BURNIN = 10000

chain = np.zeros((N_SAMPLES, 5))
x = x0.copy()
lp = logprior(x) + loglike(x)
accepted = 0
t0 = time.time()

for i in range(N_SAMPLES):
    x_prop = x + proposal * np.random.randn(5)
    lp_prop = logprior(x_prop)
    if lp_prop > -np.inf:
        lp_prop += loglike(x_prop)
    if np.log(np.random.rand()) < lp_prop - lp:
        x = x_prop
        lp = lp_prop
        accepted += 1
    chain[i] = x
    if (i + 1) % 10000 == 0:
        print(f"  {i+1}/{N_SAMPLES}: accept={accepted/(i+1):.3f}, logpost={lp:.1f}", flush=True)

elapsed = time.time() - t0
print(f"\nMCMC: {N_SAMPLES} steps in {elapsed:.0f}s, accept={accepted/N_SAMPLES:.3f}")

samples = chain[N_BURNIN:]
means = samples.mean(axis=0)
stds = samples.std(axis=0)

print(f"\nPosteriors:")
for i, n in enumerate(param_names):
    print(f"  {n:>15s} = {means[i]:.4f} ± {stds[i]:.4f}")

cov_post = np.cov(samples.T)
alpha = cov_post[0, 3] / cov_post[3, 3]
r = cov_post[0, 3] / np.sqrt(cov_post[0, 0] * cov_post[3, 3])

print(f"\n  α = dH₀/dα_B = {alpha:.3f}")
print(f"  r(H₀, α_B)   = {r:.4f}")

# Save to analysis sessions
outdir = os.path.expanduser("~/Desktop/dgf_analysis_sessions/alpha_measurement_v2/")
os.makedirs(outdir, exist_ok=True)

with open(os.path.join(outdir, "posteriors.txt"), 'w') as f:
    f.write("α_B GPU Chain v2 (merged 2500 training points)\n")
    f.write("=" * 60 + "\n")
    for i, n in enumerate(param_names):
        f.write(f"  {n:>15s} = {means[i]:.4f} ± {stds[i]:.4f}\n")
    f.write(f"\n  α = dH₀/dα_B = {alpha:.3f}\n")
    f.write(f"  r(H₀, α_B)   = {r:.4f}\n")
    f.write(f"  Runtime: {elapsed:.0f}s\n")
    f.write(f"  Training: {n_samples} points, NN err {rel_err_nn.mean():.4%}\n")

np.savetxt(os.path.join(outdir, "chain.txt"), chain, header=" ".join(param_names))

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.hist(samples[:, 0], bins=80, density=True, color='steelblue', alpha=0.7)
ax.set_xlabel(r'$H_0$'); ax.set_title(f'H₀ = {means[0]:.2f} ± {stds[0]:.2f}')

ax = axes[1]
ax.hist(samples[:, 3], bins=80, density=True, color='darkorange', alpha=0.7)
ax.axvline(1.0, color='red', ls='--', lw=2, label='DGF fiducial')
ax.set_xlabel(r'$\alpha_B$ scale'); ax.set_title(f'α_B = {means[3]:.3f} ± {stds[3]:.3f}')
ax.legend()

ax = axes[2]
ax.scatter(samples[::5, 3], samples[::5, 0], s=1, alpha=0.3, c='steelblue')
ax.set_xlabel(r'$\alpha_B$ scale'); ax.set_ylabel(r'$H_0$')
ax.set_title(f'α = {alpha:.3f}, r = {r:.3f}')

plt.suptitle(f'α_B Chain v2 — α = dH₀/dα_B = {alpha:.3f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(outdir, "alpha_posteriors_v2.png"), dpi=150, bbox_inches='tight')
print(f"\nSaved to {outdir}")
