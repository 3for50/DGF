#!/usr/bin/env python3
"""
J6 — Flash Field Equilibrium Test (analytic factorisation)

Since H0_matter has no likelihood term, the joint posterior factorises:
  P(H0, params, H0_matter | data) = P(H0, params | data) × N(H0_matter | 72.5, 1.5)

We use the existing converged Planck+DGF chain for the light channel,
and draw H0_matter independently from its Gaussian prior.
"""
import numpy as np
import os, shutil

# ── Load existing Planck chain ──
chain_file = "/home/joe-research/chains/dgf_planck_h0free.1.txt"
data = np.loadtxt(chain_file)
n_samples = len(data)

# Column mapping from header
weights = data[:, 0]
H0_light = data[:, 2]       # H0 (light channel, CMB-constrained)
omega_cdm = data[:, 3]
logA = data[:, 4]
A_planck = data[:, 5]
sigma8 = data[:, 7]
Omega_m = data[:, 8]

# Burn-in: drop first 30%
burnin = int(0.3 * n_samples)
weights = weights[burnin:]
H0_light = H0_light[burnin:]
omega_cdm = omega_cdm[burnin:]
logA = logA[burnin:]
A_planck = A_planck[burnin:]
sigma8 = sigma8[burnin:]
Omega_m = Omega_m[burnin:]
n_eff = len(weights)

print(f"Loaded {n_samples} samples, using {n_eff} after burn-in")

# ── Draw H0_matter from its Gaussian prior ──
np.random.seed(61)
H0_matter = np.random.normal(72.5, 1.5, size=n_eff)

# ── Compute derived Flash Field quantities ──
F_arithmetic = (H0_light + H0_matter) / 2.0
w_light = 1.0 / 0.19**2   # ≈ 27.7 (Planck H0 precision)
w_matter = 1.0 / 1.5**2    # ≈ 0.444
F_weighted = (H0_light * w_light + H0_matter * w_matter) / (w_light + w_matter)

# ─�� Weighted statistics ──
def wstats(x, w):
    m = np.average(x, weights=w)
    v = np.average((x - m)**2, weights=w)
    s = np.sqrt(v)
    p16, p50, p84 = np.percentile(x, [16, 50, 84])
    return m, s, p50, p16, p84

F_TARGET = 69.47
DELTA = 2.075

print("\n" + "=" * 65)
print("J6 — FLASH FIELD EQUILIBRIUM TEST")
print("=" * 65)
print(f"Light channel:  Planck TTTEEE + low-l (DGF hi_class, H0 free)")
print(f"Matter channel: N(72.5, 1.5) combined corpus prior")
print(f"Effective samples: {n_eff}")
print("=" * 65)

lines = []
def report(name, x):
    m, s, med, p16, p84 = wstats(x, weights)
    line = f"  {name:>20s} = {m:.4f} +/- {s:.4f}  (68% CI: [{p16:.4f}, {p84:.4f}])"
    print(line)
    lines.append(line)
    return m, s

h0_m, h0_s = report("H0 (light)", H0_light)
hm_m, hm_s = report("H0_matter", H0_matter)
ocdm_m, ocdm_s = report("omega_cdm", omega_cdm)
logA_m, logA_s = report("logA", logA)
ap_m, ap_s = report("A_planck", A_planck)
s8_m, s8_s = report("sigma8", sigma8)
om_m, om_s = report("Omega_m", Omega_m)
fa_m, fa_s = report("F_arithmetic", F_arithmetic)
fw_m, fw_s = report("F_weighted", F_weighted)

print()

# Key diagnostics
nsig_arith = abs(fa_m - F_TARGET) / fa_s
nsig_weight = abs(fw_m - F_TARGET) / fw_s
gap = fa_m - fw_m

# Correlation (should be ~0 since independent)
cov_matrix = np.cov(H0_light, H0_matter, aweights=weights)
r_corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])

b_arith = (fa_m - F_TARGET) / DELTA
b_weight = (fw_m - F_TARGET) / DELTA

print(f"  F_arithmetic from 69.47:           {nsig_arith:.2f}σ")
print(f"  F_weighted from 69.47:             {nsig_weight:.1f}σ")
print(f"  Gap (F_arith - F_weighted):        {gap:+.2f} km/s/Mpc")
print(f"  Correlation r(H0, H0_matter):      {r_corr:.4f}")
print(f"  b_arithmetic:                      {b_arith:+.4f}")
print(f"  b_weighted:                        {b_weight:+.4f}")
print()

# Precision ratio
ratio = w_light / w_matter
print(f"  Precision ratio (light/matter):    {ratio:.1f}:1")
print(f"  w_light = 1/{h0_s:.2f}^2 = {1/h0_s**2:.1f}")
print(f"  w_matter = 1/1.50^2 = {w_matter:.3f}")
print()

# ── Interpretation ──
print("INTERPRETATION:")
if nsig_arith < 1.0:
    print(f"  F_arithmetic is consistent with 69.47 ({nsig_arith:.2f}σ)")
    print(f"  → The geometric midpoint of the two channels lands ON the DGF prediction")
else:
    print(f"  F_arithmetic deviates from 69.47 by {nsig_arith:.2f}σ")

if nsig_weight > 3.0:
    print(f"  F_weighted is {nsig_weight:.0f}σ from 69.47")
    print(f"  → Precision asymmetry ({ratio:.0f}:1) pulls the statistical mean to the light channel")

print(f"  The {gap:.1f} km/s/Mpc gap between F_arith and F_weighted")
print(f"  IS the Flash Field effect — distance equilibrium ≠ statistical equilibrium")
print()

# ── Plots ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

outdir = os.path.expanduser("~/Desktop/dgf_master_findings/J6_flash_field/")
os.makedirs(outdir, exist_ok=True)

# Main figure: 2x2 panel
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: H0 light vs H0 matter scatter
ax = axes[0, 0]
idx = np.random.choice(n_eff, min(5000, n_eff), replace=False)
ax.scatter(H0_light[idx], H0_matter[idx], s=1, alpha=0.3, c='steelblue')
ax.axhline(72.5, color='red', ls='--', alpha=0.5, lw=1)
ax.axvline(h0_m, color='blue', ls='--', alpha=0.5, lw=1)
ax.plot([55, 85], [55, 85], 'k:', alpha=0.3)  # equality line
ax.set_xlabel(r'$H_0^{\rm light}$ [km/s/Mpc]', fontsize=11)
ax.set_ylabel(r'$H_0^{\rm matter}$ [km/s/Mpc]', fontsize=11)
ax.set_title(f'r = {r_corr:.4f} (independent by construction)', fontsize=10)

# Panel 2: F_arithmetic
ax = axes[0, 1]
ax.hist(F_arithmetic, bins=80, weights=weights, density=True, color='darkorange', alpha=0.7)
ax.axvline(F_TARGET, color='red', lw=2.5, ls='--', label=f'DGF: {F_TARGET}')
ax.axvline(fa_m, color='black', lw=2, label=f'Mean: {fa_m:.2f}±{fa_s:.2f}')
ax.set_xlabel(r'$\mathcal{F}_{\rm arithmetic}$ [km/s/Mpc]', fontsize=11)
ax.set_title(f'Arithmetic midpoint ({nsig_arith:.2f}σ from 69.47)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

# Panel 3: F_weighted
ax = axes[1, 0]
ax.hist(F_weighted, bins=80, weights=weights, density=True, color='purple', alpha=0.7)
ax.axvline(F_TARGET, color='red', lw=2.5, ls='--', label=f'DGF: {F_TARGET}')
ax.axvline(fw_m, color='black', lw=2, label=f'Mean: {fw_m:.2f}±{fw_s:.2f}')
ax.set_xlabel(r'$\mathcal{F}_{\rm weighted}$ [km/s/Mpc]', fontsize=11)
ax.set_title(f'Precision-weighted ({nsig_weight:.1f}σ from 69.47)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

# Panel 4: Both overlaid + interpretation
ax = axes[1, 1]
bins_a = np.linspace(min(F_arithmetic.min(), F_weighted.min()) - 1,
                     max(F_arithmetic.max(), F_weighted.max()) + 1, 100)
ax.hist(F_arithmetic, bins=bins_a, weights=weights, density=True,
        color='darkorange', alpha=0.5, label=f'Arithmetic: {fa_m:.2f}±{fa_s:.2f}')
ax.hist(F_weighted, bins=bins_a, weights=weights, density=True,
        color='purple', alpha=0.5, label=f'Weighted: {fw_m:.2f}±{fw_s:.2f}')
ax.axvline(F_TARGET, color='red', lw=2.5, ls='--', label=f'DGF prediction: {F_TARGET}')
ax.annotate(f'Gap = {gap:.1f} km/s/Mpc', xy=(0.5, 0.92), xycoords='axes fraction',
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel(r'$\mathcal{F}$ [km/s/Mpc]', fontsize=11)
ax.set_title('Flash Field: distance vs statistical equilibrium', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

plt.suptitle('J6 — Flash Field Equilibrium Test', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "J6_flash_field.png"), dpi=150, bbox_inches='tight')
print(f"Saved: {outdir}J6_flash_field.png")

# H0 channel posteriors
fig, ax = plt.subplots(figsize=(10, 5))
bins_h = np.linspace(58, 80, 100)
ax.hist(H0_light, bins=bins_h, weights=weights, density=True,
        color='steelblue', alpha=0.7, label=f'Light channel: {h0_m:.2f}±{h0_s:.2f}')
ax.hist(H0_matter, bins=bins_h, density=True,
        color='crimson', alpha=0.7, label=f'Matter channel: {hm_m:.2f}±{hm_s:.2f}')
ax.axvline(F_TARGET, color='green', lw=2.5, ls='--', label=f'Flash Field: {F_TARGET}')
ax.axvline(fa_m, color='darkorange', lw=2, label=f'Arithmetic midpoint: {fa_m:.2f}')
ax.axvline(fw_m, color='purple', lw=2, label=f'Weighted midpoint: {fw_m:.2f}')
ax.set_xlabel(r'$H_0$ [km/s/Mpc]', fontsize=12)
ax.set_ylabel('Posterior density', fontsize=12)
ax.set_title('J6: Light vs Matter Channel — Flash Field Equilibrium', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(outdir, "J6_H0_channels.png"), dpi=150, bbox_inches='tight')
print(f"Saved: {outdir}J6_H0_channels.png")

# ── Save posteriors ──
with open(os.path.join(outdir, "posteriors.txt"), 'w') as f:
    f.write("J6 — Flash Field Equilibrium Test\n")
    f.write("=" * 65 + "\n")
    f.write("Light channel: Planck TTTEEE + low-l TT/EE (DGF hi_class, H0 free)\n")
    f.write("Matter channel: N(72.5, 1.5) combined corpus prior\n")
    f.write(f"Effective samples: {n_eff}\n\n")
    for line in lines:
        f.write(line + "\n")
    f.write(f"\n  F_arithmetic from 69.47: {nsig_arith:.2f}σ\n")
    f.write(f"  F_weighted from 69.47:   {nsig_weight:.1f}σ\n")
    f.write(f"  Gap (arith - weighted):  {gap:+.2f} km/s/Mpc\n")
    f.write(f"  r(H0, H0_matter):        {r_corr:.4f}\n")
    f.write(f"  b_arithmetic:            {b_arith:+.4f}\n")
    f.write(f"  b_weighted:              {b_weight:+.4f}\n")
    f.write(f"  Precision ratio:         {ratio:.1f}:1\n")
print(f"Saved: {outdir}posteriors.txt")

# Save config
shutil.copy2("/home/joe-research/chains/dgf_J6_flash_field.yaml", os.path.join(outdir, "config.yaml"))

# Save chain samples
chain_out = np.column_stack([weights, H0_light, H0_matter, omega_cdm, logA,
                              F_arithmetic, F_weighted])
np.savetxt(os.path.join(outdir, "chain.txt"), chain_out,
           header="weight H0_light H0_matter omega_cdm logA F_arithmetic F_weighted")
print(f"Saved: {outdir}chain.txt")
print("\nJ6 COMPLETE.")
