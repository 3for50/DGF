#!/usr/bin/env python3
"""
Task 3: J6 Flash Field with TDCOSMO prior N(73.3, 1.7).
Analytic factorisation — same as before but different matter prior.
"""
import numpy as np
import os

chain_file = "/home/joe-research/chains/dgf_planck_h0free.1.txt"
data = np.loadtxt(chain_file)
n = len(data)
burnin = int(0.3 * n)
weights = data[burnin:, 0]
H0_light = data[burnin:, 2]
n_eff = len(weights)

# TDCOSMO prior: N(73.3, 1.7)
np.random.seed(73)
H0_matter = np.random.normal(73.3, 1.7, size=n_eff)

F_arith = (H0_light + H0_matter) / 2.0
w_light = 1.0 / np.std(H0_light)**2  # from chain
w_matter = 1.0 / 1.7**2
F_weight = (H0_light * w_light + H0_matter * w_matter) / (w_light + w_matter)

F_TARGET = 69.47
DELTA = 2.075

def wstats(x, w):
    m = np.average(x, weights=w)
    s = np.sqrt(np.average((x - m)**2, weights=w))
    return m, s

h0_m, h0_s = wstats(H0_light, weights)
hm_m, hm_s = wstats(H0_matter, weights)
fa_m, fa_s = wstats(F_arith, weights)
fw_m, fw_s = wstats(F_weight, weights)

nsig_arith = abs(fa_m - F_TARGET) / fa_s
gap = fa_m - fw_m
b_arith = (fa_m - F_TARGET) / DELTA

cov = np.cov(H0_light, H0_matter, aweights=weights)
r_corr = cov[0,1] / np.sqrt(cov[0,0]*cov[1,1])

print("=" * 60)
print("J6 — Flash Field (TDCOSMO prior)")
print("=" * 60)
print(f"  Light: Planck TTTEEE + low-l (DGF hi_class)")
print(f"  Matter: TDCOSMO N(73.3, 1.7)")
print(f"  Samples: {n_eff}")
print()
print(f"  H0 (light)     = {h0_m:.4f} ± {h0_s:.4f}")
print(f"  H0_matter       = {hm_m:.4f} ± {hm_s:.4f}")
print(f"  F_arithmetic    = {fa_m:.4f} ± {fa_s:.4f}")
print(f"  F_weighted      = {fw_m:.4f} ± {fw_s:.4f}")
print()
print(f"  F_arith from 69.47: {nsig_arith:.2f}σ")
print(f"  Gap (arith - weighted): {gap:+.2f} km/s/Mpc")
print(f"  r(H0, H0_matter): {r_corr:.4f}")
print(f"  b_arithmetic: {b_arith:+.4f}")
print()

# Compare with original N(72.5, 1.5)
np.random.seed(61)
H0_matter_orig = np.random.normal(72.5, 1.5, size=n_eff)
fa_orig_m, fa_orig_s = wstats((H0_light + H0_matter_orig) / 2.0, weights)
print(f"  COMPARISON:")
print(f"  Original N(72.5,1.5): F_arith = {fa_orig_m:.4f} ± {fa_orig_s:.4f} ({abs(fa_orig_m-F_TARGET)/fa_orig_s:.2f}σ from 69.47)")
print(f"  TDCOSMO N(73.3,1.7): F_arith = {fa_m:.4f} ± {fa_s:.4f} ({nsig_arith:.2f}σ from 69.47)")
shift = fa_m - fa_orig_m
print(f"  Shift: {shift:+.3f} km/s/Mpc ({'closer' if abs(fa_m-F_TARGET) < abs(fa_orig_m-F_TARGET) else 'further'} to 69.47)")

# Save
outdir = os.path.expanduser("~/Desktop/dgf_master_findings/J6_flash_field_tdcosmo/")
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "posteriors.txt"), 'w') as f:
    f.write("J6 — Flash Field (TDCOSMO prior N(73.3, 1.7))\n")
    f.write("=" * 60 + "\n")
    f.write(f"  H0 (light)     = {h0_m:.4f} ± {h0_s:.4f}\n")
    f.write(f"  H0_matter       = {hm_m:.4f} ± {hm_s:.4f}\n")
    f.write(f"  F_arithmetic    = {fa_m:.4f} ± {fa_s:.4f}\n")
    f.write(f"  F_weighted      = {fw_m:.4f} ± {fw_s:.4f}\n")
    f.write(f"  F_arith from 69.47: {nsig_arith:.2f}σ\n")
    f.write(f"  Gap: {gap:+.2f} km/s/Mpc\n")
    f.write(f"  b_arithmetic: {b_arith:+.4f}\n")
print(f"Saved: {outdir}posteriors.txt")
