#!/usr/bin/env python3
"""
Task 2: Formal chi2 of DGF vs ΛCDM at l=2-30 using Planck PR3 Commander data.
Uses asymmetric errors (lower/upper) — take the appropriate one based on residual sign.
"""
import numpy as np

# Planck PR3 Commander observed D_l^TT (l=2-30)
obs = {
    2: (225.9, 132.4, 533.1), 3: (936.9, 450.5, 1212.3), 4: (692.2, 294.1, 666.5),
    5: (1501.7, 574.4, 1155.8), 6: (557.6, 201.2, 375.8), 7: (1152.6, 381.6, 670.8),
    8: (615.8, 192.0, 323.4), 9: (697.8, 214.6, 349.6), 10: (803.7, 232.8, 367.9),
    11: (869.6, 240.3, 371.0), 12: (764.1, 205.0, 310.6), 13: (599.0, 158.6, 235.6),
    14: (808.1, 199.4, 290.8), 15: (1022.1, 243.5, 349.7), 16: (621.3, 151.0, 215.8),
    17: (1035.4, 236.2, 332.1), 18: (674.6, 156.3, 216.9), 19: (951.6, 211.6, 291.0),
    20: (659.9, 144.8, 197.6), 21: (601.9, 136.1, 185.3), 22: (605.3, 131.7, 177.0),
    23: (748.5, 155.9, 208.7), 24: (850.5, 175.6, 233.2), 25: (814.3, 169.4, 224.3),
    26: (823.8, 164.4, 215.4), 27: (1004.0, 190.3, 247.9), 28: (1149.9, 215.1, 279.1),
    29: (979.6, 183.1, 236.7), 30: (1102.8, 274.7, 274.7),
}

# DGF and LCDM predictions from the hi_class runs (Task B results)
dgf = {2: 879.4, 3: 958.4, 4: 978.0, 5: 974.3, 6: 963.0, 7: 950.1, 8: 937.8,
       9: 927.5, 10: 919.5, 11: 913.3, 12: 909.1, 13: 907.0, 14: 906.5, 15: 907.7,
       16: 910.7, 17: 914.6, 18: 920.0, 19: 926.8, 20: 934.5, 21: 943.0, 22: 952.5,
       23: 962.8, 24: 973.7, 25: 985.3, 26: 997.4, 27: 1010.3, 28: 1023.7, 29: 1037.4,
       30: 1051.4}

lcdm = {2: 969.3, 3: 921.8, 4: 875.5, 5: 840.7, 6: 816.9, 7: 801.8, 8: 793.5,
        9: 789.6, 10: 789.7, 11: 792.5, 12: 797.2, 13: 803.9, 14: 811.9, 15: 820.8,
        16: 830.8, 17: 841.6, 18: 853.0, 19: 865.2, 20: 877.6, 21: 890.6, 22: 904.1,
        23: 918.0, 24: 932.1, 25: 946.5, 26: 961.3, 27: 976.5, 28: 991.8, 29: 1007.3,
        30: 1023.2}

def chi2_asym(obs_val, pred, sigma_lo, sigma_hi):
    """Chi2 with asymmetric errors: use lower error if pred > obs, upper if pred < obs."""
    residual = pred - obs_val
    sigma = sigma_hi if residual < 0 else sigma_lo
    return (residual / sigma)**2

print("=" * 95)
print("PLANCK PR3 LOW-l TT: FORMAL χ² COMPARISON (asymmetric errors)")
print("=" * 95)
print(f"{'l':>3s}  {'Obs':>8s}  {'DGF':>8s}  {'ΛCDM':>8s}  {'χ²_DGF':>8s}  {'χ²_ΛCDM':>8s}  {'Δχ²':>8s}  {'Favours':>8s}")
print("-" * 95)

total_chi2_dgf = 0
total_chi2_lcdm = 0
dgf_wins = 0
lcdm_wins = 0

for ell in range(2, 31):
    d, sig_lo, sig_hi = obs[ell]
    c2_dgf = chi2_asym(d, dgf[ell], sig_lo, sig_hi)
    c2_lcdm = chi2_asym(d, lcdm[ell], sig_lo, sig_hi)
    delta = c2_lcdm - c2_dgf  # positive = DGF better
    total_chi2_dgf += c2_dgf
    total_chi2_lcdm += c2_lcdm

    if delta > 0:
        fav = "DGF"
        dgf_wins += 1
    else:
        fav = "ΛCDM"
        lcdm_wins += 1

    print(f"{ell:3d}  {d:8.1f}  {dgf[ell]:8.1f}  {lcdm[ell]:8.1f}  {c2_dgf:8.3f}  {c2_lcdm:8.3f}  {delta:+8.3f}  {fav:>8s}")

print("-" * 95)
total_delta = total_chi2_lcdm - total_chi2_dgf
print(f"{'TOT':>3s}  {'':>8s}  {'':>8s}  {'':>8s}  {total_chi2_dgf:8.3f}  {total_chi2_lcdm:8.3f}  {total_delta:+8.3f}  {'DGF' if total_delta > 0 else 'ΛCDM':>8s}")
print()
print(f"  Total χ²_DGF  = {total_chi2_dgf:.3f}  (χ²/dof = {total_chi2_dgf/29:.3f})")
print(f"  Total χ²_ΛCDM = {total_chi2_lcdm:.3f}  (χ²/dof = {total_chi2_lcdm/29:.3f})")
print(f"  Δχ² (ΛCDM − DGF) = {total_delta:+.3f}")
print(f"  {'DGF' if total_delta > 0 else 'ΛCDM'} fits the Planck sky better at l=2-30")
print(f"  Multipoles favouring DGF: {dgf_wins}/29, ΛCDM: {lcdm_wins}/29")
