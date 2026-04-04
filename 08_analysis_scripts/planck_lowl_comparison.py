#!/usr/bin/env python3
"""
Task B: Planck PR3 low-l TT comparison — DGF vs ΛCDM vs observed.
"""
import numpy as np
import subprocess, sys, os, tempfile

# ── Planck PR3 observed D_l^TT (Commander, l=2–30) ──
planck_obs = {
    2: 225.9, 3: 936.9, 4: 692.2, 5: 1501.7, 6: 557.6,
    7: 1152.6, 8: 615.8, 9: 697.8, 10: 803.7, 11: 869.6,
    12: 764.1, 13: 599.0, 14: 808.1, 15: 1022.1, 16: 621.3,
    17: 1035.4, 18: 674.6, 19: 951.6, 20: 659.9, 21: 601.9,
    22: 605.3, 23: 748.5, 24: 850.5, 25: 814.3, 26: 823.8,
    27: 1004.0, 28: 1149.9, 29: 979.6, 30: 1102.8,
}

# ── Parameters ──
H0 = 65.73
omega_cdm = 0.1190
logA = 3.017
A_s = np.exp(logA) * 1e-10
omega_b = 0.02238280
n_s = 0.9660499
tau_reio = 0.054

# hi_class path
HICLASS = "/home/joe-research/hi_class"

def run_class(ini_content, label, root_prefix):
    """Run hi_class with given .ini content, return dict of l -> D_l^TT."""
    ini_path = os.path.join(HICLASS, f"{root_prefix}.ini")
    with open(ini_path, 'w') as f:
        f.write(ini_content)

    try:
        result = subprocess.run(
            [os.path.join(HICLASS, "class"), ini_path],
            capture_output=True, text=True, timeout=120, cwd=HICLASS
        )
        if result.returncode != 0:
            print(f"[{label}] CLASS failed: {result.stderr[:500]}")
            return None

        # hi_class appends 00_ to root
        cl_file = os.path.join(HICLASS, f"{root_prefix}00_cl_lensed.dat")
        if not os.path.exists(cl_file):
            cl_file = os.path.join(HICLASS, f"{root_prefix}00_cl.dat")
        if not os.path.exists(cl_file):
            print(f"[{label}] No output cl file found")
            for f2 in sorted(os.listdir(HICLASS)):
                if f2.startswith(root_prefix):
                    print(f"  Found: {f2}")
            return None

        data = np.loadtxt(cl_file, comments='#')
        # With format=camb: columns are l, TT, EE, BB, TE, ... in D_l [uK^2]
        dl_tt = {}
        for row in data:
            ell = int(row[0])
            if 2 <= ell <= 30:
                dl_tt[ell] = row[1]

        # Cleanup
        os.unlink(ini_path)
        for f2 in os.listdir(HICLASS):
            if f2.startswith(root_prefix):
                os.unlink(os.path.join(HICLASS, f2))

        return dl_tt
    except Exception as e:
        print(f"[{label}] Error: {e}")
        return None

# ── ΛCDM run ──
lcdm_ini = f"""output = tCl pCl lCl
lensing = yes
l_max_scalars = 50
format = camb
A_s = {A_s:.6e}
n_s = {n_s}
H0 = {H0}
omega_b = {omega_b}
omega_cdm = {omega_cdm}
tau_reio = {tau_reio}
N_ur = 2.0328
N_ncdm = 1
m_ncdm = 0.06
T_ncdm = 0.71611
root = lowl_lcdm_
"""

# ── DGF run ──
dgf_ini = f"""output = tCl pCl lCl
lensing = yes
l_max_scalars = 50
format = camb
A_s = {A_s:.6e}
n_s = {n_s}
H0 = {H0}
omega_b = {omega_b}
omega_cdm = {omega_cdm}
tau_reio = {tau_reio}
N_ur = 2.0328
N_ncdm = 1
m_ncdm = 0.06
T_ncdm = 0.71611
gravity_model = tabulated_alphas
alpha_functions_file = {HICLASS}/dgf_background_alphas_tabfmt.dat
Omega_smg = -1
expansion_model = wowa
expansion_smg = 0.685, -0.933, 0.
Omega_fld = 0
Omega_Lambda = 0
skip_stability_tests_smg = yes
kineticity_safe_smg = 1e-4
cs2_safe_smg = 1e-4
root = lowl_dgf_
"""

print("Running ΛCDM...")
lcdm_dl = run_class(lcdm_ini, "LCDM", "lowl_lcdm_")

print("Running DGF...")
dgf_dl = run_class(dgf_ini, "DGF", "lowl_dgf_")

if lcdm_dl is None or dgf_dl is None:
    print("FATAL: One or both CLASS runs failed.")
    sys.exit(1)

# ── Comparison table ──
print()
print("=" * 90)
print("PLANCK PR3 LOW-l TT COMPARISON: DGF vs ΛCDM vs OBSERVED")
print("=" * 90)
print(f"Parameters: H0={H0}, omega_cdm={omega_cdm}, logA={logA}, omega_b={omega_b}, n_s={n_s}, tau={tau_reio}")
print()
print(f"{'l':>3s}  {'Obs D_l':>10s}  {'DGF D_l':>10s}  {'ΛCDM D_l':>10s}  {'|Obs-DGF|':>10s}  {'|Obs-ΛCDM|':>10s}  {'Closer':>8s}  {'DGF sign':>10s}")
print("-" * 90)

dgf_wins = 0
lcdm_wins = 0
ties = 0
sign_pattern = []

for ell in range(2, 31):
    obs = planck_obs[ell]
    dgf = dgf_dl.get(ell, float('nan'))
    lcdm = lcdm_dl.get(ell, float('nan'))

    diff_dgf = abs(obs - dgf)
    diff_lcdm = abs(obs - lcdm)

    if diff_dgf < diff_lcdm:
        closer = "DGF"
        dgf_wins += 1
    elif diff_lcdm < diff_dgf:
        closer = "ΛCDM"
        lcdm_wins += 1
    else:
        closer = "TIE"
        ties += 1

    # Sign: DGF prediction relative to ΛCDM
    if dgf > lcdm:
        sign = "+"  # DGF enhanced
    elif dgf < lcdm:
        sign = "−"  # DGF suppressed
    else:
        sign = "0"
    sign_pattern.append(sign)

    print(f"{ell:3d}  {obs:10.1f}  {dgf:10.1f}  {lcdm:10.1f}  {diff_dgf:10.1f}  {diff_lcdm:10.1f}  {closer:>8s}  {sign:>10s}")

print("-" * 90)
print(f"DGF closer: {dgf_wins}/{dgf_wins+lcdm_wins+ties}  |  ΛCDM closer: {lcdm_wins}/{dgf_wins+lcdm_wins+ties}  |  Ties: {ties}")
print()
print(f"DGF-vs-ΛCDM sign pattern (l=2..30): {''.join(sign_pattern)}")
print()

# Chi-squared comparison using symmetric error approximation
# Use average of asymmetric errors for rough chi2
planck_err = {
    2: (132.4+533.1)/2, 3: (450.5+1212.3)/2, 4: (294.1+666.5)/2,
    5: (574.4+1155.8)/2, 6: (201.2+375.8)/2, 7: (381.6+670.8)/2,
    8: (192.0+323.4)/2, 9: (214.6+349.6)/2, 10: (232.8+367.9)/2,
    11: (240.3+371.0)/2, 12: (205.0+310.6)/2, 13: (158.6+235.6)/2,
    14: (199.4+290.8)/2, 15: (243.5+349.7)/2, 16: (151.0+215.8)/2,
    17: (236.2+332.1)/2, 18: (156.3+216.9)/2, 19: (211.6+291.0)/2,
    20: (144.8+197.6)/2, 21: (136.1+185.3)/2, 22: (131.7+177.0)/2,
    23: (155.9+208.7)/2, 24: (175.6+233.2)/2, 25: (169.4+224.3)/2,
    26: (164.4+215.4)/2, 27: (190.3+247.9)/2, 28: (215.1+279.1)/2,
    29: (183.1+236.7)/2, 30: (274.7+274.7)/2,
}

chi2_dgf = sum((planck_obs[l] - dgf_dl.get(l, 0))**2 / planck_err[l]**2 for l in range(2, 31))
chi2_lcdm = sum((planck_obs[l] - lcdm_dl.get(l, 0))**2 / planck_err[l]**2 for l in range(2, 31))

print(f"Approximate chi2 (l=2-30):")
print(f"  DGF:  {chi2_dgf:.2f}  (chi2/dof = {chi2_dgf/29:.3f})")
print(f"  ΛCDM: {chi2_lcdm:.2f}  (chi2/dof = {chi2_lcdm/29:.3f})")
print(f"  Δχ2 = {chi2_dgf - chi2_lcdm:+.2f} (positive = ΛCDM better)")
print()

# ── Plot ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1]})

ells = np.arange(2, 31)
obs_vals = [planck_obs[l] for l in ells]
dgf_vals = [dgf_dl.get(l, np.nan) for l in ells]
lcdm_vals = [lcdm_dl.get(l, np.nan) for l in ells]
err_lo = [planck_err[l] * 0.6 for l in ells]  # rough lower
err_hi = [planck_err[l] * 1.4 for l in ells]  # rough upper

ax1.errorbar(ells, obs_vals, yerr=[err_lo, err_hi], fmt='ko', markersize=5, capsize=3, label='Planck PR3 (Commander)')
ax1.plot(ells, dgf_vals, 'r-o', markersize=4, linewidth=1.5, label='DGF')
ax1.plot(ells, lcdm_vals, 'b--s', markersize=4, linewidth=1.5, label='ΛCDM')
ax1.set_ylabel(r'$D_\ell^{TT}$ [$\mu K^2$]', fontsize=12)
ax1.set_title(r'Planck PR3 low-$\ell$ TT: DGF vs $\Lambda$CDM', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.set_xlim(1.5, 30.5)
ax1.grid(True, alpha=0.3)

# Residuals
residual_dgf = [(planck_obs[l] - dgf_dl.get(l, np.nan)) for l in ells]
residual_lcdm = [(planck_obs[l] - lcdm_dl.get(l, np.nan)) for l in ells]
ax2.bar(ells - 0.15, residual_dgf, 0.3, color='red', alpha=0.7, label='Obs − DGF')
ax2.bar(ells + 0.15, residual_lcdm, 0.3, color='blue', alpha=0.7, label='Obs − ΛCDM')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_xlabel(r'Multipole $\ell$', fontsize=12)
ax2.set_ylabel(r'$\Delta D_\ell$ [$\mu K^2$]', fontsize=12)
ax2.legend(fontsize=10)
ax2.set_xlim(1.5, 30.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
outdir = os.path.expanduser("~/Desktop/dgf_master_findings/planck_lowl_comparison/")
os.makedirs(outdir, exist_ok=True)
plot_path = os.path.join(outdir, "planck_lowl_DGF_vs_LCDM.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved: {plot_path}")
