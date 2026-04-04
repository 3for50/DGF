#!/usr/bin/env python3
"""
Task 1: Generate α_B-only training data.
500 points varying ONLY α_B scale (0.1–2.0), all other params fixed at DGF fiducial.
"""
import numpy as np
import subprocess, os, sys, time, json, tempfile, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

HICLASS = "/home/joe-research/hi_class"
ALPHA_FILE = os.path.join(HICLASS, "dgf_background_alphas_tabfmt.dat")
OUTDIR = "/home/joe-research/dgf_training/cl_emulator_alphaB_v2"
os.makedirs(OUTDIR, exist_ok=True)

N_TRAIN = 500
LMAX = 2508

# Fixed fiducial parameters
H0 = 65.9
OMEGA_CDM = 0.119
LOGA = 3.017
A_S = np.exp(LOGA) * 1e-10
OMEGA_B = 0.02238280
N_S = 0.9660499
TAU = 0.054

# α_B scale range
AB_MIN, AB_MAX = 0.1, 2.0

alpha_data = np.loadtxt(ALPHA_FILE)

def run_single(idx, alpha_B_scale):
    tmpdir = tempfile.mkdtemp(prefix=f"hiclass_{idx}_")
    scaled = alpha_data.copy()
    scaled[:, 2] *= alpha_B_scale
    alpha_path = os.path.join(tmpdir, "alphas.dat")
    np.savetxt(alpha_path, scaled, header="# a  alpha_K  alpha_B  alpha_M  alpha_T", comments='')

    ini = f"""output = tCl pCl lCl
lensing = yes
l_max_scalars = {LMAX}
format = camb
A_s = {A_S:.8e}
n_s = {N_S}
H0 = {H0}
omega_b = {OMEGA_B}
omega_cdm = {OMEGA_CDM}
tau_reio = {TAU}
N_ur = 2.0328
N_ncdm = 1
m_ncdm = 0.06
T_ncdm = 0.71611
gravity_model = tabulated_alphas
alpha_functions_file = {alpha_path}
Omega_smg = -1
expansion_model = wowa
expansion_smg = 0.685, -0.933, 0.
Omega_fld = 0
Omega_Lambda = 0
skip_stability_tests_smg = yes
kineticity_safe_smg = 1e-4
cs2_safe_smg = 1e-4
root = {tmpdir}/cl_
"""
    ini_path = os.path.join(tmpdir, "run.ini")
    with open(ini_path, 'w') as f:
        f.write(ini)

    try:
        subprocess.run([os.path.join(HICLASS, "class"), ini_path],
                      capture_output=True, text=True, timeout=120, cwd=HICLASS)
        cl_file = os.path.join(tmpdir, "cl_00_cl_lensed.dat")
        if not os.path.exists(cl_file):
            return (idx, alpha_B_scale, None)
        data = np.loadtxt(cl_file, comments='#')
        ells = data[:, 0].astype(int)
        mask = (ells >= 2) & (ells <= LMAX)
        return (idx, alpha_B_scale, data[mask, 1], data[mask, 2], data[mask, 4])
    except:
        return (idx, alpha_B_scale, None)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# Uniform grid
ab_values = np.linspace(AB_MIN, AB_MAX, N_TRAIN)

print(f"Generating {N_TRAIN} hi_class evaluations (α_B only)...", flush=True)
print(f"α_B range: [{AB_MIN}, {AB_MAX}], fixed H0={H0}, ω_cdm={OMEGA_CDM}, logA={LOGA}", flush=True)

cl_tt_all = np.zeros((N_TRAIN, LMAX - 1))
cl_ee_all = np.zeros((N_TRAIN, LMAX - 1))
cl_te_all = np.zeros((N_TRAIN, LMAX - 1))
success = np.zeros(N_TRAIN, dtype=bool)
params_all = np.zeros((N_TRAIN, 4))  # H0, omega_cdm, logA, alpha_B_scale

t0 = time.time()
n_done = 0
n_fail = 0

with ProcessPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(run_single, i, ab_values[i]): i for i in range(N_TRAIN)}
    for future in as_completed(futures):
        result = future.result()
        idx = result[0]
        n_done += 1
        if result[2] is None:
            n_fail += 1
        else:
            cl_tt_all[idx] = result[2]
            cl_ee_all[idx] = result[3]
            cl_te_all[idx] = result[4]
            success[idx] = True
            params_all[idx] = [H0, OMEGA_CDM, LOGA, result[1]]
        if n_done % 50 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (N_TRAIN - n_done) / rate / 60
            print(f"  {n_done}/{N_TRAIN} done ({n_fail} failed), {rate:.1f}/s, ETA {eta:.0f} min", flush=True)

elapsed = time.time() - t0
n_ok = success.sum()
print(f"\nCompleted: {n_ok}/{N_TRAIN} in {elapsed/60:.1f} min ({n_fail} failures)", flush=True)

ell_array = np.arange(2, LMAX + 1)
np.save(os.path.join(OUTDIR, "params.npy"), params_all[success])
np.save(os.path.join(OUTDIR, "cl_tt.npy"), cl_tt_all[success])
np.save(os.path.join(OUTDIR, "cl_ee.npy"), cl_ee_all[success])
np.save(os.path.join(OUTDIR, "cl_te.npy"), cl_te_all[success])
np.save(os.path.join(OUTDIR, "ell.npy"), ell_array)

# Also save the full 4D training data (merge with the existing 2000-point LHS)
# Load existing
OLD = "/home/joe-research/dgf_training/cl_emulator_alphaB"
old_params = np.load(os.path.join(OLD, "params.npy"))
old_tt = np.load(os.path.join(OLD, "cl_tt.npy"))
old_ee = np.load(os.path.join(OLD, "cl_ee.npy"))
old_te = np.load(os.path.join(OLD, "cl_te.npy"))

merged_params = np.vstack([old_params, params_all[success]])
merged_tt = np.vstack([old_tt, cl_tt_all[success]])
merged_ee = np.vstack([old_ee, cl_ee_all[success]])
merged_te = np.vstack([old_te, cl_te_all[success]])

MERGED = os.path.join(OUTDIR, "merged")
os.makedirs(MERGED, exist_ok=True)
np.save(os.path.join(MERGED, "params.npy"), merged_params)
np.save(os.path.join(MERGED, "cl_tt.npy"), merged_tt)
np.save(os.path.join(MERGED, "cl_ee.npy"), merged_ee)
np.save(os.path.join(MERGED, "cl_te.npy"), merged_te)
np.save(os.path.join(MERGED, "ell.npy"), ell_array)

meta = {
    "param_names": ["H0", "omega_cdm", "logA", "alpha_B_scale"],
    "n_new": int(n_ok), "n_old": len(old_params), "n_merged": len(merged_params),
    "alpha_B_range": [AB_MIN, AB_MAX], "n_failed": int(n_fail),
    "generation_time_min": round(elapsed/60, 1),
}
with open(os.path.join(OUTDIR, "metadata.json"), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"Saved: {OUTDIR}/ ({n_ok} new + {len(old_params)} old = {len(merged_params)} merged)", flush=True)
