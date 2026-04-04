#!/usr/bin/env python3
"""
Generate training data for CosmoPower emulator with α_B as free dimension.

Parameters: H0, omega_cdm, logA, alpha_B_scale
  alpha_B_scale multiplies the entire α_B(a) column in the tabulated file.
  DGF fiducial = 1.0 (α_B(a=1) = 1.018).
  Range: [0.0, 2.0] — from GR (no braiding) to 2× DGF braiding.

Generates 2000 Latin hypercube samples, runs hi_class for each,
saves lensed C_l^TT, C_l^EE, C_l^TE for l=2..2508.
"""
import numpy as np
import subprocess, os, sys, time, json, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

HICLASS = "/home/joe-research/hi_class"
ALPHA_FILE = os.path.join(HICLASS, "dgf_background_alphas_tabfmt.dat")
OUTDIR = "/home/joe-research/dgf_training/cl_emulator_alphaB"
os.makedirs(OUTDIR, exist_ok=True)

N_TRAIN = 2000
LMAX = 2508

# Parameter ranges
RANGES = {
    "H0":             (60.0, 80.0),
    "omega_cdm":      (0.09, 0.16),
    "logA":           (2.7, 3.4),
    "alpha_B_scale":  (0.0, 2.0),
}

# Fixed parameters
OMEGA_B = 0.02238280
N_S = 0.9660499
TAU = 0.054

# Load fiducial alpha table
alpha_data = np.loadtxt(ALPHA_FILE)
# Columns: a, alpha_K, alpha_B, alpha_M, alpha_T

def latin_hypercube(n, d, seed=42):
    """Simple LHS sampling."""
    rng = np.random.default_rng(seed)
    samples = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.uniform(size=n)) / n
    return samples

def make_scaled_alpha_file(alpha_B_scale, tmpdir):
    """Create alpha table with scaled α_B column."""
    scaled = alpha_data.copy()
    scaled[:, 2] *= alpha_B_scale  # Scale α_B column
    # Also scale α_K proportionally to maintain consistency
    # α_K includes kinetic contributions from G3, scales with α_B^2 approximately
    # But for the emulator we want to isolate α_B effect, so keep α_K at DGF value
    # Actually: α_K = 2*X + 12*A^2*X^2*H^2, where A = 6*phi*alpha_B_scale
    # For simplicity and to isolate the braiding effect, only scale α_B
    path = os.path.join(tmpdir, f"alphas_scaled.dat")
    header = "# a  alpha_K  alpha_B  alpha_M  alpha_T"
    np.savetxt(path, scaled, header=header, comments='')
    return path

def run_single(idx, params):
    """Run hi_class for one parameter set, return (idx, cl_tt, cl_ee, cl_te) or (idx, None)."""
    H0, omega_cdm, logA, alpha_B_scale = params
    A_s = np.exp(logA) * 1e-10

    tmpdir = tempfile.mkdtemp(prefix=f"hiclass_{idx}_")
    alpha_path = make_scaled_alpha_file(alpha_B_scale, tmpdir)

    ini = f"""output = tCl pCl lCl
lensing = yes
l_max_scalars = {LMAX}
format = camb
A_s = {A_s:.8e}
n_s = {N_S}
H0 = {H0:.6f}
omega_b = {OMEGA_B}
omega_cdm = {omega_cdm:.8f}
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
        result = subprocess.run(
            [os.path.join(HICLASS, "class"), ini_path],
            capture_output=True, text=True, timeout=120, cwd=HICLASS
        )

        cl_file = os.path.join(tmpdir, "cl_00_cl_lensed.dat")
        if not os.path.exists(cl_file):
            return (idx, None)

        data = np.loadtxt(cl_file, comments='#')
        # format=camb: columns l, TT, EE, BB, TE, ...
        # Extract l=2..LMAX
        ells = data[:, 0].astype(int)
        mask = (ells >= 2) & (ells <= LMAX)
        cl_tt = data[mask, 1]   # D_l^TT in uK^2
        cl_ee = data[mask, 2]   # D_l^EE
        cl_te = data[mask, 4]   # D_l^TE

        if len(cl_tt) != LMAX - 1:
            return (idx, None)

        return (idx, cl_tt, cl_ee, cl_te)

    except Exception as e:
        return (idx, None)
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

# ── Generate LHS samples ──
print(f"Generating {N_TRAIN} Latin Hypercube samples...")
lhs = latin_hypercube(N_TRAIN, 4, seed=42)

params_list = []
param_names = list(RANGES.keys())
for i in range(N_TRAIN):
    p = []
    for j, name in enumerate(param_names):
        lo, hi = RANGES[name]
        p.append(lo + lhs[i, j] * (hi - lo))
    params_list.append(p)

params_array = np.array(params_list)

# ── Run hi_class in parallel ──
N_WORKERS = 4  # Adjust based on CPU cores
print(f"Running {N_TRAIN} hi_class evaluations with {N_WORKERS} workers...")
print(f"Estimated time: {N_TRAIN * 8 / N_WORKERS / 60:.0f} minutes")

cl_tt_all = np.zeros((N_TRAIN, LMAX - 1))
cl_ee_all = np.zeros((N_TRAIN, LMAX - 1))
cl_te_all = np.zeros((N_TRAIN, LMAX - 1))
success = np.zeros(N_TRAIN, dtype=bool)

t0 = time.time()
n_done = 0
n_fail = 0

with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(run_single, i, params_list[i]): i
               for i in range(N_TRAIN)}

    for future in as_completed(futures):
        result = future.result()
        idx = result[0]
        n_done += 1

        if result[1] is None:
            n_fail += 1
        else:
            cl_tt_all[idx] = result[1]
            cl_ee_all[idx] = result[2]
            cl_te_all[idx] = result[3]
            success[idx] = True

        if n_done % 50 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (N_TRAIN - n_done) / rate / 60
            print(f"  {n_done}/{N_TRAIN} done ({n_fail} failed), "
                  f"{rate:.1f}/s, ETA {eta:.0f} min", flush=True)

elapsed = time.time() - t0
n_success = success.sum()
print(f"\nCompleted: {n_success}/{N_TRAIN} successful in {elapsed/60:.1f} min "
      f"({n_fail} failures)")

# ── Save ──
# Keep only successful samples
mask = success
ell_array = np.arange(2, LMAX + 1)

np.save(os.path.join(OUTDIR, "params.npy"), params_array[mask])
np.save(os.path.join(OUTDIR, "cl_tt.npy"), cl_tt_all[mask])
np.save(os.path.join(OUTDIR, "cl_ee.npy"), cl_ee_all[mask])
np.save(os.path.join(OUTDIR, "cl_te.npy"), cl_te_all[mask])
np.save(os.path.join(OUTDIR, "ell.npy"), ell_array)

metadata = {
    "param_names": ["H0", "omega_cdm", "logA", "alpha_B_scale"],
    "ranges": {k: list(v) for k, v in RANGES.items()},
    "lmax": LMAX,
    "n_train": int(n_success),
    "n_failed": int(n_fail),
    "fixed": {"omega_b": OMEGA_B, "n_s": N_S, "tau_reio": TAU},
    "notes": "alpha_B_scale multiplies the DGF alpha_B(a) profile. 1.0 = fiducial DGF.",
    "generation_time_min": round(elapsed / 60, 1),
}
with open(os.path.join(OUTDIR, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved to {OUTDIR}/")
print(f"  params.npy: {params_array[mask].shape}")
print(f"  cl_tt.npy:  {cl_tt_all[mask].shape}")
print(f"  metadata.json")
