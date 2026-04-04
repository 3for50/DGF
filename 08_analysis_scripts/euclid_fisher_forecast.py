#!/home/joe-research/dgf_env/bin/python3
"""
Euclid-like weak lensing Fisher forecast: DGF vs LCDM.

Computes tomographic cosmic shear C_l using the DGF CosmoPower emulator
and CAMB for LCDM, then evaluates:
  - Fisher matrix for (omega_cdm, ln10A_s) under Euclid survey specs
  - Delta chi2 between DGF and LCDM C_l spectra

DGF emulator: P(k,z) in (Mpc/h)^3, k in h/Mpc
Euclid specs: 15,000 deg^2, 10 tomo bins, n_eff=30/arcmin^2, sigma_e=0.26
"""

import os
import sys
import json
import time
import warnings

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline, interp1d

# ── Constants and fixed cosmology ──────────────────────────────────────────

H0       = 72.1
h        = H0 / 100.0
omega_b  = 0.02238280
n_s      = 0.9660499
W0       = -0.933
WA       = 0.0
C_KMS    = 299792.458   # km/s

T_CMB    = 2.7255
omega_g  = 4.48162687e-7 * T_CMB**4
N_UR     = 2.0328
omega_nu_rel = N_UR * (7./8.) * (4./11.)**(4./3.) * omega_g
omega_r  = omega_g + omega_nu_rel

# Fiducial parameters
OMEGA_CDM_FID = 0.1678
LN10AS_FID    = 2.294

# Euclid survey specs
F_SKY    = 15000.0 / 41253.0      # 15,000 deg^2
N_EFF    = 30.0                    # gal/arcmin^2
SIGMA_E  = 0.26                    # shape noise per component
N_TOMO   = 10                      # tomographic bins
Z_MIN    = 0.001
Z_MAX    = 2.5

# Multipole range
ELL_MIN  = 10
ELL_MAX  = 5000
N_ELL    = 100   # log-spaced

# Numerical derivative step sizes
D_OMEGA_CDM = 0.001
D_LN10AS    = 0.01

MODEL_DIR = '/home/joe-research/dgf_training/models'
OUTPUT_DIR = '/home/joe-research/Desktop/dgf_master_findings/euclid_forecast'


# ── DGF background ────────────────────────────────────────────────────────

def E2(z):
    """H(z)^2 / H0^2 for DGF w0-wa background (flat)."""
    a = 1.0 / (1.0 + z)
    omega_m = omega_b + OMEGA_CDM_FID + 0.0006   # + massive nu approximation
    Omega_m = omega_m / h**2
    Omega_r = omega_r / h**2
    Omega_DE = 1.0 - Omega_m - Omega_r
    f_de = a**(-3 * (1 + W0 + WA)) * np.exp(-3 * WA * (1 - a))
    return Omega_m * (1 + z)**3 + Omega_r * (1 + z)**4 + Omega_DE * f_de


def chi_of_z(z):
    """Comoving distance chi(z) in Mpc for DGF background."""
    if np.isscalar(z):
        if z <= 0:
            return 0.0
        val, _ = quad(lambda zp: C_KMS / H0 / np.sqrt(E2(zp)), 0, z, limit=200)
        return val
    return np.array([chi_of_z(zi) for zi in z])


# ── Smail n(z) and tomographic binning ─────────────────────────────────────

def smail_nz(z, z0=0.9 / np.sqrt(2)):
    """Smail distribution: n(z) = (z/z0)^2 * exp(-(z/z0)^1.5)."""
    return (z / z0)**2 * np.exp(-(z / z0)**1.5)


def build_tomo_bins(n_tomo=N_TOMO, z_min=Z_MIN, z_max=Z_MAX, nz_pts=500):
    """
    Split the Smail n(z) into n_tomo equipopulated bins.

    Returns:
        z_fine : array of z values
        nz_bins : list of n_tomo arrays, normalised n_i(z) for each bin
        z_edges : bin edges
        n_gal_per_bin : galaxies per steradian per bin
    """
    z_fine = np.linspace(z_min, z_max, nz_pts)
    nz_full = smail_nz(z_fine)

    # CDF for equipopulation
    cdf = np.cumsum(nz_full)
    cdf /= cdf[-1]

    # Bin edges at equal CDF quantiles
    z_edges = np.zeros(n_tomo + 1)
    z_edges[0] = z_min
    z_edges[-1] = z_max
    for i in range(1, n_tomo):
        z_edges[i] = np.interp(i / n_tomo, cdf, z_fine)

    # Total galaxy number density in per steradian
    # N_EFF is per arcmin^2; convert: 1 arcmin^2 = (pi/180/60)^2 sr
    n_total_per_sr = N_EFF / (np.pi / 180.0 / 60.0)**2

    # Build normalised n_i(z)
    nz_bins = []
    n_gal_per_bin = np.zeros(n_tomo)
    for i in range(n_tomo):
        mask = (z_fine >= z_edges[i]) & (z_fine < z_edges[i + 1])
        nz_i = np.where(mask, nz_full, 0.0)
        # Fraction of galaxies in this bin
        integral = np.trapz(nz_full[mask], z_fine[mask]) / np.trapz(nz_full, z_fine)
        n_gal_per_bin[i] = n_total_per_sr * integral
        # Normalise n_i(z) so integral = 1
        norm = np.trapz(nz_i, z_fine)
        if norm > 0:
            nz_i = nz_i / norm
        nz_bins.append(nz_i)

    return z_fine, nz_bins, z_edges, n_gal_per_bin


# ── Load DGF emulator ─────────────────────────────────────────────────────

def load_emulator():
    """Load DGF CosmoPower emulator models and normalisation."""
    import tensorflow as tf

    with open(os.path.join(MODEL_DIR, 'normalisation.json')) as f:
        norm = json.load(f)

    k_grid = np.array(norm['k_grid'])
    z_grid = np.array(norm['z_grid'])
    pk_mean = np.array(norm['pk_mean'])
    pk_std = np.array(norm['pk_std'])
    param_mean = np.array(norm['param_mean'], dtype=np.float32)
    param_std = np.array(norm['param_std'], dtype=np.float32)

    models = []
    for iz in range(len(z_grid)):
        m = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, f'pk_z{iz:02d}.keras'))
        models.append(m)

    print(f"  Loaded {len(z_grid)} DGF emulator models, "
          f"k range [{k_grid[0]:.4e}, {k_grid[-1]:.1f}] h/Mpc")
    print(f"  z_grid: {z_grid}")

    return models, norm, k_grid, z_grid, pk_mean, pk_std, param_mean, param_std


def predict_dgf_pk(omega_cdm, ln10A_s, models, norm_data):
    """
    Evaluate DGF emulator P(k,z).

    Returns: pk_all (n_z, n_k) in (Mpc/h)^3, k_grid in h/Mpc, z_grid
    """
    k_grid, z_grid, pk_mean, pk_std, param_mean, param_std = norm_data

    params_in = np.array([[omega_cdm, ln10A_s]], dtype=np.float32)
    params_norm = (params_in - param_mean) / param_std

    pk_all = np.zeros((len(z_grid), len(k_grid)))
    for iz, model in enumerate(models):
        pred_norm = model(params_norm, training=False).numpy()[0]
        log_pk = pred_norm * pk_std[iz] + pk_mean[iz]
        pk_all[iz] = 10**log_pk

    return pk_all, k_grid, z_grid


# ── LCDM P(k,z) from CAMB ─────────────────────────────────────────────────

def compute_lcdm_pk(omega_cdm, ln10A_s, k_grid, z_grid):
    """
    Compute LCDM P(k,z) using CAMB at the same cosmology.

    Returns: pk_all (n_z, n_k) in (Mpc/h)^3, with k in h/Mpc
    """
    import camb

    As = np.exp(ln10A_s) * 1e-10

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=omega_b, omch2=omega_cdm,
                       mnu=0.06, omk=0)
    pars.InitPower.set_params(As=As, ns=n_s)
    pars.set_matter_power(redshifts=sorted(z_grid, reverse=True),
                          kmax=k_grid[-1] * 1.1,
                          nonlinear=True)
    pars.NonLinear = camb.model.NonLinear_both

    results = camb.get_results(pars)

    # Get the interpolator once (returns P in Mpc^3 with k in 1/Mpc)
    pk_interp_camb = results.get_matter_power_interpolator(
        nonlinear=True, var1='delta_tot', var2='delta_tot')

    pk_all = np.zeros((len(z_grid), len(k_grid)))
    for iz, z in enumerate(z_grid):
        k_mpc = k_grid * h           # h/Mpc -> 1/Mpc
        pk_mpc = pk_interp_camb.P(z, k_mpc)
        pk_all[iz] = pk_mpc * h**3   # Mpc^3 -> (Mpc/h)^3

    return pk_all


# ── P(k,z) interpolator ───────────────────────────────────────────────────

def build_pk_interpolator(pk_all, k_grid, z_grid):
    """
    Build a 2D interpolator for P(k,z).
    Input/output in (Mpc/h)^3 with k in h/Mpc.
    """
    log_k = np.log(k_grid)
    log_pk = np.log(np.clip(pk_all, 1e-30, None))
    spline = RectBivariateSpline(z_grid, log_k, log_pk, kx=3, ky=3)

    def pk_interp(z, k):
        """P(k,z) in (Mpc/h)^3 with k in h/Mpc."""
        return np.exp(spline(np.atleast_1d(z),
                             np.log(np.atleast_1d(k)), grid=False))

    return pk_interp


# ── Lensing kernel and C_l computation ─────────────────────────────────────

def compute_lensing_kernels(z_fine, nz_bins, chi_interp):
    """
    Compute lensing efficiency q_i(chi) for each tomographic bin.

    q_i(chi) = (3/2) Omega_m (H0/c)^2 (chi/a)
               int_{chi}^{chi_H} dchi' n_i(z(chi')) (chi'-chi)/chi'
    """
    omega_m = omega_b + OMEGA_CDM_FID + 0.0006
    Omega_m = omega_m / h**2
    prefactor = 1.5 * Omega_m * (H0 / C_KMS)**2   # 1/Mpc^2

    # Build chi(z) and z(chi) mappings on a dense grid
    z_dense = np.linspace(Z_MIN, Z_MAX, 2000)
    chi_dense = chi_interp(z_dense)

    z_of_chi_interp = interp1d(chi_dense, z_dense, kind='cubic',
                                bounds_error=False,
                                fill_value=(Z_MIN, Z_MAX))

    chi_max = chi_interp(Z_MAX)

    # Integration grid in chi
    n_chi = 500
    chi_grid = np.linspace(chi_interp(Z_MIN), chi_max, n_chi)
    z_at_chi = z_of_chi_interp(chi_grid)
    a_at_chi = 1.0 / (1.0 + z_at_chi)

    # dz/dchi = H(z)/c  for converting n_i(z) -> n_i(chi)
    Hz_at_chi = H0 * np.sqrt(E2(z_at_chi))
    dzdchi = Hz_at_chi / C_KMS

    kernels = []
    for i_bin in range(len(nz_bins)):
        nz_i_interp = interp1d(z_fine, nz_bins[i_bin], kind='linear',
                                bounds_error=False, fill_value=0.0)

        n_i_chi = nz_i_interp(z_at_chi) * dzdchi

        # q_i(chi_j) = prefactor * chi_j / a_j * int_{chi_j}^{chi_max}
        #              n_i(chi') (chi' - chi_j) / chi' dchi'
        q_i = np.zeros(n_chi)
        for j in range(n_chi - 1):
            chi_j = chi_grid[j]
            if chi_j < 1.0:
                continue
            chi_above = chi_grid[j:]
            n_above = n_i_chi[j:]
            integrand = n_above * (chi_above - chi_j) / chi_above
            q_i[j] = (prefactor * chi_j / a_at_chi[j]
                       * np.trapz(integrand, chi_above))

        kernels.append(q_i)

    return chi_grid, z_at_chi, kernels


def compute_cl(ell_arr, pk_func, chi_grid, z_at_chi, kernels, n_tomo):
    """
    Compute angular power spectra C_l^{ij} using Limber approximation.

    C_l^{ij} = int dchi/chi^2 q_i(chi) q_j(chi) P_3D((l+0.5)/chi, z(chi))

    The emulator P(k,z) is in (Mpc/h)^3 with k in h/Mpc.
    chi is in Mpc, so k_Mpc = (l+0.5)/chi [1/Mpc], and we convert:
      k_hmpc = k_Mpc / h
      P_Mpc3 = P_hmpc3 / h^3
    """
    n_ell = len(ell_arr)
    n_pairs = n_tomo * (n_tomo + 1) // 2

    # Avoid chi ~ 0 singularity
    valid = chi_grid > 10.0
    chi_v = chi_grid[valid]
    z_v = z_at_chi[valid]
    kernels_v = [k[valid] for k in kernels]

    cl_all = np.zeros((n_pairs, n_ell))

    for il, ell in enumerate(ell_arr):
        k_hmpc = (ell + 0.5) / chi_v / h   # h/Mpc

        # Clip to emulator/interpolation range
        k_min, k_max = 1e-4, 50.0
        mask = (k_hmpc >= k_min) & (k_hmpc <= k_max)
        if np.sum(mask) < 5:
            continue

        pk_vals = np.zeros(len(chi_v))
        pk_vals[mask] = pk_func(z_v[mask], k_hmpc[mask])

        inv_chi2 = 1.0 / chi_v**2
        base_integrand = inv_chi2 * pk_vals / h**3

        idx = 0
        for i in range(n_tomo):
            for j in range(i, n_tomo):
                integrand = kernels_v[i] * kernels_v[j] * base_integrand
                cl_all[idx, il] = np.trapz(integrand, chi_v)
                idx += 1

    return cl_all


# ── Covariance and Fisher matrix ───────────────────────────────────────────

def gaussian_covariance(cl_ij, ell_arr, n_gal_per_bin, n_tomo):
    """
    Gaussian covariance for C_l.

    Cov(C_l^{ij}, C_l^{mn}) = [(C^{im}+N^{im})(C^{jn}+N^{jn})
                              + (C^{in}+N^{in})(C^{jm}+N^{jm})]
                              / ((2l+1) f_sky Delta_l)

    N_l^{ij} = sigma_e^2 / n_i  delta_{ij}
    """
    n_ell = len(ell_arr)
    n_pairs = n_tomo * (n_tomo + 1) // 2

    # Pair index mapping (i,j) -> flat index (i<=j)
    pair_idx = {}
    idx = 0
    for i in range(n_tomo):
        for j in range(i, n_tomo):
            pair_idx[(i, j)] = idx
            pair_idx[(j, i)] = idx
            idx += 1

    # Shape noise per bin (per steradian)
    noise = SIGMA_E**2 / n_gal_per_bin

    # Bin widths for log-spaced ell
    log_ell = np.log(ell_arr)
    delta_ell = np.zeros(n_ell)
    for il in range(n_ell):
        if il == 0:
            delta_ell[il] = np.exp(log_ell[1]) - np.exp(log_ell[0])
        elif il == n_ell - 1:
            delta_ell[il] = np.exp(log_ell[-1]) - np.exp(log_ell[-2])
        else:
            delta_ell[il] = ((np.exp(log_ell[il + 1])
                              - np.exp(log_ell[il - 1])) / 2.0)

    def get_cl_plus_nl(il, i, j):
        cl = cl_ij[pair_idx[(i, j)], il]
        nl = noise[i] if i == j else 0.0
        return cl + nl

    cov = np.zeros((n_ell, n_pairs, n_pairs))

    for il in range(n_ell):
        ell = ell_arr[il]
        pref = 1.0 / ((2 * ell + 1) * F_SKY * delta_ell[il])

        idx_a = 0
        for i in range(n_tomo):
            for j in range(i, n_tomo):
                idx_b = 0
                for m in range(n_tomo):
                    for n in range(m, n_tomo):
                        t1 = (get_cl_plus_nl(il, i, m)
                              * get_cl_plus_nl(il, j, n))
                        t2 = (get_cl_plus_nl(il, i, n)
                              * get_cl_plus_nl(il, j, m))
                        cov[il, idx_a, idx_b] = pref * (t1 + t2)
                        idx_b += 1
                idx_a += 1

    return cov, delta_ell


def compute_fisher(cl_fid, dcl_dtheta, cov, ell_arr, n_params):
    """
    Fisher matrix:
    F_{ab} = sum_l  (dC/dtheta_a)^T  Cov^{-1}  (dC/dtheta_b)
    """
    n_ell = len(ell_arr)
    fisher = np.zeros((n_params, n_params))

    for il in range(n_ell):
        cov_l = cov[il]
        try:
            cov_inv = np.linalg.inv(cov_l)
        except np.linalg.LinAlgError:
            continue

        for a in range(n_params):
            for b in range(a, n_params):
                val = (dcl_dtheta[a][:, il]
                       @ cov_inv
                       @ dcl_dtheta[b][:, il])
                fisher[a, b] += val
                if a != b:
                    fisher[b, a] += val

    return fisher


def compute_delta_chi2(cl_dgf, cl_lcdm, cov):
    """
    Delta chi^2 = sum_l (C^DGF - C^LCDM)^T Cov^{-1} (C^DGF - C^LCDM)
    """
    n_ell = cov.shape[0]
    delta_chi2 = 0.0

    for il in range(n_ell):
        diff = cl_dgf[:, il] - cl_lcdm[:, il]
        try:
            cov_inv = np.linalg.inv(cov[il])
        except np.linalg.LinAlgError:
            continue
        delta_chi2 += diff @ cov_inv @ diff

    return delta_chi2


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    warnings.filterwarnings('ignore')

    print("=" * 72)
    print("  Euclid-like Weak Lensing Fisher Forecast: DGF vs LCDM")
    print("=" * 72)

    # ── 1. Load DGF emulator ──────────────────────────────────────────────
    print("\n[1] Loading DGF CosmoPower emulator...")
    (models, norm, k_grid, z_grid,
     pk_mean, pk_std, param_mean, param_std) = load_emulator()
    norm_data = (k_grid, z_grid, pk_mean, pk_std, param_mean, param_std)

    # ── 2. Evaluate DGF P(k,z) at fiducial ───────────────────────────────
    print("\n[2] Evaluating DGF P(k,z) at fiducial "
          f"(omega_cdm={OMEGA_CDM_FID:.4f}, ln10A_s={LN10AS_FID:.3f})...")
    pk_dgf, _, _ = predict_dgf_pk(OMEGA_CDM_FID, LN10AS_FID,
                                   models, norm_data)
    pk_dgf_interp = build_pk_interpolator(pk_dgf, k_grid, z_grid)

    # ── 3. Compute LCDM P(k,z) via CAMB ──────────────────────────────────
    print("\n[3] Computing LCDM P(k,z) via CAMB at same cosmology...")
    pk_lcdm = compute_lcdm_pk(OMEGA_CDM_FID, LN10AS_FID, k_grid, z_grid)
    pk_lcdm_interp = build_pk_interpolator(pk_lcdm, k_grid, z_grid)

    # Report P(k) ratio at z~0.5
    iz_05 = np.argmin(np.abs(z_grid - 0.5))
    ratio = pk_dgf[iz_05] / pk_lcdm[iz_05]
    print(f"  P_DGF/P_LCDM at z={z_grid[iz_05]:.1f}: "
          f"mean={np.mean(ratio):.4f}, max={np.max(ratio):.4f} "
          f"(expected ~1.08 from G_eff enhancement)")

    # ── 4. Build tomographic bins ─────────────────────────────────────────
    print(f"\n[4] Building {N_TOMO}-bin tomographic n(z) "
          f"(Smail, equipopulated)...")
    z_fine, nz_bins, z_edges, n_gal_per_bin = build_tomo_bins()
    print(f"  Bin edges: "
          f"{np.array2string(z_edges, precision=3, separator=', ')}")
    print(f"  Galaxies/bin (per sr): "
          f"{np.array2string(n_gal_per_bin, precision=0, separator=', ')}")

    # ── 5. Compute lensing kernels ────────────────────────────────────────
    print("\n[5] Computing lensing efficiency kernels...")
    z_for_chi = np.linspace(0, 3.5, 3000)
    chi_arr = chi_of_z(z_for_chi)
    chi_interp = interp1d(z_for_chi, chi_arr, kind='cubic',
                           bounds_error=False,
                           fill_value=(0, chi_arr[-1]))

    chi_grid, z_at_chi, kernels = compute_lensing_kernels(
        z_fine, nz_bins, chi_interp)
    print(f"  chi range: [{chi_grid[0]:.1f}, {chi_grid[-1]:.1f}] Mpc")

    # ── 6. Compute C_l for DGF and LCDM ──────────────────────────────────
    ell_arr = np.logspace(np.log10(ELL_MIN), np.log10(ELL_MAX),
                          N_ELL).astype(int)
    ell_arr = np.unique(ell_arr).astype(float)
    n_ell = len(ell_arr)
    n_pairs = N_TOMO * (N_TOMO + 1) // 2

    print(f"\n[6] Computing C_l for {n_ell} multipoles, "
          f"{n_pairs} bin pairs...")
    print("  DGF C_l...")
    cl_dgf = compute_cl(ell_arr, pk_dgf_interp,
                         chi_grid, z_at_chi, kernels, N_TOMO)
    print("  LCDM C_l...")
    cl_lcdm = compute_cl(ell_arr, pk_lcdm_interp,
                          chi_grid, z_at_chi, kernels, N_TOMO)

    # ── 7. Numerical derivatives for Fisher ───────────────────────────────
    print("\n[7] Computing numerical derivatives "
          "dC_l/d(omega_cdm), dC_l/d(ln10A_s)...")

    # d/d(omega_cdm)
    pk_p, _, _ = predict_dgf_pk(OMEGA_CDM_FID + D_OMEGA_CDM, LN10AS_FID,
                                 models, norm_data)
    pk_m, _, _ = predict_dgf_pk(OMEGA_CDM_FID - D_OMEGA_CDM, LN10AS_FID,
                                 models, norm_data)
    cl_p = compute_cl(ell_arr, build_pk_interpolator(pk_p, k_grid, z_grid),
                      chi_grid, z_at_chi, kernels, N_TOMO)
    cl_m = compute_cl(ell_arr, build_pk_interpolator(pk_m, k_grid, z_grid),
                      chi_grid, z_at_chi, kernels, N_TOMO)
    dcl_domega = (cl_p - cl_m) / (2 * D_OMEGA_CDM)

    # d/d(ln10A_s)
    pk_p, _, _ = predict_dgf_pk(OMEGA_CDM_FID, LN10AS_FID + D_LN10AS,
                                 models, norm_data)
    pk_m, _, _ = predict_dgf_pk(OMEGA_CDM_FID, LN10AS_FID - D_LN10AS,
                                 models, norm_data)
    cl_p = compute_cl(ell_arr, build_pk_interpolator(pk_p, k_grid, z_grid),
                      chi_grid, z_at_chi, kernels, N_TOMO)
    cl_m = compute_cl(ell_arr, build_pk_interpolator(pk_m, k_grid, z_grid),
                      chi_grid, z_at_chi, kernels, N_TOMO)
    dcl_dln10As = (cl_p - cl_m) / (2 * D_LN10AS)

    dcl_dtheta = [dcl_domega, dcl_dln10As]

    # ── 8. Gaussian covariance ────────────────────────────────────────────
    print("\n[8] Computing Gaussian covariance (using DGF fiducial)...")
    cov, delta_ell = gaussian_covariance(cl_dgf, ell_arr,
                                          n_gal_per_bin, N_TOMO)

    # ── 9. Fisher matrix ──────────────────────────────────────────────────
    print("\n[9] Computing Fisher matrix...")
    fisher = compute_fisher(cl_dgf, dcl_dtheta, cov, ell_arr, n_params=2)

    print("\n  Fisher matrix:")
    print(f"    F(omega_cdm, omega_cdm)  = {fisher[0, 0]:.6e}")
    print(f"    F(omega_cdm, ln10A_s)    = {fisher[0, 1]:.6e}")
    print(f"    F(ln10A_s, ln10A_s)      = {fisher[1, 1]:.6e}")

    # Parameter constraints
    fisher_inv = None
    sigma_omega_cdm = np.nan
    sigma_ln10As = np.nan
    rho = np.nan
    try:
        fisher_inv = np.linalg.inv(fisher)
        sigma_omega_cdm = np.sqrt(fisher_inv[0, 0])
        sigma_ln10As = np.sqrt(fisher_inv[1, 1])
        rho = fisher_inv[0, 1] / (sigma_omega_cdm * sigma_ln10As)

        print("\n  Marginalised 1-sigma constraints (Euclid WL only):")
        print(f"    sigma(omega_cdm)  = {sigma_omega_cdm:.6f}  "
              f"({sigma_omega_cdm / OMEGA_CDM_FID * 100:.2f}%)")
        print(f"    sigma(ln10A_s)    = {sigma_ln10As:.6f}  "
              f"({sigma_ln10As / LN10AS_FID * 100:.2f}%)")
        print(f"    correlation       = {rho:.4f}")
    except np.linalg.LinAlgError:
        print("  WARNING: Fisher matrix is singular, cannot invert.")

    # ── 10. Delta chi2: DGF vs LCDM ──────────────────────────────────────
    print("\n[10] Computing Delta chi^2 between DGF and LCDM C_l...")
    dchi2 = compute_delta_chi2(cl_dgf, cl_lcdm, cov)
    n_data = n_pairs * n_ell
    significance = np.sqrt(dchi2)

    print(f"\n  Delta chi^2 (DGF vs LCDM)  = {dchi2:.2f}")
    print(f"  N_data (ell x pairs)       = {n_data}")
    print(f"  Detection significance     = {significance:.1f} sigma")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Survey: Euclid-like, 15000 deg^2, {N_TOMO} tomo bins, "
          f"n_eff={N_EFF:.0f}/arcmin^2, sigma_e={SIGMA_E}")
    print(f"  Multipole range: ell = [{ELL_MIN}, {ELL_MAX}], {n_ell} bins")
    print(f"  DGF fiducial: omega_cdm={OMEGA_CDM_FID}, "
          f"ln10A_s={LN10AS_FID}")
    print(f"  DGF gravity: alpha_K=4.005, alpha_B=1.018, "
          f"alpha_M=0, alpha_T=0")
    print(f"  G_eff enhancement: ~+8.2% at z~0.5")
    print()
    print(f"  Fisher constraints (DGF model, marginalised):")
    print(f"    sigma(omega_cdm) = {sigma_omega_cdm:.6f}")
    print(f"    sigma(ln10A_s)   = {sigma_ln10As:.6f}")
    print()
    print(f"  DGF vs LCDM distinguishability:")
    print(f"    Delta chi^2      = {dchi2:.2f}")
    print(f"    Significance     = {significance:.1f} sigma")
    print("=" * 72)

    elapsed = time.time() - t0
    print(f"\n  Elapsed time: {elapsed:.1f} s")

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        'survey': {
            'name': 'Euclid-like',
            'area_deg2': 15000,
            'n_tomo': N_TOMO,
            'n_eff_arcmin2': N_EFF,
            'sigma_e': SIGMA_E,
            'z_range': [Z_MIN, Z_MAX],
            'ell_range': [ELL_MIN, ELL_MAX],
            'n_ell': int(n_ell),
        },
        'fiducial': {
            'omega_cdm': OMEGA_CDM_FID,
            'ln10A_s': LN10AS_FID,
            'H0': H0,
            'omega_b': omega_b,
            'n_s': n_s,
            'w0': W0,
        },
        'dgf_gravity': {
            'alpha_K': 4.005,
            'alpha_B': 1.018,
            'alpha_M': 0.0,
            'alpha_T': 0.0,
            'G_eff_peak_percent': 8.2,
            'G_eff_peak_z': 0.5,
        },
        'fisher_matrix': fisher.tolist(),
        'fisher_inverse': (fisher_inv.tolist()
                           if fisher_inv is not None else None),
        'constraints': {
            'sigma_omega_cdm': float(sigma_omega_cdm),
            'sigma_ln10As': float(sigma_ln10As),
            'correlation': float(rho),
        },
        'dgf_vs_lcdm': {
            'delta_chi2': float(dchi2),
            'n_data': int(n_data),
            'significance_sigma': float(significance),
        },
        'tomo_bin_edges': z_edges.tolist(),
    }

    results_file = os.path.join(OUTPUT_DIR, 'euclid_fisher_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    # Save C_l arrays
    spectra_file = os.path.join(OUTPUT_DIR, 'euclid_cl_spectra.npz')
    np.savez(spectra_file,
             ell=ell_arr,
             cl_dgf=cl_dgf,
             cl_lcdm=cl_lcdm,
             z_edges=z_edges,
             fisher=fisher)
    print(f"  C_l spectra saved to: {spectra_file}")


if __name__ == '__main__':
    main()
