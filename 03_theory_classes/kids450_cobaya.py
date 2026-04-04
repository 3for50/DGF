"""
KiDS-450 cosmic shear likelihood for cobaya.

Ported from MontePython kids450_cf_likelihood_public (Hildebrandt et al. 2017).
Optimised for DGF: fixed background means lensing kernels g_i(r) are pre-computed once.

References:
  Hildebrandt et al. 2017, MNRAS 465 1454
  Harnois-Deraps et al. 2014 (baryon feedback)
  Koehlinger et al. 2019, MNRAS 484 3126

Usage in cobaya YAML:
  likelihood:
    kids450_cobaya.KiDS450:
      data_directory: /path/to/KiDS-450_COSMIC_SHEAR_DATA_RELEASE
"""

import numpy as np
import os
import math
from scipy import interpolate as itp
from scipy import special
from scipy.linalg import cholesky, solve_triangular
from cobaya.likelihood import Likelihood

# Fixed DGF cosmology (must match theory class)
H0_DGF   = 72.1
h_DGF    = H0_DGF / 100.0
OMEGA_DE = 0.685
W0       = -0.933
WA       = 0.0

# Critical density constant for IA (solar masses per Mpc^3)
_Mpc_cm   = 3.08568025e24
_Msun_g   = 1.98892e33
_G_cgs    = 6.673e-8
_H100_s   = 100. / (_Mpc_cm * 1e-5)
_G_Mpc_Msun = _Msun_g * _G_cgs / _Mpc_cm**3

# Baryon feedback AGN parameters (Harnois-Deraps et al. 2014, Table 2)
_AGN = {
    'A2': -0.11900, 'B2': 0.1300, 'C2': 0.6000, 'D2': 0.002110, 'E2': -2.0600,
    'A1':  0.30800, 'B1': -0.6600, 'C1': -0.7600, 'D1': -0.002950, 'E1': 1.8400,
    'A0':  0.15000, 'B0': 1.2200, 'C0': 1.3800, 'D0': 0.001300, 'E0': 3.5700,
}


def _baryon_feedback(k_hMpc, z, A_bary):
    """Baryon feedback bias^2 (AGN model). k in h/Mpc."""
    x = np.log10(k_hMpc)
    a = 1. / (1. + z)
    a2 = a * a
    A_z = _AGN['A2']*a2 + _AGN['A1']*a + _AGN['A0']
    B_z = _AGN['B2']*a2 + _AGN['B1']*a + _AGN['B0']
    C_z = _AGN['C2']*a2 + _AGN['C1']*a + _AGN['C0']
    D_z = _AGN['D2']*a2 + _AGN['D1']*a + _AGN['D0']
    E_z = _AGN['E2']*a2 + _AGN['E1']*a + _AGN['E0']
    return 1. - A_bary * (A_z * np.exp((B_z * x - C_z)**3) - D_z * x * np.exp(E_z * x))


def _E2_dgf(z):
    """H(z)^2/H0^2 for fixed DGF wowa background (flat, fiducial omega_cdm)."""
    omega_b_val   = 0.02238280
    omega_cdm_val = 0.1678
    omega_nu      = 0.0006
    omega_g  = 4.48162687e-7 * 2.7255**4
    omega_nu_rel = 2.0328 * (7./8.) * (4./11.)**(4./3.) * omega_g
    Omega_m  = (omega_b_val + omega_cdm_val + omega_nu) / h_DGF**2
    Omega_r  = (omega_g + omega_nu_rel) / h_DGF**2
    Omega_DE = 1.0 - Omega_m - Omega_r
    a = 1. / (1. + z)
    f_de = a**(-3*(1 + W0 + WA)) * np.exp(-3 * WA * (1 - a))
    return Omega_m * (1+z)**3 + Omega_r * (1+z)**4 + Omega_DE * f_de


class KiDS450(Likelihood):

    # Cobaya parameters
    data_directory: str = '/home/joe-research/montepython_gpu/data/KiDS-450_COSMIC_SHEAR_DATA_RELEASE'
    nzmax: int = 70
    nz_method: str = 'DIR'
    bootstrap_photoz_errors: bool = True
    index_bootstrap_low: int = 1
    index_bootstrap_high: int = 1000
    marginalize_over_multiplicative_bias_uncertainty: bool = True
    err_multiplicative_bias: float = 0.01
    use_cut_theta: bool = True
    cutvalues_file: str = 'cut_values_fiducial.txt'
    lmax: int = 60000
    dlnl: float = 0.4
    xmax: float = 50.
    dx_below_threshold: float = 0.05
    dx_above_threshold: float = 0.15
    dx_threshold: float = 0.4
    dlntheta: float = 0.25
    k_max_h_by_Mpc: float = 100.
    baryon_model: str = 'AGN'

    def initialize(self):
        self._setup_l_grid()
        self._load_data()
        self._precompute_background()
        self._precompute_lensing_kernels()
        self._setup_theta_grid()

    # ── l grid ────────────────────────────────────────────────────────────────

    def _setup_l_grid(self):
        self.nlmax = int(np.log(self.lmax) / self.dlnl) + 1
        self.dlnl  = np.log(self.lmax) / (self.nlmax - 1)
        self.l     = np.exp(self.dlnl * np.arange(self.nlmax))

        self.z_bins_min = [0.1, 0.3, 0.5, 0.7]
        self.z_bins_max = [0.3, 0.5, 0.7, 0.9]
        self.nzbins  = 4
        self.nzcorrs = self.nzbins * (self.nzbins + 1) // 2
        self.ntheta  = 9
        self.a2r     = math.pi / (180. * 60.)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        # Load data vector
        xi_data = self._load_data_vector()
        self.theta_bins = xi_data[:, 0]
        self.xi_obs     = self._get_xi_obs(xi_data[:, 1:])

        # Load covariance
        covmat = np.loadtxt(os.path.join(self.data_directory,
            'COV_MAT/Cov_mat_all_scales_use_with_kids450_cf_likelihood_public.dat'))

        # Build mask
        if self.use_cut_theta:
            cut_path = os.path.join(self.data_directory, 'CUT_VALUES', self.cutvalues_file)
            cut_values = np.loadtxt(cut_path)
            mask = self._get_mask(cut_values)
        else:
            mask = np.ones(2 * self.nzcorrs * self.ntheta)
        self.mask_indices = np.where(mask == 1)[0]

        # Multiplicative bias uncertainty
        if self.marginalize_over_multiplicative_bias_uncertainty:
            xi_cut = self.xi_obs[self.mask_indices]
            cov_m  = np.outer(xi_cut, xi_cut) * 4. * self.err_multiplicative_bias**2
            covmat = covmat[self.mask_indices][:, self.mask_indices] + cov_m
        else:
            covmat = covmat[np.ix_(self.mask_indices, self.mask_indices)]

        self.cholesky_transform = cholesky(covmat, lower=True)

        # Load mean n(z) for all bins
        self.z_p = np.linspace(0.025, 3.475, self.nzmax)
        zbin_labels = ['{:.1f}t{:.1f}'.format(lo, hi)
                       for lo, hi in zip(self.z_bins_min, self.z_bins_max)]
        self.pz      = np.zeros((self.nzmax, self.nzbins))
        self.pz_norm = np.zeros(self.nzbins)
        for zbin, label in enumerate(zbin_labels):
            fname = os.path.join(self.data_directory,
                f'Nz_{self.nz_method}/Nz_{self.nz_method}_Mean/'
                f'Nz_{self.nz_method}_z{label}.asc')
            zt, hz = np.loadtxt(fname, usecols=[0, 1], unpack=True)
            shift = np.diff(zt)[0] / 2.
            spl = itp.splrep(zt + shift, hz)
            mask_z = (self.z_p >= zt.min()) & (self.z_p <= zt.max())
            self.pz[mask_z, zbin] = itp.splev(self.z_p[mask_z], spl)
            dz = self.z_p[1:] - self.z_p[:-1]
            self.pz_norm[zbin] = np.sum(0.5 * (self.pz[1:, zbin] + self.pz[:-1, zbin]) * dz)

        # Store bootstrap files pattern for later
        self._zbin_labels = zbin_labels
        self.zmax = self.z_p.max()

    def _load_data_vector(self):
        dxip = np.zeros((self.ntheta, self.nzcorrs + 1))
        dxim = np.zeros((self.ntheta, self.nzcorrs + 1))
        idx = 0
        for b1 in range(self.nzbins):
            for b2 in range(b1, self.nzbins):
                fname = os.path.join(self.data_directory,
                    f'DATA_VECTOR/KiDS-450_xi_pm_files/'
                    f'KiDS-450_xi_pm_tomo_{b1+1}_{b2+1}_logbin_mcor.dat')
                theta, xip, xim = np.loadtxt(fname, unpack=True)
                if idx == 0:
                    dxip[:, 0] = theta
                    dxim[:, 0] = theta
                dxip[:, idx + 1] = xip
                dxim[:, idx + 1] = xim
                idx += 1
        return np.concatenate((dxip, dxim))

    def _get_xi_obs(self, temp):
        xi = np.zeros(self.ntheta * self.nzcorrs * 2)
        k = 0
        for j in range(self.nzcorrs):
            for i in range(2 * self.ntheta):
                xi[k] = temp[i, j]
                k += 1
        return xi

    def _get_mask(self, cut_values):
        mask = np.zeros(2 * self.nzcorrs * self.ntheta)
        iz = 0
        for izl in range(self.nzbins):
            for izh in range(izl, self.nzbins):
                iz += 1
                for i in range(self.ntheta):
                    j = (iz - 1) * 2 * self.ntheta
                    lo_p = max(cut_values[izl, 0], cut_values[izh, 0])
                    hi_p = max(cut_values[izl, 1], cut_values[izh, 1])
                    lo_m = max(cut_values[izl, 2], cut_values[izh, 2])
                    hi_m = max(cut_values[izl, 3], cut_values[izh, 3])
                    if lo_p < self.theta_bins[i] < hi_p:
                        mask[j + i] = 1
                    if lo_m < self.theta_bins[i] < hi_m:
                        mask[self.ntheta + j + i] = 1
        return mask

    # ── Fixed DGF background (pre-computed once) ───────────────────────────────

    def _precompute_background(self):
        """Compute r(z) and dzdr for fixed DGF background. Done ONCE."""
        from scipy.integrate import quad
        C_KMS = 299792.458

        z_arr = self.z_p
        r_arr = np.zeros_like(z_arr)
        for i, z in enumerate(z_arr):
            if z > 0:
                val, _ = quad(lambda zp: C_KMS / H0_DGF / np.sqrt(_E2_dgf(zp)),
                              0, z, limit=200)
                r_arr[i] = val

        # dzdr = 1 / (dr/dz) = H(z)/c
        dzdr_arr = H0_DGF * np.sqrt(_E2_dgf(z_arr)) / C_KMS
        dzdr_arr[0] = dzdr_arr[1]  # avoid division by zero at z=0

        self._r_dgf    = r_arr
        self._dzdr_dgf = dzdr_arr

        # Omega_m for IA
        omega_b   = 0.02238280
        omega_cdm = 0.1678
        omega_nu  = 0.0006
        self._Omega_m = (omega_b + omega_cdm + omega_nu) / h_DGF**2
        self._rho_crit = 3. * (h_DGF * _H100_s)**2 / (8. * math.pi * _G_Mpc_Msun)

        self.log.info("DGF background pre-computed (r(z), dzdr fixed).")

    def _precompute_lensing_kernels(self):
        """
        Compute g_i(r) for each tomographic bin using fixed n(z) mean.
        These are FIXED in DGF (since r(z) is fixed), so we pre-compute them.
        They will be recomputed per step only when bootstrap n(z) is used.
        """
        self._g_mean = self._compute_g(self.pz, self.pz_norm, self._r_dgf, self._dzdr_dgf)

    def _compute_g(self, pz, pz_norm, r, dzdr):
        """Lensing efficiency g_i(r) for each bin."""
        pr = pz * (dzdr[:, np.newaxis] / pz_norm)   # (nzmax, nzbins)
        g  = np.zeros_like(pr)
        for nr in range(len(r) - 1):
            for Bin in range(self.nzbins):
                fun = pr[nr:, Bin] * (r[nr:] - r[nr]) / np.where(r[nr:] > 0, r[nr:], 1e-10)
                g[nr, Bin] = np.sum(0.5 * (fun[1:] + fun[:-1]) * (r[nr+1:] - r[nr:-1]))
                g[nr, Bin] *= 2. * r[nr] * (1. + self.z_p[nr])
        return g

    # ── Theta grid and Bessel kernel ──────────────────────────────────────────

    def _setup_theta_grid(self):
        thetamin = np.min(self.theta_bins) * 0.8
        thetamax = np.max(self.theta_bins) * 1.2
        nthetatot = int(np.ceil(math.log(thetamax / thetamin) / self.dlntheta)) + 1
        theta = np.array([thetamin * math.exp(self.dlntheta * it) for it in range(nthetatot)])

        # Build l grid for Hankel transform
        ll, il = 1., 0
        while ll * theta[-1] * self.a2r < self.dx_threshold:
            ll += self.dx_below_threshold / theta[-1] / self.a2r
            il += 1
        for it in range(nthetatot):
            while (ll * theta[nthetatot-1-it] * self.a2r < self.xmax and
                   ll + self.dx_above_threshold / theta[nthetatot-1-it] / self.a2r < self.lmax):
                ll += self.dx_above_threshold / theta[nthetatot-1-it] / self.a2r
                il += 1
        nl = il + 1

        lll    = np.zeros(nl)
        il_max = np.zeros(nthetatot, dtype=int)
        il = 0
        lll[0] = 1.
        while lll[il] * theta[-1] * self.a2r < self.dx_threshold:
            il += 1
            lll[il] = lll[il-1] + self.dx_below_threshold / theta[-1] / self.a2r
        for it in range(nthetatot):
            while (lll[il] * theta[nthetatot-1-it] * self.a2r < self.xmax and
                   lll[il] + self.dx_above_threshold / theta[nthetatot-1-it] / self.a2r < self.lmax):
                il += 1
                lll[il] = lll[il-1] + self.dx_above_threshold / theta[nthetatot-1-it] / self.a2r
            il_max[nthetatot-1-it] = il

        ldl       = np.zeros(nl)
        ldl[0]    = lll[0] * 0.5 * (lll[1] - lll[0])
        ldl[1:-1] = lll[1:-1] * 0.5 * (lll[2:] - lll[:-2])
        ldl[-1]   = lll[-1]  * 0.5 * (lll[-1] - lll[-2])

        self._theta    = theta
        self._nthetatot = nthetatot
        self._lll      = lll
        self._il_max   = il_max
        self._ldl      = ldl

    # ── Main likelihood calculation ───────────────────────────────────────────

    def get_requirements(self):
        return {
            'Pk_interpolator': {
                'z':          list(np.linspace(0, self.zmax, 20)),
                'k_max':      self.k_max_h_by_Mpc,
                'nonlinear':  True,
                'vars_pairs': [('delta_tot', 'delta_tot')],
            },
        }

    def logp(self, **params_values):
        A_bary = params_values.get('A_bary', 1.0)
        A_IA   = params_values.get('A_IA', 0.0)

        pk_interp = self.provider.get_Pk_interpolator(
            var_pair=('delta_tot', 'delta_tot'), nonlinear=True)

        r    = self._r_dgf
        dzdr = self._dzdr_dgf
        z_p  = self.z_p

        # Bootstrap n(z) if enabled
        if self.bootstrap_photoz_errors:
            idx = np.random.randint(self.index_bootstrap_low, self.index_bootstrap_high + 1)
            pz      = np.zeros((self.nzmax, self.nzbins))
            pz_norm = np.zeros(self.nzbins)
            for zbin, label in enumerate(self._zbin_labels):
                fname = os.path.join(self.data_directory,
                    f'Nz_{self.nz_method}/Nz_{self.nz_method}_Bootstrap/'
                    f'Nz_z{label}_boot{idx}_{self.nz_method}.asc')
                zt, hz = np.loadtxt(fname, usecols=(0, 1), unpack=True)
                shift = np.diff(zt)[0] / 2.
                spl = itp.splrep(zt + shift, hz)
                mask_z = (z_p >= zt.min()) & (z_p <= zt.max())
                pz[mask_z, zbin] = itp.splev(z_p[mask_z], spl)
                dz = z_p[1:] - z_p[:-1]
                pz_norm[zbin] = np.sum(0.5 * (pz[1:, zbin] + pz[:-1, zbin]) * dz)
            g = self._compute_g(pz, pz_norm, r, dzdr)
        else:
            g = self._g_mean

        # Intrinsic alignment: linear growth factor D(z)
        # For fixed DGF background we can precompute this too, but A_IA varies so
        # the full IA term must be recomputed each step regardless.
        use_IA = (A_IA != 0.0)
        if use_IA:
            # Growth factor D(z) normalized to 1 at z=0
            # For DGF we approximate D(z) ∝ 1/H(z) × integral (rough but consistent with HPC)
            # More precisely: D(z) from growth ODE — use simple fitting for now
            # (HPC didn't vary alpha parameters so growth is also fixed)
            from scipy.integrate import quad as _quad
            D = np.zeros(self.nzmax)
            for i, z in enumerate(z_p):
                if z == 0:
                    D[i] = 1.0
                else:
                    # Integral approximation: D(a) ∝ H(z) ∫ dz' / H(z')^3
                    val, _ = _quad(lambda zp: 1.0 / (1+zp) / _E2_dgf(zp)**1.5,
                                   0, z, limit=100)
                    D[i] = np.sqrt(_E2_dgf(z)) * val
            D /= D[0] if D[0] != 0 else 1.0

            const_IA = 5e-14 / h_DGF**2  # Mpc^3/Msol
            z0 = 0.3
            factor_IA = (-A_IA * const_IA * self._rho_crit * self._Omega_m
                         / np.where(D > 0, D, 1e-10)
                         * ((1. + z_p) / (1. + z0))**0)   # exp_IA=0 per HPC

        # Compute C_l for each correlation pair
        dr = r[1:] - r[:-1]
        Cl    = np.zeros((self.nlmax, self.nzcorrs))
        kmax_invMpc = self.k_max_h_by_Mpc * h_DGF

        for il in range(self.nlmax):
            # P(k = l/r, z) with baryon feedback
            k_invMpc = (self.l[il] + 0.5) / np.where(r > 0, r, 1e-10)
            k_hMpc   = k_invMpc / h_DGF
            pk_dm    = np.where(
                k_invMpc < kmax_invMpc,
                pk_interp.P(z_p, k_hMpc, grid=False),
                0.0
            )
            pk_dm /= h_DGF**3   # convert (Mpc/h)^3 → Mpc^3

            # Apply baryon feedback
            bfb = _baryon_feedback(k_hMpc, z_p, A_bary)
            pk_eff = pk_dm * bfb

            for Bin1 in range(self.nzbins):
                for Bin2 in range(Bin1, self.nzbins):
                    idx_c = self._one_dim_index(Bin1, Bin2)

                    integrand_GG = g[:, Bin1] * g[:, Bin2] / np.where(r**2 > 0, r**2, 1.) * pk_eff
                    Cl_GG = np.sum(0.5 * (integrand_GG[1:] + integrand_GG[:-1]) * dr)
                    Cl_GG *= 9./16. * self._Omega_m**2 * (h_DGF / 2997.9)**4

                    if use_IA:
                        pr_b1 = g[:, Bin1]  # already multiplied by n(z)/norm
                        pr_b2 = g[:, Bin2]
                        # For NLA: pr should be dn/dr not g(r); re-derive
                        pr_arr = (pz if self.bootstrap_photoz_errors else self.pz) * (dzdr[:, np.newaxis] / (pz_norm if self.bootstrap_photoz_errors else self.pz_norm))
                        fi = factor_IA * pk_eff / r**2

                        Cl_II = np.sum(0.5 * ((pr_arr[:, Bin1]*pr_arr[:, Bin2]*fi)[1:] +
                                               (pr_arr[:, Bin1]*pr_arr[:, Bin2]*fi)[:-1]) * dr)

                        Cl_GI = np.sum(0.5 * (((g[:, Bin1]*pr_arr[:, Bin2] + g[:, Bin2]*pr_arr[:, Bin1])*fi)[1:] +
                                               ((g[:, Bin1]*pr_arr[:, Bin2] + g[:, Bin2]*pr_arr[:, Bin1])*fi)[:-1]) * dr)
                        Cl_GI *= 3./4. * self._Omega_m * (h_DGF / 2997.9)**2

                        Cl[il, idx_c] = Cl_GG + Cl_GI + Cl_II
                    else:
                        Cl[il, idx_c] = Cl_GG

        # Hankel transform: C_l → xi+/-
        Cll = np.zeros((self.nzcorrs, len(self._lll)))
        for Bin in range(self.nzcorrs):
            spl = itp.splrep(self.l, Cl[:, Bin])
            Cll[Bin] = itp.splev(self._lll, spl)

        xi1 = np.zeros((self._nthetatot, self.nzcorrs))
        xi2 = np.zeros((self._nthetatot, self.nzcorrs))
        for it in range(self._nthetatot):
            ilmax = self._il_max[it]
            B0 = special.j0(self._lll[:ilmax] * self._theta[it] * self.a2r)
            B4 = special.jv(4, self._lll[:ilmax] * self._theta[it] * self.a2r)
            xi1[it] = np.sum(self._ldl[:ilmax] * Cll[:, :ilmax] * B0, axis=1)
            xi2[it] = np.sum(self._ldl[:ilmax] * Cll[:, :ilmax] * B4, axis=1)

        xi1 /= (2. * math.pi)
        xi2 /= (2. * math.pi)

        # Evaluate at measured theta bins
        xi_theory = np.zeros(np.size(self.xi_obs))
        iz = 0
        for Bin in range(self.nzcorrs):
            iz += 1
            j = (iz - 1) * 2 * self.ntheta
            spl1 = itp.splrep(self._theta, xi1[:, Bin])
            spl2 = itp.splrep(self._theta, xi2[:, Bin])
            xi_theory[j:j + self.ntheta]             = itp.splev(self.theta_bins[:self.ntheta], spl1)
            xi_theory[j + self.ntheta:j + 2*self.ntheta] = itp.splev(self.theta_bins[:self.ntheta], spl2)

        # Chi2
        vec = xi_theory[self.mask_indices] - self.xi_obs[self.mask_indices]
        if np.any(np.isinf(vec)) or np.any(np.isnan(vec)):
            return -1e30

        yt   = solve_triangular(self.cholesky_transform, vec, lower=True)
        chi2 = yt.dot(yt)

        return -chi2 / 2.

    def _one_dim_index(self, Bin1, Bin2):
        if Bin1 <= Bin2:
            return Bin2 + self.nzbins * Bin1 - (Bin1 * (Bin1 + 1)) // 2
        return Bin1 + self.nzbins * Bin2 - (Bin2 * (Bin2 + 1)) // 2
