"""
KiDS-1000 cosmic shear likelihood for cobaya.

Port of Asgari et al. 2021 (arXiv:2007.15633) xi+/xi- analysis.
Optimised for DGF: fixed background pre-computes r(z) and g_i(r) once.

Data: FITS file with xi+/xi- (270 points: 15 pairs × 9 angles × 2)
Covariance: 270×270 analytical from FITS + SOM calibration contribution

Usage in cobaya YAML:
  likelihood:
    kids1000_cobaya.KiDS1000:
      data_directory: /path/to/KiDS-1000
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

# Physical constants
_Mpc_cm    = 3.08568025e24
_Msun_g    = 1.98892e33
_G_cgs     = 6.673e-8
_H100_s    = 100. / (_Mpc_cm * 1e-5)
_G_Mpc_Msun = _Msun_g * _G_cgs / _Mpc_cm**3

# Baryon feedback AGN parameters (Harnois-Deraps et al. 2014, Table 2)
_AGN = {
    'A2': -0.11900, 'B2':  0.1300, 'C2':  0.6000, 'D2':  0.002110, 'E2': -2.0600,
    'A1':  0.30800, 'B1': -0.6600, 'C1': -0.7600, 'D1': -0.002950, 'E1':  1.8400,
    'A0':  0.15000, 'B0':  1.2200, 'C0':  1.3800, 'D0':  0.001300, 'E0':  3.5700,
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
    exp1 = np.clip((B_z * x - C_z)**3, -50., 50.)
    exp2 = np.clip(E_z * x, -50., 50.)
    return 1. - A_bary * (A_z * np.exp(exp1) - D_z * x * np.exp(exp2))


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


class KiDS1000(Likelihood):

    data_directory: str = '/home/joe-research/montepython_gpu/data/KiDS-1000'
    data_file: str = ('xipm_KIDS1000_BlindC_with_m_bias_V1.0.0A_ugriZYJHKs_photoz_SG_mask_'
                      'LF_svn_309c_2Dbins_v2_goldclasses_Flag_SOM_Fid.fits')
    use_som_cov: bool = False
    som_cov_file: str = 'SOM_cov_multiplied.asc'
    marginalize_over_multiplicative_bias_uncertainty: bool = True
    err_multiplicative_bias: float = 0.02   # KiDS-1000: ~2% per bin
    nzmax: int = 120
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
        self.nzbins  = 5
        self.nzcorrs = self.nzbins * (self.nzbins + 1) // 2  # 15
        self.ntheta  = 9
        self.a2r     = math.pi / (180. * 60.)

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

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        from astropy.io import fits

        fpath = os.path.join(self.data_directory, self.data_file)
        with fits.open(fpath) as f:
            xip_data = f['xiP'].data
            xim_data = f['xiM'].data
            cov_full = np.array(f['COVMAT'].data, dtype=float)
            nz_data  = f['NZ_SOURCE'].data

        # Build data vector: xi+ pairs then xi- pairs, each in pair order then angle order
        # Order: (1,1)_ang1..9, (1,2)_ang1..9, ..., (5,5)_ang1..9, then xi-
        self.theta_bins = np.unique(xip_data['ANG'])   # 9 angles in arcmin

        xi_obs = np.zeros(2 * self.nzcorrs * self.ntheta)
        offset = 0
        for arr in [xip_data, xim_data]:
            for b1 in range(1, self.nzbins + 1):
                for b2 in range(b1, self.nzbins + 1):
                    mask = (arr['BIN1'] == b1) & (arr['BIN2'] == b2)
                    vals = arr['VALUE'][mask]
                    xi_obs[offset:offset + self.ntheta] = vals
                    offset += self.ntheta
        self.xi_obs = xi_obs

        # Covariance: analytical from FITS
        covmat = cov_full.copy()

        # Add SOM calibration uncertainty
        if self.use_som_cov:
            som_path = os.path.join(self.data_directory, self.som_cov_file)
            if os.path.exists(som_path):
                som_cov = np.loadtxt(som_path)   # (5,5) per-bin covariance
                # Expand to data vector space: for each pair (i,j), the SOM contribution
                # is som_cov[i,j] added to covariance block.
                # Simplified treatment: add diagonal per-pair contribution.
                for ic, (b1, b2) in enumerate(self._pair_list()):
                    for jc, (b3, b4) in enumerate(self._pair_list()):
                        val = som_cov[b1, b2] + som_cov[b3, b4]
                        for it in range(self.ntheta):
                            for jt in range(self.ntheta):
                                # xi+ block
                                covmat[ic*self.ntheta + it, jc*self.ntheta + jt] += val * 0.5
                                # xi- block
                                n2 = self.nzcorrs * self.ntheta
                                covmat[n2 + ic*self.ntheta + it, n2 + jc*self.ntheta + jt] += val * 0.5

        # Multiplicative bias uncertainty (m-bias): additive to covariance
        if self.marginalize_over_multiplicative_bias_uncertainty:
            xi_cut = self.xi_obs
            cov_m  = np.outer(xi_cut, xi_cut) * 4. * self.err_multiplicative_bias**2
            covmat = covmat + cov_m

        self.cholesky_transform = cholesky(covmat, lower=True)

        # n(z) from FITS NZ_SOURCE
        z_mid = nz_data['Z_MID']
        self.z_p = np.linspace(0., z_mid.max(), self.nzmax)
        self.pz      = np.zeros((self.nzmax, self.nzbins))
        self.pz_norm = np.zeros(self.nzbins)
        for zbin in range(self.nzbins):
            hz = nz_data['BIN{:d}'.format(zbin + 1)]
            spl = itp.interp1d(np.concatenate(([0.], z_mid)),
                               np.concatenate(([0.], hz)),
                               kind='linear', bounds_error=False, fill_value=0.)
            self.pz[:, zbin] = spl(self.z_p)
            dz = self.z_p[1:] - self.z_p[:-1]
            self.pz_norm[zbin] = np.sum(0.5*(self.pz[1:,zbin] + self.pz[:-1,zbin]) * dz)
            if self.pz_norm[zbin] == 0:
                self.pz_norm[zbin] = 1.

        self.zmax = self.z_p.max()
        self._z_pk_max = 3.5   # emulator trained up to z=3.5

    def _pair_list(self):
        pairs = []
        for b1 in range(self.nzbins):
            for b2 in range(b1, self.nzbins):
                pairs.append((b1, b2))
        return pairs

    def _one_dim_index(self, Bin1, Bin2):
        if Bin1 <= Bin2:
            return Bin2 + self.nzbins * Bin1 - (Bin1 * (Bin1 + 1)) // 2
        return Bin1 + self.nzbins * Bin2 - (Bin2 * (Bin2 + 1)) // 2

    # ── Fixed DGF background ──────────────────────────────────────────────────

    def _precompute_background(self):
        from scipy.integrate import quad
        C_KMS = 299792.458

        r_arr    = np.zeros(self.nzmax)
        dzdr_arr = np.zeros(self.nzmax)
        for i, z in enumerate(self.z_p):
            if z > 0:
                val, _ = quad(lambda zp: C_KMS / H0_DGF / np.sqrt(_E2_dgf(zp)),
                              0, z, limit=200)
                r_arr[i] = val
            dzdr_arr[i] = H0_DGF * np.sqrt(_E2_dgf(z)) / C_KMS

        dzdr_arr[0] = dzdr_arr[1]
        self._r_dgf    = r_arr
        self._dzdr_dgf = dzdr_arr

        omega_b   = 0.02238280
        omega_cdm = 0.1678
        omega_nu  = 0.0006
        self._Omega_m  = (omega_b + omega_cdm + omega_nu) / h_DGF**2
        self._rho_crit = 3. * (h_DGF * _H100_s)**2 / (8. * math.pi * _G_Mpc_Msun)

        self.log.info("DGF background pre-computed (KiDS-1000, 5 bins).")

    def _precompute_lensing_kernels(self):
        self._g_mean = self._compute_g(self.pz, self.pz_norm,
                                       self._r_dgf, self._dzdr_dgf)

    def _compute_g(self, pz, pz_norm, r, dzdr):
        pr = pz * (dzdr[:, np.newaxis] / pz_norm)
        g  = np.zeros_like(pr)
        for nr in range(len(r) - 1):
            for Bin in range(self.nzbins):
                fun = pr[nr:, Bin] * (r[nr:] - r[nr]) / np.where(r[nr:] > 0, r[nr:], 1e-10)
                g[nr, Bin] = np.sum(0.5 * (fun[1:] + fun[:-1]) * (r[nr+1:] - r[nr:-1]))
                g[nr, Bin] *= 2. * r[nr] * (1. + self.z_p[nr])
        return g

    # ── Theta grid ────────────────────────────────────────────────────────────

    def _setup_theta_grid(self):
        thetamin = np.min(self.theta_bins) * 0.8
        thetamax = np.max(self.theta_bins) * 1.2
        nthetatot = int(np.ceil(math.log(thetamax / thetamin) / self.dlntheta)) + 1
        theta = np.array([thetamin * math.exp(self.dlntheta * it) for it in range(nthetatot)])

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

        lll = np.zeros(nl)
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

        ldl = np.zeros(nl)
        ldl[0]    = lll[0]  * 0.5 * (lll[1] - lll[0])
        ldl[1:-1] = lll[1:-1] * 0.5 * (lll[2:] - lll[:-2])
        ldl[-1]   = lll[-1] * 0.5 * (lll[-1] - lll[-2])

        self._theta     = theta
        self._nthetatot = nthetatot
        self._lll       = lll
        self._il_max    = il_max
        self._ldl       = ldl

    # ── Main likelihood ───────────────────────────────────────────────────────

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
        A_IA   = params_values.get('A_IA',   0.0)

        pk_interp = self.provider.get_Pk_interpolator(
            var_pair=('delta_tot', 'delta_tot'), nonlinear=True)

        r    = self._r_dgf
        dzdr = self._dzdr_dgf
        z_p  = self.z_p
        g    = self._g_mean

        # Intrinsic alignment: growth factor D(z)
        use_IA = (A_IA != 0.0)
        if use_IA:
            from scipy.integrate import quad as _quad
            D = np.zeros(self.nzmax)
            for i, z in enumerate(z_p):
                if z == 0:
                    D[i] = 1.0
                else:
                    val, _ = _quad(lambda zp: 1.0 / (1+zp) / _E2_dgf(zp)**1.5,
                                   0, z, limit=100)
                    D[i] = np.sqrt(_E2_dgf(z)) * val
            D /= D[0] if D[0] != 0 else 1.0

            const_IA = 5e-14 / h_DGF**2
            z0 = 0.3
            factor_IA = (-A_IA * const_IA * self._rho_crit * self._Omega_m
                         / np.where(D > 0, D, 1e-10)
                         * ((1. + z_p) / (1. + z0))**0)

        # C_l integration
        dr = r[1:] - r[:-1]
        Cl = np.zeros((self.nlmax, self.nzcorrs))
        kmax_invMpc = self.k_max_h_by_Mpc * h_DGF
        pr_arr = self.pz * (dzdr[:, np.newaxis] / self.pz_norm)

        r2_safe = np.where(r > 0, r**2, 1.)   # avoid 0/0; g=0 at r=0 anyway
        z_clipped = np.clip(z_p, 0., self._z_pk_max)  # don't extrapolate emulator

        for il in range(self.nlmax):
            k_invMpc = (self.l[il] + 0.5) / np.where(r > 0, r, 1e-10)
            k_hMpc   = k_invMpc / h_DGF
            in_range = k_invMpc < kmax_invMpc
            # Only evaluate P(k,z) for k within range to avoid
            # extrapolation errors from CAMB-based interpolators
            k_safe   = np.where(in_range, k_hMpc, 1.0)
            pk_raw   = pk_interp.P(z_clipped, k_safe, grid=False)
            pk_dm    = np.where(in_range, pk_raw, 0.0)
            pk_dm /= h_DGF**3
            pk_eff = pk_dm * _baryon_feedback(k_hMpc, z_p, A_bary)

            for Bin1 in range(self.nzbins):
                for Bin2 in range(Bin1, self.nzbins):
                    idx_c = self._one_dim_index(Bin1, Bin2)

                    integrand = g[:, Bin1] * g[:, Bin2] / r2_safe * pk_eff
                    Cl_GG = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * dr)
                    Cl_GG *= 9./16. * self._Omega_m**2 * (h_DGF / 2997.9)**4

                    if use_IA:
                        fi = factor_IA * pk_eff / r2_safe
                        Cl_II = np.sum(0.5 * ((pr_arr[:,Bin1]*pr_arr[:,Bin2]*fi)[1:] +
                                               (pr_arr[:,Bin1]*pr_arr[:,Bin2]*fi)[:-1]) * dr)
                        Cl_GI = np.sum(0.5 * (((g[:,Bin1]*pr_arr[:,Bin2] +
                                                 g[:,Bin2]*pr_arr[:,Bin1])*fi)[1:] +
                                               ((g[:,Bin1]*pr_arr[:,Bin2] +
                                                 g[:,Bin2]*pr_arr[:,Bin1])*fi)[:-1]) * dr)
                        Cl_GI *= 3./4. * self._Omega_m * (h_DGF / 2997.9)**2
                        Cl[il, idx_c] = Cl_GG + Cl_GI + Cl_II
                    else:
                        Cl[il, idx_c] = Cl_GG

        # Hankel transform C_l → xi+/-
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

        # Assemble theory vector matching data vector ordering
        xi_theory = np.zeros(2 * self.nzcorrs * self.ntheta)
        offset = 0
        for xi_grid in [xi1, xi2]:
            for Bin1 in range(self.nzbins):
                for Bin2 in range(Bin1, self.nzbins):
                    idx_c = self._one_dim_index(Bin1, Bin2)
                    spl = itp.splrep(self._theta, xi_grid[:, idx_c])
                    xi_theory[offset:offset + self.ntheta] = itp.splev(
                        self.theta_bins, spl)
                    offset += self.ntheta

        vec = xi_theory - self.xi_obs
        if np.any(np.isinf(vec)) or np.any(np.isnan(vec)):
            return -1e30

        yt   = solve_triangular(self.cholesky_transform, vec, lower=True)
        chi2 = yt.dot(yt)
        return -chi2 / 2.
