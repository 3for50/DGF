#!/home/joe-research/dgf_env/bin/python3
"""
Cobaya theory class: Direct hi_class DGF computation.

No emulator — calls hi_class directly for each likelihood evaluation.
Slower (~5-10 sec/eval) but exact. For publication-grade results.

Provides P(k,z) from hi_class with tabulated_alphas DGF gravity + native halofit.
Distances computed from the self-consistent hi_class background.
"""

import numpy as np
import os
import sys
import logging
from cobaya.theory import Theory

HICLASS_PY = '/home/joe-research/hi_class/python/build/lib.linux-x86_64-cpython-312'
ALPHA_FILE = '/home/joe-research/hi_class/dgf_background_alphas_tabfmt.dat'

# Fixed DGF cosmology
H0_DEFAULT = 72.1
OMEGA_B    = 0.02238280
N_S        = 0.9660499
TAU_REIO   = 0.054

C_KMS = 299792.458

# Redshifts for P(k,z) output
Z_PK = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

# k grid for P(k) output
K_MIN = 1e-4
K_MAX = 50.
N_K   = 300


class DGFHiClass(Theory):
    """
    Direct hi_class theory for DGF.
    Exact computation — no emulator approximation.
    """

    def initialize(self):
        self.log = logging.getLogger('DGFHiClass')

        # Import classy
        if HICLASS_PY not in sys.path:
            sys.path.insert(0, HICLASS_PY)

        import classy
        self._classy = classy

        self._k_grid = np.geomspace(K_MIN, K_MAX, N_K)
        self._z_grid = np.array(Z_PK)

        self.log.info("DGFHiClass initialised (direct hi_class, no emulator)")

    def get_allow_agnostic(self):
        return True

    def get_requirements(self):
        return {}

    def must_provide(self, **requirements):
        return {}

    def get_can_provide(self):
        return ['Pk_interpolator', 'comoving_radial_distance', 'Hubble',
                'angular_diameter_distance', 'sigma8', 'rdrag']

    def get_can_provide_params(self):
        return ['sigma8', 'Omega_m', 'H0', 'rdrag']

    def calculate(self, state, want_derived=True, **params_values_dict):
        omega_cdm = params_values_dict['omega_cdm']
        ln10A_s   = params_values_dict['ln10A_s']
        H0_cur    = params_values_dict.get('H0', H0_DEFAULT)
        h_cur     = H0_cur / 100.0

        A_s = np.exp(ln10A_s) * 1e-10

        # Run hi_class
        classy = self._classy
        cosmo = classy.Class()
        cosmo.set({
            'gravity_model':        'tabulated_alphas',
            'alpha_functions_file': ALPHA_FILE,
            'Omega_smg':            -1,
            'expansion_model':      'wowa',
            'expansion_smg':        '0.685, -0.933, 0.',
            'Omega_fld':            0,
            'Omega_Lambda':         0,
            'skip_stability_tests_smg': 'yes',
            'kineticity_safe_smg':  1e-4,
            'cs2_safe_smg':         1e-4,
            'H0':        H0_cur,
            'omega_b':   OMEGA_B,
            'omega_cdm': omega_cdm,
            'A_s':       A_s,
            'n_s':       N_S,
            'tau_reio':  TAU_REIO,
            'N_ur':      2.0328,
            'N_ncdm':    1,
            'm_ncdm':    0.06,
            'T_ncdm':    0.71611,
            'output':    'mPk',
            'non linear': 'halofit',
            'P_k_max_h/Mpc': K_MAX,
            'z_pk':      ', '.join(str(z) for z in Z_PK),
        })

        try:
            cosmo.compute()
        except Exception as e:
            self.log.debug(f"hi_class failed: {e}")
            state['pk_all'] = np.full((len(Z_PK), N_K), 1e-30)
            state['k_grid'] = self._k_grid
            state['z_grid'] = self._z_grid
            state['H0_cur'] = H0_cur
            state['h_cur']  = h_cur
            if want_derived:
                state['derived'] = {
                    'sigma8': 0.0, 'Omega_m': 0.0, 'H0': H0_cur, 'rdrag': 0.0,
                }
            return False

        # Extract P(k,z)
        k_grid = self._k_grid
        pk_all = np.zeros((len(Z_PK), N_K))
        for iz, z in enumerate(Z_PK):
            for ik, k in enumerate(k_grid):
                try:
                    # hi_class pk returns P(k) in Mpc^3 with k in 1/Mpc
                    # Convert to (Mpc/h)^3 with k in h/Mpc
                    pk_all[iz, ik] = cosmo.pk(k * h_cur, z) * h_cur**3
                except Exception:
                    pk_all[iz, ik] = 1e-30

        # Extract background for distances
        bg = cosmo.get_background()
        z_bg = bg['z']
        idx = np.argsort(z_bg)
        z_bg_sorted = z_bg[idx]
        # comov. dist. in Mpc
        chi_bg = bg['comov. dist.'][idx]
        # H in 1/Mpc
        H_bg = bg['H [1/Mpc]'][idx]

        sigma8_val = cosmo.sigma8()

        # rdrag from CAMB (more accurate than hi_class fitting)
        try:
            import camb
            cpars = camb.CAMBparams()
            cpars.set_cosmology(H0=H0_cur, ombh2=OMEGA_B, omch2=omega_cdm, mnu=0.06, omk=0)
            cpars.InitPower.set_params(As=A_s, ns=N_S)
            cresults = camb.get_results(cpars)
            rdrag = cresults.get_derived_params()['rdrag']
        except Exception:
            rdrag = 147.0  # fallback

        cosmo.struct_cleanup()
        cosmo.empty()

        state['pk_all']     = pk_all
        state['k_grid']     = k_grid
        state['z_grid']     = self._z_grid
        state['H0_cur']     = H0_cur
        state['h_cur']      = h_cur
        state['z_bg']       = z_bg_sorted
        state['chi_bg']     = chi_bg
        state['H_bg']       = H_bg

        if want_derived:
            Omega_m = (OMEGA_B + omega_cdm) / h_cur**2
            state['derived'] = {
                'sigma8':  sigma8_val,
                'Omega_m': Omega_m,
                'H0':      H0_cur,
                'rdrag':   rdrag,
            }

    # ── Providers ──────────────────────────────────────────────────────────────

    def get_Pk_interpolator(self, var_pair=('delta_tot', 'delta_tot'),
                             nonlinear=True, extrap_kmax=None):
        from scipy.interpolate import RectBivariateSpline

        pk_all = self.current_state['pk_all']
        k_grid = self.current_state['k_grid']
        z_grid = self.current_state['z_grid']

        log_k  = np.log(k_grid)
        log_pk = np.log(np.maximum(pk_all, 1e-30))
        spline = RectBivariateSpline(z_grid, log_k, log_pk, kx=3, ky=3)

        class _PkInterp:
            def P(self, z, k, grid=False):
                log_k_in = np.log(np.atleast_1d(k))
                z_in     = np.atleast_1d(z)
                return np.exp(spline(z_in, log_k_in, grid=grid))
        return _PkInterp()

    def get_comoving_radial_distance(self, z):
        z_bg  = self.current_state['z_bg']
        chi_bg = self.current_state['chi_bg']
        z = np.atleast_1d(z)
        return np.interp(z, z_bg, chi_bg)

    def get_angular_diameter_distance(self, z):
        z = np.atleast_1d(z)
        return self.get_comoving_radial_distance(z) / (1 + z)

    def get_Hubble(self, z, units='km/s/Mpc'):
        z_bg = self.current_state['z_bg']
        H_bg = self.current_state['H_bg']  # in 1/Mpc
        z = np.atleast_1d(z)
        H_invMpc = np.interp(z, z_bg, H_bg)
        if units == "1/Mpc":
            return H_invMpc
        return H_invMpc * C_KMS

    def get_sigma8(self):
        return self.current_state['derived']['sigma8']

    def get_rdrag(self):
        return self.current_state['derived']['rdrag']
