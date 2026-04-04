#!/home/joe-research/dgf_env/bin/python3
"""
Cobaya theory class: DGF CosmoPower emulator.

Provides P(k, z) for KiDS weak lensing and BAO distances for BAO likelihoods.
Uses GPU-accelerated NN trained on true DGF physics (tabulated_alphas).

Fixed background (H0=72.1, wowa): BAO distances are pre-computed once.
Free params: omega_cdm, ln10A_s (cosmo) + A_bary, A_IA (nuisance, pass-through).
"""

import numpy as np
import os
import json
import logging

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from cobaya.theory import Theory

MODEL_DIR = '/home/joe-research/dgf_training/models'

# Fixed DGF cosmology (must match training)
H0       = 72.1
h        = H0 / 100.0
omega_b  = 0.02238280
omega_cdm_fid = 0.1678  # fiducial (for background only — background is fixed)
n_s      = 0.9660499
tau_reio = 0.054

# DGF background: wowa with Omega_de=0.685, w0=-0.933, wa=0
OMEGA_DE = 0.685
W0       = -0.933
WA       = 0.0
OMEGA_K  = 0.0

# c in km/s
C_KMS = 299792.458


T_CMB    = 2.7255
omega_g  = 4.48162687e-7 * T_CMB**4          # photon physical density
N_UR     = 2.0328
omega_nu_rel = N_UR * (7./8.) * (4./11.)**(4./3.) * omega_g  # massless nu
omega_r  = omega_g + omega_nu_rel             # total radiation (no massive nu)


def _E2(z, omega_cdm_val=None, h_val=None, w0_val=None):
    """H(z)^2 / H0^2 for DGF wowa background (flat universe enforced)."""
    if omega_cdm_val is None:
        omega_cdm_val = omega_cdm_fid
    if h_val is None:
        h_val = h
    if w0_val is None:
        w0_val = W0
    a = 1.0 / (1.0 + z)
    omega_m  = omega_b + omega_cdm_val + 0.0006
    Omega_m  = omega_m / h_val**2
    Omega_r  = omega_r / h_val**2
    Omega_DE = 1.0 - Omega_m - Omega_r   # flat universe: Omega_smg closes the budget
    f_de = a**(-3*(1 + w0_val + WA)) * np.exp(-3 * WA * (1 - a))
    return Omega_m * (1+z)**3 + Omega_r * (1+z)**4 + Omega_DE * f_de


def _compute_rdrag(omega_cdm_val, H0_val=None, w0_val=None):
    """Sound horizon at drag epoch r_drag (Mpc) using CAMB for accuracy."""
    import camb
    if H0_val is None:
        H0_val = H0
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=omega_b, omch2=omega_cdm_val,
                       mnu=0.06, omk=0)
    pars.InitPower.set_params(As=2.1e-9, ns=0.966)
    results = camb.get_results(pars)
    return results.get_derived_params()['rdrag']


def _comoving_distance(z_arr):
    """Comoving distance chi(z) in Mpc for DGF fixed background."""
    from scipy.integrate import quad
    chi = np.zeros(len(z_arr))
    for i, z in enumerate(z_arr):
        if z <= 0:
            chi[i] = 0.0
        else:
            val, _ = quad(lambda zp: C_KMS / H0 / np.sqrt(_E2(zp)), 0, z,
                          limit=200)
            chi[i] = val
    return chi


class DGFCosmoPower(Theory):
    """
    GPU-accelerated DGF theory for cobaya.
    Emulates P(k,z) using trained NNs, provides distances from fixed DGF background.
    """

    # Class-level model cache
    _loaded = False
    _models = None
    _norm   = None
    _k_grid = None
    _z_grid = None

    # Pre-computed background (fixed DGF)
    _bg_computed = False
    _chi_interp  = None   # comoving distance chi(z) interpolator

    def initialize(self):
        self.log = logging.getLogger('DGFCosmoPower')

        if not DGFCosmoPower._loaded:
            self.log.info(f"Loading DGF emulator from {MODEL_DIR}")

            with open(os.path.join(MODEL_DIR, 'normalisation.json')) as f:
                DGFCosmoPower._norm = json.load(f)

            norm = DGFCosmoPower._norm
            DGFCosmoPower._k_grid = np.array(norm['k_grid'])
            DGFCosmoPower._z_grid = np.array(norm['z_grid'])
            n_z = len(DGFCosmoPower._z_grid)

            DGFCosmoPower._models = []
            for iz in range(n_z):
                m = tf.keras.models.load_model(
                    os.path.join(MODEL_DIR, f'pk_z{iz:02d}.keras'))
                DGFCosmoPower._models.append(m)

            DGFCosmoPower._loaded = True
            self.log.info(f"Loaded {n_z} P(k) models (GPU: {len(tf.config.list_physical_devices('GPU'))} devices)")

        # Pre-compute DGF background distances (fixed, only done once)
        if not DGFCosmoPower._bg_computed:
            self._precompute_background()

    def _precompute_background(self):
        """Compute comoving distances for fixed DGF background."""
        from scipy.interpolate import interp1d
        self.log.info("Pre-computing DGF background distances (one-time)...")

        # Dense z grid for interpolation
        z_bg = np.linspace(0, 10, 2000)
        chi_bg = _comoving_distance(z_bg)

        DGFCosmoPower._chi_interp = interp1d(
            z_bg, chi_bg, kind='cubic', bounds_error=False,
            fill_value=(0, chi_bg[-1]))

        DGFCosmoPower._bg_computed = True
        self.log.info("Background pre-computation done.")

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
        # Allow H0 and w0 to be free (for BAO); default to fixed DGF values
        H0_cur    = params_values_dict.get('H0', H0)
        h_cur     = H0_cur / 100.0
        w0_cur    = params_values_dict.get('w0_dgf', W0)

        norm = DGFCosmoPower._norm
        k_grid = DGFCosmoPower._k_grid
        z_grid = DGFCosmoPower._z_grid

        # Normalise input parameters
        param_mean = np.array(norm['param_mean'], dtype=np.float32)
        param_std  = np.array(norm['param_std'],  dtype=np.float32)
        params_in  = np.array([[omega_cdm, ln10A_s]], dtype=np.float32)
        params_norm = (params_in - param_mean) / param_std

        # Run emulator for each redshift
        pk_mean = np.array(norm['pk_mean'])  # (n_z, n_k)
        pk_std  = np.array(norm['pk_std'])

        pk_all = np.zeros((len(z_grid), len(k_grid)))
        for iz, model in enumerate(DGFCosmoPower._models):
            pred_norm = model(params_norm, training=False).numpy()[0]
            log_pk    = pred_norm * pk_std[iz] + pk_mean[iz]
            pk_all[iz] = 10**log_pk

        state['pk_all']    = pk_all
        state['k_grid']    = k_grid
        state['z_grid']    = z_grid
        state['omega_cdm'] = omega_cdm
        state['H0_cur']    = H0_cur
        state['h_cur']     = h_cur
        state['w0_cur']    = w0_cur

        # Derived parameters
        if want_derived:
            # sigma8 — integrate P(k,z=0) with W_th(kR) window
            # All in h/Mpc units: P in (Mpc/h)^3, k in h/Mpc, R=8 Mpc/h
            pk_z0 = pk_all[0]  # z=0 slice, (Mpc/h)^3
            R = 8.0  # Mpc/h
            x = k_grid * R
            W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
            W[x < 1e-3] = 1.0
            integrand = pk_z0 * W**2 * k_grid**2 / (2 * np.pi**2)
            sigma8 = np.sqrt(np.trapz(integrand, k_grid))

            Omega_m = (omega_b + omega_cdm) / h_cur**2

            state['derived'] = {
                'sigma8':  sigma8,
                'Omega_m': Omega_m,
                'H0':      H0_cur,
                'rdrag':   _compute_rdrag(omega_cdm, H0_cur, w0_cur),
            }

    # ── Providers ──────────────────────────────────────────────────────────────

    def get_Pk_interpolator(self, var_pair=('delta_tot', 'delta_tot'),
                             nonlinear=True, extrap_kmax=None):
        """Return P(k,z) interpolator for KiDS likelihood."""
        from scipy.interpolate import RectBivariateSpline

        pk_all = self.current_state['pk_all']   # (n_z, n_k)
        k_grid = self.current_state['k_grid']
        z_grid = self.current_state['z_grid']

        # RectBivariateSpline expects sorted z and log(k)
        log_k  = np.log(k_grid)
        log_pk = np.log(pk_all)

        spline = RectBivariateSpline(z_grid, log_k, log_pk, kx=3, ky=3)

        class _PkInterp:
            def P(self, z, k, grid=False):
                """P(k, z) in (Mpc/h)^3, k in h/Mpc."""
                log_k_in = np.log(np.atleast_1d(k))
                z_in     = np.atleast_1d(z)
                return np.exp(spline(z_in, log_k_in, grid=grid))

        return _PkInterp()

    def get_comoving_radial_distance(self, z):
        from scipy.integrate import quad
        omega_cdm_cur = self.current_state['omega_cdm']
        H0_cur = self.current_state.get('H0_cur', H0)
        h_cur = H0_cur / 100.0
        w0_cur = self.current_state.get('w0_cur', W0)
        z = np.atleast_1d(z)
        chi = np.zeros(len(z))
        for i, zi in enumerate(z):
            if zi <= 0:
                chi[i] = 0.
            else:
                chi[i], _ = quad(lambda zp: C_KMS / H0_cur / np.sqrt(_E2(zp, omega_cdm_cur, h_cur, w0_cur)),
                                 0, zi, limit=100)
        return chi

    def get_angular_diameter_distance(self, z):
        z = np.atleast_1d(z)
        return self.get_comoving_radial_distance(z) / (1 + z)

    def get_Hubble(self, z, units='km/s/Mpc'):
        omega_cdm_cur = self.current_state['omega_cdm']
        H0_cur = self.current_state.get('H0_cur', H0)
        h_cur = H0_cur / 100.0
        w0_cur = self.current_state.get('w0_cur', W0)
        z = np.atleast_1d(z)
        H_kms = H0_cur * np.sqrt(_E2(z, omega_cdm_cur, h_cur, w0_cur))
        if units == "1/Mpc":
            return H_kms / C_KMS
        return H_kms

    def get_sigma8(self):
        return self.current_state['derived']['sigma8']

    def get_rdrag(self):
        return self.current_state['rdrag']
