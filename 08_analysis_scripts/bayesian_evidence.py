#!/home/joe-research/dgf_env/bin/python3
"""
Bayesian evidence comparison: DGF vs LCDM vs wCDM using nautilus nested sampling.

Computes ln(Z) for each model with KiDS-1000 cosmic shear likelihood,
then reports Bayes factors.

DGF:   CosmoPower emulator (w0=-0.933 fixed, 4 free params)
LCDM:  CAMB (w=-1 fixed, 4 free params)
wCDM:  CAMB (w0 free, 5 free params — Occam penalty applies)

Free parameters (DGF & LCDM, identical priors):
  omega_cdm  ~ U(0.01, 0.99)
  ln10A_s    ~ U(2.0, 4.0)
  A_bary     ~ U(0.0, 10.0)
  A_IA       ~ U(-6.0, 6.0)

wCDM adds: w0 ~ U(-2.0, 0.0)

Fixed: H0=72.1, omega_b=0.02238280, n_s=0.9660499, tau_reio=0.054
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import time
import json
import warnings
import numpy as np
from nautilus import Sampler
from cobaya.model import get_model

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = os.path.expanduser(
    '~/Desktop/dgf_master_findings/bayesian_evidence')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prior bounds (same for both models)
PARAM_NAMES = ['omega_cdm', 'ln10A_s', 'A_bary', 'A_IA']
PRIOR_LOW   = np.array([0.01,  2.0,  0.0, -6.0])
PRIOR_HIGH  = np.array([0.99,  4.0, 10.0,  6.0])
N_DIM = len(PARAM_NAMES)

# Fixed cosmological parameters
H0       = 72.1
OMEGA_B  = 0.02238280
N_S      = 0.9660499
TAU_REIO = 0.054

# Nautilus settings
N_LIVE = 500


def prior_transform(u):
    """Map unit hypercube [0,1]^4 to physical parameter space."""
    return PRIOR_LOW + u * (PRIOR_HIGH - PRIOR_LOW)


# ============================================================================
# 1. DGF Model
# ============================================================================

def build_dgf_model():
    """Build cobaya Model for DGF + KiDS-1000."""
    info = {
        'params': {
            # Sampled
            'omega_cdm': {'prior': {'min': 0.01, 'max': 0.99}, 'ref': 0.1678},
            'ln10A_s':   {'prior': {'min': 2.0,  'max': 4.0},  'ref': 2.294},
            'A_bary':    {'prior': {'min': 0.0,  'max': 10.0}, 'ref': 0.59},
            'A_IA':      {'prior': {'min': -6.0, 'max': 6.0},  'ref': 0.0},
            # Fixed
            'omega_b':   OMEGA_B,
            'n_s':       N_S,
            'tau_reio':  TAU_REIO,
            'H0':        H0,
        },
        'theory': {
            'dgf_cosmopower_theory.DGFCosmoPower': {
                'python_path': '/home/joe-research/dgf_training',
            },
        },
        'likelihood': {
            'kids1000_cobaya.KiDS1000': {
                'python_path': '/home/joe-research/dgf_training',
                'data_directory': '/home/joe-research/montepython_gpu/data/KiDS-1000',
            },
        },
    }
    return get_model(info)


# ============================================================================
# 2. LCDM Model (CAMB)
# ============================================================================

def build_lcdm_model():
    """
    Build cobaya Model for LCDM (CAMB) + KiDS-1000.

    CAMB parameter mapping:
      - omch2 = omega_cdm  (identical quantity)
      - As    = exp(ln10A_s) * 1e-10  (derived from sampled ln10A_s)
      - ombh2 = omega_b (fixed)
      - ns    = n_s (fixed)
      - tau   = tau_reio (fixed)
      - H0    = 72.1 (fixed)

    ln10A_s is marked 'drop: True' so it is not passed to CAMB directly;
    instead As is computed from it via a lambda and fed to CAMB.

    A_bary and A_IA are nuisance parameters consumed by the KiDS likelihood
    (not by CAMB).
    """
    info = {
        'params': {
            # Sampled -- omega_cdm mapped to CAMB's omch2
            'omch2':     {'prior': {'min': 0.01, 'max': 0.99}, 'ref': 0.1678},
            # Sampled -- ln10A_s is dropped; As is derived from it for CAMB
            'ln10A_s':   {'prior': {'min': 2.0, 'max': 4.0}, 'ref': 2.294,
                          'drop': True},
            'As':        {'value': 'lambda ln10A_s: 1e-10 * np.exp(ln10A_s)',
                          'latex': 'A_\\mathrm{s}'},
            # Nuisance (passed to likelihood, transparent to CAMB)
            'A_bary':    {'prior': {'min': 0.0,  'max': 10.0}, 'ref': 0.59},
            'A_IA':      {'prior': {'min': -6.0, 'max': 6.0},  'ref': 0.0},
            # Fixed
            'ombh2':     OMEGA_B,
            'ns':        N_S,
            'tau':       TAU_REIO,
            'H0':        H0,
        },
        'theory': {
            'camb': {
                'extra_args': {
                    'num_massive_neutrinos': 1,
                    'nnu': 3.044,
                    'halofit_version': 'mead2020',
                },
            },
        },
        'likelihood': {
            'kids1000_cobaya.KiDS1000': {
                'python_path': '/home/joe-research/dgf_training',
                'data_directory': '/home/joe-research/montepython_gpu/data/KiDS-1000',
                'input_params': ['A_bary', 'A_IA'],
            },
        },
    }
    return get_model(info)


# ============================================================================
# 3. wCDM Model (CAMB, w0 free — 5 params, Occam penalty)
# ============================================================================

# wCDM prior bounds (5D)
WCDM_PARAM_NAMES = ['omega_cdm', 'ln10A_s', 'A_bary', 'A_IA', 'w0']
WCDM_PRIOR_LOW   = np.array([0.01,  2.0,  0.0, -6.0, -2.0])
WCDM_PRIOR_HIGH  = np.array([0.99,  4.0, 10.0,  6.0,  0.0])
N_DIM_WCDM = 5


def wcdm_prior_transform(u):
    return WCDM_PRIOR_LOW + u * (WCDM_PRIOR_HIGH - WCDM_PRIOR_LOW)


def build_wcdm_model():
    """Build cobaya Model for wCDM (CAMB, w0 free) + KiDS-1000."""
    info = {
        'params': {
            'omch2':     {'prior': {'min': 0.01, 'max': 0.99}, 'ref': 0.1678},
            'ln10A_s':   {'prior': {'min': 2.0, 'max': 4.0}, 'ref': 2.294,
                          'drop': True},
            'As':        {'value': 'lambda ln10A_s: 1e-10 * np.exp(ln10A_s)',
                          'latex': 'A_\\mathrm{s}'},
            'w':         {'prior': {'min': -2.0, 'max': 0.0}, 'ref': -1.0},
            'A_bary':    {'prior': {'min': 0.0,  'max': 10.0}, 'ref': 0.59},
            'A_IA':      {'prior': {'min': -6.0, 'max': 6.0},  'ref': 0.0},
            'ombh2':     OMEGA_B,
            'ns':        N_S,
            'tau':       TAU_REIO,
            'H0':        H0,
        },
        'theory': {
            'camb': {
                'extra_args': {
                    'num_massive_neutrinos': 1,
                    'nnu': 3.044,
                    'halofit_version': 'mead2020',
                },
            },
        },
        'likelihood': {
            'kids1000_cobaya.KiDS1000': {
                'python_path': '/home/joe-research/dgf_training',
                'data_directory': '/home/joe-research/montepython_gpu/data/KiDS-1000',
                'input_params': ['A_bary', 'A_IA'],
            },
        },
    }
    return get_model(info)


# ============================================================================
# Nautilus likelihood wrapper
# ============================================================================

def make_loglike_fn(cobaya_model, param_names):
    """
    Return a nautilus-compatible log-likelihood function.

    The function takes a 1-D array of parameter values (in the order given
    by param_names) and returns the total log-likelihood from cobaya.
    """
    def log_likelihood(params):
        point = {name: float(val) for name, val in zip(param_names, params)}
        try:
            logls, derived = cobaya_model.loglikes(point)
            logl = float(logls[0])
        except Exception:
            logl = -1e30
        if not np.isfinite(logl):
            logl = -1e30
        return logl
    return log_likelihood


# ============================================================================
# Evidence runner
# ============================================================================

def run_evidence(label, cobaya_model, param_names, filepath):
    """Run nautilus nested sampling and return ln(Z)."""
    print(f"\n{'=' * 70}")
    print(f"  Running nested sampling: {label}")
    print(f"  n_live = {N_LIVE}, n_dim = {N_DIM}")
    print(f"  Parameters: {param_names}")
    print(f"  Checkpoint: {filepath}")
    print(f"{'=' * 70}\n")

    log_likelihood = make_loglike_fn(cobaya_model, param_names)

    sampler = Sampler(
        prior_transform,
        log_likelihood,
        n_dim=N_DIM,
        n_live=N_LIVE,
        filepath=filepath,
        resume=True,
        seed=42,
    )

    t0 = time.time()
    sampler.run(verbose=True)
    dt = time.time() - t0

    log_z = sampler.log_z
    print(f"\n  {label}: ln(Z) = {log_z:.4f}")
    print(f"  Wall time: {dt / 60:.1f} min")

    return log_z, dt


# ============================================================================
# Jeffreys scale interpretation
# ============================================================================

def interpret_jeffreys(ln_B):
    """Interpret |ln(B)| on the Jeffreys scale."""
    absB = abs(ln_B)
    if absB < 1.0:
        strength = "Not worth more than a bare mention"
    elif absB < 2.5:
        strength = "Substantial"
    elif absB < 5.0:
        strength = "Strong"
    else:
        strength = "Decisive"

    favoured = "DGF" if ln_B > 0 else ("LCDM" if ln_B < 0 else "Neither")
    return f"{strength} evidence in favour of {favoured}"


# ============================================================================
# Main
# ============================================================================

def main():
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    print("\n" + "=" * 70)
    print("  Bayesian Evidence Computation")
    print("  nautilus nested sampling + cobaya model evaluation")
    print("  KiDS-1000 cosmic shear | DGF vs LCDM vs wCDM")
    print("=" * 70)

    # ---- Build models ----
    print("\nBuilding DGF cobaya model...")
    dgf_model = build_dgf_model()
    print("DGF model ready.")

    print("\nBuilding LCDM (CAMB) cobaya model...")
    lcdm_model = build_lcdm_model()
    print("LCDM model ready.")

    print("\nBuilding wCDM (CAMB, w0 free) cobaya model...")
    wcdm_model = build_wcdm_model()
    print("wCDM model ready.")

    # ---- DGF evidence (4 free params) ----
    dgf_param_names = ['omega_cdm', 'ln10A_s', 'A_bary', 'A_IA']
    fp_dgf = os.path.join(OUTPUT_DIR, 'nautilus_dgf.hdf5')
    log_z_dgf, dt_dgf = run_evidence("DGF", dgf_model, dgf_param_names, fp_dgf)

    # ---- LCDM evidence (4 free params) ----
    lcdm_param_names = ['omch2', 'ln10A_s', 'A_bary', 'A_IA']
    fp_lcdm = os.path.join(OUTPUT_DIR, 'nautilus_lcdm.hdf5')
    log_z_lcdm, dt_lcdm = run_evidence("LCDM", lcdm_model, lcdm_param_names,
                                        fp_lcdm)

    # ---- wCDM evidence (5 free params — Occam penalty) ----
    wcdm_param_names = ['omch2', 'ln10A_s', 'A_bary', 'A_IA', 'w']
    fp_wcdm = os.path.join(OUTPUT_DIR, 'nautilus_wcdm.hdf5')

    # wCDM needs its own prior_transform (5D)
    def wcdm_loglike(params):
        point = {name: float(val) for name, val in zip(wcdm_param_names, params)}
        try:
            logls, derived = wcdm_model.loglikes(point)
            logl = float(logls[0])
        except Exception:
            logl = -1e30
        if not np.isfinite(logl):
            logl = -1e30
        return logl

    print(f"\n{'=' * 70}")
    print(f"  Running nested sampling: wCDM (w0 free, 5 params)")
    print(f"  n_live = {N_LIVE}, n_dim = {N_DIM_WCDM}")
    print(f"{'=' * 70}\n")

    sampler_wcdm = Sampler(
        wcdm_prior_transform, wcdm_loglike,
        n_dim=N_DIM_WCDM, n_live=N_LIVE,
        filepath=fp_wcdm, resume=True, seed=42,
    )
    t0 = time.time()
    sampler_wcdm.run(verbose=True)
    dt_wcdm = time.time() - t0
    log_z_wcdm = sampler_wcdm.log_z
    print(f"\n  wCDM: ln(Z) = {log_z_wcdm:.4f}")

    # ---- Bayes factors ----
    ln_B_dgf_lcdm = log_z_dgf - log_z_lcdm
    ln_B_dgf_wcdm = log_z_dgf - log_z_wcdm
    ln_B_lcdm_wcdm = log_z_lcdm - log_z_wcdm

    # ---- Print results ----
    print("\n" + "=" * 70)
    print("  BAYESIAN EVIDENCE RESULTS")
    print("=" * 70)
    print(f"  ln(Z_DGF)  = {log_z_dgf:.4f}   (4 free params)")
    print(f"  ln(Z_LCDM) = {log_z_lcdm:.4f}   (4 free params)")
    print(f"  ln(Z_wCDM) = {log_z_wcdm:.4f}   (5 free params)")
    print()
    print(f"  DGF vs LCDM:  ln(B) = {ln_B_dgf_lcdm:+.4f}  {interpret_jeffreys(ln_B_dgf_lcdm)}")
    print(f"  DGF vs wCDM:  ln(B) = {ln_B_dgf_wcdm:+.4f}  {interpret_jeffreys(ln_B_dgf_wcdm)}")
    print(f"  LCDM vs wCDM: ln(B) = {ln_B_lcdm_wcdm:+.4f}  {interpret_jeffreys(ln_B_lcdm_wcdm)}")
    print()
    print("  Jeffreys scale: |ln B| < 1 inconclusive, 1-2.5 substantial,")
    print("                  2.5-5 strong, >5 decisive")
    print("=" * 70)

    # ---- Save results ----
    results = {
        'log_z_dgf':        float(log_z_dgf),
        'log_z_lcdm':       float(log_z_lcdm),
        'log_z_wcdm':       float(log_z_wcdm),
        'ln_B_dgf_lcdm':    float(ln_B_dgf_lcdm),
        'ln_B_dgf_wcdm':    float(ln_B_dgf_wcdm),
        'ln_B_lcdm_wcdm':   float(ln_B_lcdm_wcdm),
        'interp_dgf_lcdm':  interpret_jeffreys(ln_B_dgf_lcdm),
        'interp_dgf_wcdm':  interpret_jeffreys(ln_B_dgf_wcdm),
        'n_live':           N_LIVE,
        'n_params_dgf':     4,
        'n_params_lcdm':    4,
        'n_params_wcdm':    5,
        'wall_time_dgf_min':  round(dt_dgf / 60, 1),
        'wall_time_lcdm_min': round(dt_lcdm / 60, 1),
        'wall_time_wcdm_min': round(dt_wcdm / 60, 1),
    }
    results_path = os.path.join(OUTPUT_DIR, 'bayesian_evidence_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
