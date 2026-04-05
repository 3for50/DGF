DGF REPRODUCIBILITY PACKAGE — FINAL
====================================
J. Shields, 2026

Deep Geometry Framework (DGF): zero-free-parameter Horndeski nKGB
  G2 = X - V(phi),  V = 5.518*(phi - ln(phi))
  G3 = 9.708 * X,   G4 = 1/2,  G5 = 0
  c = phi_gr = 1.618033988749895 (golden ratio)
  w0 = -0.933, alpha_K = 4.005, alpha_B = 1.018, alpha_M = 0, alpha_T = 0


FOLDER STRUCTURE
================

00_mac_origins/            Original Mac development (Lagrangian to first runs)
  configs/                 19 config files tracing the DGF model evolution:
                           params_DGF.ini → params_DGF_True.ini →
                           params_DGF_Final.ini → params_DGF_Ultimate.ini
                           EFTCAMB_dgf_eftcamb.ini (RPH mapping)
                           EFTCAMB_dgf_params.ini (EFT parameters)
                           Planck 2018 configs, hi_class Makefile
  environment/             Mac system: Python 3.14, homebrew gfortran/gcc,
                           pip packages, conda env, environment variables
  git_state/               EFTCAMB commit bb16b39, MGCAMB commit cb1a03e,
                           local patches, hi_class file list
  SUMMARY.md               Original package summary

  Key file: configs/params_DGF_Final.ini contains the original
  MGCAMB mu(z) spline with 11 nodes mapping the +8.2% G_eff peak
  at z~0.5, plus w0=-0.933, H0=72.1, all DGF fixed parameters.

01_chain_configs/          Every cobaya YAML config used
  dgf/                     DGF chains (hi_class + tabulated_alphas)
  lcdm/                    LCDM comparison chains (CAMB)

02_chain_results/          Converged chain outputs (posteriors, plots, covmats)
  01_kids1000/             KiDS-1000 cosmic shear (5 chains)
  02_desi_bao/             DESI 2024 BAO (2 chains)
  03_planck/               Planck TTTEEE + low-l (1 chain)

03_theory_classes/         Cobaya theory providers
  dgf_cosmopower_theory.py   GPU emulator theory class
  dgf_hiclass_theory.py      Direct hi_class theory class
  kids1000_cobaya.py         KiDS-1000 cosmic shear likelihood

04_alpha_tables/           DGF alpha function tables for hi_class
  dgf_background_alphas_tabfmt.dat   500-point (a, aK, aB, aM, aT) table

05_patches/                All bug fixes
  cobaya_classy_fixes.patch   Cobaya classy wrapper patches
  cobaya_classy_hiclass_compat.patch  Original hi_class compatibility
  008p9_DGF.f90              H-EFTCAMB DGF model scaffold
  known_bugs_and_fixes.txt   13 documented bugs with fixes

06_heftcamb/               H-EFTCAMB DGF Fortran model
  008p9_DGF.f90             Potential functions + EFT mapping stubs

07_emulator/               PCA+NN C_l emulator with alpha_B
  alphaB_v2_model/         Trained model (2500 points, 30 PCA, sklearn NN)
  metadata.json            Training metadata

08_analysis_scripts/       Every analysis script
  J6_flash_field_analytic.py    Flash Field equilibrium test
  h0_channel_diagnostic.py      H0 channel diagnostic (b parameter)
  planck_lowl_comparison.py     Low-l TT: DGF vs LCDM vs Planck
  run_NS5_evidence.py           Bayesian evidence (nautilus)
  run_alphaB_gpu_chain.py       alpha = dH0/d(alpha_B) measurement
  euclid_fisher_forecast.py     Euclid detection forecast
  + 12 more scripts

09_analysis_results/       All diagnostic and analysis outputs
  01_h0_tension/           Flash Field, channel diagnostic
  02_alpha_transfer/       alpha_B measurement (v1 + v2)
  03_gpu_chains/           GPU chains A-G, J1-J5
  04_bayesian_evidence/    NS1-NS5 nested sampling
  05_lowl_cmb/             Low-l multipole analysis
  06_diagnostics/          Model consistency tests
  07_forecasts/            Euclid + pipelines
  08_theory_plots/         ISW, P(k), geometry plots

10_session_outputs/        Quick-access task outputs
  NS4_evidence/            NS4 results
  NS5_evidence/            NS5 final evidence
  task1-6 outputs          All session task results

11_cosmosis/               CosmoSIS TUI application
  app.py                   Main application
  globe.py                 Braille globe widget
  panels.py                Status panels + bug database
  backend.py               Backend automation (5 solvers, 12 datasets)
  cosmosis.tcss            Theme
  run.sh                   Launch script


KEY RESULTS
===========

KiDS-1000 only:        S8 = 0.817 +/- 0.079, chi2/dof = 0.96
KiDS+DESI v2:          S8 = 0.689 +/- 0.038, chi2_BAO = 59
KiDS+DESI w0-free:     w0 = -1.11 +/- 0.03, S8 = 0.593 +/- 0.043
DESI H0-free:          H0 = 67.1 +/- 0.7, chi2_BAO = 14.8
DESI w0+H0-free:       w0 = -1.02 +/- 0.14, H0 = 69.3 +/- 3.8
Planck H0-free:        H0 = 65.7 +/- 0.3

Bayesian evidence:     ln(B) DGF vs LCDM = +80.6 (decisive)
Flash Field:           F_arith = 69.09, 0.50 sigma from 69.47
Flash Field (TDCOSMO): F_arith = 69.51, 0.04 sigma from 69.47
alpha_B transfer:      alpha = dH0/d(alpha_B) = +1.92, r = 0.49
Euclid forecast:       209 sigma detection significance


DEPENDENCIES
============

Python 3.12, cobaya, hi_class v3.2, CAMB, nautilus, numpy, scipy,
matplotlib, sklearn, pyyaml, textual (for CosmoSIS TUI)

Install: pip install cobaya nautilus-sampler textual pyyaml scikit-learn
hi_class: compile from source with tabulated_alphas support
Apply patches from 05_patches/ to cobaya before running DGF chains.


TO REPRODUCE
============

1. Apply patches:
   cd $(python -c "import cobaya; print(cobaya.__path__[0])")
   patch -p1 < /path/to/05_patches/cobaya_classy_fixes.patch

2. Copy alpha table:
   cp 04_alpha_tables/dgf_background_alphas_tabfmt.dat ~/hi_class/

3. Run any chain:
   cobaya-run 01_chain_configs/dgf/dgf_planck_h0free.yaml


