# Deep Geometry Framework

Zero-free-parameter Horndeski cosmology (nKGB subclass) with all parameters fixed by the golden ratio.

```
G₂ = X − V(φ),  V = 5.518·(φ − ln φ)
G₃ = 9.708·X,   G₄ = ½,  G₅ = 0
c = φ_gr = 1.618033988749895
w₀ = −0.933,  α_K = 4.005,  α_B = 1.018,  α_M = 0,  α_T = 0
```

## Results

| Dataset | Key constraint | Value |
|---------|---------------|-------|
| KiDS-1000 only | S₈ | 0.817 ± 0.079 |
| KiDS+DESI v2 | S₈ | 0.689 ± 0.038 |
| KiDS+DESI w₀-free | w₀ | −1.11 ± 0.03 |
| DESI H₀-free | H₀ | 67.1 ± 0.7 |
| Planck H₀-free | H₀ | 65.7 ± 0.3 |
| Bayesian evidence | ln(B) DGF vs ΛCDM | +80.6 (decisive) |
| Flash Field | F_arith | 69.09 (0.50σ from 69.47) |
| Flash Field + TDCOSMO | F_arith | 69.51 (0.04σ from 69.47) |
| α_B transfer | α = dH₀/dα_B | +1.92, r = 0.49 |
| Euclid forecast | Detection significance | 209σ |

## What's here

```
00_mac_origins/          Original development environment + configs
01_chain_configs/        Cobaya YAML configs (DGF + ΛCDM comparison)
02_chain_results/        Converged MCMC chains, posteriors, plots
03_theory_classes/       Cobaya theory providers (emulator + hi_class)
04_alpha_tables/         Tabulated α functions for hi_class
05_patches/              Bug fixes for cobaya/hi_class compatibility
06_heftcamb/             H-EFTCAMB DGF Fortran model
07_emulator/             PCA+NN Cₗ emulator (α_B, 2500 training points)
08_analysis_scripts/     All analysis scripts (evidence, forecasts, diagnostics)
09_analysis_results/     Full outputs: plots, chains, nested sampling
10_session_outputs/      Quick-access task outputs (NS4/NS5 evidence)
```

## Quick start

Install dependencies:

```bash
pip install cobaya nautilus-sampler numpy scipy matplotlib scikit-learn pyyaml
```

[hi_class](https://github.com/miguelzuma/hi_class_public) v3.2 needs to be compiled from source with `tabulated_alphas` support.

Apply patches to cobaya:

```bash
cd $(python -c "import cobaya; print(cobaya.__path__[0])")
patch -p1 < /path/to/05_patches/cobaya_classy_fixes.patch
```

Copy the alpha table:

```bash
cp 04_alpha_tables/dgf_background_alphas_tabfmt.dat ~/hi_class/
```

Run any chain:

```bash
cobaya-run 01_chain_configs/dgf/dgf_planck_h0free.yaml
```

## Chain results

**KiDS-1000** (5 chains): emulator-only, DESI-fixed, DESI v2, w₀-free, hi_class exact

**DESI BAO** (2 chains): H₀-free, w₀+H₀-free

**Planck** (1 chain): TTTEEE + low-l, H₀-free

Each folder in `02_chain_results/` contains `chain.txt`, `posteriors.txt`, `config.yaml`, and posterior plots.

## Bayesian evidence

Nested sampling (nautilus) comparing DGF vs ΛCDM across KiDS-1000 + DESI BAO:

**ln(B) = +80.6** — decisive preference for DGF.

Results in `09_analysis_results/04_bayesian_evidence/`.

## Dependencies

Python 3.12, cobaya, hi_class v3.2, CAMB, nautilus, numpy, scipy, matplotlib, scikit-learn, pyyaml

## License

MIT
