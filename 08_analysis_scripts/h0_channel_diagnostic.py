#!/usr/bin/env python3
"""
DGF H0 Channel Diagnostic — Full computation with published measurements.

Channel parameter: b = (H0 - 69.47) / 2.075
  b < 0  → Light channel (CMB/BAO dominated)
  b > 0  → Matter channel (local/lensing dominated)
  b ≈ 0  → Consistent with DGF midpoint
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── DGF constants ──
H0_MID = 69.47    # DGF channel midpoint
DELTA  = 2.075    # channel half-width

def b_channel(H0):
    return (H0 - H0_MID) / DELTA

# ── Results: (name, H0, sigma, category) ──
results = [
    # Programme MCMC chains
    ("KiDS-1000 only",              65.9,  4.2,  "MCMC"),
    ("KiDS+DESI fixed",             67.1,  0.7,  "MCMC"),
    ("DESI H0-free",                67.1,  0.7,  "MCMC"),
    ("DESI w0+H0-free",             69.3,  3.8,  "MCMC"),
    ("KiDS+DESI w0-free",           65.9,  4.2,  "MCMC"),
    ("Matter channel BAO",          68.0,  0.9,  "MCMC"),

    # Programme Boltzmann/theory
    ("DGF geometric prediction",    72.1,  0.0,  "Theory"),
    ("DGF matter channel",          71.55, 2.0,  "Theory"),
    ("DGF light channel",           67.4,  2.0,  "Theory"),

    # Published H0 measurements
    ("SH0ES (Riess+ 2022)",         73.04, 1.04, "Published"),
    ("H0LiCOW (Wong+ 2020)",        73.3,  1.7,  "Published"),
    ("TDCOSMO (Birrer+ 2020)",      74.5,  5.6,  "Published"),
    ("CCHP (Freedman+ 2021)",       69.96, 1.7,  "Published"),
    ("GW170817 (Abbott+ 2017)",     70.0,  12.0, "Published"),
    ("SBF (Blakeslee+ 2021)",       73.3,  2.5,  "Published"),

    # CMB inverse distance ladder
    ("Planck 2018 ΛCDM",            67.36, 0.54, "Published"),
]

print("=" * 72)
print("DGF H0 CHANNEL DIAGNOSTIC (with published measurements)")
print("=" * 72)
print(f"  Channel midpoint: H0 = {H0_MID}")
print(f"  Channel width:    Δ  = {DELTA}")
print(f"  b = (H0 - {H0_MID}) / {DELTA}")
print(f"  b < 0 → Light channel | b > 0 → Matter channel")
print("=" * 72)
print()

# Compute b for each, accumulate weighted mean
weighted_b_sum = 0.0
weight_sum = 0.0
cat_data = {}

print(f"{'Name':<30s} {'H0':>6s} {'σ':>6s} {'b':>7s} {'Channel':<12s} {'Category'}")
print("-" * 80)

for name, H0, sigma, cat in results:
    b = b_channel(H0)
    channel = "MATTER" if b > 0 else "LIGHT" if b < 0 else "MIDPOINT"

    if sigma > 0:
        w = 1.0 / sigma**2
        weighted_b_sum += w * b
        weight_sum += w
        if cat not in cat_data:
            cat_data[cat] = {"wb": 0.0, "w": 0.0, "n": 0}
        cat_data[cat]["wb"] += w * b
        cat_data[cat]["w"] += w
        cat_data[cat]["n"] += 1
        print(f"  {name:<28s} {H0:6.2f} {sigma:6.2f} {b:+7.3f}  {channel:<12s} {cat}")
    else:
        print(f"  {name:<28s} {H0:6.2f}  {'fixed':>5s} {b:+7.3f}  {channel:<12s} {cat} (no weight)")

print("-" * 80)

# Weighted mean
mean_b = weighted_b_sum / weight_sum
sigma_b = 1.0 / np.sqrt(weight_sum)
channel_verdict = "MATTER-BIASED" if mean_b > 0 else "LIGHT-BIASED" if mean_b < 0 else "BALANCED"

print()
print(f"  Weighted mean b = {mean_b:+.4f} ± {sigma_b:.4f}")
print(f"  Sign: {channel_verdict}")
print(f"  Significance: {abs(mean_b)/sigma_b:.1f}σ from zero")
print()

# Category breakdown
print("Category breakdown:")
for cat in ["MCMC", "Theory", "Published"]:
    if cat in cat_data:
        d = cat_data[cat]
        cb = d["wb"] / d["w"]
        cs = 1.0 / np.sqrt(d["w"])
        ch = "matter" if cb > 0 else "light"
        print(f"  {cat:<12s} (n={d['n']:2d}): mean b = {cb:+.4f} ± {cs:.4f}  [{ch}]")

# Published-only weighted mean
if "Published" in cat_data:
    d = cat_data["Published"]
    pub_b = d["wb"] / d["w"]
    pub_s = 1.0 / np.sqrt(d["w"])
    print()
    print(f"  Published measurements alone: b = {pub_b:+.4f} ± {pub_s:.4f}")
    pub_H0 = pub_b * DELTA + H0_MID
    print(f"  → Weighted mean H0 = {pub_H0:.2f} (published)")

# Overall weighted H0
mean_H0 = mean_b * DELTA + H0_MID
print(f"  → Overall weighted H0 = {mean_H0:.2f}")
print()

# ── Plot ──
outdir = os.path.expanduser("~/Desktop/dgf_master_findings/channel_diagnostic/")
os.makedirs(outdir, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 8))

colors = {"MCMC": "#2196F3", "Theory": "#FF9800", "Published": "#4CAF50"}
markers = {"MCMC": "o", "Theory": "D", "Published": "s"}

y_pos = 0
y_labels = []
y_ticks = []

for cat in ["Published", "MCMC", "Theory"]:
    entries = [(n, h, s, b_channel(h)) for n, h, s, c in results if c == cat]
    for name, H0, sigma, b in entries:
        color = colors[cat]
        marker = markers[cat]
        if sigma > 0:
            ax.errorbar(b, y_pos, xerr=sigma/DELTA, fmt=marker, color=color,
                       markersize=8, capsize=4, elinewidth=1.5, label=cat if y_pos == 0 or cat not in [r[3] for r in results[:y_pos]] else "")
        else:
            ax.plot(b, y_pos, marker=marker, color=color, markersize=10, markeredgewidth=2)
        y_labels.append(name)
        y_ticks.append(y_pos)
        y_pos += 1

# Shaded regions
ax.axvspan(-10, 0, alpha=0.08, color='blue', label='Light channel')
ax.axvspan(0, 10, alpha=0.08, color='red', label='Matter channel')
ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(mean_b, color='purple', linestyle='-', linewidth=2, alpha=0.7)
ax.axvspan(mean_b - sigma_b, mean_b + sigma_b, alpha=0.15, color='purple', label=f'Weighted mean: b={mean_b:+.2f}±{sigma_b:.2f}')

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel('Channel parameter b = (H₀ − 69.47) / 2.075', fontsize=12)
ax.set_title('DGF H₀ Channel Diagnostic', fontsize=14, fontweight='bold')
ax.set_xlim(-5, 5)

# Remove duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=9)

ax.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(outdir, "channel_diagnostic.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved: {plot_path}")

# Save summary
summary_path = os.path.join(outdir, "summary.txt")
with open(summary_path, 'w') as f:
    f.write("DGF CHANNEL DIAGNOSTIC (with published H0 measurements)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Weighted mean b = {mean_b:+.4f} ± {sigma_b:.4f}\n")
    f.write(f"Sign: {channel_verdict}\n")
    f.write(f"Significance: {abs(mean_b)/sigma_b:.1f}σ from zero\n\n")
    f.write("Category breakdown:\n")
    for cat in ["MCMC", "Theory", "Published"]:
        if cat in cat_data:
            d = cat_data[cat]
            cb = d["wb"] / d["w"]
            cs = 1.0 / np.sqrt(d["w"])
            f.write(f"  {cat:<12s} (n={d['n']:2d}): mean b = {cb:+.4f} ± {cs:.4f}\n")
    f.write(f"\nTotal results: {len(results)}\n")
    f.write(f"Total weight: {weight_sum:.0f}\n")
    f.write(f"Overall weighted H0: {mean_H0:.2f}\n")
print(f"Saved: {summary_path}")
