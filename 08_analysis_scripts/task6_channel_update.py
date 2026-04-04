#!/usr/bin/env python3
"""
Task 6: Channel diagnostic update with additional published light-channel measurements.
"""
import numpy as np
import os

H0_MID = 69.47
DELTA = 2.075

results = [
    # Programme MCMC chains
    ("KiDS-1000 only",              65.9,  4.2,  "MCMC"),
    ("KiDS+DESI fixed",             67.1,  0.7,  "MCMC"),
    ("DESI H0-free",                67.1,  0.7,  "MCMC"),
    ("DESI w0+H0-free",             69.3,  3.8,  "MCMC"),
    ("KiDS+DESI w0-free",           65.9,  4.2,  "MCMC"),
    ("Matter channel BAO",          68.0,  0.9,  "MCMC"),

    # Programme theory
    ("DGF geometric",               72.1,  0.0,  "Theory"),
    ("DGF matter channel",          71.55, 2.0,  "Theory"),
    ("DGF light channel",           67.4,  2.0,  "Theory"),

    # Published matter-channel
    ("SH0ES (Riess+ 2022)",         73.04, 1.04, "Pub-Matter"),
    ("H0LiCOW (Wong+ 2020)",        73.3,  1.7,  "Pub-Matter"),
    ("TDCOSMO (Birrer+ 2020)",      74.5,  5.6,  "Pub-Matter"),
    ("CCHP (Freedman+ 2021)",       69.96, 1.7,  "Pub-Matter"),
    ("GW170817 (Abbott+ 2017)",     70.0,  12.0, "Pub-Matter"),
    ("SBF (Blakeslee+ 2021)",       73.3,  2.5,  "Pub-Matter"),

    # Published light-channel (NEW)
    ("Planck 2018 ΛCDM",            67.36, 0.54, "Pub-Light"),
    ("ACT DR4 (Aiola+ 2020)",       68.3,  1.5,  "Pub-Light"),
    ("BOSS BAO (Alam+ 2021)",       67.6,  0.6,  "Pub-Light"),
    ("eBOSS (Alam+ 2021)",          67.4,  1.1,  "Pub-Light"),
    ("SPT-3G (Balkenhol+ 2023)",    67.9,  1.5,  "Pub-Light"),
]

print("=" * 85)
print("DGF CHANNEL DIAGNOSTIC (updated with light-channel published measurements)")
print("=" * 85)
print(f"  b = (H0 - {H0_MID}) / {DELTA}")
print()
print(f"{'Name':<30s} {'H0':>6s} {'σ':>5s} {'b':>7s} {'Channel':<8s} {'Category'}")
print("-" * 85)

wb_sum = 0.0
w_sum = 0.0
cat_data = {}

for name, H0, sigma, cat in results:
    b = (H0 - H0_MID) / DELTA
    ch = "MATTER" if b > 0 else "LIGHT"

    if sigma > 0:
        w = 1.0 / sigma**2
        wb_sum += w * b
        w_sum += w
        if cat not in cat_data:
            cat_data[cat] = {"wb": 0.0, "w": 0.0, "n": 0}
        cat_data[cat]["wb"] += w * b
        cat_data[cat]["w"] += w
        cat_data[cat]["n"] += 1
        print(f"  {name:<28s} {H0:6.2f} {sigma:5.2f} {b:+7.3f} {ch:<8s} {cat}")
    else:
        print(f"  {name:<28s} {H0:6.2f} {'fix':>5s} {b:+7.3f} {ch:<8s} {cat}")

print("-" * 85)

mean_b = wb_sum / w_sum
sigma_b = 1.0 / np.sqrt(w_sum)

print(f"\n  Global weighted b = {mean_b:+.4f} ± {sigma_b:.4f}")
print(f"  Significance: {abs(mean_b)/sigma_b:.1f}σ from zero")
print(f"  Overall weighted H0 = {mean_b * DELTA + H0_MID:.2f}")
print()

# Category breakdown
print("Category breakdown:")
for cat in ["MCMC", "Theory", "Pub-Matter", "Pub-Light"]:
    if cat in cat_data:
        d = cat_data[cat]
        cb = d["wb"] / d["w"]
        cs = 1.0 / np.sqrt(d["w"])
        ch = "matter" if cb > 0 else "light"
        print(f"  {cat:<12s} (n={d['n']:2d}): b = {cb:+.4f} ± {cs:.4f}  [{ch}]")

# Published-only (matter + light combined)
pub_m = cat_data.get("Pub-Matter", {"wb": 0, "w": 0, "n": 0})
pub_l = cat_data.get("Pub-Light", {"wb": 0, "w": 0, "n": 0})
pub_wb = pub_m["wb"] + pub_l["wb"]
pub_w = pub_m["w"] + pub_l["w"]
pub_b = pub_wb / pub_w
pub_s = 1.0 / np.sqrt(pub_w)
print(f"\n  All published (n={pub_m['n']+pub_l['n']}): b = {pub_b:+.4f} ± {pub_s:.4f}")
print(f"  Published weighted H0 = {pub_b * DELTA + H0_MID:.2f}")

# Compare to previous (without the 4 new light-channel entries)
old_wb = pub_m["wb"] + cat_data["Pub-Light"]["wb"] - sum(
    (H0 - H0_MID)/DELTA / sigma**2 for _, H0, sigma, cat in results
    if cat == "Pub-Light" and "Planck" not in _
) if "Pub-Light" in cat_data else pub_m["wb"]

# Actually just recompute without the new 4
prev_pub_wb = pub_m["wb"]
prev_pub_w = pub_m["w"]
# Add only Planck from light
for name, H0, sigma, cat in results:
    if cat == "Pub-Light" and "Planck" in name:
        prev_pub_wb += (H0 - H0_MID)/DELTA / sigma**2
        prev_pub_w += 1.0/sigma**2
prev_pub_b = prev_pub_wb / prev_pub_w
print(f"\n  Previous published (without ACT/BOSS/eBOSS/SPT): b = {prev_pub_b:+.4f}")
print(f"  New published (with ACT/BOSS/eBOSS/SPT):         b = {pub_b:+.4f}")
shift = pub_b - prev_pub_b
print(f"  Shift: {shift:+.4f} ({'more light' if shift < 0 else 'more matter'})")

outdir = os.path.expanduser("~/Desktop/dgf_master_findings/channel_diagnostic/")
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "summary_v2.txt"), 'w') as f:
    f.write("DGF CHANNEL DIAGNOSTIC v2 (with light-channel published)\n")
    f.write(f"Global weighted b = {mean_b:+.4f} ± {sigma_b:.4f}\n")
    f.write(f"Significance: {abs(mean_b)/sigma_b:.1f}σ\n")
    f.write(f"Overall H0 = {mean_b * DELTA + H0_MID:.2f}\n")
print(f"\nSaved: {outdir}summary_v2.txt")
