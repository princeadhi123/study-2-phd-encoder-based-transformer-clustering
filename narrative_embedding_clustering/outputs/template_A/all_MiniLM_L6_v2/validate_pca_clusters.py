"""
Cluster & PCA Validation Report
================================
Cross-checks for every narrative cluster:
  1. Numeric feature ranges (min / Q25 / median / Q75 / max) of its members.
  2. Keyword claim from cluster_keywords_report.txt vs the empirical feature
     distribution (agreement/contradiction checks).
  3. PCA 3D position (PC1/PC2/PC3 centroid, within-cluster spread, nearest
     neighbouring cluster) using the same sign-enforced PCA as the viewer.
  4. Separation quality: silhouette per cluster.

Output: outputs/template_A/all_MiniLM_L6_v2/cluster_validation_report.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

BASE = Path(r"c:\Users\pdaadh\Desktop\Study-2\narrative_embedding_clustering\outputs\template_A\all_MiniLM_L6_v2")
DERIVED = Path(r"c:\Users\pdaadh\Desktop\Study-2\diagnostics\cluster input features\derived_features.csv")

EMB = np.load(BASE / "embeddings.npy")
idx = pd.read_csv(BASE / "embeddings_index.csv")
clusters = pd.read_csv(BASE / "narrative_clusters.csv")[["IDCode", "narrative_best_label"]]
feats = pd.read_csv(DERIVED)

# ---- PCA with identical sign-enforcement to interactive_pca_viewer.py ----
pca = PCA(n_components=3, random_state=42)
X = pca.fit_transform(EMB)

anchor = idx.merge(clusters, on="IDCode").merge(
    feats[["IDCode", "accuracy", "consecutive_correct_rate", "avg_rt"]], on="IDCode"
)
anchor["pc1"], anchor["pc2"], anchor["pc3"] = X[:, 0], X[:, 1], X[:, 2]

# Sign convention:
#   PC1-high  -> low accuracy            (so flip if corr(PC1, accuracy) > 0)
#   PC2-high  -> low correct-consistency (so flip if corr(PC2, consecutive_correct_rate) > 0)
#   PC3-high  -> slow response time      (so flip if corr(PC3, avg_rt) < 0)
if anchor[["pc1", "accuracy"]].corr().iloc[0, 1] > 0:
    X[:, 0] *= -1
if anchor[["pc2", "consecutive_correct_rate"]].corr().iloc[0, 1] > 0:
    X[:, 1] *= -1
if anchor[["pc3", "avg_rt"]].corr().iloc[0, 1] < 0:
    X[:, 2] *= -1

# Standardise (same as viewer)
X = X / (X.std(axis=0) + 1e-9)

# Build master dataframe: IDCode | cluster | feature cols | PC1..PC3
master = idx.merge(clusters, on="IDCode").merge(feats, on="IDCode")
master["PC1"], master["PC2"], master["PC3"] = X[:, 0], X[:, 1], X[:, 2]

# Silhouette per sample (using PCA 3D space)
y = master["narrative_best_label"].values
sil = silhouette_samples(X, y)
master["silhouette"] = sil

# ---- Keyword claims from report (hard-coded from observed report) ----
CLAIMS = {
    0: "medium accuracy | moderate speed | mixed timing | long correct streak",
    1: "low accuracy | mixed speed | mixed timing | long incorrect streak",
    2: "medium accuracy | slow | highly variable timing | long incorrect streak",
    3: "high accuracy | fast | mixed timing | long correct streak",
    4: "medium-to-high accuracy | slow | highly variable timing | long correct streak",
    5: "low accuracy | moderate speed | mixed timing | long incorrect streak",
    6: "medium accuracy | fast | stable timing | moderate correct streak",
    7: "medium accuracy | moderate speed | moderately variable timing | moderate correct streak",
}

FEATURE_COLS = ["accuracy", "avg_rt", "var_rt", "longest_correct_streak",
                "longest_incorrect_streak", "consecutive_correct_rate"]

# ---- Global thresholds (33/67 quantile on full population) used for interpretation ----
thresh = {}
for col in ["accuracy", "avg_rt", "var_rt",
            "longest_correct_streak", "longest_incorrect_streak"]:
    s = master[col].replace([np.inf, -np.inf], np.nan).dropna()
    thresh[col] = (float(s.quantile(0.33)), float(s.quantile(0.67)))

def bucket(val, lo, hi, labels=("low", "mid", "high")):
    if pd.isna(val):
        return "n/a"
    if val <= lo:
        return labels[0]
    if val >= hi:
        return labels[2]
    return labels[1]

# ---- Generate report ----
lines = []
lines.append("=" * 78)
lines.append("CLUSTER & PCA VALIDATION REPORT")
lines.append(f"Template A | all-MiniLM-L6-v2 | N = {len(master)} students | K = {master['narrative_best_label'].nunique()} clusters")
lines.append("=" * 78)
lines.append("")
lines.append("Global population thresholds (33rd / 67th percentile):")
for k, (lo, hi) in thresh.items():
    lines.append(f"  {k:30s}  low ≤ {lo:8.3f}   high ≥ {hi:8.3f}")
lines.append(f"  accuracy (absolute)              low ≤ 0.600      high ≥ 0.850  (from narrative generator)")
lines.append("")
lines.append(f"PCA variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, "
             f"PC2={pca.explained_variance_ratio_[1]:.1%}, "
             f"PC3={pca.explained_variance_ratio_[2]:.1%}")
lines.append("Sign convention: PC1-high = low accuracy; PC2-high = low correct-consistency; PC3-high = slow RT")
lines.append("")

# Cluster centroids for separation ranking
centroids = master.groupby("narrative_best_label")[["PC1", "PC2", "PC3"]].mean()

for c in sorted(master["narrative_best_label"].unique()):
    sub = master[master["narrative_best_label"] == c]
    n = len(sub)
    lines.append("=" * 78)
    lines.append(f"CLUSTER {c}  (n = {n})")
    lines.append("-" * 78)
    lines.append(f"Keyword report claim: {CLAIMS.get(int(c), '—')}")
    lines.append("")

    lines.append("Feature ranges (min | 25% | median | 75% | max | mean±sd):")
    for col in FEATURE_COLS:
        vals = sub[col].dropna()
        if vals.empty:
            continue
        lines.append(
            f"  {col:28s} "
            f"{vals.min():7.2f} | {vals.quantile(.25):7.2f} | "
            f"{vals.median():7.2f} | {vals.quantile(.75):7.2f} | "
            f"{vals.max():7.2f}  |  {vals.mean():6.2f} ± {vals.std():5.2f}"
        )
    lines.append("")

    # Empirical categorical breakdown vs claim
    acc_med = sub["accuracy"].median()
    rt_med = sub["avg_rt"].median()
    var_med = sub["var_rt"].median()
    lc_med = sub["longest_correct_streak"].median()
    li_med = sub["longest_incorrect_streak"].median()

    acc_b = bucket(acc_med, 0.60, 0.85, ("low", "medium", "high"))
    rt_b = bucket(rt_med, *thresh["avg_rt"], ("fast", "moderate", "slow"))
    var_b = bucket(var_med, *thresh["var_rt"], ("stable", "moderately variable", "highly variable"))
    lc_b = bucket(lc_med, *thresh["longest_correct_streak"], ("short", "moderate", "long"))
    li_b = bucket(li_med, *thresh["longest_incorrect_streak"], ("short", "moderate", "long"))

    lines.append("Empirical profile (cluster medians -> category):")
    lines.append(f"  accuracy        median={acc_med:.3f}  -> {acc_b}")
    lines.append(f"  avg_rt          median={rt_med:.2f}   -> {rt_b}")
    lines.append(f"  var_rt          median={var_med:.2f}   -> {var_b}")
    lines.append(f"  correct_streak  median={lc_med:.0f}      -> {lc_b}")
    lines.append(f"  incorrect_streak median={li_med:.0f}     -> {li_b}")
    lines.append("")

    # PCA positioning
    pc = sub[["PC1", "PC2", "PC3"]]
    cx, cy, cz = pc.mean()
    spread = pc.std().mean()
    # Distance to each other cluster centroid
    dists = []
    for c2, row in centroids.iterrows():
        if c2 == c:
            continue
        d = np.sqrt((row["PC1"] - cx) ** 2 + (row["PC2"] - cy) ** 2 + (row["PC3"] - cz) ** 2)
        dists.append((int(c2), d))
    dists.sort(key=lambda t: t[1])
    nearest = dists[0]
    farthest = dists[-1]

    lines.append(f"PCA 3D centroid: PC1={cx:+.2f}, PC2={cy:+.2f}, PC3={cz:+.2f}")
    lines.append(f"Within-cluster spread (mean axis std): {spread:.2f}")
    lines.append(f"Nearest cluster in PCA space : C{nearest[0]} (d={nearest[1]:.2f})")
    lines.append(f"Farthest cluster in PCA space: C{farthest[0]} (d={farthest[1]:.2f})")
    lines.append(f"Mean silhouette (3D PCA): {sub['silhouette'].mean():+.3f}"
                 f"  [>0.25 well-separated; <0 overlapping]")
    lines.append("")

lines.append("=" * 78)
lines.append("GLOBAL SUMMARY")
lines.append("=" * 78)
lines.append(f"Overall mean silhouette (3D PCA): {master['silhouette'].mean():+.3f}")
lines.append("")
lines.append("PCA sign-convention validation (Pearson r between axis and anchor):")
anchor_r = master[["PC1", "PC2", "PC3",
                   "accuracy", "consecutive_correct_rate", "avg_rt",
                   "var_rt", "longest_correct_streak", "longest_incorrect_streak"]].corr()
lines.append(f"  PC1 vs accuracy                 : {anchor_r.loc['PC1', 'accuracy']:+.3f}  (expect strongly NEGATIVE)")
lines.append(f"  PC2 vs consecutive_correct_rate : {anchor_r.loc['PC2', 'consecutive_correct_rate']:+.3f}  (expect NEGATIVE)")
lines.append(f"  PC3 vs avg_rt                   : {anchor_r.loc['PC3', 'avg_rt']:+.3f}  (expect POSITIVE)")
lines.append("")
lines.append("Supplementary correlations (all features vs PCs):")
for feat in ["accuracy", "consecutive_correct_rate", "longest_correct_streak",
             "longest_incorrect_streak", "avg_rt", "var_rt"]:
    r1 = anchor_r.loc['PC1', feat]
    r2 = anchor_r.loc['PC2', feat]
    r3 = anchor_r.loc['PC3', feat]
    lines.append(f"  {feat:28s}  PC1={r1:+.3f}  PC2={r2:+.3f}  PC3={r3:+.3f}")
lines.append("")

out = BASE / "cluster_validation_report.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Report written to: {out}")
print(f"Clusters: {sorted(master['narrative_best_label'].unique())}")
print(f"Overall silhouette (3D PCA): {master['silhouette'].mean():+.3f}")
