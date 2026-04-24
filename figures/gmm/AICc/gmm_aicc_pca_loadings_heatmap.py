"""
PCA Loadings Heatmap — Numeric GMM (AICc best: k=5, cov=full)
Output: gmm_aicc_pca_loadings_heatmap.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE       = Path(r"c:\Users\pdaadh\Desktop\Study-2")
FEATS_PATH = BASE / "diagnostics" / "cluster input features" / "derived_features.csv"
CLUST_PATH = BASE / "diagnostics" / "student cluster labels" / "student_clusters.csv"
OUT_PNG    = Path(__file__).parent / "gmm_aicc_pca_loadings_heatmap.png"

FEATURE_COLS = [
    "n_items",
    "accuracy",
    "avg_rt",
    "var_rt",
    "rt_cv",
    "longest_correct_streak",
    "longest_incorrect_streak",
    "consecutive_correct_rate",
]

FEATURE_LABELS = [
    "N Items",
    "Accuracy",
    "Avg RT",
    "Var RT",
    "RT CV",
    "Longest Correct Streak",
    "Longest Incorrect Streak",
    "Consecutive Correct Rate",
]

# ── Load & scale ───────────────────────────────────────────────────────────────
feats    = pd.read_csv(FEATS_PATH)
clusters = pd.read_csv(CLUST_PATH)[["IDCode", "gmm_aicc_best_label"]]
df       = feats.merge(clusters, on="IDCode", how="inner")
df       = df.dropna(subset=FEATURE_COLS)

Xs  = StandardScaler().fit_transform(df[FEATURE_COLS].values)
pca = PCA(n_components=3, random_state=42)
pca.fit(Xs)
var = pca.explained_variance_ratio_

# loadings: shape (n_features, 3)
loadings = pca.components_.T

pc_labels = [
    f"PC1\n({var[0]:.1%})",
    f"PC2\n({var[1]:.1%})",
    f"PC3\n({var[2]:.1%})",
]

print("PCA LOADINGS — AICc best (k=5, cov=full)")
load_df = pd.DataFrame(loadings, index=FEATURE_LABELS, columns=pc_labels)
print(load_df.to_string(float_format=lambda x: f"{x:+.3f}"))
print(f"\nTotal variance explained: {sum(var):.1%}")

# ── Plot (matching narrative embedding heatmap style) ─────────────────────────
# Transpose so PCs are rows and features are columns (horizontal layout)
pc_row_labels = [f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})", f"PC3 ({var[2]:.1%})"]
loadings_df = pd.DataFrame(loadings.T, index=pc_row_labels, columns=FEATURE_LABELS)

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(
    loadings_df,
    annot=True,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Correlation"},
)
ax.set_title("PCA Loadings: Feature Contributions to Axes", fontsize=14, pad=15)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

fig.tight_layout()
fig.savefig(str(OUT_PNG), dpi=300)
print(f"\nSaved: {OUT_PNG}")
plt.close()
