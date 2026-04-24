"""
PCA Loadings Heatmap — Numeric GMM (AICc best: k=5, cov=full)
Output: gmm_aicc_pca_loadings_heatmap.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# ── Plot ───────────────────────────────────────────────────────────────────────
# Adjust figure size for better layout and spacing
fig_width = max(5, len(pc_labels) * 1.2 + 1)
fig_height = max(5, len(FEATURE_LABELS) * 0.6 + 2)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

cmap = 'coolwarm'
norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

im = ax.imshow(loadings, cmap=cmap, norm=norm, aspect="auto")

# ── Cell annotations ──────────────────────────────────────────────────────────
for i in range(len(FEATURE_LABELS)):
    for j in range(3):
        val = loadings[i, j]
        txt_color = "white" if abs(val) > 0.45 else "#222222"
        ax.text(j, i, f"{val:+.3f}",
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=txt_color)

# ── Axes formatting ────────────────────────────────────────────────────────────
ax.set_xticks(range(3))
ax.set_xticklabels(pc_labels, fontsize=10)
ax.set_yticks(range(len(FEATURE_LABELS)))
ax.set_yticklabels(FEATURE_LABELS, fontsize=10)
ax.tick_params(length=0)

# grid lines between cells
for x in np.arange(-0.5, 3, 1):
    ax.axvline(x, color="white", linewidth=1.5)
for y in np.arange(-0.5, len(FEATURE_LABELS), 1):
    ax.axhline(y, color="white", linewidth=1.5)

ax.set_title(
    "PCA Loadings: Feature Contributions to Axes",
    fontsize=12, pad=14
)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Loading", fontsize=10)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
fig.savefig(str(OUT_PNG), dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")
plt.close()
