"""
Combine the two 3D-PCA cluster plots side-by-side.

Strategy for larger legend labels
──────────────────────────────────
We do NOT modify the source images (avoids overwriting 3D content).
Instead, for each matplotlib subplot we:
  1. Cover the original tiny Plotly legend with a white Rectangle patch
     (using imshow data coordinates = pixel coordinates).
  2. Sample the cluster dot colours directly from that legend area in the
     original image so the new legend colours are pixel-perfect.
  3. Overlay a fresh matplotlib legend with fontsize=11 (clearly readable).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from PIL import Image

img_numeric_path   = Path(r"C:\Users\pdaadh\Desktop\Study-2\figures\gmm\AICc\numeric clusters.png")
img_narrative_path = Path(r"C:\Users\pdaadh\Desktop\Study-2\narrative_embedding_clustering\outputs\template_A\all_MiniLM_L6_v2\figures\narrative clusters.png")
output_path        = Path(r"C:\Users\pdaadh\Desktop\Study-2\narrative_embedding_clustering\final result\combined_3d_pca_comparison.png")

pil_num = Image.open(img_numeric_path).convert("RGB")
pil_nar = Image.open(img_narrative_path).convert("RGB")
arr_num = np.array(pil_num)
arr_nar = np.array(pil_nar)

W, H = pil_num.size
print(f"Image size: {W} × {H} px")

# ── Exact colours from each pca_3d_paper_figure.py source script ─────────────
# Numeric  → figures/gmm/AICc/pca_3d_paper_figure.py
colors_num = ["#e6194b", "#3cb44b", "#4363d8", "#ff9900", "#9b0dff"]

# Narrative → outputs/template_A/all_MiniLM_L6_v2/pca_3d_paper_figure.py
colors_nar = ["#e6194b", "#3cb44b", "#4363d8", "#ff9900", "#9b0dff",
              "#00c8c8", "#ff69b4", "#8b4513"]

# ── Build legend handles ──────────────────────────────────────────────────────
def make_handles(colors):
    return [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=c, markersize=9,
               label=f'Cluster {i}')
        for i, c in enumerate(colors)
    ]

# ── Build combined figure ─────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor("white")

for ax, arr, title, colors, leg_h in [
    (axs[0], arr_num, "(a) Numeric Baseline (GMM-AICc, K=5)",   colors_num, 190),
    (axs[1], arr_nar, "(b) Narrative Clustering (Strategy C + MiniLM, K=8)", colors_nar, 240),
]:
    ax.imshow(arr)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)

    # Cover original tiny legend with a white rectangle (data coords = pixels)
    ax.add_patch(mpatches.Rectangle(
        (0, 28), 175, leg_h,
        facecolor="white", edgecolor="none",
        transform=ax.transData, zorder=5
    ))

    # Overlay a new, larger legend
    leg = ax.legend(
        handles=make_handles(colors),
        title="Clusters", title_fontsize=11,
        fontsize=10.5, loc="upper left",
        framealpha=0.95, edgecolor="#bbbbbb",
        handletextpad=0.4, borderpad=0.7, labelspacing=0.45,
        bbox_to_anchor=(0.005, 0.99), bbox_transform=ax.transAxes,
    )
    leg.set_zorder(6)

plt.tight_layout(pad=0.5)
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {output_path}")
