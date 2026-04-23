"""
Interactive LDA Viewer
======================
Generates an interactive HTML file with Plotly for exploring LDA plots.
Features: zoom, pan, hover tooltips, toggle clusters, lasso select, 3D rotation.

LDA (Linear Discriminant Analysis) maximizes separation between narrative clusters.
Unlike PCA (unsupervised), LDA is supervised by cluster labels.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR
EMB_PATH = OUTPUT_DIR / "embeddings.npy"
INDEX_PATH = OUTPUT_DIR / "embeddings_index.csv"
CLUSTERS_PATH = OUTPUT_DIR / "narrative_clusters.csv"

# Load data
print("Loading embeddings...")
embeddings = np.load(EMB_PATH)
index_df = pd.read_csv(INDEX_PATH)
clusters_df = pd.read_csv(CLUSTERS_PATH)

# Merge data
df = index_df.merge(clusters_df[["IDCode", "narrative_best_label"]], on="IDCode")

# Prepare for LDA
y = df["narrative_best_label"].to_numpy()
mask = ~np.isnan(y)
if not mask.all():
    y = y[mask]
    X_for_lda = embeddings[mask]
else:
    X_for_lda = embeddings

n_classes = len(np.unique(y))
print(f"Found {n_classes} narrative clusters")

if n_classes < 4:
    raise SystemExit(f"Need at least 4 clusters for 3D LDA (found {n_classes})")

# Pre-reduce with PCA to avoid degeneracy on high-dim embeddings
n_pre = min(X_for_lda.shape[0] - 1, n_classes * 6, 50)
print(f"Pre-reducing to {n_pre} dims with PCA before LDA...")
pca_pre = PCA(n_components=n_pre, random_state=42)
X_pre = pca_pre.fit_transform(X_for_lda)

# Compute LDA with 3 components for 3D visualization
n_lda_components = min(3, n_classes - 1)
print(f"Computing LDA with {n_lda_components} components...")
lda = LinearDiscriminantAnalysis(n_components=n_lda_components)
X_lda = lda.fit_transform(X_pre, y)

# Standardize each axis to unit variance
X_lda = X_lda / (X_lda.std(axis=0) + 1e-9)

# Add LDA coordinates to dataframe
if not mask.all():
    df_lda = df[mask].copy()
else:
    df_lda = df.copy()

df_lda["LD1"] = X_lda[:, 0]
df_lda["LD2"] = X_lda[:, 1] if n_lda_components >= 2 else 0
df_lda["LD3"] = X_lda[:, 2] if n_lda_components >= 3 else 0

# Explained variance ratio
expl = lda.explained_variance_ratio_
ld2_pct = expl[1] if len(expl) > 1 else 0.0
ld3_pct = expl[2] if len(expl) > 2 else 0.0
print(f"LDA Variance Explained: LD1={expl[0]:.1%}, LD2={ld2_pct:.1%}, LD3={ld3_pct:.1%}")

# Color palette for clusters (distinct colors)
cluster_colors = [
    "#E53935",  # Red
    "#43A047",  # Green  
    "#1976D2",  # Blue
    "#F57C00",  # Orange
    "#8E24AA",  # Purple
    "#00897B",  # Teal
    "#FBC02D",  # Yellow
    "#5E35B1",  # Deep Purple
    "#039BE5",  # Light Blue
    "#795548",  # Brown
]

# Create interactive 2D plot (LD1 vs LD2)
print("Creating interactive 2D LDA plot...")
fig_2d = go.Figure()

clusters = sorted(df_lda["narrative_best_label"].unique())

for i, cluster in enumerate(clusters):
    cluster_data = df_lda[df_lda["narrative_best_label"] == cluster]
    color = cluster_colors[i % len(cluster_colors)]
    
    fig_2d.add_trace(go.Scatter(
        x=cluster_data["LD1"],
        y=cluster_data["LD2"],
        mode="markers",
        name=f"Cluster {int(cluster)}",
        marker=dict(
            size=10,
            color=color,
            opacity=0.7,
            line=dict(width=1, color="white")
        ),
        text=cluster_data["IDCode"],
        hovertemplate="<b>ID:</b> %{text}<br><b>LD1:</b> %{x:.2f}<br><b>LD2:</b> %{y:.2f}<extra></extra>",
        selected=dict(marker=dict(size=12, color="black", opacity=1)),
    ))

fig_2d.update_layout(
    title=dict(
        text=f"Interactive LDA: LD1 vs LD2 (MiniLM Narrative Clusters)",
        font=dict(size=16)
    ),
    xaxis_title=f"LD1 ({expl[0]*100:.1f}% between-cluster var)",
    yaxis_title=f"LD2 ({expl[1]*100:.1f}% between-cluster var)" if len(expl) > 1 else "LD2",
    template="plotly_white",
    width=1000,
    height=700,
    hovermode="closest",
    dragmode="zoom",
    legend=dict(
        title="Narrative Cluster",
        itemsizing="constant",
        traceorder="normal",
        font=dict(size=11)
    ),
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(label="Reset Zoom", method="relayout", args=[{"xaxis.range": None, "yaxis.range": None}]),
            ],
            pad={"r": 10, "t": 10},
            showactive=False,
            x=0.02,
            xanchor="left",
            y=1.02,
            yanchor="top"
        )
    ]
)

# Calculate data ranges for better centering
ld1_range = df_lda["LD1"].max() - df_lda["LD1"].min()
ld2_range = df_lda["LD2"].max() - df_lda["LD2"].min()
ld3_range = df_lda["LD3"].max() - df_lda["LD3"].min()
ld1_center = (df_lda["LD1"].max() + df_lda["LD1"].min()) / 2
ld2_center = (df_lda["LD2"].max() + df_lda["LD2"].min()) / 2
ld3_center = (df_lda["LD3"].max() + df_lda["LD3"].min()) / 2

# Create 3D interactive plot
print("Creating interactive 3D LDA plot...")
fig_3d = go.Figure()

for i, cluster in enumerate(clusters):
    cluster_data = df_lda[df_lda["narrative_best_label"] == cluster]
    color = cluster_colors[i % len(cluster_colors)]
    
    fig_3d.add_trace(go.Scatter3d(
        x=cluster_data["LD1"],
        y=cluster_data["LD2"],
        z=cluster_data["LD3"],
        mode="markers",
        name=f"Cluster {int(cluster)}",
        marker=dict(
            size=7,
            color=color,
            opacity=0.8,
            line=dict(width=0.5, color="white")
        ),
        text=cluster_data["IDCode"],
        hovertemplate="<b>ID:</b> %{text}<br><b>LD1:</b> %{x:.2f}<br><b>LD2:</b> %{y:.2f}<br><b>LD3:</b> %{z:.2f}<extra></extra>",
    ))

# Determine max range for consistent aspect ratio
max_range = max(ld1_range, ld2_range, ld3_range) / 2

fig_3d.update_layout(
    title=dict(
        text=f"Interactive 3D LDA: LD1 vs LD2 vs LD3",
        font=dict(size=16),
        x=0.5,
        xanchor="center"
    ),
    scene=dict(
        xaxis=dict(
            title=f"LD1 ({expl[0]*100:.1f}%)",
            range=[ld1_center - max_range, ld1_center + max_range],
            showbackground=True,
            backgroundcolor="rgb(245, 245, 245)",
            gridcolor="white",
            zerolinecolor="gray",
        ),
        yaxis=dict(
            title=f"LD2 ({expl[1]*100:.1f}%)" if len(expl) > 1 else "LD2",
            range=[ld2_center - max_range, ld2_center + max_range],
            showbackground=True,
            backgroundcolor="rgb(245, 245, 245)",
            gridcolor="white",
            zerolinecolor="gray",
        ),
        zaxis=dict(
            title=f"LD3 ({expl[2]*100:.1f}%)" if len(expl) > 2 else "LD3",
            range=[ld3_center - max_range, ld3_center + max_range],
            showbackground=True,
            backgroundcolor="rgb(245, 245, 245)",
            gridcolor="white",
            zerolinecolor="gray",
        ),
        aspectmode="cube",
        camera=dict(
            eye=dict(x=2.0, y=2.0, z=1.5),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
    ),
    template="plotly_white",
    width=1100,
    height=850,
    margin=dict(l=50, r=50, t=80, b=50),
    hovermode="closest",
    legend=dict(
        title="Narrative Cluster",
        itemsizing="constant",
        font=dict(size=11),
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
)

# Save HTML files
output_2d = OUTPUT_DIR / "interactive_lda_2d.html"
output_3d = OUTPUT_DIR / "interactive_lda_3d.html"

fig_2d.write_html(str(output_2d), include_plotlyjs="cdn", full_html=True)
fig_3d.write_html(str(output_3d), include_plotlyjs="cdn", full_html=True)

print(f"\n✓ Interactive 2D LDA saved to: {output_2d}")
print(f"✓ Interactive 3D LDA saved to: {output_3d}")
print("\nOpen these HTML files in your browser to:")
print("  • Zoom with mouse wheel")
print("  • Pan by dragging")
print("  • Hover over points to see student IDs")
print("  • Click legend items to toggle clusters on/off")
print("  • In 3D view: rotate by dragging, zoom with scroll")
print("\nNote: LDA maximizes cluster separation (supervised), unlike PCA (unsupervised).")
