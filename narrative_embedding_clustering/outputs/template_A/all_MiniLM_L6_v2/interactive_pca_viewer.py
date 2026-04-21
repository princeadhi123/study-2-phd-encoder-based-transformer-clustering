"""
Interactive PCA Viewer
======================
Generates an interactive HTML file with Plotly for exploring PCA plots.
Features: zoom, pan, hover tooltips, toggle clusters, lasso select.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "template_A" / "all_MiniLM_L6_v2"
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

# Perform PCA
print("Computing PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(embeddings)
pca = PCA(n_components=10)  # Compute first 10 PCs
X_pca = pca.fit_transform(X_scaled)

# Add PCA coordinates to dataframe
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
df["PC3"] = X_pca[:, 2]
df["PC4"] = X_pca[:, 3]
df["PC5"] = X_pca[:, 4]

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
]

# Create interactive 2D plot (PC1 vs PC2)
print("Creating interactive 2D plot...")
fig_2d = go.Figure()

clusters = sorted(df["narrative_best_label"].unique())

for i, cluster in enumerate(clusters):
    cluster_data = df[df["narrative_best_label"] == cluster]
    color = cluster_colors[i % len(cluster_colors)]
    
    fig_2d.add_trace(go.Scatter(
        x=cluster_data["PC1"],
        y=cluster_data["PC2"],
        mode="markers",
        name=f"Cluster {cluster}",
        marker=dict(
            size=10,
            color=color,
            opacity=0.7,
            line=dict(width=1, color="white")
        ),
        text=cluster_data["IDCode"],
        hovertemplate="<b>ID:</b> %{text}<br><b>PC1:</b> %{x:.2f}<br><b>PC2:</b> %{y:.2f}<extra></extra>",
        selected=dict(marker=dict(size=12, color="black", opacity=1)),
    ))

fig_2d.update_layout(
    title=dict(
        text="Interactive PCA: PC1 vs PC2 (MiniLM Narrative Clusters)",
        font=dict(size=16)
    ),
    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
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
pc1_range = df["PC1"].max() - df["PC1"].min()
pc2_range = df["PC2"].max() - df["PC2"].min()
pc3_range = df["PC3"].max() - df["PC3"].min()
pc1_center = (df["PC1"].max() + df["PC1"].min()) / 2
pc2_center = (df["PC2"].max() + df["PC2"].min()) / 2
pc3_center = (df["PC3"].max() + df["PC3"].min()) / 2

# Create 3D interactive plot
print("Creating interactive 3D plot...")
fig_3d = go.Figure()

for i, cluster in enumerate(clusters):
    cluster_data = df[df["narrative_best_label"] == cluster]
    color = cluster_colors[i % len(cluster_colors)]
    
    fig_3d.add_trace(go.Scatter3d(
        x=cluster_data["PC1"],
        y=cluster_data["PC2"],
        z=cluster_data["PC3"],
        mode="markers",
        name=f"Cluster {cluster}",
        marker=dict(
            size=7,
            color=color,
            opacity=0.8,
            line=dict(width=0.5, color="white")
        ),
        text=cluster_data["IDCode"],
        hovertemplate="<b>ID:</b> %{text}<br><b>PC1:</b> %{x:.2f}<br><b>PC2:</b> %{y:.2f}<br><b>PC3:</b> %{z:.2f}<extra></extra>",
    ))

# Determine max range for consistent aspect ratio
max_range = max(pc1_range, pc2_range, pc3_range) / 2

fig_3d.update_layout(
    title=dict(
        text="Interactive 3D PCA: PC1 vs PC2 vs PC3",
        font=dict(size=16),
        x=0.5,
        xanchor="center"
    ),
    scene=dict(
        xaxis=dict(
            title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            range=[pc1_center - max_range, pc1_center + max_range],
            showbackground=True,
            backgroundcolor="rgb(245, 245, 245)",
            gridcolor="white",
            zerolinecolor="gray",
        ),
        yaxis=dict(
            title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            range=[pc2_center - max_range, pc2_center + max_range],
            showbackground=True,
            backgroundcolor="rgb(245, 245, 245)",
            gridcolor="white",
            zerolinecolor="gray",
        ),
        zaxis=dict(
            title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)",
            range=[pc3_center - max_range, pc3_center + max_range],
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
output_2d = OUTPUT_DIR / "interactive_pca_2d.html"
output_3d = OUTPUT_DIR / "interactive_pca_3d.html"

fig_2d.write_html(str(output_2d), include_plotlyjs="cdn", full_html=True)
fig_3d.write_html(str(output_3d), include_plotlyjs="cdn", full_html=True)

print(f"\n✓ Interactive 2D PCA saved to: {output_2d}")
print(f"✓ Interactive 3D PCA saved to: {output_3d}")
print("\nOpen these HTML files in your browser to:")
print("  • Zoom with mouse wheel")
print("  • Pan by dragging")
print("  • Hover over points to see student IDs")
print("  • Click legend items to toggle clusters on/off")
print("  • In 3D view: rotate by dragging, zoom with scroll")
