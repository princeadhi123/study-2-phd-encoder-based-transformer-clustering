"""
Publication-Quality 3D PCA Figure — Numeric GMM (AICc best)
===========================================================
Generates a clean, high-resolution 3D PCA plot suitable for research papers.
- Clean white background
- Distinct but muted color palette (colorblind-friendly)
- Smaller, less opaque markers to reduce crowding
- Clear axis labels with interpretation
- Legend outside plot area
- Multiple camera angles saved as separate PNGs
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

BASE = Path(r"c:\Users\pdaadh\Desktop\Study-2")
DERIVED = BASE / "diagnostics" / "cluster input features" / "derived_features.csv"
CLUSTERS = BASE / "diagnostics" / "student cluster labels" / "student_clusters.csv"
OUTPUT_DIR = BASE / "figures" / "gmm" / "AICc"

# Feature columns for PCA
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

# Load data
feats = pd.read_csv(DERIVED)
clusters = pd.read_csv(CLUSTERS)[["IDCode", "gmm_aicc_best_label"]]

# Merge
master = feats.merge(clusters, on="IDCode", how="inner")
master = master.dropna(subset=FEATURE_COLS)

# Standardize and PCA
Xs = StandardScaler().fit_transform(master[FEATURE_COLS].values)
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(Xs)

# Sign enforcement (align with behavioral anchors)
anchor = master[["accuracy", "consecutive_correct_rate", "avg_rt"]].copy()
anchor["PC1"], anchor["PC2"], anchor["PC3"] = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]

if anchor[["PC1", "accuracy"]].corr().iloc[0, 1] > 0:
    X_pca[:, 0] *= -1
if anchor[["PC2", "consecutive_correct_rate"]].corr().iloc[0, 1] > 0:
    X_pca[:, 1] *= -1
if anchor[["PC3", "avg_rt"]].corr().iloc[0, 1] < 0:
    X_pca[:, 2] *= -1

X_pca = X_pca / (X_pca.std(axis=0) + 1e-9)
master["PC1"], master["PC2"], master["PC3"] = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]

# Colorblind-friendly palette (Okabe-Ito)
colors = {
    0: "#e6194b",   # vivid red
    1: "#3cb44b",   # vivid green
    2: "#4363d8",   # vivid blue
    3: "#ff9900",   # vivid amber
    4: "#9b0dff",   # vivid violet
}

cluster_names = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
    4: "Cluster 4",
}

# Calculate overall silhouette
sil_score = silhouette_score(X_pca, master["gmm_aicc_best_label"].values)

# Centroids
centroids = master.groupby("gmm_aicc_best_label")[["PC1", "PC2", "PC3"]].mean()

def _convex_hull_trace(pts, color, opacity=0.35):
    """Return a Mesh3d trace for the convex hull of pts (N×3 array)."""
    hull = ConvexHull(pts)
    verts = pts[hull.vertices]
    # Re-index simplices to hull.vertices
    idx_map = {v: i for i, v in enumerate(hull.vertices)}
    simplices = np.array([[idx_map[s] for s in simplex] for simplex in hull.simplices])
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=simplices[:, 0], j=simplices[:, 1], k=simplices[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting=dict(ambient=0.8, diffuse=0.5),
        hoverinfo="skip",
        showlegend=False
    )


def create_figure(camera_eye, title_suffix=""):
    """Create a 3D figure with specified camera angle."""
    fig = go.Figure()
    
    for cluster_id in sorted(master["gmm_aicc_best_label"].unique()):
        sub = master[master["gmm_aicc_best_label"] == cluster_id]
        
        fig.add_trace(go.Scatter3d(
            x=sub["PC1"],
            y=sub["PC2"],
            z=sub["PC3"],
            mode="markers",
            name=cluster_names.get(int(cluster_id), f"Cluster {int(cluster_id)}"),
            marker=dict(
                size=3,
                color=colors.get(int(cluster_id), "#333333"),
                opacity=0.6,
                line=dict(width=0.3, color="white")
            ),
            showlegend=True
        ))
    
    # ---- Convex hull envelopes — one per cluster ----
    for cluster_id in sorted(master["gmm_aicc_best_label"].unique()):
        sub = master[master["gmm_aicc_best_label"] == cluster_id]
        pts = sub[["PC1", "PC2", "PC3"]].values
        if len(pts) < 4:
            continue
        col = colors.get(int(cluster_id), "#888888")
        try:
            fig.add_trace(_convex_hull_trace(pts, col, opacity=0.30))
        except Exception:
            pass  # skip degenerate clusters

    # Cluster labels — large bold black, always on top
    label_z_offset = 0.25
    for cluster_id, row in centroids.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row["PC1"]],
            y=[row["PC2"]],
            z=[row["PC3"] + label_z_offset],
            mode="markers+text",
            marker=dict(size=6, color="black", symbol="circle"),
            text=[f"  C{int(cluster_id)}"],
            textposition="middle right",
            textfont=dict(size=20, color="black", family="Arial Black"),
            hoverinfo="skip",
            showlegend=False
        ))
    
    # Axis titles with interpretation
    fig.update_layout(
        title=dict(
            text=f"3D PCA — Numeric GMM (AICc best: k=5, cov=full) [N={len(master)}, silhouette={sil_score:.3f}]{title_suffix}",
            font=dict(size=14, family="Arial"),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", font=dict(size=10)),
                showgrid=True,
                gridwidth=0.5,
                gridcolor="lightgray",
                showbackground=True,
                backgroundcolor="rgba(240,240,255,0.5)"
            ),
            yaxis=dict(
                title=dict(text=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", font=dict(size=10)),
                showgrid=True,
                gridwidth=0.5,
                gridcolor="lightgray",
                showbackground=True,
                backgroundcolor="rgba(240,255,240,0.5)"
            ),
            zaxis=dict(
                title=dict(text=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)", font=dict(size=10)),
                showgrid=True,
                gridwidth=0.5,
                gridcolor="lightgray",
                showbackground=True,
                backgroundcolor="rgba(255,245,235,0.5)"
            ),
            camera=dict(
                eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode="data"
        ),
        legend=dict(
            title=dict(text="Clusters", font=dict(size=10)),
            font=dict(size=9),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=0.5
        ),
        width=950,
        height=750,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        template="simple_white"
    )
    
    return fig

print("Generating publication-quality figure...")
print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.1%}, "
      f"PC3={pca.explained_variance_ratio_[2]:.1%}")
print(f"Overall silhouette score: {sil_score:.3f}")
print()

# ---- Find best camera angle: maximise min pairwise centroid separation ----
def _best_camera(centroids_df, n_phi=18, n_theta=24):
    """Grid-search over spherical angles; pick view maximising min pairwise
    projected distance between all cluster centroids."""
    pts = centroids_df[["PC1", "PC2", "PC3"]].values
    best_score, best_eye = -1, (1.5, 1.5, 1.2)
    for phi in np.linspace(10, 80, n_phi):          # elevation (degrees)
        for theta in np.linspace(0, 360, n_theta, endpoint=False):  # azimuth
            phi_r = np.radians(phi)
            theta_r = np.radians(theta)
            # Camera direction unit vector
            ex = np.cos(phi_r) * np.cos(theta_r)
            ey = np.cos(phi_r) * np.sin(theta_r)
            ez = np.sin(phi_r)
            # Right vector (perpendicular in xy-plane)
            rx = -np.sin(theta_r)
            ry =  np.cos(theta_r)
            rz = 0.0
            # Up vector (cross product)
            ux = ry * ez - rz * ey
            uy = rz * ex - rx * ez
            uz = rx * ey - ry * ex
            # Project centroids onto (right, up) screen plane
            proj = pts @ np.array([[rx, ux], [ry, uy], [rz, uz]])
            # Min pairwise distance in 2D projection
            dists = []
            for i in range(len(proj)):
                for j in range(i + 1, len(proj)):
                    dists.append(np.linalg.norm(proj[i] - proj[j]))
            score = min(dists)
            if score > best_score:
                best_score = score
                best_eye = (ex * 2.2, ey * 2.2, ez * 2.2)
    print(f"Best camera eye: {best_eye[0]:.2f}, {best_eye[1]:.2f}, {best_eye[2]:.2f}  "
          f"(min separation={best_score:.3f})")
    return best_eye

best_eye = _best_camera(centroids)

# Generate interactive HTML only (instant — no headless browser needed)
# Open in browser, rotate to your preferred angle, then screenshot for paper
fig = create_figure(best_eye)
html_path = OUTPUT_DIR / "pca_3d_paper_interactive.html"
fig.write_html(str(html_path), include_plotlyjs="cdn")
print(f"Saved: {html_path}")
print("Open the HTML in your browser, rotate to the best angle, then use")
print("the camera icon (top-right toolbar) or a screenshot to save for paper.")

# Print cluster stats
print("\n" + "="*60)
print("CLUSTER CENTROIDS IN 3D PCA SPACE")
print("="*60)
print(f"{'Cluster':>8} {'PC1':>10} {'PC2':>10} {'PC3':>10} {'n':>6}")
print("-"*60)
for cluster_id in sorted(master["gmm_aicc_best_label"].unique()):
    sub = master[master["gmm_aicc_best_label"] == cluster_id]
    c = centroids.loc[cluster_id]
    print(f"{int(cluster_id):>8} {c['PC1']:>+10.2f} {c['PC2']:>+10.2f} {c['PC3']:>+10.2f} {len(sub):>6}")

print("\nAll files saved to:", OUTPUT_DIR)
