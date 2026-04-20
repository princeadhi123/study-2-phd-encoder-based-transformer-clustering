import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from config import OUTPUT_DIR, STUDENT_CLUSTERS_PATH, make_versioned_filename, DERIVED_FEATURES_PATH, NARRATIVE_TEMPLATE_VERSION, EMBEDDING_MODEL_NAME


def _load_embeddings():

    emb_filename = make_versioned_filename("embeddings.npy")
    index_filename = make_versioned_filename("embeddings_index.csv")
    emb_path = OUTPUT_DIR / emb_filename
    index_path = OUTPUT_DIR / index_filename
    if not emb_path.exists() or not index_path.exists():
        raise SystemExit("Embeddings or index not found. Run 02_compute_embeddings.py first.")
    X = np.load(emb_path)
    index_df = pd.read_csv(index_path)
    return X, index_df


def _load_clusters():

    narrative_clusters_filename = make_versioned_filename("narrative_clusters.csv")
    narrative_clusters_path = OUTPUT_DIR / narrative_clusters_filename
    if not narrative_clusters_path.exists():
        raise SystemExit("Missing narrative_clusters.csv. Run 03_cluster_embeddings.py first.")
    nar_clusters = pd.read_csv(narrative_clusters_path)

    stud_clusters = pd.read_csv(STUDENT_CLUSTERS_PATH)

    return nar_clusters, stud_clusters


def _prepare_coords(index_df: pd.DataFrame, nar_clusters: pd.DataFrame, stud_clusters: pd.DataFrame) -> pd.DataFrame:
    """Merge embedding index with narrative and numeric cluster labels.

    This does *not* compute any projection; projections are computed separately
    and added as columns when plotting.
    """
    coords = index_df.merge(nar_clusters[["IDCode", "narrative_best_label"]], on="IDCode", how="left")
    coords = coords.merge(
        stud_clusters[["IDCode", "gmm_bic_best_label", "gmm_aic_best_label"]], on="IDCode", how="left"
    )
    return coords


def _plot_scatter(
    coords: pd.DataFrame,
    color_col: str,
    title: str,
    filename: str,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for visualization. Install it with:\n"
            "pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 6))

    labels = sorted(coords[color_col].dropna().unique())
    if color_col == "narrative_best_label":
        cluster_palette = {
            0: "#0072B2",
            1: "#D55E00",
            2: "#009E73",
            3: "#CC79A7",
            4: "#999933",
            5: "#56B4E9",
            6: "#E69F00",
            7: "#000000",
            8: "#999999",
        }

        for label in labels:
            label_int = int(label)
            subset = coords[coords[color_col] == label]
            ax.scatter(
                subset[x_col],
                subset[y_col],
                color=cluster_palette.get(label_int, "#333333"),
                s=15,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.3,
                label=f"Cluster {label_int}",
            )

        ax.legend(title="Narrative Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        scatter = ax.scatter(
            coords[x_col],
            coords[y_col],
            c=coords[color_col],
            cmap="tab10",
            s=15,
            alpha=0.8,
        )

        handles, _ = scatter.legend_elements(prop="colors")
        legend_labels = [f"{color_col} = {int(l)}" for l in labels]
        ax.legend(handles, legend_labels, title=color_col, bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved {title} plot to {out_path}")



def _enforce_pca_orientation(coords_pca: pd.DataFrame, df_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns PCA signs to ensure consistent interpretation:
    - PC1: High Performance (Accuracy) -> Negative values
    - PC2: High Variance (Response Variance) -> Negative values
    - PC3: Slow Speed (Avg Response Time) -> Positive values
    
    Returns:
        coords_pca: The dataframe with potentially flipped PC columns
        loadings: The correlation matrix (transposed for plotting)
    """
    # Feature Map (Internal Name -> Display Name). Must mirror the canonical
    # 8-feature set used in build_final_comparison.py / reviewer_ablation.py /
    # the narrative builder, so the PCA-loadings heatmap reports *all* inputs.
    feature_map = {
        "n_items": "Items Attempted",
        "accuracy": "Accuracy",
        "consecutive_correct_rate": "Consec. Correct Rate",
        "longest_correct_streak": "Max Correct Streak",
        "longest_incorrect_streak": "Max Incorrect Streak",
        "avg_rt": "Avg Response Time",
        "var_rt": "RT Variance",
        "rt_cv": "RT Coeff. Variation",
    }

    # Merge PCA coords with features
    valid_cols = [c for c in feature_map.keys() if c in df_features.columns]
    pca_cols = [c for c in coords_pca.columns if c.startswith("dim")]
    
    merged = coords_pca.merge(df_features[["IDCode"] + valid_cols], on="IDCode", how="inner")
    
    if merged.empty:
        print("Warning: No matching IDs found for feature correlation. PCA orientation not enforced.")
        return coords_pca, pd.DataFrame()

    # Calculate initial correlations
    cols_to_corr = pca_cols + valid_cols
    correlations = merged[cols_to_corr].corr(method="pearson")
    
    # We only need rows=valid_cols, cols=pca_cols
    raw_loadings = correlations.loc[valid_cols, pca_cols]

    # Check Anchors and Flip if needed
    # PC1 Anchor: Accuracy should be Negative ( < 0 )
    if "dim1" in pca_cols and "accuracy" in valid_cols:
        if raw_loadings.loc["accuracy", "dim1"] > 0:
            print("  -> Flipping PC1 sign (forcing High Accuracy to Negative)")
            coords_pca["dim1"] *= -1
            raw_loadings["dim1"] *= -1

    # PC2 Anchor: RT Variance should be Negative ( < 0 ). `var_rt` is the
    # canonical response-time variance feature (was previously mis-named
    # `response_variance`, which does not exist in the feature matrix).
    if "dim2" in pca_cols and "var_rt" in valid_cols:
        if raw_loadings.loc["var_rt", "dim2"] > 0:
            print("  -> Flipping PC2 sign (forcing High RT Variance to Negative)")
            coords_pca["dim2"] *= -1
            raw_loadings["dim2"] *= -1
            
    # PC3 Anchor: Avg Response Time should be Positive ( > 0 )
    if "dim3" in pca_cols and "avg_rt" in valid_cols:
        if raw_loadings.loc["avg_rt", "dim3"] < 0:
            print("  -> Flipping PC3 sign (forcing Slow RT to Positive)")
            coords_pca["dim3"] *= -1
            raw_loadings["dim3"] *= -1

    # Format loadings for plotting (Rename cols/rows)
    pc_names = [f"PC{i+1}" for i in range(len(pca_cols))]
    raw_loadings.columns = pc_names
    
    # Transpose: PCs on Y, Features on X
    final_loadings = raw_loadings.T
    final_loadings = final_loadings.rename(columns=feature_map)

    # Reorder columns (performance block first, then streaks, then timing).
    desired_order = [
        "Items Attempted", "Accuracy", "Consec. Correct Rate",
        "Max Correct Streak", "Max Incorrect Streak",
        "Avg Response Time", "RT Variance", "RT Coeff. Variation",
    ]
    plot_cols = [c for c in desired_order if c in final_loadings.columns]
    final_loadings = final_loadings[plot_cols]
    
    return coords_pca, final_loadings


def _plot_pca_heatmap(loadings: pd.DataFrame) -> None:
    """Plots the pre-calculated and aligned loadings heatmap."""
    if loadings.empty:
        return

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("seaborn not installed, skipping heatmap.")
        return

    # Features with zero variance (e.g. n_items = 57 for every student) have
    # undefined Pearson correlations with the PCs. Show them explicitly as
    # zero-loading columns tagged "(const.)" so the reader sees all analysis
    # inputs rather than having them silently disappear.
    const_mask = loadings.isna().all(axis=0)
    if const_mask.any():
        const_cols = loadings.columns[const_mask]
        loadings = loadings.copy()
        loadings[const_cols] = 0.0
        loadings = loadings.rename(columns={c: f"{c}\n(const.)" for c in const_cols})

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        loadings,
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

    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / make_versioned_filename("pca_loadings_heatmap.png")
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved PCA loadings heatmap to {out_path}")


def _plot_narrative_k_selection() -> None:
    # Load the latest model results
    pattern = "model_results_narrative*.csv"
    files = sorted(OUTPUT_DIR.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        print("No narrative model results found. Run 03_cluster_embeddings.py first.")
        return
    
    latest_file = files[0]
    print(f"Loading model results from: {latest_file.name}")
    df = pd.read_csv(latest_file)
    
    if "bic" in df.columns:
        _plot_metric_selection(df, "bic", "BIC", "bic_vs_k")
    
    if "aicc" in df.columns:
        _plot_metric_selection(df, "aicc", "AICc", "aicc_vs_k")


def _plot_metric_selection(df: pd.DataFrame, metric: str, title_suffix: str, filename_suffix: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter out infinite values for plotting
    df_clean = df[np.isfinite(df[metric])].copy()
    
    if df_clean.empty:
        print(f"No valid {metric} values to plot.")
        plt.close(fig)
        return

    for cov, sub in df_clean.groupby("covariance_type"):
        sub_sorted = sub.sort_values("K")
        ax.plot(sub_sorted["K"], sub_sorted[metric], marker="o", label=cov)

    best_idx = df_clean[metric].idxmin()
    best_row = df_clean.loc[best_idx]
    best_k = int(best_row["K"])
    best_cov = str(best_row["covariance_type"])
    best_val = float(best_row[metric])
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.7)

    ax.set_xlabel("Number of narrative clusters (K)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Narrative GMM model selection by {title_suffix} (best K={best_k}, cov={best_cov})")
    ax.grid(True, alpha=0.2)
    ax.legend(title="covariance_type")

    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / make_versioned_filename(f"narrative_gmm_{filename_suffix}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(
        "Selected narrative GMM by {metric}: K={k}, cov={cov}, {metric}={val:.2f}".format(
            metric=metric.upper(), k=best_k, cov=best_cov, val=best_val
        )
    )
    print(f"Saved narrative GMM {metric.upper()} plot to {out_path}")


def main() -> None:
    _plot_narrative_k_selection()
    try:
        X, index_df = _load_embeddings()
        nar_clusters, stud_clusters = _load_clusters()
        coords = _prepare_coords(index_df, nar_clusters, stud_clusters)
        
        # Additional visualizations could be added here
        
    except Exception as e:
        print(f"Visualization error: {e}")
        return


    # --- PCA projection (3 components to capture Performance, Intensity, Timing) ---
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    coords_pca = coords.copy()
    coords_pca["dim1"] = X_pca[:, 0]
    coords_pca["dim2"] = X_pca[:, 1]
    coords_pca["dim3"] = X_pca[:, 2]

    # Calculate Variance Explained
    var_exp = pca.explained_variance_ratio_
    print(f"PCA Variance Explained: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}, PC3={var_exp[2]:.1%}")

    # --- Load Features and Enforce PCA Orientation ---
    if DERIVED_FEATURES_PATH.exists():
        df_features = pd.read_csv(DERIVED_FEATURES_PATH)
        # Recalculate derived columns only if missing (new 8-feature schema already has them)
        with np.errstate(divide="ignore", invalid="ignore"):
            if "accuracy" not in df_features.columns and "total_correct" in df_features.columns:
                df_features["accuracy"] = df_features["total_correct"] / df_features["n_items"].replace(0, np.nan)
            if "rt_cv" not in df_features.columns and "var_rt" in df_features.columns and "avg_rt" in df_features.columns:
                std_rt = np.sqrt(df_features["var_rt"].clip(lower=0.0))
                df_features["rt_cv"] = std_rt / df_features["avg_rt"].replace(0, np.nan)
        
        # Enforce consistent signs (e.g. PC1 = Low Perf, PC3 = Slow)
        print("Checking PCA orientation against behavioral anchors...")
        coords_pca, loadings = _enforce_pca_orientation(coords_pca, df_features)
        
        # Plot aligned heatmap
        _plot_pca_heatmap(loadings)
    else:
        print("Derived features not found; skipping PCA alignment and heatmap.")

    # Calculate 2D Silhouette Scores to identify the best view (using aligned coords)
    labels = coords["narrative_best_label"]
    # Re-extract numpy array from coords_pca because columns might have been flipped
    X_pca_aligned = coords_pca[["dim1", "dim2", "dim3"]].values

    sil_12 = silhouette_score(X_pca_aligned[:, [0, 1]], labels)
    sil_13 = silhouette_score(X_pca_aligned[:, [0, 2]], labels)
    sil_23 = silhouette_score(X_pca_aligned[:, [1, 2]], labels)

    print(f"2D Silhouette Scores: PC1-PC2={sil_12:.3f}, PC1-PC3={sil_13:.3f}, PC2-PC3={sil_23:.3f}")

    # Plot 1: Standard PC1 vs PC2
    _plot_scatter(
        coords_pca,
        color_col="narrative_best_label",
        title=f"Narrative Clusters: PC1 vs PC2 (Sil={sil_12:.2f})",
        filename=make_versioned_filename("embeddings_pca_PC1_vs_PC2.png"),
        x_col="dim1",
        y_col="dim2",
        x_label=f"PC1 ({var_exp[0]:.1%} var)",
        y_label=f"PC2 ({var_exp[1]:.1%} var)",
    )

    # Plot 2: PC1 vs PC3 (Performance vs Consistency) - Often the BEST separator
    # Standard scatter
    _plot_scatter(
        coords_pca,
        color_col="narrative_best_label",
        title=f"Narrative Clusters: PC1 vs PC3 (Sil={sil_13:.2f})",
        filename=make_versioned_filename("embeddings_pca_PC1_vs_PC3.png"),
        x_col="dim1",
        y_col="dim3",
        x_label=f"PC1 ({var_exp[0]:.1%} var)",
        y_label=f"PC3 ({var_exp[2]:.1%} var)",
    )

    # Plot 3: PC2 vs PC3
    _plot_scatter(
        coords_pca,
        color_col="narrative_best_label",
        title=f"Narrative Clusters: PC2 vs PC3 (Sil={sil_23:.2f})",
        filename=make_versioned_filename("embeddings_pca_PC2_vs_PC3.png"),
        x_col="dim2",
        y_col="dim3",
        x_label=f"PC2 ({var_exp[1]:.1%} var)",
        y_label=f"PC3 ({var_exp[2]:.1%} var)",
    )

    # Keep the numeric cluster comparisons on PC1/PC2 for reference
    _plot_scatter(
        coords_pca,
        color_col="gmm_bic_best_label",
        title="Numeric GMM-BIC clusters (PCA: PC1 vs PC2)",
        filename=make_versioned_filename("embeddings_pca_gmm_bic_clusters.png"),
        x_col="dim1",
        y_col="dim2",
        x_label="PC1",
        y_label="PC2",
    )

    _plot_scatter(
        coords_pca,
        color_col="gmm_aic_best_label",
        title="Numeric GMM-AIC clusters (PCA: PC1 vs PC2)",
        filename=make_versioned_filename("embeddings_pca_gmm_aic_clusters.png"),
        x_col="dim1",
        y_col="dim2",
        x_label="PC1",
        y_label="PC2",
    )

    # --- UMAP projection (narrative clusters only) ---
    try:
        import umap

        umap_model = umap.UMAP(n_components=2, random_state=42)
        X_umap = umap_model.fit_transform(X)
        coords_umap = coords.copy()
        coords_umap["dim1"] = X_umap[:, 0]
        coords_umap["dim2"] = X_umap[:, 1]

        _plot_scatter(
            coords_umap,
            color_col="narrative_best_label",
            title="Narrative GMM-BIC clusters (embedding UMAP)",
            filename=make_versioned_filename("embeddings_umap_narrative_clusters.png"),
            x_col="dim1",
            y_col="dim2",
            x_label="UMAP1 (narrative embeddings)",
            y_label="UMAP2 (narrative embeddings)",
        )
    except ImportError:
        print(
            "umap-learn is not installed; skipping UMAP plot. "
            "Install it with: pip install umap-learn"
        )

    # --- t-SNE projection (narrative clusters only) ---
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X)
    coords_tsne = coords.copy()
    coords_tsne["dim1"] = X_tsne[:, 0]
    coords_tsne["dim2"] = X_tsne[:, 1]

    _plot_scatter(
        coords_tsne,
        color_col="narrative_best_label",
        title="Narrative GMM-BIC clusters (embedding t-SNE)",
        filename=make_versioned_filename("embeddings_tsne_narrative_clusters.png"),
        x_col="dim1",
        y_col="dim2",
        x_label="t-SNE1 (narrative embeddings)",
        y_label="t-SNE2 (narrative embeddings)",
    )

    # --- LDA projection (supervised by narrative clusters) ---
    y = coords["narrative_best_label"].to_numpy()
    # Remove any NaN labels if they exist
    mask = ~np.isnan(y)
    if not mask.all():
        y = y[mask]
        X_for_lda = X[mask]
    else:
        X_for_lda = X

    n_classes = len(np.unique(y))
    
    if n_classes < 3:
        print(f"Skipping LDA plot: Requires at least 3 clusters for 2D projection (found {n_classes}).")
    else:
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_lda = lda.fit_transform(X_for_lda, y)
        coords_lda = coords.copy()
        coords_lda["dim1"] = X_lda[:, 0]
        coords_lda["dim2"] = X_lda[:, 1]

        _plot_scatter(
            coords_lda,
            color_col="narrative_best_label",
            title="Narrative GMM-BIC clusters (embedding LDA)",
            filename=make_versioned_filename("embeddings_lda_narrative_clusters.png"),
            x_col="dim1",
            y_col="dim2",
            x_label="LDA1 (narrative embeddings)",
            y_label="LDA2 (narrative embeddings)",
        )


if __name__ == "__main__":
    main()
