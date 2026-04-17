from pathlib import Path

import pandas as pd


TEMPLATES = ["A", "B", "C"]
EMBEDDING_IDS = ["all_MiniLM_L6_v2", "all_mpnet_base_v2"]


def load_metrics() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    output_root = base_dir / "outputs"

    rows: list[dict] = []
    for template in TEMPLATES:
        for embedding_id in EMBEDDING_IDS:
            template_dir = output_root / f"template_{template}" / embedding_id
            if template == "A":
                fname = "gmm_vs_narrative_metrics.csv"
            else:
                fname = f"gmm_vs_narrative_metrics_{template}.csv"
            path = template_dir / fname
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                rows.append(
                    {
                        "template": template,
                        "embedding_id": embedding_id,
                        "baseline": str(r["baseline"]),
                        "metric": str(r["metric"]),
                        "value": float(r["value"]),
                    }
                )

    if not rows:
        raise SystemExit("No metrics found for any template.")

    return pd.DataFrame(rows)


def plot_internal_metrics(metrics: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    base_dir = Path(__file__).resolve().parent
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    internal_metrics = ["silhouette_cosine", "calinski_harabasz", "davies_bouldin"]
    baselines = ["narrative_best_label", "gmm_aicc_best_label"]

    for metric_name in internal_metrics:
        sub = metrics[metrics["metric"] == metric_name]
        sub = sub[sub["baseline"].isin(baselines)]
        if sub.empty:
            continue

        # One combined figure per metric, with a subplot for each embedding model.
        n_emb = len(EMBEDDING_IDS)
        fig, axes = plt.subplots(1, n_emb, figsize=(5 * n_emb, 5), sharey=True)
        if n_emb == 1:
            axes = [axes]

        handles = None
        labels = None

        for ax, embedding_id in zip(axes, EMBEDDING_IDS):
            emb_sub = sub[sub["embedding_id"] == embedding_id]
            if emb_sub.empty:
                ax.set_visible(False)
                continue

            pivot = emb_sub.pivot(index="template", columns="baseline", values="value").reindex(TEMPLATES)

            x = range(len(pivot.index))
            width = 0.25
            offsets = {
                "narrative_best_label": -width / 2,
                "gmm_aicc_best_label": width / 2,
            }

            for baseline in baselines:
                if baseline not in pivot.columns:
                    continue
                values = pivot[baseline].values
                bars = ax.bar(
                    [xi + offsets[baseline] for xi in x],
                    values,
                    width=width,
                    label=baseline,
                )
                if handles is None:
                    handles = []
                    labels = []
                # Capture legend entries from the first subplot only.
                if baseline not in (labels or []):
                    handles.append(bars)
                    labels.append(baseline)

            ax.set_xticks(list(x))
            ax.set_xticklabels(pivot.index)
            ax.set_xlabel("Template")
            ax.set_title(embedding_id)
            ax.grid(True, axis="y", alpha=0.2)

        # Shared y-label on the leftmost axis if at least one axis is visible.
        if isinstance(axes, list) and axes:
            axes[0].set_ylabel(metric_name)

        # Put a shared legend at the bottom, outside the plotting area.
        if handles and labels:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=len(labels),
            )

        fig.suptitle(f"Internal metric: {metric_name} by template and clustering", y=0.96)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))

        out_path = figures_dir / f"internal_{metric_name}_by_template.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"Saved combined {metric_name} plot to {out_path}")


def plot_ari(metrics: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    base_dir = Path(__file__).resolve().parent
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sub = metrics[metrics["metric"] == "adjusted_rand_index"]
    if sub.empty:
        return

    baselines = ["gmm_aicc_best_label"]
    sub = sub[sub["baseline"].isin(baselines)]

    import matplotlib.pyplot as plt  # type: ignore[no-redef]

    # One combined figure with a subplot per embedding model.
    n_emb = len(EMBEDDING_IDS)
    fig, axes = plt.subplots(1, n_emb, figsize=(5 * n_emb, 5), sharey=True)
    if n_emb == 1:
        axes = [axes]

    handles = None
    labels = None

    for ax, embedding_id in zip(axes, EMBEDDING_IDS):
        emb_sub = sub[sub["embedding_id"] == embedding_id]
        if emb_sub.empty:
            ax.set_visible(False)
            continue

        pivot = emb_sub.pivot(index="template", columns="baseline", values="value").reindex(TEMPLATES)

        x = range(len(pivot.index))
        width = 0.25
        offsets = {
            "gmm_aicc_best_label": 0.0,
        }

        for baseline in baselines:
            if baseline not in pivot.columns:
                continue
            values = pivot[baseline].values
            bars = ax.bar(
                [xi + offsets[baseline] for xi in x],
                values,
                width=width,
                label=baseline,
            )
            if handles is None:
                handles = []
                labels = []
            if baseline not in (labels or []):
                handles.append(bars)
                labels.append(baseline)

        ax.set_xticks(list(x))
        ax.set_xticklabels(pivot.index)
        ax.set_xlabel("Template")
        ax.set_title(embedding_id)
        ax.grid(True, axis="y", alpha=0.2)

    # Shared y-label on the leftmost axis if at least one axis is visible.
    if isinstance(axes, list) and axes:
        axes[0].set_ylabel("Adjusted Rand index")

    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(labels),
        )

    fig.suptitle(
        "External metric: ARI between numeric and narrative clusters", y=0.96
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))

    out_path = figures_dir / "external_ari_by_template.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved combined ARI plot to {out_path}")


def load_anova_results() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    output_root = base_dir / "outputs"

    rows: list[dict] = []

    # First, load the global numeric ANOVA (BIC/AIC), which does not depend on
    # narrative template or embedding model.
    numeric_path = output_root / "numeric_marks_anova_by_cluster.csv"
    if numeric_path.exists():
        df_num = pd.read_csv(numeric_path)
        if not df_num.empty:
            for _, r in df_num.iterrows():
                eta_val = r.get("eta_squared")
                if pd.isna(eta_val):
                    continue
                rows.append(
                    {
                        # Mark numeric ANOVA as global rather than tied to a
                        # specific template or embedding model.
                        "template": "GLOBAL",
                        "embedding_id": "GLOBAL",
                        "cluster_label": str(r["cluster_label"]),
                        "outcome": str(r["outcome"]),
                        "eta_squared": float(eta_val),
                    }
                )
    # Then, load template-specific narrative ANOVA for each template and
    # embedding model.
    for template in TEMPLATES:
        for embedding_id in EMBEDDING_IDS:
            template_dir = output_root / f"template_{template}" / embedding_id
            if template == "A":
                fname = "marks_anova_by_cluster.csv"
            else:
                fname = f"marks_anova_by_cluster_{template}.csv"
            path = template_dir / fname
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            for _, r in df.iterrows():
                eta_val = r.get("eta_squared")
                if pd.isna(eta_val):
                    continue
                rows.append(
                    {
                        "template": template,
                        "embedding_id": embedding_id,
                        "cluster_label": str(r["cluster_label"]),
                        "outcome": str(r["outcome"]),
                        "eta_squared": float(eta_val),
                    }
                )

    if not rows:
        raise SystemExit("No ANOVA results found for any template.")

    return pd.DataFrame(rows)


def plot_anova_subjects(anova_df: pd.DataFrame) -> None:
    """Entry point for subject-wise ANOVA plots.

    - Numeric ANOVA (BIC/AIC) is shown once, aggregated across templates and
      embedding models.
    - Narrative ANOVA is shown separately for each template and embedding
      model.
    """

    plot_anova_subjects_numeric(anova_df)
    plot_anova_subjects_numeric_by_outcome(anova_df)
    plot_anova_subjects_narrative_per_template_model(anova_df)


def plot_anova_subjects_numeric(anova_df: pd.DataFrame) -> None:
    """Plot mean eta-squared for subject marks for numeric GMM clusters only.

    This aggregates over templates and embedding models, since the numeric
    clusters and marks do not depend on the narrative template.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    base_dir = Path(__file__).resolve().parent
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    subject_outcomes = {"S1", "S2", "S3", "S5", "S6"}
    baselines = ["gmm_aicc_best_label"]

    sub = anova_df[anova_df["outcome"].isin(subject_outcomes)]
    sub = sub[sub["cluster_label"].isin(baselines)]
    if sub.empty:
        return

    # Average eta-squared across templates and embedding models for each
    # numeric clustering.
    grouped = sub.groupby("cluster_label", as_index=False)["eta_squared"].mean()

    fig, ax = plt.subplots(figsize=(6, 5))

    x = range(len(grouped.index))
    values = grouped["eta_squared"].values
    labels = grouped["cluster_label"].tolist()

    bars = ax.bar(x, values, width=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean eta-squared (subjects S1, S2, S3, S5, S6)")
    ax.set_title("ANOVA effect size (subjects): numeric GMM clusters")
    ax.grid(True, axis="y", alpha=0.2)

    out_path = figures_dir / "anova_eta_subjects_numeric.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved numeric ANOVA subjects eta-squared plot to {out_path}")


def plot_anova_subjects_numeric_by_outcome(anova_df: pd.DataFrame) -> None:
    """Plot eta-squared for each subject separately for numeric GMM clusters.

    X-axis: subject (S1, S2, S3, S5, S6).
    Bars: gmm_aic_best_label.
    Aggregated over templates and embedding models.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    base_dir = Path(__file__).resolve().parent
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    subject_outcomes = ["S1", "S2", "S3", "S5", "S6"]
    baselines = ["gmm_aicc_best_label"]

    sub = anova_df[anova_df["outcome"].isin(subject_outcomes)]
    sub = sub[sub["cluster_label"].isin(baselines)]
    if sub.empty:
        return

    grouped = (
        sub.groupby(["outcome", "cluster_label"], as_index=False)["eta_squared"]
        .mean()
    )

    # Pivot so that rows are outcomes and columns are the two numeric clusterings.
    pivot = grouped.pivot(index="outcome", columns="cluster_label", values="eta_squared")
    pivot = pivot.reindex(subject_outcomes)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = range(len(subject_outcomes))
    width = 0.35
    offsets = {
        "gmm_aicc_best_label": 0.0,
    }

    handles = []
    labels = []

    for label in baselines:
        if label not in pivot.columns:
            continue
        values = pivot[label].values
        bars = ax.bar(
            [xi + offsets[label] for xi in x],
            values,
            width=width,
            label=label,
        )
        handles.append(bars)
        labels.append(label)

    ax.set_xticks(list(x))
    ax.set_xticklabels(subject_outcomes)
    ax.set_ylabel("Eta-squared (per subject)")
    ax.set_xlabel("Subject")
    ax.set_title("ANOVA eta-squared by subject: numeric GMM clusters")
    ax.grid(True, axis="y", alpha=0.2)

    if handles and labels:
        ax.legend(handles, labels)

    out_path = figures_dir / "anova_eta_subjects_numeric_by_outcome.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved numeric ANOVA eta-squared by subject plot to {out_path}")


def plot_anova_subjects_narrative_per_template_model(anova_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    base_dir = Path(__file__).resolve().parent
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    subject_outcomes = ["S1", "S2", "S3", "S5", "S6"]

    sub = anova_df[anova_df["outcome"].isin(subject_outcomes)]
    sub = sub[sub["cluster_label"] == "narrative_best_label"]
    if sub.empty:
        return

    for template in TEMPLATES:
        for embedding_id in EMBEDDING_IDS:
            te_sub = sub[
                (sub["template"] == template)
                & (sub["embedding_id"] == embedding_id)
            ]
            if te_sub.empty:
                continue

            grouped = (
                te_sub.groupby("outcome", as_index=False)["eta_squared"].mean()
            )
            grouped = grouped.set_index("outcome").reindex(subject_outcomes)

            fig, ax = plt.subplots(figsize=(6, 5))

            x = range(len(subject_outcomes))
            values = grouped["eta_squared"].values

            ax.bar(x, values, width=0.6)
            ax.set_xticks(list(x))
            ax.set_xticklabels(subject_outcomes)
            ax.set_ylabel("Eta-squared (per subject)")
            ax.set_xlabel("Subject")
            ax.set_title(
                f"ANOVA eta-squared: Template {template}, {embedding_id}"
            )
            ax.grid(True, axis="y", alpha=0.2)

            out_path = (
                figures_dir
                / f"anova_eta_subjects_narrative_template_{template}_{embedding_id}.png"
            )
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

            print(
                "Saved narrative ANOVA eta-squared by subject for template "
                f"{template}, {embedding_id} to {out_path}"
            )


def plot_mark_correlations() -> None:
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("seaborn or matplotlib not installed, skipping correlation plot.")
        return

    base_dir = Path(__file__).resolve().parent
    OUTPUT_DIR = base_dir / "outputs"  # Local redefinition or import if accessible, but using local is safer here
    marks_path = base_dir.parent / "diagnostics" / "cluster input features" / "marks_with_clusters.csv"
    
    if not marks_path.exists():
        print(f"Marks file not found at {marks_path}, skipping correlation plot.")
        return

    df = pd.read_csv(marks_path)
    # Filter for Subject columns only (S1, S2, etc.)
    subj_cols = [c for c in df.columns if c.startswith("S") and c[1:].isdigit()]
    
    if not subj_cols:
        return

    corr = df[subj_cols].corr()
    
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Subject Marks")
    fig.tight_layout()
    
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "subject_correlation_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved correlation matrix to {out_path}")


def plot_mean_eta_comparison(anova_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("seaborn or matplotlib not installed, skipping mean eta plot.")
        return

    base_dir = Path(__file__).resolve().parent
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    subjects = ["S1", "S2", "S3", "S5", "S6"]
    
    # Filter for relevant outcomes
    df_sub = anova_df[anova_df["outcome"].isin(subjects)].copy()
    
    results = []

    # 1. Numeric Baseline
    # template="GLOBAL", cluster_label="gmm_aicc_best_label"
    num_mask = (df_sub["template"] == "GLOBAL") & (df_sub["cluster_label"] == "gmm_aicc_best_label")
    if num_mask.any():
        mean_eta = df_sub.loc[num_mask, "eta_squared"].mean()
        results.append({
            "Label": "Numeric (GMM-AICc)",
            "Type": "Numeric Baseline",
            "Mean_Eta_Squared": mean_eta
        })

    # 2. Narrative Models
    # template in TEMPLATES, embedding_id in EMBEDDING_IDS, cluster_label="narrative_best_label"
    nar_mask = (df_sub["template"].isin(TEMPLATES)) & \
               (df_sub["embedding_id"].isin(EMBEDDING_IDS)) & \
               (df_sub["cluster_label"] == "narrative_best_label")
    
    if nar_mask.any():
        nar_df = df_sub[nar_mask]
        grouped = nar_df.groupby(["template", "embedding_id"])["eta_squared"].mean().reset_index()
        
        for _, row in grouped.iterrows():
            template = row["template"]
            emb_id = row["embedding_id"]
            val = row["eta_squared"]
            
            # Clean model name for display
            # emb_id is like all_MiniLM_L6_v2
            clean_model = emb_id.replace("all_", "").replace("_v2", "").replace("_", "-")
            
            label = f"Template {template}\n({clean_model})"
            
            results.append({
                "Label": label,
                "Type": "Narrative Clusters",
                "Mean_Eta_Squared": val
            })

    if not results:
        print("No data for mean eta comparison.")
        return

    res_df = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    palette = {"Numeric Baseline": "#7f7f7f", "Narrative Clusters": "#1f77b4"}
    
    ax = sns.barplot(
        data=res_df, 
        x="Label", 
        y="Mean_Eta_Squared", 
        hue="Type", 
        dodge=False,
        palette=palette
    )
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2., 
                height + 0.01, 
                f'{height:.2f}', 
                ha="center", 
                fontsize=10, 
                fontweight='bold'
            )

    plt.title("Predictive Power Comparison: Mean Eta-Squared (S1-S6)", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Eta-Squared (Higher is Stronger)", fontsize=12)
    plt.xlabel("")
    plt.ylim(0, 1.0)
    plt.legend(title=None)
    plt.tight_layout()
    
    out_path = figures_dir / "final_mean_eta_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved combined Mean Eta plot to {out_path}")


def main() -> None:
    metrics = load_metrics()
    plot_internal_metrics(metrics)
    plot_ari(metrics)

    anova_df = load_anova_results()
    plot_anova_subjects(anova_df)
    plot_mean_eta_comparison(anova_df)
    
    plot_mark_correlations()


if __name__ == "__main__":
    main()
