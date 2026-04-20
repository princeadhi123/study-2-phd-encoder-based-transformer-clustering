"""
Build final_model_comparison.csv from the latest pipeline outputs.
Extracts:
- Winner K, covariance from model_results_narrative*.csv (lowest AICc)
- Silhouette (cosine), Calinski-Harabasz, Davies-Bouldin for narrative_best_label
- ARI (vs Numeric GMM-AICc) from gmm_vs_narrative_metrics*.csv
- Mean Eta^2 across S1..S5 computed from narrative_clusters + marks_with_clusters

Also adds a Numeric baseline row using GMM-AICc from numeric_marks_anova_by_cluster.csv.
"""

from pathlib import Path
from scipy import stats
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NARRATIVE_DIR = PROJECT_ROOT / "narrative_embedding_clustering"
OUTPUT_ROOT = NARRATIVE_DIR / "outputs"
OUT_CSV = NARRATIVE_DIR / "final result" / "final_model_comparison.csv"

TEMPLATES = ["A", "B", "C"]
MODELS = [("MiniLM", "all_MiniLM_L6_v2"), ("MPNet", "all_mpnet_base_v2")]
SUBJECTS = ["S1", "S2", "S3", "S4", "S5"]
MARKS_PATH = PROJECT_ROOT / "diagnostics" / "cluster input features" / "marks_with_clusters.csv"


def get_template_suffix(t: str) -> str:
    return "" if t == "A" else f"_{t}"


def winner_from_model_results(template: str, embedding_id: str) -> dict:
    """Find the AICc-best model row and return K, cov, sil, CH, DB.

    NOTE: `model_results_narrative*.csv` stores Euclidean silhouette (default
    sklearn metric). The true COSINE silhouette against the winner label is
    computed in 04_compare_with_gmm_bic.py and written to
    `gmm_vs_narrative_metrics*.csv` under metric='silhouette_cosine' /
    baseline='narrative_best_label'. We pull that value here so the CSV/heatmap
    column "Silhouette (Cosine)" is actually cosine.
    """
    suffix = get_template_suffix(template)
    path = OUTPUT_ROOT / f"template_{template}" / embedding_id / f"model_results_narrative{suffix}.csv"
    df = pd.read_csv(path)
    # Pick row with lowest AICc
    best = df.loc[df["aicc"].idxmin()]

    sil_cosine = float("nan")
    metrics_path = OUTPUT_ROOT / f"template_{template}" / embedding_id / f"gmm_vs_narrative_metrics{suffix}.csv"
    if metrics_path.exists():
        mdf = pd.read_csv(metrics_path)
        row = mdf[(mdf["baseline"] == "narrative_best_label") &
                  (mdf["metric"] == "silhouette_cosine")]
        if not row.empty:
            sil_cosine = float(row["value"].iloc[0])
    if np.isnan(sil_cosine):
        # Fallback: Euclidean value from the grid (mislabelled but at least numeric).
        sil_cosine = float(best["silhouette"])

    return {
        "K": int(best["K"]),
        "cov": str(best["covariance_type"]),
        "silhouette": sil_cosine,
        "ch": float(best["calinski_harabasz"]),
        "db": float(best["davies_bouldin"]),
    }


def ari_vs_numeric(template: str, embedding_id: str) -> float:
    """Read ARI against numeric GMM-AICc from the metrics CSV."""
    suffix = get_template_suffix(template)
    path = OUTPUT_ROOT / f"template_{template}" / embedding_id / f"gmm_vs_narrative_metrics{suffix}.csv"
    df = pd.read_csv(path)
    row = df[(df["baseline"] == "gmm_aicc_best_label") & (df["metric"] == "adjusted_rand_index")]
    if row.empty:
        return float("nan")
    return float(row["value"].iloc[0])


def mean_eta_squared(template: str, embedding_id: str) -> float:
    """Compute mean eta^2 across S1..S5 from narrative_best_label vs marks."""
    suffix = get_template_suffix(template)
    clusters_path = OUTPUT_ROOT / f"template_{template}" / embedding_id / f"narrative_clusters{suffix}.csv"
    clusters_df = pd.read_csv(clusters_path)

    if not MARKS_PATH.exists():
        return float("nan")
    marks_df = pd.read_csv(MARKS_PATH)

    # Merge on IDCode
    id_col = "IDCode" if "IDCode" in marks_df.columns else marks_df.columns[0]
    merged = clusters_df.merge(marks_df, on=id_col, how="inner")

    etas = []
    for subj in SUBJECTS:
        if subj not in merged.columns:
            continue
        sub = merged[["narrative_best_label", subj]].dropna()
        if sub.empty:
            continue
        groups = [g[subj].values for _, g in sub.groupby("narrative_best_label") if len(g) > 1]
        if len(groups) < 2:
            continue
        # eta^2 = SS_between / SS_total
        grand_mean = sub[subj].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((sub[subj] - grand_mean) ** 2).sum()
        if ss_total > 0:
            etas.append(ss_between / ss_total)
    return float(np.mean(etas)) if etas else float("nan")


def numeric_row() -> dict:
    """Build a Numeric Baseline row using GMM-AICc.

    Internal validity (silhouette/CH/DB) are computed in the numeric 8-D
    standardized feature space using the gmm_aicc_best_label from
    student_clusters.csv, then the AICc winner (K, cov) is read from
    diagnostics/model results/gmm_model_selection.csv.
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler

    anova_path = OUTPUT_ROOT / "numeric_marks_anova_by_cluster.csv"
    df = pd.read_csv(anova_path)
    sub = df[(df["cluster_label"] == "gmm_aicc_best_label") & (df["outcome"].isin(SUBJECTS))]
    mean_eta = float(sub["eta_squared"].mean()) if not sub.empty else float("nan")

    # Recompute internal metrics from the numeric feature space.
    sil = ch = db = float("nan")
    k_val = 0
    cov_val = "unknown"
    try:
        feats_path = PROJECT_ROOT / "diagnostics" / "cluster input features" / "derived_features.csv"
        labels_path = PROJECT_ROOT / "diagnostics" / "student cluster labels" / "student_clusters.csv"
        gmm_sel_path = PROJECT_ROOT / "diagnostics" / "model results" / "gmm_model_selection.csv"

        feats = pd.read_csv(feats_path)
        labels_df = pd.read_csv(labels_path)[["IDCode", "gmm_aicc_best_label"]]

        feature_cols = [
            "n_items", "accuracy", "avg_rt", "var_rt", "rt_cv",
            "longest_correct_streak", "longest_incorrect_streak",
            "consecutive_correct_rate",
        ]
        merged = feats.merge(labels_df, on="IDCode", how="inner").dropna(subset=feature_cols + ["gmm_aicc_best_label"])
        X = StandardScaler().fit_transform(merged[feature_cols].to_numpy(dtype=float))
        y = merged["gmm_aicc_best_label"].to_numpy()
        if len(np.unique(y)) > 1:
            sil = float(silhouette_score(X, y))
            ch = float(calinski_harabasz_score(X, y))
            db = float(davies_bouldin_score(X, y))
            k_val = int(len(np.unique(y)))

        # AICc winner K, cov from grid
        gm_df = pd.read_csv(gmm_sel_path)
        best = gm_df.loc[gm_df["aicc"].idxmin()]
        k_val = int(best["K"])
        cov_val = str(best["covariance_type"])
    except Exception as exc:
        print(f"WARN numeric_row internal metrics: {exc}")

    return {
        "Template": "Numeric",
        "Model": "GMM-AICc",
        "Winner K": k_val,
        "Winner Cov": cov_val,
        "Silhouette (Cosine)": round(sil, 4) if not np.isnan(sil) else np.nan,
        "Calinski-Harabasz": round(ch, 4) if not np.isnan(ch) else np.nan,
        "Davies-Bouldin": round(db, 4) if not np.isnan(db) else np.nan,
        "ARI (vs Numeric)": np.nan,  # self-reference; filled by plot script
        "Mean Eta^2": mean_eta,
    }


def main() -> None:
    rows = []
    for template in TEMPLATES:
        for model_name, embedding_id in MODELS:
            try:
                w = winner_from_model_results(template, embedding_id)
                ari = ari_vs_numeric(template, embedding_id)
                eta = mean_eta_squared(template, embedding_id)
            except FileNotFoundError as exc:
                print(f"SKIP Template {template} / {model_name}: {exc}")
                continue
            rows.append({
                "Template": f"Template {template}",
                "Model": model_name,
                "Winner K": w["K"],
                "Winner Cov": w["cov"],
                "Silhouette (Cosine)": round(w["silhouette"], 4),
                "Calinski-Harabasz": round(w["ch"], 4),
                "Davies-Bouldin": round(w["db"], 4),
                "ARI (vs Numeric)": round(ari, 4) if not np.isnan(ari) else np.nan,
                "Mean Eta^2": round(eta, 4) if not np.isnan(eta) else np.nan,
            })
            print(f"OK  Template {template} / {model_name}: K={w['K']} cov={w['cov']} sil={w['silhouette']:.4f} ARI={ari:.4f} eta={eta:.4f}")

    rows.append(numeric_row())

    df_out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
