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
    """Find the AICc-best model row and return K, cov, sil, CH, DB."""
    suffix = get_template_suffix(template)
    path = OUTPUT_ROOT / f"template_{template}" / embedding_id / f"model_results_narrative{suffix}.csv"
    df = pd.read_csv(path)
    # Pick row with lowest AICc
    best = df.loc[df["aicc"].idxmin()]
    return {
        "K": int(best["K"]),
        "cov": str(best["covariance_type"]),
        "silhouette": float(best["silhouette"]),
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
    """Build a Numeric Baseline row using GMM-AICc."""
    anova_path = OUTPUT_ROOT / "numeric_marks_anova_by_cluster.csv"
    df = pd.read_csv(anova_path)
    sub = df[(df["cluster_label"] == "gmm_aicc_best_label") & (df["outcome"].isin(SUBJECTS))]
    mean_eta = float(sub["eta_squared"].mean()) if not sub.empty else float("nan")
    # Numeric K=4 (AICc) is the standard baseline; sil/CH/DB are not computed for numeric against narrative embedding space.
    # Use placeholders sourced from the paper's original final_model_comparison (K=4, full cov).
    # We set internal metrics to NaN so the heatmap normalization uses narrative rows only.
    return {
        "Template": "Numeric",
        "Model": "GMM-AICc",
        "Winner K": 4,
        "Winner Cov": "full",
        "Silhouette (Cosine)": np.nan,
        "Calinski-Harabasz": np.nan,
        "Davies-Bouldin": np.nan,
        "ARI (vs Numeric)": np.nan,  # filled by plot script from narrative mean
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
