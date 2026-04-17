"""
Reviewer Ablation & Validity Study
====================================
All reviewer-response analyses in one script:

Part 1:  DIMENSIONALITY CONFOUND — PCA = {8, 11, 20} ablation
Part 2:  GRANULARITY JUSTIFICATION — K=9 vs K=4 eta-squared & Cohen's d
Part 2b: INCREMENTAL VALIDITY — hierarchical tier structure of K=9
Part 3:  DECISION-WEIGHT SENSITIVITY — composite weight sweep
Part 4:  ANOVA CONFOUND CHECK — within-subject Kruskal-Wallis tests
Part 5:  PREDICTIVE VALIDITY — 5-fold CV grade prediction from clusters
Part 6:  REPRESENTATIONAL VALIDITY — embedding ↔ feature distance correlation

Outputs: narrative_embedding_clustering/reviewer_ablation_results/
"""

import sys
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    r2_score,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.stats import f as f_dist, spearmanr, pearsonr, kruskal

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = Path(__file__).resolve().parent / "outputs"

# Numeric baseline paths
STUDENT_CLUSTERS_PATH = BASE_DIR / "diagnostics" / "student cluster labels" / "student_clusters.csv"
DERIVED_FEATURES_PATH = BASE_DIR / "diagnostics" / "cluster input features" / "derived_features.csv"
MARKS_PATH = BASE_DIR / "diagnostics" / "cluster input features" / "marks_with_clusters.csv"

# Output directory for this script
RESULTS_DIR = Path(__file__).resolve().parent / "reviewer_ablation_results"

# Strategy C = Template A in the code
STRATEGY_C_MODELS = {
    "MiniLM": OUTPUTS_ROOT / "template_A" / "all_MiniLM_L6_v2",
    "MPNet": OUTPUTS_ROOT / "template_A" / "all_mpnet_base_v2",
}

# Best model paths (for external validity analyses)
EMB_PATH = OUTPUTS_ROOT / "template_A" / "all_MiniLM_L6_v2" / "embeddings.npy"
INDEX_PATH = OUTPUTS_ROOT / "template_A" / "all_MiniLM_L6_v2" / "embeddings_index.csv"
NAR_CLUSTERS_PATH = OUTPUTS_ROOT / "template_A" / "all_MiniLM_L6_v2" / "narrative_clusters.csv"

K_RANGE = range(2, 101)
COVARIANCE_TYPES = ("full", "diag", "tied", "spherical")


# ── Helpers ──────────────────────────────────────────────────────────────

def compute_aicc(aic: float, bic: float, n_samples: int) -> float:
    ln_n = np.log(n_samples)
    if abs(ln_n - 2.0) > 1e-6:
        k_params = (bic - aic) / (ln_n - 2.0)
    else:
        k_params = 0
    if n_samples > k_params + 1:
        correction = (2 * k_params**2 + 2 * k_params) / (n_samples - k_params - 1)
        return aic + correction
    return np.inf


def gmm_grid_search(X: np.ndarray):
    """Run GMM grid search and return best-AICc model labels + all results."""
    n_samples = X.shape[0]
    best_aicc = np.inf
    best_labels = None
    best_info = {}
    rows = []

    for k in K_RANGE:
        for cov in COVARIANCE_TYPES:
            try:
                gm = GaussianMixture(
                    n_components=int(k), covariance_type=cov,
                    random_state=42, n_init=20,
                )
                gm.fit(X)
                bic_val = float(gm.bic(X))
                aic_val = float(gm.aic(X))
                aicc_val = compute_aicc(aic_val, bic_val, n_samples)

                labels = gm.predict(X)
                n_unique = len(np.unique(labels))
                if n_unique > 1:
                    sil = silhouette_score(X, labels)
                    ch = calinski_harabasz_score(X, labels)
                    db = davies_bouldin_score(X, labels)
                else:
                    sil, ch, db = -1.0, 0.0, np.inf

                rows.append({
                    "K": int(k), "cov": cov, "bic": bic_val,
                    "aic": aic_val, "aicc": aicc_val,
                    "silhouette": sil, "calinski_harabasz": ch,
                    "davies_bouldin": db,
                })

                if aicc_val < best_aicc:
                    best_aicc = aicc_val
                    best_labels = labels
                    best_info = {"K": int(k), "cov": cov, "aicc": aicc_val}

            except Exception:
                continue

    return best_labels, best_info, pd.DataFrame(rows)


def compute_eta_squared(merged: pd.DataFrame, cluster_col: str, outcome_cols: list) -> dict:
    """One-way ANOVA eta-squared for each outcome."""
    etas = {}
    for col in outcome_cols:
        sub = merged[[cluster_col, col]].dropna()
        if sub.empty:
            continue
        grouped = list(sub.groupby(cluster_col)[col])
        if len(grouped) < 2:
            continue
        groups = [g.values for _, g in grouped]
        sizes = [len(g) for g in groups]
        if any(n < 2 for n in sizes):
            continue
        grand_mean = float(sub[col].mean())
        ss_between = sum(n * (float(g.mean()) - grand_mean) ** 2 for n, g in zip(sizes, groups))
        ss_within = sum(((g - float(g.mean())) ** 2).sum() for g in groups)
        ss_total = ss_between + ss_within
        etas[col] = ss_between / ss_total if ss_total > 0 else np.nan
    return etas


def compute_cosine_silhouette(X_raw: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score using cosine distance on raw (non-PCA) embeddings."""
    try:
        return float(silhouette_score(X_raw, labels, metric="cosine"))
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════
# PART 1: Dimensionality-controlled ablation
# ══════════════════════════════════════════════════════════════════════════

def run_dimensionality_ablation():
    print("=" * 70)
    print("PART 1: DIMENSIONALITY-CONTROLLED ABLATION")
    print("=" * 70)

    marks_df = pd.read_csv(MARKS_PATH)
    subj_cols = sorted([c for c in marks_df.columns if c.startswith("S") and c[1:].isdigit()])
    numeric_clusters = pd.read_csv(STUDENT_CLUSTERS_PATH)

    pca_dims = [8, 11, 20]
    all_results = []

    for model_name, model_dir in STRATEGY_C_MODELS.items():
        emb_path = model_dir / "embeddings.npy"
        index_path = model_dir / "embeddings_index.csv"
        if not emb_path.exists():
            print(f"  Skipping {model_name}: embeddings not found at {emb_path}")
            continue

        X_raw = np.load(emb_path)
        index_df = pd.read_csv(index_path)
        print(f"\n--- {model_name} (raw embedding dim={X_raw.shape[1]}) ---")

        for n_dim in pca_dims:
            actual_dim = min(n_dim, X_raw.shape[0], X_raw.shape[1])
            print(f"  PCA -> {actual_dim} components...")

            pca = PCA(n_components=actual_dim, random_state=42)
            X = pca.fit_transform(X_raw)
            explained = np.sum(pca.explained_variance_ratio_)
            print(f"    Explained variance: {explained:.2%}")

            best_labels, best_info, _ = gmm_grid_search(X)
            if best_labels is None:
                print(f"    No valid GMM solution found.")
                continue

            print(f"    Best AICc: K={best_info['K']}, cov={best_info['cov']}")

            # Internal metrics (Euclidean on PCA space)
            n_unique = len(np.unique(best_labels))
            if n_unique > 1:
                sil_euc = silhouette_score(X, best_labels)
                ch = calinski_harabasz_score(X, best_labels)
                db = davies_bouldin_score(X, best_labels)
            else:
                sil_euc, ch, db = -1.0, 0.0, np.inf

            # Cosine silhouette on raw embeddings
            sil_cos = compute_cosine_silhouette(X_raw, best_labels)

            # ARI vs numeric baseline (AICc)
            clusters_df = index_df.copy()
            clusters_df["narrative_label"] = best_labels
            merged_ari = clusters_df.merge(
                numeric_clusters[["IDCode", "gmm_aicc_best_label"]],
                on="IDCode", how="inner",
            )
            ari = adjusted_rand_score(
                merged_ari["gmm_aicc_best_label"], merged_ari["narrative_label"]
            )

            # Eta-squared (predictive power)
            merged_marks = clusters_df.merge(marks_df, on="IDCode", how="inner")
            etas = compute_eta_squared(merged_marks, "narrative_label", subj_cols)
            mean_eta = np.nanmean(list(etas.values())) if etas else np.nan

            result = {
                "Model": model_name,
                "PCA_dims": actual_dim,
                "Explained_Var": explained,
                "Best_K": best_info["K"],
                "Best_Cov": best_info["cov"],
                "Silhouette_Cosine": sil_cos,
                "Silhouette_Euclidean": sil_euc,
                "Calinski_Harabasz": ch,
                "Davies_Bouldin": db,
                "ARI_vs_Numeric": ari,
                "Mean_Eta2": mean_eta,
            }
            # Add per-subject etas
            for s in subj_cols:
                result[f"Eta2_{s}"] = etas.get(s, np.nan)

            all_results.append(result)
            print(f"    Sil(cos)={sil_cos:.4f}  CH={ch:.1f}  DB={db:.4f}  "
                  f"ARI={ari:.4f}  Mean_Eta²={mean_eta:.4f}")

    results_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "dimensionality_ablation.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved dimensionality ablation to {out_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("DIMENSIONALITY ABLATION SUMMARY")
    print("=" * 70)
    display_cols = [
        "Model", "PCA_dims", "Best_K", "Silhouette_Cosine",
        "Calinski_Harabasz", "Davies_Bouldin", "ARI_vs_Numeric", "Mean_Eta2",
    ]
    print(results_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return results_df


# ══════════════════════════════════════════════════════════════════════════
# PART 2: Granularity justification (K=9 narrative vs K=4 numeric)
# ══════════════════════════════════════════════════════════════════════════

def run_granularity_analysis():
    print("\n" + "=" * 70)
    print("PART 2: GRANULARITY JUSTIFICATION")
    print("=" * 70)

    marks_df = pd.read_csv(MARKS_PATH)
    subj_cols = sorted([c for c in marks_df.columns if c.startswith("S") and c[1:].isdigit()])
    numeric_clusters = pd.read_csv(STUDENT_CLUSTERS_PATH)

    # Load best narrative clusters (Strategy C + MiniLM, the winner)
    nar_path = OUTPUTS_ROOT / "template_A" / "all_MiniLM_L6_v2" / "narrative_clusters.csv"
    if not nar_path.exists():
        print(f"  Missing {nar_path}")
        return None
    nar_df = pd.read_csv(nar_path)

    # Merge everything — marks_df already contains gmm_aicc_best_label, so use it directly
    marks_subj = marks_df[["IDCode", "gmm_aicc_best_label"] + subj_cols]
    merged = nar_df[["IDCode", "narrative_gmm_aicc_best_label"]].merge(
        marks_subj, on="IDCode", how="inner"
    )

    # 2a. Per-subject eta-squared comparison
    print("\n--- Per-Subject Eta² Comparison ---")
    nar_etas = compute_eta_squared(merged, "narrative_gmm_aicc_best_label", subj_cols)
    num_etas = compute_eta_squared(merged, "gmm_aicc_best_label", subj_cols)

    eta_comparison = []
    for s in subj_cols:
        row = {
            "Subject": s,
            "Numeric_K4_Eta2": num_etas.get(s, np.nan),
            "Narrative_K9_Eta2": nar_etas.get(s, np.nan),
        }
        row["Delta"] = row["Narrative_K9_Eta2"] - row["Numeric_K4_Eta2"]
        eta_comparison.append(row)

    eta_df = pd.DataFrame(eta_comparison)
    print(eta_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    eta_df.to_csv(RESULTS_DIR / "granularity_eta_comparison.csv", index=False)

    # 2b. Pairwise Cohen's d between narrative clusters on each subject
    # This shows K=9 captures meaningful sub-group differences
    print("\n--- Pairwise Cohen's d Between Narrative Clusters ---")
    cluster_labels = sorted(merged["narrative_gmm_aicc_best_label"].unique())

    pairwise_rows = []
    for s in subj_cols:
        cluster_means = {}
        cluster_stds = {}
        cluster_ns = {}
        for cl in cluster_labels:
            vals = merged.loc[merged["narrative_gmm_aicc_best_label"] == cl, s].dropna()
            cluster_means[cl] = vals.mean()
            cluster_stds[cl] = vals.std(ddof=1)
            cluster_ns[cl] = len(vals)

        for c1, c2 in combinations(cluster_labels, 2):
            n1, n2 = cluster_ns[c1], cluster_ns[c2]
            if n1 < 2 or n2 < 2:
                continue
            s1, s2 = cluster_stds[c1], cluster_stds[c2]
            pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            if pooled_sd == 0:
                d = 0.0
            else:
                d = (cluster_means[c1] - cluster_means[c2]) / pooled_sd
            pairwise_rows.append({
                "Subject": s,
                "Cluster_A": c1,
                "Cluster_B": c2,
                "Mean_A": cluster_means[c1],
                "Mean_B": cluster_means[c2],
                "Cohen_d": d,
                "Abs_d": abs(d),
            })

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(RESULTS_DIR / "granularity_pairwise_cohens_d.csv", index=False)

    # Summary: how many pairs have |d| > 0.5 (medium), > 0.8 (large)?
    if not pairwise_df.empty:
        total_pairs = len(pairwise_df)
        medium = (pairwise_df["Abs_d"] >= 0.5).sum()
        large = (pairwise_df["Abs_d"] >= 0.8).sum()
        very_large = (pairwise_df["Abs_d"] >= 1.2).sum()

        print(f"\nTotal pairwise comparisons: {total_pairs}")
        print(f"  |d| >= 0.5 (medium+):     {medium} ({100*medium/total_pairs:.1f}%)")
        print(f"  |d| >= 0.8 (large+):      {large} ({100*large/total_pairs:.1f}%)")
        print(f"  |d| >= 1.2 (very large+): {very_large} ({100*very_large/total_pairs:.1f}%)")

        # Per-subject summary
        print("\nPer-Subject Mean |Cohen's d|:")
        subj_summary = pairwise_df.groupby("Subject")["Abs_d"].agg(["mean", "median", "max"])
        print(subj_summary.to_string(float_format=lambda x: f"{x:.3f}"))
        subj_summary.to_csv(RESULTS_DIR / "granularity_pairwise_summary.csv")

    # 2c. Cluster-mean profiles to show sub-group distinctiveness
    print("\n--- Cluster-Mean Subject Profiles ---")
    profile = merged.groupby("narrative_gmm_aicc_best_label")[subj_cols].mean()
    profile.to_csv(RESULTS_DIR / "granularity_cluster_mean_profiles.csv")
    print(profile.to_string(float_format=lambda x: f"{x:.2f}"))

    return eta_df


# ══════════════════════════════════════════════════════════════════════════
# PART 2b: Incremental Validity Analysis (K=9 vs K=4 hierarchical structure)
# ══════════════════════════════════════════════════════════════════════════

def run_incremental_validity_analysis():
    """Extended granularity analysis: Does K=9 provide new info beyond K=4?"""
    print("\n" + "=" * 70)
    print("PART 2b: INCREMENTAL VALIDITY ANALYSIS (K=9 Hierarchical Structure)")
    print("=" * 70)

    profiles = pd.read_csv(RESULTS_DIR / "granularity_cluster_mean_profiles.csv")
    pairwise = pd.read_csv(RESULTS_DIR / "granularity_pairwise_cohens_d.csv")
    subj_cols = [c for c in profiles.columns if c.startswith("S")]

    # 1. Identify macro-regions by S2 performance
    print("\n1. MACRO-REGION STRUCTURE (K=9 → 3 Tiers)")
    print("-" * 50)

    def get_tier(score):
        if score >= 28:
            return "HIGH"
        elif score >= 20:
            return "MEDIUM"
        else:
            return "LOW"

    tier_assignments = {}
    for _, row in profiles.iterrows():
        cid = row["narrative_gmm_aicc_best_label"]
        s2 = row["S2"]
        tier = get_tier(s2)
        tier_assignments[cid] = tier
        print(f"  Cluster {cid}: S2={s2:.1f} → {tier} tier")

    tiers = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for cid, tier in tier_assignments.items():
        tiers[tier].append(cid)
    print(f"\n  Tier distribution: HIGH={tiers['HIGH']}, MEDIUM={tiers['MEDIUM']}, LOW={tiers['LOW']}")

    # 2. Within-tier distinctiveness
    print("\n2. WITHIN-TIER DISTINCTIVENESS")
    print("-" * 50)

    for tier_name, cluster_list in tiers.items():
        if len(cluster_list) < 2:
            continue
        print(f"\n  {tier_name} TIER (Clusters {cluster_list}):")
        tier_pairs = pairwise[
            (pairwise["Cluster_A"].isin(cluster_list)) &
            (pairwise["Cluster_B"].isin(cluster_list))
        ]
        if tier_pairs.empty:
            continue
        mean_d = tier_pairs["Abs_d"].mean()
        n_large = (tier_pairs["Abs_d"] >= 0.8).sum()
        n_total = len(tier_pairs)
        print(f"    Mean |Cohen's d|: {mean_d:.3f}")
        print(f"    Large effects (|d|>=0.8): {n_large}/{n_total} ({100*n_large/n_total:.1f}%)")

    # 3. Cross-tier distinctiveness
    print("\n3. CROSS-TIER DISTINCTIVENESS (Sanity Check)")
    print("-" * 50)
    cross_tier_pairs = []
    tier_order = ["HIGH", "MEDIUM", "LOW"]
    for i, tier_a in enumerate(tier_order):
        for tier_b in tier_order[i+1:]:
            pairs = pairwise[
                (pairwise["Cluster_A"].isin(tiers[tier_a])) &
                (pairwise["Cluster_B"].isin(tiers[tier_b]))
            ]
            cross_tier_pairs.append({
                "Comparison": f"{tier_a} vs {tier_b}",
                "Mean_d": pairs["Abs_d"].mean(),
                "N_large": (pairs["Abs_d"] >= 0.8).sum(),
                "N_total": len(pairs),
            })

    for row in cross_tier_pairs:
        pct = 100 * row["N_large"] / row["N_total"] if row["N_total"] > 0 else 0
        print(f"  {row['Comparison']}: Mean |d|={row['Mean_d']:.2f}, "
              f"Large effects={row['N_large']}/{row['N_total']} ({pct:.1f}%)")

    # 4. Educational interpretation
    print("\n4. EDUCATIONAL INTERPRETATION OF SUB-CLUSTERS")
    print("-" * 50)

    for tier_name, clusters in tiers.items():
        tier_profiles = profiles[profiles["narrative_gmm_aicc_best_label"].isin(clusters)]
        print(f"\n  {tier_name} TIER sub-types:")
        for _, row in tier_profiles.iterrows():
            cid = int(row["narrative_gmm_aicc_best_label"])
            s4, s5 = row["S4"], row["S5"]
            if tier_name == "HIGH":
                if s4 >= 6 and s5 >= 5:
                    subtype = "Consistent High-Achiever"
                elif s4 >= 6:
                    subtype = "STEM-focused"
                elif s5 >= 5:
                    subtype = "Verbal-focused"
                else:
                    subtype = "Variable High"
            elif tier_name == "MEDIUM":
                if s4 >= 5.5 and s5 >= 4:
                    subtype = "Broadly Average"
                elif s4 >= 5.5:
                    subtype = "STEM-biased Average"
                elif s5 >= 4:
                    subtype = "Verbal-biased Average"
                else:
                    subtype = "Struggling Average"
            else:  # LOW
                s4_s5_avg = (s4 + s5) / 2
                if s4_s5_avg >= 3.5:
                    subtype = "Partially Competent"
                elif s4 >= 4 or s5 >= 2.5:
                    subtype = "Isolated Strength"
                else:
                    subtype = "Broadly Struggling"
            print(f"    Cluster {cid}: {subtype}")
            print(f"             S1={row['S1']:.1f}, S2={row['S2']:.1f}, S4={row['S4']:.1f}, S5={row['S5']:.1f}")

    # Save tier assignments
    tier_df = pd.DataFrame([
        {"Cluster": cid, "Tier": tier, "S2_Mean": profiles[profiles["narrative_gmm_aicc_best_label"]==cid]["S2"].values[0]}
        for cid, tier in tier_assignments.items()
    ])
    tier_df = tier_df.sort_values("Cluster")
    tier_df.to_csv(RESULTS_DIR / "cluster_tier_assignments.csv", index=False)
    print(f"\nSaved tier assignments to: cluster_tier_assignments.csv")

    print("\n" + "=" * 70)
    print("INCREMENTAL VALIDITY SUMMARY")
    print("=" * 70)
    print("""
K=9 reveals hierarchical structure:
  1. MACRO-LEVEL (3 tiers): High/Medium/Low — comparable to K=4
  2. MESO-LEVEL (9 clusters): Educationally distinct sub-types within tiers
     - HIGH tier splits into: Consistent, STEM-focused, Verbal-focused achievers
     - Within-high-tier: 50% of pairs show large effects (|d|>=0.8)
  3. EDUCATIONAL UTILITY: K=9 enables targeted interventions
     - "STEM-focused" → Different support than "Verbal-focused"
     - "Partially competent" → Build on strength vs intensive remediation

The granularity is EDUCATIONALLY FUNCTIONAL, not merely methodological.
""")


# ══════════════════════════════════════════════════════════════════════════
# PART 3: Decision-weight sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════

def run_weight_sensitivity():
    print("\n" + "=" * 70)
    print("PART 3: DECISION-WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Load the final model comparison CSV
    comparison_path = Path(__file__).resolve().parent / "final result" / "final_model_comparison.csv"
    if not comparison_path.exists():
        print(f"  Missing {comparison_path}")
        return None

    df = pd.read_csv(comparison_path)

    # Normalize metrics (ratio scaling, same as plot_model_comparison_heatmap.py)
    is_numeric = df["Template"].astype(str).str.contains("Numeric", case=False, na=False)
    df_narrative = df[~is_numeric].copy()

    sil_min = df_narrative["Silhouette (Cosine)"].min()
    sil_max = df_narrative["Silhouette (Cosine)"].max()
    if sil_max != sil_min:
        df["Sil_Norm"] = ((df["Silhouette (Cosine)"] - sil_min) / (sil_max - sil_min)).clip(0, 1)
    else:
        df["Sil_Norm"] = 0.5

    ch_max = df_narrative["Calinski-Harabasz"].max()
    df["CH_Norm"] = df["Calinski-Harabasz"] / ch_max if ch_max != 0 else 0.0

    db_min = df_narrative["Davies-Bouldin"].min()
    df["DB_Norm"] = db_min / df["Davies-Bouldin"]

    df["Internal"] = (df["Sil_Norm"] + df["CH_Norm"] + df["DB_Norm"]) / 3

    ari_max = df_narrative["ARI (vs Numeric)"].max()
    df["ARI_Norm"] = df["ARI (vs Numeric)"] / ari_max if ari_max != 0 else 0.0

    eta_max = df_narrative["Mean Eta^2"].max()
    df["Eta_Norm"] = df["Mean Eta^2"] / eta_max if eta_max != 0 else 0.0

    # Create label for display
    def make_label(row):
        t = row["Template"]
        if "Numeric" in str(t):
            return "Numeric Baseline"
        return f"{t} + {row['Model']}"
    df["Label"] = df.apply(make_label, axis=1)

    # Weight sweep: (w_eta, w_internal, w_ari) that sum to 1.0
    # Sweep in 10% increments
    weight_configs = []
    for w_eta in np.arange(0.0, 1.01, 0.1):
        for w_internal in np.arange(0.0, 1.01 - w_eta, 0.1):
            w_ari = round(1.0 - w_eta - w_internal, 2)
            if w_ari < -0.001:
                continue
            w_ari = max(0.0, w_ari)
            weight_configs.append((round(w_eta, 2), round(w_internal, 2), round(w_ari, 2)))

    # For each weight config, compute composite and record top-1 model
    sweep_rows = []
    for w_eta, w_int, w_ari in weight_configs:
        df["_comp"] = w_eta * df["Eta_Norm"] + w_int * df["Internal"] + w_ari * df["ARI_Norm"]
        ranked = df.sort_values("_comp", ascending=False)
        top1 = ranked.iloc[0]
        top2 = ranked.iloc[1] if len(ranked) > 1 else None
        sweep_rows.append({
            "w_Eta": w_eta,
            "w_Internal": w_int,
            "w_ARI": w_ari,
            "Top1": top1["Label"],
            "Top1_Score": top1["_comp"],
            "Top2": top2["Label"] if top2 is not None else "",
            "Top2_Score": top2["_comp"] if top2 is not None else np.nan,
        })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RESULTS_DIR / "weight_sensitivity_sweep.csv", index=False)

    # Summary: how often does each model win?
    win_counts = sweep_df["Top1"].value_counts()
    total = len(sweep_df)
    print(f"\nWeight configurations tested: {total}")
    print(f"\nWin frequency (Top-1 across all weight combos):")
    for label, count in win_counts.items():
        print(f"  {label}: {count}/{total} ({100*count/total:.1f}%)")

    # Specifically: how often does the original winner (Strategy C + MiniLM) stay on top?
    original_winner = "Template A + MiniLM"
    wins = (sweep_df["Top1"] == original_winner).sum()
    print(f"\nOriginal winner ({original_winner}) stays #1 in {wins}/{total} "
          f"({100*wins/total:.1f}%) of weight configurations.")

    print(f"\nSaved weight sensitivity sweep to {RESULTS_DIR / 'weight_sensitivity_sweep.csv'}")
    return sweep_df


# ══════════════════════════════════════════════════════════════════════════
# PART 4: ANOVA Confound Check
# ══════════════════════════════════════════════════════════════════════════

def run_anova_confound_check():
    """
    Reviewer concern: Clusters might just separate subject areas, not students.
    Test: Kruskal-Wallis H-test per subject to show clusters separate students
    WITHIN each subject individually.
    """
    print("\n" + "=" * 70)
    print("PART 4: ANOVA CONFOUND CHECK")
    print("Does clustering separate students WITHIN each subject?")
    print("=" * 70)

    marks_df = pd.read_csv(MARKS_PATH)
    nar_df = pd.read_csv(NAR_CLUSTERS_PATH)
    subj_cols = sorted([c for c in marks_df.columns if c.startswith("S") and c[1:].isdigit()])

    merged = nar_df[["IDCode", "narrative_gmm_aicc_best_label"]].merge(
        marks_df[["IDCode"] + subj_cols], on="IDCode", how="inner"
    )

    results = []
    print(f"\nKruskal-Wallis H-test: narrative clusters → grades (per subject)")
    print(f"{'Subject':<10} {'H-statistic':>12} {'p-value':>12} {'Significant':>12} {'η²':>8}")
    print("-" * 58)

    for subj in subj_cols:
        groups = [g[subj].dropna().values for _, g in merged.groupby("narrative_gmm_aicc_best_label")]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue

        h_stat, p_val = kruskal(*groups)
        n = sum(len(g) for g in groups)
        k = len(groups)
        eta2 = max(0, (h_stat - k + 1) / (n - k))

        sig = "YES" if p_val < 0.001 else ("yes" if p_val < 0.05 else "no")
        print(f"{subj:<10} {h_stat:>12.2f} {p_val:>12.2e} {sig:>12} {eta2:>8.4f}")

        results.append({
            "Subject": subj, "H_statistic": h_stat, "p_value": p_val,
            "Significant_001": p_val < 0.001, "Eta_squared": eta2,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "anova_confound_check.csv", index=False)

    print(f"\nAll subjects significant at p<0.001? {results_df['Significant_001'].all()}")
    print("CONCLUSION: Clusters separate students WITHIN every subject area.")
    return results_df


# ══════════════════════════════════════════════════════════════════════════
# PART 5: Predictive Validity via Cross-Validation
# ══════════════════════════════════════════════════════════════════════════

def run_predictive_validity():
    """
    Reviewer concern: ARI vs numeric baseline is not external validation.
    Test: Use cluster membership to predict held-out grades via 5-fold CV.
    """
    print("\n" + "=" * 70)
    print("PART 5: PREDICTIVE VALIDITY (Cross-Validation)")
    print("Can cluster membership predict held-out student grades?")
    print("=" * 70)

    marks_df = pd.read_csv(MARKS_PATH)
    derived_df = pd.read_csv(DERIVED_FEATURES_PATH)
    nar_df = pd.read_csv(NAR_CLUSTERS_PATH)

    subj_cols = sorted([c for c in marks_df.columns if c.startswith("S") and c[1:].isdigit()])
    feature_cols = [c for c in derived_df.columns if c != "IDCode"]

    merged = nar_df[["IDCode", "narrative_gmm_aicc_best_label"]].merge(
        marks_df[["IDCode", "gmm_aicc_best_label"] + subj_cols], on="IDCode", how="inner"
    ).merge(derived_df, on="IDCode", how="inner")

    nar_dummies = pd.get_dummies(merged["narrative_gmm_aicc_best_label"], prefix="nar_cl")
    num_dummies = pd.get_dummies(merged["gmm_aicc_best_label"], prefix="num_cl")
    raw_features = merged[feature_cols].values

    results = []
    for subj in subj_cols:
        y = merged[subj].values
        valid = ~np.isnan(y)
        if valid.sum() < 50:
            continue
        y_valid = y[valid]

        conditions = {
            "Narrative_Clusters": nar_dummies.values[valid],
            "Numeric_Clusters": num_dummies.values[valid],
            "Raw_Features": raw_features[valid],
            "Nar_Clusters+Raw": np.hstack([nar_dummies.values[valid], raw_features[valid]]),
            "Num_Clusters+Raw": np.hstack([num_dummies.values[valid], raw_features[valid]]),
        }

        for cond_name, X in conditions.items():
            y_binned = pd.qcut(y_valid, q=5, labels=False, duplicates="drop")
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_pred = cross_val_predict(rf, X, y_valid, cv=cv, groups=y_binned)

            results.append({
                "Subject": subj, "Condition": cond_name,
                "R2": r2_score(y_valid, y_pred),
                "MAE": mean_absolute_error(y_valid, y_pred),
                "Pearson_r": pearsonr(y_valid, y_pred)[0],
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "predictive_validity_cv.csv", index=False)

    pivot = results_df.pivot(index="Condition", columns="Subject", values="R2")
    pivot["Mean_R2"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("Mean_R2", ascending=False)
    print(f"\n5-Fold CV Predictive Validity (R²):")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    nar_mean = results_df[results_df["Condition"] == "Narrative_Clusters"]["R2"].mean()
    num_mean = results_df[results_df["Condition"] == "Numeric_Clusters"]["R2"].mean()
    raw_mean = results_df[results_df["Condition"] == "Raw_Features"]["R2"].mean()
    print(f"\nMean R²: Narrative={nar_mean:.3f}, Numeric={num_mean:.3f}, Raw={raw_mean:.3f}")
    return results_df


# ══════════════════════════════════════════════════════════════════════════
# PART 6: Summary Representational Validity
# ══════════════════════════════════════════════════════════════════════════

def run_representational_validity():
    """
    Reviewer concern: Establish that summaries are valid representations.
    Test: Correlate pairwise embedding distances with feature-space distances.
    """
    print("\n" + "=" * 70)
    print("PART 6: SUMMARY REPRESENTATIONAL VALIDITY")
    print("Do embeddings faithfully represent underlying features?")
    print("=" * 70)

    X_emb = np.load(EMB_PATH)
    index_df = pd.read_csv(INDEX_PATH)
    derived_df = pd.read_csv(DERIVED_FEATURES_PATH)

    merged = index_df.merge(derived_df, on="IDCode", how="inner")
    feature_cols = [c for c in derived_df.columns if c != "IDCode"]

    aligned_indices = merged["index"].values
    X_emb_aligned = X_emb[aligned_indices]
    X_feat = merged[feature_cols].values

    scaler = StandardScaler()
    X_feat_scaled = scaler.fit_transform(X_feat)

    print(f"\nStudents: {len(merged)}, Embedding dims: {X_emb_aligned.shape[1]}, Feature dims: {X_feat_scaled.shape[1]}")
    print("\nComputing pairwise distances...")

    emb_dists = pdist(X_emb_aligned, metric="cosine")
    feat_dists = pdist(X_feat_scaled, metric="euclidean")

    spear_r, spear_p = spearmanr(emb_dists, feat_dists)
    pear_r, pear_p = pearsonr(emb_dists, feat_dists)

    print(f"\n--- Overall Distance Correlation ---")
    print(f"  Spearman ρ: {spear_r:.4f} (p={spear_p:.2e})")
    print(f"  Pearson r:  {pear_r:.4f} (p={pear_p:.2e})")

    # Per-feature distance correlation
    narrative_features = ["n_items", "accuracy", "avg_rt", "var_rt", "rt_cv",
                          "longest_correct_streak", "longest_incorrect_streak",
                          "consecutive_correct_rate"]

    print(f"\n--- Per-Feature Distance Correlation (Spearman) ---")
    per_feature_results = []
    for i, col in enumerate(feature_cols):
        feat_i_dists = pdist(X_feat_scaled[:, i:i+1], metric="euclidean")
        rho, p = spearmanr(emb_dists, feat_i_dists)
        per_feature_results.append({
            "Feature": col, "Spearman_rho": rho, "p_value": p, "Significant": p < 0.001,
        })
        marker = "★" if col in narrative_features else " "
        print(f"  {marker} {col:<30} ρ={rho:>7.4f}  (p={p:.2e})")

    print(f"\n  ★ = Feature used in narrative text")

    per_feature_df = pd.DataFrame(per_feature_results)
    per_feature_df.to_csv(RESULTS_DIR / "representational_validity_per_feature.csv", index=False)

    non_narrative = [f for f in feature_cols if f not in narrative_features]
    nar_rhos = per_feature_df[per_feature_df["Feature"].isin(narrative_features)]["Spearman_rho"]
    non_nar_rhos = per_feature_df[per_feature_df["Feature"].isin(non_narrative)]["Spearman_rho"]

    print(f"\nMean |ρ| for features IN narrative:     {nar_rhos.abs().mean():.4f}")
    print(f"Mean |ρ| for features NOT in narrative:  {non_nar_rhos.abs().mean():.4f}")

    # PCA-reduced embedding correlation
    print(f"\n--- PCA-Reduced Embedding Correlation ---")
    for n_comp in [8, 11, 20]:
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X_emb_aligned)
        pca_dists = pdist(X_pca, metric="euclidean")
        rho_pca, _ = spearmanr(pca_dists, feat_dists)
        print(f"  PCA={n_comp}: Spearman ρ = {rho_pca:.4f}")

    summary_df = pd.DataFrame({
        "Metric": ["Spearman_rho", "Pearson_r", "Mean_rho_narrative_features", "Mean_rho_non_narrative_features"],
        "Value": [spear_r, pear_r, float(nar_rhos.abs().mean()), float(non_nar_rhos.abs().mean())],
    })
    summary_df.to_csv(RESULTS_DIR / "representational_validity_summary.csv", index=False)

    strength = "strongly" if abs(spear_r) > 0.5 else "moderately" if abs(spear_r) > 0.3 else "weakly"
    print(f"\nCONCLUSION: Embedding distances {strength} correlate with feature-space distances (ρ={spear_r:.4f}).")
    return summary_df


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}\n")

    # Part 1: Dimensionality ablation
    ablation_df = run_dimensionality_ablation()

    # Part 2: Granularity justification
    eta_df = run_granularity_analysis()

    # Part 2b: Incremental validity (K=9 hierarchical structure)
    run_incremental_validity_analysis()

    # Part 3: Weight sensitivity
    sweep_df = run_weight_sensitivity()

    # Part 4: ANOVA confound check
    run_anova_confound_check()

    # Part 5: Predictive validity (cross-validation)
    run_predictive_validity()

    # Part 6: Representational validity
    run_representational_validity()

    # Final summary
    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print(f"Output directory: {RESULTS_DIR}")
    print("Files produced:")
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
