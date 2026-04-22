import re
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from scipy.stats import f as f_dist

from config import (
    MARKS_WITH_CLUSTERS_PATH,
    NARRATIVE_TEMPLATE_VERSION,
    OUTPUT_DIR,
    OUTPUT_ROOT,
    STUDENT_CLUSTERS_PATH,
    make_versioned_filename,
)


def _detect_id_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() in {"idcode", "id", "studentid", "student_id"}]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if c.lower().startswith("id"):
            return c
    raise ValueError("Could not find an ID column in marks_with_clusters.csv")


def _detect_subject_cols(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        cl = c.lower().strip()
        if re.fullmatch(r"s\d+", cl):
            cols.append(c)
    return sorted(cols, key=lambda x: int(re.findall(r"\d+", x)[0])) if cols else cols


def _zmean(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean()
    sd = df.std(ddof=0).replace(0, np.nan)
    return (df - mu) / sd


def _save_zmean_heatmap(df: pd.DataFrame, cluster_col: str, value_cols: list, out_path: Path, title: str):
    if not value_cols:
        return
        
    z = _zmean(df[value_cols])
    mat = z.join(df[cluster_col]).groupby(cluster_col)[value_cols].mean()
    # Order clusters by overall z-mean (row mean) so high-performing clusters
    # appear on top, moderate in the middle, low-performing at the bottom.
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]
    vmax = 2.0
    vmin = -2.0
    n_rows, n_cols = mat.shape
    annot_size = 6
    fig_w = min(4.2, max(3.0, (0.52 * n_cols) + 1.10))
    fig_h = min(4.6, max(3.2, (0.36 * n_rows) + 1.05))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        mat,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        cbar=True,
        cbar_kws={"label": "z-mean", "shrink": 0.9, "pad": 0.015, "aspect": 40, "ticks": [-2, -1, 0, 1, 2]},
        annot_kws={"size": annot_size, "weight": "normal"},
        linewidths=0.4,
        linecolor="#f0f0f0",
        ax=ax,
    )
    ax.set_xlabel("Subject area", fontsize=annot_size)
    ax.set_ylabel("Cluster", fontsize=annot_size)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.tick_params(axis="both", labelsize=annot_size)
    
    counts = df[cluster_col].value_counts().sort_index()
    ylabels = []
    for k in mat.index.tolist():
        ylabels.append(f"{k} (n={int(counts.get(k, 0))})")
    ax.set_yticklabels(ylabels, rotation=0)
    
    for text in ax.texts:
        try:
            val = float(text.get_text())
        except Exception:
            continue
        text.set_color("white" if abs(val) > (0.6 * vmax) else "black")
    
    ax.set_title(title.replace("\n", " "), fontsize=annot_size, fontweight="normal", pad=2)
    cbar = ax.collections[0].colorbar
    cbar.set_label("z-mean", fontsize=annot_size)
    cbar.ax.tick_params(labelsize=annot_size)
    fig.tight_layout(pad=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.005)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def _pct(count: int, total: int) -> str:
    """Return a percentage string, e.g. '72.3%'."""
    return f"{100 * count / total:.1f}%" if total > 0 else "0.0%"


def save_cluster_keywords(df: pd.DataFrame, text_col: str, cluster_col: str, top_n: int = 3) -> None:
    clusters = sorted(df[cluster_col].unique())

    PHRASES = {
        'Accuracy': [
            'high accuracy',
            'medium accuracy',
            'low accuracy',
        ],
        'Speed': [
            'responses are fast',
            'responses are moderate',
            'responses are slow',
        ],
        'Timing': [
            'stable in timing',
            'moderately variable in timing',
            'highly variable in timing',
        ],
        'Streak': [
            'short longest correct streak',
            'moderate longest correct streak',
            'long longest correct streak',
            'short longest incorrect streak',
            'moderate longest incorrect streak',
            'long longest incorrect streak',
        ],
    }

    rows = []
    report_lines = [
        "=" * 72,
        "CLUSTER KEYWORD SELECTION REPORT",
        f"Generated from column '{cluster_col}' | {len(df)} students total",
        "=" * 72,
        "",
    ]

    for cluster_label in clusters:
        sub_df = df[df[cluster_col] == cluster_label]
        combined_text = " ".join(sub_df[text_col].astype(str).tolist())
        n_students = len(sub_df)

        cluster_parts = []
        cluster_report = [
            f"CLUSTER {int(cluster_label)}  (n={n_students} students)",
            "-" * 50,
        ]

        # ── 1. Accuracy ──────────────────────────────────────────────────────
        acc_counts = {p: combined_text.count(p) for p in PHRASES['Accuracy']}
        total_acc = sum(acc_counts.values()) or 1
        high_n = acc_counts['high accuracy']
        med_n  = acc_counts['medium accuracy']
        low_n  = acc_counts['low accuracy']
        best_acc = max(acc_counts, key=acc_counts.get)

        cluster_report.append("  [Accuracy]")
        for p, c in acc_counts.items():
            cluster_report.append(f"    {p:30s}: {c:3d} students  ({_pct(c, n_students)})")

        if high_n > 0 and med_n > 0 and min(high_n, med_n) / max(high_n, med_n) >= 0.50:
            clean_acc = "With medium-to-high accuracy"
            cluster_report.append(
                f"  → SELECTED: '{clean_acc}'  "
                f"[Reason: high ({_pct(high_n, n_students)}) and medium ({_pct(med_n, n_students)}) "
                f"are within 50% of each other — mixed accuracy cluster]"
            )
        else:
            clean_acc = "With " + best_acc
            cluster_report.append(
                f"  → SELECTED: '{clean_acc}'  "
                f"[Reason: dominant label ({_pct(acc_counts[best_acc], n_students)} of students)]"
            )
        cluster_parts.append(clean_acc)

        # ── 2. Speed ─────────────────────────────────────────────────────────
        _DOMINANCE = 0.60  # winner must represent at least 60% of students to be labelled clearly
        speed_counts = {p: combined_text.count(p) for p in PHRASES['Speed']}
        best_speed = max(speed_counts, key=speed_counts.get)
        best_speed_pct = speed_counts[best_speed] / n_students

        cluster_report.append("  [Speed]")
        for p, c in speed_counts.items():
            cluster_report.append(f"    {p:30s}: {c:3d} students  ({_pct(c, n_students)})")
        if best_speed_pct >= _DOMINANCE:
            speed_label = best_speed
            speed_reason = f"highest frequency ({_pct(speed_counts[best_speed], n_students)} of students, above 60% threshold)"
        else:
            speed_label = "responses are mixed speed"
            speed_reason = (
                f"no single speed dominates (best: '{best_speed}' at "
                f"{_pct(speed_counts[best_speed], n_students)}, below 60% threshold)"
            )
        cluster_report.append(f"  → SELECTED: '{speed_label}'  [Reason: {speed_reason}]")
        cluster_parts.append(speed_label)

        # ── 3. Timing variability ─────────────────────────────────────────────
        timing_counts = {p: combined_text.count(p) for p in PHRASES['Timing']}
        best_timing = max(timing_counts, key=timing_counts.get)
        best_timing_pct = timing_counts[best_timing] / n_students

        cluster_report.append("  [Timing Variability]")
        for p, c in timing_counts.items():
            cluster_report.append(f"    {p:35s}: {c:3d} students  ({_pct(c, n_students)})")
        if best_timing_pct >= _DOMINANCE:
            timing_label = best_timing
            timing_reason = f"highest frequency ({_pct(timing_counts[best_timing], n_students)} of students, above 60% threshold)"
        else:
            timing_label = "mixed timing variability"
            timing_reason = (
                f"no single timing pattern dominates (best: '{best_timing}' at "
                f"{_pct(timing_counts[best_timing], n_students)}, below 60% threshold)"
            )
        cluster_report.append(f"  → SELECTED: '{timing_label}'  [Reason: {timing_reason}]")
        cluster_parts.append(timing_label)

        # ── 4. Streak ─────────────────────────────────────────────────────────
        streak_counts = {p: combined_text.count(p) for p in PHRASES['Streak']}
        correct_streaks  = {k: v for k, v in streak_counts.items() if 'correct streak'  in k and 'incorrect' not in k}
        incorrect_streaks = {k: v for k, v in streak_counts.items() if 'incorrect streak' in k}

        best_correct_p   = max(correct_streaks,   key=correct_streaks.get)   if correct_streaks   else None
        best_incorrect_p = max(incorrect_streaks, key=incorrect_streaks.get) if incorrect_streaks else None

        is_sig_correct   = best_correct_p   and (best_correct_p.startswith('long')   or best_correct_p.startswith('moderate'))
        is_sig_incorrect = best_incorrect_p and (best_incorrect_p.startswith('long') or best_incorrect_p.startswith('moderate'))

        if is_sig_correct and not is_sig_incorrect:
            selected_streak = best_correct_p
            streak_reason = (
                f"correct streak is significant ('{best_correct_p}', "
                f"{_pct(correct_streaks[best_correct_p], n_students)}) while incorrect streak "
                f"is short/minor ('{best_incorrect_p}', "
                f"{_pct(incorrect_streaks.get(best_incorrect_p, 0), n_students)})"
            )
        elif is_sig_incorrect and not is_sig_correct:
            selected_streak = best_incorrect_p
            streak_reason = (
                f"incorrect streak is significant ('{best_incorrect_p}', "
                f"{_pct(incorrect_streaks[best_incorrect_p], n_students)}) while correct streak "
                f"is short/minor ('{best_correct_p}', "
                f"{_pct(correct_streaks.get(best_correct_p, 0), n_students)})"
            )
        else:
            selected_streak = best_correct_p if best_correct_p else best_incorrect_p
            streak_reason = (
                f"both streaks are significant or both minor — defaulting to correct streak "
                f"('{selected_streak}', {_pct(streak_counts.get(selected_streak, 0), n_students)})"
            )

        cluster_report.append("  [Streak]")
        for p, c in streak_counts.items():
            cluster_report.append(f"    {p:40s}: {c:3d}  ({_pct(c, n_students)})")

        if selected_streak:
            clean_streak = selected_streak.replace("longest ", "")
            cluster_parts.append(clean_streak)
            cluster_report.append(f"  → SELECTED: '{clean_streak}'  [Reason: {streak_reason}]")

        # ── Final keyword string ──────────────────────────────────────────────
        keywords_str = ", ".join(cluster_parts)
        rows.append({"Cluster": cluster_label, "Keywords": keywords_str})
        cluster_report.append("")
        cluster_report.append(f"  ★ FINAL KEYWORD: \"{keywords_str}\"")
        cluster_report.append("")

        report_lines.extend(cluster_report)

    # Save CSV
    out_path = OUTPUT_DIR / make_versioned_filename("cluster_keywords_frequency_based.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved cluster keywords to {out_path}")

    # Save full report
    report_path = OUTPUT_DIR / make_versioned_filename("cluster_keywords_report.txt")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved keyword selection report to {report_path}")


def main() -> None:
    narrative_clusters_path = OUTPUT_DIR / make_versioned_filename("narrative_clusters.csv")
    narratives_path = OUTPUT_DIR / make_versioned_filename("narratives.csv")
    if not narrative_clusters_path.exists():
        raise SystemExit(f"Missing narrative_clusters.csv at {narrative_clusters_path}")
    if not narratives_path.exists():
        raise SystemExit(f"Missing narratives.csv at {narratives_path}")

    stud_clusters = pd.read_csv(STUDENT_CLUSTERS_PATH)
    nar_clusters = pd.read_csv(narrative_clusters_path)
    narratives = pd.read_csv(narratives_path)

    # Include both BIC-, AIC-, and AICc-based numeric GMM labels for comparison.
    base = stud_clusters[["IDCode", "gmm_bic_best_label", "gmm_aic_best_label", "gmm_aicc_best_label"]].merge(
        nar_clusters[["IDCode", "narrative_best_label"]], on="IDCode", how="inner"
    )
    base = base.merge(narratives[["IDCode", "narrative_text"]], on="IDCode", how="left")

    if base.empty:
        raise SystemExit("No overlapping students between numeric and narrative clustering.")

    # Extract and save keywords for narrative clusters
    save_cluster_keywords(base, "narrative_text", "narrative_best_label")

    # Overlap and ARI for BIC-based numeric GMM vs narrative GMM.
    overlap_bic = pd.crosstab(base["gmm_bic_best_label"], base["narrative_best_label"])
    overlap_bic_filename = make_versioned_filename("gmm_bic_vs_narrative_overlap.csv")
    overlap_bic_path = OUTPUT_DIR / overlap_bic_filename
    overlap_bic.to_csv(overlap_bic_path)

    ari_bic = float(adjusted_rand_score(base["gmm_bic_best_label"], base["narrative_best_label"]))

    # Overlap and ARI for AIC-based numeric GMM vs narrative GMM.
    overlap_aic = pd.crosstab(base["gmm_aic_best_label"], base["narrative_best_label"])
    overlap_aic_filename = make_versioned_filename("gmm_aic_vs_narrative_overlap.csv")
    overlap_aic_path = OUTPUT_DIR / overlap_aic_filename
    overlap_aic.to_csv(overlap_aic_path)

    ari_aic = float(adjusted_rand_score(base["gmm_aic_best_label"], base["narrative_best_label"]))

    # Overlap and ARI for AICc-based numeric GMM vs narrative GMM.
    overlap_aicc = pd.crosstab(base["gmm_aicc_best_label"], base["narrative_best_label"])
    overlap_aicc_filename = make_versioned_filename("gmm_aicc_vs_narrative_overlap.csv")
    overlap_aicc_path = OUTPUT_DIR / overlap_aicc_filename
    overlap_aicc.to_csv(overlap_aicc_path)

    ari_aicc = float(adjusted_rand_score(base["gmm_aicc_best_label"], base["narrative_best_label"]))

    metric_descriptions = {
        "adjusted_rand_index": (
            "Agreement between baseline cluster labels and narrative_best_label "
            "(1 = perfect match, 0 ≈ random)."
        ),
        "silhouette_cosine": (
            "Cluster separation in narrative embedding space using cosine distance "
            "(higher is better, max = 1)."
        ),
        "calinski_harabasz": (
            "Calinski-Harabasz index in narrative embedding space (higher is better)."
        ),
        "davies_bouldin": (
            "Davies-Bouldin index in narrative embedding space (lower is better)."
        ),
    }

    metrics_rows = [
        {
            "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
            "baseline": "gmm_bic_best_label",
            "metric": "adjusted_rand_index",
            "value": ari_bic,
            "description": metric_descriptions["adjusted_rand_index"],
        },
        {
            "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
            "baseline": "gmm_aic_best_label",
            "metric": "adjusted_rand_index",
            "value": ari_aic,
            "description": metric_descriptions["adjusted_rand_index"],
        },
        {
            "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
            "baseline": "gmm_aicc_best_label",
            "metric": "adjusted_rand_index",
            "value": ari_aicc,
            "description": metric_descriptions["adjusted_rand_index"],
        },
    ]

    # Embedding-space internal indices for each clustering (numeric BIC, numeric AIC, narrative).
    emb_filename = make_versioned_filename("embeddings.npy")
    emb_index_filename = make_versioned_filename("embeddings_index.csv")
    emb_path = OUTPUT_DIR / emb_filename
    emb_index_path = OUTPUT_DIR / emb_index_filename

    # These will be reused later to select central example profiles.
    X_all: np.ndarray | None = None
    emb_base: pd.DataFrame | None = None

    if emb_path.exists() and emb_index_path.exists():
        try:
            X_all = np.load(emb_path)
            index_df = pd.read_csv(emb_index_path)

            # Restrict to students present in both embeddings and base (i.e., have all labels).
            emb_base = index_df.merge(
                base[[
                    "IDCode",
                    "gmm_bic_best_label",
                    "gmm_aic_best_label",
                    "gmm_aicc_best_label",
                    "narrative_best_label",
                ]],
                on="IDCode",
                how="inner",
            )

            if not emb_base.empty:
                indices = emb_base["index"].to_numpy(dtype=int)
                X = X_all[indices]

                label_sets = {
                    "narrative_best_label": emb_base["narrative_best_label"].to_numpy(),
                    "gmm_bic_best_label": emb_base["gmm_bic_best_label"].to_numpy(),
                    "gmm_aic_best_label": emb_base["gmm_aic_best_label"].to_numpy(),
                    "gmm_aicc_best_label": emb_base["gmm_aicc_best_label"].to_numpy(),
                }

                for baseline_name, labels in label_sets.items():
                    label_counts = pd.Series(labels).value_counts()
                    # Need at least 2 clusters, each with at least 2 members, for stable indices.
                    if len(label_counts) < 2 or not (label_counts > 1).all():
                        continue

                    sil_val = None
                    ch_val = None
                    db_val = None

                    try:
                        sil_val = float(silhouette_score(X, labels, metric="cosine"))
                    except Exception:
                        pass

                    try:
                        ch_val = float(calinski_harabasz_score(X, labels))
                    except Exception:
                        pass

                    try:
                        db_val = float(davies_bouldin_score(X, labels))
                    except Exception:
                        pass

                    if sil_val is not None:
                        metrics_rows.append(
                            {
                                "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
                                "baseline": baseline_name,
                                "metric": "silhouette_cosine",
                                "value": sil_val,
                                "description": metric_descriptions["silhouette_cosine"],
                            }
                        )

                    if ch_val is not None:
                        metrics_rows.append(
                            {
                                "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
                                "baseline": baseline_name,
                                "metric": "calinski_harabasz",
                                "value": ch_val,
                                "description": metric_descriptions["calinski_harabasz"],
                            }
                        )

                    if db_val is not None:
                        metrics_rows.append(
                            {
                                "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
                                "baseline": baseline_name,
                                "metric": "davies_bouldin",
                                "value": db_val,
                                "description": metric_descriptions["davies_bouldin"],
                            }
                        )
        except Exception:
            # If anything goes wrong with embedding-based metrics, skip them silently
            # so that ARI results are still produced.
            X_all = None
            emb_base = None

    metrics_df = pd.DataFrame(metrics_rows)
    ari_filename = make_versioned_filename("gmm_vs_narrative_metrics.csv")
    ari_path = OUTPUT_DIR / ari_filename
    metrics_df.to_csv(ari_path, index=False)

    if MARKS_WITH_CLUSTERS_PATH.exists():
        marks_df = pd.read_csv(MARKS_WITH_CLUSTERS_PATH)
        id_col = _detect_id_col(marks_df)
        # Treat cluster-label columns from the marks file as labels, not as marks to be averaged.
        # Exclude them from numeric_cols so that the base gmm_bic_best_label column is preserved
        # and not duplicated/renamed during the merge.
        label_cols = {"gmm_bic_best_label", "narrative_best_label", "gmm_aic_best_label", "gmm_aicc_best_label"}
        numeric_cols = [
            c
            for c in marks_df.columns
            if c != id_col and c not in label_cols and pd.api.types.is_numeric_dtype(marks_df[c])
        ]
        merged = base.merge(marks_df[[id_col] + numeric_cols], left_on="IDCode", right_on=id_col, how="left")

        if numeric_cols:
            gmm_group = merged.groupby("gmm_bic_best_label")[numeric_cols].mean().reset_index()
            narrative_group = merged.groupby("narrative_best_label")[numeric_cols].mean().reset_index()

            gmm_marks_filename = make_versioned_filename("marks_by_gmm_bic.csv")
            narrative_marks_filename = make_versioned_filename("marks_by_narrative.csv")
            gmm_marks_path = OUTPUT_DIR / gmm_marks_filename
            narrative_marks_path = OUTPUT_DIR / narrative_marks_filename
            gmm_group.to_csv(gmm_marks_path, index=False)
            narrative_group.to_csv(narrative_marks_path, index=False)

            # --- Generate Subject-wise Heatmap for Narrative Clusters ---
            # Identify subject columns (s1..s6)
            subj_cols = _detect_subject_cols(merged)
            if subj_cols:
                figs_dir = OUTPUT_DIR / "figures"
                figs_dir.mkdir(parents=True, exist_ok=True)
                
                heatmap_path = figs_dir / "subject_zmean_by_narrative_cluster.png"
                try:
                    _save_zmean_heatmap(
                        merged, 
                        "narrative_best_label", 
                        subj_cols, 
                        heatmap_path, 
                        "Subject-wise z-mean by Narrative Cluster\n(Strategy C + MiniLM)"
                    )
                except Exception as e:
                    print(f"Could not save narrative heatmap: {e}")

            # One-way ANOVA and effect sizes for marks by cluster label.
            anova_rows: list[dict] = []
            cluster_labels = ("gmm_bic_best_label", "gmm_aic_best_label", "gmm_aicc_best_label", "narrative_best_label")

            for cluster_col in cluster_labels:
                if cluster_col not in merged.columns:
                    continue

                for col in numeric_cols:
                    # Skip ANOVA for the overall missing-work metric; it will still be
                    # included in the descriptive marks_by_* summaries.
                    if col.lower() == "missing_total":
                        continue

                    sub = merged[[cluster_col, col]].dropna()
                    if sub.empty:
                        continue

                    grouped = list(sub.groupby(cluster_col)[col])
                    if len(grouped) < 2:
                        continue

                    groups = [g.values for _, g in grouped]
                    sizes = [len(g) for g in groups]
                    # Require at least 2 observations per group for stability.
                    if any(n < 2 for n in sizes):
                        continue

                    n_total = sum(sizes)
                    if n_total <= len(groups):
                        continue

                    grand_mean = float(sub[col].mean())
                    ss_between = float(
                        sum(n * (float(g.mean()) - grand_mean) ** 2 for n, g in zip(sizes, groups))
                    )
                    ss_within = float(sum(((g - float(g.mean())) ** 2).sum() for g in groups))
                    df_between = len(groups) - 1
                    df_within = n_total - len(groups)
                    if df_within <= 0 or ss_within <= 0.0:
                        continue

                    ms_between = ss_between / df_between
                    ms_within = ss_within / df_within
                    f_stat = ms_between / ms_within

                    try:
                        p_value = float(1.0 - f_dist.cdf(f_stat, df_between, df_within))
                    except Exception:
                        p_value = float("nan")

                    ss_total = ss_between + ss_within
                    eta_squared = ss_between / ss_total if ss_total > 0.0 else float("nan")

                    anova_rows.append(
                        {
                            "template_version": NARRATIVE_TEMPLATE_VERSION.upper(),
                            "cluster_label": cluster_col,
                            "outcome": col,
                            "f_statistic": f_stat,
                            "df_between": df_between,
                            "df_within": df_within,
                            "p_value": p_value,
                            "eta_squared": eta_squared,
                        }
                    )

            if anova_rows:
                anova_df = pd.DataFrame(anova_rows)

                # --- Split ANOVA outputs into narrative (template-specific) and
                # numeric (template-independent) parts.
                numeric_mask = anova_df["cluster_label"].isin(
                    ["gmm_bic_best_label", "gmm_aic_best_label", "gmm_aicc_best_label"]
                )
                narrative_mask = anova_df["cluster_label"] == "narrative_best_label"

                # Narrative ANOVA: keep only narrative_best_label rows in each
                # template/model directory so these files truly reflect the
                # template-specific narrative clustering.
                if narrative_mask.any():
                    narrative_df = anova_df[narrative_mask].copy()
                    anova_filename = make_versioned_filename("marks_anova_by_cluster.csv")
                    anova_path = OUTPUT_DIR / anova_filename
                    narrative_df.to_csv(anova_path, index=False)

                # Numeric ANOVA (BIC/AIC): write a single global file under the
                # outputs/ root, marking template_version as GLOBAL so it is
                # clear these results are not tied to any particular narrative
                # template.
                if numeric_mask.any():
                    numeric_df = anova_df[numeric_mask].copy()
                    numeric_df["template_version"] = "GLOBAL"
                    numeric_anova_path = OUTPUT_ROOT / "numeric_marks_anova_by_cluster.csv"
                    numeric_df.to_csv(numeric_anova_path, index=False)

    # --- Example profiles: choose central students per narrative cluster.
    # If we have embeddings and an alignment between embeddings and labels
    # (emb_base), pick the most central students in embedding space for each
    # narrative cluster. Otherwise, fall back to taking the first 5.
    examples: list[pd.DataFrame] = []

    if X_all is not None and emb_base is not None and not emb_base.empty:
        # Map base by ID for easy lookup in the desired order.
        base_by_id = base.set_index("IDCode")

        for label_val, group in emb_base.groupby("narrative_best_label"):
            # Embedding indices for this narrative cluster.
            idx_array = group["index"].to_numpy(dtype=int)
            if idx_array.size == 0:
                continue
            X_group = X_all[idx_array]
            # Cluster centroid in embedding space.
            center = X_group.mean(axis=0)
            # Euclidean distance to the centroid.
            dists = np.linalg.norm(X_group - center, axis=1)
            order = np.argsort(dists)[:5]
            central_ids = group.iloc[order]["IDCode"].tolist()

            # Recover the corresponding rows (including narrative_text) in
            # the order of increasing distance to the centroid.
            try:
                sample = base_by_id.loc[central_ids].reset_index()
            except Exception:
                # If anything goes wrong, skip this cluster.
                continue

            examples.append(sample)
    else:
        # Fallback: simple head(5) per narrative cluster as before.
        for _, group in base.groupby("narrative_best_label"):
            sample = group.head(5).copy()
            examples.append(sample)

    if examples:
        examples_df = pd.concat(examples, ignore_index=True)
        examples_filename = make_versioned_filename("example_profiles.csv")
        examples_path = OUTPUT_DIR / examples_filename
        examples_df.to_csv(examples_path, index=False)

    print(f"Saved BIC overlap table to {overlap_bic_path}")
    print(f"Saved AIC overlap table to {overlap_aic_path}")
    print(f"Saved AICc overlap table to {overlap_aicc_path}")
    print(f"Saved ARI metrics to {ari_path}")
    if MARKS_WITH_CLUSTERS_PATH.exists():
        print("Saved marks summaries by cluster if numeric marks were found.")
    print("Saved example profiles for narrative clusters.")


if __name__ == "__main__":
    main()
