import math
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
    DERIVED_FEATURES_PATH,
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


def _compute_thresholds(series: pd.Series, method: str = "quantile",
                         abs_low: float | None = None, abs_high: float | None = None) -> tuple[float, float]:
    """Replicates the thresholding logic of 01_build_narratives.py so categories
    derived here match those that were embedded in the narrative text."""
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 0.0, 0.0
    if method == "absolute" and abs_low is not None and abs_high is not None:
        return abs_low, abs_high
    low = float(s.quantile(0.33))
    high = float(s.quantile(0.67))
    if math.isclose(low, high):
        high = float(s.max())
    return low, high


def _categorise(value: float, low: float, high: float,
                 low_label: str, mid_label: str, high_label: str) -> str:
    if pd.isna(value):
        return "unknown"
    if value <= low:
        return low_label
    if value >= high:
        return high_label
    return mid_label


def _build_categories(features_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-student categorical labels from numeric features using the
    same thresholds as narrative construction. Returns a frame with IDCode +
    acc_cat / speed_cat / timing_cat / streak_correct_cat / streak_incorrect_cat.
    Template-independent: these columns drive the keyword extractor regardless
    of which narrative template (A, B, or C) was used for embedding."""
    feats = features_df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if "accuracy" not in feats.columns and "total_correct" in feats.columns:
            feats["accuracy"] = feats["total_correct"] / feats["n_items"].replace(0, np.nan)
        if "rt_cv" not in feats.columns and "var_rt" in feats.columns and "avg_rt" in feats.columns:
            std_rt = np.sqrt(feats["var_rt"].clip(lower=0.0))
            feats["rt_cv"] = std_rt / feats["avg_rt"].replace(0, np.nan)
    for col in ["accuracy", "avg_rt", "var_rt", "longest_correct_streak", "longest_incorrect_streak"]:
        if col in feats.columns:
            median_val = feats[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            feats[col] = feats[col].fillna(median_val)

    acc_low, acc_high = _compute_thresholds(feats["accuracy"], method="absolute", abs_low=0.60, abs_high=0.85)
    rt_low, rt_high   = _compute_thresholds(feats["avg_rt"])
    var_low, var_high = _compute_thresholds(feats["var_rt"])
    lc_low, lc_high   = _compute_thresholds(feats["longest_correct_streak"])
    li_low, li_high   = _compute_thresholds(feats["longest_incorrect_streak"])

    feats["acc_cat"]              = feats["accuracy"].apply(lambda v: _categorise(v, acc_low, acc_high, "low", "medium", "high"))
    feats["speed_cat"]            = feats["avg_rt"].apply(lambda v: _categorise(v, rt_low, rt_high, "fast", "moderate", "slow"))
    feats["timing_cat"]           = feats["var_rt"].apply(lambda v: _categorise(v, var_low, var_high, "stable", "moderately variable", "highly variable"))
    feats["streak_correct_cat"]   = feats["longest_correct_streak"].apply(lambda v: _categorise(v, lc_low, lc_high, "short", "moderate", "long"))
    feats["streak_incorrect_cat"] = feats["longest_incorrect_streak"].apply(lambda v: _categorise(v, li_low, li_high, "short", "moderate", "long"))
    return feats[["IDCode", "acc_cat", "speed_cat", "timing_cat", "streak_correct_cat", "streak_incorrect_cat"]]


def save_cluster_keywords(df: pd.DataFrame, cluster_col: str) -> None:
    """Generate cluster keyword report from pre-computed categorical labels.

    Expected columns on df: acc_cat, speed_cat, timing_cat,
    streak_correct_cat, streak_incorrect_cat plus the cluster column.

    This function is template-independent — results depend only on the
    categorised numeric features, not on the narrative phrasing, so the
    same logic produces identical keywords for Templates A, B, and C
    (they describe the same 537 students in different words)."""
    clusters = sorted(df[cluster_col].unique())
    _GAP_THRESHOLD = 0.15

    def _counts(series: pd.Series, levels: list[str]) -> dict[str, int]:
        vc = series.value_counts().to_dict()
        return {lvl: int(vc.get(lvl, 0)) for lvl in levels}

    rows = []
    report_lines = [
        "=" * 72,
        "CLUSTER KEYWORD SELECTION REPORT",
        f"Generated from column '{cluster_col}' | {len(df)} students total",
        "=" * 72,
        "",
    ]

    for cluster_label in clusters:
        sub = df[df[cluster_col] == cluster_label]
        n_students = len(sub)
        cluster_parts: list[str] = []
        cluster_report = [
            f"CLUSTER {int(cluster_label)}  (n={n_students} students)",
            "-" * 50,
        ]

        # ── 1. Accuracy ──────────────────────────────────────────────────────
        acc = _counts(sub["acc_cat"], ["high", "medium", "low"])
        best_acc = max(acc, key=acc.get)
        cluster_report.append("  [Accuracy]")
        for lvl in ["high", "medium", "low"]:
            label = f"{lvl} accuracy"
            cluster_report.append(f"    {label:30s}: {acc[lvl]:3d} students  ({_pct(acc[lvl], n_students)})")

        if acc["high"] > 0 and acc["medium"] > 0 and min(acc["high"], acc["medium"]) / max(acc["high"], acc["medium"]) >= 0.50:
            clean_acc = "With medium-to-high accuracy"
            cluster_report.append(
                f"  → SELECTED: '{clean_acc}'  "
                f"[Reason: high ({_pct(acc['high'], n_students)}) and medium ({_pct(acc['medium'], n_students)}) "
                f"are within 50% of each other — mixed accuracy cluster]"
            )
        else:
            clean_acc = f"With {best_acc} accuracy"
            cluster_report.append(
                f"  → SELECTED: '{clean_acc}'  "
                f"[Reason: dominant label ({_pct(acc[best_acc], n_students)} of students)]"
            )
        cluster_parts.append(clean_acc)

        # ── 2. Speed ─────────────────────────────────────────────────────────
        speed = _counts(sub["speed_cat"], ["fast", "moderate", "slow"])
        cluster_report.append("  [Speed]")
        for lvl in ["fast", "moderate", "slow"]:
            label = f"responses are {lvl}"
            cluster_report.append(f"    {label:30s}: {speed[lvl]:3d} students  ({_pct(speed[lvl], n_students)})")
        sorted_v = sorted(speed.values(), reverse=True)
        top_pct = sorted_v[0] / n_students if n_students else 0.0
        sec_pct = sorted_v[1] / n_students if n_students and len(sorted_v) > 1 else 0.0
        best_speed = max(speed, key=speed.get)
        if (top_pct - sec_pct) >= _GAP_THRESHOLD:
            speed_label = f"responses are {best_speed}"
            diff = round(top_pct * 100, 1) - round(sec_pct * 100, 1)
            speed_reason = (
                f"clear winner '{speed_label}' ({_pct(speed[best_speed], n_students)}) "
                f"leads 2nd place by {diff:.1f}% pts"
            )
        else:
            speed_label = "responses are mixed speed"
            speed_reason = (
                f"top two options within {_GAP_THRESHOLD*100:.0f}pp of each other "
                f"('responses are {best_speed}' at {_pct(speed[best_speed], n_students)} vs 2nd at {sec_pct*100:.1f}%)"
            )
        cluster_report.append(f"  → SELECTED: '{speed_label}'  [Reason: {speed_reason}]")
        cluster_parts.append(speed_label)

        # ── 3. Timing variability ────────────────────────────────────────────
        timing_levels = ["stable", "moderately variable", "highly variable"]
        timing = _counts(sub["timing_cat"], timing_levels)
        cluster_report.append("  [Timing Variability]")
        for lvl in timing_levels:
            label = f"{lvl} in timing"
            cluster_report.append(f"    {label:35s}: {timing[lvl]:3d} students  ({_pct(timing[lvl], n_students)})")
        sorted_v = sorted(timing.values(), reverse=True)
        top_pct = sorted_v[0] / n_students if n_students else 0.0
        sec_pct = sorted_v[1] / n_students if n_students and len(sorted_v) > 1 else 0.0
        best_timing = max(timing, key=timing.get)
        if (top_pct - sec_pct) >= _GAP_THRESHOLD:
            timing_label = f"{best_timing} in timing"
            diff = round(top_pct * 100, 1) - round(sec_pct * 100, 1)
            timing_reason = (
                f"clear winner '{timing_label}' ({_pct(timing[best_timing], n_students)}) "
                f"leads 2nd place by {diff:.1f}% pts"
            )
        else:
            timing_label = "mixed timing variability"
            timing_reason = (
                f"top two options within {_GAP_THRESHOLD*100:.0f}pp of each other "
                f"('{best_timing} in timing' at {_pct(timing[best_timing], n_students)} vs 2nd at {sec_pct*100:.1f}%)"
            )
        cluster_report.append(f"  → SELECTED: '{timing_label}'  [Reason: {timing_reason}]")
        cluster_parts.append(timing_label)

        # ── 4. Streak ────────────────────────────────────────────────────────
        cor = _counts(sub["streak_correct_cat"],   ["short", "moderate", "long"])
        inc = _counts(sub["streak_incorrect_cat"], ["short", "moderate", "long"])
        cluster_report.append("  [Streak]")
        for lvl in ["short", "moderate", "long"]:
            label = f"{lvl} correct streak"
            cluster_report.append(f"    {label:40s}: {cor[lvl]:3d}  ({_pct(cor[lvl], n_students)})")
        for lvl in ["short", "moderate", "long"]:
            label = f"{lvl} incorrect streak"
            cluster_report.append(f"    {label:40s}: {inc[lvl]:3d}  ({_pct(inc[lvl], n_students)})")

        cor_sig_n = cor["moderate"] + cor["long"]
        inc_sig_n = inc["moderate"] + inc["long"]
        cor_sig_label = "long correct streak" if cor["long"] >= cor["moderate"] else "moderate correct streak"
        inc_sig_label = "long incorrect streak" if inc["long"] >= inc["moderate"] else "moderate incorrect streak"
        cor_sig_count = max(cor["moderate"], cor["long"])
        inc_sig_count = max(inc["moderate"], inc["long"])

        # A streak family is "significant" (moderate+long dominate) when they
        # beat short outright OR when short leads by less than _GAP_THRESHOLD —
        # i.e. the gap is too close to call "mostly short".
        cor_gap = (cor["short"] - cor_sig_n) / n_students if n_students else 0.0
        inc_gap = (inc["short"] - inc_sig_n) / n_students if n_students else 0.0
        is_sig_correct   = cor_sig_n >= cor["short"] or cor_gap < _GAP_THRESHOLD
        is_sig_incorrect = inc_sig_n >= inc["short"] or inc_gap < _GAP_THRESHOLD

        if is_sig_correct and not is_sig_incorrect:
            selected_streak = cor_sig_label
            streak_reason = (
                f"correct streak is significant (moderate+long = {_pct(cor_sig_n, n_students)} "
                f"beats short {_pct(cor['short'], n_students)}); "
                f"incorrect streaks are mostly short ({_pct(inc['short'], n_students)} vs "
                f"{_pct(inc_sig_n, n_students)} moderate+long)"
            )
        elif is_sig_incorrect and not is_sig_correct:
            selected_streak = inc_sig_label
            streak_reason = (
                f"incorrect streak is significant (moderate+long = {_pct(inc_sig_n, n_students)} "
                f"beats short {_pct(inc['short'], n_students)}); "
                f"correct streaks are mostly short ({_pct(cor['short'], n_students)} vs "
                f"{_pct(cor_sig_n, n_students)} moderate+long)"
            )
        elif is_sig_correct and is_sig_incorrect:
            if cor_sig_n >= inc_sig_n:
                selected_streak = cor_sig_label
                streak_reason = (
                    f"both streaks significant — correct wins by combined count "
                    f"(correct moderate+long = {cor_sig_n} vs incorrect moderate+long = {inc_sig_n})"
                )
            else:
                selected_streak = inc_sig_label
                streak_reason = (
                    f"both streaks significant — incorrect wins by combined count "
                    f"(incorrect moderate+long = {inc_sig_n} vs correct moderate+long = {cor_sig_n})"
                )
        else:
            selected_streak = cor_sig_label if cor_sig_count >= inc_sig_count else inc_sig_label
            streak_reason = (
                f"neither streak family is significant (correct short={_pct(cor['short'], n_students)}, "
                f"incorrect short={_pct(inc['short'], n_students)}); "
                f"reporting larger non-short bucket for context"
            )

        cluster_parts.append(selected_streak)
        cluster_report.append(f"  → SELECTED: '{selected_streak}'  [Reason: {streak_reason}]")

        # ── Final keyword string ──────────────────────────────────────────────
        keywords_str = ", ".join(cluster_parts)
        rows.append({"Cluster": cluster_label, "Keywords": keywords_str})
        cluster_report.append("")
        cluster_report.append(f"  ★ FINAL KEYWORD: \"{keywords_str}\"")
        cluster_report.append("")

        report_lines.extend(cluster_report)

    # Build summary table for end of report
    report_lines.append("")
    report_lines.append("=" * 72)
    report_lines.append("FINAL KEYWORD SUMMARY TABLE")
    report_lines.append("=" * 72)
    report_lines.append("")
    report_lines.append(f"{'Cluster':<10} {'Keywords'}")
    report_lines.append("-" * 72)
    for row in rows:
        report_lines.append(f"{int(row['Cluster']):<10} {row['Keywords']}")
    report_lines.append("=" * 72)

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

    # Build template-agnostic categorical labels from the underlying numeric
    # features so the keyword extractor works identically for Templates A, B
    # and C (they describe the same 537 students in different words).
    if DERIVED_FEATURES_PATH.exists():
        derived = pd.read_csv(DERIVED_FEATURES_PATH)
        cats = _build_categories(derived)
        base_with_cats = base.merge(cats, on="IDCode", how="left")
        save_cluster_keywords(base_with_cats, "narrative_best_label")
    else:
        print(f"[warn] Derived features not found at {DERIVED_FEATURES_PATH}; skipping keyword extraction.")

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
