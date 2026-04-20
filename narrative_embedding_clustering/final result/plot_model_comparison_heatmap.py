import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Instead of calculating from scratch, load the FINAL comparison CSV which now includes Numeric
INPUT_FILE = PROJECT_ROOT / "narrative_embedding_clustering" / "final result" / "final_model_comparison.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"

def normalize_column(series, invert=False):
    """Normalize a pandas series to 0-1 range. """
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    
    if invert:
        # Lower is better (1.0)
        return (max_val - series) / (max_val - min_val)
    else:
        # Higher is better (1.0)
        return (series - min_val) / (max_val - min_val)

def plot_heatmap(df):
    if df.empty:
        print("No data to plot.")
        return

    strategy_c_mean_ari = df.loc[df["Template"].eq("Template A"), "ARI (vs Numeric)"].mean()
    if pd.notna(strategy_c_mean_ari):
        strategy_c_mean_ari = float(Decimal(str(strategy_c_mean_ari)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
        df.loc[df["Template"].astype(str).str.contains("Numeric", case=False, na=False), "ARI (vs Numeric)"] = float(strategy_c_mean_ari)

    # User Request: Rename "Template" to "Strategy"
    # The CSV has "Template", "Model", etc.
    # Convert to "Model Name" first
    
    # 1. Rename columns if needed or just handle in label creation
    # The CSV has: Template, Model, ...

    df_norm = df.copy()

    def map_template_to_strategy(template: str) -> str:
        if template == "Template A":
            return "Strategy C"
        if template == "Template C":
            return "Strategy A"
        return template.replace("Template", "Strategy")
    
    # Create Labels
    # Logic: Rename Template->Strategy. Handle Numeric Baseline.
    def create_label(row):
        if "Numeric" in row["Template"]:
            base = "Numeric Baseline"
        else:
            strategy = map_template_to_strategy(row["Template"])
            base = f"{strategy} + {row['Model']}"

        if pd.notna(row.get("Winner K")):
            return f"{base} (K={int(row['Winner K'])})"

        return base

    df_norm["Model Name"] = df_norm.apply(create_label, axis=1)
    
    # Sorting
    # Numeric First, then Strategy A->C
    df_norm['SortKey'] = df_norm['Template'].apply(lambda x: 0 if 'Numeric' in x else 1)
    df_norm['StrategyName'] = df_norm['Template'].apply(map_template_to_strategy)
    strategy_order = {"Strategy A": 1, "Strategy B": 2, "Strategy C": 3}
    df_norm['StrategySort'] = df_norm['StrategyName'].map(strategy_order).fillna(999).astype(int)
    df_norm = df_norm.sort_values(['SortKey', 'StrategySort', 'Model'], ascending=[True, True, False])
    
    df_norm = df_norm.set_index("Model Name")

    # Compute normalization stats from non-numeric models (so Numeric Baseline doesn't distort scaling)
    # NOTE: the index label includes K (e.g., "Numeric Baseline (K=4)"), so exclude via Template.
    is_numeric_baseline = df_norm["Template"].astype(str).str.contains("Numeric", case=False, na=False)
    df_stats = df_norm.loc[~is_numeric_baseline]
    
    # Normalize metrics to 0-1 score for the Decision Matrix
    # Calculate normalization stats from the filtered dataframe

    # Normalize
    # Use Ratio Scaling (x / max) for positive "higher is better" metrics to avoid zeroing out the min
    # Use Inverse Ratio (min / x) for "lower is better" (DB)
    # Use Min-Max for Silhouette (since it can be negative)
    sil_min = df_stats["Silhouette (Cosine)"].min()
    sil_max = df_stats["Silhouette (Cosine)"].max()
    if sil_max == sil_min:
        df_norm["Sil_Norm"] = pd.Series(0.5, index=df_norm.index)
    else:
        df_norm["Sil_Norm"] = (df_norm["Silhouette (Cosine)"] - sil_min) / (sil_max - sil_min)
    df_norm["Sil_Norm"] = df_norm["Sil_Norm"].clip(lower=0.0, upper=1.0)

    ch_max = df_stats["Calinski-Harabasz"].max()
    df_norm["CH_Norm"] = df_norm["Calinski-Harabasz"] / ch_max if ch_max != 0 else 0.0

    db_min = df_stats["Davies-Bouldin"].min()
    df_norm["DB_Norm"] = db_min / df_norm["Davies-Bouldin"]

    ari_max = df_stats["ARI (vs Numeric)"].max()
    df_norm["ARI_Norm"] = df_norm["ARI (vs Numeric)"] / ari_max if ari_max != 0 else 0.0

    eta_max = df_stats["Mean Eta^2"].max()
    df_norm["Eta_Norm"] = df_norm["Mean Eta^2"] / eta_max if eta_max != 0 else 0.0

    # Combine internal metrics into a single score (equal weights)
    df_norm["Internal Score"] = (
        df_norm["Sil_Norm"]
        + df_norm["CH_Norm"]
        + df_norm["DB_Norm"]
    ) / 3
    
    # Composite Score calculation
    # Weights: equal (1/3 each) across Eta, Internal, and ARI
    w_eta = w_int = w_ari = 1.0 / 3.0
    df_norm["Composite"] = (w_eta * df_norm["Eta_Norm"]) + \
                           (w_int * df_norm["Internal Score"]) + \
                           (w_ari * df_norm["ARI_Norm"])
    
    # Select columns to plot
    cols_to_plot = [
        "Internal Score",
        "ARI_Norm",
        "Eta_Norm",
        "Composite"
    ]

    # Rename for display. All four columns are normalized so "higher = better";
    # Davies-Bouldin is already inverted inside Internal Score. The ↑ arrow
    # makes that direction explicit to the reader.
    plot_data = df_norm[cols_to_plot].rename(columns={
        "Internal Score": "Internal\nScore ↑",
        "ARI_Norm": "ARI (vs Numeric)\nScore ↑",
        "Eta_Norm": "Mean Eta^2\nScore ↑",
        "Composite": "Composite ↑",
    })
    
    # Plot - Compact Size for Paper
    plt.figure(figsize=(9, 3.5))
    sns.set_context("paper", font_scale=1.0)
    
    # RdBu_r palette (reversed red-blue diverging), for visual consistency
    # across the final-result figures.
    cmap = "RdBu_r"
    
    ax = sns.heatmap(
        plot_data, annot=True, fmt=".4f", cmap=cmap, linewidths=.5,
        vmin=0.0, vmax=1.0,
        cbar_kws={'label': 'Normalized Score', 'shrink': 1.0, 'ticks': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
    )
    
    plt.title("Model Decision Matrix (Normalized Scores)", fontsize=11, fontweight='bold', pad=10)
    plt.ylabel("") # Hide "Label" label
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    internal_note = "Internal Score = mean(Silhouette_norm, Calinski-Harabasz_norm, Davies-Bouldin_norm)."
    plt.figtext(0.5, 0.01, internal_note, ha='center', fontsize=8)
    plt.tight_layout(pad=0.5, rect=[0, 0.06, 1, 1])
    
    if not FIGURES_DIR.exists():
        FIGURES_DIR.mkdir(exist_ok=True, parents=True)
        
    output_path = FIGURES_DIR / "model_decision_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")
    print(internal_note)
    
    # Save the computed scores to CSV
    scores_path = FIGURES_DIR / "model_decision_scores.csv"
    # plot_data already has the correct column names and data
    df_export = plot_data.sort_values("Composite ↑", ascending=False)
    df_export.to_csv(scores_path)
    print(f"Composite scores saved to {scores_path}")
    print("\nTop 3 Models by Composite Score:")
    print(df_export.head(3)[["Composite ↑"]])

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Please run the comparison script first.")
    else:
        df = pd.read_csv(INPUT_FILE)
        plot_heatmap(df)
