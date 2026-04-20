import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "narrative_embedding_clustering" / "final result" / "final_model_comparison.csv"
OUTPUT_FILE = PROJECT_ROOT / "narrative_embedding_clustering" / "final result" / "final_model_comparison_heatmap.png"

def normalize_column(series, invert=False):
    """Normalize a pandas series to 0-1 range. 
    If invert is True, lower values get higher scores (closer to 1)."""
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

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    strategy_c_mean_ari = df.loc[df["Template"].eq("Template A"), "ARI (vs Numeric)"].mean()
    if pd.notna(strategy_c_mean_ari):
        strategy_c_mean_ari = float(Decimal(str(strategy_c_mean_ari)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
        df.loc[df["Template"].astype(str).str.contains("Numeric", case=False, na=False), "ARI (vs Numeric)"] = float(strategy_c_mean_ari)

    def map_template_to_strategy(template: str) -> str:
        if template == "Template A":
            return "Strategy C"
        if template == "Template C":
            return "Strategy A"
        return template.replace("Template", "Strategy")
    
    # Sort: Numeric first, then by Template A->C
    # We can achieve this by a custom sort key
    df['SortKey'] = df['Template'].apply(lambda x: 0 if 'Numeric' in x else 1)
    df['StrategyName'] = df['Template'].apply(map_template_to_strategy)
    strategy_order = {"Strategy A": 1, "Strategy B": 2, "Strategy C": 3}
    df['StrategySort'] = df['StrategyName'].map(strategy_order).fillna(999).astype(int)
    df = df.sort_values(['SortKey', 'StrategySort', 'Model'], ascending=[True, True, False])

    # Create descriptive index
    # User requested "Numeric Baseline" and putting GMM-AICc in title
    # Also requested renaming "Template" to "Strategy"
    df['Label'] = df.apply(
        lambda row: (
            (map_template_to_strategy(row['Template']) + " + " + row['Model'])
            if 'Numeric' not in row['Template']
            else "Numeric Baseline"
        )
        + (f" (K={int(row['Winner K'])})" if pd.notna(row.get('Winner K')) else ""),
        axis=1
    )
    df = df.set_index('Label')
    
    # Rename columns to match desired output
    df = df.rename(columns={
        'Silhouette (Cosine)': 'Silhouette\nScore',
        'Calinski-Harabasz': 'Calinski-Harabasz\nScore',
        'Davies-Bouldin': 'Davies-Bouldin\nScore',
        'ARI (vs Numeric)': 'ARI (vs Numeric)\nScore',
        'Mean Eta^2': 'Mean Eta^2\nScore'
    })
    
    # Select numeric metrics for the heatmap
    metrics = [
        'Silhouette\nScore', 
        'Calinski-Harabasz\nScore', 
        'Davies-Bouldin\nScore', 
        'ARI (vs Numeric)\nScore', 
        'Mean Eta^2\nScore'
    ]
    
    # Create annotation dataframe (Raw Values)
    annot_df = df[metrics]
    
    # Create color dataframe (Normalized 0-1)
    color_df = pd.DataFrame(index=annot_df.index, columns=annot_df.columns)
    
    # Apply normalization logic
    # For ARI: normalize relative to best narrative model (exclude Numeric Baseline)
    is_numeric_baseline = df.index.str.contains("Numeric", case=False, na=False)
    df_narrative = df.loc[~is_numeric_baseline]
    
    for col in metrics:
        invert = ('Davies-Bouldin' in col)
        if 'ARI' in col:
            # Relative ARI normalization: exclude Numeric Baseline from stats, set baseline to 1.0
            ari_max = df_narrative[col].max()
            if ari_max != 0:
                color_df[col] = df[col] / ari_max
            else:
                color_df[col] = 0.0
        else:
            color_df[col] = normalize_column(annot_df[col], invert=invert)

    # Plotting - Compact Size for Paper (e.g., 1 column width)
    # 8 inches wide is roughly a full page width, 3.5 inches high is compact
    plt.figure(figsize=(9, 3.5)) 
    sns.set_context("paper", font_scale=1.0) # Use 'paper' context for smaller fonts
    
    # Create heatmap
    ax = sns.heatmap(
        data=color_df, 
        annot=annot_df, 
        fmt=".4f", 
        cmap="coolwarm", 
        linewidths=.5,
        cbar_kws={'label': 'Relative Performance', 'shrink': 1.0}
    )
    
    # Compact Title
    plt.title("Model Comparison (GMM-AICc)", fontsize=11, fontweight='bold', pad=10)
    plt.ylabel("") # Hide "Label" label
    
    # Rotate x-axis labels
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout(pad=0.5)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
