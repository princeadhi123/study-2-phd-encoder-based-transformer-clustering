import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from config import K_RANGE, NARRATIVE_TEMPLATE_VERSION, OUTPUT_DIR, make_versioned_filename


def main() -> None:
    emb_filename = make_versioned_filename("embeddings.npy")
    index_filename = make_versioned_filename("embeddings_index.csv")
    emb_path = OUTPUT_DIR / emb_filename
    index_path = OUTPUT_DIR / index_filename
    if not emb_path.exists() or not index_path.exists():
        raise SystemExit("Embeddings or index not found. Run 02_compute_embeddings.py first.")
    X_raw = np.load(emb_path)
    index_df = pd.read_csv(index_path)

    # 1. Fix Curse of Dimensionality: Apply PCA
    # Reduce to 8 components to match the 8 narrative variables used in Template C,
    # ensuring a fair apples-to-apples comparison against the 8-feature numeric baseline.
    n_components = min(8, X_raw.shape[0], X_raw.shape[1])
    print(f"Applying PCA to reduce dimensionality (target components={n_components})...")
    pca = PCA(n_components=n_components, random_state=42)
    X = pca.fit_transform(X_raw)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA reduced shape: {X.shape}, Explained Variance: {explained_var:.2%}")

    best_model_info: dict | None = None
    best_bic = np.inf
    best_labels: np.ndarray | None = None
    
    # Track best AICc model as well
    best_aicc = np.inf
    best_aicc_labels: np.ndarray | None = None

    rows: list[dict] = []

    covariance_types = ("full", "diag", "tied", "spherical")

    # 2. Stability Testing: Run multiple initializations (n_init increased)
    # We will also compute Silhouette for reference but select on BIC
    print("Running GMM grid search...")
    for k in K_RANGE:
        for cov in covariance_types:
            try:
                # n_init=20 ensures we restart 20 times and pick best log-likelihood
                gm = GaussianMixture(n_components=int(k), covariance_type=cov, random_state=42, n_init=20)
                gm.fit(X)
                n_samples = X.shape[0]
                bic_val = float(gm.bic(X))
                aic_val = float(gm.aic(X))

                # Calculate AICc (Corrected AIC)
                # k_params = (BIC - AIC) / (ln(n) - 2)
                ln_n = np.log(n_samples)
                if abs(ln_n - 2.0) > 1e-6:
                    k_params = (bic_val - aic_val) / (ln_n - 2.0)
                else:
                    k_params = 0
                
                if n_samples > k_params + 1:
                    correction = (2 * k_params**2 + 2 * k_params) / (n_samples - k_params - 1)
                    aicc_val = aic_val + correction
                else:
                    aicc_val = np.inf
                
                labels = gm.predict(X)
                if len(np.unique(labels)) > 1:
                    sil = silhouette_score(X, labels)
                    ch = calinski_harabasz_score(X, labels)
                    db = davies_bouldin_score(X, labels)
                else:
                    sil = -1.0
                    ch = None
                    db = None

                rows.append({
                    "K": int(k), 
                    "covariance_type": cov, 
                    "bic": bic_val,
                    "aic": aic_val,
                    "aicc": aicc_val,
                    "silhouette": sil,
                    "calinski_harabasz": ch,
                    "davies_bouldin": db
                })

                if bic_val < best_bic:
                    best_bic = bic_val
                    best_model_info = {"k": int(k), "covariance_type": cov}
                    best_labels = labels
                
                if aicc_val < best_aicc:
                    best_aicc = aicc_val
                    best_aicc_model_info = {"k": int(k), "covariance_type": cov}
                    best_aicc_labels = labels

            except Exception as e:
                print(f"Skipping K={k}, cov={cov} due to error: {e}")
                continue

    if not rows:
        raise SystemExit("Failed to fit any GMM models on embeddings.")

    results_df = pd.DataFrame(rows)
    model_results_filename = make_versioned_filename("model_results_narrative.csv")
    model_results_path = OUTPUT_DIR / model_results_filename
    results_df.to_csv(model_results_path, index=False)

    if best_labels is None or best_model_info is None:
        raise SystemExit("Failed to obtain a valid GMM solution for any K / covariance_type.")

    clusters_df = index_df.copy()
    clusters_df["narrative_gmm_bic_best_label"] = best_labels
    if best_aicc_labels is not None:
        clusters_df["narrative_gmm_aicc_best_label"] = best_aicc_labels
        # Set AICc as the default "best" label for downstream use
        clusters_df["narrative_best_label"] = best_aicc_labels
        print(f"Using AICc-selected model (K={best_aicc_model_info['k']}, cov={best_aicc_model_info['covariance_type']}) as primary.")
    else:
        clusters_df["narrative_gmm_aicc_best_label"] = -1
        # Fallback to BIC if AICc failed (unlikely with PCA=20)
        clusters_df["narrative_best_label"] = best_labels
        print(f"AICc model invalid, falling back to BIC model (K={best_model_info['k']}) as primary.")
    out_filename = make_versioned_filename("narrative_clusters.csv")
    out_path = OUTPUT_DIR / out_filename
    clusters_df.to_csv(out_path, index=False)

    print(
        f"Saved narrative clustering results (template {NARRATIVE_TEMPLATE_VERSION.upper()}) to {out_path}"
    )
    print(
        "Best GMM by BIC: k={k}, cov={cov}, BIC={bic:.2f}".format(
            k=best_model_info["k"], cov=best_model_info["covariance_type"], bic=best_bic
        )
    )


if __name__ == "__main__":
    main()
