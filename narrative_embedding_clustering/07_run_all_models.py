import os
import sys
import argparse
from pathlib import Path
import subprocess


TEMPLATES = ("A", "B", "C")
MODELS = ("all-mpnet-base-v2", "all-MiniLM-L6-v2")
SCRIPTS = (
    "01_build_narratives.py",
    "02_compute_embeddings.py",
    "03_cluster_embeddings.py",
    "04_compare_with_gmm_bic.py",
    "05_visualize_embeddings.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full narrative embedding clustering pipeline.")
    parser.add_argument("--batch_size", type=str, default="32", help="Batch size for embedding generation (default: 32)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    for template in TEMPLATES:
        for model in MODELS:
            print("==> Running pipeline for template", template, "and model", model)

            env = os.environ.copy()
            env["NARRATIVE_TEMPLATE_VERSION"] = template
            env["EMBEDDING_MODEL_NAME"] = model
            env["BATCH_SIZE"] = args.batch_size

            for script in SCRIPTS:
                script_path = base_dir / script
                print(f"[template {template} | model {model}] Running {script} with BATCH_SIZE={args.batch_size}")
                subprocess.run([sys.executable, str(script_path)], check=True, env=env)

    # Final step: Generate metrics plots (Template A/B/C comparisons)
    print("\n==> Running metrics plotting (06_plot_metrics.py)")
    plot_script = base_dir / "06_plot_metrics.py"
    subprocess.run([sys.executable, str(plot_script)], check=True)
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
