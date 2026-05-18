import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from augmented_classification.data_builder import build_augmented_train, generated_path_for_ratio
from augmented_classification.metrics import compute_metrics
from augmented_classification.run_augmented_classification import evaluate_augmented_classification


def evaluate_augmented(config_path: str | Path) -> dict[str, object]:
    return evaluate_augmented_classification(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real-only and diffusion-augmented classifiers.")
    parser.add_argument("--config", default="configs/classification.yaml")
    args = parser.parse_args()
    print(json.dumps(evaluate_augmented(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
