from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.prepare_data import feature_columns
from diffusion.diffusion_process import SpectralDiffusion
from diffusion.train_diffusion import build_model_from_config, labels_to_onehot
from utils.config import load_yaml, project_root, resolve_path, set_seed


def generated_count_by_class(train_df: pd.DataFrame, ratio: float) -> dict[str, int]:
    counts = train_df["label"].value_counts().to_dict()
    return {label: int(count * ratio) for label, count in counts.items()}


def load_checkpoint(path: Path, config: dict, spec_len: int, device):
    import torch

    checkpoint = torch.load(path, map_location=device)
    model_config = checkpoint.get("config", config) if isinstance(checkpoint, dict) else config
    model = build_model_from_config(model_config, spec_len=checkpoint.get("spec_len", spec_len) if isinstance(checkpoint, dict) else spec_len)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model.to(device)


def generate_spectra(config_path: str | Path, ratio: float) -> dict[str, object]:
    import torch

    root = project_root()
    config = load_yaml(config_path)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config.get("data", {})
    split_dir = resolve_path(data_cfg.get("split_dir", "data/splits"), root)
    train_df = pd.read_csv(split_dir / "train.csv")
    columns = feature_columns(train_df)
    class_map = config.get("classes", {"healthy": 0, "lung_adenocarcinoma": 1})
    num_classes = int(config.get("model", {}).get("num_classes", len(class_map)))

    train_cfg = config.get("training", {})
    checkpoint_dir = resolve_path(train_cfg.get("checkpoint_dir", "outputs/checkpoints/diffusion"), root)
    checkpoint_path = checkpoint_dir / "diffusion_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing diffusion checkpoint: {checkpoint_path}")

    model = load_checkpoint(checkpoint_path, config, len(columns), device)
    noise_cfg = config.get("noise_schedule", {})
    diffusion = SpectralDiffusion(
        timesteps=int(noise_cfg.get("timesteps", 1000)),
        beta_start=float(noise_cfg.get("beta_start", 0.0001)),
        beta_end=float(noise_cfg.get("beta_end", 0.02)),
    ).to(device)

    gen_cfg = config.get("generation", {})
    output_dir = resolve_path(gen_cfg.get("output_dir", "outputs/generated"), root)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = gen_cfg.get("sampler", "ddim")
    eta = float(gen_cfg.get("eta", 0.0))
    batch_size = int(gen_cfg.get("batch_size", 32))

    generated_frames = []
    target_counts = generated_count_by_class(train_df, ratio)
    for label_name, count in target_counts.items():
        if count <= 0:
            continue
        label_df = pd.DataFrame({"label": [label_name]})
        label_onehot = labels_to_onehot(label_df["label"], class_map, num_classes)[0].to(device)
        batches = []
        with torch.no_grad():
            for start in range(0, count, batch_size):
                size = min(batch_size, count - start)
                y = label_onehot.unsqueeze(0).expand(size, -1)
                x_gen = diffusion.sample(
                    model,
                    shape=(size, len(columns)),
                    label=y,
                    device=device,
                    sampler=sampler,
                    eta=eta,
                )
                batches.append(x_gen.cpu().numpy())
        spectra = np.concatenate(batches, axis=0)
        df = pd.DataFrame(spectra, columns=columns)
        df.insert(0, "label", label_name)
        generated_frames.append(df)

    generated = pd.concat(generated_frames, ignore_index=True)
    generated = generated.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    ratio_label = str(ratio).replace(".", "_")
    output_path = output_dir / f"raman_generated_{ratio_label}x.csv"
    generated.to_csv(output_path, index=False)
    summary = {"output": str(output_path), "counts": generated["label"].value_counts().to_dict()}
    (output_dir / f"raman_generated_{ratio_label}x.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffusion.yaml")
    parser.add_argument("--ratio", type=float, required=True)
    args = parser.parse_args()
    summary = generate_spectra(args.config, args.ratio)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
