from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.prepare_data import feature_columns
from diffusion.diffusion_process import SpectralDiffusion
from diffusion.model import SpectralDiffusionUNet
from utils.config import load_yaml, project_root, resolve_path, set_seed


def labels_to_onehot(labels: pd.Series, class_map: dict[str, int], num_classes: int):
    import torch

    indices = labels.map(class_map).to_numpy()
    if pd.isna(indices).any():
        raise ValueError("Found labels that are missing from the class map")
    y = torch.zeros(len(indices), num_classes, dtype=torch.float32)
    y[torch.arange(len(indices)), torch.tensor(indices, dtype=torch.long)] = 1.0
    return y


def load_split_csv(path: Path, class_map: dict[str, int], num_classes: int):
    import torch

    df = pd.read_csv(path)
    columns = feature_columns(df)
    x = torch.tensor(df[columns].to_numpy(), dtype=torch.float32)
    y = labels_to_onehot(df["label"], class_map, num_classes)
    return x, y


def build_model_from_config(config: dict, spec_len: int) -> SpectralDiffusionUNet:
    model_cfg = config.get("model", {})
    return SpectralDiffusionUNet(
        spec_len=int(model_cfg.get("spectrum_length") or spec_len),
        num_classes=int(model_cfg.get("num_classes", 2)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 4])),
        time_emb_dim=int(model_cfg.get("time_embed_dim", 128)),
        label_dim=int(model_cfg.get("label_embed_dim", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        reshape_height=model_cfg.get("reshape_height"),
        reshape_width=model_cfg.get("reshape_width"),
    )


def train_diffusion(config_path: str | Path) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    root = project_root()
    config = load_yaml(config_path)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config.get("data", {})
    split_dir = resolve_path(data_cfg.get("split_dir", "data/splits"), root)
    class_map = config.get("classes", {"healthy": 0, "lung_adenocarcinoma": 1})
    num_classes = int(config.get("model", {}).get("num_classes", len(class_map)))
    x_train, y_train = load_split_csv(split_dir / "train.csv", class_map, num_classes)
    x_val, y_val = load_split_csv(split_dir / "val.csv", class_map, num_classes)

    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 10000))
    learning_rate = float(train_cfg.get("learning_rate", 1e-4))
    validate_every = int(train_cfg.get("validate_every", train_cfg.get("save_every", 100)))
    early_stopping = train_cfg.get("early_stopping_patience")

    noise_cfg = config.get("noise_schedule", {})
    diffusion = SpectralDiffusion(
        timesteps=int(noise_cfg.get("timesteps", 1000)),
        beta_start=float(noise_cfg.get("beta_start", 0.0001)),
        beta_end=float(noise_cfg.get("beta_end", 0.02)),
    ).to(device)
    model = build_model_from_config(config, spec_len=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    checkpoint_dir = resolve_path(train_cfg.get("checkpoint_dir", "outputs/checkpoints/diffusion"), root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "diffusion_best.pt"
    history_path = checkpoint_dir / "training_history.json"

    best_val = float("inf")
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            loss = diffusion.compute_loss(model, x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        if epoch % validate_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_loss = diffusion.compute_loss(model, x_val.to(device), y_val.to(device)).item()
            train_loss = train_loss / len(x_train)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "spec_len": x_train.shape[1],
                        "class_map": class_map,
                        "best_val_loss": best_val,
                        "epoch": epoch,
                    },
                    best_path,
                )
            else:
                patience += 1
            if early_stopping is not None and patience >= int(early_stopping):
                break

    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"checkpoint": str(best_path), "best_val_loss": best_val, "history": str(history_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffusion.yaml")
    args = parser.parse_args()
    summary = train_diffusion(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
