"""
Entraînement principal (à implémenter par l'étudiant·e).

Doit exposer un main() exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification si applicable)
- supporter le flag --overfit_small (si True, sur-apprendre sur un très petit échantillon)
"""
import argparse
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from configs.config import get_config
from src.data_loading import get_dataloaders
from src.model import build_model


def set_seed(seed: int) -> None:
    """Fixer toutes les seeds pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Pour rendre les choses un peu plus déterministes (au prix d'un peu de perf)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config_train: Dict[str, Any]) -> torch.device:
    """Choisir le device en fonction de la config."""
    requested = config_train.get("device", "auto")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = train_cfg["optimizer"]
    name = opt_cfg["name"].lower()
    lr = opt_cfg["lr"]
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    momentum = opt_cfg.get("momentum", 0.9)

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=momentum, weight_decay=weight_decay
        )
    elif name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    else:
        raise ValueError(f"Optimiseur inconnu dans la config : {name}")


def make_overfit_loader(
    train_loader: DataLoader,
    train_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
) -> DataLoader:
    """Construire un DataLoader restreint à overfit_size premiers exemples."""
    overfit_size = int(train_cfg.get("overfit_size", 32))

    base_dataset = train_loader.dataset
    n_total = len(base_dataset)
    n_used = min(overfit_size, n_total)
    indices = list(range(n_used))

    subset = Subset(base_dataset, indices)

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(dataset_cfg.get("num_workers", 4))
    shuffle = True  # on shuffle même ce petit subset

    small_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return small_loader


def train_overfit_small(
    config: Dict[str, Any],
    seed: int | None = None,
    max_epochs: int | None = None,
    max_steps: int | None = None,
) -> None:
    """Boucle d'entraînement pour M3 : overfit sur un très petit sous-ensemble."""
    # 1) Seed
    train_cfg = config["train"]
    if seed is None:
        seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    # 2) Device
    device = get_device(train_cfg)
    print(f"[M3] Utilisation du device : {device}")

    # 3) Données
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    # Remplacer train_loader par un loader restreint à overfit_size
    train_loader = make_overfit_loader(
        train_loader=train_loader,
        train_cfg=train_cfg,
        dataset_cfg=config["dataset"],
    )

    # 4) Modèle
    model = build_model(config)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[M3] Modèle construit avec {num_params} paramètres entraînables.")

    # 5) Loss / Optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)

    # 6) TensorBoard
    runs_dir = config["paths"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)

    run_name = "overfit_small_M3"
    log_dir = os.path.join(runs_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[M3] Logs TensorBoard dans : {log_dir}")

    # 7) Boucle d'entraînement
    epochs = max_epochs if max_epochs is not None else int(train_cfg.get("epochs", 30))
    global_step = 0

    model.train()
    for epoch in range(epochs):
        print(f"[M3] Epoch {epoch + 1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Logging scalars
            global_step += 1
            writer.add_scalar("train/loss", loss.item(), global_step)

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
            writer.add_scalar("train/accuracy", acc, global_step)

            if batch_idx % 10 == 0:
                print(
                    f"[M3] epoch {epoch+1} batch {batch_idx} "
                    f"loss={loss.item():.4f} acc={acc:.4f}"
                )

            if max_steps is not None and global_step >= max_steps:
                print("[M3] max_steps atteint, arrêt de l'entraînement.")
                writer.close()
                return

    writer.close()
    print("[M3] Entraînement overfit_small terminé.")

def train_full(
    config: Dict[str, Any],
    seed: int | None = None,
):
    """Entraînement complet pour M6 (train + validation + best.ckpt)."""

    # 1) Seed
    train_cfg = config["train"]
    if seed is None:
        seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    # 2) Device
    device = get_device(train_cfg)
    print(f"[M6] Device utilisé : {device}")

    # 3) Chargement des données
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    # 4) Modèle
    model = build_model(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[M6] Paramètres du modèle : {num_params}")

    # 5) Loss + Optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)

    # 6) TensorBoard
    runs_dir = config["paths"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)
    log_dir = os.path.join(runs_dir, "train_full")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[M6] Logs TensorBoard → {log_dir}")

    # 7) Entraînement
    epochs = int(train_cfg.get("epochs", 20))
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(config["paths"]["artifacts_dir"], "best.ckpt")
    os.makedirs(config["paths"]["artifacts_dir"], exist_ok=True)

    global_step = 0

    for epoch in range(epochs):
        print(f"\n[M6] Epoch {epoch+1}/{epochs}")

        # TRAIN LOOP
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

        train_loss = running_loss / len(train_loader)
        writer.add_scalar("train/loss", train_loss, epoch)

        # VAL LOOP
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)

        print(f"[M6] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | acc={val_acc:.4f}")

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"[M6] Nouveau meilleur modèle sauvegardé → {best_ckpt_path}")

    writer.close()
    print("[M6] Entraînement complet terminé.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    # Charger la config YAML avec la fonction fournie par le dépôt
    config = get_config(args.config)

    # Pour l'instant, on ne gère que le cas M3 (overfit_small)
    do_overfit = args.overfit_small or config["train"].get("overfit_small", False)

    if not do_overfit:
        print("[M6] Entraînement complet activé.")
        train_full(config=config, seed=args.seed)
        return

    train_overfit_small(
        config=config,
        seed=args.seed,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

