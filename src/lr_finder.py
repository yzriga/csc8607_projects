"""
Recherche de taux d'apprentissage (LR finder) — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.lr_finder --config configs/config.yaml

Exigences minimales :
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard ou équivalent.
"""

import argparse
import math
import torch
from torch.utils.tensorboard import SummaryWriter

from configs.config import get_config
from src.data_loading import get_dataloaders
from src.model import build_model


def lr_finder(config, num_steps=200, initial_lr=1e-7, final_lr=1.0):

    # Chargement des données
    train_loader, _, _, _ = get_dataloaders(config)

    # Modèle
    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimiseur
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    # Critère
    criterion = torch.nn.CrossEntropyLoss()

    # TensorBoard
    writer = SummaryWriter("runs/lr_finder")

    # Progression exponentielle du LR
    lr_mult = (final_lr / initial_lr) ** (1 / num_steps)

    lr = initial_lr
    optimizer.param_groups[0]["lr"] = lr

    model.train()

    step = 0
    best_loss = float("inf")
    avg_loss = 0.0
    beta = 0.98  # smoothing constant

    data_iter = iter(train_loader)

    print("\n[LR Finder] Démarrage...\n")

    while step < num_steps:
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        # Smoothing EMA
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed = avg_loss / (1 - beta ** (step + 1))

        # Log TensorBoard
        writer.add_scalar("lr_finder/loss", smoothed, step)
        writer.add_scalar("lr_finder/lr", lr, step)

        # Si la loss explose → stop
        if smoothed > 10 * best_loss:
            print("[LR Finder] STOP : explosion de la perte.")
            break

        if smoothed < best_loss:
            best_loss = smoothed

        # Backward + update
        loss.backward()
        optimizer.step()

        # Update du LR (exponentiel)
        lr *= lr_mult
        optimizer.param_groups[0]["lr"] = lr

        step += 1

        # Affichage console
        if step % 20 == 0:
            print(f"[step {step}/{num_steps}] lr={lr:.6f}  smoothed_loss={smoothed:.4f}")

    writer.close()
    print("\n[LR Finder] Terminé. Voir résultats dans runs/lr_finder/\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = get_config(args.config)

    # Hyperparamètres
    NUM_STEPS = 200
    INITIAL_LR = 1e-7
    FINAL_LR = 1.0

    lr_finder(config, num_steps=NUM_STEPS,
              initial_lr=INITIAL_LR, final_lr=FINAL_LR)


if __name__ == "__main__":
    main()

