"""
Mini grid search — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.grid_search --config configs/config.yaml

Exigences minimales :
- lire la section 'hparams' de la config
- lancer plusieurs runs en variant les hyperparamètres
- journaliser les hparams et résultats de chaque run (ex: TensorBoard HParams ou équivalent)
"""

import argparse
import itertools
import yaml
import torch
import copy
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import accuracy_fn
from src.utils import save_config_snapshot

# 1) Fonction d’entraînement court pour une config donnée
def train_one_config(config, lr, batch_size, weight_decay, run_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # mettre à jour temporairement la config
    config["train"]["batch_size"] = batch_size
    config["train"]["optimizer"]["lr"] = lr
    config["train"]["optimizer"]["weight_decay"] = weight_decay
    config["train"]["epochs"] = 2

    # dataloaders
    train_loader, val_loader, _, meta = get_dataloaders(config)

    # modèle
    model = build_model(config).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    writer = SummaryWriter(run_dir)

    best_val_loss = float("inf")

    for epoch in range(1, config["train"]["epochs"] + 1):
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

        train_loss = running_loss / len(train_loader)
        writer.add_scalar("train/loss", train_loss, epoch)

        # validation
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
        writer.add_scalar("val/acc", val_acc, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"[GridSearch] Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")

    writer.close()
    return best_val_loss

# 2) Grid search
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Charger config.yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    save_config_snapshot(config, config["paths"]["artifacts_dir"])

    hparams = config["hparams"]

    lr_list = hparams["lr"]
    batch_list = hparams["batch_size"]
    wd_list = hparams["weight_decay"]
    block_list = hparams["blocks"]
    channel_list = hparams["channels"]

    combinations = list(itertools.product(
        lr_list,
        batch_list,
        wd_list,
        block_list,
        channel_list
    ))

    print(f"[GridSearch] Nombre de combinaisons : {len(combinations)}")
    
    results = []

    for (lr, batch_size, weight_decay, blocks, channels) in combinations:

        run_name = (
            f"lr{lr}_bs{batch_size}_wd{weight_decay}"
            f"_blocks{blocks}_ch{channels}"
        )

        run_dir = f"{config['paths']['runs_dir']}/grid/{run_name}"

        print(f"\n[GridSearch] Test config : LR={lr} | BS={batch_size} | WD={weight_decay}")
        print(f"[GridSearch] blocks = {blocks} | channels = {channels}")
        print(f"[GridSearch] Logs → {run_dir}")
        
        # mettre à jour les hyperparamètres du modèle
        config_copy = copy.deepcopy(config)
        config_copy["model"]["blocks"] = blocks
        config_copy["model"]["channels"] = channels

        val_loss = train_one_config(
            config=config_copy,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            run_dir=run_dir,
        )

        results.append((val_loss, lr, batch_size, weight_decay, blocks, channels))

    # Trier par meilleure val_loss
    results.sort(key=lambda x: x[0])

    best = results[0]
    print("\n")
    print("Meilleure configuration trouvée :")
    print(f"  val_loss = {best[0]:.4f}")
    print(f"  LR       = {best[1]}")
    print(f"  batch    = {best[2]}")
    print(f"  wd       = {best[3]}")
    print(f"  blocks  = {best[4]}")
    print(f"  channels = {best[5]}")

    
if __name__ == "__main__":
    main()
