"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import accuracy_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # Charger la config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device utilisé : {device}")


    if device.type == "cuda":
        print(f"[Evaluate] GPU : {torch.cuda.get_device_name(0)}")

    print("[Evaluate] Chargement du dataset...")
    _, _, test_loader, meta = get_dataloaders(config)

    # Construire le modèle
    print("[Evaluate] Construction du modèle...")
    model = build_model(config).to(device)

    # Charger le checkpoint
    print(f"[Evaluate] Chargement du checkpoint : {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    print("[Evaluate] Évaluation en cours...")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples

    print("\nRésultats test")
    print(f"Test loss     : {avg_loss:.4f}")
    print(f"Test accuracy : {accuracy:.4f}\n")
    
    # Log TensorBoard
    writer = SummaryWriter(f"{config['paths']['runs_dir']}/evaluate")
    writer.add_scalar("test/loss", avg_loss)
    writer.add_scalar("test/accuracy", accuracy)
    writer.close()

    print("[Evaluate] Résultats loggés dans TensorBoard.")

if __name__ == "__main__":
    main()
