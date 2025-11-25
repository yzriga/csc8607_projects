import torch
from collections import Counter
import matplotlib.pyplot as plt

from configs.config import get_config
from src.data_loading import get_dataloaders

# Charger config + dataloaders
config = get_config("configs/config.yaml")
train_loader, val_loader, test_loader, meta = get_dataloaders(config)

# Compter les labels du train
counter = Counter()

print("Collecte des labels...")
for _, labels in train_loader:
    counter.update(labels.tolist())

# Convertir en liste ordonnée
class_ids = list(range(meta["num_classes"]))
counts = [counter[i] for i in class_ids]

# Afficher un aperçu
print("Exemples par classe (extrait):")
print(counts[:20])  # premières classes

# Histogramme
plt.figure(figsize=(12, 4))
plt.bar(class_ids, counts)
plt.xlabel("Classe")
plt.ylabel("Nombre d'images")
plt.title("Distribution des classes — Tiny ImageNet (train)")
plt.tight_layout()
plt.savefig("class_distribution.png")

print("Histogramme sauvegardé dans class_distribution.png")
