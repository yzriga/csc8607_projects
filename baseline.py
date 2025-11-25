import torch
from configs.config import get_config
from src.data_loading import get_dataloaders

# Charger les DataLoaders
config = get_config("configs/config.yaml")
_, val_loader, _, meta = get_dataloaders(config)

num_classes = meta["num_classes"]

def accuracy(preds, labels):
    return (preds == labels).float().mean().item()

# Baseline : Classe majoritaire
majority_class = 0  # peu importe car distribution = uniforme
acc_majority = 0.0

for _, labels in val_loader:
    preds = torch.full_like(labels, fill_value=majority_class)
    acc_majority += accuracy(preds, labels)

acc_majority /= len(val_loader)


# Baseline : Prédiction aléatoire uniforme
acc_random = 0.0

for _, labels in val_loader:
    preds = torch.randint(low=0, high=num_classes, size=labels.shape)
    acc_random += accuracy(preds, labels)

acc_random /= len(val_loader)

print("Baseline classe majoritaire  :", acc_majority)
print("Baseline prédiction aléatoire :", acc_random)
