"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

import os
import random
import yaml
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Initialise toutes les seeds pour produire des résultats reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = "auto") -> str:
    """Retourne 'cpu' ou 'cuda'. Si prefer='cpu', force CPU; si 'cuda', force GPU."""
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model) -> int:
    """Compte le nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """Sauvegarde une copie YAML de la configuration dans out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config_snapshot.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"[utils] Config sauvegardée dans : {path}")


def accuracy_fn(logits, labels):
    """
    Calcule l’accuracy simple pour classification multi-classe.
    logits : (B, num_classes)
    labels : (B,)
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()
