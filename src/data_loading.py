"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch

# Import fonctions de prétraitement et d'augmentation
from src.preprocessing import get_preprocess_transforms
from src.augmentation import get_augmentation_transforms
from sklearn.model_selection import train_test_split

class TinyImageNetTorchDataset(Dataset):
    """
    Adapter le dataset HuggingFace en Dataset PyTorch.
    """
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        img = item["image"].convert("RGB")          # PIL Image
        label = item["label"]        # int

        if self.transform is not None:
            img = self.transform(img)

        return img, label

def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    """
    # 1) Chargement du dataset HuggingFace
    dataset = load_dataset("zh-plus/tiny-imagenet")

    train_hf = dataset["train"]
    full_valid = dataset["valid"]

    indices = list(range(len(full_valid)))
    val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    val_hf  = full_valid.select(val_idx)
    test_hf = full_valid.select(test_idx)

    # 2) Transforms (définis dans preprocessing et augmentation)
    preprocess = get_preprocess_transforms(config)        # liste
    augment    = get_augmentation_transforms(config)      # liste

    # Compose :
    train_transform = transforms.Compose(augment + preprocess)
    val_transform   = transforms.Compose(preprocess)
    test_transform  = transforms.Compose(preprocess)

    # 3) PyTorch Datasets
    train_dataset = TinyImageNetTorchDataset(train_hf, train_transform)
    val_dataset   = TinyImageNetTorchDataset(val_hf,   val_transform)
    test_dataset  = TinyImageNetTorchDataset(test_hf,  test_transform)

    # 4) DataLoaders
    batch_size = config["train"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 5) Meta (utilisé pour construire la tête du modèle)
    meta = {
        "num_classes": 200,
        "input_shape": (3, 64, 64)
    }

    return train_loader, val_loader, test_loader, meta
