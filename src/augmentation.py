"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""

from torchvision import transforms

def get_augmentation_transforms(config: dict):
    """Retourne les transformations d'augmentation."""
    augment_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(
            size=64,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),
    ]

    return augment_transforms 
