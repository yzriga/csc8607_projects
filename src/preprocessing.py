"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""

from torchvision import transforms

def get_preprocess_transforms(config: dict):
    """Retourne les transformations de pré-traitement."""
    # Valeurs standard pour ImageNet (conviennent pour Tiny ImageNet)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Résolution cible
    target_size = 64  # Tiny ImageNet = 64x64

    preprocess_transforms = [
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    return preprocess_transforms
