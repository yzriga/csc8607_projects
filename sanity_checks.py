import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from configs.config import get_config
from src.data_loading import get_dataloaders

# Charger DataLoaders
config = get_config("configs/config.yaml")
train_loader, val_loader, test_loader, meta = get_dataloaders(config)

# Prendre un batch
images, labels = next(iter(train_loader))

# Fonction pour dé-normaliser (pour visualisation)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denorm(img):
    return img * std + mean

# Afficher 3 images
plt.figure(figsize=(8,4))
for i in range(3):
    img = denorm(images[i]).clamp(0,1)  # dénormaliser
    plt.subplot(1, 3, i+1)
    plt.imshow(F.to_pil_image(img))
    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("sanity_examples.png")
print("Image sauvegardée dans sanity_examples.png")
