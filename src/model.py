"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

from typing import Tuple
import torch
import torch.nn as nn


class PreActResidualBlock(nn.Module):
    """
    Bloc résiduel pré-activation :

    x ----> BN -> ReLU -> Conv3x3 (stride s)
            -> BN -> ReLU -> Conv3x3 (stride 1)
            + chemin court (identité ou projection 1x1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_projection: bool = False,
    ):
        super().__init__()

        # Pré-activation sur l'entrée
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        # Deuxième pré-activation
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Chemin court : identité ou projection 1x1
        if use_projection:
            self.proj = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.proj is not None:
            shortcut = self.proj(x)

        out = out + shortcut
        return out


class PreActResNetTinyImageNet(nn.Module):
    """
    Réseau convolutionnel pour Tiny ImageNet avec blocs résiduels pré-activation.

    - Entrée : image 3x64x64
    - Couche d'initialisation : Conv 3x3, 64 canaux, stride 1, padding 1
    - Stage 1 : B1 blocs (64 canaux, résolution inchangée)
    - Stage 2 : B2 blocs (C2 canaux, première conv du premier bloc stride 2)
    - Stage 3 : B3 blocs (C3 canaux, première conv du premier bloc stride 2)
    - Tête : Global Average Pooling -> Linear (C3 -> 200)
    """

    def __init__(
        self,
        num_classes: int,
        blocks_per_stage: Tuple[int, int, int],
        c2: int,
        c3: int,
    ):
        super().__init__()

        b1, b2, b3 = blocks_per_stage

        # Couche d'initialisation (stem)
        self.stem = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Stage 1 : 64 canaux, résolution inchangée
        self.stage1 = self._make_stage(
            in_channels=64,
            out_channels=64,
            num_blocks=b1,
            first_stride=1,
        )

        # Stage 2 : passage de 64 -> C2, réduction de résolution /2
        self.stage2 = self._make_stage(
            in_channels=64,
            out_channels=c2,
            num_blocks=b2,
            first_stride=2,
        )

        # Stage 3 : passage de C2 -> C3, réduction de résolution /2
        self.stage3 = self._make_stage(
            in_channels=c2,
            out_channels=c3,
            num_blocks=b3,
            first_stride=2,
        )

        # Tête : Global Average Pooling + Linear
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, num_classes)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        first_stride: int,
    ) -> nn.Sequential:
        """
        Construit un "stage" de blocs résiduels.

        - Premier bloc : stride = first_stride, projection 1x1 si
          in_channels != out_channels ou stride != 1.
        - Blocs suivants : stride = 1, chemin court identité.
        """
        blocks = []

        use_proj = (in_channels != out_channels) or (first_stride != 1)
        blocks.append(
            PreActResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=first_stride,
                use_projection=use_proj,
            )
        )

        for _ in range(1, num_blocks):
            blocks.append(
                PreActResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    use_projection=False,
                )
            )

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 3, 64, 64)
        x = self.stem(x)      # -> (B, 64, 64, 64)

        x = self.stage1(x)    # -> (B, 64, 64, 64)
        x = self.stage2(x)    # -> (B, C2, 32, 32)
        x = self.stage3(x)    # -> (B, C3, 16, 16)

        x = self.avgpool(x)   # -> (B, C3, 1, 1)
        x = torch.flatten(x, 1)  # -> (B, C3)
        logits = self.fc(x)      # -> (B, num_classes)

        return logits


def build_model(config: dict) -> nn.Module:
    """
    Fabrique le modèle à partir de la config YAML.

    On lit :
    - model.num_classes
    - model.blocks_per_stage  (liste [B1, B2, B3])
    - model.c2, model.c3
    """
    model_cfg = config["model"]

    num_classes = int(model_cfg.get("num_classes", 200))
    blocks = model_cfg.get("blocks_per_stage", [2, 2, 2])
    if len(blocks) != 3:
        raise ValueError("model.blocks_per_stage doit avoir exactement 3 valeurs [B1, B2, B3].")

    b1, b2, b3 = [int(x) for x in blocks]

    c2 = int(model_cfg.get("c2", 128))
    c3 = int(model_cfg.get("c3", 256))

    model = PreActResNetTinyImageNet(
        num_classes=num_classes,
        blocks_per_stage=(b1, b2, b3),
        c2=c2,
        c3=c3,
    )
    return model
