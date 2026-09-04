import torch
import torch.nn as nn
import timm


class DeepShieldViT(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        # Pretrained Vision Transformer
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):

        return self.vit(x)