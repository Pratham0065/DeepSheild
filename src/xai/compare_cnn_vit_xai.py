import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import random_split


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORT THE EXACT ORIGINAL CNN MODEL
# ============================================================

from models.cnn_model import DeepShieldCNN


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = PROJECT_ROOT / "dataset" / "images"

CNN_PATH = PROJECT_ROOT / "results" / "best_cnn_model.pth"
VIT_PATH = PROJECT_ROOT / "results" / "best_vit_model.pth"

CNN_XAI_DIR = PROJECT_ROOT / "results" / "xai" / "gradcam_cnn"

OUTPUT_DIR = PROJECT_ROOT / "results" / "xai" / "comparison"


# ============================================================
# CLASSES
# ============================================================

dataset = datasets.ImageFolder(
    root=DATASET_DIR
)

classes = dataset.classes

print("\nClasses:", classes)
print("Total images:", len(dataset))


# ============================================================
# SAME DATASET SPLIT USED DURING TRAINING
# ============================================================

train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

_, _, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print("\nTest dataset size:", len(test_dataset))


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# ORIGINAL CNN CASES
# ============================================================

cases = {
    "correct": [32, 156, 114, 87],
    "uncertain": [33, 46],
    "incorrect": [15, 51]
}


# ============================================================
# LOAD CNN
# ============================================================

print("\nLoading CNN...")

cnn = DeepShieldCNN().to(device)

cnn.load_state_dict(
    torch.load(
        CNN_PATH,
        map_location=device
    )
)

cnn.eval()

print("CNN model loaded successfully!")


# ============================================================
# CNN GRAD-CAM IMAGE LOADER
# ============================================================

def load_cnn_gradcam(index):

    path = CNN_XAI_DIR / f"gradcam_{index}.png"

    if not path.exists():
        raise FileNotFoundError(
            f"CNN Grad-CAM not found:\n{path}"
        )

    return Image.open(path).convert("RGB")


# ============================================================
# VIT MODEL
# ============================================================

class DeepShieldViT(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):

        return self.vit(x)


# ============================================================
# LOAD VIT
# ============================================================

print("\nLoading ViT...")

vit = DeepShieldViT().to(device)

vit.load_state_dict(
    torch.load(
        VIT_PATH,
        map_location=device
    )
)

vit.eval()

print("ViT model loaded successfully!")


# ============================================================
# DISABLE FUSED ATTENTION
# ============================================================

for block in vit.vit.blocks:

    if hasattr(block.attn, "fused_attn"):

        block.attn.fused_attn = False


# ============================================================
# ATTENTION STORAGE
# ============================================================

attention_maps = []


# ============================================================
# PATCH ATTENTION FOR EXPLICIT EXTRACTION
# ============================================================

def attention_forward(
    self,
    x,
    attn_mask=None,
    is_causal=False,
    **kwargs
):

    B, N, C = x.shape

    qkv = self.qkv(x)

    qkv = qkv.reshape(
        B,
        N,
        3,
        self.num_heads,
        C // self.num_heads
    )

    qkv = qkv.permute(
        2,
        0,
        3,
        1,
        4
    )

    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * self.scale

    attn = q @ k.transpose(-2, -1)

    attn = attn.softmax(dim=-1)

    attention_maps.append(
        attn.detach()
    )

    attn = self.attn_drop(attn)

    x = attn @ v

    x = x.transpose(1, 2).reshape(
        B,
        N,
        C
    )

    x = self.proj(x)

    x = self.proj_drop(x)

    return x


# ============================================================
# APPLY ATTENTION PATCH
# ============================================================

for block in vit.vit.blocks:

    block.attn.forward = (
        attention_forward.__get__(
            block.attn,
            type(block.attn)
        )
    )


# ============================================================
# VIT ATTENTION ROLLOUT
# ============================================================

def get_vit_rollout(image_tensor):

    global attention_maps

    attention_maps = []

    with torch.no_grad():

        _ = vit(
            image_tensor
        )

    rollout = torch.eye(
        attention_maps[0].size(-1),
        device=device
    )

    for attention in attention_maps:

        # Average attention heads
        attention = attention.mean(
            dim=1
        )[0]

        # Add residual connection
        attention = (
            attention
            + torch.eye(
                attention.size(0),
                device=device
            )
        )

        # Normalize
        attention = attention / attention.sum(
            dim=-1,
            keepdim=True
        )

        # Rollout
        rollout = attention @ rollout

    # CLS token → patch tokens
    mask = rollout[0, 1:]

    # 196 patches = 14 x 14
    mask = mask.reshape(
        14,
        14
    )

    mask = mask.detach().cpu().numpy()

    # Normalize
    mask = (
        mask - mask.min()
    ) / (
        mask.max() - mask.min() + 1e-8
    )

    # Resize to 224 x 224
    mask = np.array(
        Image.fromarray(
            np.uint8(mask * 255)
        ).resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )
    ) / 255.0

    return mask


# ============================================================
# CREATE VIT OVERLAY
# ============================================================

def create_overlay(
    original,
    mask
):

    original = np.asarray(
        original
    ) / 255.0

    cmap = plt.get_cmap("jet")

    heatmap = cmap(mask)[..., :3]

    overlay = (
        0.5 * original
        + 0.5 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    return overlay


# ============================================================
# PROCESS CASES
# ============================================================

print("\n========================================")
print("CNN vs ViT XAI COMPARISON")
print("========================================")


for category, indices in cases.items():

    category_dir = (
        OUTPUT_DIR / category
    )

    category_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nProcessing {category.upper()} cases..."
    )

    for index in indices:

        print(
            f"  Processing image {index}..."
        )

        # ----------------------------------------------------
        # GET TEST IMAGE
        # ----------------------------------------------------

        image_path, true_label = (
            test_dataset.dataset.samples[
                test_dataset.indices[index]
            ]
        )

        original = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # PREPROCESS FOR VIT
        # ----------------------------------------------------

        image_tensor = transform(
            original
        ).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # VIT PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = vit(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_label = (
                probabilities.argmax(
                    dim=1
                ).item()
            )

            confidence = (
                probabilities[
                    0,
                    predicted_label
                ].item()
            )

        # ----------------------------------------------------
        # VIT ATTENTION
        # ----------------------------------------------------

        mask = get_vit_rollout(
            image_tensor
        )

        overlay = create_overlay(
            original.resize((224, 224)),
            mask
        )

        # ----------------------------------------------------
        # LOAD EXISTING CNN GRAD-CAM
        # ----------------------------------------------------

        cnn_gradcam = load_cnn_gradcam(
            index
        )

        # ----------------------------------------------------
        # CREATE COMPARISON FIGURE
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4)
        )

        # Original
        axes[0].imshow(
            original
        )

        axes[0].set_title(
            f"Original\nTrue: {classes[true_label]}"
        )

        # CNN Grad-CAM
        axes[1].imshow(
            cnn_gradcam
        )

        axes[1].set_title(
            "CNN Grad-CAM"
        )

        # ViT Attention
        axes[2].imshow(
            mask,
            cmap="jet"
        )

        axes[2].set_title(
            f"ViT Attention\n"
            f"Pred: {classes[predicted_label]}"
        )

        # ViT Overlay
        axes[3].imshow(
            overlay
        )

        axes[3].set_title(
            f"ViT Overlay\n"
            f"{confidence * 100:.2f}%"
        )

        for ax in axes:

            ax.axis("off")

        plt.tight_layout()

        output_path = (
            category_dir
            / f"comparison_{index}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"  Saved: {output_path}"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("CNN vs ViT XAI COMPARISON COMPLETE")
print("========================================")

print(
    f"Results saved in:\n{OUTPUT_DIR}"
)