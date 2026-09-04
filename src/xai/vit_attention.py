import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import random_split

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))

from models.vit_model import DeepShieldViT


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

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "best_vit_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "xai"
    / "vit_attention"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMAGE TRANSFORM
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
# LOAD DATASET
# ============================================================

dataset = datasets.ImageFolder(
    root=DATASET_DIR,
    transform=transform
)

print("\nClasses:", dataset.classes)
print("Total images:", len(dataset))


# ============================================================
# SAME DATASET SPLIT AS TRAINING
# ============================================================

train_size = int(
    0.70 * len(dataset)
)

val_size = int(
    0.15 * len(dataset)
)

test_size = (
    len(dataset)
    - train_size
    - val_size
)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [
        train_size,
        val_size,
        test_size
    ],
    generator=torch.Generator().manual_seed(42)
)

print(
    "Test dataset size:",
    len(test_dataset)
)


# ============================================================
# CREATE ViT MODEL
# ============================================================

print("\nLoading Vision Transformer...")

model = DeepShieldViT(
    num_classes=2
).to(device)


# ============================================================
# LOAD TRAINED CHECKPOINT
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print(
    "Best ViT model loaded successfully!"
)


# ============================================================
# ATTENTION STORAGE
# ============================================================

attention_maps = []


# ============================================================
# CUSTOM ATTENTION FORWARD
# ============================================================

def make_attention_forward(attention_module):

    original_forward = attention_module.forward

    def forward_with_attention(
        x,
        attn_mask=None,
        is_causal=False,
        **kwargs
    ):

        # ----------------------------------------------------
        # Input shape
        # ----------------------------------------------------

        B, N, C = x.shape

        # ----------------------------------------------------
        # QKV
        # ----------------------------------------------------

        qkv = (
            attention_module.qkv(x)
            .reshape(
                B,
                N,
                3,
                attention_module.num_heads,
                C // attention_module.num_heads
            )
            .permute(
                2,
                0,
                3,
                1,
                4
            )
        )

        q, k, v = qkv.unbind(0)

        # ----------------------------------------------------
        # Scale Query
        # ----------------------------------------------------

        q = q * attention_module.scale

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        attention = (
            q @ k.transpose(-2, -1)
        )

        # ----------------------------------------------------
        # Attention mask
        # ----------------------------------------------------

        if attn_mask is not None:

            attention = attention + attn_mask

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        if is_causal:

            causal_mask = torch.triu(
                torch.ones(
                    N,
                    N,
                    device=x.device,
                    dtype=torch.bool
                ),
                diagonal=1
            )

            attention = attention.masked_fill(
                causal_mask,
                float("-inf")
            )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention = torch.softmax(
            attention,
            dim=-1
        )

        # ----------------------------------------------------
        # SAVE ATTENTION MAP
        # ----------------------------------------------------

        attention_maps.append(
            attention.detach()
        )

        # ----------------------------------------------------
        # Attention × Value
        # ----------------------------------------------------

        attention_output = (
            attention @ v
        )

        # ----------------------------------------------------
        # Rearrange
        # ----------------------------------------------------

        attention_output = (
            attention_output
            .transpose(1, 2)
            .reshape(
                B,
                N,
                C
            )
        )

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------

        attention_output = (
            attention_module.proj(
                attention_output
            )
        )

        attention_output = (
            attention_module.proj_drop(
                attention_output
            )
        )

        return attention_output

    attention_module.forward = (
        forward_with_attention
    )

    return original_forward


# ============================================================
# PATCH ALL ViT ATTENTION BLOCKS
# ============================================================

original_forwards = []

print("\nPreparing ViT attention extraction...")

for block in model.vit.blocks:

    # IMPORTANT:
    # Disable timm fused attention.
    # Otherwise the attention matrix is not accessible.

    if hasattr(
        block.attn,
        "fused_attn"
    ):

        block.attn.fused_attn = False

    original_forward = (
        make_attention_forward(
            block.attn
        )
    )

    original_forwards.append(
        (
            block.attn,
            original_forward
        )
    )


print(
    "Attention extraction ready."
)


# ============================================================
# ATTENTION ROLLOUT
# ============================================================

def attention_rollout(attentions):

    """
    Compute Attention Rollout.

    Each attention matrix:

        [batch, heads, tokens, tokens]

    ViT-B/16 with 224x224 input:

        196 image patches
        + 1 CLS token
        = 197 tokens
    """

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not attentions:

        raise RuntimeError(
            "No attention maps were captured."
        )

    result = None

    # --------------------------------------------------------
    # Process every Transformer block
    # --------------------------------------------------------

    for attention in attentions:

        # Average all attention heads
        attention = attention.mean(
            dim=1
        )

        B, N, _ = attention.shape

        # ----------------------------------------------------
        # Identity matrix
        # ----------------------------------------------------

        identity = torch.eye(
            N,
            device=attention.device
        ).unsqueeze(0)

        # ----------------------------------------------------
        # Add residual connection
        # ----------------------------------------------------

        attention = (
            attention
            + identity
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        attention = (
            attention
            / attention.sum(
                dim=-1,
                keepdim=True
            )
        )

        # ----------------------------------------------------
        # Rollout multiplication
        # ----------------------------------------------------

        if result is None:

            result = attention

        else:

            result = (
                attention @ result
            )

    # --------------------------------------------------------
    # CLS token → image patches
    # --------------------------------------------------------

    rollout = result[
        :,
        0,
        1:
    ]

    return rollout


# ============================================================
# CREATE 14 × 14 HEATMAP
# ============================================================

def create_heatmap(rollout):

    """
    ViT-B/16:

        Input = 224 × 224
        Patch = 16 × 16

        224 / 16 = 14

        14 × 14 = 196 patches
    """

    rollout = rollout.reshape(
        14,
        14
    )

    rollout = (
        rollout
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    rollout = (
        rollout
        - rollout.min()
    )

    if rollout.max() > 0:

        rollout = (
            rollout
            / rollout.max()
        )

    # --------------------------------------------------------
    # Resize to original image
    # --------------------------------------------------------

    rollout = cv2.resize(
        rollout,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    return rollout


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    original_image,
    heatmap,
    true_class,
    predicted_class,
    confidence,
    output_path
):

    # --------------------------------------------------------
    # Undo ImageNet normalization
    # --------------------------------------------------------

    image = original_image.clone()

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(
        3,
        1,
        1
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(
        3,
        1,
        1
    )

    image = (
        image * std
        + mean
    )

    image = torch.clamp(
        image,
        0,
        1
    )

    image = (
        image
        .permute(1, 2, 0)
        .numpy()
    )

    # --------------------------------------------------------
    # Convert heatmap
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = (
        cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    colored_heatmap = (
        colored_heatmap.astype(
            np.float32
        )
        / 255.0
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = (
        0.55 * image
        + 0.45 * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Original
    axes[0].imshow(
        image
    )

    axes[0].set_title(
        "Original Image"
    )

    # Attention
    axes[1].imshow(
        heatmap
    )

    axes[1].set_title(
        "ViT Attention Rollout"
    )

    # Overlay
    axes[2].imshow(
        overlay
    )

    axes[2].set_title(
        f"Prediction: {predicted_class}\n"
        f"True: {true_class}\n"
        f"Confidence: {confidence:.2f}"
    )

    # Remove axes
    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# REPRESENTATIVE EXAMPLES
# ============================================================

correct_examples = []

incorrect_examples = []

uncertain_examples = []


# ============================================================
# GENERATE EXPLANATIONS
# ============================================================

print(
    "\nGenerating ViT Attention Rollout...\n"
)


for index in range(
    len(test_dataset)
):

    image, label = (
        test_dataset[index]
    )

    # --------------------------------------------------------
    # Clear previous attention
    # --------------------------------------------------------

    attention_maps.clear()

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    input_tensor = (
        image
        .unsqueeze(0)
        .to(device)
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = (
            torch.max(
                probabilities,
                dim=1
            )
        )

    prediction = prediction.item()

    confidence = confidence.item()

    # --------------------------------------------------------
    # Verify attention was captured
    # --------------------------------------------------------

    if not attention_maps:

        raise RuntimeError(
            "Attention maps were not captured "
            "during the ViT forward pass."
        )

    # --------------------------------------------------------
    # Attention Rollout
    # --------------------------------------------------------

    rollout = attention_rollout(
        attention_maps
    )

    heatmap = create_heatmap(
        rollout[0]
    )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    true_class = dataset.classes[
        label
    ]

    predicted_class = dataset.classes[
        prediction
    ]

    # --------------------------------------------------------
    # Correct
    # --------------------------------------------------------

    if (
        prediction == label
        and len(correct_examples) < 3
    ):

        correct_examples.append(
            (
                index,
                image.clone(),
                label,
                prediction,
                confidence,
                heatmap
            )
        )

    # --------------------------------------------------------
    # Incorrect
    # --------------------------------------------------------

    if (
        prediction != label
        and len(incorrect_examples) < 3
    ):

        incorrect_examples.append(
            (
                index,
                image.clone(),
                label,
                prediction,
                confidence,
                heatmap
            )
        )

    # --------------------------------------------------------
    # Uncertain
    # --------------------------------------------------------

    if (
        confidence < 0.65
        and len(uncertain_examples) < 3
    ):

        uncertain_examples.append(
            (
                index,
                image.clone(),
                label,
                prediction,
                confidence,
                heatmap
            )
        )

    # --------------------------------------------------------
    # Stop after enough examples
    # --------------------------------------------------------

    if (
        len(correct_examples) >= 3
        and len(incorrect_examples) >= 3
        and len(uncertain_examples) >= 3
    ):

        break


# ============================================================
# SAVE EXAMPLES
# ============================================================

def save_examples(
    examples,
    category
):

    category_dir = (
        OUTPUT_DIR
        / category
    )

    category_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for count, example in enumerate(
        examples,
        start=1
    ):

        (
            index,
            image,
            label,
            prediction,
            confidence,
            heatmap
        ) = example

        true_class = dataset.classes[
            label
        ]

        predicted_class = dataset.classes[
            prediction
        ]

        output_path = (
            category_dir
            / f"{count}_"
              f"{predicted_class}_"
              f"true_{true_class}.png"
        )

        save_visualization(
            image,
            heatmap,
            true_class,
            predicted_class,
            confidence,
            output_path
        )

        print(
            f"Saved: {output_path}"
        )


# ============================================================
# SAVE CORRECT
# ============================================================

save_examples(
    correct_examples,
    "correct"
)


# ============================================================
# SAVE UNCERTAIN
# ============================================================

save_examples(
    uncertain_examples,
    "uncertain"
)


# ============================================================
# SAVE INCORRECT
# ============================================================

save_examples(
    incorrect_examples,
    "incorrect"
)


# ============================================================
# RESTORE ORIGINAL ATTENTION FORWARDS
# ============================================================

for module, original in (
    original_forwards
):

    module.forward = original


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "ViT EXPLAINABILITY COMPLETE"
)

print(
    "========================================"
)

print(
    f"Correct examples   : "
    f"{len(correct_examples)}"
)

print(
    f"Uncertain examples : "
    f"{len(uncertain_examples)}"
)

print(
    f"Incorrect examples : "
    f"{len(incorrect_examples)}"
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)