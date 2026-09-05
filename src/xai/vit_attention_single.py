import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make project root available for imports
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT MODEL
# ============================================================

from models.vit_model import DeepShieldViT


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = PROJECT_ROOT / "results" / "best_vit_model.pth"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "xai"
    / "vit_attention_single"
)

IMAGE_SIZE = 224

CLASS_NAMES = {
    0: "fake",
    1: "real"
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# Same normalization used during ViT evaluation
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
# GLOBAL ATTENTION STORAGE
# ============================================================

attention_maps = []


# ============================================================
# ATTENTION CAPTURE FUNCTION
# ============================================================

def make_attention_capture(attention_module):

    def forward_with_attention(
        x,
        attn_mask=None,
        is_causal=False,
        **kwargs
    ):
        """
        Custom attention forward pass.

        This reproduces the standard timm ViT attention
        calculation while additionally storing the attention
        matrix for attention rollout.
        """

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
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv.unbind(0)

        # ----------------------------------------------------
        # Scale Query
        # ----------------------------------------------------

        q = q * attention_module.scale

        # ----------------------------------------------------
        # Attention Scores
        # ----------------------------------------------------

        attn = q @ k.transpose(-2, -1)

        # ----------------------------------------------------
        # Attention Mask
        # ----------------------------------------------------

        if attn_mask is not None:
            attn = attn + attn_mask

        # ----------------------------------------------------
        # Causal Mask
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

            attn = attn.masked_fill(
                causal_mask,
                float("-inf")
            )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attn = torch.softmax(
            attn,
            dim=-1
        )

        # ----------------------------------------------------
        # SAVE ATTENTION
        # ----------------------------------------------------

        attention_maps.append(
            attn.detach()
        )

        # ----------------------------------------------------
        # Attention Output
        # ----------------------------------------------------

        x = attn @ v

        x = (
            x
            .transpose(1, 2)
            .reshape(B, N, C)
        )

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------

        x = attention_module.proj(x)

        x = attention_module.proj_drop(x)

        return x

    return forward_with_attention


# ============================================================
# PATCH ViT ATTENTION
# ============================================================

def patch_attention_modules(model):

    for block in model.vit.blocks:

        # Disable fused attention because we need access
        # to the actual attention matrix.
        block.attn.fused_attn = False

        block.attn.forward = make_attention_capture(
            block.attn
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 60)
    print("LOADING VISION TRANSFORMER")
    print("=" * 60)

    print()
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print()
    print(f"Model: {MODEL_PATH}")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel checkpoint not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = DeepShieldViT(
        num_classes=2
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Handle checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)

    model.eval()

    print()
    print("ViT loaded successfully!")

    # --------------------------------------------------------
    # Patch attention modules
    # --------------------------------------------------------

    patch_attention_modules(model)

    print(
        f"Attention capture enabled for "
        f"{len(model.vit.blocks)} transformer blocks."
    )

    return model


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"\nImage not found:\n{image_path}"
        )

    print()
    print("=" * 60)
    print("LOADING IMAGE")
    print("=" * 60)

    print()
    print(f"Image: {image_path}")

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original_image = np.array(
        image
    )

    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    input_tensor = transform(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        DEVICE
    )

    print(
        f"Original size: "
        f"{image.width} x {image.height}"
    )

    print(
        f"Model input: "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    return (
        image,
        original_image,
        input_tensor
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(model, input_tensor):

    print()
    print("=" * 60)
    print("VIT PREDICTION")
    print("=" * 60)

    # Clear any old attention
    attention_maps.clear()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    fake_probability = (
        probabilities[0, 0]
        .item()
        * 100
    )

    real_probability = (
        probabilities[0, 1]
        .item()
        * 100
    )

    predicted_label = CLASS_NAMES[
        predicted_class
    ]

    confidence = (
        probabilities[0, predicted_class]
        .item()
        * 100
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()

    print(
        f"FAKE probability : "
        f"{fake_probability:.2f}%"
    )

    print(
        f"REAL probability : "
        f"{real_probability:.2f}%"
    )

    print()

    print(
        f"Prediction       : "
        f"{predicted_label.upper()}"
    )

    print(
        f"Confidence       : "
        f"{confidence:.2f}%"
    )

    print()

    print(
        f"Captured attention maps: "
        f"{len(attention_maps)}"
    )

    return (
        predicted_class,
        predicted_label,
        confidence,
        fake_probability,
        real_probability
    )


# ============================================================
# ATTENTION ROLLOUT
# ============================================================

def attention_rollout():

    print()
    print("=" * 60)
    print("GENERATING ATTENTION ROLLOUT")
    print("=" * 60)

    if len(attention_maps) == 0:

        raise RuntimeError(
            "No attention maps were captured."
        )

    # --------------------------------------------------------
    # Start with identity matrix
    # --------------------------------------------------------

    num_tokens = (
        attention_maps[0]
        .shape[-1]
    )

    device = attention_maps[0].device

    rollout = torch.eye(
        num_tokens,
        device=device
    )

    # --------------------------------------------------------
    # Process each transformer block
    # --------------------------------------------------------

    for layer_index, attention in enumerate(
        attention_maps
    ):

        # ----------------------------------------------------
        # Average attention across heads
        # ----------------------------------------------------

        attention = attention.mean(
            dim=1
        )[0]

        # ----------------------------------------------------
        # Add residual connection
        # ----------------------------------------------------

        identity = torch.eye(
            num_tokens,
            device=device
        )

        attention = (
            attention + identity
        )

        # ----------------------------------------------------
        # Normalize rows
        # ----------------------------------------------------

        attention = attention / (
            attention.sum(
                dim=-1,
                keepdim=True
            ) + 1e-8
        )

        # ----------------------------------------------------
        # Accumulate attention
        # ----------------------------------------------------

        rollout = (
            attention @ rollout
        )

    # --------------------------------------------------------
    # CLS token attention to image patches
    # --------------------------------------------------------

    cls_attention = rollout[
        0,
        1:
    ]

    # --------------------------------------------------------
    # Number of image patches
    # --------------------------------------------------------

    num_patches = cls_attention.shape[0]

    grid_size = int(
        np.sqrt(num_patches)
    )

    if grid_size * grid_size != num_patches:

        raise RuntimeError(
            f"Unexpected number of patches: "
            f"{num_patches}"
        )

    # --------------------------------------------------------
    # Convert to 2D map
    # --------------------------------------------------------

    attention_map = (
        cls_attention
        .reshape(
            grid_size,
            grid_size
        )
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    attention_map = (
        attention_map
        - attention_map.min()
    )

    max_value = attention_map.max()

    if max_value > 0:

        attention_map = (
            attention_map
            / max_value
        )

    print()
    print(
        f"Attention tokens : "
        f"{num_tokens}"
    )

    print(
        f"Image patches    : "
        f"{num_patches}"
    )

    print(
        f"Attention grid   : "
        f"{grid_size} x {grid_size}"
    )

    print()
    print("Attention rollout generated successfully.")

    return attention_map


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    attention_map,
    original_image
):

    # --------------------------------------------------------
    # Resize attention map to image size
    # --------------------------------------------------------

    heatmap = cv2.resize(
        attention_map,
        (
            original_image.shape[1],
            original_image.shape[0]
        ),
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # Convert to 0-255
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # --------------------------------------------------------
    # Apply color map
    # --------------------------------------------------------

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # OpenCV uses BGR
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    original_uint8 = np.uint8(
        original_image
    )

    overlay = cv2.addWeighted(
        original_uint8,
        0.55,
        heatmap_color,
        0.45,
        0
    )

    return (
        heatmap,
        heatmap_color,
        overlay
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    image_path,
    original_image,
    heatmap,
    heatmap_color,
    overlay,
    predicted_label,
    confidence,
    fake_probability,
    real_probability
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_name = Path(
        image_path
    ).stem

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    heatmap_path = (
        OUTPUT_DIR
        / f"{image_name}_attention_heatmap.png"
    )

    overlay_path = (
        OUTPUT_DIR
        / f"{image_name}_attention_overlay.png"
    )

    combined_path = (
        OUTPUT_DIR
        / f"{image_name}_attention_result.png"
    )

    # --------------------------------------------------------
    # Save heatmap
    # --------------------------------------------------------

    cv2.imwrite(
        str(heatmap_path),
        cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_RGB2BGR
        )
    )

    # --------------------------------------------------------
    # Save overlay
    # --------------------------------------------------------

    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    # --------------------------------------------------------
    # Combined visualization
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(15, 5)
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    ax1 = fig.add_subplot(
        1,
        3,
        1
    )

    ax1.imshow(
        original_image
    )

    ax1.set_title(
        "Original Image"
    )

    ax1.axis("off")

    # --------------------------------------------------------
    # Attention heatmap
    # --------------------------------------------------------

    ax2 = fig.add_subplot(
        1,
        3,
        2
    )

    ax2.imshow(
        heatmap,
        cmap="jet"
    )

    ax2.set_title(
        "ViT Attention Rollout"
    )

    ax2.axis("off")

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    ax3 = fig.add_subplot(
        1,
        3,
        3
    )

    ax3.imshow(
        overlay
    )

    ax3.set_title(
        f"Prediction: {predicted_label.upper()}\n"
        f"Confidence: {confidence:.2f}%"
    )

    ax3.axis("off")

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        f"DeepS ViT Attention Analysis\n"
        f"Fake: {fake_probability:.2f}% | "
        f"Real: {real_probability:.2f}%",
        fontsize=14
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.savefig(
        combined_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Print paths
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)

    print()
    print(
        f"Heatmap : {heatmap_path}"
    )

    print(
        f"Overlay : {overlay_path}"
    )

    print(
        f"Combined: {combined_path}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check command-line argument
    # --------------------------------------------------------

    if len(sys.argv) != 2:

        print()
        print(
            "Usage:"
        )

        print()

        print(
            'python -u '
            '"src\\xai\\vit_attention_single.py" '
            '"C:\\path\\to\\image.jpg"'
        )

        print()

        sys.exit(1)

    image_path = sys.argv[1]

    print()
    print("=" * 60)
    print("DEEPS SINGLE IMAGE ViT ATTENTION ANALYSIS")
    print("=" * 60)

    print()
    print(
        f"Input image: {image_path}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    (
        image,
        original_image,
        input_tensor
    ) = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    (
        predicted_class,
        predicted_label,
        confidence,
        fake_probability,
        real_probability
    ) = predict(
        model,
        input_tensor
    )

    # --------------------------------------------------------
    # Attention rollout
    # --------------------------------------------------------

    attention_map = attention_rollout()

    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    (
        heatmap,
        heatmap_color,
        overlay
    ) = create_heatmap(
        attention_map,
        original_image
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        image_path,
        original_image,
        heatmap,
        heatmap_color,
        overlay,
        predicted_label,
        confidence,
        fake_probability,
        real_probability
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print()
    print(
        f"Final ViT prediction: "
        f"{predicted_label.upper()}"
    )

    print(
        f"Model confidence: "
        f"{confidence:.2f}%"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()