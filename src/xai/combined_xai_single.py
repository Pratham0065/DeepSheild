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
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT MODELS
# ============================================================

from models.cnn_model import DeepShieldCNN
from models.vit_model import DeepShieldViT


# ============================================================
# CONFIGURATION
# ============================================================

CNN_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "best_cnn_model.pth"
)

VIT_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "best_vit_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "xai"
    / "combined_single"
)

IMAGE_SIZE = 224

# Final ensemble weights
CNN_WEIGHT = 0.5
VIT_WEIGHT = 0.5

CLASS_NAMES = {
    0: "fake",
    1: "real"
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 70)
print("DEEPS COMPLETE SINGLE-IMAGE XAI ANALYSIS")
print("=" * 70)

print()
print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# IMAGE TRANSFORM
# Same preprocessing used during model evaluation
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
# CNN GRAD-CAM STORAGE
# ============================================================

cnn_activations = None
cnn_gradients = None


# ============================================================
# CNN FORWARD HOOK
# ============================================================

def cnn_forward_hook(module, input, output):

    global cnn_activations

    cnn_activations = output


# ============================================================
# CNN BACKWARD HOOK
# ============================================================

def cnn_backward_hook(
    module,
    grad_input,
    grad_output
):

    global cnn_gradients

    cnn_gradients = grad_output[0]


# ============================================================
# ViT ATTENTION STORAGE
# ============================================================

vit_attention_maps = []


# ============================================================
# ViT ATTENTION CAPTURE
# ============================================================

def make_attention_capture(attention_module):

    def forward_with_attention(
        x,
        attn_mask=None,
        is_causal=False,
        **kwargs
    ):

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
        # Scale query
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

            attention = (
                attention + attn_mask
            )

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
        # Store attention
        # ----------------------------------------------------

        vit_attention_maps.append(
            attention.detach()
        )

        # ----------------------------------------------------
        # Attention output
        # ----------------------------------------------------

        attention_output = (
            attention @ v
        )

        attention_output = (
            attention_output
            .transpose(1, 2)
            .reshape(B, N, C)
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

    return forward_with_attention


# ============================================================
# PATCH ViT ATTENTION
# ============================================================

def patch_vit_attention(model):

    for block in model.vit.blocks:

        # Disable fused attention so that the
        # attention matrix can be captured.
        block.attn.fused_attn = False

        block.attn.forward = (
            make_attention_capture(
                block.attn
            )
        )


# ============================================================
# LOAD CNN
# ============================================================

def load_cnn():

    print()
    print("=" * 70)
    print("LOADING CNN")
    print("=" * 70)

    print()
    print(
        "Checkpoint:",
        CNN_MODEL_PATH
    )

    if not CNN_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"CNN checkpoint not found:\n"
            f"{CNN_MODEL_PATH}"
        )

    model = DeepShieldCNN()

    model.load_state_dict(
        torch.load(
            CNN_MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    # --------------------------------------------------------
    # Grad-CAM target layer
    # --------------------------------------------------------

    target_layer = model.features[6]

    target_layer.register_forward_hook(
        cnn_forward_hook
    )

    target_layer.register_full_backward_hook(
        cnn_backward_hook
    )

    print()
    print(
        "CNN model loaded successfully!"
    )

    print(
        "Grad-CAM target layer:",
        "model.features[6]"
    )

    return model


# ============================================================
# LOAD ViT
# ============================================================

def load_vit():

    print()
    print("=" * 70)
    print("LOADING VISION TRANSFORMER")
    print("=" * 70)

    print()
    print(
        "Checkpoint:",
        VIT_MODEL_PATH
    )

    if not VIT_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"ViT checkpoint not found:\n"
            f"{VIT_MODEL_PATH}"
        )

    model = DeepShieldViT(
        num_classes=2
    )

    checkpoint = torch.load(
        VIT_MODEL_PATH,
        map_location=device
    )

    # --------------------------------------------------------
    # Handle different checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint["state_dict"]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    model.eval()

    # --------------------------------------------------------
    # Enable attention capture
    # --------------------------------------------------------

    patch_vit_attention(
        model
    )

    print()
    print(
        "ViT model loaded successfully!"
    )

    print(
        "Attention capture enabled for",
        len(model.vit.blocks),
        "transformer blocks."
    )

    return model


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )

    print()
    print("=" * 70)
    print("LOADING IMAGE")
    print("=" * 70)

    print()
    print(
        "Image:",
        image_path
    )

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    original_image = np.array(
        image
    )

    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    input_tensor = transform(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        device
    )

    print()
    print(
        "Original size:",
        image.width,
        "x",
        image.height
    )

    print(
        "Model input:",
        IMAGE_SIZE,
        "x",
        IMAGE_SIZE
    )

    return (
        image,
        original_image,
        input_tensor
    )


# ============================================================
# RUN CNN + GRAD-CAM
# ============================================================

def run_cnn(
    model,
    input_tensor
):

    global cnn_activations
    global cnn_gradients

    print()
    print("=" * 70)
    print("CNN PREDICTION + GRAD-CAM")
    print("=" * 70)

    cnn_activations = None
    cnn_gradients = None

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(
        input_tensor
    )

    probabilities = F.softmax(
        output,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = (
        probabilities[0, prediction]
        .item()
        * 100
    )

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

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    model.zero_grad()

    target_score = output[
        0,
        prediction
    ]

    target_score.backward()

    # --------------------------------------------------------
    # Check hooks
    # --------------------------------------------------------

    if cnn_activations is None:

        raise RuntimeError(
            "CNN activations were not captured."
        )

    if cnn_gradients is None:

        raise RuntimeError(
            "CNN gradients were not captured."
        )

    # --------------------------------------------------------
    # Feature maps and gradients
    # --------------------------------------------------------

    feature_maps = (
        cnn_activations[0]
    )

    gradients = (
        cnn_gradients[0]
    )

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    weights = gradients.mean(
        dim=(1, 2)
    )

    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    cam = torch.zeros(
        feature_maps.shape[1:],
        device=device
    )

    for i, weight in enumerate(
        weights
    ):

        cam += (
            weight
            * feature_maps[i]
        )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = F.relu(
        cam
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    cam -= cam.min()

    if cam.max() != 0:

        cam /= cam.max()

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    cam = (
        cam
        .detach()
        .cpu()
        .numpy()
    )

    print()
    print(
        f"CNN Fake probability : "
        f"{fake_probability:.2f}%"
    )

    print(
        f"CNN Real probability : "
        f"{real_probability:.2f}%"
    )

    print()
    print(
        f"CNN Prediction       : "
        f"{CLASS_NAMES[prediction].upper()}"
    )

    print(
        f"CNN Confidence       : "
        f"{confidence:.2f}%"
    )

    return (
        cam,
        probabilities.detach().cpu().numpy()[0],
        prediction,
        confidence
    )


# ============================================================
# RUN ViT + ATTENTION ROLLOUT
# ============================================================

def run_vit(
    model,
    input_tensor
):

    global vit_attention_maps

    print()
    print("=" * 70)
    print("VIT PREDICTION + ATTENTION ROLLOUT")
    print("=" * 70)

    # --------------------------------------------------------
    # Clear previous attention
    # --------------------------------------------------------

    vit_attention_maps.clear()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = F.softmax(
            output,
            dim=1
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = (
        probabilities[0, prediction]
        .item()
        * 100
    )

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

    print()
    print(
        f"ViT Fake probability : "
        f"{fake_probability:.2f}%"
    )

    print(
        f"ViT Real probability : "
        f"{real_probability:.2f}%"
    )

    print()
    print(
        f"ViT Prediction       : "
        f"{CLASS_NAMES[prediction].upper()}"
    )

    print(
        f"ViT Confidence       : "
        f"{confidence:.2f}%"
    )

    print()
    print(
        "Captured attention maps:",
        len(vit_attention_maps)
    )

    if len(vit_attention_maps) == 0:

        raise RuntimeError(
            "No ViT attention maps were captured."
        )

    # ========================================================
    # ATTENTION ROLLOUT
    # ========================================================

    num_tokens = (
        vit_attention_maps[0]
        .shape[-1]
    )

    rollout = torch.eye(
        num_tokens,
        device=device
    )

    # --------------------------------------------------------
    # Process every transformer block
    # --------------------------------------------------------

    for attention in (
        vit_attention_maps
    ):

        # Average attention across heads
        attention = attention.mean(
            dim=1
        )[0]

        # Residual connection
        identity = torch.eye(
            num_tokens,
            device=device
        )

        attention = (
            attention + identity
        )

        # Row normalization
        attention = (
            attention
            / (
                attention.sum(
                    dim=-1,
                    keepdim=True
                )
                + 1e-8
            )
        )

        # Accumulate
        rollout = (
            attention @ rollout
        )

    # --------------------------------------------------------
    # CLS token → image patches
    # --------------------------------------------------------

    cls_attention = rollout[
        0,
        1:
    ]

    num_patches = (
        cls_attention.shape[0]
    )

    grid_size = int(
        np.sqrt(num_patches)
    )

    if (
        grid_size * grid_size
        != num_patches
    ):

        raise RuntimeError(
            f"Unexpected number of "
            f"ViT patches: {num_patches}"
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

    attention_map -= (
        attention_map.min()
    )

    if attention_map.max() > 0:

        attention_map /= (
            attention_map.max()
        )

    print()
    print(
        "Attention tokens:",
        num_tokens
    )

    print(
        "Image patches:",
        num_patches
    )

    print(
        "Attention grid:",
        f"{grid_size} x {grid_size}"
    )

    return (
        attention_map,
        probabilities.detach().cpu().numpy()[0],
        prediction,
        confidence
    )


# ============================================================
# CREATE CNN VISUALIZATION
# ============================================================

def create_cnn_visualization(
    cam,
    original_image
):

    height = (
        original_image.shape[0]
    )

    width = (
        original_image.shape[1]
    )

    # --------------------------------------------------------
    # Resize Grad-CAM
    # --------------------------------------------------------

    cam_resized = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # Convert to color heatmap
    # --------------------------------------------------------

    heatmap = np.uint8(
        255 * cam_resized
    )
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Overlay heatmap on original image
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return (
        heatmap,
        overlay
    )


# ============================================================
# CREATE ViT VISUALIZATION
# ============================================================

def create_vit_visualization(
    attention_map,
    original_image
):

    height = (
        original_image.shape[0]
    )

    width = (
        original_image.shape[1]
    )

    # --------------------------------------------------------
    # Resize attention map
    # --------------------------------------------------------

    attention_resized = cv2.resize(
        attention_map,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # Convert to 8-bit
    # --------------------------------------------------------

    heatmap = np.uint8(
        255 * attention_resized
    )

    # --------------------------------------------------------
    # Apply color map
    # --------------------------------------------------------

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Overlay attention on original image
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return (
        heatmap,
        overlay
    )


# ============================================================
# SAVE INDIVIDUAL XAI RESULTS
# ============================================================

def save_individual_results(
    image_name,
    original_image,
    cnn_heatmap,
    cnn_overlay,
    vit_heatmap,
    vit_overlay
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original_path = (
        OUTPUT_DIR
        / f"{image_name}_original.png"
    )

    Image.fromarray(
        original_image
    ).save(
        original_path
    )

    # --------------------------------------------------------
    # CNN heatmap
    # --------------------------------------------------------

    cnn_heatmap_path = (
        OUTPUT_DIR
        / f"{image_name}_cnn_gradcam_heatmap.png"
    )

    Image.fromarray(
        cnn_heatmap
    ).save(
        cnn_heatmap_path
    )

    # --------------------------------------------------------
    # CNN overlay
    # --------------------------------------------------------

    cnn_overlay_path = (
        OUTPUT_DIR
        / f"{image_name}_cnn_gradcam_overlay.png"
    )

    Image.fromarray(
        cnn_overlay
    ).save(
        cnn_overlay_path
    )

    # --------------------------------------------------------
    # ViT heatmap
    # --------------------------------------------------------

    vit_heatmap_path = (
        OUTPUT_DIR
        / f"{image_name}_vit_attention_heatmap.png"
    )

    Image.fromarray(
        vit_heatmap
    ).save(
        vit_heatmap_path
    )

    # --------------------------------------------------------
    # ViT overlay
    # --------------------------------------------------------

    vit_overlay_path = (
        OUTPUT_DIR
        / f"{image_name}_vit_attention_overlay.png"
    )

    Image.fromarray(
        vit_overlay
    ).save(
        vit_overlay_path
    )

    print()
    print("=" * 70)
    print("INDIVIDUAL XAI RESULTS SAVED")
    print("=" * 70)

    print()
    print(
        "Original:",
        original_path
    )

    print(
        "CNN heatmap:",
        cnn_heatmap_path
    )

    print(
        "CNN overlay:",
        cnn_overlay_path
    )

    print(
        "ViT heatmap:",
        vit_heatmap_path
    )

    print(
        "ViT overlay:",
        vit_overlay_path
    )


# ============================================================
# CREATE COMPLETE XAI FIGURE
# ============================================================

def create_complete_figure(
    image_name,
    original_image,
    cnn_heatmap,
    cnn_overlay,
    vit_heatmap,
    vit_overlay,
    cnn_probs,
    vit_probs,
    ensemble_probs,
    cnn_prediction,
    vit_prediction,
    ensemble_prediction
):

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11)
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    axes[0, 0].imshow(
        original_image
    )

    axes[0, 0].set_title(
        "Original Image",
        fontsize=15,
        fontweight="bold"
    )

    axes[0, 0].axis("off")

    # --------------------------------------------------------
    # CNN Grad-CAM heatmap
    # --------------------------------------------------------

    axes[0, 1].imshow(
        cnn_heatmap
    )

    axes[0, 1].set_title(
        "CNN Grad-CAM",
        fontsize=15,
        fontweight="bold"
    )

    axes[0, 1].axis("off")

    # --------------------------------------------------------
    # CNN overlay
    # --------------------------------------------------------

    axes[0, 2].imshow(
        cnn_overlay
    )

    axes[0, 2].set_title(
        (
            f"CNN Overlay\n"
            f"Prediction: "
            f"{CLASS_NAMES[cnn_prediction].upper()}\n"
            f"Fake: {cnn_probs[0] * 100:.2f}% | "
            f"Real: {cnn_probs[1] * 100:.2f}%"
        ),
        fontsize=13,
        fontweight="bold"
    )

    axes[0, 2].axis("off")

    # --------------------------------------------------------
    # ViT attention heatmap
    # --------------------------------------------------------

    axes[1, 0].imshow(
        vit_heatmap
    )

    axes[1, 0].set_title(
        "ViT Attention Rollout",
        fontsize=15,
        fontweight="bold"
    )

    axes[1, 0].axis("off")

    # --------------------------------------------------------
    # ViT overlay
    # --------------------------------------------------------

    axes[1, 1].imshow(
        vit_overlay
    )

    axes[1, 1].set_title(
        (
            f"ViT Overlay\n"
            f"Prediction: "
            f"{CLASS_NAMES[vit_prediction].upper()}\n"
            f"Fake: {vit_probs[0] * 100:.2f}% | "
            f"Real: {vit_probs[1] * 100:.2f}%"
        ),
        fontsize=13,
        fontweight="bold"
    )

    axes[1, 1].axis("off")

    # --------------------------------------------------------
    # Ensemble result
    # --------------------------------------------------------

    axes[1, 2].imshow(
        original_image
    )

    ensemble_confidence = (
        ensemble_probs[
            ensemble_prediction
        ]
        * 100
    )

    axes[1, 2].set_title(
        (
            f"FINAL 50:50 ENSEMBLE\n"
            f"Prediction: "
            f"{CLASS_NAMES[ensemble_prediction].upper()}\n"
            f"Confidence: "
            f"{ensemble_confidence:.2f}%\n"
            f"Fake: {ensemble_probs[0] * 100:.2f}% | "
            f"Real: {ensemble_probs[1] * 100:.2f}%"
        ),
        fontsize=13,
        fontweight="bold"
    )

    axes[1, 2].axis("off")

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        (
            "DeepS — CNN Grad-CAM + "
            "ViT Attention Rollout + "
            "50:50 Ensemble"
        ),
        fontsize=19,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{image_name}_complete_xai.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print()
    print("=" * 70)
    print("COMPLETE XAI FIGURE SAVED")
    print("=" * 70)

    print()
    print(
        output_path
    )

    return output_path


# ============================================================
# RUN 50:50 ENSEMBLE
# ============================================================

def run_ensemble(
    cnn_probs,
    vit_probs
):

    print()
    print("=" * 70)
    print("FINAL 50:50 CNN + ViT ENSEMBLE")
    print("=" * 70)

    # --------------------------------------------------------
    # Weighted probability fusion
    # --------------------------------------------------------

    ensemble_probs = (
        CNN_WEIGHT * cnn_probs
        +
        VIT_WEIGHT * vit_probs
    )

    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    prediction = int(
        np.argmax(
            ensemble_probs
        )
    )

    confidence = (
        ensemble_probs[prediction]
        * 100
    )

    fake_probability = (
        ensemble_probs[0]
        * 100
    )

    real_probability = (
        ensemble_probs[1]
        * 100
    )

    print()
    print(
        f"CNN weight           : "
        f"{CNN_WEIGHT:.2f}"
    )

    print(
        f"ViT weight           : "
        f"{VIT_WEIGHT:.2f}"
    )

    print()
    print(
        f"Ensemble Fake probability : "
        f"{fake_probability:.2f}%"
    )

    print(
        f"Ensemble Real probability : "
        f"{real_probability:.2f}%"
    )

    print()
    print(
        f"FINAL PREDICTION     : "
        f"{CLASS_NAMES[prediction].upper()}"
    )

    print(
        f"FINAL CONFIDENCE     : "
        f"{confidence:.2f}%"
    )

    return (
        ensemble_probs,
        prediction,
        confidence
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check command-line argument
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            "python -u "
            "\"src\\xai\\combined_xai_single.py\" "
            "\"path\\to\\image.jpg\""
        )

        print()

        sys.exit(1)

    image_path = sys.argv[1]

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    cnn_model = load_cnn()

    vit_model = load_vit()

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
    # Image name
    # --------------------------------------------------------

    image_name = (
        Path(image_path)
        .stem
    )

    # --------------------------------------------------------
    # CNN + Grad-CAM
    # --------------------------------------------------------

    (
        cnn_cam,
        cnn_probs,
        cnn_prediction,
        cnn_confidence
    ) = run_cnn(
        cnn_model,
        input_tensor
    )

    # --------------------------------------------------------
    # ViT + Attention Rollout
    # --------------------------------------------------------

    (
        vit_attention_map,
        vit_probs,
        vit_prediction,
        vit_confidence
    ) = run_vit(
        vit_model,
        input_tensor
    )

    # --------------------------------------------------------
    # Final ensemble
    # --------------------------------------------------------

    (
        ensemble_probs,
        ensemble_prediction,
        ensemble_confidence
    ) = run_ensemble(
        cnn_probs,
        vit_probs
    )

    # --------------------------------------------------------
    # Create visualizations
    # --------------------------------------------------------

    (
        cnn_heatmap,
        cnn_overlay
    ) = create_cnn_visualization(
        cnn_cam,
        original_image
    )

    (
        vit_heatmap,
        vit_overlay
    ) = create_vit_visualization(
        vit_attention_map,
        original_image
    )

    # --------------------------------------------------------
    # Save individual results
    # --------------------------------------------------------

    save_individual_results(
        image_name,
        original_image,
        cnn_heatmap,
        cnn_overlay,
        vit_heatmap,
        vit_overlay
    )

    # --------------------------------------------------------
    # Create complete figure
    # --------------------------------------------------------

    complete_figure_path = (
        create_complete_figure(
            image_name,
            original_image,
            cnn_heatmap,
            cnn_overlay,
            vit_heatmap,
            vit_overlay,
            cnn_probs,
            vit_probs,
            ensemble_probs,
            cnn_prediction,
            vit_prediction,
            ensemble_prediction
        )
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DEEPS XAI ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        "Image:",
        image_path
    )

    print()
    print(
        "CNN:"
    )

    print(
        f"  Prediction : "
        f"{CLASS_NAMES[cnn_prediction].upper()}"
    )

    print(
        f"  Fake       : "
        f"{cnn_probs[0] * 100:.2f}%"
    )

    print(
        f"  Real       : "
        f"{cnn_probs[1] * 100:.2f}%"
    )

    print()
    print(
        "ViT:"
    )

    print(
        f"  Prediction : "
        f"{CLASS_NAMES[vit_prediction].upper()}"
    )

    print(
        f"  Fake       : "
        f"{vit_probs[0] * 100:.2f}%"
    )

    print(
        f"  Real       : "
        f"{vit_probs[1] * 100:.2f}%"
    )

    print()
    print(
        "50:50 Ensemble:"
    )

    print(
        f"  Prediction : "
        f"{CLASS_NAMES[ensemble_prediction].upper()}"
    )

    print(
        f"  Fake       : "
        f"{ensemble_probs[0] * 100:.2f}%"
    )

    print(
        f"  Real       : "
        f"{ensemble_probs[1] * 100:.2f}%"
    )

    print(
        f"  Confidence : "
        f"{ensemble_confidence:.2f}%"
    )

    print()
    print(
        "Complete visualization:"
    )

    print(
        complete_figure_path
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()