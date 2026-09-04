import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))

from models.cnn_model import DeepShieldCNN
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
# MODEL PATHS
# ============================================================

CNN_MODEL_PATH = (
    PROJECT_ROOT /
    "results" /
    "best_cnn_model.pth"
)

VIT_MODEL_PATH = (
    PROJECT_ROOT /
    "results" /
    "best_vit_model.pth"
)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "fake",
    1: "real"
}


# ============================================================
# ENSEMBLE WEIGHTS
# ============================================================

CNN_WEIGHT = 0.3
VIT_WEIGHT = 0.7


# ============================================================
# IMAGE TRANSFORMATION
# SAME AS EVALUATION
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
# LOAD CNN
# ============================================================

print("\nLoading CNN...")

cnn = DeepShieldCNN().to(device)

cnn.load_state_dict(
    torch.load(
        CNN_MODEL_PATH,
        map_location=device
    )
)

cnn.eval()

print("CNN loaded successfully!")


# ============================================================
# LOAD ViT
# ============================================================

print("\nLoading Vision Transformer...")

vit = DeepShieldViT(
    num_classes=2
).to(device)

vit.load_state_dict(
    torch.load(
        VIT_MODEL_PATH,
        map_location=device
    )
)

vit.eval()

print("ViT loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    print("\n========================================")
    print("DEEPS SINGLE IMAGE PREDICTION")
    print("========================================")

    print("Image:", image_path)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # --------------------------------------------------------
    # MODEL PREDICTIONS
    # --------------------------------------------------------

    with torch.no_grad():

        cnn_output = cnn(image_tensor)

        vit_output = vit(image_tensor)

        # Convert logits to probabilities
        cnn_probs = F.softmax(
            cnn_output,
            dim=1
        )[0]

        vit_probs = F.softmax(
            vit_output,
            dim=1
        )[0]

        # ----------------------------------------------------
        # ENSEMBLE
        # ----------------------------------------------------

        ensemble_probs = (
            CNN_WEIGHT * cnn_probs
            +
            VIT_WEIGHT * vit_probs
        )

    # --------------------------------------------------------
    # GET PREDICTIONS
    # --------------------------------------------------------

    cnn_prediction = torch.argmax(
        cnn_probs
    ).item()

    vit_prediction = torch.argmax(
        vit_probs
    ).item()

    ensemble_prediction = torch.argmax(
        ensemble_probs
    ).item()

    # --------------------------------------------------------
    # CONFIDENCES
    # --------------------------------------------------------

    cnn_confidence = (
        cnn_probs[cnn_prediction].item()
        * 100
    )

    vit_confidence = (
        vit_probs[vit_prediction].item()
        * 100
    )

    ensemble_confidence = (
        ensemble_probs[ensemble_prediction].item()
        * 100
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n----------------------------------------")
    print("CNN")
    print("----------------------------------------")

    print(
        f"Prediction : "
        f"{CLASS_NAMES[cnn_prediction].upper()}"
    )

    print(
        f"Confidence : "
        f"{cnn_confidence:.2f}%"
    )

    print("\n----------------------------------------")
    print("VISION TRANSFORMER")
    print("----------------------------------------")

    print(
        f"Prediction : "
        f"{CLASS_NAMES[vit_prediction].upper()}"
    )

    print(
        f"Confidence : "
        f"{vit_confidence:.2f}%"
    )

    print("\n----------------------------------------")
    print("CNN + ViT ENSEMBLE")
    print("----------------------------------------")

    print(
        f"CNN Weight : {CNN_WEIGHT}"
    )

    print(
        f"ViT Weight : {VIT_WEIGHT}"
    )

    print(
        f"Prediction : "
        f"{CLASS_NAMES[ensemble_prediction].upper()}"
    )

    print(
        f"Confidence : "
        f"{ensemble_confidence:.2f}%"
    )

    print("\n========================================")
    print("FINAL DEEPS PREDICTION")
    print("========================================")

    print(
        f"Result: "
        f"{CLASS_NAMES[ensemble_prediction].upper()}"
    )

    print(
        f"Confidence: "
        f"{ensemble_confidence:.2f}%"
    )

    print("========================================")

    return {
        "cnn_prediction":
            CLASS_NAMES[cnn_prediction],

        "cnn_confidence":
            cnn_confidence,

        "vit_prediction":
            CLASS_NAMES[vit_prediction],

        "vit_confidence":
            vit_confidence,

        "ensemble_prediction":
            CLASS_NAMES[ensemble_prediction],

        "ensemble_confidence":
            ensemble_confidence
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python src/inference/predict.py "
            "<image_path>"
        )

        print(
            "\nExample:"
        )

        print(
            "python src/inference/predict.py "
            "dataset/images/fake/example.jpg"
        )

        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():

        print(
            f"\nError: Image not found:"
            f"\n{image_path}"
        )

        sys.exit(1)

    predict_image(image_path) 
