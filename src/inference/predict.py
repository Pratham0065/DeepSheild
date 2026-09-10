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


# ============================================================
# MODEL IMPORTS
# ============================================================

from models.cnn_model import DeepShieldCNN
from models.vit_model import DeepShieldViT

# Human-readable explanation engine
from web.explanation_engine import explain_ensemble


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

CNN_WEIGHT = 0.5
VIT_WEIGHT = 0.5


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

        # CNN prediction
        cnn_output = cnn(image_tensor)

        # ViT prediction
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
        # 50:50 ENSEMBLE
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
    # HUMAN-READABLE EXPLANATION
    # ========================================================

    explanation = explain_ensemble(

        cnn_fake=cnn_probs[0].item(),

        cnn_real=cnn_probs[1].item(),

        vit_fake=vit_probs[0].item(),

        vit_real=vit_probs[1].item(),

        ensemble_fake=ensemble_probs[0].item(),

        ensemble_real=ensemble_probs[1].item()
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

    print(
        f"Fake       : "
        f"{cnn_probs[0].item() * 100:.2f}%"
    )

    print(
        f"Real       : "
        f"{cnn_probs[1].item() * 100:.2f}%"
    )

    # --------------------------------------------------------

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

    print(
        f"Fake       : "
        f"{vit_probs[0].item() * 100:.2f}%"
    )

    print(
        f"Real       : "
        f"{vit_probs[1].item() * 100:.2f}%"
    )

    # --------------------------------------------------------

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

    print(
        f"Fake       : "
        f"{ensemble_probs[0].item() * 100:.2f}%"
    )

    print(
        f"Real       : "
        f"{ensemble_probs[1].item() * 100:.2f}%"
    )

    # ========================================================
    # FINAL DEEPS PREDICTION
    # ========================================================

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

    # ========================================================
    # HUMAN-READABLE EXPLANATION
    # ========================================================

    print("\n========================================")
    print("WHY DID DEEPS MAKE THIS PREDICTION?")
    print("========================================")

    print(
        explanation["explanation"]
    )

    print(
        f"\nConfidence Level: "
        f"{explanation['confidence']}"
    )

    print("========================================")

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        "cnn_prediction":
            CLASS_NAMES[cnn_prediction],

        "cnn_confidence":
            cnn_confidence,

        "cnn_fake_probability":
            cnn_probs[0].item(),

        "cnn_real_probability":
            cnn_probs[1].item(),

        # ----------------------------------------------------
        # ViT
        # ----------------------------------------------------

        "vit_prediction":
            CLASS_NAMES[vit_prediction],

        "vit_confidence":
            vit_confidence,

        "vit_fake_probability":
            vit_probs[0].item(),

        "vit_real_probability":
            vit_probs[1].item(),

        # ----------------------------------------------------
        # Ensemble
        # ----------------------------------------------------

        "ensemble_prediction":
            CLASS_NAMES[ensemble_prediction],

        "ensemble_confidence":
            ensemble_confidence,

        "ensemble_fake_probability":
            ensemble_probs[0].item(),

        "ensemble_real_probability":
            ensemble_probs[1].item(),

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        "explanation":
            explanation
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CHECK ARGUMENT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # IMAGE PATH
    # --------------------------------------------------------

    image_path = Path(sys.argv[1])

    # --------------------------------------------------------
    # CHECK IMAGE EXISTS
    # --------------------------------------------------------

    if not image_path.exists():

        print(
            f"\nError: Image not found:"
            f"\n{image_path}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------------

    predict_image(image_path)