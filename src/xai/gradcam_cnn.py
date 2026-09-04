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

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "best_cnn_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "xai"
    / "gradcam_cnn"
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
# SAME SPLIT AS TRAINING
# ============================================================

train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = (
    len(dataset)
    - train_size
    - val_size
)

_, _, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print("Test dataset size:", len(test_dataset))


# ============================================================
# CREATE CNN
# ============================================================

model = DeepShieldCNN().to(device)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained CNN...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("CNN model loaded successfully!")


# ============================================================
# GRAD-CAM STORAGE
# ============================================================

activations = None
gradients = None


# ============================================================
# FORWARD HOOK
# ============================================================

def forward_hook(module, input, output):

    global activations

    activations = output


# ============================================================
# BACKWARD HOOK
# ============================================================

def backward_hook(module, grad_input, grad_output):

    global gradients

    gradients = grad_output[0]


# ============================================================
# TARGET LAYER
# ============================================================

target_layer = model.features[6]

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(image_tensor):

    global activations
    global gradients

    activations = None
    gradients = None

    image_tensor = image_tensor.unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(image_tensor)

    probabilities = F.softmax(
        output,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        prediction
    ].item()

    # --------------------------------------------------------
    # Clear previous gradients
    # --------------------------------------------------------

    model.zero_grad()

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    target_score = output[
        0,
        prediction
    ]

    target_score.backward()

    # --------------------------------------------------------
    # Get activations and gradients
    # --------------------------------------------------------

    feature_maps = activations[0]

    grads = gradients[0]

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    weights = grads.mean(
        dim=(1, 2)
    )

    # --------------------------------------------------------
    # Weighted combination
    # --------------------------------------------------------

    cam = torch.zeros(
        feature_maps.shape[1:],
        device=device
    )

    for i, weight in enumerate(weights):

        cam += (
            weight
            * feature_maps[i]
        )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = F.relu(cam)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    cam -= cam.min()

    if cam.max() != 0:

        cam /= cam.max()

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    cam = cam.detach().cpu().numpy()

    return cam, prediction, confidence


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(index):

    image_tensor, label = test_dataset[index]

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    cam, prediction, confidence = (
        generate_gradcam(image_tensor)
    )

    # --------------------------------------------------------
    # Convert normalized tensor back to image
    # --------------------------------------------------------

    image = image_tensor.permute(
        1, 2, 0
    ).cpu().numpy()

    mean = np.array(
        [0.485, 0.456, 0.406]
    )

    std = np.array(
        [0.229, 0.224, 0.225]
    )

    image = (
        image * std
    ) + mean

    image = np.clip(
        image,
        0,
        1
    )

    # --------------------------------------------------------
    # Resize CAM
    # --------------------------------------------------------

    cam = cv2.resize(
        cam,
        (224, 224)
    )

    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    heatmap = np.uint8(
        255 * cam
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
    # Convert original image
    # --------------------------------------------------------

    original = np.uint8(
        image * 255
    )

    # --------------------------------------------------------
    # Overlay heatmap
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    true_class = dataset.classes[label]

    predicted_class = (
        dataset.classes[prediction]
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 4)
    )

    plt.subplot(1, 3, 1)

    plt.imshow(original)

    plt.title(
        f"Original\nTrue: {true_class}"
    )

    plt.axis("off")

    plt.subplot(1, 3, 2)

    plt.imshow(heatmap)

    plt.title("Grad-CAM")

    plt.axis("off")

    plt.subplot(1, 3, 3)

    plt.imshow(overlay)

    plt.title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence * 100:.2f}%"
    )

    plt.axis("off")

    plt.tight_layout()

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"gradcam_{index}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )

    print(
        f"True class: {true_class}"
    )

    print(
        f"Prediction: {predicted_class}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )


# ============================================================
# SELECT REPRESENTATIVE XAI CASES
# ============================================================

print("\n========================================")
print("SELECTING REPRESENTATIVE XAI CASES")
print("========================================")


results = []


# ------------------------------------------------------------
# Evaluate every test image
# ------------------------------------------------------------

for i in range(len(test_dataset)):

    image_tensor, label = test_dataset[i]

    image_input = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_input)

        probabilities = F.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            prediction
        ].item()

    results.append({
        "index": i,
        "true": label,
        "prediction": prediction,
        "confidence": confidence,
        "correct": prediction == label
    })


# ============================================================
# SEPARATE CASES
# ============================================================

real_correct = [
    r for r in results
    if r["true"] == dataset.class_to_idx["real"]
    and r["correct"]
]

fake_correct = [
    r for r in results
    if r["true"] == dataset.class_to_idx["fake"]
    and r["correct"]
]

incorrect = [
    r for r in results
    if not r["correct"]
]


# ============================================================
# SORT BY CONFIDENCE
# ============================================================

real_correct_high = sorted(
    real_correct,
    key=lambda x: x["confidence"],
    reverse=True
)[:2]


fake_correct_high = sorted(
    fake_correct,
    key=lambda x: x["confidence"],
    reverse=True
)[:2]


# Lowest-confidence correct predictions

correct_predictions = [
    r for r in results
    if r["correct"]
]

low_confidence = sorted(
    correct_predictions,
    key=lambda x: x["confidence"]
)[:2]


# ============================================================
# BUILD FINAL CASE LIST
# ============================================================

selected_cases = []

selected_cases.extend(
    real_correct_high
)

selected_cases.extend(
    fake_correct_high
)

selected_cases.extend(
    low_confidence
)


# Add incorrect predictions if available

if len(incorrect) > 0:

    incorrect_sorted = sorted(
        incorrect,
        key=lambda x: x["confidence"],
        reverse=True
    )

    selected_cases.extend(
        incorrect_sorted[:2]
    )


# Remove duplicate image indices

unique_cases = {}

for case in selected_cases:

    unique_cases[
        case["index"]
    ] = case


selected_cases = list(
    unique_cases.values()
)


# ============================================================
# PRINT SELECTED CASES
# ============================================================

print("\nSelected XAI cases:")

for case in selected_cases:

    true_class = dataset.classes[
        case["true"]
    ]

    predicted_class = dataset.classes[
        case["prediction"]
    ]

    print(
        f"Image {case['index']} | "
        f"True: {true_class} | "
        f"Prediction: {predicted_class} | "
        f"Confidence: "
        f"{case['confidence'] * 100:.2f}%"
    )


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

print("\n========================================")
print("GENERATING REPRESENTATIVE GRAD-CAM")
print("========================================")


for count, case in enumerate(
    selected_cases,
    start=1
):

    print(
        f"\nProcessing XAI case "
        f"{count}/{len(selected_cases)}"
    )

    process_image(
        case["index"]
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("CNN XAI EVALUATION COMPLETE")
print("========================================")

print(
    f"Results saved in:\n{OUTPUT_DIR}"
)