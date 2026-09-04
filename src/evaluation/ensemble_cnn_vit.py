import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


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
# PATHS
# ============================================================

DATASET_DIR = PROJECT_ROOT / "dataset" / "images"

CNN_MODEL_PATH = PROJECT_ROOT / "results" / "best_cnn_model.pth"
VIT_MODEL_PATH = PROJECT_ROOT / "results" / "best_vit_model.pth"


# ============================================================
# IMAGE TRANSFORMATIONS
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
# SAME DATASET SPLIT AS CNN AND ViT
# ============================================================

train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print("\nDataset split:")
print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Testing:", len(test_dataset))


# ============================================================
# DATALOADERS
# ============================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD CNN
# ============================================================

print("\nLoading CNN...")

cnn_model = DeepShieldCNN().to(device)

cnn_model.load_state_dict(
    torch.load(
        CNN_MODEL_PATH,
        map_location=device
    )
)

cnn_model.eval()

print("Best CNN model loaded successfully!")


# ============================================================
# LOAD ViT
# ============================================================

print("\nLoading Vision Transformer...")

vit_model = DeepShieldViT(
    num_classes=2
).to(device)

vit_model.load_state_dict(
    torch.load(
        VIT_MODEL_PATH,
        map_location=device
    )
)

vit_model.eval()

print("Best ViT model loaded successfully!")


# ============================================================
# GET CNN + ViT PROBABILITIES
# ============================================================

def get_probabilities(loader):

    labels = []

    cnn_probs = []

    vit_probs = []

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(device)

            # CNN
            cnn_outputs = cnn_model(images)

            batch_cnn_probs = torch.softmax(
                cnn_outputs,
                dim=1
            )

            # ViT
            vit_outputs = vit_model(images)

            batch_vit_probs = torch.softmax(
                vit_outputs,
                dim=1
            )

            labels.extend(
                batch_labels.numpy()
            )

            cnn_probs.extend(
                batch_cnn_probs.cpu().numpy()
            )

            vit_probs.extend(
                batch_vit_probs.cpu().numpy()
            )

    return (
        labels,
        cnn_probs,
        vit_probs
    )


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n========================================")
print("COLLECTING VALIDATION PREDICTIONS")
print("========================================")

val_labels, val_cnn_probs, val_vit_probs = get_probabilities(
    val_loader
)


# ============================================================
# WEIGHT OPTIMIZATION
# ============================================================

print("\n========================================")
print("ENSEMBLE WEIGHT OPTIMIZATION")
print("========================================")

print("\nTesting CNN/ViT weight combinations...")
print("Weights are selected using validation data only.\n")


best_weight = 0.5
best_f1 = -1.0

weight_results = []


for cnn_weight in [i / 10 for i in range(11)]:

    vit_weight = 1.0 - cnn_weight

    # Convert probabilities to tensors
    cnn_probs_tensor = torch.tensor(
        val_cnn_probs
    )

    vit_probs_tensor = torch.tensor(
        val_vit_probs
    )

    # Soft-voting ensemble
    ensemble_probs = (
        cnn_weight * cnn_probs_tensor
        + vit_weight * vit_probs_tensor
    )

    # Final prediction
    ensemble_predictions = torch.argmax(
        ensemble_probs,
        dim=1
    ).numpy()

    # Metrics
    accuracy = accuracy_score(
        val_labels,
        ensemble_predictions
    )

    precision = precision_score(
        val_labels,
        ensemble_predictions,
        zero_division=0
    )

    recall = recall_score(
        val_labels,
        ensemble_predictions,
        zero_division=0
    )

    f1 = f1_score(
        val_labels,
        ensemble_predictions,
        zero_division=0
    )

    weight_results.append(
        (
            cnn_weight,
            vit_weight,
            accuracy,
            precision,
            recall,
            f1
        )
    )

    print(
        f"CNN {cnn_weight:.1f} | "
        f"ViT {vit_weight:.1f} | "
        f"Accuracy: {accuracy * 100:.2f}% | "
        f"Precision: {precision * 100:.2f}% | "
        f"Recall: {recall * 100:.2f}% | "
        f"F1: {f1 * 100:.2f}%"
    )

    # Select best weight using F1
    if f1 > best_f1:

        best_f1 = f1

        best_weight = cnn_weight


best_vit_weight = 1.0 - best_weight


# ============================================================
# BEST WEIGHTS
# ============================================================

print("\n========================================")
print("BEST ENSEMBLE WEIGHT")
print("========================================")

print(f"CNN Weight: {best_weight:.1f}")
print(f"ViT Weight: {best_vit_weight:.1f}")
print(f"Validation F1: {best_f1 * 100:.2f}%")


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n========================================")
print("COLLECTING TEST PREDICTIONS")
print("========================================")

test_labels, test_cnn_probs, test_vit_probs = get_probabilities(
    test_loader
)


# ============================================================
# CONVERT TEST PROBABILITIES
# ============================================================

test_cnn_probs_tensor = torch.tensor(
    test_cnn_probs
)

test_vit_probs_tensor = torch.tensor(
    test_vit_probs
)


# ============================================================
# FINAL OPTIMIZED ENSEMBLE
# ============================================================

test_ensemble_probs = (
    best_weight * test_cnn_probs_tensor
    + best_vit_weight * test_vit_probs_tensor
)


# ============================================================
# INDIVIDUAL MODEL PREDICTIONS
# ============================================================

test_cnn_predictions = torch.argmax(
    test_cnn_probs_tensor,
    dim=1
).numpy()

test_vit_predictions = torch.argmax(
    test_vit_probs_tensor,
    dim=1
).numpy()

test_ensemble_predictions = torch.argmax(
    test_ensemble_probs,
    dim=1
).numpy()


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(labels, predictions):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# FINAL TEST METRICS
# ============================================================

cnn_metrics = calculate_metrics(
    test_labels,
    test_cnn_predictions
)

vit_metrics = calculate_metrics(
    test_labels,
    test_vit_predictions
)

ensemble_metrics = calculate_metrics(
    test_labels,
    test_ensemble_predictions
)


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

print("\n========================================")
print("FINAL TEST MODEL COMPARISON")
print("========================================")

print(
    f"\n{'Model':<12}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1 Score':>12}"
)

print("-" * 60)

print(
    f"{'CNN':<12}"
    f"{cnn_metrics[0] * 100:>11.2f}%"
    f"{cnn_metrics[1] * 100:>11.2f}%"
    f"{cnn_metrics[2] * 100:>11.2f}%"
    f"{cnn_metrics[3] * 100:>11.2f}%"
)

print(
    f"{'ViT':<12}"
    f"{vit_metrics[0] * 100:>11.2f}%"
    f"{vit_metrics[1] * 100:>11.2f}%"
    f"{vit_metrics[2] * 100:>11.2f}%"
    f"{vit_metrics[3] * 100:>11.2f}%"
)

print(
    f"{'Ensemble':<12}"
    f"{ensemble_metrics[0] * 100:>11.2f}%"
    f"{ensemble_metrics[1] * 100:>11.2f}%"
    f"{ensemble_metrics[2] * 100:>11.2f}%"
    f"{ensemble_metrics[3] * 100:>11.2f}%"
)


# ============================================================
# OPTIMIZED ENSEMBLE RESULTS
# ============================================================

print("\n========================================")
print("OPTIMIZED ENSEMBLE TEST RESULTS")
print("========================================")

print(f"CNN Weight : {best_weight:.1f}")
print(f"ViT Weight : {best_vit_weight:.1f}")

print(
    f"\nAccuracy : {ensemble_metrics[0] * 100:.2f}%"
)

print(
    f"Precision: {ensemble_metrics[1] * 100:.2f}%"
)

print(
    f"Recall   : {ensemble_metrics[2] * 100:.2f}%"
)

print(
    f"F1 Score : {ensemble_metrics[3] * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_ensemble_predictions
)

print("\n========================================")
print("OPTIMIZED ENSEMBLE CONFUSION MATRIX")
print("========================================")

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("OPTIMIZED ENSEMBLE CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        test_labels,
        test_ensemble_predictions,
        target_names=dataset.classes,
        zero_division=0
    )
)


# ============================================================
# CLASS MAPPING
# ============================================================

print("\nClass mapping:")

for index, class_name in enumerate(dataset.classes):

    print(f"{index} = {class_name}")


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("OPTIMIZED CNN + ViT ENSEMBLE COMPLETE")
print("========================================")