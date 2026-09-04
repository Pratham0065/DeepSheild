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

MODEL_PATH = PROJECT_ROOT / "results" / "best_cnn_model.pth"


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
# SAME DATASET SPLIT AS TRAINING
# ============================================================

train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print("\nTest dataset size:", len(test_dataset))


# ============================================================
# TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CREATE CNN MODEL
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

print("Best CNN model loaded successfully!")


# ============================================================
# EVALUATION
# ============================================================

model.eval()

all_labels = []
all_predictions = []


print("\nEvaluating CNN on test dataset...\n")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("========================================")
print("CNN TEST RESULTS")
print("========================================")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        all_labels,
        all_predictions,
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
print("CNN EVALUATION COMPLETE")
print("========================================")