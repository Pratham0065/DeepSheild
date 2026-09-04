from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL RESULTS
# ============================================================

models = [
    "CNN",
    "ViT",
    "CNN + ViT Ensemble"
]

accuracy = [
    93.29,
    92.07,
    93.29
]

precision = [
    97.30,
    90.48,
    92.68
]

recall = [
    88.89,
    93.83,
    93.83
]

f1 = [
    92.90,
    92.12,
    93.25
]


# ============================================================
# CONFUSION MATRICES
# ============================================================

confusion_matrices = {

    "CNN": np.array([
        [81, 2],
        [9, 72]
    ]),

    "ViT": np.array([
        [75, 8],
        [5, 76]
    ]),

    "CNN + ViT Ensemble": np.array([
        [77, 6],
        [5, 76]
    ])
}


# ============================================================
# FUNCTION: SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(matrix, title, filename):

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(matrix)

    ax.set_title(title)

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Fake",
        "Real"
    ])

    ax.set_yticklabels([
        "Fake",
        "Real"
    ])

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# GENERATE CONFUSION MATRICES
# ============================================================

print("\n========================================")
print("GENERATING CONFUSION MATRICES")
print("========================================")

save_confusion_matrix(
    confusion_matrices["CNN"],
    "CNN Confusion Matrix",
    "cnn_confusion_matrix.png"
)

save_confusion_matrix(
    confusion_matrices["ViT"],
    "ViT Confusion Matrix",
    "vit_confusion_matrix.png"
)

save_confusion_matrix(
    confusion_matrices["CNN + ViT Ensemble"],
    "CNN + ViT Ensemble Confusion Matrix",
    "ensemble_confusion_matrix.png"
)


# ============================================================
# MODEL COMPARISON GRAPH
# ============================================================

print("\n========================================")
print("GENERATING MODEL COMPARISON")
print("========================================")

x = np.arange(
    len(models)
)

width = 0.18

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    x - 1.5 * width,
    accuracy,
    width,
    label="Accuracy"
)

ax.bar(
    x - 0.5 * width,
    precision,
    width,
    label="Precision"
)

ax.bar(
    x + 0.5 * width,
    recall,
    width,
    label="Recall"
)

ax.bar(
    x + 1.5 * width,
    f1,
    width,
    label="F1 Score"
)

ax.set_ylabel(
    "Score (%)"
)

ax.set_title(
    "CNN vs ViT vs CNN + ViT Ensemble"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    models
)

ax.set_ylim(
    80,
    100
)

ax.legend()

plt.tight_layout()

output_path = (
    OUTPUT_DIR
    / "model_comparison.png"
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


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("FINAL EVALUATION VISUALIZATION COMPLETE")
print("========================================")

print(
    f"Results saved in:\n{OUTPUT_DIR}"
)