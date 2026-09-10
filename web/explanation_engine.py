"""
DeepS - Human Readable Explanation Engine

Converts CNN, ViT and Ensemble predictions into
human-readable explanations for the web dashboard.
"""


def format_probability(value):
    """Convert probability to percentage string."""
    return f"{value * 100:.2f}%"


def get_class_name(prob_fake, prob_real):
    """Return class with the higher probability."""
    return "FAKE" if prob_fake > prob_real else "REAL"


def get_confidence_level(probability):
    """
    Determine confidence level from the winning probability.
    """

    if probability >= 0.90:
        return "Very High"
    elif probability >= 0.75:
        return "High"
    elif probability >= 0.60:
        return "Moderate"
    elif probability >= 0.50:
        return "Low"
    else:
        return "Very Low"


def explain_model(model_name, fake_probability, real_probability):
    """
    Generate a human-readable explanation for one model.
    """

    prediction = get_class_name(fake_probability, real_probability)

    winning_probability = max(fake_probability, real_probability)

    confidence = get_confidence_level(winning_probability)

    if prediction == "REAL":
        explanation = (
            f"The {model_name} predicts the image as REAL with "
            f"{format_probability(real_probability)} probability. "
            f"The model assigns {format_probability(fake_probability)} "
            f"probability to the Fake class."
        )
    else:
        explanation = (
            f"The {model_name} predicts the image as FAKE with "
            f"{format_probability(fake_probability)} probability. "
            f"The model assigns {format_probability(real_probability)} "
            f"probability to the Real class."
        )

    return {
        "model": model_name,
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "confidence": confidence,
        "explanation": explanation,
    }


def explain_ensemble(
    cnn_fake,
    cnn_real,
    vit_fake,
    vit_real,
    ensemble_fake,
    ensemble_real,
):
    """
    Generate the final DeepS explanation.

    The current DeepS ensemble uses equal weighting
    between CNN and ViT.
    """

    final_prediction = get_class_name(
        ensemble_fake,
        ensemble_real
    )

    final_probability = max(
        ensemble_fake,
        ensemble_real
    )

    confidence = get_confidence_level(final_probability)

    cnn_prediction = get_class_name(cnn_fake, cnn_real)
    vit_prediction = get_class_name(vit_fake, vit_real)

    models_agree = cnn_prediction == vit_prediction

    if models_agree:

        if final_prediction == "REAL":
            explanation = (
                f"The final DeepS prediction is REAL with "
                f"{format_probability(ensemble_real)} probability. "
                f"Both the CNN and ViT models support the Real class. "
                f"The CNN assigns {format_probability(cnn_real)} probability "
                f"to Real, while the ViT assigns "
                f"{format_probability(vit_real)} probability to Real. "
                f"Because both models agree, the ensemble prediction "
                f"has stronger support."
            )

        else:
            explanation = (
                f"The final DeepS prediction is FAKE with "
                f"{format_probability(ensemble_fake)} probability. "
                f"Both the CNN and ViT models support the Fake class. "
                f"The CNN assigns {format_probability(cnn_fake)} probability "
                f"to Fake, while the ViT assigns "
                f"{format_probability(vit_fake)} probability to Fake. "
                f"Because both models agree, the ensemble prediction "
                f"has stronger support."
            )

    else:

        explanation = (
            f"The final DeepS prediction is {final_prediction} with "
            f"{format_probability(final_probability)} probability. "
            f"The CNN and ViT models disagree: the CNN predicts "
            f"{cnn_prediction}, while the ViT predicts {vit_prediction}. "
            f"Because the models provide conflicting evidence, the "
            f"ensemble confidence is lower than it would be if both "
            f"models strongly agreed."
        )

    return {
        "prediction": final_prediction,
        "probability": final_probability,
        "confidence": confidence,
        "cnn_prediction": cnn_prediction,
        "vit_prediction": vit_prediction,
        "models_agree": models_agree,
        "explanation": explanation,
    }


def generate_xai_explanation(
    cnn_prediction,
    vit_prediction,
):
    """
    Generate a general explanation for the XAI section.

    The actual spatial interpretation will later be connected
    to Grad-CAM and ViT Attention Rollout outputs.
    """

    cnn_text = (
        "The CNN explanation uses Grad-CAM to highlight image "
        "regions that contributed strongly to the CNN's prediction."
    )

    vit_text = (
        "The ViT explanation uses Attention Rollout to visualize "
        "how attention is distributed across image regions."
    )

    if cnn_prediction == vit_prediction:
        agreement_text = (
            "The CNN and ViT predictions are consistent, providing "
            "agreement between the two model architectures."
        )
    else:
        agreement_text = (
            "The CNN and ViT predictions differ. This disagreement "
            "is important because the final ensemble prediction "
            "combines evidence from both models."
        )

    return {
        "cnn": cnn_text,
        "vit": vit_text,
        "agreement": agreement_text,
    }
if __name__ == "__main__":

    result = explain_ensemble(
        cnn_fake=0.0009,
        cnn_real=0.9991,
        vit_fake=0.5822,
        vit_real=0.4178,
        ensemble_fake=0.2916,
        ensemble_real=0.7084,
    )

    print("\n========================================")
    print("DEEPS EXPLANATION ENGINE")
    print("========================================")

    print("\nFinal Prediction:")
    print(result["prediction"])

    print("\nProbability:")
    print(format_probability(result["probability"]))

    print("\nConfidence:")
    print(result["confidence"])

    print("\nCNN Prediction:")
    print(result["cnn_prediction"])

    print("\nViT Prediction:")
    print(result["vit_prediction"])

    print("\nModels Agree:")
    print(result["models_agree"])

    print("\nExplanation:")
    print(result["explanation"])