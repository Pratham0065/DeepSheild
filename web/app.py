import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from models.cnn_model import DeepShieldCNN
from models.vit_model import DeepShieldViT
from web.explanation_engine import explain_ensemble


# ============================================================
# CONFIGURATION
# ============================================================

CNN_MODEL_PATH = PROJECT_ROOT / "results" / "best_cnn_model.pth"
VIT_MODEL_PATH = PROJECT_ROOT / "results" / "best_vit_model.pth"

IMAGE_SIZE = 224

CLASS_NAMES = {
    0: "fake",
    1: "real"
}

CNN_WEIGHT = 0.5
VIT_WEIGHT = 0.5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DeepS | Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.15rem;
        color: #777;
        margin-bottom: 2rem;
    }

    .result-real {
        padding: 25px;
        border-radius: 15px;
        background-color: #e8f5e9;
        border: 2px solid #43a047;
        text-align: center;
    }

    .result-fake {
        padding: 25px;
        border-radius: 15px;
        background-color: #ffebee;
        border: 2px solid #e53935;
        text-align: center;
    }

    .result-title {
        font-size: 2.3rem;
        font-weight: 800;
    }

    .result-confidence {
        font-size: 1.4rem;
        margin-top: 8px;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .model-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        text-align: center;
    }

    .model-name {
        font-size: 1.2rem;
        font-weight: 700;
    }

    .disclaimer {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff8e1;
        border: 1px solid #ffca28;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # IMPORTANT:
    # CNN constructor in models/cnn_model.py is DeepShieldCNN()
    # ViT constructor is DeepShieldViT(num_classes=2)
    # --------------------------------------------------------

    cnn = DeepShieldCNN()

    vit = DeepShieldViT(
        num_classes=2
    )

    # --------------------------------------------------------
    # Load CNN checkpoint
    # --------------------------------------------------------

    cnn_checkpoint = torch.load(
        CNN_MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(cnn_checkpoint, dict):

        if "model_state_dict" in cnn_checkpoint:

            cnn.load_state_dict(
                cnn_checkpoint["model_state_dict"]
            )

        elif "state_dict" in cnn_checkpoint:

            cnn.load_state_dict(
                cnn_checkpoint["state_dict"]
            )

        else:

            cnn.load_state_dict(
                cnn_checkpoint
            )

    else:

        cnn.load_state_dict(
            cnn_checkpoint
        )


    # --------------------------------------------------------
    # Load ViT checkpoint
    # --------------------------------------------------------

    vit_checkpoint = torch.load(
        VIT_MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(vit_checkpoint, dict):

        if "model_state_dict" in vit_checkpoint:

            vit.load_state_dict(
                vit_checkpoint["model_state_dict"]
            )

        elif "state_dict" in vit_checkpoint:

            vit.load_state_dict(
                vit_checkpoint["state_dict"]
            )

        else:

            vit.load_state_dict(
                vit_checkpoint
            )

    else:

        vit.load_state_dict(
            vit_checkpoint
        )


    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    cnn = cnn.to(DEVICE)
    vit = vit.to(DEVICE)

    cnn.eval()
    vit.eval()

    return cnn, vit


# ============================================================
# RUN INFERENCE
# ============================================================

def run_inference(image, cnn_model, vit_model):

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)


    # --------------------------------------------------------
    # CNN + ViT prediction
    # --------------------------------------------------------

    with torch.no_grad():

        cnn_output = cnn_model(
            image_tensor
        )

        vit_output = vit_model(
            image_tensor
        )


        cnn_probs = F.softmax(
            cnn_output,
            dim=1
        )[0]


        vit_probs = F.softmax(
            vit_output,
            dim=1
        )[0]


    # --------------------------------------------------------
    # 50:50 ENSEMBLE
    # --------------------------------------------------------

    ensemble_probs = (
        CNN_WEIGHT * cnn_probs
        +
        VIT_WEIGHT * vit_probs
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    cnn_prediction_index = int(
        torch.argmax(cnn_probs).item()
    )

    vit_prediction_index = int(
        torch.argmax(vit_probs).item()
    )

    ensemble_prediction_index = int(
        torch.argmax(ensemble_probs).item()
    )


    cnn_prediction = CLASS_NAMES[
        cnn_prediction_index
    ]

    vit_prediction = CLASS_NAMES[
        vit_prediction_index
    ]

    ensemble_prediction = CLASS_NAMES[
        ensemble_prediction_index
    ]


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    cnn_confidence = float(
        cnn_probs[
            cnn_prediction_index
        ].item()
    )

    vit_confidence = float(
        vit_probs[
            vit_prediction_index
        ].item()
    )

    ensemble_confidence = float(
        ensemble_probs[
            ensemble_prediction_index
        ].item()
    )


    # --------------------------------------------------------
    # Human-readable explanation
    # --------------------------------------------------------

    explanation = explain_ensemble(
        cnn_fake=float(
            cnn_probs[0].item()
        ),

        cnn_real=float(
            cnn_probs[1].item()
        ),

        vit_fake=float(
            vit_probs[0].item()
        ),

        vit_real=float(
            vit_probs[1].item()
        ),

        ensemble_fake=float(
            ensemble_probs[0].item()
        ),

        ensemble_real=float(
            ensemble_probs[1].item()
        )
    )


    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    return {

        "cnn_fake": float(
            cnn_probs[0].item()
        ),

        "cnn_real": float(
            cnn_probs[1].item()
        ),

        "vit_fake": float(
            vit_probs[0].item()
        ),

        "vit_real": float(
            vit_probs[1].item()
        ),

        "ensemble_fake": float(
            ensemble_probs[0].item()
        ),

        "ensemble_real": float(
            ensemble_probs[1].item()
        ),

        "cnn_prediction": cnn_prediction,

        "vit_prediction": vit_prediction,

        "ensemble_prediction": ensemble_prediction,

        "cnn_confidence": cnn_confidence,

        "vit_confidence": vit_confidence,

        "ensemble_confidence": ensemble_confidence,

        "explanation": explanation
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ DeepS")

    st.markdown(
        "### Deepfake Image Detection"
    )

    st.divider()

    st.markdown("### System")

    if torch.cuda.is_available():

        st.success(
            "GPU: CUDA available"
        )

        st.caption(
            torch.cuda.get_device_name(0)
        )

    else:

        st.info(
            "Running on CPU"
        )

    st.divider()

    st.markdown("### Detection Pipeline")

    st.write("✓ CNN")
    st.write("✓ Vision Transformer")
    st.write("✓ 50:50 Ensemble")
    st.write("✓ Human-readable Explanation")
    st.write("✓ Explainable AI")

    st.divider()

    st.caption(
        "DeepS is a research prototype for "
        "deepfake image detection."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ DeepS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deepfake Image Detection using CNN + Vision Transformer'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

try:

    with st.spinner(
        "Loading DeepS models..."
    ):

        cnn_model, vit_model = load_models()

    st.success(
        "DeepS models loaded successfully."
    )

except Exception as e:

    st.error(
        "Unable to load the trained models."
    )

    st.exception(e)

    st.stop()


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📤 Upload Image'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# IMAGE + ANALYZE BUTTON
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    col1, col2 = st.columns(
        [1, 1]
    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with col1:

        st.markdown(
            "### Input Image"
        )

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    with col2:

        st.markdown(
            "### Analysis"
        )

        st.write(
            "DeepS will analyze this image using:"
        )

        st.write(
            "• CNN"
        )

        st.write(
            "• Vision Transformer"
        )

        st.write(
            "• 50:50 Ensemble"
        )

        st.write("")

        analyze_button = st.button(
            "🔍 ANALYZE IMAGE",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    if analyze_button:

        with st.spinner(
            "Running DeepS analysis..."
        ):

            try:

                results = run_inference(
                    image,
                    cnn_model,
                    vit_model
                )

                st.session_state[
                    "results"
                ] = results

                st.session_state[
                    "image"
                ] = image

            except Exception as e:

                st.error(
                    "An error occurred during inference."
                )

                st.exception(e)


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state[
        "results"
    ]


    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🎯 DeepS Final Prediction'
        '</div>',
        unsafe_allow_html=True
    )


    prediction = results[
        "ensemble_prediction"
    ].upper()

    confidence = results[
        "ensemble_confidence"
    ]


    if prediction == "REAL":

        st.markdown(
            f"""
            <div class="result-real">

                <div class="result-title">
                    ✅ REAL
                </div>

                <div class="result-confidence">
                    Confidence:
                    <b>{confidence:.2%}</b>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-fake">

                <div class="result-title">
                    ⚠️ FAKE
                </div>

                <div class="result-confidence">
                    Confidence:
                    <b>{confidence:.2%}</b>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # CONFIDENCE BAR
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Ensemble Confidence'
        '</div>',
        unsafe_allow_html=True
    )

    st.progress(
        confidence
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🤖 Model Comparison'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    with col1:

        st.markdown(
            '<div class="model-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="model-name">'
            'CNN'
            '</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Fake Probability",
            f"{results['cnn_fake']:.2%}"
        )

        st.metric(
            "Real Probability",
            f"{results['cnn_real']:.2%}"
        )

        st.write(
            "**Prediction:** "
            + results[
                "cnn_prediction"
            ].upper()
        )

        st.write(
            "**Confidence:** "
            + f"{results['cnn_confidence']:.2%}"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # VIT
    # --------------------------------------------------------

    with col2:

        st.markdown(
            '<div class="model-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="model-name">'
            'Vision Transformer'
            '</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Fake Probability",
            f"{results['vit_fake']:.2%}"
        )

        st.metric(
            "Real Probability",
            f"{results['vit_real']:.2%}"
        )

        st.write(
            "**Prediction:** "
            + results[
                "vit_prediction"
            ].upper()
        )

        st.write(
            "**Confidence:** "
            + f"{results['vit_confidence']:.2%}"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # ENSEMBLE
    # --------------------------------------------------------

    with col3:

        st.markdown(
            '<div class="model-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="model-name">'
            '50:50 Ensemble'
            '</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Fake Probability",
            f"{results['ensemble_fake']:.2%}"
        )

        st.metric(
            "Real Probability",
            f"{results['ensemble_real']:.2%}"
        )

        st.write(
            "**Prediction:** "
            + results[
                "ensemble_prediction"
            ].upper()
        )

        st.write(
            "**Confidence:** "
            + f"{results['ensemble_confidence']:.2%}"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY TABLE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Probability Breakdown'
        '</div>',
        unsafe_allow_html=True
    )


    probability_data = {

        "Model": [
            "CNN",
            "Vision Transformer",
            "50:50 Ensemble"
        ],

        "Fake": [
            f"{results['cnn_fake']:.2%}",
            f"{results['vit_fake']:.2%}",
            f"{results['ensemble_fake']:.2%}"
        ],

        "Real": [
            f"{results['cnn_real']:.2%}",
            f"{results['vit_real']:.2%}",
            f"{results['ensemble_real']:.2%}"
        ],

        "Prediction": [
            results[
                "cnn_prediction"
            ].upper(),

            results[
                "vit_prediction"
            ].upper(),

            results[
                "ensemble_prediction"
            ].upper()
        ]
    }


    st.table(
        probability_data
    )


    # ========================================================
    # EXPLANATION
    # ========================================
    st.markdown(
        '<div class="section-title">'
        '🧠 Why did DeepS make this prediction?'
        '</div>',
        unsafe_allow_html=True
    )


    explanation = results[
        "explanation"
    ]


    if isinstance(
        explanation,
        dict
    ):

        if "summary" in explanation:

            st.info(
                explanation[
                    "summary"
                ]
            )

        if "confidence" in explanation:

            st.write(
                "**Confidence Level:** "
                + str(
                    explanation[
                        "confidence"
                    ]
                )
            )

        if "agreement" in explanation:

            st.write(
                "**Model Agreement:** "
                + str(
                    explanation[
                        "agreement"
                    ]
                )
            )

    else:

        st.info(
            str(
                explanation
            )
        )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🔬 Explainable AI'
        '</div>',
        unsafe_allow_html=True
    )


    st.info(
        "DeepS uses CNN Grad-CAM and Vision Transformer "
        "Attention Rollout to visualize image regions "
        "associated with the model predictions."
    )


    xai_col1, xai_col2 = st.columns(2)


    # --------------------------------------------------------
    # CNN GRAD-CAM
    # --------------------------------------------------------

    with xai_col1:

        st.markdown(
            "### 🔥 CNN Grad-CAM"
        )

        st.caption(
            "Highlights image regions that strongly "
            "influence the CNN prediction."
        )

        st.warning(
            "CNN Grad-CAM integration is the next step."
        )


    # --------------------------------------------------------
    # VIT ATTENTION
    # --------------------------------------------------------

    with xai_col2:

        st.markdown(
            "### 🧩 ViT Attention Rollout"
        )

        st.caption(
            "Visualizes regions receiving stronger "
            "attention from the Vision Transformer."
        )

        st.warning(
            "ViT Attention Rollout integration is the next step."
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
        """
        <div class="disclaimer">

        <b>⚠️ Research Prototype:</b>
        DeepS provides an AI-based prediction and
        explainability visualization. The result should
        not be treated as definitive proof that an image
        is authentic or manipulated.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DeepS — CNN + Vision Transformer Deepfake Detection "
    "Research Prototype"
)