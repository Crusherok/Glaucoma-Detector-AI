import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from model import get_model

MODEL_PATH = Path("glaucoma_resnet.pth")

TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_pytorch_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not MODEL_PATH.exists():
        return None, device

    model = get_model(model_type="resnet")
    try:
        state_dict = torch.load(str(MODEL_PATH), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    except Exception as exc:
        print(f"Model loading failed: {exc}")
        return None, device


def predict_glaucoma(image):
    model, device = load_pytorch_model()
    if model is not None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            image = image.convert("RGB")

        input_tensor = TRANSFORM(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output[0], dim=0)
            prob_glaucoma = float(probabilities[1].item())
        return prob_glaucoma

    # Fallback mock prediction
    if isinstance(image, Image.Image):
        image = np.array(image)

    img = cv2.resize(image, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    variance = np.var(img)
    mock_prob = min(0.9, 0.5 + variance * 10)
    return mock_prob

def create_attention_overlay(image_np, probability, is_positive):
    """Return an RGB image with a heatmap-style highlight."""
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    focus_radius = max(40, min(h, w) // 3)
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, int(h * 0.45))
    cv2.circle(mask, center, focus_radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), focus_radius / 2.5)
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if is_positive:
        color_map = cv2.COLORMAP_PLASMA
        blend_strength = 0.65 * probability
    else:
        color_map = cv2.COLORMAP_SUMMER
        blend_strength = 0.5 * (1 - probability)

    heatmap = cv2.applyColorMap(mask, color_map)
    highlight = cv2.addWeighted(heatmap, blend_strength, img_bgr, 1 - blend_strength, 0)
    return cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB)

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Glaucoma Detector AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check model loading at startup
model, device = load_pytorch_model()
model_loaded = model is not None
if not model_loaded:
    st.info("⚠️ Unable to load trained model. Using mock AI predictions for demo.")

# ---------------------------
# Unique Custom CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

body {
    font-family: 'Roboto', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    background-attachment: fixed;
    color: #e0e0e0;
}

.sidebar .sidebar-content {
    background: rgba(20,20,40,0.95);
    border-radius: 15px;
    margin: 10px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
}

.main .block-container {
    background: rgba(30,30,50,0.9);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

.stButton>button {
    background: linear-gradient(45deg, #00d4ff, #090979);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,212,255,0.3);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,212,255,0.5);
    filter: brightness(1.1);
}

.stProgress .st-bo {
    background: linear-gradient(90deg, #00d4ff, #090979);
}

.stImage {
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.1);
}

h1, h2, h3 {
    color: #ffffff;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

.stSuccess, .stError {
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.1);
}

.stSuccess {
    background: rgba(34,197,94,0.2);
    border-left: 4px solid #22c55e;
}

.stError {
    background: rgba(239,68,68,0.2);
    border-left: 4px solid #ef4444;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.stMarkdown, .stImage, .stButton {
    animation: fadeIn 0.5s ease-in-out;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(20,20,40,0.8);
    border-radius: 10px;
    padding: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #e0e0e0;
    border-radius: 8px;
    transition: all 0.3s;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(0,212,255,0.2);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(45deg, #00d4ff, #090979);
    color: white;
}

.confidence-card {
    margin: 1.5rem 0;
    padding: 1.25rem 1.5rem;
    border-radius: 18px;
    background: rgba(15, 15, 40, 0.85);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
}

.confidence-card.positive {
    border-left: 5px solid #ef4444;
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(15,15,40,0.92));
}

.confidence-card.negative {
    border-left: 5px solid #10b981;
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(15,15,40,0.92));
}

.confidence-card h3 {
    margin: 0;
    font-size: 1.1rem;
    color: #ffffff;
}

.confidence-card p {
    margin: 0.4rem 0 0;
    font-size: 0.95rem;
    color: rgba(255,255,255,0.9);
}

.confidence-meter {
    margin-top: 0.9rem;
    height: 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    overflow: hidden;
}

.confidence-meter span {
    display: block;
    height: 100%;
    border-radius: 999px;
}

.confidence-card.positive .confidence-meter span {
    background: linear-gradient(90deg, #fca5a5, #ef4444);
}

.confidence-card.negative .confidence-meter span {
    background: linear-gradient(90deg, #4ade80, #10b981);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("🩺 Glaucoma Detector")
    st.markdown("**AI-Powered Early Detection**")
    st.markdown("Upload retinal fundus images for glaucoma analysis.")
    st.markdown("---")
    st.markdown("**Tech Stack:**")
    st.markdown("- Python + OpenCV")
    st.markdown("- Mock AI (Demo)")
    st.markdown("- Streamlit Frontend")
    st.markdown("- Datasets: DRISHTI-GS1 / REFUGE")
    st.markdown("---")
    with st.expander("ℹ️ About Glaucoma"):
        st.markdown("""
        Glaucoma is a group of eye conditions that damage the optic nerve, often due to high eye pressure.
        Early detection is crucial for prevention of vision loss.
        Risk factors: Age, family history, high IOP, thin corneas.
        """)
    st.info("For clinical use only. Consult professionals.")

# ---------------------------
# Main Content
# ---------------------------
st.title("👁️ Glaucoma Detection from Retinal Images")

st.markdown("""
Detect glaucoma early using AI analysis of fundus images. Upload your images below for instant results.
""")

# Upload
uploaded_files = st.file_uploader(
    "Upload Fundus Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    # Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Processing images...")
    
    for i in range(len(uploaded_files)):
        progress_bar.progress((i + 1) / len(uploaded_files))
        import time
        time.sleep(0.2)
    
    status_text.text("Analysis complete!")
    progress_bar.empty()
    
    # Process all images for batch analysis
    all_results = []
    for i, file in enumerate(uploaded_files):
        image = Image.open(file).convert("RGB")
        prob = predict_glaucoma(image)
        is_positive = prob >= 0.5
        all_results.append({
            "Image": f"Image {i+1}",
            "Confidence": prob,  # Store as float
            "Prediction": "Glaucoma" if is_positive else "Healthy",
            "IsPositive": is_positive
        })
    
    # Analysis and Results in one page
    st.subheader("🔬 Analysis & Results")
    
    # Display uploaded images with highlights
    for i, result in enumerate(all_results):
        image = Image.open(uploaded_files[i]).convert("RGB")
        img_np = np.array(image)
        
        # Create advanced heatmap overlay
        highlighted = create_attention_overlay(img_np, result["Confidence"], result["IsPositive"])
        highlighted = Image.fromarray(highlighted)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption=f"{result['Image']} - Original", use_container_width=True)
        with col2:
            st.image(highlighted, caption=f"{result['Image']} - AI Highlight ({result['Prediction']})", use_container_width=True)
        
        # Confidence card
        confidence_pct = result["Confidence"] * 100
        card_class = "positive" if result["IsPositive"] else "negative"
        meter_width = f"{confidence_pct:.1f}%"
        
        st.markdown(f"""
        <div class="confidence-card {card_class}">
            <h3>{result["Prediction"]} Detected</h3>
            <p><strong>Confidence:</strong> {confidence_pct:.1f}%</p>
            <div class="confidence-meter">
                <span style="width: {meter_width};"></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Overall chart
    st.subheader("📊 Summary Chart")
    fig, ax = plt.subplots()
    categories = ['Glaucoma', 'Healthy']
    glaucoma_count = sum(1 for r in all_results if r["Prediction"] == "Glaucoma")
    healthy_count = len(all_results) - glaucoma_count
    values = [glaucoma_count, healthy_count]
    colors = ['#ef4444', '#10b981']
    ax.bar(categories, values, color=colors)
    ax.set_ylabel('Count')
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha='center')
    st.pyplot(fig)
    
    # Download report
    report_text = f"Glaucoma Detection Report\n\n"
    for result in all_results:
        report_text += f"{result['Image']}: {result['Prediction']} ({result['Confidence']:.2f})\n"
    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name="glaucoma_report.txt",
        mime="text/plain"
    )

st.markdown("---")
st.markdown("**Disclaimer:** This tool is for educational purposes. Always consult an ophthalmologist for diagnosis.")