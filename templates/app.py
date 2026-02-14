import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import datetime

# --- Page Config & Professional Styling ---
st.set_page_config(page_title="Adenocarcinoma Detection AI", layout="wide")

# Custom CSS for a refined "Clinical Dark Mode"
st.markdown("""
    <style>
    /* Main background and font improvements */
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    
    /* Custom Card Styling */
    .metric-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Improve Button Aesthetics */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border: none;
        color: white;
    }

    /* Status Highlights */
    .status-positive { color: #ff4b4b; font-weight: bold; border: 1px solid #ff4b4b; padding: 10px; border-radius: 8px; }
    .status-negative { color: #00eb93; font-weight: bold; border: 1px solid #00eb93; padding: 10px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- Initialize Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Model Loading ---
MODEL_PATH = r"artifacts/trained_model/model.h5"

@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_trained_model()

# --- Sidebar: Metadata & History ---
with st.sidebar:
    st.title("📂 Diagnostic Info")
    st.divider()
    
    st.subheader("📊 Scan Properties")
    meta_placeholder = st.empty()
    
    if not st.session_state.history:
        meta_placeholder.info("Upload a scan to view data.")
    
    st.divider()
    
    st.subheader("📜 Recent Analysis")
    if not st.session_state.history:
        st.write("No history available.")
    else:
        for item in st.session_state.history[::-1][:5]: # Last 5 records
            color = "#ff4b4b" if item['result'] == "Positive" else "#00eb93"
            with st.container():
                st.markdown(f"""
                <div style="font-size: 0.85rem; margin-bottom: 10px; padding: 5px; border-bottom: 1px solid #30363d;">
                    <span style="color: {color}; font-weight: bold;">{item['result']}</span><br>
                    <small>{item['time']} | Conf: {item['conf']}</small>
                </div>
                """, unsafe_allow_html=True)

# --- Main UI ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🫁 Adenocarcinoma Chest Cancer Detection")
    st.markdown("##### *Deep Learning Analysis for Computed Tomography (CT) Imaging*")



st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Step 1: Input Data")
    uploaded_file = st.file_uploader("Select CT-Scan image (JPG/PNG)", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    container = st.container(border=True)
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        container.image(image, caption="Analyzable Medical Surface", use_container_width=True)
        
        # Update Metadata in Sidebar
        meta_placeholder.markdown(f"""
        - **File:** `{uploaded_file.name}`
        - **Size:** `{uploaded_file.size/1024:.1f} KB`
        - **Res:** `{image.size[0]}x{image.size[1]}`
        - **Time:** `{datetime.datetime.now().strftime('%H:%M')}`
        """)
    else:
        container.info("Please upload a CT scan to begin analysis.")

with col2:
    st.markdown("### 🔍 Step 2: Verdict")
    
    if uploaded_file and model:
        if st.button("RUN DIAGNOSTIC ANALYSIS"):
            with st.spinner("Analyzing Voxel Patterns..."):
                # Preprocessing
                img = image.resize((224, 224)) 
                img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
                
                # Prediction
                prediction = model.predict(img_array)
                risk_score = float(prediction[0][0])
                
                # Layout for Metrics
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Probability", f"{risk_score:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with m_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Confidence", f"{100 - abs(risk_score-0.5)*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.write("")
                st.progress(risk_score)
                
                # Verdict Display
                if risk_score > 0.5:
                    st.markdown('<div class="status-positive">🚨 POSITIVE: Adenocarcinoma Detected</div>', unsafe_allow_html=True)
                    result_text = "Positive"
                else:
                    st.markdown('<div class="status-negative">✅ NEGATIVE: No Malignancy Detected</div>', unsafe_allow_html=True)
                    result_text = "Negative"
                
                # Save to History
                st.session_state.history.append({
                    "time": datetime.datetime.now().strftime("%H:%M"),
                    "result": result_text,
                    "conf": f"{risk_score:.1%}"
                })
                
                st.toast(f"Analysis Complete: {result_text}")
    
    elif not model:
        st.warning(f"System Error: Model missing at {MODEL_PATH}")
    else:
        st.info("System Ready. Awaiting scan input...")

# --- Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
f_col1, f_col2 = st.columns(2)
f_col1.caption("© 2026 | Clinical Grade Diagnostic Tool")
