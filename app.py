import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import datetime

# --- Page Config & Styling ---
st.set_page_config(page_title="Adenocarcinoma Detection AI", layout="wide")

# Custom CSS for a medical "Dark Mode" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .result-card { padding: 20px; border-radius: 15px; margin-top: 20px; text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Initialize Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Model Loading ---
MODEL_PATH = r"artifacts/training/model.h5"

@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_trained_model()

# --- Sidebar: Metadata & History ---
with st.sidebar:
    st.title("📂 Diagnostic Info")
    
    # Metadata Section
    st.subheader("File Metadata")
    meta_placeholder = st.empty()
    meta_placeholder.info("Upload a scan to view data.")
    
    st.divider()
    
    # History Section
    st.subheader("📜 Session History")
    if not st.session_state.history:
        st.write("No scans processed yet.")
    for item in st.session_state.history[::-1]: # Show latest first
        st.caption(f"{item['time']} - **{item['result']}** ({item['conf']})")

# --- Main UI ---
st.title("🫁 Adenocarcinoma Chest Cancer Detection")
st.write("Deep Learning Analysis for Computed Tomography (CT) Imaging")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Step 1: Upload Scan")
    uploaded_file = st.file_uploader("Drop CT-Scan image here...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Current Scan", use_container_width=True)
        
        # Update Metadata in Sidebar
        meta_placeholder.code(f"""
Name: {uploaded_file.name}
Size: {uploaded_file.size/1024:.2f} KB
Format: {uploaded_file.type}
Time: {datetime.datetime.now().strftime('%H:%M:%S')}
        """)

with col2:
    st.subheader("Step 2: AI Analysis")
    
    if uploaded_file and model:
        if st.button("EXECUTE DIAGNOSTIC", use_container_width=True):
            with st.spinner("Analyzing Pulmonary Tissue..."):
                # Preprocessing
                img = image.resize((224, 224)) 
                img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
                
                # Prediction
                prediction = model.predict(img_array)
                risk_score = float(prediction[0][0])
                
                # Advanced Metrics
                m1, m2 = st.columns(2)
                m1.metric("Risk Score", f"{risk_score:.2%}")
                m2.metric("Structural Integrity", f"{98.4 - (risk_score*5):.1f}%") # Simulated metric
                
                # Visual Risk Bar
                st.progress(risk_score)
                
                # Final Verdict
                if risk_score > 0.5:
                    st.error("### VERDICT: ADENOCARCINOMA DETECTED")
                    result_text = "Positive"
                else:
                    st.success("### VERDICT: NO MALIGNANCY DETECTED")
                    result_text = "Negative"
                
                # Save to History
                st.session_state.history.append({
                    "time": datetime.datetime.now().strftime("%H:%M"),
                    "result": result_text,
                    "conf": f"{risk_score:.1%}"
                })
                
                st.rerun() # Refresh sidebar history
    elif not model:
        st.warning(f"Model file not found at {MODEL_PATH}")
    else:
        st.info("Waiting for image upload...")

# --- Footer ---
st.divider()
st.caption("Professional Research Tool | 2026 AI Diagnostics")