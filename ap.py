import streamlit as st
from PIL import Image
import io
import pandas as pd
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from inference import load_image_model, predict_image, load_text_model, predict_text

# ----------------------------
# SESSION STATE
# ----------------------------
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'image_model' not in st.session_state:
    st.session_state.image_model = None
    st.session_state.image_classes = []
    st.session_state.image_ready = False
if 'text_model' not in st.session_state:
    st.session_state.text_model = None
    st.session_state.text_ready = False

# ----------------------------
# APP CONFIG
# ----------------------------
st.set_page_config(
    page_title="🩺 Smart Medical Diagnosis Assistant",
    layout="wide",
    page_icon="🧬"
)

# ----------------------------
# CUSTOM STYLES
# ----------------------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: white;
        background: linear-gradient(90deg, #0077B6, #00B4D8);
        padding: 1.2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #0077B6;
        color: white;
        height: 3rem;
        font-size: 1rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00B4D8;
        transform: scale(1.02);
    }
    .pulse {
        width: 60px;
        height: 60px;
        background: #ff1744;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        margin: 20px auto;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.7; }
        70% { transform: scale(1); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<div class="main-title">
    <h1>🧬 Smart Medical Diagnosis Dashboard</h1>
    <p>AI-powered diagnosis using medical images & patient symptoms</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODELS
# ----------------------------
IMAGE_MODEL_PATH = 'models/modelfinal30.pth'
TEXT_MODEL_PATH = 'models/text_model.joblib'

if not st.session_state.image_ready:
    try:
        st.session_state.image_model, st.session_state.image_classes = load_image_model(IMAGE_MODEL_PATH)
        st.session_state.image_ready = True
        st.success("✅ Image model loaded successfully.")
    except Exception as e:
        st.warning(f"⚠️ Image model not loaded: {e}")

if not st.session_state.text_ready:
    try:
        st.session_state.text_model = load_text_model(TEXT_MODEL_PATH)
        st.session_state.text_ready = True
        st.success("✅ Text model loaded successfully.")
    except Exception as e:
        st.warning(f"⚠️ Text model not loaded: {e}")

# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------
st.sidebar.header("🧭 DashBoard")
menu = st.sidebar.radio("Go to", [
    "🏥 Patient Dashboard",
    "📷 Image Diagnosis",
    "💬 Symptom Analysis",
    "🧩 Combined Diagnosis",
    "📊 Patient History"
])

if st.session_state.current_patient:
    st.sidebar.info(f"👤 Active: {st.session_state.current_patient['name']} ({st.session_state.current_patient['id']})")

# ----------------------------
# PATIENT DASHBOARD
# ----------------------------
if menu == "🏥 Patient Dashboard":
    st.subheader("🏥 Patient Registration")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("👤 Name")
        with col2:
            age = st.number_input("🎂 Age", min_value=0, max_value=120, value=30)
        with col3:
            gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])

        patient_id = st.text_input("🆔 Patient ID", placeholder="e.g. P-2025")
        notes = st.text_area("🩺 Doctor Notes", placeholder="Enter brief medical history or current observations")

        submitted = st.form_submit_button("💾 Save Patient Record")

    if submitted:
        if not name or not patient_id:
            st.error("⚠️ Please fill in Name and Patient ID.")
        else:
            st.session_state.current_patient = {
                "id": patient_id,
                "name": name,
                "age": age,
                "gender": gender,
                "notes": notes,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.success(f"✅ Patient {name} registered successfully.")
            st.json(st.session_state.current_patient)

# ----------------------------
# IMAGE DIAGNOSIS
# ----------------------------
elif menu == "📷 Image Diagnosis":
    st.subheader("📷 Upload Medical Image")

    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    if uploaded:
        if not st.session_state.current_patient:
            st.warning("⚠️ Please register a patient first.")
        else:
            image = Image.open(io.BytesIO(uploaded.read())).convert('RGB')
            st.image(image, caption="🩻 Uploaded Image", use_column_width=True)

            if st.button("🔍 Analyze Image"):
                st.markdown('<div class="pulse"></div>', unsafe_allow_html=True)
                st.info("Analyzing image... Please wait ⏳")

                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.015)
                    progress.progress(i + 1)

                results = predict_image(st.session_state.image_model, st.session_state.image_classes, image)
                df = pd.DataFrame(results, columns=["Disease", "Probability"])
                top = df.iloc[0]

                st.success("✅ Analysis complete")
                st.markdown(f"""
                ### 🤖 AI Suggests:
                **{top['Disease']}** — confidence **{top['Probability']*100:.2f}%**
                """)

                # Confidence progress bars
                st.subheader("📊 Confidence Scores")
                for i, row in df.iterrows():
                    st.write(f"{row['Disease']}: **{row['Probability']*100:.2f}%**")
                    st.progress(float(row["Probability"]))

                st.session_state.history.append({
                    "mode": "Image",
                    "patient": st.session_state.current_patient,
                    "results": results
                })

# ----------------------------
# SYMPTOM ANALYSIS
# ----------------------------
elif menu == "💬 Symptom Analysis":
    st.subheader("💬 Enter Patient Symptoms")

    symptoms = st.text_area("Type symptoms here (e.g., headache, fatigue, fever)", height=150)

    if st.button("🧠 Analyze Symptoms"):
        if not st.session_state.text_ready:
            st.error("❌ Text model not available.")
        elif not st.session_state.current_patient:
            st.warning("⚠️ Register a patient first.")
        else:
            st.markdown('<div class="pulse"></div>', unsafe_allow_html=True)
            st.info("Analyzing symptoms... Please wait 🧬")

            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.012)
                progress.progress(i + 1)

            results = predict_text(st.session_state.text_model, symptoms)
            df = pd.DataFrame(results, columns=["Disease", "Probability"])
            top = df.iloc[0]

            st.success("✅ Analysis Complete")
            st.markdown(f"""
            ### 🩺 AI Suggests:
            Patient **{st.session_state.current_patient['name']}** may have **{top['Disease']}**  
            with **{top['Probability']*100:.2f}%** confidence.
            """)

            st.subheader("📊 Confidence Scores")
            for i, row in df.iterrows():
                st.write(f"{row['Disease']}: **{row['Probability']*100:.2f}%**")
                st.progress(float(row["Probability"]))

            st.session_state.history.append({
                "mode": "Text",
                "patient": st.session_state.current_patient,
                "symptoms": symptoms,
                "results": results
            })

# ----------------------------
# COMBINED DIAGNOSIS
# ----------------------------
elif menu == "🧩 Combined Diagnosis":
    st.subheader("🧩 Combined Diagnosis (Image + Text)")

    uploaded = st.file_uploader("Upload Image", type=['jpg','jpeg','png'])
    text_input = st.text_area("Enter Symptoms", height=120)

    if st.button("🔬 Run Combined Diagnosis"):
        if uploaded and text_input.strip():
            img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            st.image(img, use_column_width=True)

            st.markdown('<div class="pulse"></div>', unsafe_allow_html=True)
            st.info("Analyzing both sources... Please wait 🔄")

            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i + 1)

            img_res = predict_image(st.session_state.image_model, st.session_state.image_classes, img)
            txt_res = predict_text(st.session_state.text_model, text_input)

            combined = {}
            for d, p in img_res: combined[d] = combined.get(d, 0) + p * 0.6
            for d, p in txt_res: combined[d] = combined.get(d, 0) + p * 0.4
            combined_res = sorted(combined.items(), key=lambda x: x[1], reverse=True)

            df = pd.DataFrame(combined_res, columns=["Disease", "Probability"])
            top = df.iloc[0]

            st.balloons()
            st.success("✅ Combined AI Diagnosis Ready")
            st.markdown(f"""
            ### 🤖 Final Suggestion:
            **{top['Disease']}**  
            Confidence: **{top['Probability']*100:.2f}%**
            """)

            st.subheader("📊 Confidence Scores")
            for i, row in df.iterrows():
                st.write(f"{row['Disease']}: **{row['Probability']*100:.2f}%**")
                st.progress(float(row["Probability"]))

            st.session_state.history.append({
                "mode": "Combined",
                "patient": st.session_state.current_patient,
                "results": combined_res
            })
        else:
            st.warning("⚠️ Please provide both image and text inputs.")

# ----------------------------
# PATIENT HISTORY
# ----------------------------
elif menu == "📊 Patient History":
    st.subheader("📊 Diagnosis History")
    if not st.session_state.history:
        st.info("No history yet. Perform some diagnoses first.")
    else:
        for rec in st.session_state.history[::-1]:
            p = rec["patient"]
            st.markdown(f"""
            ### 🧾 {rec['mode']} Diagnosis for {p['name']} (ID: {p['id']})
            🕒 {p['time']} | 🎂 {p['age']} | ⚧ {p['gender']}
            """)
            df = pd.DataFrame(rec["results"], columns=["Disease", "Probability"])
            st.table(df.style.format({"Probability": "{:.2%}"}))
            st.markdown("---")
