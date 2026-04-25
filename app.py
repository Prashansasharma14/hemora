import streamlit as st
import numpy as np
import pickle
from datetime import datetime
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="HemoraAI", layout="wide")

# ------------------ DARK THEME CSS ------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #ff4b4b;
}
.stButton>button {
    background: linear-gradient(90deg, #ff4b4b, #ff0000);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open('pph_model.pkl', 'rb'))

# ------------------ HERO SECTION ------------------
st.markdown("""
<h1 style='text-align: center;'>🩸 HemoraAI</h1>
<p style='text-align: center; font-size:18px; color:gray;'>
AI-powered Postpartum Hemorrhage Risk Prediction
</p>
""", unsafe_allow_html=True)

st.caption("Clinical decision support tool | Not a substitute for medical judgment")

st.markdown("---")

# ------------------ INPUT LAYOUT ------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👩 Patient Profile")
    age = st.number_input("Age", 15, 60)
    parity = st.number_input("Parity", 0, 20)
    hb = st.number_input("Hemoglobin (g/dL)", 0.0, 17.0)
    bmi = st.number_input("BMI", 15.0, 40.0)

    # 🔥 Smart alert
    if hb < 7:
        st.error("⚠️ Severe anemia detected")

with col2:
    st.markdown("### ⚕️ Clinical Risk Factors")

    prev_lscs = st.selectbox("Previous LSCS", ["No", "Yes"])
    induction = st.selectbox("Induction of Labor", ["No", "Yes"])
    prolonged = st.selectbox("Prolonged Labor", ["No", "Yes"])
    multiple = st.selectbox("Multiple Pregnancy", ["No", "Yes"])
    bp = st.selectbox("Hypertension", ["No", "Yes"])
    prev_pph = st.selectbox("Previous PPH", ["No", "Yes"])
    placenta = st.selectbox("Placenta Issues", ["No", "Yes"])

# Convert
prev_lscs = 1 if prev_lscs == "Yes" else 0
induction = 1 if induction == "Yes" else 0
prolonged = 1 if prolonged == "Yes" else 0
multiple = 1 if multiple == "Yes" else 0
bp = 1 if bp == "Yes" else 0
prev_pph = 1 if prev_pph == "Yes" else 0
placenta = 1 if placenta == "Yes" else 0

st.markdown("---")

# ------------------ PREDICTION ------------------
if st.button("🚀 Predict Risk"):

    input_data = np.array([[age, parity, hb, prev_lscs, induction,
                            prolonged, multiple, bmi, bp,
                            prev_pph, placenta]])

    prob = model.predict_proba(input_data)[0][1]

    # ------------------ RISK CARD ------------------
    if prob > 0.7:
        risk_label = "High Risk"
        color = "#ff4b4b"
    elif prob > 0.4:
        risk_label = "Moderate Risk"
        color = "#ffa500"
    else:
        risk_label = "Low Risk"
        color = "#00c853"

    st.markdown(f"""
    <div style='background: linear-gradient(90deg, {color}, #222);
                padding:25px;
                border-radius:15px;
                text-align:center'>
    <h2>{risk_label}</h2>
    <h1>{round(prob*100,2)}%</h1>
    </div>
    """, unsafe_allow_html=True)

    # ------------------ PROGRESS ------------------
    st.progress(int(prob * 100))

    # ------------------ CONFIDENCE ------------------
    confidence = "High" if prob > 0.8 or prob < 0.2 else "Moderate"
    st.markdown(f"### 🔎 Model Confidence: {confidence}")

    st.markdown("---")

    # ------------------ CLINICAL INTERPRETATION ------------------
    st.markdown("### 📌 Clinical Interpretation")

    if prob > 0.7:
        st.error("High likelihood of PPH → Active management required.")
    elif prob > 0.4:
        st.warning("Moderate risk → Close monitoring advised.")
    else:
        st.success("Low risk → Routine care sufficient.")

    # ------------------ SUMMARY ------------------
    st.markdown("### 🧾 Patient Summary")
    st.write(f"""
    Age: {age}  
    Parity: {parity}  
    Hb: {hb}  
    BMI: {bmi}  
    """)

    # ------------------ EXPLAINABILITY ------------------
    st.markdown("### 🧬 Clinical Drivers")

    reasons = []

    if hb < 9:
        reasons.append("Anemia")
    if parity >= 3:
        reasons.append("High Parity")
    if prev_pph:
        reasons.append("Previous PPH")
    if prolonged:
        reasons.append("Prolonged Labor")
    if multiple:
        reasons.append("Multiple Pregnancy")
    if placenta:
        reasons.append("Placenta Issues")
    if bp:
        reasons.append("Hypertension")

    if reasons:
        for r in reasons:
            st.markdown(f"✔ {r}")
    else:
        st.info("No major contributing factors")

    st.markdown("---")

    # ------------------ RECOMMENDATIONS ------------------
    st.markdown("### 🏥 Suggested Plan")

    if prob > 0.7:
        st.error("Emergency preparedness, blood arrangement, ICU readiness.")
    elif prob > 0.4:
        st.warning("Close monitoring, uterotonics ready.")
    else:
        st.success("Routine monitoring.")

    st.markdown("---")

    # ------------------ DOWNLOAD ------------------
    report = f"""
HemoraAI Report
Generated: {datetime.now()}

Risk: {risk_label}
Score: {round(prob*100,2)}%

Factors: {', '.join(reasons) if reasons else 'None'}
"""

    st.download_button("📥 Download Report", report)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Version 4.0 | HemoraAI | Dr. Prashansa Sharma")