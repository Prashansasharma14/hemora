import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="HemoraAI | PPH Risk Predictor", layout="wide")

# Load model
model = pickle.load(open('pph_model.pkl', 'rb'))

# Header
st.title("🩸 HemoraAI")
st.subheader("AI-Based Postpartum Hemorrhage Risk Predictor")
st.caption("Clinical Decision Support Tool (For educational use only)")

st.markdown("---")

# Layout
col1, col2 = st.columns(2)

# LEFT SIDE (Patient Data)
with col1:
    st.markdown("### 👩 Patient Profile")

    age = st.number_input("Age", 15, 60)
    parity = st.number_input("Parity", 0, 20)
    hb = st.number_input("Hemoglobin (g/dL)", 0.0, 17.0)
    bmi = st.number_input("BMI", 15.0, 40.0)

# RIGHT SIDE (Clinical Data)
with col2:
    st.markdown("### ⚕️ Clinical Risk Factors")

    prev_lscs = st.selectbox("Previous LSCS", ["No", "Yes"])
    induction = st.selectbox("Induction of Labor", ["No", "Yes"])
    prolonged = st.selectbox("Prolonged Labor", ["No", "Yes"])
    multiple = st.selectbox("Multiple Pregnancy", ["No", "Yes"])
    bp = st.selectbox("Hypertension", ["No", "Yes"])
    prev_pph = st.selectbox("Previous PPH", ["No", "Yes"])
    placenta = st.selectbox("Placenta Issues", ["No", "Yes"])

# Convert categorical inputs
prev_lscs = 1 if prev_lscs == "Yes" else 0
induction = 1 if induction == "Yes" else 0
prolonged = 1 if prolonged == "Yes" else 0
multiple = 1 if multiple == "Yes" else 0
bp = 1 if bp == "Yes" else 0
prev_pph = 1 if prev_pph == "Yes" else 0
placenta = 1 if placenta == "Yes" else 0

st.markdown("---")

# Predict button
if st.button("🔍 Predict Risk"):

    input_data = np.array([[age, parity, hb, prev_lscs, induction,
                            prolonged, multiple, bmi, bp,
                            prev_pph, placenta]])

    prob = model.predict_proba(input_data)[0][1]

    # Risk Score
    st.markdown("## 🧠 Risk Assessment")
    st.metric(label="Risk Score", value=f"{round(prob*100,2)} %")

    # Risk Levels
    if prob > 0.7:
        st.error("🔴 High Risk of PPH")
    elif prob > 0.4:
        st.warning("🟠 Moderate Risk")
    else:
        st.success("🟢 Low Risk")

    st.markdown("---")

    # Patient Summary
    st.markdown("### 🧾 Patient Summary")
    st.write(f"""
    - Age: {age}
    - Parity: {parity}
    - Hb: {hb}
    - BMI: {bmi}
    - Hypertension: {'Yes' if bp else 'No'}
    - Previous PPH: {'Yes' if prev_pph else 'No'}
    """)

    st.markdown("---")

    # Explainability
    st.markdown("### 📊 Key Risk Factors")

    reasons = []

    if hb < 9:
        reasons.append("Low Hemoglobin")
    if parity >= 3:
        reasons.append("High Parity")
    if prev_pph:
        reasons.append("Previous PPH")
    if prolonged:
        reasons.append("Prolonged Labor")
    if multiple:
        reasons.append("Multiple Pregnancy")
    if placenta:
        reasons.append("Placenta-related Risk")
    if bp:
        reasons.append("Hypertension")

    if reasons:
        st.warning("⚠️ Contributing Factors:")
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.info("No major high-risk factors detected")

    st.markdown("---")

    # Clinical Recommendation
    st.markdown("### 🏥 Suggested Plan")

    if prob > 0.7:
        st.error("""
        - Active management of 3rd stage
        - Arrange blood products
        - Senior supervision
        - ICU readiness
        """)
    elif prob > 0.4:
        st.warning("""
        - Close monitoring
        - Prepare uterotonics
        - Senior review advised
        """)
    else:
        st.success("""
        - Routine care
        - Standard monitoring
        """)

# Footer
st.markdown("---")
st.caption("Version 2.0 | HemoraAI | Developed by Dr. Prashansa Sharma")