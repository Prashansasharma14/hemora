import streamlit as st
import numpy as np
import pickle
from datetime import datetime
import pandas as pd

# Page config
st.set_page_config(page_title="HemoraAI | PPH Risk Predictor", layout="wide")

# Load model
model = pickle.load(open('pph_model.pkl', 'rb'))

# 🔥 HERO HEADER (UPGRADED UI)
st.markdown("""
<h1 style='text-align: center; color: #b30000;'>🩸 HemoraAI</h1>
<p style='text-align: center; font-size:18px;'>
AI-powered prediction of Postpartum Hemorrhage risk
</p>
""", unsafe_allow_html=True)

st.caption("Clinical Decision Support Tool (For educational use only)")
st.markdown("---")

# Layout
col1, col2 = st.columns(2)

# LEFT SIDE
with col1:
    st.markdown("### 👩 Patient Profile")

    age = st.number_input("Age", 15, 60)
    parity = st.number_input("Parity", 0, 20)
    hb = st.number_input("Hemoglobin (g/dL)", 0.0, 17.0)
    bmi = st.number_input("BMI", 15.0, 40.0)

# RIGHT SIDE
with col2:
    st.markdown("### ⚕️ Clinical Risk Factors")

    prev_lscs = st.selectbox("Previous LSCS", ["No", "Yes"])
    induction = st.selectbox("Induction of Labor", ["No", "Yes"])
    prolonged = st.selectbox("Prolonged Labor", ["No", "Yes"])
    multiple = st.selectbox("Multiple Pregnancy", ["No", "Yes"])
    bp = st.selectbox("Hypertension", ["No", "Yes"])
    prev_pph = st.selectbox("Previous PPH", ["No", "Yes"])
    placenta = st.selectbox("Placenta Issues", ["No", "Yes"])

# Convert to numeric
prev_lscs = 1 if prev_lscs == "Yes" else 0
induction = 1 if induction == "Yes" else 0
prolonged = 1 if prolonged == "Yes" else 0
multiple = 1 if multiple == "Yes" else 0
bp = 1 if bp == "Yes" else 0
prev_pph = 1 if prev_pph == "Yes" else 0
placenta = 1 if placenta == "Yes" else 0

st.markdown("---")

# Prediction
if st.button("🔍 Predict Risk"):

    input_data = np.array([[age, parity, hb, prev_lscs, induction,
                            prolonged, multiple, bmi, bp,
                            prev_pph, placenta]])

    prob = model.predict_proba(input_data)[0][1]

    # 🔥 RISK DISPLAY CARD
    if prob > 0.7:
        risk_label = "High Risk"
        st.markdown(f"""
        <div style='background-color:#ffcccc;padding:20px;border-radius:10px'>
        <h2>🔴 High Risk ({round(prob*100,2)}%)</h2>
        </div>
        """, unsafe_allow_html=True)

    elif prob > 0.4:
        risk_label = "Moderate Risk"
        st.markdown(f"""
        <div style='background-color:#fff3cd;padding:20px;border-radius:10px'>
        <h2>🟠 Moderate Risk ({round(prob*100,2)}%)</h2>
        </div>
        """, unsafe_allow_html=True)

    else:
        risk_label = "Low Risk"
        st.markdown(f"""
        <div style='background-color:#d4edda;padding:20px;border-radius:10px'>
        <h2>🟢 Low Risk ({round(prob*100,2)}%)</h2>
        </div>
        """, unsafe_allow_html=True)

    # 🔥 PROGRESS BAR
    st.progress(int(prob * 100))

    # 🔥 CONFIDENCE
    confidence = "High" if prob > 0.8 or prob < 0.2 else "Moderate"
    st.markdown(f"### 🔎 Model Confidence: {confidence}")

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

    # 🔍 Explainability
    st.markdown("### 🧬 Clinical Drivers")

    reasons = []

    if hb < 9:
        reasons.append("Low Hemoglobin (Anemia)")
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
        for r in reasons:
            st.markdown(f"✅ {r}")
    else:
        st.info("No major high-risk factors detected")

    # 🤖 Model Insights
    try:
        st.markdown("### 🤖 Model Insights")
        feature_names = ['Age','Parity','Hb','Prev_LSCS','Induction',
                         'Prolonged','Multiple','BMI','BP','Prev_PPH','Placenta']

        importance = model.feature_importances_
        sorted_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

        for f, v in sorted_features[:3]:
            st.write(f"• {f} has high influence")
    except:
        pass

    st.markdown("---")

    # 📊 Risk Graph
    st.markdown("### 📊 Risk Visualization")

    chart = pd.DataFrame({
        "Risk": [prob]
    })

    st.bar_chart(chart)

    st.markdown("---")

    # Clinical Recommendations
    st.markdown("### 🏥 Suggested Plan")

    if prob > 0.7:
        st.error("Prepare emergency response, blood products, ICU readiness.")
    elif prob > 0.4:
        st.warning("Close monitoring and uterotonic preparedness required.")
    else:
        st.success("Routine care with standard monitoring.")

    st.markdown("---")

    # Download report
    st.markdown("### 📥 Download Report")

    report = f"""
HemoraAI PPH Risk Report
Generated: {datetime.now()}

Age: {age}
Parity: {parity}
Hb: {hb}
BMI: {bmi}

Risk Score: {round(prob*100,2)}%
Risk Category: {risk_label}

Factors:
{', '.join(reasons) if reasons else 'None'}
"""

    st.download_button(
        label="Download Report",
        data=report,
        file_name="pph_risk_report.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.caption("Version 3.0 | HemoraAI | Developed by Dr. Prashansa Sharma")