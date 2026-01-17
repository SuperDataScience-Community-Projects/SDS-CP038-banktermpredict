
# Streamlit Web App for Bank Term Deposit Prediction 

import streamlit as st
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('catboost').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)

# --- Page Config ---
st.set_page_config(page_title="Term Deposit Prediction", page_icon="🏦", layout="centered")

# --- Load Model ---
MODEL_FILE = "bankterm_pipeline.pkl"   # <-- same folder as app.py
LOCKED_THRESHOLD = 0.45

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

model = load_model(MODEL_FILE)

# --- Custom CSS (CHANGED: colors + design) ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #fff7f0 0%, #f4fbff 55%, #f3f0ff 100%);
            font-family: 'Segoe UI', sans-serif;
        }
        .big-title {
            font-size: 2.5em;
            color: #1f2d3d;
            font-weight: 800;
            letter-spacing: 0.8px;
            text-align: center;
            margin-bottom: 0.15em;
        }
        .subtitle {
            color: #6b4eff;
            font-size: 1.05em;
            text-align: center;
            margin-bottom: 1.2em;
        }
        .input-card {
            background: #fff0e6;
            border-radius: 16px;
            border: 1px solid rgba(107, 78, 255, 0.18);
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
            padding: 1.4em 1.1em 0.4em 1.1em;
            margin-bottom: 1em;
            margin-top: 1em;
        }
        .result-card {
            background: #ffffff;
            border-radius: 20px;
            border: 1px solid rgba(31, 45, 61, 0.10);
            box-shadow: 0 6px 24px rgba(31, 45, 61, 0.10);
            padding: 1.8em;
            margin: 1.8em auto;
            max-width: 520px;
        }
        .prob-meter {
            font-size: 1.35em;
            font-weight: 800;
            color: #1f2d3d;
            margin: 0.5em 0;
        }
        .success-label {
            color: #0f7b3d;
            font-weight: 800;
        }
        .fail-label {
            color: #c0392b;
            font-weight: 800;
        }
        .byline {
            font-size: 0.95em;
            color: #6b7280;
            margin-top: 2em;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Title (CHANGED text) ---
st.markdown('<div class="big-title">🏦 Term Deposit Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Mahtab’s Streamlit App • Locked threshold decision</div>', unsafe_allow_html=True)
st.info(f"Threshold (locked): {LOCKED_THRESHOLD:.2f}")

# --- Input Section Card (CHANGED text) ---
st.markdown("""
<div class="input-card">
  <span style="font-size:1.15em; color:#1f2d3d; font-weight:700;">🧾 Client Information</span>
  <div style="font-size:0.95em; color:#4b5563; margin-bottom: 0.5em;">
    Enter values below, then click <b>Predict</b> to see probability and YES/NO output.
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔧 Show/Hide Feature Inputs", expanded=True):
    age = st.number_input("Age", min_value=18, max_value=100, value=35, help="Client's age (18-100)")
    job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
                              "self-employed", "services", "student", "technician", "unemployed", "unknown"])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Has Credit in Default?", ["no", "yes"])
    balance = st.number_input("Account Balance", value=1000, help="Current balance")
    housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])
    day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
    month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    campaign = st.number_input("Number of Contacts During Campaign", value=1)
    pdays = st.number_input("Days Since Last Contact (-1 means never)", value=-1)
    previous = st.number_input("Number of Contacts Before This Campaign", value=0)
    poutcome = st.selectbox("Previous Outcome", ["unknown", "other", "failure", "success"])

X_input = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}])

# --- Predict Button (CHANGED hint text) ---
st.markdown("""<div style="margin-top: -0.8em; font-size:1.05em; color:#6b4eff; font-weight:600;">⬇️ Click “Predict” to get the result</div>""",
            unsafe_allow_html=True)

if st.button("🎯 Predict"):
    y_pred_proba = model.predict_proba(X_input)[:, 1][0]
    y_pred_label = "yes" if y_pred_proba >= LOCKED_THRESHOLD else "no"

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("Prediction Output")
    st.markdown(f"<div class='prob-meter'>Probability of YES: <b>{y_pred_proba:.1%}</b></div>", unsafe_allow_html=True)
    st.progress(y_pred_proba)

    if y_pred_label == "yes":
        st.markdown("<div class='success-label'>✅ Prediction: YES (Likely to Subscribe)</div>", unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("<div class='fail-label'>⚠️ Prediction: NO (Unlikely to Subscribe)</div>", unsafe_allow_html=True)

    st.write(f"**Threshold used:** {LOCKED_THRESHOLD:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer (CHANGED text) ---
st.markdown("---")
st.markdown("""
**What is this for?**  
This app helps estimate which clients are more likely to subscribe to a term deposit, using a trained ML pipeline.
""")
st.markdown('<div class="byline">Bank Term Deposit Predictor • Built by Mahtab</div>', unsafe_allow_html=True)
