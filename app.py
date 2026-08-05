import time
import datetime
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load
from pathlib import Path


MODEL = load(Path(__file__).parents[0]/"models/credit_risk_prediction_model.joblib")
FEATURES = load(Path(__file__).parents[0]/"models/feature_columns.joblib")
ENCODERS = load(Path(__file__).parents[0]/"models/label_encoders.joblib")  
EXPLAINER = shap.TreeExplainer(MODEL)


def encode(col, value):
    """Reuse the exact LabelEncoder fitted during training for this column."""
    return int(ENCODERS[col].transform([value])[0])



st.set_page_config(page_title="BNPL Credit Risk Scanner", page_icon="💳", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 16px; }

h1 { font-size: 1.9rem !important; font-weight: 700 !important; }
h2 { font-size: 1.35rem !important; font-weight: 600 !important; }
h3 { font-size: 1.1rem !important;  font-weight: 600 !important; }
p, li, label, .stMarkdown { font-size: 0.95rem !important; line-height: 1.5 !important; }

.app-tagline { font-size: 0.95rem; color: #9a9a9a; margin-top: -0.6rem; }

.metric-box {
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
    background: #14202e; border: 1px solid #1f3347;
}
.metric-box .label { font-size: 0.8rem; color: #8ba3b8; text-transform: uppercase; letter-spacing: .05em; }
.metric-box .value { font-size: 1.6rem; font-weight: 700; margin-top: .2rem; color: #eaf2f8; }

.decision-banner {
    border-radius: 10px; padding: 1rem 1.3rem; font-size: 1rem; font-weight: 600;
    display: flex; align-items: center; gap: .6rem;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 BNPL Credit Risk Scanner")
st.markdown('<p class="app-tagline">Estimate a customer\'s default probability on a Buy-Now-Pay-Later purchase.</p>', unsafe_allow_html=True)
st.divider()


with st.sidebar:
    st.header("Applicant Details")

    st.subheader("Demographics")
    age = st.slider("Age", 18, 59, 30)
    employment_type = st.selectbox("Employment Type", sorted(ENCODERS['employment_type'].classes_))
    monthly_income = st.number_input("Monthly Income ($)", 0.0, 200000.0, 50000.0, step=500.0)
    location = st.selectbox("Location", sorted(ENCODERS['location'].classes_))

    st.subheader("Credit Profile")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    debt_to_income_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.2, step=0.01)
    risk_score = st.number_input("Risk Score", 0.0, 400.0, 150.0, step=1.0)
    customer_segment = st.selectbox("Customer Segment", sorted(ENCODERS['customer_segment'].classes_))

    st.subheader("Purchase & Repayment")
    purchase_amount = st.number_input("Purchase Amount ($)", 100.0, 5000.0, 2000.0, step=50.0)
    product_category = st.selectbox("Product Category", sorted(ENCODERS['product_category'].classes_))
    bnpl_installments = st.slider("BNPL Installments", 3, 12, 6)
    repayment_delay_days = st.slider("Repayment Delay (days)", 0, 33, 0)
    missed_payments = st.slider("Missed Payments", 0, 7, 0)
    app_usage_frequency = st.slider("App Usage Frequency (per week)", 1.0, 10.0, 5.0, step=0.1)
    transaction_date = st.date_input("Transaction Date", datetime.date.today())

    st.markdown("")
    run_prediction = st.button("Scan Applicant", type="primary", use_container_width=True)


if not run_prediction:
    st.info("Fill in the applicant's details in the sidebar, then click **Scan Applicant**.")
else:
    row = {
        'age': age,
        'employment_type': encode('employment_type', employment_type),
        'monthly_income': monthly_income,
        'credit_score': credit_score,
        'purchase_amount': purchase_amount,
        'product_category': encode('product_category', product_category),
        'bnpl_installments': bnpl_installments,
        'repayment_delay_days': repayment_delay_days,
        'missed_payments': missed_payments,
        'app_usage_frequency': app_usage_frequency,
        'location': encode('location', location),
        'debt_to_income_ratio': debt_to_income_ratio,
        'risk_score': risk_score,
        'customer_segment': encode('customer_segment', customer_segment),
        'year': transaction_date.year,
        'month': transaction_date.month,
        'day': transaction_date.day,
        'weekday': transaction_date.weekday(),
    }
    input_df = pd.DataFrame([row])[FEATURES]

    with st.spinner("Scanning applicant..."):
        time.sleep(0.5)
        proba = MODEL.predict_proba(input_df)[0][1]

    if proba >= 0.6:
        level, icon, bg, border = "High risk", "🔴", "#3a1518", "#a83232"
        note = "High default probability — recommend manual review or decline."
    elif proba >= 0.3:
        level, icon, bg, border = "Medium risk", "🟡", "#3a2f10", "#a88632"
        note = "Moderate default probability — consider added conditions."
    else:
        level, icon, bg, border = "Low risk", "🟢", "#123a1a", "#329a4e"
        note = "Low default probability — safe to approve."

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="label">Default Probability</div>'
                     f'<div class="value">{proba*100:.1f}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="label">Risk Level</div>'
                     f'<div class="value">{level}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="label">Credit Score</div>'
                     f'<div class="value">{credit_score}</div></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="decision-banner" style="background:{bg};border:1px solid {border};margin-top:1rem;">'
        f'{icon} {note}</div>', unsafe_allow_html=True)

    progress = st.progress(0)
    for pct in range(int(proba * 100) + 1):
        progress.progress(pct)
        time.sleep(0.003)

  
    st.divider()
    st.header("What's driving this score")
    st.caption("Bars show how much each factor pushed the prediction toward or away from default.")

    explanation = EXPLAINER(input_df)
    exp_default_class = explanation[0, :, 1]  # class 1 = default

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(exp_default_class, max_display=10, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.clf()
