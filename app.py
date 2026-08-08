import time
import uuid
import math
import datetime
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load

# ----------------------------- Load artifacts -----------------------------
MODEL = load("models/credit_risk_prediction_model.joblib")
FEATURES = load("models/feature_columns.joblib")
ENCODERS = load("models/label_encoders.joblib")  # {col: fitted LabelEncoder}
EXPLAINER = shap.TreeExplainer(MODEL)


def encode(col, value):
    """Reuse the exact LabelEncoder fitted during training for this column."""
    return int(ENCODERS[col].transform([value])[0])


# ----------------------------- Page setup -----------------------------
st.set_page_config(page_title="BNPL Credit Risk Scanner", page_icon="💳", layout="wide")

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; font-size: 16px; }

/* app background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 0%, #1b1035 0%, #0c0e1c 45%, #060713 100%);
}
[data-testid="stHeader"] { background: transparent; }

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #120b28 0%, #0a0c18 100%);
    border-right: 1px solid rgba(148, 92, 255, 0.15);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Sora', sans-serif;
    color: #d9c9ff;
}

/* headings */
h1 { font-family: 'Sora', sans-serif !important; font-size: 2.3rem !important; font-weight: 800 !important;
     background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
     -webkit-background-clip: text; background-clip: text; color: transparent !important;
     letter-spacing: -0.02em; }
h2 { font-family: 'Sora', sans-serif !important; font-size: 1.4rem !important; font-weight: 700 !important; color: #eae6ff !important; }
h3 { font-family: 'Sora', sans-serif !important; font-size: 1.1rem !important; font-weight: 600 !important; color: #cfc6f5 !important; }
p, li, label, .stMarkdown { font-size: 0.95rem !important; line-height: 1.55 !important; color: #b9b6d6; }

.app-tagline { font-size: 1rem; color: #9791c2; margin-top: -0.8rem; }

/* glass cards */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(167,139,250,0.18);
    border-radius: 18px;
    padding: 1.4rem 1.2rem;
    backdrop-filter: blur(10px);
    text-align: center;
    transition: transform .25s ease, box-shadow .25s ease;
}
.glass-card:hover { transform: translateY(-4px); box-shadow: 0 10px 30px rgba(120,80,255,0.25); }
.glass-card .icon { font-size: 1.6rem; }
.glass-card .label { font-size: 0.78rem; color: #9791c2; text-transform: uppercase; letter-spacing: .08em; margin-top: .3rem; }
.glass-card .value { font-family: 'Sora', sans-serif; font-size: 1.7rem; font-weight: 700; color: #f3f1ff; margin-top: .1rem; }

/* verdict banner */
.decision-banner {
    border-radius: 16px; padding: 1.1rem 1.4rem; font-size: 1.02rem; font-weight: 600;
    display: flex; align-items: center; gap: .7rem;
    animation: fadeSlideUp .6s ease;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* section card wrapper */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(167,139,250,0.14);
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    margin-top: .5rem;
}

/* hero pulse rings */
.hero-wrap { display:flex; align-items:center; gap: 1rem; }
.pulse-dot {
    width: 12px; height: 12px; border-radius: 50%;
    background: #34d399; box-shadow: 0 0 0 rgba(52,211,153,.6);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(52,211,153,.6); }
    70%  { box-shadow: 0 0 0 12px rgba(52,211,153,0); }
    100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}

/* buttons */
.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; padding: 0.6rem 1rem !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(124,58,237,.35) !important; }

hr { border-color: rgba(167,139,250,0.15) !important; }
</style>
""", unsafe_allow_html=True)


def gauge_svg(proba: float, color: str, glow: str) -> str:
    """Animated circular gauge that fills to the churn/default probability."""
    r = 70
    circumference = 2 * math.pi * r
    target_offset = circumference * (1 - proba)
    uid = uuid.uuid4().hex[:8]
    return f"""
    <div style="display:flex;justify-content:center;">
    <svg width="200" height="200" viewBox="0 0 200 200">
        <circle cx="100" cy="100" r="{r}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="14"/>
        <circle id="gauge-{uid}" cx="100" cy="100" r="{r}" fill="none" stroke="{color}" stroke-width="14"
            stroke-linecap="round" stroke-dasharray="{circumference:.2f}"
            stroke-dashoffset="{circumference:.2f}"
            transform="rotate(-90 100 100)"
            style="filter: drop-shadow(0 0 8px {glow}); animation: fillGauge-{uid} 1.1s ease-out forwards;"/>
        <text x="100" y="94" text-anchor="middle" font-family="Sora, sans-serif" font-size="30"
              font-weight="800" fill="#f3f1ff">{proba*100:.0f}%</text>
        <text x="100" y="118" text-anchor="middle" font-family="Inter, sans-serif" font-size="12"
              fill="#9791c2">default risk</text>
    </svg>
    <style>
    @keyframes fillGauge-{uid} {{
        from {{ stroke-dashoffset: {circumference:.2f}; }}
        to   {{ stroke-dashoffset: {target_offset:.2f}; }}
    }}
    </style>
    </div>
    """


st.markdown(
    '<div class="hero-wrap"><span class="pulse-dot"></span>'
    '<span style="color:#9791c2; font-size:.85rem; letter-spacing:.1em; text-transform:uppercase;">live scoring model</span></div>',
    unsafe_allow_html=True
)
st.title("💳 BNPL Credit Risk Scanner")
st.markdown('<p class="app-tagline">Estimate a customer\'s default probability on a Buy-Now-Pay-Later purchase — instantly.</p>', unsafe_allow_html=True)
st.write("")

# ----------------------------- Sidebar inputs -----------------------------
with st.sidebar:
    st.header("🧾 Applicant Details")

    st.subheader("👤 Demographics")
    age = st.slider("Age", 18, 59, 30)
    employment_type = st.selectbox("Employment Type", sorted(ENCODERS['employment_type'].classes_))
    monthly_income = st.number_input("Monthly Income ($)", 0.0, 200000.0, 50000.0, step=500.0)
    location = st.selectbox("Location", sorted(ENCODERS['location'].classes_))

    st.subheader("📊 Credit Profile")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    debt_to_income_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.2, step=0.01)
    risk_score = st.number_input("Risk Score", 0.0, 400.0, 150.0, step=1.0)
    customer_segment = st.selectbox("Customer Segment", sorted(ENCODERS['customer_segment'].classes_))

    st.subheader("🛍️ Purchase & Repayment")
    purchase_amount = st.number_input("Purchase Amount ($)", 100.0, 5000.0, 2000.0, step=50.0)
    product_category = st.selectbox("Product Category", sorted(ENCODERS['product_category'].classes_))
    bnpl_installments = st.slider("BNPL Installments", 3, 12, 6)
    repayment_delay_days = st.slider("Repayment Delay (days)", 0, 33, 0)
    missed_payments = st.slider("Missed Payments", 0, 7, 0)
    app_usage_frequency = st.slider("App Usage Frequency (per week)", 1.0, 10.0, 5.0, step=0.1)
    transaction_date = st.date_input("Transaction Date", datetime.date.today())

    st.markdown("")
    run_prediction = st.button("⚡ Scan Applicant", type="primary", use_container_width=True)

# ----------------------------- Prediction -----------------------------
if not run_prediction:
    st.markdown(
        '<div class="section-card">🔎 Fill in the applicant\'s details in the sidebar, then click '
        '<b>Scan Applicant</b> to generate a live risk score.</div>',
        unsafe_allow_html=True
    )
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
        level, icon, bg, border, accent, glow = "High risk", "🔴", "#3a1518", "#a83232", "#ef4444", "rgba(239,68,68,.55)"
        note = "High default probability — recommend manual review or decline."
    elif proba >= 0.3:
        level, icon, bg, border, accent, glow = "Medium risk", "🟡", "#3a2f10", "#a88632", "#f59e0b", "rgba(245,158,11,.55)"
        note = "Moderate default probability — consider added conditions."
    else:
        level, icon, bg, border, accent, glow = "Low risk", "🟢", "#123a1a", "#329a4e", "#22c55e", "rgba(34,197,94,.55)"
        note = "Low default probability — safe to approve."

    gauge_col, stats_col = st.columns([1, 1.4])
    with gauge_col:
        st.markdown(gauge_svg(proba, accent, glow), unsafe_allow_html=True)

    with stats_col:
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f'<div class="glass-card"><div class="icon">🎯</div>'
                         f'<div class="label">Risk Level</div><div class="value">{level}</div></div>',
                         unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="glass-card"><div class="icon">📈</div>'
                         f'<div class="label">Credit Score</div><div class="value">{credit_score}</div></div>',
                         unsafe_allow_html=True)
        st.write("")
        st.markdown(
            f'<div class="decision-banner" style="background:{bg};border:1px solid {border};">'
            f'{icon} {note}</div>', unsafe_allow_html=True)

    if level == "Low risk":
        st.balloons()

    # ----------------------------- SHAP explanation -----------------------------
    st.write("")
    st.header("🧠 What's driving this score")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.caption("Bars show how much each factor pushed the prediction toward or away from default.")

    explanation = EXPLAINER(input_df)
    exp_default_class = explanation[0, :, 1]  # class 1 = default

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0c0e1c")
    ax.set_facecolor("#0c0e1c")
    shap.plots.waterfall(exp_default_class, max_display=10, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.clf()
    st.markdown('</div>', unsafe_allow_html=True)
