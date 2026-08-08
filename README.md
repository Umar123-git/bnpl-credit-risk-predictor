# 💳 BNPL Credit Risk Scanner

A Streamlit dashboard that estimates a customer's probability of default on a Buy-Now-Pay-Later (BNPL) purchase, with a live SHAP explanation of what's driving the score.

## Features

- **Live risk scoring** — RandomForest model trained on BNPL transaction data, balanced with SMOTE.
- **Explainable predictions** — SHAP waterfall chart shows exactly which factors pushed the score up or down.
- **Modern UI** — dark gradient theme, glassmorphism cards, animated circular gauge, responsive sidebar form.
- **Consistent categorical encoding** — each categorical column has its own saved `LabelEncoder`, so dashboard inputs are encoded exactly the way the model was trained (no train/inference mismatch).

## Tech Stack

- Python, scikit-learn, imbalanced-learn (SMOTE)
- SHAP for model explainability
- Streamlit for the dashboard
- Matplotlib for the waterfall chart

## Project Structure

```
.
├── app.py                  # Streamlit dashboard
├── train.py                # Model training script
├── requirements.txt
├── Dataset/
│   └── Buy_Now_Pay_Later_BNPL_CreditRisk_Dataset.csv
└── models/
    ├── credit_risk_prediction_model.joblib   # trained RandomForest
    ├── feature_columns.joblib                # feature order used at inference
    └── label_encoders.joblib                 # per-column LabelEncoders
```

## Setup

```bash
git clone <this-repo-url>
cd bnpl-credit-risk-scanner
pip install -r requirements.txt
```

## Retraining the model

```bash
python train.py
```

This reads `Dataset/Buy_Now_Pay_Later_BNPL_CreditRisk_Dataset.csv`, engineers date features (year/month/day/weekday) from `transaction_date`, applies SMOTE to balance the classes, trains a `RandomForestClassifier`, and saves the model, feature order, and label encoders to `models/`.

## Running the dashboard

```bash
streamlit run app.py
```

Fill in the applicant's details in the sidebar and click **Scan Applicant** to get a default probability, risk level, and a SHAP breakdown of the top contributing factors.

## Model

| Metric | Score |
|---|---|
| ROC-AUC | ~0.77 |
| Average Precision | ~0.69 |

Top predictive features: `credit_score`, `risk_score`, `repayment_delay_days`, `monthly_income`, `debt_to_income_ratio`.

## Deployment

Deployed via [Streamlit Community Cloud](https://share.streamlit.io) — connect this repo, set `app.py` as the main file, and it installs from `requirements.txt` automatically.

## Disclaimer

This is a personal/educational project. Predictions are based on a sample dataset and should not be used for real lending or credit decisions.
