import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = Path(__file__).parents[1] / "Dataset" / "Buy_Now_Pay_Later_BNPL_CreditRisk_Dataset.csv"
MODEL_PATH = Path(__file__).parents[1] / "models"

CATEGORICAL_COLS = ["employment_type", "product_category", "location", "customer_segment"]


def load_and_engineer(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.drop("user_id", axis=1)


    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    df["date"] = pd.to_datetime(df["transaction_date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df = df.drop(columns=["date", "transaction_date"])

    return df, encoders


if __name__ == "__main__":
    df, encoders = load_and_engineer(DATASET_PATH)

    X = df.drop("default_flag", axis=1)
    y = df["default_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.2, stratify=y, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Average Precision:", average_precision_score(y_test, y_pred_proba))

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print(feat_imp.head(10))

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH / "credit_risk_prediction_model.joblib")
    dump(list(X.columns), MODEL_PATH / "feature_columns.joblib")
    dump(encoders, MODEL_PATH / "label_encoders.joblib")
    print("Feature columns:", list(X.columns))
