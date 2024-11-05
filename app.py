# FastAPI framework
from fastapi import FastAPI
from pydantic import BaseModel

# Data processing and model handling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Joblib for saving and loading the model (if needed for future reuse)
import joblib

# Warnings filter (optional, for cleaner output)
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

# Load the model if it's saved, or define it in the app
try:
    model = joblib.load("best_logistic_model.pkl")
except:
    # Define and train the model here (similar to your notebook code)
    df = pd.read_csv("creditcard.csv")
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Fraud_Spike"] = df["Class"].rolling(window=1000).mean()
    bins = [0, 50, 100, 200, 500, 1000, 5000, 10000, 50000]
    labels = [
        "0-50",
        "51-100",
        "101-200",
        "201-500",
        "501-1000",
        "1001-5000",
        "5001-10000",
        "10001+",
    ]
    df["Amount Range"] = pd.cut(df["Amount"], bins=bins, labels=labels, right=False)

    fraud_df = df.loc[df["Class"] == 1]
    non_fraud_df = df.loc[df["Class"] == 0][:492]
    new_df = (
        pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42).dropna()
    )

    label_encoder = LabelEncoder()
    for column in new_df.columns:
        if (
            new_df[column].dtype == "object"
            or new_df[column].apply(lambda x: isinstance(x, str)).any()
        ):
            new_df[column] = label_encoder.fit_transform(new_df[column].astype(str))

    X = new_df.drop("Class", axis=1)
    y = new_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log_reg_params = {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }
    grid_log_reg = GridSearchCV(
        LogisticRegression(solver="liblinear"), log_reg_params, cv=5
    )
    grid_log_reg.fit(X_train, y_train)

    best_log_reg = grid_log_reg.best_estimator_
    best_log_reg.fit(X_train, y_train)

    # Save model for future API restarts
    joblib.dump(best_log_reg, "best_logistic_model.pkl")

    model = best_log_reg


# Define a Pydantic model for input data validation
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.post("/predict")
async def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])

    # Preprocess any needed columns (e.g., feature engineering)
    data["Hour"] = (data["Time"] // 3600) % 24
    # Apply other preprocessing as necessary

    # Predict
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}


@app.get("/metrics")
async def metrics():
    # Verifica si ya tienes datos de prueba, si no, crea el conjunto de prueba
    try:
        y_pred = model.predict(X_test)
    except NameError:
        # Entrenamiento inicial (debe estar definido en el bloque de carga del modelo)
        df = pd.read_csv("creditcard.csv")
        # ... incluye el mismo código de procesamiento previo ...
        X = new_df.drop("Class", axis=1)
        y = new_df["Class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        y_pred = model.predict(X_test)

    # Cálculo de métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(
        y_test, y_pred
    ).tolist()  # Convierte a lista para JSON

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix,
    }


@app.get("/best_params")
async def best_params():
    # Devuelve los mejores parámetros del modelo obtenidos con GridSearchCV
    return {"best_params": grid_log_reg.best_params_}
