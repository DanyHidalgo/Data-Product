# FastAPI framework
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import json
import os

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

import joblib

# Warnings filter (optional, for cleaner output)
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

# importar el modelo.pkl
try:
    model = joblib.load("best_logistic_model.pkl")
except FileNotFoundError:
    raise Exception(
        "Modelo no encontrado. Asegúrate de ejecutar credit_card.py primero para entrenar y guardar el modelo."
    )

# Cargar las métricas .pkl
try:
    metrics = joblib.load("metrics.pkl")
except FileNotFoundError:
    raise Exception(
        "Métricas no encontradas. Asegúrate de ejecutar credit_card.py primero para calcular y guardar las métricas."
    )


# Modelo para entrenar en tiempo real
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
    Class: int
    Hour: float
    Fraud_Spike: float


# Logs
def log_request(endpoint: str, request_data: dict, response_data: dict):
    timestamp = datetime.now()
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "request": request_data,
        "response": response_data,
    }

    # Ruta de carpetas
    log_dir = f"{endpoint}/year={timestamp.year}/month={timestamp.month:02d}/day={timestamp.day:02d}/hour={timestamp.hour:02d}"
    os.makedirs(log_dir, exist_ok=True)

    # Nombre del archivo de log
    log_file = os.path.join(log_dir, "log.json")

    # Escribir el log en formato JSON
    with open(log_file, "a") as file:
        file.write(json.dumps(log_entry) + "\n")


# Endpoint en tiempo real
@app.get("/metrics")
async def get_metrics():
    # Devuelve las métricas cargadas desde el archivo
    return metrics


# Endpoint en prediccion
@app.post("/predict")
async def predict(transaction: Transaction):
    # Convertir la entrada a DataFrame
    data = pd.DataFrame([transaction.dict()])
    data["Hour"] = (data["Time"] // 3600) % 24

    # Realizar la predicción
    prediction = model.predict(data)[0]
    response = {"prediction": int(prediction)}

    # Registrar el request y response
    log_request("predict", transaction.dict(), response)

    return response
