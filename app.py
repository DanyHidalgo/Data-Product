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


@app.get("/metrics")
async def get_metrics():
    # Devuelve las métricas cargadas desde el archivo
    return metrics
