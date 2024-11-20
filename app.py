# FastAPI framework
from fastapi import FastAPI, UploadFile, File
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


@app.get("/alive")
async def isalive():

    return True


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


"""
@app.post("/predict_batch_file")
async def predict_batch_file(file: UploadFile = File(...)):
    # Leer el archivo en un DataFrame
    data = pd.read_csv(file.file)
    data["Hour"] = (data["Time"] // 3600) % 24

    # Realizar predicciones
    predictions = model.predict(data)
    response = {"predictions": predictions.tolist()}

    # Registrar el request y response
    log_request("predict_batch_file", {"file_name": file.filename}, {"num_predictions": len(predictions)})

    return response
"""


@app.post("/predict_batch_file")
async def predict_batch_file(file: UploadFile = File(...)):
    try:
        # Leer el archivo en un DataFrame
        data = pd.read_csv(file.file)

        # Validar que las columnas requeridas estén presentes
        required_columns = [
            "Time",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",
            "Amount",
            "Class",
        ]
        if not all(col in data.columns for col in required_columns):
            return {"error": "El archivo no contiene todas las columnas requeridas."}

        # Calcular la columna Hour si no está presente
        if "Hour" not in data.columns:
            data["Hour"] = (data["Time"] // 3600) % 24

        # Dividir en características (X) y etiquetas (y)
        X = data.drop(columns=["Class"])
        y = data["Class"]

        # Dividir en 80% entrenamiento y 20% prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Crear y entrenar un nuevo modelo LogisticRegression
        new_model = LogisticRegression(max_iter=1000, random_state=42)
        new_model.fit(X_train, y_train)

        # Hacer predicciones en los datos de prueba
        y_pred = new_model.predict(X_test)
        y_pred_prob = new_model.predict_proba(X_test)[:, 1]

        # Calcular métricas
        new_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        # Actualizar las métricas en metrics.pkl
        joblib.dump(new_metrics, "metrics.pkl")

        # Registrar en los logs
        log_request(
            "predict_batch_file",
            {"file_name": file.filename, "batch_size": len(data)},
            {"status": "success", "metrics": new_metrics},
        )

        # Responder con las métricas calculadas
        return {
            "status": "success",
            "message": "Modelo reentrenado con nuevos datos.",
            "metrics": new_metrics,
        }

    except Exception as e:
        # Manejo de errores
        log_request(
            "predict_batch_file",
            {"file_name": file.filename},
            {"status": "error", "error": str(e)},
        )
        return {
            "status": "error",
            "message": "Fallo en el reentrenamiento del modelo.",
            "details": str(e),
        }
