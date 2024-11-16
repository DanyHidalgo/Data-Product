import pandas as pd
import joblib

# visualización
import matplotlib.pyplot as plt
import seaborn as sns


# modelo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression

# warnings
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("./creditcard.csv")

df.shape
# Distribución de clases
class_counts = df["Class"].value_counts()
# print("Class Distribution:")
# print(class_counts)

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

corr_matrix = df.drop(columns=["Amount Range"]).corr()

df = df.drop(columns=["Amount Range"])


plt.figure(figsize=(20, 16))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Heatmap of Features (Excluding Amount Range)")
plt.show()

# Balance de clases

df = df.sample(frac=1)

fraud_df = df.loc[df["Class"] == 1]
non_fraud_df = df.loc[df["Class"] == 0][:492]

norm_distributed_df = pd.concat([fraud_df, non_fraud_df])

new_df = norm_distributed_df.sample(frac=1, random_state=42)

new_df = new_df.dropna()

for column in new_df.columns:
    if new_df[column].dtype == "object":
        new_df[column] = new_df[column].str.strip()
        new_df[column] = new_df[column].replace({"0-50": 25, "51-100": 75})
        new_df[column] = pd.to_numeric(new_df[column], errors="coerce")

label_encoder = LabelEncoder()
for column in new_df.columns:
    if (
        new_df[column].dtype == "object"
        or new_df[column].apply(lambda x: isinstance(x, str)).any()
    ):
        new_df[column] = label_encoder.fit_transform(new_df[column].astype(str))


new_df = new_df.dropna()

X = new_df.drop("Class", axis=1)
y = new_df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# MODELO
classifier = LogisticRegression()

# ENTRENAMIENTO
classifier.fit(X_train, y_train)

# PARAMETROS
log_reg_params = {"penalty": ["l1", "l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(
    LogisticRegression(solver="liblinear"), log_reg_params, cv=5
)
grid_log_reg.fit(X_train, y_train)

best_log_reg = grid_log_reg.best_estimator_

# ENTRENAMIENTO 2
best_log_reg.fit(X_train, y_train)

# PREDICCION
y_pred = best_log_reg.predict(X_test)

# GUARDAR MODELO COMO .PKL
joblib.dump(best_log_reg, "best_logistic_model.pkl")


# GUARDAR X_test y y_test
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

# DICCIONARIO METRICAS
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

# GUARDAR METRICAS
joblib.dump(metrics, "metrics.pkl")

# Guardar parametros en variables
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Mejores parámetros:", grid_log_reg.best_params_)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# Matriz de confusion grafica
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Matriz de Confusión")
plt.show()

# Data.scv
df_clean = df.dropna()

df_subset1 = df_clean.sample(n=50)
df_subset1.to_csv('Data1.csv', index=False)

df_subset2 = df_clean.sample(n=50)
df_subset2.to_csv('Data2.csv', index=False)

df_subset3 = df_clean.sample(n=50)
df_subset3.to_csv('Data3.csv', index=False)