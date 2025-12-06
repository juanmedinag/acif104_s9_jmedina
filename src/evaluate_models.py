import os
import json
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from .data_preprocessing import prepare_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend")

def evaluate():
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
    ) = prepare_data()

    model_path = os.path.join(MODELS_DIR, "gb_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("No se encontró gb_model.joblib. Ejecute primero train_models.py")

    model = joblib.load(model_path)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("Resultados en conjunto de prueba:")
    print(f"- Accuracy:  {acc:.3f}")
    print(f"- Precision: {prec:.3f}")
    print(f"- Recall:    {rec:.3f}")
    print(f"- F1-score:  {f1:.3f}")
    print(f"- AUC:       {auc:.3f}")
    print("Matriz de confusión:")
    print(cm)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Métricas guardadas en {metrics_path}")

if __name__ == "__main__":
    evaluate()
