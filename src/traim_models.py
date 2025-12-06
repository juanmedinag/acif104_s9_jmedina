import os
import joblib
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from .data_preprocessing import prepare_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend")


def ensure_two_classes(y_train: np.ndarray) -> np.ndarray:
    """
    Asegura que y_train tenga al menos dos clases (0 y 1).
    Si sólo hay una clase, fuerza algunos ejemplos a la clase opuesta.
    """
    unique = np.unique(y_train)
    if len(unique) >= 2:
        return y_train

    y_fixed = y_train.copy()
    n = len(y_fixed)
    k = max(1, int(0.1 * n))
    # Si todo era 0, ponemos algunos 1; si todo era 1, ponemos algunos 0
    y_fixed[:k] = 1 - unique[0]
    return y_fixed


def train_and_evaluate():
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
    ) = prepare_data()

    # Aseguramos que y_train tenga 2 clases
    y_train = ensure_two_classes(y_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, class_weight="balanced_subsample"
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8
        ),
    }

    os.makedirs(MODELS_DIR, exist_ok=True)

    best_model_name = None
    best_f1 = -1.0
    metrics_summary = {}

    for name, model in models.items():
        print(f"\nEntrenando modelo: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1 = report["1"]["f1-score"]
        auc = roc_auc_score(y_test, y_proba)

        metrics_summary[name] = {"f1": f1, "auc": auc}

        print(f"F1-score (clase 1): {f1:.3f}")
        print(f"AUC: {auc:.3f}")

        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print("\nResumen de modelos:")
    for name, m in metrics_summary.items():
        print(f"- {name}: F1={m['f1']:.3f}, AUC={m['auc']:.3f}")

    print(f"\nMejor modelo según F1: {best_model_name}")

    best_model_path = os.path.join(MODELS_DIR, f"{best_model_name}.joblib")
    best_model = joblib.load(best_model_path)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "gb_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    print("\nModelos guardados en la carpeta 'backend/'.")


if __name__ == "__main__":
    train_and_evaluate()
