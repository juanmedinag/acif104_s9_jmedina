import os
import joblib
import shap
import matplotlib.pyplot as plt

from .data_preprocessing import prepare_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks", "figuras")

os.makedirs(FIG_DIR, exist_ok=True)

def run_shap():
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

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    summary_path = os.path.join(FIG_DIR, "shap_summary.png")
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()

    print(f"Gráfico SHAP summary guardado en {summary_path}")

if __name__ == "__main__":
    run_shap()
