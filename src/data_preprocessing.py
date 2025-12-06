import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carpeta donde están los datos
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_dataset(filename: str = "ai4i_2020.csv") -> pd.DataFrame:
    """
    Carga el dataset principal. Si no existe, usa la muestra de ejemplo.
    """
    full_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(full_path):
        full_path = os.path.join(DATA_PATH, "ai4i_2020_sample.csv")

    df = pd.read_csv(full_path)
    return df


def prepare_data(
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepara los datos para el modelado:

    - Carga el dataset.
    - Construye la etiqueta binaria 'failure' a partir de 'Tool wear [min]'.
    - Separa X (features) e y (etiqueta).
    - Particiona en train / val / test (sin stratify).
    - Estandariza las variables numéricas con StandardScaler.
    """

    df = load_dataset()

    if "Tool wear [min]" not in df.columns:
        raise ValueError("El dataset no tiene la columna 'Tool wear [min]' necesaria para construir la etiqueta.")

    # Construimos la etiqueta binaria: falla si el desgaste es alto
    y = (df["Tool wear [min]"] > 200).astype(int).values

    # Features: todas las columnas numéricas menos la etiqueta y, opcionalmente,
    # podemos excluir 'Machine failure' si viene en el dataset original.
    drop_cols = ["Machine failure"] if "Machine failure" in df.columns else []
    drop_cols.append("Tool wear [min]")  # la usamos solo para construir y
    X = df.drop(columns=drop_cols, errors="ignore").values

    # 1) Train + (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(test_size + val_size),
        random_state=random_state,
    )

    # 2) (val + test)
    relative_test_size = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
    )

    # 3) Estandarización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
    )
