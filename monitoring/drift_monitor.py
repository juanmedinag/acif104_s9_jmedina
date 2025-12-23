import numpy as np


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calcula el Population Stability Index (PSI) entre una distribución de
    referencia (expected) y una distribución actual (actual).

    Parameters
    ----------
    expected : np.ndarray
        Distribución de referencia (por ejemplo, datos de entrenamiento).
    actual : np.ndarray
        Distribución actual (por ejemplo, datos recientes en producción).
    buckets : int
        Número de bins para discretizar las distribuciones.

    Returns
    -------
    float
        Valor del PSI. Valores altos indican mayor grado de drift.
    """

    def _scale_range(data, bins):
        breakpoints = np.linspace(0, 100, bins + 1)
        return np.percentile(data, breakpoints)

    breakpoints = _scale_range(expected, buckets)
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Evitar divisiones por cero
    expected_percents = np.where(expected_percents == 0, 1e-6, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-6, actual_percents)

    psi_value = np.sum(
        (expected_percents - actual_percents)
        * np.log(expected_percents / actual_percents)
    )

    return psi_value


def check_drift(psi_value: float, threshold: float = 0.2) -> bool:
    """
    Evalúa si existe drift significativo según un umbral definido.

    Parameters
    ----------
    psi_value : float
        Valor del PSI calculado.
    threshold : float
        Umbral a partir del cual se considera que existe drift.

    Returns
    -------
    bool
        True si se detecta drift significativo, False en caso contrario.
    """
    return psi_value > threshold


if __name__ == "__main__":
    # Ejemplo conceptual de uso (no productivo)
    expected_data = np.random.normal(0, 1, 1000)
    actual_data = np.random.normal(0.5, 1.2, 1000)

    psi = compute_psi(expected_data, actual_data)
    drift_detected = check_drift(psi)

    print(f"PSI calculado: {psi:.4f}")
    print(f"Drift detectado: {drift_detected}")
