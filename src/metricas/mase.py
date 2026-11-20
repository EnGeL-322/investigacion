"""MASE: error absoluto medio escalado (Hyndman y Koehler, 2006).

    MASE = MAE del modelo / MAE del pronóstico ingenuo en entrenamiento

Valores menores que 1 indican que el modelo supera al pronóstico ingenuo.
"""
import numpy as np

from .mae import mae


def mase(y, f, y_train):
    """MAE del modelo dividido entre el MAE ingenuo (y_t = y_{t-1}) de la serie de entrenamiento."""
    y_train = np.asarray(y_train, dtype=float)
    y_train = y_train[np.isfinite(y_train)]
    if len(y_train) < 2:
        raise ValueError("MASE requiere al menos 2 datos de entrenamiento")
    escala = np.mean(np.abs(np.diff(y_train)))
    if escala == 0:
        raise ValueError("escala nula")
    return mae(y, f) / escala
