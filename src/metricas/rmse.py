"""RMSE: raíz del error cuadrático medio.

    RMSE = sqrt( (1/n) * Σ (y - f)² )

Penaliza más los errores grandes que el MAE; siempre RMSE >= MAE. Menor es mejor.
"""
import numpy as np

from ._comun import preparar


def rmse(y, f):
    """Raíz del error cuadrático medio entre observados `y` y pronósticos `f`."""
    y, f = preparar(y, f)
    return float(np.sqrt(np.mean((y - f) ** 2)))
