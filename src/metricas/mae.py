"""MAE: error absoluto medio.

    MAE = (1/n) * Σ |y - f|

Se expresa en las mismas unidades que la variable (aquí, puntos del IDR). Menor es mejor.
"""
import numpy as np

from ._comun import preparar


def mae(y, f):
    """Error absoluto medio entre observados `y` y pronósticos `f`."""
    y, f = preparar(y, f)
    return float(np.mean(np.abs(y - f)))
