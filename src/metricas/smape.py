"""sMAPE: error porcentual absoluto medio simétrico.

    sMAPE = (100/n) * Σ 2|y - f| / (|y| + |f|)

Está acotado entre 0 % y 200 %; el caso 0/0 se toma como 0. Menor es mejor.
"""
import numpy as np

from ._comun import preparar


def smape(y, f):
    """sMAPE en % (variante con |y| + |f| en el denominador)."""
    y, f = preparar(y, f)
    den = np.abs(y) + np.abs(f)
    termino = np.where(den == 0, 0.0, 2 * np.abs(y - f) / np.where(den == 0, 1, den))
    return float(100 * np.mean(termino))
