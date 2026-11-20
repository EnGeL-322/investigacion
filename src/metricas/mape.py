"""MAPE: error porcentual absoluto medio.

    MAPE = (100/n) * Σ |y - f| / |y|

No está definido cuando y = 0 y depende del origen de la escala. En un índice min–max
es solo una métrica complementaria. Menor es mejor.
"""
import numpy as np

from ._comun import preparar


def mape(y, f, eps=0.0, return_n=False):
    """MAPE en %. Excluye los pares con |y| <= eps (con eps = 0, solo y exactamente 0).

    Si return_n es True, devuelve también cuántos pares se usaron.
    """
    y, f = preparar(y, f)
    keep = np.abs(y) > eps
    if not keep.any():
        raise ValueError("MAPE no definido: todos los observados son 0")
    valor = float(100 * np.mean(np.abs(y[keep] - f[keep]) / np.abs(y[keep])))
    return (valor, int(keep.sum())) if return_n else valor
