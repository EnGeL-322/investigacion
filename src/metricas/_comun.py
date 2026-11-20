"""Funciones auxiliares compartidas por las métricas."""
import numpy as np


def preparar(y, f):
    """Convierte a vectores numpy, verifica longitudes y descarta pares con NaN o infinito."""
    y = np.asarray(y, dtype=float).ravel()
    f = np.asarray(f, dtype=float).ravel()
    if y.shape != f.shape:
        raise ValueError("y y f deben tener la misma longitud")
    ok = np.isfinite(y) & np.isfinite(f)
    if not ok.any():
        raise ValueError("no hay pares válidos")
    return y[ok], f[ok]
