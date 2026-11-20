"""R²: coeficiente de determinación.

    R² = 1 - Σ (y - f)² / Σ (y - ȳ)²

Mide ajuste, no capacidad de pronóstico fuera de la muestra.
"""
import numpy as np

from ._comun import preparar


def r2(y, f):
    """Coeficiente de determinación; NaN si y es constante."""
    y, f = preparar(y, f)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - f) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
