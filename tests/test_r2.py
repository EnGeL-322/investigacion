"""Pruebas del R² (coeficiente de determinación)."""
import math

import numpy as np
from sklearn.metrics import r2_score

from metricas import r2


def test_r2_coincide_con_scikit_learn():
    rng = np.random.default_rng(4)
    y = rng.normal(size=100)
    f = y + rng.normal(0, 0.5, 100)
    assert math.isclose(r2(y, f), r2_score(y, f))


def test_r2_es_uno_si_el_ajuste_es_perfecto():
    assert r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_r2_es_nan_si_y_es_constante():
    assert math.isnan(r2([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))
