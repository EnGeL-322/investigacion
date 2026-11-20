"""Pruebas del MAE (error absoluto medio)."""
import math

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from metricas import mae

Y = np.array([100.0, 200.0, 300.0, 400.0])
F = np.array([110.0, 180.0, 330.0, 400.0])


def test_mae_calculado_a_mano():
    # errores absolutos: 10, 20, 30, 0  ->  60 / 4 = 15
    assert math.isclose(mae(Y, F), 15.0)


def test_mae_coincide_con_scikit_learn():
    rng = np.random.default_rng(1)
    y = rng.uniform(0.1, 1, 500)
    f = y + rng.normal(0, 0.1, 500)
    assert math.isclose(mae(y, f), mean_absolute_error(y, f))


def test_mae_es_cero_si_el_pronostico_es_perfecto():
    assert mae(Y, Y) == 0.0


def test_mae_no_cambia_al_desplazar_la_escala():
    assert math.isclose(mae(Y + 1000, F + 1000), mae(Y, F))


def test_mae_ignora_pares_con_nan():
    assert math.isclose(mae([1.0, np.nan, 3.0], [2.0, 5.0, 3.0]), 0.5)


def test_mae_rechaza_longitudes_distintas():
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])
