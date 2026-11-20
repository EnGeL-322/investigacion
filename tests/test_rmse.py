"""Pruebas del RMSE (raíz del error cuadrático medio)."""
import math

import numpy as np
from sklearn.metrics import mean_squared_error

from metricas import mae, rmse

Y = np.array([100.0, 200.0, 300.0, 400.0])
F = np.array([110.0, 180.0, 330.0, 400.0])


def test_rmse_calculado_a_mano():
    # errores al cuadrado: 100, 400, 900, 0  ->  sqrt(1400 / 4) = sqrt(350)
    assert math.isclose(rmse(Y, F), math.sqrt(350))


def test_rmse_coincide_con_scikit_learn():
    rng = np.random.default_rng(1)
    y = rng.uniform(0.1, 1, 500)
    f = y + rng.normal(0, 0.1, 500)
    assert math.isclose(rmse(y, f), math.sqrt(mean_squared_error(y, f)))


def test_rmse_siempre_mayor_o_igual_que_mae():
    rng = np.random.default_rng(2)
    for _ in range(50):
        y = rng.normal(size=30)
        f = rng.normal(size=30)
        assert rmse(y, f) >= mae(y, f) - 1e-12


def test_rmse_penaliza_mas_un_error_grande():
    # mismo MAE (2), pero un solo error grande sube el RMSE
    parejo = rmse([0, 0, 0, 0], [2, 2, 2, 2])
    concentrado = rmse([0, 0, 0, 0], [8, 0, 0, 0])
    assert concentrado > parejo
