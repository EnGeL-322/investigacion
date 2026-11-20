"""Pruebas del MAPE (error porcentual absoluto medio)."""
import math

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_percentage_error

from metricas import mae, mape

Y = np.array([100.0, 200.0, 300.0, 400.0])
F = np.array([110.0, 180.0, 330.0, 400.0])


def test_mape_calculado_a_mano():
    # (10/100 + 20/200 + 30/300 + 0) / 4 * 100 = 7,5 %
    assert math.isclose(mape(Y, F), 7.5)


def test_mape_coincide_con_scikit_learn():
    rng = np.random.default_rng(1)
    y = rng.uniform(0.1, 1, 500)
    f = y + rng.normal(0, 0.1, 500)
    assert math.isclose(mape(y, f), 100 * mean_absolute_percentage_error(y, f), rel_tol=1e-9)


def test_mape_excluye_observados_iguales_a_cero():
    valor, n = mape([0.0, 0.5, 1.0], [0.1, 0.4, 1.1], return_n=True)
    assert n == 2


def test_mape_falla_si_todos_los_observados_son_cero():
    with pytest.raises(ValueError):
        mape([0.0, 0.0], [1.0, 1.0])


def test_mape_es_asimetrico():
    assert not math.isclose(mape([100.0], [150.0]), mape([150.0], [100.0]))


def test_mape_depende_del_origen_de_la_escala():
    """Mismo error absoluto (0,03): al sumar 1 al índice, el MAPE cae a menos de la mitad y el MAE no cambia."""
    y = np.array([0.15, 0.30, 0.55, 0.80])
    f = y + np.array([0.03, -0.03, 0.03, -0.03])
    assert math.isclose(mae(y + 1, f + 1), mae(y, f))
    assert mape(y + 1, f + 1) < mape(y, f) / 2
