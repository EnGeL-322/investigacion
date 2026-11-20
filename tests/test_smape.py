"""Pruebas del sMAPE (error porcentual absoluto medio simétrico)."""
import math

import numpy as np

from metricas import smape

Y = np.array([100.0, 200.0, 300.0, 400.0])
F = np.array([110.0, 180.0, 330.0, 400.0])


def test_smape_calculado_a_mano():
    esperado = 100 * (20 / 210 + 40 / 380 + 60 / 630 + 0) / 4
    assert math.isclose(smape(Y, F), esperado)


def test_smape_esta_entre_0_y_200():
    rng = np.random.default_rng(3)
    y = rng.normal(size=200)
    f = rng.normal(size=200)
    assert 0 <= smape(y, f) <= 200


def test_smape_maximo_es_200():
    assert math.isclose(smape([1.0], [0.0]), 200.0)


def test_smape_cero_entre_cero_vale_cero():
    assert smape([0.0], [0.0]) == 0.0


def test_smape_es_simetrico():
    assert math.isclose(smape([100.0], [150.0]), smape([150.0], [100.0]))


def test_smape_es_cero_si_el_pronostico_es_perfecto():
    assert smape(Y, Y) == 0.0
