"""Pruebas del MASE (error absoluto medio escalado)."""
import math

import pytest

from metricas import mase


def test_mase_calculado_a_mano():
    # diferencias de entrenamiento |1|, |2|, |3| -> escala 2 ; MAE = 1 -> MASE = 0,5
    assert math.isclose(mase([5.0], [4.0], [1.0, 2.0, 4.0, 7.0]), 0.5)


def test_mase_rechaza_entrenamiento_de_un_solo_dato():
    with pytest.raises(ValueError):
        mase([5.0], [4.0], [1.0])


def test_mase_rechaza_escala_nula():
    with pytest.raises(ValueError):
        mase([5.0], [4.0], [3.0, 3.0, 3.0])
