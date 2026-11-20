"""Pruebas de la regresión múltiple (MCO) y de Ridge."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from idr_lib import DIMS, FEATURES, pairs


def test_r2_descriptivo_de_la_regresion_con_4_dimensiones(resultado):
    parte_a = resultado("reg_parte_a.csv", index_col=0).iloc[:, 0]
    assert int(parte_a["n"]) == 213
    assert round(float(parte_a["r2"]), 2) == 0.90


def test_mco_coincide_con_ecuaciones_normales(indice):
    """Los coeficientes de statsmodels coinciden con (X'X)^-1 X'y calculado a mano."""
    _, _, ix = indice
    p = pairs(ix, FEATURES).dropna(subset=["idr_y"] + [d + "_t" for d in DIMS])
    X = sm.add_constant(p[[d + "_t" for d in DIMS]].to_numpy())
    y = p.idr_y.to_numpy()
    beta_manual = np.linalg.solve(X.T @ X, X.T @ y)
    beta_sm = sm.OLS(y, X).fit().params
    assert np.allclose(beta_manual, beta_sm)


def test_mco_16_indicadores_y_persistencia(resultado):
    m = resultado("reg_metricas.csv", index_col=0)
    assert round(m.loc["persistencia", "MAE"], 4) == 0.0648
    assert round(m.loc["persistencia", "RMSE"], 4) == 0.0801
    assert round(m.loc["ols16", "MAE"], 4) == 0.0885
    assert round(m.loc["ols16", "RMSE"], 4) == 0.1048


def test_ninguna_regresion_supera_a_la_persistencia(resultado):
    m = resultado("reg_metricas.csv", index_col=0)
    assert m.MAE.idxmin() == "persistencia"
