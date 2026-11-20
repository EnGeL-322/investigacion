"""Pruebas de la construcción del IDR (ACP calibrado en 2015-2017 y cortes de Jenks)."""
import numpy as np
import pandas as pd

from idr_lib import DATA, FEATURES, Index


def publicado():
    pub = pd.read_csv(DATA / "idr_publicado.csv").melt("Territorio", var_name="year", value_name="celda")
    pub = pub[pub.celda.astype(str).str.contains(" ")]
    pub[["idr_pub", "cat_pub"]] = pub.celda.str.split(" ", expand=True)
    pub["idr_pub"] = pub.idr_pub.astype(float)
    pub["year"] = pub.year.astype(int)
    return pub.rename(columns={"Territorio": "region"})


def test_calibracion_usa_72_casos_de_2015_a_2017(indice):
    _, idx, _ = indice
    assert all(d["n_calibration"] == 72 for d in idx.diag)


def test_cortes_de_jenks(indice):
    _, idx, _ = indice
    assert abs(idx.breaks[1] - 0.360138) < 1e-4
    assert abs(idx.breaks[2] - 0.635553) < 1e-4


def test_se_reconstruyen_237_indices(indice):
    _, _, ix = indice
    assert ix.idr.notna().sum() == 237


def test_las_237_categorias_coinciden_con_las_publicadas(indice):
    _, _, ix = indice
    m = ix.merge(publicado(), on=["region", "year"])
    letras = m.category.map({0: "B", 1: "M", 2: "A"})
    assert len(m) == 237
    assert (letras == m.cat_pub).all()


def test_diferencia_maxima_con_el_idr_publicado_es_de_redondeo(indice):
    _, _, ix = indice
    m = ix.merge(publicado(), on=["region", "year"])
    assert (m.idr - m.idr_pub).abs().max() < 0.001


def test_cambiar_datos_futuros_no_altera_la_calibracion(indice):
    """Si se alteran los años 2018-2024, los parámetros del índice no cambian (no hay fuga del futuro)."""
    panel, idx, _ = indice
    alterado = panel.copy()
    alterado.loc[alterado.year >= 2018, FEATURES] *= 1.5
    idx2 = Index(FEATURES).fit(alterado[alterado.year <= 2017])
    assert np.isclose(idx.lo, idx2.lo) and np.isclose(idx.hi, idx2.hi)
    assert np.allclose(idx.breaks, idx2.breaks)
