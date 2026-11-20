"""Pruebas de los pares t -> t+1 usados para pronosticar."""
from idr_lib import FEATURES, pairs


def test_hay_225_pares_y_213_con_objetivo_observable(indice):
    _, _, ix = indice
    p = pairs(ix, FEATURES)
    assert len(p) == 225
    assert p.y.notna().sum() == 213


def test_el_objetivo_es_siempre_el_anio_siguiente(indice):
    _, _, ix = indice
    p = pairs(ix, FEATURES)
    assert (p.year_y == p.year_x + 1).all()


def test_prueba_externa_tiene_117_objetivos_2020_2024(indice):
    _, _, ix = indice
    p = pairs(ix, FEATURES)
    prueba = p[(p.year_y >= 2020) & p.y.notna()]
    assert len(prueba) == 117
    assert prueba.groupby("year_y").size().tolist() == [24, 24, 23, 23, 23]
