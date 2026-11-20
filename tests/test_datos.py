"""Pruebas de integridad del panel de datos (data/raw)."""
import pandas as pd

from idr_lib import DATA, FEATURES

PANEL = DATA / "panel_indicadores_2015_2024.csv"


def leer():
    return pd.read_csv(PANEL)


def test_panel_tiene_250_filas_y_16_indicadores():
    p = leer()
    assert p.shape == (250, 18)
    assert list(p.columns[2:]) == FEATURES


def test_panel_tiene_25_territorios_y_10_anios():
    p = leer()
    assert p.region.nunique() == 25
    assert sorted(p.year.unique()) == list(range(2015, 2025))


def test_cada_territorio_anio_aparece_una_sola_vez():
    assert not leer().duplicated(["region", "year"]).any()


def test_hay_71_valores_faltantes():
    assert int(leer()[FEATURES].isna().sum().sum()) == 71


def test_porcentajes_entre_0_y_100():
    p = leer()
    pct = [c for c in FEATURES if c not in ("ingreso_laboral", "escolaridad")]
    assert ((p[pct].dropna(how="all") >= 0) | p[pct].isna()).all().all()
    assert ((p[pct] <= 100) | p[pct].isna()).all().all()


def test_lima_no_tiene_panel_completo():
    p = leer()
    assert p[p.region == "Lima"][FEATURES].isna().any(axis=1).all()
