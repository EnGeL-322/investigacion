"""Pruebas de los métodos de series de tiempo (Tabla 3 del artículo)."""
import pytest

ARTICULO = {  # MAE y RMSE informados en el artículo
    "ingenuo": (0.0648, 0.0801),
    "ar1": (0.0656, 0.0817),
    "ses": (0.0705, 0.0847),
    "deriva": (0.0717, 0.0884),
    "holt_amortiguado": (0.0791, 0.0986),
}


@pytest.mark.parametrize("modelo", list(ARTICULO))
def test_errores_coinciden_con_el_articulo(resultado, modelo):
    m = resultado("ts_metricas.csv", index_col=0)
    mae_art, rmse_art = ARTICULO[modelo]
    assert int(m.loc[modelo, "n"]) == 117
    assert round(m.loc[modelo, "MAE"], 4) == mae_art
    assert round(m.loc[modelo, "RMSE"], 4) == rmse_art


def test_el_ingenuo_tiene_el_menor_error(resultado):
    assert resultado("ts_metricas.csv", index_col=0).MAE.idxmin() == "ingenuo"


def test_ingenuo_repite_el_valor_del_anio_anterior(resultado, indice):
    """El pronóstico ingenuo de t+1 es exactamente el IDR observado en t (solo usa el pasado)."""
    _, _, ix = indice
    pred = resultado("ts_predicciones.csv")
    ing = pred[pred.model == "ingenuo"].copy()
    ing["year_x"] = ing.year_y - 1
    m = ing.merge(ix[["region", "year", "idr"]], left_on=["region", "year_x"], right_on=["region", "year"])
    assert len(m) == len(ing)
    assert (m.pred - m.idr).abs().max() < 1e-12
