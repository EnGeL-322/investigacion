"""Pruebas de la evaluación de clasificación (Tabla 2 del artículo)."""
import pytest

# (accuracy, F1 macro) informados en el artículo para 117 objetivos 2020-2024
ARTICULO = {
    "logistica": (0.7692, 0.7728),
    "persistencia": (0.7778, 0.7665),
    "random_forest": (0.7350, 0.7426),
    "svm": (0.6752, 0.6834),
    "mayoritaria": (0.4530, 0.2078),
}


@pytest.mark.parametrize("modelo", list(ARTICULO))
def test_resultados_coinciden_con_el_articulo(resultado, modelo):
    tabla = resultado("clasif_replica.csv", index_col=0)
    acc, f1 = ARTICULO[modelo]
    assert tabla.loc[modelo, "n"] == 117
    assert round(tabla.loc[modelo, "accuracy"], 4) == acc
    assert round(tabla.loc[modelo, "F1_macro"], 4) == f1


def test_ningun_modelo_gana_en_las_dos_metricas(resultado):
    tabla = resultado("clasif_replica.csv", index_col=0).drop(index="xgboost")
    assert tabla.accuracy.idxmax() == "persistencia"
    assert tabla.F1_macro.idxmax() == "logistica"


@pytest.mark.xfail(reason="Discrepancia abierta: el artículo informa F1 0,7686 para XGBoost y este código obtiene 0,7758", strict=True)
def test_xgboost_coincide_con_el_articulo(resultado):
    tabla = resultado("clasif_replica.csv", index_col=0)
    assert round(tabla.loc["xgboost", "F1_macro"], 4) == 0.7686


def test_predicciones_solo_de_2020_a_2024(resultado):
    pred = resultado("clasif_predicciones.csv")
    assert set(pred.year_y) == {2020, 2021, 2022, 2023, 2024}
