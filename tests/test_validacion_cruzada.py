"""Pruebas de la validación cruzada: el esquema cambia la conclusión."""
import pytest

# MAE de Ridge (alfa = 10) informado en el artículo sobre los mismos 117 objetivos
ARTICULO = {
    "2. Un territorio fuera": 0.062412,
    "3. Un anio objetivo fuera": 0.075509,
    "4. Bloques de anios": 0.062284,
    "5. Ventana expansiva": 0.082151,
}
PERSISTENCIA = 0.0648


def ridge(resultado):
    cv = resultado("cv_resultados.csv")
    r = cv[(cv.tarea == "regresion") & (cv.modelo == "ridge") & cv.muestra.str.startswith("mismas 117")]
    return r.set_index("esquema").MAE


@pytest.mark.parametrize("esquema", list(ARTICULO))
def test_mae_de_ridge_coincide_con_el_articulo(resultado, esquema):
    assert round(ridge(resultado)[esquema], 6) == pytest.approx(ARTICULO[esquema], abs=1e-6)


def test_kfold_aleatorio_parece_mejor_que_la_persistencia(resultado):
    assert ridge(resultado)["1. K-fold aleatorio"] < PERSISTENCIA


def test_ventana_expansiva_es_peor_que_la_persistencia(resultado):
    assert ridge(resultado)["5. Ventana expansiva"] > PERSISTENCIA


def test_todos_los_esquemas_evaluan_las_mismas_117_observaciones(resultado):
    cv = resultado("cv_resultados.csv")
    mismas = cv[cv.muestra.str.startswith("mismas 117")]
    assert (mismas.n == 117).all()
