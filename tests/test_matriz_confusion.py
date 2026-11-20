"""Pruebas de la matriz de confusión y de las métricas por categoría."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

RUTA = Path(__file__).resolve().parent.parent / "src" / "07_matriz_confusion.py"
spec = importlib.util.spec_from_file_location("matriz_confusion", RUTA)
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)


def test_matriz_calculada_a_mano():
    y = [0, 0, 1, 1, 2, 2]
    p = [0, 1, 1, 2, 2, 2]
    esperado = np.array([[1, 1, 0],
                         [0, 1, 1],
                         [0, 0, 2]])
    assert (mc.matriz_confusion(y, p) == esperado).all()


def test_matriz_coincide_con_scikit_learn():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, 200)
    p = rng.integers(0, 3, 200)
    assert (mc.matriz_confusion(y, p) == confusion_matrix(y, p, labels=[0, 1, 2])).all()


def test_metricas_por_clase_coinciden_con_scikit_learn():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 3, 300)
    p = np.where(rng.random(300) < 0.7, y, rng.integers(0, 3, 300))
    pc = mc.metricas_por_clase(mc.matriz_confusion(y, p))
    pr, rc, f1, sop = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    assert np.allclose(pc.precision, pr)
    assert np.allclose(pc.recall, rc)
    assert np.allclose(pc.F1, f1)
    assert (pc.soporte.to_numpy() == sop).all()


def test_vp_fp_fn_vn_suman_el_total():
    m = mc.matriz_confusion([0, 1, 2, 2, 1, 0, 2], [0, 2, 2, 1, 1, 0, 0])
    pc = mc.metricas_por_clase(m)
    assert ((pc.VP + pc.FP + pc.FN + pc.VN) == m.sum()).all()


def test_matrices_del_estudio(resultado):
    mat = resultado("matrices_confusion.csv").set_index(["modelo", "real"])
    # logística: 33/38/19 en la diagonal; persistencia: 36/40/15
    assert [mat.loc[("logistica", c), f"pred_{c}"] for c in mc.CATEGORIAS] == [33, 38, 19]
    assert [mat.loc[("persistencia", c), f"pred_{c}"] for c in mc.CATEGORIAS] == [36, 40, 15]


def test_ningun_modelo_confunde_bajo_con_alto(resultado):
    res = resultado("matriz_confusion_resumen.csv")
    entrenados = res[res.modelo.isin(["logistica", "persistencia", "xgboost", "random_forest", "svm"])]
    assert (entrenados.errores_bajo_alto == 0).all()


def test_aciertos_de_logistica_y_persistencia(resultado):
    res = resultado("matriz_confusion_resumen.csv").set_index("modelo")
    assert res.loc["logistica", "aciertos"] == 90
    assert res.loc["persistencia", "aciertos"] == 91


@pytest.mark.parametrize("modelo,recall_alto", [("logistica", 0.95), ("persistencia", 0.75)])
def test_recall_de_la_categoria_alta(resultado, modelo, recall_alto):
    pc = resultado("metricas_por_clase.csv")
    fila = pc[(pc.modelo == modelo) & (pc.categoria == "Alto")].iloc[0]
    assert round(fila.recall, 3) == recall_alto
