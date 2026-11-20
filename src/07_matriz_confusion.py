"""Paso 7. Matriz de confusión, falsos positivos/negativos y métricas por categoría.

Usa las 117 predicciones de clasificación (2020-2024) que deja el paso 2 en results/clasif_predicciones.csv.
Para cada modelo calcula:
  - la matriz de confusión 3x3 (filas = categoría real, columnas = categoría predicha);
  - por categoría: verdaderos positivos (VP), falsos positivos (FP), falsos negativos (FN),
    verdaderos negativos (VN), precisión, recall (sensibilidad), especificidad y F1;
  - accuracy y F1 macro, que deben coincidir con el paso 2.
Escribe tablas CSV y una figura en results/.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from idr_lib import OUT

warnings.filterwarnings("ignore")
CATEGORIAS = ["Bajo", "Medio", "Alto"]          # 0, 1, 2
MODELOS = {"logistica": "Regresión logística", "xgboost": "XGBoost", "persistencia": "Persistencia",
           "random_forest": "Random Forest", "svm": "Máquina de soporte vectorial", "mayoritaria": "Clase mayoritaria"}


def matriz_confusion(y, pred, k=3):
    """Matriz k x k: m[i, j] = casos de la categoría real i predichos como j."""
    m = np.zeros((k, k), dtype=int)
    for a, b in zip(np.asarray(y, int), np.asarray(pred, int)):
        m[a, b] += 1
    return m


def metricas_por_clase(m):
    """VP, FP, FN, VN, precisión, recall, especificidad y F1 de cada categoría a partir de la matriz."""
    total = m.sum()
    filas = []
    for i, nombre in enumerate(CATEGORIAS):
        vp = int(m[i, i])
        fp = int(m[:, i].sum() - vp)      # predijo i pero no era i
        fn = int(m[i, :].sum() - vp)      # era i pero predijo otra
        vn = int(total - vp - fp - fn)
        precision = vp / (vp + fp) if vp + fp else 0.0
        recall = vp / (vp + fn) if vp + fn else 0.0
        especificidad = vn / (vn + fp) if vn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        filas.append(dict(categoria=nombre, soporte=vp + fn, VP=vp, FP=fp, FN=fn, VN=vn,
                          precision=precision, recall=recall, especificidad=especificidad, F1=f1))
    return pd.DataFrame(filas)


if __name__ == "__main__":
    pred = pd.read_csv(OUT / "clasif_predicciones.csv")
    replica = pd.read_csv(OUT / "clasif_replica.csv", index_col=0)

    matrices, por_clase, resumen = [], [], []
    for modelo, nombre in MODELOS.items():
        d = pred[pred.model == modelo]
        m = matriz_confusion(d.y, d.pred)
        assert m.sum() == 117, modelo
        pc = metricas_por_clase(m)
        acc = np.trace(m) / m.sum()
        f1_macro = pc.F1.mean()
        # deben coincidir con el paso 2 (scikit-learn)
        assert np.isclose(acc, replica.loc[modelo, "accuracy"]), modelo
        assert np.isclose(f1_macro, replica.loc[modelo, "F1_macro"]), modelo

        for i, real in enumerate(CATEGORIAS):
            matrices.append(dict(modelo=modelo, real=real, **{f"pred_{c}": int(m[i, j]) for j, c in enumerate(CATEGORIAS)}))
        pc.insert(0, "modelo", modelo)
        por_clase.append(pc)
        errores_extremos = int(m[0, 2] + m[2, 0])     # Bajo<->Alto
        resumen.append(dict(modelo=modelo, aciertos=int(np.trace(m)), errores=int(m.sum() - np.trace(m)),
                            errores_bajo_alto=errores_extremos, accuracy=acc, F1_macro=f1_macro))

        print(f"\n=== {nombre}: {np.trace(m)} aciertos de {m.sum()} (accuracy {acc:.4f}, F1 macro {f1_macro:.4f})")
        print(pd.DataFrame(m, index=[f"real {c}" for c in CATEGORIAS], columns=[f"pred {c}" for c in CATEGORIAS]).to_string())
        print(pc.drop(columns="modelo").round(3).to_string(index=False))

    pd.DataFrame(matrices).to_csv(OUT / "matrices_confusion.csv", index=False)
    pd.concat(por_clase).to_csv(OUT / "metricas_por_clase.csv", index=False)
    pd.DataFrame(resumen).to_csv(OUT / "matriz_confusion_resumen.csv", index=False)

    # figura: matrices de los 6 modelos
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    for ax, (modelo, nombre) in zip(axes.ravel(), MODELOS.items()):
        d = pred[pred.model == modelo]
        m = matriz_confusion(d.y, d.pred)
        ax.imshow(m, cmap="Blues", vmin=0, vmax=55)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=13,
                        color="white" if m[i, j] > 30 else "#1B2733", fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(3), CATEGORIAS)
        ax.set_yticks(range(3), CATEGORIAS)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(f"{nombre}\n{np.trace(m)}/117 aciertos", fontsize=11)
    fig.suptitle("Matrices de confusión · 117 predicciones de categoría, 2020-2024", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_matrices_confusion.png", dpi=160)
    print("\nOK: matrices de confusión verificadas contra accuracy y F1 macro del paso 2")
