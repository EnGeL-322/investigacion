"""Paso 6. Figuras y resumen numerico (JSON) que alimentan el articulo."""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from idr_lib import OUT
cv = pd.read_csv(OUT/"cv_resultados.csv")
m = cv[(cv.tarea == "regresion") & (cv.muestra == "mismas 117 (2020-2024)")]
esq = sorted(m.esquema.unique()); lab = ["K-fold\naleatorio", "Un territorio\nfuera", "Un año\nfuera", "Bloques\nde años", "Ventana\nexpansiva"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
w = .27; x = np.arange(len(esq))
for k, (mod, col, nom) in enumerate([("ridge", "#176b87", "Ridge"), ("ols4", "#6aa84f", "MCO (4 dimensiones)"), ("rf", "#e69138", "Random Forest")]):
    ax[0].bar(x + (k - 1) * w, [m[(m.esquema == e) & (m.modelo == mod)].MAE.iloc[0] for e in esq], w, label=nom, color=col)
pers = m[m.modelo == "persistencia"].MAE.iloc[0]; ax[0].axhline(pers, color="k", ls="--", lw=1.2, label=f"Persistencia ({pers:.4f})")
ax[0].set(xticks=x, xticklabels=lab, ylabel="MAE (IDR)", title="Error según el esquema de validación\n(mismas 117 observaciones)"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_ylim(.04, .095)
c = cv[(cv.tarea == "clasificacion") & (cv.muestra == "mismas 117 (2020-2024)")]
for k, (mod, col, nom) in enumerate([("logistica", "#176b87", "Logística (C=1)"), ("persistencia", "#999999", "Persistencia")]):
    ax[1].bar(x + (k - .5) * .38, [c[(c.esquema == e) & (c.modelo == mod)].F1_macro.iloc[0] for e in esq], .38, label=nom, color=col)
ax[1].set(xticks=x, xticklabels=lab, ylabel="F1 macro", title="Clasificación (Bajo/Medio/Alto)", ylim=(.5, .85)); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT/"fig_validacion_cruzada.png", dpi=170)

reg = pd.read_csv(OUT/"reg_metricas.csv", index_col=0); ts = pd.read_csv(OUT/"ts_metricas.csv", index_col=0)
fig, ax = plt.subplots(figsize=(8.5, 4)); names = {"persistencia":"Persistencia","ar1":"AR(1) agrupado","deriva":"Deriva global","ols4":"MCO 4 dim.","xgb":"XGBoost","ridge":"Ridge","rf":"Random Forest","ols16":"MCO 16 ind."}
r = reg.sort_values("MAE"); ax.barh([names[i] for i in r.index][::-1], r.MAE[::-1], color=["#b5483a" if i != "persistencia" else "#0b3c5d" for i in r.index][::-1])
for i, v in enumerate(r.MAE[::-1]): ax.text(v + .001, i, f"{v:.4f}", va="center", fontsize=8)
ax.set(xlabel="MAE fuera de muestra (117 predicciones, 2020-2024)", title="Predicción del IDR del año siguiente: MAE por modelo"); ax.set_xlim(0, .1); fig.tight_layout(); fig.savefig(OUT/"fig_mae_modelos.png", dpi=170)

res = dict(reg=reg.round(4).to_dict("index"), ts=ts.round(4).to_dict("index"),
           reg_boot=pd.read_csv(OUT/"reg_bootstrap.csv", index_col=0).round(4).to_dict("index"), ts_boot=pd.read_csv(OUT/"ts_bootstrap.csv", index_col=0).round(4).to_dict("index"),
           parte_a=pd.read_csv(OUT/"reg_parte_a.csv", index_col=0).iloc[:, 0].to_dict(), ts_agg=pd.read_csv(OUT/"ts_agregado.csv", index_col=0).iloc[:, 0].to_dict(),
           coef=pd.read_csv(OUT/"reg_ols_dimensiones.csv", index_col=0).round(4).to_dict("index"))
json.dump(res, open(OUT/"resumen_articulo.json", "w"), ensure_ascii=False, indent=1, default=float); print("OK: figuras y JSON")
