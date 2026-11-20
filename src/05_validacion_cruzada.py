"""Paso 5. Validacion cruzada: ¿cuanto cambia el error estimado segun el esquema?

Esquemas (todos sobre pares t -> t+1 con objetivo observado):
  1. K-fold aleatorio (k=5, 10 repeticiones)       -> mezcla anios y territorios (filtra informacion futura y del mismo territorio)
  2. Dejar un territorio fuera (LOTO)              -> predice territorios no vistos (con anios futuros en el entrenamiento)
  3. Dejar un anio objetivo fuera (LOYO)           -> usa anios posteriores para predecir uno anterior
  4. Bloques contiguos de anios (Bergmeir 2012)     -> 5 bloques, entrena con el resto
  5. Ventana expansiva / origen movil (articulo)    -> solo entrena con el pasado; objetivos 2020-2024
Hiperparametros fijos (Ridge alpha=10, RF max_depth=5, Logistica C=1) para aislar el efecto del esquema.
Se reporta el error (a) con todas las observaciones que cada esquema evalua y (b) en las MISMAS 117 observaciones 2020-2024.
"""
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from idr_lib import *
from metricas import *
warnings.filterwarnings("ignore")
DT = [d + "_t" for d in DIMS]
panel, idx, ix = build_indexed()
D = pairs(ix, FEATURES)
S = D[D.idr_y.notna() & D.idr_t.notna()].reset_index(drop=True)
print(f"Observaciones evaluables: {len(S)} (objetivos {S.year_y.min()}-{S.year_y.max()}), territorios: {S.region.nunique()}")
mask117 = (S.year_y >= 2020).values; assert mask117.sum() == 117

# ------------------------------------------------------------ generadores de particiones (con pruebas)
def esquemas(S, seed=0):
    n = len(S); out = {}
    reps = []
    for r in range(10):
        reps.append([(tr, te) for tr, te in KFold(5, shuffle=True, random_state=SEED + r).split(np.arange(n))])
    out["1. K-fold aleatorio"] = reps
    g = pd.factorize(S.region)[0]; out["2. Un territorio fuera"] = [list(LeaveOneGroupOut().split(np.arange(n), groups=g))]
    out["3. Un anio objetivo fuera"] = [[(np.where(S.year_y != y)[0], np.where(S.year_y == y)[0]) for y in sorted(S.year_y.unique())]]
    yrs = sorted(S.year_y.unique()); bl = [yrs[k:k+2] for k in range(0, len(yrs), 2)]   # 2016-17, 2018-19, 2020-21, 2022-23, 2024
    out["4. Bloques de anios"] = [[(np.where(~S.year_y.isin(b))[0], np.where(S.year_y.isin(b))[0]) for b in bl]]
    out["5. Ventana expansiva"] = [[(np.where(S.year_y < y)[0], np.where(S.year_y == y)[0]) for y in range(2020, 2025)]]
    return out
ESQ = esquemas(S)
# PRUEBAS de las particiones
for name, reps in ESQ.items():
    for folds in reps:
        te_all = np.concatenate([te for _, te in folds])
        assert all(len(set(tr) & set(te)) == 0 for tr, te in folds), name                  # sin solapamiento
        if name.startswith(("1", "2", "3", "4")): assert sorted(te_all) == list(range(len(S))), name   # cada fila se evalua exactamente 1 vez
        if name.startswith("2"): assert all(not (set(S.region[tr]) & set(S.region[te])) for tr, te in folds)   # territorio no visto
        if name.startswith("3"): assert all(not (set(S.year_y[tr]) & set(S.year_y[te])) for tr, te in folds)
        if name.startswith("5"): assert all(S.year_y[tr].max() < S.year_y[te].min() for tr, te in folds)       # solo pasado
print("OK: particiones verificadas (sin solapamiento, cobertura completa, territorio/anio no visto, ventana solo con pasado)")
leak = np.mean([(S.year_y[tr].max() > S.year_y[te].min()) for tr, te in ESQ["1. K-fold aleatorio"][0]])
print(f"   en K-fold aleatorio, {leak:.0%} de los pliegues entrena con anios POSTERIORES al de prueba\n")

# ------------------------------------------------------------ modelos
def reg_pred(name, tr, te):
    if name == "persistencia": return te.idr_t.values
    if name == "ols4": return LinearRegression().fit(tr[DT], tr.idr_y).predict(te[DT])
    if name == "ridge": return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)).fit(tr[FEATURES], tr.idr_y).predict(te[FEATURES])
    if name == "rf": return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=180, max_depth=5, min_samples_leaf=3, random_state=SEED, n_jobs=1)).fit(tr[FEATURES], tr.idr_y).predict(te[FEATURES])
def clf_pred(name, tr, te):
    if name == "persistencia": return te.category_t.fillna(2).astype(int).values
    m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LogisticRegression(C=1, max_iter=3000, class_weight="balanced"))
    return m.fit(tr[FEATURES], tr.y.astype(int)).predict(te[FEATURES])
REG = ["persistencia", "ols4", "ridge", "rf"]; rows = []
Sy = S.y.astype(int).values
for esq, reps in ESQ.items():
    for ri, folds in enumerate(reps):
        for mod in REG:
            pred = np.full(len(S), np.nan)
            for tr, te in folds: pred[te] = reg_pred(mod, S.iloc[tr], S.iloc[te])
            ev = ~np.isnan(pred)
            for etiqueta, sel in (("todas las evaluadas", ev), ("mismas 117 (2020-2024)", ev & mask117)):
                if sel.sum() == 0: continue
                rows.append(dict(esquema=esq, rep=ri, tarea="regresion", modelo=mod, muestra=etiqueta, n=int(sel.sum()), **{k: v for k, v in todas(S.idr_y[sel], pred[sel]).items() if k in ("MAE", "RMSE", "MAPE", "sMAPE")}))
        for mod in ("persistencia", "logistica"):
            pred = np.full(len(S), -1)
            for tr, te in folds: pred[te] = clf_pred(mod, S.iloc[tr], S.iloc[te])
            ev = pred >= 0
            for etiqueta, sel in (("todas las evaluadas", ev), ("mismas 117 (2020-2024)", ev & mask117)):
                if sel.sum() == 0: continue
                rows.append(dict(esquema=esq, rep=ri, tarea="clasificacion", modelo=mod, muestra=etiqueta, n=int(sel.sum()),
                                 F1_macro=f1_score(Sy[sel], pred[sel], labels=[0,1,2], average="macro", zero_division=0), accuracy=accuracy_score(Sy[sel], pred[sel])))
    print(esq, "ok", flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT/"cv_resultados_detalle.csv", index=False)
agg = R.groupby(["tarea", "muestra", "esquema", "modelo"]).agg(n=("n", "first"), **{c: (c, "mean") for c in ["MAE", "RMSE", "MAPE", "sMAPE", "F1_macro", "accuracy"] if c in R}).reset_index()
agg.to_csv(OUT/"cv_resultados.csv", index=False)
pd.set_option("display.width", 200)
for muestra in ("mismas 117 (2020-2024)", "todas las evaluadas"):
    print(f"\n=== REGRESION | muestra: {muestra} ==="); t = agg[(agg.tarea == "regresion") & (agg.muestra == muestra)]
    print(t.pivot(index="esquema", columns="modelo", values="MAE").round(4).to_string()); 
print("\n=== CLASIFICACION | F1 macro (mismas 117) ==="); t = agg[(agg.tarea == "clasificacion") & (agg.muestra == "mismas 117 (2020-2024)")]
print(t.pivot(index="esquema", columns="modelo", values="F1_macro").round(3).to_string())
sd = R[(R.esquema == "1. K-fold aleatorio") & (R.tarea == "regresion") & (R.modelo == "ridge") & (R.muestra == "todas las evaluadas")].MAE.std()
print(f"\nVariabilidad entre las 10 repeticiones del K-fold aleatorio (Ridge, MAE): sd={sd:.5f}")

# ------------------------------------------------------------ diagnostico: sesgo por anio objetivo (Ridge)
rows = []
for esq in ["3. Un anio objetivo fuera", "4. Bloques de anios", "5. Ventana expansiva"]:
    pred = np.full(len(S), np.nan)
    for tr, te in ESQ[esq][0]: pred[te] = reg_pred("ridge", S.iloc[tr], S.iloc[te])
    for y, g in S.assign(p=pred).query("year_y >= 2020").groupby("year_y"):
        rows.append(dict(esquema=esq, year_y=int(y), sesgo=(g.p - g.idr_y).mean(), MAE=(g.p - g.idr_y).abs().mean(), MAE_persistencia=(g.idr_t - g.idr_y).abs().mean()))
sg = pd.DataFrame(rows); sg.to_csv(OUT/"cv_sesgo_por_anio.csv", index=False)
print("\nSesgo medio de Ridge (pronostico - observado) por anio objetivo:"); print(sg.pivot(index="year_y", columns="esquema", values="sesgo").round(4).to_string())
