"""Paso 3. Regresion multiple para el IDR del anio siguiente.

Parte A (descriptiva, dentro de muestra): MCO de idr(t+1) sobre los 4 puntajes dimensionales de t, con errores
         robustos agrupados por territorio, VIF y diagnosticos.  NO es causal.
Parte B (predictiva, fuera de muestra): ventana expansiva, objetivos 2020-2024, mismas 117 observaciones que la
         clasificacion. Modelos: persistencia, deriva, AR(1) agrupado, MCO (4 dimensiones), MCO (16 indicadores),
         Ridge, Random Forest y XGBoost.  Metricas: MAE, RMSE, MAPE, sMAPE.
"""
import numpy as np, pandas as pd, statsmodels.api as sm, warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from idr_lib import *
from metricas import *
warnings.filterwarnings("ignore")
DT = [d + "_t" for d in DIMS]

panel, idx, ix = build_indexed()
D = pairs(ix, FEATURES)

# ---------------------------------------------------------------- PARTE A: inferencia descriptiva
A = D.dropna(subset=["idr_y"] + DT).copy()
X = sm.add_constant(A[DT]); mod = sm.OLS(A.idr_y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(A.region)[0]})
print(f"PARTE A | n = {int(mod.nobs)} pares | territorios = {A.region.nunique()}")
print(mod.summary().tables[1])
print(f"R2 = {mod.rsquared:.4f} | R2 ajustado = {mod.rsquared_adj:.4f} | F = {mod.fvalue:.1f}")
# prueba: ¿las cuatro dimensiones pesan igual? (el IDR es el promedio simple => coeficientes iguales)
R = np.array([[0,1,-1,0,0],[0,0,1,-1,0],[0,0,0,1,-1]]); wald = mod.wald_test(R, scalar=True)
print(f"Wald H0: 4 coeficientes iguales -> estadistico={float(wald.statistic):.2f}, p={float(wald.pvalue):.4f}")
vif = pd.DataFrame({"variable": DT, "VIF": [variance_inflation_factor(X.values, i+1) for i in range(len(DT))]})
bp = het_breuschpagan(mod.resid, X)[1]; dw = durbin_watson(mod.resid); sh = stats.shapiro(mod.resid)[1]
print(vif.round(2).to_string(index=False)); print(f"Breusch-Pagan p={bp:.4f} | Durbin-Watson={dw:.2f} | Shapiro p={sh:.4f}")
coef = pd.DataFrame({"coef": mod.params, "ee_robusto": mod.bse, "p": mod.pvalues, "IC95_inf": mod.conf_int()[0], "IC95_sup": mod.conf_int()[1]})
coef.to_csv(OUT/"reg_ols_dimensiones.csv")
# regresion con los 16 indicadores: ilustra multicolinealidad
B = D.dropna(subset=["idr_y"] + FEATURES).copy(); X16 = sm.add_constant(B[FEATURES]); m16 = sm.OLS(B.idr_y, X16).fit()
vif16 = pd.Series([variance_inflation_factor(X16.values, i+1) for i in range(len(FEATURES))], index=FEATURES)
print(f"\n16 indicadores | n={int(m16.nobs)} R2={m16.rsquared:.4f} R2aj={m16.rsquared_adj:.4f} | VIF max={vif16.max():.1f} ({vif16.idxmax()}) | #VIF>10: {(vif16>10).sum()} | n.condicion={m16.condition_number:.0f}")
parte_a = dict(n=int(mod.nobs), r2=mod.rsquared, r2_adj=mod.rsquared_adj, F=float(mod.fvalue), wald_stat=float(wald.statistic), wald_p=float(wald.pvalue),
               bp_p=bp, dw=dw, shapiro_p=sh, vif_max_dim=float(vif.VIF.max()), n16=int(m16.nobs), r2_16=m16.rsquared, r2adj_16=m16.rsquared_adj,
               vif16_max=float(vif16.max()), vif16_var=vif16.idxmax(), vif16_gt10=int((vif16 > 10).sum()), cond16=float(m16.condition_number))
# prueba de verificacion: coeficientes MCO == solucion de ecuaciones normales
Xm = np.c_[np.ones(len(A)), A[DT].values]; beta = np.linalg.solve(Xm.T @ Xm, Xm.T @ A.idr_y.values)
assert np.allclose(beta, mod.params.values, atol=1e-8); print("OK: coeficientes = ecuaciones normales")

# ---------------------------------------------------------------- PARTE B: prediccion fuera de muestra
def make(name, par):
    if name == "ols16": return make_pipeline(StandardScaler(), LinearRegression())
    if name == "ridge": return make_pipeline(StandardScaler(), Ridge(alpha=par))
    if name == "rf": return RandomForestRegressor(n_estimators=180, max_depth=par, min_samples_leaf=3, random_state=SEED, n_jobs=1)
    if name == "xgb": return XGBRegressor(n_estimators=100, max_depth=par, learning_rate=.05, subsample=.85, colsample_bytree=.85, random_state=SEED, n_jobs=1)
FEAT = {"ols16": FEATURES, "ridge": FEATURES, "rf": FEATURES, "xgb": FEATURES}
GRID = {"ridge": [.1, 1, 10, 100], "rf": [3, 5, None], "xgb": [1, 2, 3]}

def fit_predict(name, tr, te, par=None):
    if name == "persistencia": return te.idr_t.values
    if name == "deriva": return te.idr_t.values + (tr.idr_y - tr.idr_t).mean()
    if name == "ar1":
        m = LinearRegression().fit(tr[["idr_t"]], tr.idr_y); return m.predict(te[["idr_t"]])
    if name == "ols4":
        m = LinearRegression().fit(tr[DT], tr.idr_y); return m.predict(te[DT])
    m = make(name, par).fit(tr[FEAT[name]].fillna(tr[FEAT[name]].median()), tr.idr_y)
    return m.predict(te[FEAT[name]].fillna(tr[FEAT[name]].median()))

rows, chosen = [], []
for year in range(2020, 2025):
    tr = D[(D.year_y < year) & D.idr_y.notna() & D.idr_t.notna()]; te = D[(D.year_y == year) & D.idr_y.notna()]
    assert tr.year_y.max() < year and (te.year_y == year).all()      # prueba: nunca se entrena con el futuro
    for name in ["persistencia","deriva","ar1","ols4","ols16","ridge","rf","xgb"]:
        par = None
        if name in GRID:                                              # ajuste de hiperparametro con ventana expansiva interna
            sc = []
            for p_ in GRID[name]:
                v = []
                for inner in range(max(2018, year-3), year):
                    a, b = tr[tr.year_y < inner], tr[tr.year_y == inner]
                    if len(b) == 0: continue
                    v.append(mae(b.idr_y, fit_predict(name, a, b, p_)))
                sc.append(np.mean(v))
            par = GRID[name][int(np.argmin(sc))]; chosen.append(dict(year=year, model=name, param=str(par)))
        f = fit_predict(name, tr, te, par)
        rows += [dict(model=name, region=r, year_y=year, y=y_, pred=f_) for r, y_, f_ in zip(te.region, te.idr_y, f)]
    print(year, "ok", flush=True)
P = pd.DataFrame(rows); P.to_csv(OUT/"reg_predicciones.csv", index=False); pd.DataFrame(chosen).to_csv(OUT/"reg_hiperparametros.csv", index=False)
assert P.pred.notna().all() and P.groupby("model").size().nunique() == 1
res = pd.DataFrame([dict(model=m, **todas(g.y, g.pred)) for m, g in P.groupby("model")]).set_index("model")
# MASE: escala = error medio de la persistencia (sobre los mismos objetivos de prueba) -> MAE relativo a la persistencia
res["MAE_rel_persistencia"] = res.MAE / res.loc["persistencia", "MAE"]
res = res.sort_values("MAE"); print("\nPARTE B | 117 predicciones, objetivos 2020-2024"); print(res.round(4).to_string()); res.to_csv(OUT/"reg_metricas.csv")

# Bootstrap por territorio: diferencia de MAE y RMSE contra la persistencia (1000 remuestreos)
rng = np.random.default_rng(SEED); regs = np.array(sorted(P.region.unique())); by = {(m, r): g for (m, r), g in P.groupby(["model","region"])}
def stat(m, sample, fn):
    g = pd.concat([by[(m, r)] for r in sample]); return fn(g.y, g.pred)
boot = []
for m in res.index.drop("persistencia"):
    dm, dr = [], []
    for _ in range(1000):
        s = rng.choice(regs, len(regs), replace=True)
        dm.append(stat(m, s, mae) - stat("persistencia", s, mae)); dr.append(stat(m, s, rmse) - stat("persistencia", s, rmse))
    boot.append(dict(model=m, dMAE=res.loc[m, "MAE"] - res.loc["persistencia", "MAE"], MAE_lo=np.quantile(dm, .025), MAE_hi=np.quantile(dm, .975),
                     dRMSE=res.loc[m, "RMSE"] - res.loc["persistencia", "RMSE"], RMSE_lo=np.quantile(dr, .025), RMSE_hi=np.quantile(dr, .975)))
boot = pd.DataFrame(boot).set_index("model"); print("\nDiferencia vs persistencia (negativo = mejor):"); print(boot.round(4).to_string())
boot.to_csv(OUT/"reg_bootstrap.csv")
pd.Series(parte_a).to_csv(OUT/"reg_parte_a.csv")
