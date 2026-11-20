"""Paso 4. Series de tiempo del IDR.

Parte A: pronostico a un paso (origen movil) para cada territorio, objetivos 2020-2024, usando solo su propia historia:
         ingenuo, deriva, suavizamiento exponencial simple, Holt amortiguado y AR(1) con constante.
Parte B: dinamica agregada: tendencia (Mann-Kendall y pendiente de Sen), prueba ADF, convergencia sigma y beta.
Aviso: son series ANUALES de 10 puntos; las pruebas de raiz unitaria y los modelos ARIMA tienen muy poca potencia.
"""
import numpy as np, pandas as pd, warnings, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from idr_lib import *
from metricas import *
warnings.filterwarnings("ignore")
panel, idx, ix = build_indexed()
W = ix.pivot(index="region", columns="year", values="idr")
print("Territorios con IDR:", W.dropna(how="all").shape[0], "| serie completa 10 anios:", int(W.notna().all(axis=1).sum()),
      "| incompletos:", W.dropna(how="all").index[W.dropna(how="all").isna().any(axis=1)].tolist())

# ---------------------------------------------------------------- PARTE A
def forecast(name, h):
    """h = historia (array) hasta t; devuelve pronostico de t+1. Retorna (valor, fallo)."""
    last = h[-1]
    try:
        if name == "ingenuo": return last, False
        if name == "deriva": return last + (h[-1] - h[0]) / (len(h) - 1), False
        if name == "ses":    return float(SimpleExpSmoothing(h, initialization_method="estimated").fit().forecast(1)[0]), False
        if name == "holt_amortiguado":
            return float(ExponentialSmoothing(h, trend="add", damped_trend=True, initialization_method="estimated").fit().forecast(1)[0]), False
        if name == "ar1":
            r = ARIMA(h, order=(1, 0, 0), trend="c").fit(); v = float(r.forecast(1)[0])
            if not np.isfinite(v) or abs(v - last) > 0.5: return last, True      # estimacion inestable -> se reemplaza por el ingenuo y se cuenta
            return v, False
    except Exception: return last, True
MODELS = ["ingenuo", "deriva", "ses", "holt_amortiguado", "ar1"]
rows, fails = [], {m: 0 for m in MODELS}
for region, s in W.iterrows():
    s = s.dropna()
    for year in range(2020, 2025):
        if year not in s.index: continue
        h = s[s.index < year]
        if len(h) < 5: continue
        assert h.index.max() < year                                    # prueba: solo historia pasada
        for m in MODELS:
            f, fail = forecast(m, h.values); fails[m] += fail
            rows.append(dict(model=m, region=region, year_y=year, y=s[year], pred=f))
P = pd.DataFrame(rows); P.to_csv(OUT/"ts_predicciones.csv", index=False)
assert P.groupby("model").size().nunique() == 1 and P.pred.notna().all()
res = pd.DataFrame([dict(model=m, **todas(g.y, g.pred)) for m, g in P.groupby("model")]).set_index("model")
res["MAE_rel_ingenuo"] = res.MAE / res.loc["ingenuo", "MAE"]; res["fallos_ajuste"] = pd.Series(fails)
res = res.sort_values("MAE"); print("\nPARTE A | pronostico por territorio con su propia historia"); print(res.round(4).to_string()); res.to_csv(OUT/"ts_metricas.csv")
# bootstrap por territorio contra el ingenuo
rng = np.random.default_rng(SEED); regs = np.array(sorted(P.region.unique())); by = {(m, r): g for (m, r), g in P.groupby(["model", "region"])}
st = lambda m, s, fn: fn(*[pd.concat([by[(m, r)] for r in s])[c] for c in ("y", "pred")])
boot = []
for m in [x for x in res.index if x != "ingenuo"]:
    d = [st(m, s, mae) - st("ingenuo", s, mae) for s in (rng.choice(regs, len(regs), replace=True) for _ in range(1000))]
    boot.append(dict(model=m, dMAE=res.loc[m, "MAE"] - res.loc["ingenuo", "MAE"], lo=np.quantile(d, .025), hi=np.quantile(d, .975)))
boot = pd.DataFrame(boot).set_index("model"); print("\nDiferencia de MAE vs ingenuo (negativo = mejor):"); print(boot.round(4).to_string()); boot.to_csv(OUT/"ts_bootstrap.csv")

# ---------------------------------------------------------------- PARTE B
full = W[W.notna().all(axis=1)]                       # 23 territorios con serie completa (balanceado)
mean = full.mean(); cv = full.std(ddof=1) / full.mean(); yrs = np.array(mean.index, float)
tau, p_mk = stats.kendalltau(yrs, mean.values)
sen = stats.theilslopes(mean.values, yrs)             # (pendiente, intercepto, lim inf, lim sup)
adf = adfuller(mean.values, maxlag=0, regression="c", autolag=None)
cv_tau, cv_p = stats.kendalltau(yrs, cv.values); cv_ols = stats.linregress(yrs, cv.values)
change = (full[2024] - full[2015]) / 9; init = full[2015]; beta = stats.linregress(init.values, change.values)
print(f"\nPARTE B | {len(full)} territorios con serie completa")
print(f"IDR medio 2015={mean[2015]:.3f} -> 2024={mean[2024]:.3f} | Mann-Kendall tau={tau:.2f}, p={p_mk:.4f} | Sen={sen[0]:.4f}/anio [{sen[2]:.4f}; {sen[3]:.4f}]")
print(f"ADF (c, 0 rezagos, n=10): estadistico={adf[0]:.2f}, p={adf[1]:.3f}, critico 5%={adf[4]['5%']:.2f}  <- potencia muy baja")
print(f"Sigma: CV 2015={cv[2015]:.3f} -> 2024={cv[2024]:.3f} | tau={cv_tau:.2f}, p={cv_p:.4f} | pendiente MCO={cv_ols.slope:.4f}/anio")
print(f"Beta: cambio anual = a + b*IDR2015 -> b={beta.slope:.4f}, p={beta.pvalue:.4f}, R2={beta.rvalue**2:.3f}")
pd.Series(dict(n_territorios=len(full), media_2015=mean[2015], media_2024=mean[2024], mk_tau=tau, mk_p=p_mk, sen=sen[0], sen_lo=sen[2], sen_hi=sen[3],
               adf_stat=adf[0], adf_p=adf[1], adf_crit5=adf[4]["5%"], cv_2015=cv[2015], cv_2024=cv[2024], cv_tau=cv_tau, cv_p=cv_p, cv_slope=cv_ols.slope,
               beta=beta.slope, beta_p=beta.pvalue, beta_r2=beta.rvalue**2)).to_csv(OUT/"ts_agregado.csv")
pd.DataFrame({"media": mean, "cv": cv}).to_csv(OUT/"ts_media_cv.csv")
# prueba: la pendiente de Sen coincide con la mediana de pendientes por pares
sl = [(mean.values[j]-mean.values[i])/(yrs[j]-yrs[i]) for i in range(10) for j in range(i+1, 10)]; assert np.isclose(np.median(sl), sen[0])
print("OK: pendiente de Sen = mediana de pendientes por pares")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for r, s in full.iterrows(): ax[0].plot(s.index, s.values, color="#9bb7c4", lw=.8)
ax[0].plot(mean.index, mean.values, color="#0b3c5d", lw=2.5, label="Promedio (23 territorios)"); ax[0].set(title="Trayectorias del IDR 2015-2024", xlabel="Año", ylabel="IDR"); ax[0].legend(frameon=False)
ax[1].plot(cv.index, cv.values, marker="o", color="#b5483a"); ax[1].set(title="Convergencia sigma: coeficiente de variación", xlabel="Año", ylabel="CV entre territorios")
fig.tight_layout(); fig.savefig(OUT/"fig_series_tiempo.png", dpi=170)
