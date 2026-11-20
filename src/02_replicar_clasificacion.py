"""Paso 2. Replica la evaluacion de clasificacion del articulo (4 modelos + 2 lineas base, ventana expansiva 2020-2024)."""
import numpy as np, pandas as pd, warnings
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from idr_lib import *
warnings.filterwarnings("ignore")

def model(name, param):
    if name == "logistica": m = LogisticRegression(C=param, max_iter=3000, class_weight="balanced")
    elif name == "random_forest": m = RandomForestClassifier(n_estimators=180, max_depth=param, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=1)
    elif name == "svm": m = SVC(C=param, kernel="rbf", gamma="scale", class_weight="balanced")
    else: m = XGBClassifier(n_estimators=100, max_depth=param, learning_rate=.05, subsample=.85, colsample_bytree=.85, objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=SEED, n_jobs=1)
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), m)

if __name__ == "__main__":
    panel, idx, ix = build_indexed()
    d = pairs(ix, FEATURES)
    print("pares:", len(d), "| con objetivo observable:", d.y.notna().sum())
    grids = {"logistica":[.1,1,10], "random_forest":[3,5,None], "svm":[.1,1,10], "xgboost":[1,2,3]}
    rows = []
    for year in range(2020, 2025):
        tr = d[(d.year_y < year) & d.y.notna()]; te = d[(d.year_y == year) & d.y.notna()]
        for name, grid in grids.items():
            sc = []
            for par in grid:
                v = []
                for inner in range(max(2018, year-3), year):
                    a, b = tr[tr.year_y < inner], tr[tr.year_y == inner]
                    if len(b) == 0 or a.y.nunique() != 3: continue
                    m = model(name, par).fit(a[FEATURES], a.y.astype(int))
                    v.append(f1_score(b.y.astype(int), m.predict(b[FEATURES]), labels=[0,1,2], average="macro", zero_division=0))
                sc.append(np.mean(v) if v else -np.inf)
            best = grid[int(np.argmax(sc))]
            p = model(name, best).fit(tr[FEATURES], tr.y.astype(int)).predict(te[FEATURES])
            rows += [dict(year_y=year, model=name, y=int(a_), pred=int(b_)) for a_, b_ in zip(te.y, p)]
        for name in ("persistencia", "mayoritaria"):
            p = te.category_t.fillna(tr.y.mode()[0]).astype(int).to_numpy() if name == "persistencia" else np.repeat(int(tr.y.mode()[0]), len(te))
            rows += [dict(year_y=year, model=name, y=int(a_), pred=int(b_)) for a_, b_ in zip(te.y, p)]
        print(year, "ok", flush=True)
    pr = pd.DataFrame(rows); pr.to_csv(OUT/"clasif_predicciones.csv", index=False)
    res = pr.groupby("model").apply(lambda g: pd.Series(dict(n=len(g), accuracy=accuracy_score(g.y, g.pred),
          F1_macro=f1_score(g.y, g.pred, labels=[0,1,2], average="macro", zero_division=0)))).sort_values("F1_macro", ascending=False)
    print(res.round(3)); res.to_csv(OUT/"clasif_replica.csv")
