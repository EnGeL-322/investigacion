"""Biblioteca comun: carga del panel, construccion del IDR (ACP por dimension) y pares t -> t+1.
Replica la clase Index del Anexo G del articulo."""
import numpy as np, pandas as pd, jenkspy
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"          # datos de entrada (versionados)
OUT = ROOT / "results"               # salidas generadas por los scripts
OUT.mkdir(exist_ok=True)
SEED = 20261029   # misma semilla que el Anexo G del articulo
# (variable, dimension, direccion) en el orden del catalogo del articulo (Tabla 1)
SPECS = [("ingreso_laboral","economica",1),("empleo_formal","economica",1),("titulo_vivienda","economica",1),
 ("nbi","social",-1),("escolaridad","social",1),("analfabetismo","social",-1),("desnutricion_oms","social",-1),
 ("agua_red","social",1),("internet_hogar","social",1),("electricidad","social",1),
 ("gas_cocina","ambiental",1),("alcantarillado","ambiental",1),("recojo_residuos","ambiental",1),
 ("internet_municipal","institucional",1),("plan_concertado","institucional",1),("gestion_ambiental","institucional",1)]
FEATURES = [s[0] for s in SPECS]
DIMS = ["economica","social","ambiental","institucional"]

def load_panel():
    p = pd.read_csv(DATA / "panel_indicadores_2015_2024.csv")
    p.columns = ["region","year"] + FEATURES
    return p

class Index:
    def __init__(self, features, method="pca"): self.features, self.method = features, method
    def fit(self, df):
        assert df.year.max() <= 2017
        a = df.dropna(subset=self.features); self.parts = []; self.diag = []
        for dim in DIMS:
            specs = [s for s in SPECS if s[0] in self.features and s[1] == dim]
            cols = [s[0] for s in specs]; sign = np.array([s[2] for s in specs])
            sc = StandardScaler().fit(a[cols]); z = sc.transform(a[cols]) * sign
            pca = PCA(n_components=1).fit(z)
            score = pca.transform(z).ravel() if self.method == "pca" else z.mean(axis=1)
            ori = 1 if np.corrcoef(score, z.mean(axis=1))[0, 1] >= 0 else -1
            self.parts.append((dim, cols, sign, sc, pca, ori, score.std()))
            self.diag.append(dict(dimension=dim, n_calibration=len(a), pc1_variance=float(pca.explained_variance_ratio_[0])))
        raw = self.raw(a); self.lo, self.hi = raw.min(), raw.max()
        self.breaks = jenkspy.jenks_breaks((raw - self.lo) / (self.hi - self.lo), n_classes=3)
        return self
    def raw(self, df):
        out = []
        for dim, cols, sign, sc, pca, ori, sd in self.parts:
            z = sc.transform(df[cols]) * sign
            v = pca.transform(z).ravel() if self.method == "pca" else z.mean(axis=1)
            out.append(v * ori / sd)
        return np.mean(out, axis=0)
    def dim_scores(self, df):
        """Puntaje estandarizado de cada dimension (PC1 orientado / desviacion tipica); NaN si faltan indicadores."""
        out = pd.DataFrame(np.nan, index=df.index, columns=DIMS)
        for dim, cols, sign, sc, pca, ori, sd in self.parts:
            ok = df[cols].notna().all(axis=1)
            if ok.any(): out.loc[ok, dim] = pca.transform(sc.transform(df.loc[ok, cols]) * sign).ravel() * ori / sd
        return out
    def transform(self, df):
        o = df.copy(); o["idr"] = np.nan; o["category"] = np.nan
        ok = o[self.features].notna().all(axis=1)
        s = (self.raw(o.loc[ok]) - self.lo) / (self.hi - self.lo)
        o.loc[ok, "idr"] = s
        o.loc[ok, "category"] = np.searchsorted(self.breaks[1:3], s, side="left")
        return o

def build_indexed():
    panel = load_panel()
    idx = Index(FEATURES).fit(panel[panel.year <= 2017])
    ix = idx.transform(panel)
    for c in DIMS: ix[c] = idx.dim_scores(panel)[c]
    return panel, idx, ix

def pairs(indexed, features):
    x = indexed[["region","year","category"] + features].rename(columns={"year":"year_x","category":"category_t"})
    y = indexed[["region","year","idr","category"]].rename(columns={"year":"year_y","idr":"idr_y","category":"y"})
    x["year_y"] = x.year_x + 1
    x = x.merge(indexed[["region","year","idr"] + [c for c in DIMS if c in indexed]].rename(columns={"year":"year_x","idr":"idr_t", **{c: c + "_t" for c in DIMS}}), on=["region","year_x"])
    return x.merge(y, on=["region","year_y"], validate="one_to_one")
