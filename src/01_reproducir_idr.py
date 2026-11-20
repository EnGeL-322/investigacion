"""Paso 1. Reconstruye el IDR desde el panel del Anexo y lo contrasta con el IDR publicado en el articulo."""
import numpy as np, pandas as pd
from idr_lib import *
panel, idx, ix = build_indexed()
pub = pd.read_csv(DATA/"idr_publicado.csv")
pub = pub.melt(id_vars="Territorio", var_name="year", value_name="txt")
pub["year"] = pub.year.astype(int)
pub["idr_pub"] = pub.txt.str.extract(r"([\d.]+)").astype(float)[0]
pub["cat_pub"] = pub.txt.str.extract(r"([BMA])\s*$")[0]
m = ix.merge(pub, left_on=["region","year"], right_on=["Territorio","year"], how="left")
v = m.dropna(subset=["idr","idr_pub"])
diff = (v.idr - v.idr_pub).abs()
print(f"Casos con IDR reconstruido: {len(v)} | publicado: {m.idr_pub.notna().sum()}")
print(f"Diferencia absoluta: media={diff.mean():.4f} max={diff.max():.4f} | >0.005: {(diff>0.005).sum()}")
cmap = {0:"B",1:"M",2:"A"}
print("Categoria coincide en", (v.category.map(cmap)==v.cat_pub).sum(), "de", len(v))
print("Cortes Jenks:", np.round(idx.breaks,6))
print(pd.DataFrame(idx.diag).round(3).to_string(index=False))
v[["region","year","idr","idr_pub","category","cat_pub"]].to_csv(OUT/"idr_reproduccion.csv", index=False)
ix.to_csv(OUT/"panel_con_idr.csv", index=False)
