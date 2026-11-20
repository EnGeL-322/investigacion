"""Paso 0. Extrae del manuscrito (.docx) el panel de 16 indicadores y el IDR publicado.
Uso (opcional): python src/00_extraer_panel.py ruta/al/articulo.docx
No hace falta para reproducir los resultados: data/raw ya contiene el panel extraido.
Las tablas se localizan por su encabezado, no por su posicion."""
import sys, pandas as pd
from pathlib import Path
from docx import Document
DATA = Path(__file__).resolve().parent.parent / "data" / "raw"; DATA.mkdir(parents=True, exist_ok=True)
COLS = ["region", "year", "ingreso_laboral", "empleo_formal", "titulo_vivienda", "nbi", "escolaridad", "analfabetismo",
        "desnutricion_oms", "agua_red", "internet_hogar", "electricidad", "gas_cocina", "alcantarillado", "recojo_residuos",
        "internet_municipal", "plan_concertado", "gestion_ambiental"]
def num(s):
    s = s.strip().replace("−", "-").replace(" ", "")
    if s in ("", "NA", "—", "-", "nan"): return float("nan")
    s = s.replace(",", ".") if s.count(",") == 1 and "." not in s else s.replace(",", "")
    try: return float(s)
    except ValueError: return float("nan")
def tabla(t):
    rows = [[c.text.strip() for c in r.cells] for r in t.rows]
    return pd.DataFrame(rows[1:], columns=rows[0])
d = Document(sys.argv[1])
def buscar(*cols):
    for t in d.tables:
        h = [c.text.strip() for c in t.rows[0].cells]
        if all(c in h for c in cols) and len(t.rows) > 200: return tabla(t)   # tablas largas (anexos)
    raise SystemExit(f"No se encontro la tabla con columnas {cols}")
a = buscar("Territorio", "Año", "Ingreso", "NBI"); b = buscar("Territorio", "Año", "Electricidad", "Gestión amb.")
p = a.merge(b, on=["Territorio", "Año"])
for c in p.columns[2:]: p[c] = p[c].map(num)
p["Año"] = p["Año"].astype(int)
idr = None
for t in d.tables:
    h = [c.text.strip() for c in t.rows[0].cells]
    if h[:3] == ["Territorio", "2015", "2016"] and len(h) == 11 and len(t.rows) == 26: idr = tabla(t); break
assert idr is not None, "No se encontro la tabla del IDR publicado (Anexo F)"
print("Panel:", p.shape, "| faltantes:", int(p.isna().sum().sum()), "| IDR publicado:", idr.shape)
p.columns = COLS
p.to_csv(DATA / "panel_indicadores_2015_2024.csv", index=False); idr.to_csv(DATA / "idr_publicado.csv", index=False)
