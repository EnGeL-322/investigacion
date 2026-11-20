"""Configuración compartida de las pruebas (pytest).

Agrega src/ al path y ofrece los resultados generados por run_all.py.
Las pruebas de resultados se omiten con un aviso si todavía no se ejecutó el análisis.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"


@pytest.fixture(scope="session")
def resultado():
    """Devuelve una función que lee un CSV de results/ o omite la prueba si no existe."""
    def leer(nombre, **kwargs):
        ruta = RESULTS / nombre
        if not ruta.exists():
            pytest.skip(f"Falta {ruta.name}: ejecuta primero `python run_all.py`")
        return pd.read_csv(ruta, **kwargs)
    return leer


@pytest.fixture(scope="session")
def indice():
    """Panel, índice ajustado y panel con IDR reconstruido."""
    from idr_lib import build_indexed
    return build_indexed()
