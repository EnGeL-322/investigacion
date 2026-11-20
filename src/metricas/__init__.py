"""Métricas de error de pronóstico implementadas con numpy.

Cada métrica está en su propio archivo y tiene su prueba en tests/test_<métrica>.py.
"""
from .mae import mae
from .mape import mape
from .mase import mase
from .r2 import r2
from .rmse import rmse
from .smape import smape
from ._comun import preparar

__all__ = ["mae", "rmse", "mape", "smape", "mase", "r2", "todas"]


def todas(y, f):
    """Diccionario con MAE, RMSE, MAPE, sMAPE y el número de pares usados."""
    m, n_mape = mape(y, f, return_n=True)
    return dict(MAE=mae(y, f), RMSE=rmse(y, f), MAPE=m, sMAPE=smape(y, f),
                n=len(preparar(y, f)[0]), n_mape=n_mape)
