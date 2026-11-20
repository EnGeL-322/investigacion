# Predicción del desarrollo regional peruano mediante aprendizaje automático y validación temporal (2015–2024)

Datos y código reproducible del artículo empírico sobre el **Índice de Desarrollo Regional (IDR)** del Perú.
Escuela Profesional de Ingeniería de Sistemas, Universidad Peruana Unión (Juliaca).

**Autores:** [Nombre 1], [Nombre 2], [Nombre 3]

## Entregables

| Entregable | Archivo |
|---|---|
| Documento (artículo) | [`docs/Articulo_IDR_Peru_2015_2024.docx`](docs/Articulo_IDR_Peru_2015_2024.docx) |
| Data | [`data/IDR_Peru_datos_2015_2024.xlsx`](data/IDR_Peru_datos_2015_2024.xlsx) (Excel con fuente, diccionario, panel e IDR) y los CSV de [`data/raw/`](data/raw) |
| Slides | [`docs/Exposicion_IDR_Peru_2015_2024.pptx`](docs/Exposicion_IDR_Peru_2015_2024.pptx) |
| Código | Carpeta [`src/`](src) y [`run_all.py`](run_all.py) |

## Pregunta de investigación

¿Los indicadores departamentales del año *t* predicen la categoría (Bajo, Medio o Alto) o el valor del IDR del año *t+1* mejor que la **persistencia**, es decir, mejor que repetir el último valor conocido?

**Respuesta corta:** no de forma robusta. La persistencia tuvo el menor error continuo (MAE 0,0648) y la mejor accuracy (0,778). La regresión logística solo la superó en F1 macro, y por 0,006.

## Estructura del repositorio

```
investigacion/
├── README.md
├── requirements.txt
├── run_all.py                  # ejecuta todo el análisis en orden
├── docs/
│   ├── Articulo_IDR_Peru_2015_2024.docx
│   └── Exposicion_IDR_Peru_2015_2024.pptx
├── data/
│   ├── IDR_Peru_datos_2015_2024.xlsx   # los mismos datos en Excel, con diccionario
│   ├── DICCIONARIO_DATOS.md    # variables, unidades y fuente
│   └── raw/
│       ├── panel_indicadores_2015_2024.csv   # 250 filas = 25 territorios × 10 años, 16 indicadores
│       └── idr_publicado.csv                 # IDR y categoría publicados en el artículo
├── src/
│   ├── idr_lib.py                    # carga del panel, construcción del IDR y pares t → t+1
│   ├── metricas/                     # una métrica por archivo
│   │   ├── mae.py
│   │   ├── rmse.py
│   │   ├── mape.py
│   │   ├── smape.py
│   │   ├── mase.py
│   │   └── r2.py
│   ├── 00_extraer_panel.py           # (opcional) extrae el panel desde el .docx del artículo
│   ├── 01_reproducir_idr.py          # reconstruye el IDR y lo compara con el publicado
│   ├── 02_replicar_clasificacion.py  # logística, Random Forest, SVM, XGBoost y referencias
│   ├── 03_regresion_multiple.py      # MCO, Ridge y otros modelos para el valor continuo
│   ├── 04_series_tiempo.py           # ingenuo, deriva, suavizamiento, Holt y AR(1)
│   ├── 05_validacion_cruzada.py      # cinco esquemas de validación
│   └── 06_figuras_y_resumen.py       # figuras y resumen en JSON
├── tests/                            # una prueba por método (pytest)
│   ├── test_mae.py
│   ├── test_rmse.py
│   ├── test_mape.py
│   ├── test_smape.py
│   ├── test_mase.py
│   ├── test_r2.py
│   ├── test_datos.py
│   ├── test_indice_idr.py
│   ├── test_pares_temporales.py
│   ├── test_clasificacion.py
│   ├── test_regresion_multiple.py
│   ├── test_series_tiempo.py
│   └── test_validacion_cruzada.py
└── results/                    # tablas, figuras y registro generados por run_all.py
```

## Cómo ejecutarlo

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/<tu-usuario>/<tu-repositorio>.git
cd <tu-repositorio>
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Tarda alrededor de 80 segundos. Cada script lee `data/raw/` y escribe en `results/`; al final se ejecutan todas las pruebas.

También se puede ejecutar un paso suelto desde la carpeta `src/`, por ejemplo `python 04_series_tiempo.py`. El paso 01 debe correr antes que los demás.

## Pruebas

Cada método tiene su propio archivo de pruebas en `tests/`. Para ejecutarlas solas, después de `run_all.py`:

```bash
python -m pytest tests -v                 # todas
python -m pytest tests/test_mae.py -v     # solo una
```

| Archivo | Qué comprueba |
|---|---|
| `test_mae.py`, `test_rmse.py`, `test_mape.py`, `test_smape.py`, `test_mase.py`, `test_r2.py` | Cálculo a mano, coincidencia con scikit-learn, propiedades y casos límite de cada métrica |
| `test_datos.py` | 250 filas, 25 territorios, 10 años, 71 faltantes, sin duplicados, porcentajes entre 0 y 100 |
| `test_indice_idr.py` | 72 casos de calibración, cortes de Jenks, 237 categorías iguales a las publicadas y que alterar años futuros no cambia la calibración |
| `test_pares_temporales.py` | 225 pares, 213 observables, objetivo = año siguiente, 117 objetivos de prueba |
| `test_clasificacion.py` | Accuracy y F1 de la Tabla 2 del artículo |
| `test_regresion_multiple.py` | R² = 0,90, MCO igual a las ecuaciones normales y errores de la persistencia y del MCO |
| `test_series_tiempo.py` | MAE y RMSE de la Tabla 3 y que el ingenuo solo usa el año anterior |
| `test_validacion_cruzada.py` | MAE de Ridge por esquema y que el resultado se invierte al respetar el tiempo |

Resultado actual: **68 pruebas aprobadas y 1 fallo esperado** (`xfail`). El fallo esperado es XGBoost, que no coincide con el artículo (ver abajo).

## Datos

- **Fuente:** Instituto Nacional de Estadística e Informática (INEI, 2025). *Perú: Principales Indicadores Departamentales 2014–2025* (publicación digital Lib2043). https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib2043/
- **Cobertura:** 24 departamentos y la Provincia Constitucional del Callao, de 2015 a 2024. Son 3929 de 4000 valores observados; los 71 faltantes no se imputaron en el panel.
- **Cómo se obtuvo:** el panel se transcribió de las tablas anexas del artículo, con dos decimales. Los archivos Excel originales del INEI no forman parte de este repositorio.
- **Limitaciones:** Lima no tiene IDR por falta de cobertura, y el Callao lo pierde desde 2022 por ausencia de datos laborales. El ingreso es nominal. Los datos laborales cambian de ENAHO a EPEN en 2022.

El detalle de cada variable está en [`data/DICCIONARIO_DATOS.md`](data/DICCIONARIO_DATOS.md).

## Método en resumen

1. **Índice:** los 16 indicadores se estandarizan con 2015–2017 y se orientan según su signo. Se toma un componente principal (ACP) por dimensión y se promedian las 4 dimensiones con peso de 25 % cada una. Luego se escala entre 0 y 1 y se clasifica con cortes de Jenks (0,360 y 0,636). Ningún parámetro usa los años de prueba.
2. **Evaluación temporal:** se forman pares t → t+1 (213 con objetivo observable). Con ventana expansiva se evalúan 117 objetivos de 2020 a 2024, entrenando solo con años anteriores. La semilla es 20261029.
3. **Modelos de clasificación:** regresión logística multinomial, Random Forest, máquina de soporte vectorial y XGBoost.
4. **Modelos para el valor continuo:** persistencia, deriva, suavizamiento exponencial simple, Holt amortiguado, AR(1), regresión múltiple (MCO) y Ridge.
5. **Validación cruzada:** Ridge (α = 10) con cinco esquemas: K-fold aleatorio, bloques de años, territorio fuera, año fuera y ventana expansiva.
6. **Métricas:** accuracy y F1 macro para clasificación; MAE y RMSE como principales para el valor continuo; MAPE y sMAPE como complementarias.

## Resultados principales (117 objetivos, 2020–2024)

**Clasificación**

| Modelo | Accuracy | F1 macro |
|---|---|---|
| Regresión logística | 0,7692 | 0,7728 |
| Persistencia | 0,7778 | 0,7665 |
| Random Forest | 0,7350 | 0,7426 |
| Máquina de soporte vectorial | 0,6752 | 0,6834 |
| Clase mayoritaria | 0,4530 | 0,2078 |
| XGBoost (ver nota) | 0,7692 | 0,7758 |

**Valor continuo**

| Modelo | MAE | RMSE |
|---|---|---|
| Persistencia | 0,0648 | 0,0801 |
| AR(1) por territorio | 0,0656 | 0,0817 |
| Suavizamiento exponencial simple | 0,0705 | 0,0847 |
| Deriva | 0,0717 | 0,0884 |
| Holt amortiguado | 0,0791 | 0,0986 |
| MCO con 16 indicadores | 0,0885 | 0,1048 |

**Validación cruzada (MAE de Ridge; persistencia = 0,0648)**

| Esquema | MAE |
|---|---|
| K-fold aleatorio (promedio de 10 repeticiones) | 0,0616 |
| Bloques de años | 0,0623 |
| Territorio fuera | 0,0624 |
| Año fuera | 0,0755 |
| Ventana expansiva | 0,0822 |

Con esquemas aleatorios Ridge parece superar a la persistencia; cuando se respeta el orden temporal, deja de hacerlo.

La regresión descriptiva con las 4 dimensiones obtiene R² = 0,90 dentro de la muestra. Ese valor refleja que el índice cambia poco de un año a otro; no indica capacidad de pronóstico.

## Reproducibilidad: qué coincide con el artículo

- Las 237 categorías del IDR coinciden con las publicadas (diferencia máxima 0,000574, atribuible al redondeo).
- Coinciden logística, persistencia, Random Forest, SVM y clase mayoritaria (Tabla 2), los seis errores de la Tabla 3 y los esquemas de validación cruzada salvo el K-fold.
- **K-fold aleatorio:** aquí se promedian 10 repeticiones (0,0616); el artículo informa una sola repetición (0,0606).
- **XGBoost no coincide:** el manuscrito original informa F1 0,7590 y la auditoría del artículo 0,7686; este código obtiene 0,7758. La causa no está identificada. Una posibilidad es la versión de XGBoost o de sus hilos de ejecución.
- Las sensibilidades del índice y el contraste de Spearman con el antecedente (0,952) se informan en el artículo, pero no se recalculan aquí.

Versiones con las que se probó: Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9.1, statsmodels 0.15.0, XGBoost 3.4.1, scipy 1.18.1, jenkspy 0.4.1.

## Referencias principales

- Bergmeir, C., y Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences, 191*, 192–213. https://doi.org/10.1016/j.ins.2011.12.028
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Chen, T., y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794. https://doi.org/10.1145/2939672.2939785
- Coaquira Velásquez, M. A., Tudela Mamani, J. W., y Jiménez Carrión, M. (2023). Evaluación comparativa regional: índice sintético de desarrollo regional (IDR) para Perú. *Desarrollo y Sociedad, 94*, 109–157. https://doi.org/10.13043/DYS.94.4
- Hyndman, R. J., y Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3.ª ed.). OTexts. https://otexts.com/fpp3/
- Hyndman, R. J., y Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting, 22*(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881

La lista completa está en el artículo.
"# investigacion" 
