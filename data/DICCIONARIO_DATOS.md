# Diccionario de datos

Fuente de todas las variables: INEI (2025), *Perú: Principales Indicadores Departamentales 2014–2025* (Lib2043).
Los valores se transcribieron de las tablas anexas del artículo con dos decimales. Revisa la definición exacta de cada indicador en la publicación del INEI antes de reutilizarlos.

## `raw/panel_indicadores_2015_2024.csv`

Una fila por territorio y año: 25 territorios × 10 años = 250 filas. Las celdas vacías son datos faltantes (71 en total).

| Columna | Descripción | Unidad | Dimensión | Orientación |
|---|---|---|---|---|
| `region` | Departamento o Provincia Constitucional del Callao | texto | — | — |
| `year` | Año | 2015–2024 | — | — |
| `ingreso_laboral` | Ingreso laboral promedio mensual (nominal) | soles | Económica | + |
| `empleo_formal` | Población ocupada con empleo formal | % | Económica | + |
| `titulo_vivienda` | Hogares con título de propiedad de la vivienda | % | Económica | + |
| `nbi` | Población con al menos una necesidad básica insatisfecha | % | Social | − |
| `escolaridad` | Años promedio de estudio | años | Social | + |
| `analfabetismo` | Tasa de analfabetismo | % | Social | − |
| `desnutricion_oms` | Desnutrición crónica en menores de 5 años (patrón OMS) | % | Social | − |
| `agua_red` | Hogares con agua por red pública | % | Social | + |
| `internet_hogar` | Hogares con acceso a internet | % | Social | + |
| `electricidad` | Hogares con alumbrado eléctrico por red pública | % | Social | + |
| `gas_cocina` | Hogares que cocinan con gas | % | Ambiental | + |
| `alcantarillado` | Hogares con desagüe por red pública | % | Ambiental | + |
| `recojo_residuos` | Municipalidades con servicio de recojo de residuos sólidos | % | Ambiental | + |
| `internet_municipal` | Municipalidades con acceso a internet | % | Institucional | + |
| `plan_concertado` | Municipalidades con plan de desarrollo concertado | % | Institucional | + |
| `gestion_ambiental` | Municipalidades con instrumentos de gestión ambiental | % | Institucional | + |

**Orientación:** con "+", un aumento eleva el índice. Con "−", el signo se invierte antes de construir el índice.

## `raw/idr_publicado.csv`

El IDR y su categoría tal como aparecen en el artículo, para verificar la reconstrucción.

- `Territorio`: nombre del territorio.
- `2015` … `2024`: celdas con formato `valor categoría`, por ejemplo `0.155 B`. La categoría es B (Bajo), M (Medio) o A (Alto). Una celda vacía o con guion significa que el índice no es observable.

## Notas

- Lima no tiene IDR principal porque su cobertura no lo permite. No se reemplazó por Lima Provincia ni Región Lima.
- El Callao no tiene IDR desde 2022 por falta de componentes laborales.
- Los datos laborales provienen de la ENAHO hasta 2021 y de la EPEN desde 2022. No se aplicó ningún empalme.
- El ingreso es nominal e incluye el efecto de la inflación.
