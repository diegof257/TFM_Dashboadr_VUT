# Manual Técnico — Observatorio de Mercado Inmobiliario y Presión Turística
**Trabajo Fin de Máster · Turismo Electrónico: Tecnologías aplicadas a la gestión**  
**Versión de referencia:** `app.py` · Python 3.14 · Streamlit  
**Ciudades:** Málaga, Sevilla, Jaén y Teruel · **Periodo:** 2016–2023

---

## Índice

1. [Visión general y propósito](#1-visión-general-y-propósito)
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)
3. [Estructura de archivos](#3-estructura-de-archivos)
4. [Pipeline ETL — Carga y preparación de datos](#4-pipeline-etl--carga-y-preparación-de-datos)
5. [Indicadores calculados (KPIs)](#5-indicadores-calculados-kpis)
6. [Motor analítico: Índice de Moran Bivariado](#6-motor-analítico-índice-de-moran-bivariado)
7. [Módulos del visor (tabs)](#7-módulos-del-visor-tabs)
   - 7.1 Mapa interactivo
   - 7.2 Evolución comparada
   - 7.3 Tabla de datos
   - 7.4 Modelo de regresión OLS
   - 7.5 Validación con Informe IATUR
8. [Panel de control (barra lateral)](#8-panel-de-control-barra-lateral)
9. [Sistema de caché y rendimiento](#9-sistema-de-caché-y-rendimiento)
10. [Escalabilidad — cómo añadir una nueva ciudad](#10-escalabilidad--cómo-añadir-una-nueva-ciudad)
11. [Dependencias y entorno](#11-dependencias-y-entorno)
12. [Limitaciones conocidas y decisiones de diseño](#12-limitaciones-conocidas-y-decisiones-de-diseño)

---

## 1. Visión general y propósito

El Observatorio es una aplicación web de análisis de datos geoespaciales construida con **Streamlit**. Su objetivo es cuantificar y visualizar la relación entre la concentración de Viviendas de Uso Turístico (VUT) y el precio del alquiler residencial en cuatro ciudades españolas con perfiles turísticos muy distintos.

### ¿Por qué Streamlit?

Streamlit convierte código Python puro en una aplicación web interactiva sin necesidad de JavaScript ni de un servidor dedicado. Cada interacción del usuario (cambio de ciudad, de año, activación de un análisis) relanza el script de arriba a abajo, aplicando el sistema de caché para no repetir operaciones costosas. Esta arquitectura es adecuada para un prototipo académico orientado a la gestión pública porque:

- El código es directamente auditable y reproducible.
- No requiere infraestructura de servidor.
- Los análisis estadísticos (Moran, OLS) se ejecutan en Python, la misma plataforma donde se desarrollaron.

### Enfoque data-driven

El visor no genera conclusiones predefinidas: todos los valores mostrados (precios medios, saturación turística, clústeres espaciales) se calculan en tiempo real a partir de los datos originales. El único componente estático son los coeficientes del modelo OLS, que se importan literalmente de la Tabla 4 del Informe IATUR (2024) para garantizar trazabilidad académica.

---

## 2. Arquitectura del sistema

```
Usuario (navegador web)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                  app.py (Streamlit)                  │
│                                                      │
│  ┌──────────────┐   ┌──────────────────────────────┐ │
│  │  ETL / Cache │   │  Motor analítico             │ │
│  │  ─────────── │   │  ─────────────────────────── │ │
│  │  load_and_   │   │  calcular_moran_bivariado()  │ │
│  │  merge_data()│   │  (PySAL / esda)              │ │
│  │              │   │                              │ │
│  │  load_census_│   │  Simulador OLS               │ │
│  │  indicators()│   │  (coeficientes IATUR 2024)   │ │
│  └──────────────┘   └──────────────────────────────┘ │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Interfaz de usuario (5 tabs + sidebar)        │  │
│  │  Plotly Express / Plotly Graph Objects         │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│           Datos en disco        │
│  censo_ciudades/*.gpkg          │
│  modeloRegresionLineal.csv      │
│  INE_C2021_Indicadores.xlsx     │
│  correlacion_espacial/*.gal     │
└─────────────────────────────────┘
```

El flujo de ejecución es lineal: primero se cargan y procesan los datos (una sola vez gracias al caché), después se renderizan los controles de la barra lateral, y finalmente se renderiza el contenido de cada tab filtrado por las selecciones del usuario.

---

## 3. Estructura de archivos

```
TFM_Dashboadr_VUT/
│
├── app.py                          ← Aplicación principal (este manual)
├── etl.py                          ← Script ETL auxiliar (preprocesamiento previo)
│
├── censo_ciudades/                 ← Mapas vectoriales por ciudad (GeoPackage)
│   ├── MALAGA_FINAL_CENSO2021.gpkg
│   ├── SEVILLA_FINAL_CENSO.gpkg
│   ├── JAEN_FINAL_CENSO.gpkg
│   └── Teruel_FINAL_CENSO.gpkg
│
├── correlacion_espacial/           ← Archivos de análisis espacial
│   ├── MLG_23_16.gal               ← Matriz de pesos oficial INE para Málaga
│   └── MLG_23_16.gpkg
│
├── datos/                          ← (Carpeta destino; actualmente en raíz)
│   ├── modeloRegresionLineal.csv   ← Serie histórica VUT + alquiler 2016-2023
│   └── INE_C2021_Indicadores.xlsx  ← Indicadores Censo 2021 (36.333 secciones)
│
└── .venv/                          ← Entorno virtual Python 3.14
```

### Función `find_file()`

La aplicación usa una función auxiliar que busca cada archivo primero en la subcarpeta organizada y, si no lo encuentra, lo busca en el directorio raíz. Esto permite reorganizar archivos sin romper la aplicación:

```python
def find_file(*rutas):
    for r in rutas:
        if os.path.exists(r):
            return r
    return rutas[-1]
```

---

## 4. Pipeline ETL — Carga y preparación de datos

Todo el proceso ETL está encapsulado en dos funciones decoradas con `@st.cache_data`, lo que significa que se ejecutan **una sola vez** por sesión y el resultado se almacena en memoria.

### 4.1 Paso A — Carga del CSV principal

**Archivo:** `modeloRegresionLineal.csv` (32.383 observaciones)  
**Contenido:** una fila por sección censal por año. Variables principales:

| Columna | Tipo | Descripción |
|---|---|---|
| `SSCC` | str | Código de sección censal (estandarizado a 10 dígitos) |
| `Fecha_ano` | int | Año de la observación (2016–2023) |
| `Em2` | float | Precio del alquiler en €/m² |
| `%VUT` | float | Porcentaje de VUT sobre total de viviendas |
| `VUT.Formula` | float | Número total de plazas VUT en la sección |
| `Viviendas.Formula` | float | Total de viviendas en la sección |

**Operaciones de limpieza aplicadas:**

1. El código `SSCC` se normaliza a 10 dígitos con `zfill(10)` y se elimina el sufijo `.0` que introduce pandas al leer columnas numéricas como string.
2. Los campos numéricos se convierten con `str.replace(',', '.')` para manejar el formato europeo de decimales (coma decimal en el CSV original).
3. `Fecha_ano` se fuerza a numérico con `errors='coerce'` para descartar filas corruptas.

### 4.2 Paso B — Carga de mapas vectoriales (.gpkg)

Se cargan cuatro archivos **GeoPackage**, uno por ciudad, usando `geopandas.read_file()`. Cada archivo contiene los polígonos del seccionado censal con la columna `CUSEC` como identificador geográfico oficial.

Los archivos se buscan en `censo_ciudades/` (prioritario) y luego en el directorio raíz. Si un archivo no existe, se emite un aviso pero la aplicación continúa con las ciudades disponibles.

Los cuatro GeoDataFrames se concatenan en uno solo con `pd.concat()`.

### 4.3 Paso C — Inner Join espacial

Se realiza un `merge` entre el mapa vectorial y el CSV usando `CUSEC` (mapa) = `SSCC` (CSV). El tipo de join es **inner**: solo se mantienen las secciones censales que tienen tanto geometría como datos numéricos.

```python
gdf = gdf_mapa.merge(df, left_on='CUSEC', right_on='SSCC', how='inner')
```

**Por qué inner y no left:** Las secciones sin datos históricos (años sin registro) no aportan información al análisis y generarían polígonos vacíos en el mapa. El diagnóstico de cuántas secciones quedan sin cruzar se muestra en la barra lateral como nota informativa.

Tras el merge, el sistema de referencia de coordenadas se reprojecta a **EPSG:4326** (WGS84), que es el estándar requerido por Plotly para mapas web.

### 4.4 Paso D — Enriquecimiento con Censo 2021

**Archivo:** `INE_C2021_Indicadores.xlsx` (36.333 secciones censales nacionales)

Se realiza un segundo `merge` de tipo **left** para añadir indicadores sociodemográficos. Los campos relevantes del Censo 2021 que se incorporan son:

| Campo original INE | Campo en la app | Descripción |
|---|---|---|
| `t1_1` | `pob_total` | Población total |
| `t18_1` | `viv_total_censo` | Total viviendas familiares |
| `t19_1` | `viv_principales` | Viviendas principales |
| `t19_2` | `viv_no_principales` | Viviendas no principales (turísticas, vacías, segunda residencia) |
| `t20_2` | `viv_alquiler` | Viviendas en régimen de alquiler |
| `t5_1` | `pct_extranjeros` | Porcentaje de población extranjera |
| `t10_1` | `pct_parados` | Porcentaje de parados |
| `t9_1` | `pct_estudios_sup` | Porcentaje con estudios superiores |

El join es left porque el Censo 2021 es estático (un único corte temporal) y cubre todas las secciones de España. Si un `.xlsx` no está disponible, la aplicación funciona igualmente sin los KPIs derivados del censo.

---

## 5. Indicadores calculados (KPIs)

Los tres KPIs compuestos se calculan en el ETL y están disponibles en todos los tabs:

### KPI 1 — Presión Turística

```
Presion_1000hab = (VUT.Formula / pob_total) × 1000
```

Mide cuántas plazas VUT existen por cada 1.000 habitantes residentes. Es el indicador más utilizado en literatura de tourismification para comparar ciudades de distinto tamaño. Se usa `replace(0, NaN)` para evitar divisiones por cero en secciones sin población registrada.

### KPI 2 — Ratio de Turistificación

```
Ratio_turistif = (VUT.Formula / viv_no_principales) × 100
```

Expresa qué porcentaje de las viviendas no principales (las "convertibles" en uso turístico) ya están siendo explotadas como VUT. Se considera el indicador más preciso de saturación real porque el denominador representa el universo de viviendas susceptibles, no el total.

### KPI 3 — Vulnerabilidad del Mercado de Alquiler

```
Pct_viv_alquiler = (viv_alquiler / viv_total_censo) × 100
```

Porcentaje de viviendas en alquiler sobre el total. Las secciones con alta proporción de alquiler son las más vulnerables a la competencia directa de las VUT, porque sus residentes no tienen la alternativa de la propiedad.

---

## 6. Motor analítico: Índice de Moran Bivariado

### ¿Qué mide y por qué se usa?

El **I de Moran Bivariado** mide si hay correlación espacial entre dos variables en unidades geográficas vecinas. En este caso, la variable X es la saturación VUT (`%VUT`) y la variable Y es el precio del alquiler (`Em2`). Un valor positivo y significativo indica que las secciones con alta concentración de VUT tienden a estar rodeadas de secciones con precios de alquiler elevados — el llamado **efecto de contagio espacial** o "mancha de aceite".

A diferencia de la correlación de Pearson, el índice de Moran tiene en cuenta la topología del territorio: no basta con que dos secciones estén correlacionadas, sino que deben ser geográficamente contiguas.

### Implementación técnica

La función `calcular_moran_bivariado()` recibe el GeoDataFrame ya filtrado por ciudad y año, y devuelve:
- El GeoDataFrame con la columna `Cluster_Moran` añadida
- El I Global (escalar)
- El Z-score estandarizado
- El p-valor simulado

**Paso 1 — Preparación del GeoDataFrame**

```python
gdf = gdf.dropna(subset=[col_x, col_y])
gdf = gdf.drop_duplicates(subset=['Seccion_Censal']).reset_index(drop=True)
```

Se eliminan filas con NaN en las variables de análisis y se deduplica por código de sección censal. Este último paso es crítico: el join con el CSV puede generar múltiples filas por sección (una por año u observación), y la matriz de pesos espaciales exige que cada polígono sea una unidad única. Sin esta deduplicación se produce un error de dimensiones en la multiplicación matricial (`matmul dimension mismatch`).

**Paso 2 — Construcción de la matriz de pesos W**

Para **Málaga**, si existe el archivo `correlacion_espacial/MLG_23_16.gal`, se carga la matriz de pesos oficial del INE:

```python
w_full = libpysal.io.open(gal_path).read()
ids_comun = [i for i in w_full.id_order if str(i) in ids_gdf_set]
w = libpysal.weights.util.w_subset(w_full, ids_comun)
```

El archivo `.gal` define las relaciones de vecindad entre las 436 secciones censales de Málaga usando el criterio de contigüidad Reina (Queen) calculado sobre los límites oficiales INE. Usar este archivo garantiza reproducibilidad: el mismo resultado que obtendría GeoDa o cualquier otro SIG con los mismos datos.

Para **el resto de ciudades** (y como fallback si el `.gal` falla), se calcula la contigüidad Reina al vuelo:

```python
w = libpysal.weights.Queen.from_dataframe(gdf)
```

La contigüidad Reina considera vecinas dos secciones si comparten al menos un punto de su frontera (incluye esquinas), a diferencia de la contigüidad Rook que solo considera fronteras compartidas en segmentos.

En ambos casos se aplica la transformación de estandarización por filas (`w.transform = 'r'`), que normaliza los pesos de cada observación para que sumen 1. Esto hace que el lag espacial sea una media ponderada de los vecinos, lo que facilita la interpretación.

**Paso 3 — Cálculo del I Global**

```python
moran_global = Moran_BV(x, y, w)
```

El estadístico global `moran_global.I` indica la fuerza y dirección de la autocorrelación. El Z-score (`moran_global.z_sim`) es la estandarización basada en las permutaciones aleatorias. El p-valor (`moran_global.p_sim`) indica la probabilidad de obtener ese I por azar.

Umbrales de interpretación usados en el visor:

| Z-score | Nivel de significancia | Mensaje |
|---|---|---|
| ≥ 2,58 | 99,9 % | Efecto de contagio espacial estadísticamente robusto |
| ≥ 1,96 | 95 % | Correlación espacial real |
| < 1,96 | No significativo | Patrón no concluyente |

**Paso 4 — Cálculo del I Local (LISA)**

```python
moran_local = Moran_Local_BV(x, y, w, permutations=999)
```

El análisis LISA (Local Indicators of Spatial Association) asigna a cada sección censal un tipo de clúster según el cuadrante del diagrama de dispersión de Moran en el que se sitúa, condicionado a que el p-valor simulado sea ≤ 0,05:

| Cuadrante | Código | Significado en este contexto |
|---|---|---|
| q=1 | **HH** (Alto-Alto) | VUT alta + alquiler alto en sección y vecinos → núcleo saturado |
| q=2 | **LH** (Bajo-Alto) | VUT baja pero alquiler alto en vecinos → efecto de contagio activo |
| q=3 | **LL** (Bajo-Bajo) | VUT baja + alquiler bajo → zonas residenciales con baja presión |
| q=4 | **HL** (Alto-Bajo) | VUT alta pero alquiler bajo en vecinos → anomalía o proceso emergente |
| — | **No Significativo** | p > 0,05: no se puede rechazar la hipótesis nula de aleatoriedad |

Las 999 permutaciones generan distribuciones de referencia bajo la hipótesis nula. El coste computacional de este paso (el más lento de la aplicación) es de 1–3 segundos para conjuntos de ~400 secciones.

---

## 7. Módulos del visor (tabs)

### 7.1 Mapa interactivo

**Propósito:** visualización coroplética por sección censal con panel de KPIs sincronizado.

**Controles:** los tres selectores superiores (ciudad, año, indicador) filtran el GeoDataFrame global a `df_f`, que contiene solo las secciones de la ciudad y año seleccionados.

**Modos de visualización:**

| Modo | Activación | Paleta | Descripción |
|---|---|---|---|
| Indicador continuo | Por defecto | YlOrRd / OrRd | Gradiente de color para Em2, %VUT, KPIs |
| Simulador | Checkbox sidebar | YlOrRd | Misma paleta pero con valores recalculados |
| Clústeres Moran | Checkbox sidebar | Discreta 5 colores | Resultado del análisis LISA |

**Escala de color consistente:** Los rangos `MAX_PRECIO` y `MAX_VUT` se calculan como el percentil 98 del conjunto completo (todas las ciudades y años) en el momento de carga. Esto garantiza que al cambiar de ciudad la escala no se reajusta, permitiendo comparaciones visuales directas entre Málaga y Teruel, por ejemplo.

**Centrado del mapa:** Se usa un diccionario `CAM` con coordenadas y nivel de zoom predefinidos por ciudad. Si la ciudad no está en el diccionario (caso de escalabilidad), el centro se calcula automáticamente como el centroide medio de las geometrías disponibles.

**Tooltip:** Se configura con `hover_name` (código de sección) y `hover_data` (variables numéricas con formato). El estilo del tooltip se personaliza vía `hoverlabel` (fondo blanco, fuente Inter 13px) para coherencia con el resto de la interfaz.

**Panel de KPIs (columna derecha):** Muestra métricas dinámicas para la ciudad y año seleccionados. Si el simulador está activo, los KPIs reflejan los valores simulados y se etiquetan como "KPIs SIMULADOS". Los KPIs del Censo (población, presión, ratio) solo aparecen si `INE_C2021_Indicadores.xlsx` está disponible (`TIENE_CENSO = True`).

**Exportación:** Botón de descarga que genera un CSV con las columnas visibles para la vista actual (incluyendo columnas simuladas si el modo simulador está activo).

### 7.2 Evolución comparada

**Propósito:** analizar tendencias temporales entre las cuatro ciudades en un único gráfico de líneas, para identificar divergencias entre destinos de alta y baja intensidad turística.

**Cálculo:** Se agrupan todos los datos por `(municipio, Fecha_ano)` y se calcula la **media** de cada indicador. Esto produce un único valor representativo por ciudad y año, agregando las variaciones entre secciones censales.

**Elementos del gráfico:**

- Línea vertical discontinua en 2020 con anotación "COVID-19", para contextualizar el impacto de la pandemia en la serie histórica.
- `hovermode="x unified"`: al pasar el cursor sobre un año, se muestran los valores de todas las ciudades simultáneamente.
- Paleta de colores fija por ciudad (Málaga rojo, Sevilla naranja, Jaén azul, Teruel gris) para coherencia visual entre tabs.

**Tabla resumen:** Tabla pivotada que muestra los valores en los años clave 2016, 2019, 2021 y 2023, junto con la variación porcentual acumulada 2016→2023. Esta tabla es la base de comparación directa con los datos del Informe IATUR en el Tab 5.

### 7.3 Tabla de datos

**Propósito:** inspección y descarga de los microdatos por sección censal para un municipio y año concretos.

**Columnas mostradas:**

| Columna | Fuente |
|---|---|
| `Seccion_Censal` | Código INE 10 dígitos |
| `Em2` | CSV principal |
| `%VUT` | CSV principal |
| `VUT.Formula` | CSV principal |
| `Viviendas.Formula` | CSV principal |
| `pob_total`, `viv_principales`, `viv_alquiler` | Censo 2021 (si disponible) |
| `Presion_1000hab`, `Ratio_turistif`, `Pct_viv_alquiler` | KPIs calculados |
| `pct_extranjeros`, `pct_parados` | Censo 2021 (si disponible) |

La columna `%VUT_display` se genera al vuelo escalando el valor a porcentaje legible (multiplicando por 100 si el valor máximo es ≤ 1,0, lo que indica que viene en formato decimal).

### 7.4 Modelo de regresión OLS

**Propósito:** presentar el modelo econométrico de referencia y ofrecer un simulador interactivo de escenarios.

**El modelo de efectos fijos:**

El modelo estimado por IATUR es:

```
Em2 = β₀ + β₁·(%VUT) + β₂·Sevilla + β₃·Jaén + β₄·Teruel
         + γ₂₀₁₇·D2017 + ... + γ₂₀₂₃·D2023 + ε
```

Donde:
- La constante `β₀ = 7,261` corresponde a Málaga en 2016 (ciudad y año de referencia)
- `β₁ = 0,329` es el coeficiente de interés: cada punto porcentual adicional de VUT sube el alquiler 0,329 €/m²
- Los coeficientes de ciudad capturan diferencias estructurales de precio entre mercados
- Los coeficientes de año capturan la tendencia temporal común a todas las ciudades

El modelo explica el 53,9 % de la varianza del precio del alquiler (`R² = 0,539`).

**Simulador interactivo:**

El usuario selecciona ciudad, año y porcentaje de VUT. La aplicación calcula:

```python
precio_pred = 7.261 + 0.329 * pct_vut_sim + coef_ciudad + coef_ano
alquiler_70m2 = precio_pred * 70
```

El resultado se muestra junto a un gráfico de sensibilidad que dibuja la función de precio estimado para el rango 0–20 % de VUT, manteniendo fijos ciudad y año. El punto seleccionado por el usuario se superpone como marcador.

**¿Por qué no se reestima el modelo dinámicamente?** El modelo requeriría `statsmodels` o `scikit-learn` y datos completos de todas las ciudades para producir estimaciones válidas. Al tratarse de un modelo validado y publicado, replicarlo estáticamente garantiza consistencia con la fuente académica y evita que variaciones en el subconjunto de datos cargado generen coeficientes diferentes en cada sesión.

### 7.5 Validación con Informe IATUR

**Propósito:** contrastar los resultados calculados dinámicamente por el visor con las cifras publicadas en el Informe IATUR (septiembre 2024), validando la coherencia metodológica.

**Bloque 1 — Precios y plazas (2023):**  
Calcula en tiempo real el precio medio y la variación 2016→2023 para cada ciudad, y lo compara con las tablas 5, 6 del informe. La diferencia se resalta en rojo si supera 1 €/m². Las discrepancias esperadas se deben a que el informe usa precios de oferta de portales inmobiliarios (Idealista, Fotocasa) mientras el visor usa la media del CSV a nivel de sección censal.

**Bloque 2 — Coeficientes OLS:**  
Tabla de comparación que muestra que todos los parámetros del modelo son idénticos en el visor y en el informe (fuente común). Incluye la simulación 0 % vs 10 % VUT por ciudad en 2023.

**Bloque 3 — Índice de Moran:**  
Referencia exacta publicada para Málaga 2023 (I = 0,40 · Z = 17,6 · p < 0,001) y explicación de cómo reproducirla en el visor. Un valor cercano a estos parámetros al activar el análisis confirma la validez del cálculo.

**Bloque 4 — Tabla de diferencias metodológicas:**  
Sistematiza las diferencias entre el informe (procesado con GeoDa, datos trimestrales, precios de oferta) y el visor (PySAL, datos anuales, media del CSV), para contextualizar correctamente las discrepancias numéricas observadas.

---

## 8. Panel de control (barra lateral)

### Activar Clústeres (Moran Bivariado)

Al activar este checkbox, el indicador del mapa se sustituye por el análisis LISA. El selector de indicador se deshabilita y se muestra la leyenda de clústeres en el panel de KPIs. El análisis se recalcula cada vez que cambia la ciudad o el año (no está en caché porque depende de las selecciones del usuario).

### Simulador de Impacto OLS

Al activar el checkbox aparece un slider (0–20 puntos de %VUT, paso 0,5). La app añade dos columnas simuladas al GeoDataFrame de la vista actual:

```python
df_f['%VUT_Sim'] = df_f['%VUT'] + (puntos_subida / factor)
df_f['Em2_Sim']  = df_f['Em2']  + (puntos_subida * COEF_VUT)
```

El `factor` es 1 si `%VUT` ya está en escala de porcentaje (0–100) y 100 si está en escala decimal (0–1). Este autodetector evita que el usuario tenga que conocer el formato interno de los datos.

### Añadir nueva ciudad

Bloque informativo que documenta el proceso de escalabilidad (ver sección 10).

---

## 9. Sistema de caché y rendimiento

Streamlit reejecutar el script completo en cada interacción. Para evitar recargar y procesar los datos en cada click, se usa el decorador `@st.cache_data`:

```python
@st.cache_data
def load_and_merge_data():
    ...
```

Este decorador almacena el resultado de la función en memoria (indexado por los argumentos). Las funciones ETL, que incluyen lectura de disco, parsing de GeoPackage y múltiples joins, solo se ejecutan una vez por sesión. Las operaciones que sí se recalculan en cada interacción son:

- El filtro por ciudad y año (`df_f = gdf[...]`)
- El análisis de Moran (cuando está activo)
- Todos los cálculos de KPIs mostrados en pantalla

El análisis de Moran es el paso más costoso computacionalmente (999 permutaciones Monte Carlo). Para ~400 secciones censales, tarda entre 1 y 3 segundos. Se muestra un spinner (`st.spinner`) durante la espera.

---

## 10. Escalabilidad — cómo añadir una nueva ciudad

La arquitectura está diseñada para incorporar nuevas ciudades sin modificar el código:

### Paso 1 — Preparar el GeoPackage

Crear un archivo `.gpkg` con el seccionado censal de la ciudad. El archivo debe tener:
- Una columna `CUSEC` con el código de sección censal a 10 dígitos
- Una columna `municipio` o `NMUN` con el nombre del municipio
- Geometrías en cualquier sistema de referencia (la app reprojecta a EPSG:4326 automáticamente)

Guardar como `NOMBRECIUDAD_FINAL_CENSO.gpkg` en la carpeta `censo_ciudades/`.

### Paso 2 — Añadir al CSV principal

Incluir en `modeloRegresionLineal.csv` las filas correspondientes a las secciones censales de la nueva ciudad con el mismo formato: `SSCC`, `Fecha_ano`, `Em2`, `%VUT`, `VUT.Formula`, `Viviendas.Formula`.

### Paso 3 — Registrar el archivo en app.py

Añadir el nombre del archivo a la lista `nombres_gpkg`:

```python
nombres_gpkg = [
    "MALAGA_FINAL_CENSO2021.gpkg",
    "SEVILLA_FINAL_CENSO.gpkg",
    "JAEN_FINAL_CENSO.gpkg",
    "Teruel_FINAL_CENSO.gpkg",
    "GRANADA_FINAL_CENSO.gpkg"    # nueva ciudad
]
```

Y añadir las coordenadas de centrado al diccionario `CAM`:

```python
CAM = {
    ...
    'granada': {"lat": 37.1773, "lon": -3.5986, "zoom": 12.0},
}
```

### Paso 4 — Reiniciar la aplicación

Streamlit recargará los datos y la nueva ciudad aparecerá automáticamente en el selector de ciudades.

**Nota sobre el modelo OLS:** los coeficientes de ciudad son fijos (tabla del Informe IATUR). Si se añade una ciudad no incluida en el informe original, el simulador la tratará con coeficiente 0 (equivalente a Málaga). Para usar coeficientes propios habría que reestimar el modelo con los nuevos datos.

---

## 11. Dependencias y entorno

### Librerías principales

| Librería | Versión mínima | Uso |
|---|---|---|
| `streamlit` | 1.30 | Framework web |
| `pandas` | 2.0 | Manipulación de datos tabulares |
| `geopandas` | 0.14 | Datos vectoriales geoespaciales |
| `plotly` | 5.18 | Visualizaciones interactivas (Express + Graph Objects) |
| `libpysal` | 4.9 | Pesos espaciales y lectura de archivos .gal |
| `esda` | 2.5 | Estadísticos de autocorrelación espacial (Moran) |
| `numpy` | 1.26 | Operaciones numéricas sobre arrays |
| `openpyxl` | 3.1 | Lectura de archivos .xlsx (Censo 2021) |

### Instalar dependencias

```bash
pip install streamlit pandas geopandas plotly libpysal esda numpy openpyxl
```

### Ejecutar la aplicación

Desde el directorio del proyecto, con el entorno virtual activo:

```bash
streamlit run app.py
```

La aplicación abre en `http://localhost:8501` por defecto.

---

## 12. Limitaciones conocidas y decisiones de diseño

### Datos anuales vs. trimestrales

El CSV `modeloRegresionLineal.csv` agrega los datos a nivel anual. El Informe IATUR original trabaja con datos trimestrales, lo que produce series más suavizadas y mayor precisión estadística. El visor muestra valores anuales medios, lo que puede generar diferencias en el precio medio respecto al informe.

### Censo 2021 como dato transversal

Los indicadores del Censo 2021 (población, viviendas, porcentajes sociodemográficos) son un único corte temporal aplicado a toda la serie 2016–2023. Esto implica que la presión turística calculada para 2016 usa la población de 2021 como denominador, lo que sobreestima o subestima la presión según la evolución poblacional de cada sección.

### Modelo OLS estático

Los coeficientes del simulador son fijos (Tabla 4 del Informe IATUR). El modelo no se reestima dinámicamente, por lo que no refleja cambios en el mercado posteriores a 2023 ni permite añadir nuevas variables explicativas sin modificar el código.

### Archivo .gal solo para Málaga

El archivo de pesos espaciales oficiales (`MLG_23_16.gal`) solo está disponible para Málaga. Para Sevilla, Jaén y Teruel se calcula la contigüidad Reina al vuelo, lo que es metodológicamente equivalente pero puede producir ligeras diferencias si los límites del seccionado han cambiado respecto a los usados para generar el `.gal`.

### Interpretación de clústeres LH

El clúster LH (Bajo-Alto) en el contexto de este análisis bivariado significa: sección con VUT baja rodeada de vecinos con alquiler alto. Esto puede interpretarse como el frente de avance del efecto de contagio (secciones residenciales que aún no han sido turistificadas pero ya están siendo arrastradas en precio), pero también puede ser un artefacto de borde en zonas con geometrías irregulares. La interpretación siempre debe complementarse con el conocimiento del territorio.

### Rendimiento en sesiones múltiples

Streamlit Community Cloud y otras plataformas de despliegue comparten memoria entre sesiones si se usa `@st.cache_data` sin argumentos de `ttl` (tiempo de vida). Si los datos cambian (nuevo CSV), hay que reiniciar la aplicación manualmente o añadir `@st.cache_data(ttl=3600)` para que la caché expire cada hora.

---

*Fin del manual técnico. Para consultas sobre el código, la metodología o la extensión del sistema, contactar con el autor del TFM.*
