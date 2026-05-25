# Dashboard VUT — Developer Guide

Guía técnica para desarrolladores que reciban este proyecto o quieran extenderlo.
El dashboard analiza la relación entre Viviendas de Uso Turístico (VUT) y el
mercado de alquiler residencial en cuatro ciudades españolas (Málaga, Sevilla,
Jaén y Teruel) para el período 2016–2023.

Repositorio público: https://github.com/diegof257/TFM_Dashboadr_VUT

---

## Tabla de contenidos

1. [Requisitos y arranque](#1-requisitos-y-arranque)
2. [Estructura de carpetas](#2-estructura-de-carpetas)
3. [Archivos de datos](#3-archivos-de-datos)
4. [Arquitectura de la aplicación](#4-arquitectura-de-la-aplicación)
5. [Pipeline ETL detallado](#5-pipeline-etl-detallado)
6. [Módulo de análisis espacial — Moran Bivariado](#6-módulo-de-análisis-espacial--moran-bivariado)
7. [Módulo de forecasting — SARIMAX](#7-módulo-de-forecasting--sarimax)
8. [Tabs del dashboard](#8-tabs-del-dashboard)
9. [Cómo añadir una nueva ciudad](#9-cómo-añadir-una-nueva-ciudad)
10. [Decisiones de diseño relevantes](#10-decisiones-de-diseño-relevantes)

---

## 1. Requisitos y arranque

```bash
# Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Lanzar la aplicación
streamlit run app.py
```

La app abre en `http://localhost:8501` por defecto.

**Python mínimo recomendado:** 3.11

---

## 2. Estructura de carpetas

```
TFM_Dashboadr_VUT/
│
├── app.py                      # Aplicación principal (único punto de entrada)
├── requirements.txt            # Dependencias de producción
├── DEVELOPER.md                # Este documento
├── CONTEXTO_TFM.md             # Contexto académico del proyecto
│
├── datos/                      # Datos tabulares
│   ├── modeloRegresionLineal.csv   # Fuente principal (VUT + precios por sección)
│   └── INE_C2021_Indicadores.xlsx  # Indicadores del Censo 2021
│
├── censo_ciudades/             # Geometrías vectoriales por ciudad
│   ├── MALAGA_FINAL_CENSO2021.gpkg
│   ├── SEVILLA_FINAL_CENSO.gpkg
│   ├── JAEN_FINAL_CENSO.gpkg
│   └── Teruel_FINAL_CENSO.gpkg
│
├── correlacion_espacial/       # Archivos para el análisis de Moran
│   ├── MLG_23_16.gal           # Matriz de pesos oficial INE para Málaga
│   └── MLG_23_16.gpkg          # Capa espacial auxiliar de Málaga
│
├── shapefiles/                 # Shapefiles complementarios (no cargados en runtime)
├── scripts/                    # Scripts auxiliares de preprocesamiento
└── docs/                       # Documentación adicional
```

---

## 3. Archivos de datos

### `datos/modeloRegresionLineal.csv`
**Fuente principal.** Generado por el equipo investigador del IATUR (Universidades
de Granada, Málaga y Sevilla, 2024).

| Columna | Tipo | Descripción |
|---|---|---|
| `SSCC` | str (10 dígitos) | Código de sección censal (CUSEC del INE) |
| `municipio` | str | Nombre del municipio |
| `Fecha` | str `dd/mm/yy` | Fecha de la observación (trimestral) |
| `Fecha_ano` | int | Año de la observación (corregido para Q4) |
| `Em2` | float | Precio medio del alquiler (€/m²) |
| `%VUT` | float | Proporción de VUT sobre el total de viviendas |
| `VUT.Formula` | float | Número de VUTs registradas |
| `Viviendas.Formula` | float | Total de viviendas en la sección censal |

**Nota importante sobre fechas:** El CSV usa fechas del tipo `1/1/yy`, `1/4/yy`,
`1/7/yy`, `1/10/yy` para los cuatro trimestres. El dato de Q4 de cada año se
publica el 1 de enero del año siguiente (e.g., `01/01/2024` = datos de Q4 2023).
El pipeline corrige este desfase automáticamente en el paso A del ETL.

**Nota sobre `%VUT`:** En algunas versiones del CSV, `%VUT` está almacenado como
proporción `[0, 1]` y en otras como porcentaje `[0, 100]`. El código detecta esto
en tiempo de ejecución comparando con el umbral `<= 1.0` y aplica el factor `×100`
cuando es necesario.

### `datos/INE_C2021_Indicadores.xlsx`
Indicadores socioeconómicos del Censo 2021 del INE para todas las secciones
censales de España (36.333 filas). Se hace un `left join` con el dataset principal
en el paso D del ETL. Si el archivo no existe, el dashboard funciona sin estas
columnas (no es bloqueante).

Columnas usadas: `CUSEC`, `t1_1` (población), `t18_1`–`t21_1` (viviendas),
`t5_1` (extranjeros), `t10_1` (parados), `t9_1` (estudios superiores).

### `censo_ciudades/*.gpkg`
Archivos GeoPackage con las geometrías de las secciones censales de cada ciudad.
La columna de join es `CUSEC` (código de 10 dígitos, mismo estándar que el CSV).
El CRS original de los .gpkg se detecta automáticamente; la app los reproyecta
a EPSG:25830 para calcular áreas y luego a EPSG:4326 para el mapa web.

### `correlacion_espacial/MLG_23_16.gal`
Matriz de pesos espaciales oficial del INE para las secciones censales de Málaga
(formato GAL de PySAL/libpysal). Se usa en el análisis de Moran para garantizar
reproducibilidad exacta con los valores publicados en el informe IATUR
(I = 0.40, Z = 17.6, p < 0.001). Para el resto de ciudades se calculan pesos
Queen directamente desde la geometría.

---

## 4. Arquitectura de la aplicación

`app.py` está estructurado en capas lineales; Streamlit se ejecuta de arriba
a abajo en cada interacción del usuario:

```
┌─────────────────────────────────────────────┐
│  Imports + configuración de página          │  líneas ~1–67
├─────────────────────────────────────────────┤
│  calcular_moran_bivariado()                 │  motor de análisis espacial
├─────────────────────────────────────────────┤
│  load_census_indicators()  ┐                │
│  load_and_merge_data()     ├ @st.cache_data │  ETL (sólo se ejecuta una vez)
│  load_series_forecasting() ┘                │
├─────────────────────────────────────────────┤
│  Carga inicial + rangos globales            │  gdf, MAX_PRECIO, MAX_VUT
├─────────────────────────────────────────────┤
│  Sidebar + Header + Filtros globales        │  ciudad, año, indicador
├─────────────────────────────────────────────┤
│  Tab 1: Mapa interactivo                    │  choropleth + KPIs + export CSV
│  Tab 2: Evolución comparada                 │  líneas temporales + tabla resumen
│  Tab 3: Tabla de datos                      │  tabla filtrable + export CSV
│  Tab 4: Forecasting                         │  SARIMAX + gráfico + métricas
└─────────────────────────────────────────────┘
```

**Caché:** `@st.cache_data` en las tres funciones ETL garantiza que los archivos
de datos sólo se leen y procesan una vez por sesión, incluso cuando el usuario
cambia filtros o navega entre tabs. El caché se invalida automáticamente si
cambia alguno de los argumentos de la función (en este caso ninguna recibe
argumentos variables, así que persiste toda la sesión).

---

## 5. Pipeline ETL detallado

La función `load_and_merge_data()` ejecuta siete pasos en orden:

| Paso | Qué hace | Por qué |
|---|---|---|
| **A** | Lee `modeloRegresionLineal.csv` | Fuente principal de VUT y precios |
| **B** | Lee los cuatro `.gpkg` de ciudades | Proporciona las geometrías para el mapa |
| **C** | Join mapa + CSV en `CUSEC`/`SSCC` | Une datos tabulares con geometría espacial |
| **D** | Join con Censo 2021 (opcional) | Añade indicadores socioeconómicos de contexto |
| **E** | Deduplica filas trimestrales | El CSV tiene hasta 4 filas/sección/año; conserva Q4 |
| **F** | Calcula `Tasa_crec_VUT` | Variación anual del %VUT por sección censal |
| **G** | Calcula `IDS_VUT` | Índice de Difusión Espacial (efecto "mancha de aceite") |

**Columnas derivadas generadas por el pipeline:**

| Columna | Descripción |
|---|---|
| `area_km2` | Superficie de la sección en km² (calculada en UTM 30N) |
| `VUT_km2` | Densidad de VUT por km² |
| `Pct_viv_alquiler` | % de viviendas principales en alquiler (Censo 2021) |
| `Tasa_crec_VUT` | Crecimiento anual del %VUT respecto al año anterior |
| `IDS_VUT` | Diferencia entre la tasa de crecimiento propia y el lag espacial Queen |
| `Cluster_Moran` | Etiqueta LISA (HH/LH/LL/HL/NS) — generada bajo demanda |

---

## 6. Módulo de análisis espacial — Moran Bivariado

La función `calcular_moran_bivariado()` se invoca sólo cuando el usuario activa
el checkbox **"Activar Clústeres (Moran Bivariado)"** en la barra lateral.
Calcula 999 permutaciones, lo que puede tardar entre 5 y 20 segundos según la
ciudad y el hardware.

**Lógica de pesos espaciales:**
- **Málaga:** intenta cargar `MLG_23_16.gal` (pesos oficiales INE). Si los IDs
  del archivo coinciden con ≥ 10 secciones del GeoDataFrame filtrado, usa ese
  subconjunto con `libpysal.weights.util.w_subset()`. Esto garantiza que el
  valor de I sea comparable con el publicado en el informe IATUR.
- **Resto de ciudades:** genera pesos Queen en tiempo real desde la geometría
  del GeoDataFrame con `libpysal.weights.Queen.from_dataframe()`.

En ambos casos, los pesos se normalizan por fila (`transform = 'r'`) antes de
pasarlos a `Moran_BV` y `Moran_Local_BV` de la librería `esda`.

**Interpretación de clusters LISA (p ≤ 0.05):**

| Código | Significado |
|---|---|
| HH (q=1) | Sección con alta VUT rodeada de secciones con alto alquiler |
| LH (q=2) | Baja VUT rodeada de alto alquiler — posible efecto de contagio |
| LL (q=3) | Baja VUT y bajo alquiler — zonas residenciales no presionadas |
| HL (q=4) | Alta VUT rodeada de bajo alquiler — outlier o anomalía local |

---

## 7. Módulo de forecasting — SARIMAX

El tab Forecasting usa `statsmodels.tsa.statespace.sarimax.SARIMAX`.

**Configuración del modelo:**
- Orden: `(p=1, d=1, q=1)` — ARIMA con una diferenciación estacional
- Variable exógena: `%VUT` de la ciudad seleccionada (media trimestral)
- Horizonte: 8 trimestres (cubre 2024 y 2025 completos)
- Proyección exógena: media de los últimos 4 trimestres disponibles
- Método de optimización: `lbfgs` con máximo 200 iteraciones

**Fallback:** si el modelo con variable exógena no converge (p. ej., por datos
insuficientes en ciudades pequeñas), la app cae automáticamente a un modelo
`SARIMA(1,1,0)` sin variable exógena, sin interrumpir la experiencia de usuario.

**Serie temporal:** generada por `load_series_forecasting()`. A diferencia del
ETL principal, esta función **no deduplica** los cuatro trimestres por año, de
modo que SARIMAX dispone del máximo número de observaciones posible para el
entrenamiento (~31 trimestres × 4 ciudades).

---

## 8. Tabs del dashboard

| Tab | Descripción | Características principales |
|---|---|---|
| **Mapa interactivo** | Choropleth a nivel de sección censal | 6 indicadores, modo Moran, KPIs, export CSV |
| **Evolución comparada** | Series temporales 2016–2023 por ciudad | Línea COVID-19, tabla pivot con variación acumulada |
| **Tabla de datos** | Tabla filtrada por ciudad y año | Paginación, export CSV completo |
| **Forecasting** | Predicción SARIMAX 2024–2025 | Banda de confianza 95%, selección de año |

---

## 9. Cómo añadir una nueva ciudad

1. **Añadir el GeoPackage:** coloca el `.gpkg` en `censo_ciudades/`. El nombre
   puede ser cualquiera; añade la ruta a la lista `nombres_gpkg` en
   `load_and_merge_data()` (paso B).

2. **Añadir los datos al CSV:** las filas de la nueva ciudad deben incluirse en
   `modeloRegresionLineal.csv` con el mismo formato (columnas `SSCC`, `municipio`,
   `Fecha`, `Em2`, `%VUT`, etc.).

3. **Ajustar la cámara del mapa (opcional):** añade una entrada al diccionario
   `CAM` en el bloque del Tab 1 con las coordenadas de inicio y el nivel de zoom
   adecuado para la nueva ciudad.

4. **Lanzar la app:** la ciudad aparecerá automáticamente en los selectores
   porque se derivan de los datos cargados.

No es necesario modificar ninguna otra parte del código.

---

## 10. Decisiones de diseño relevantes

**¿Por qué `@st.cache_data` y no `@st.cache_resource`?**
`cache_data` serializa el resultado (GeoDataFrame), lo que permite que Streamlit
lo comparta de forma segura entre sesiones simultáneas. `cache_resource` es
más adecuado para conexiones a bases de datos o modelos de ML persistentes.

**¿Por qué se reproyecta a EPSG:25830 para calcular áreas?**
Los `.gpkg` pueden estar en distintos CRS originales. Proyectar todos a UTM
Zona 30N (sistema oficial para la España peninsular) antes de calcular áreas
garantiza precisión métrica uniforme independientemente del CRS de origen.

**¿Por qué se usa `quantile(0.98)` para los rangos de color?**
Las secciones del centro histórico de Málaga tienen valores de VUT y precio
extremos que, si se usan como máximo de la escala, comprimen toda la variación
del resto del mapa en colores indistinguibles. El percentil 98 elimina ese
efecto sin descartar observaciones reales.

**¿Por qué `Book` como `SourceType` en las referencias Word?**
El estilo APA de Word muestra el título completo en las citas en texto cuando
el tipo es `JournalArticle`. Usando `Book` se obtiene el formato compacto
`(Autor, Año)` que corresponde al estilo deseado.

**¿Por qué el modelo SARIMAX no deduplica los trimestres?**
Con sólo 8 años de datos (2016–2023), una serie anual tendría sólo 8 puntos por
ciudad — insuficientes para SARIMAX. Usar los 4 trimestres da ~31 observaciones,
lo que permite una estimación razonable de los parámetros AR y MA.

**Sobre el `.gal` de Málaga:**
El archivo `MLG_23_16.gal` es la matriz de contigüidad Queen generada por el INE
para las secciones censales de Málaga. Usarla garantiza que el I de Moran
calculado por el dashboard coincida con el publicado en el informe IATUR
(I = 0.40, Z = 17.6). Para otras ciudades no existe este archivo público, por lo
que se generan pesos Queen dinámicamente.
