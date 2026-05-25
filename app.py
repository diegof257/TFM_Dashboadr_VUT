import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import os
import libpysal
import libpysal.io
import libpysal.weights.util
from esda.moran import Moran_Local_BV, Moran_BV
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# statsmodels es necesario para el tab de Forecasting.
# Si no está instalado, el tab muestra un aviso de instalación en lugar de lanzar un error.
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    FORECASTING_DISPONIBLE = True
except ImportError:
    FORECASTING_DISPONIBLE = False


# ---------------------------------------------------------------------------
# CONSTANTES DE DIRECTORIOS DE DATOS
# ---------------------------------------------------------------------------
CENSO_DIR    = "censo_ciudades"        # GeoPackages (.gpkg) por ciudad
DATOS_DIR    = "datos"                 # CSV y Excel con los datos fuente
ESPACIAL_DIR = "correlacion_espacial"  # Pesos espaciales (.gal) y .gpkg auxiliares


def find_file(*paths):
    """Devuelve la primera ruta existente de entre las candidatas proporcionadas.

    Soporta dos disposiciones: la empaquetada (p.ej. datos/archivo.csv) y la
    plana, donde los archivos de datos están junto a app.py. Si ninguna ruta
    existe, devuelve la última como fallback para que el mensaje de error sea
    descriptivo.
    """
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[-1]


# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILOS GLOBALES
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Dashboard VUT · Turismo y Vivienda"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@700&family=Inter:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1 { font-family: 'Lora', serif; font-size: 2.2rem !important; }
.stMetric {
    background-color: #f8f9fa;
    padding: 14px;
    border-radius: 10px;
    border-left: 5px solid #c53030;
}
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# MOTOR ANALÍTICO — I DE MORAN BIVARIADO
# ===========================================================================
def calcular_moran_bivariado(df_input, col_x='%VUT', col_y='Em2', ciudad=None):
    """Calcula el I de Moran Global y Local Bivariado (X = %VUT, Y = €/m²).

    Para Málaga intenta cargar el archivo oficial de pesos espaciales del INE (.gal)
    para garantizar la reproducibilidad de los valores publicados en el informe IATUR
    (referencia: I = 0,40, Z = 17,6, p < 0,001). Para el resto de ciudades se
    calculan pesos de contigüidad Reina (Queen) directamente desde la geometría.

    Parámetros
    ----------
    df_input : GeoDataFrame
        Datos de secciones censales para una ciudad y año concretos.
    col_x : str
        Columna usada como variable de retardo espacial (tasa de saturación VUT).
    col_y : str
        Columna usada como variable dependiente (precio del alquiler €/m²).
    ciudad : str, opcional
        Nombre del municipio, usado para seleccionar la fuente de pesos espaciales.

    Retorna
    -------
    gdf : GeoDataFrame
        Datos de entrada con la columna 'Cluster_Moran' añadida (HH/LH/LL/HL/NS).
    I : float
        Estadístico I de Moran Global.
    z_sim : float
        Z-score de la distribución de permutaciones (999 permutaciones).
    p_sim : float
        Pseudo p-valor del test de permutaciones.
    """
    gdf = gpd.GeoDataFrame(df_input, geometry='geometry').copy()
    gdf = gdf.dropna(subset=[col_x, col_y])
    gdf = gdf.drop_duplicates(subset=['Seccion_Censal']).reset_index(drop=True)

    w = None
    gal_path = os.path.join(ESPACIAL_DIR, "MLG_23_16.gal")
    es_malaga = ciudad and 'malaga' in ciudad.lower().replace('á', 'a')

    # Intentar cargar el archivo oficial de pesos del INE para Málaga
    if es_malaga and os.path.exists(gal_path):
        try:
            w_full = libpysal.io.open(gal_path).read()
            ids_gdf_set = set(gdf['Seccion_Censal'].astype(str).tolist())
            ids_comun   = [i for i in w_full.id_order if str(i) in ids_gdf_set]

            if len(ids_comun) >= 10:
                # Reordenar el GDF para que coincida exactamente con el orden del .gal antes de hacer el subset
                gdf_idx = gdf.set_index('Seccion_Censal')
                gdf_idx = gdf_idx.loc[[str(i) for i in ids_comun]]
                w_sub   = libpysal.weights.util.w_subset(w_full, ids_comun)
                w_sub.transform = 'r'

                if len(gdf_idx) == len(w_sub):
                    gdf = gdf_idx.reset_index()
                    w   = w_sub
                    st.sidebar.caption(
                        f"Pesos espaciales: archivo oficial .gal INE "
                        f"({len(ids_comun)} secciones)"
                    )
        except Exception:
            w = None  # Fallback a pesos Queen si ocurre cualquier error

    if w is None:
        w = libpysal.weights.Queen.from_dataframe(gdf)
        w.transform = 'r'

    x, y = gdf[col_x].values, gdf[col_y].values
    moran_global = Moran_BV(x, y, w)
    moran_local  = Moran_Local_BV(x, y, w, permutations=999)

    # Asignar etiquetas de clúster LISA con nivel de significancia p ≤ 0,05
    sig = 0.05
    clusters = np.full(len(gdf), 'No Significativo', dtype=object)
    clusters[(moran_local.q == 1) & (moran_local.p_sim <= sig)] = 'Alto-Alto (HH)'
    clusters[(moran_local.q == 2) & (moran_local.p_sim <= sig)] = 'Bajo-Alto (LH)'
    clusters[(moran_local.q == 3) & (moran_local.p_sim <= sig)] = 'Bajo-Bajo (LL)'
    clusters[(moran_local.q == 4) & (moran_local.p_sim <= sig)] = 'Alto-Bajo (HL)'

    gdf['Cluster_Moran'] = clusters
    return gdf, moran_global.I, moran_global.z_sim, moran_global.p_sim


# ===========================================================================
# ETL — CARGA Y ENRIQUECIMIENTO DE DATOS
# ===========================================================================
@st.cache_data
def load_census_indicators():
    """Carga los indicadores del Censo 2021 del INE para todas las secciones censales.

    El archivo Excel contiene 36.333 filas (una por sección censal) con variables
    socioeconómicas que contextualizan la presión de las VUT. Los nombres de columna
    se renombran desde los códigos internos del INE a identificadores descriptivos.

    Devuelve None si el archivo Excel no se encuentra; en ese caso el dashboard
    sigue funcionando pero las columnas derivadas del censo no estarán disponibles.
    """
    path = find_file(
        os.path.join(DATOS_DIR, "INE_C2021_Indicadores.xlsx"),
        "INE_C2021_Indicadores.xlsx"
    )
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path, dtype={'CUSEC': str})
    df['CUSEC'] = df['CUSEC'].astype(str).str.zfill(10)
    df = df.rename(columns={
        't1_1':  'pob_total',
        't18_1': 'viv_total_censo',
        't19_1': 'viv_principales',
        't19_2': 'viv_no_principales',
        't20_1': 'viv_propiedad',
        't20_2': 'viv_alquiler',
        't21_1': 'total_hogares',
        't5_1':  'pct_extranjeros',
        't10_1': 'pct_parados',
        't9_1':  'pct_estudios_sup',
    })
    cols = ['CUSEC', 'pob_total', 'viv_total_censo', 'viv_principales',
            'viv_no_principales', 'viv_propiedad', 'viv_alquiler',
            'total_hogares', 'pct_extranjeros', 'pct_parados', 'pct_estudios_sup']
    return df[cols]


@st.cache_data
def load_and_merge_data():
    """Carga, une y enriquece todos los datos espaciales y tabulares del dashboard.

    Pasos del pipeline:
      A. Lee modeloRegresionLineal.csv — registros de VUT y precios de alquiler
         por sección censal y trimestre (fuente: equipo investigador IATUR).
      B. Lee los GeoPackages (.gpkg) de cada ciudad desde CENSO_DIR.
      C. Join espacial: une la geometría del mapa con los datos tabulares
         mediante el código de sección censal (CUSEC / SSCC).
      D. Enriquece con indicadores socioeconómicos del Censo 2021 (opcional).
      E. Deduplica filas trimestrales: conserva el último snapshot por sección
         y año (Q4 = stock máximo anual de VUT).
      F. Calcula la tasa de crecimiento anual del %VUT (Tasa_crec_VUT).
      G. Calcula el Índice de Difusión Espacial (IDS_VUT).

    Retorna
    -------
    gdf : GeoDataFrame
        Una fila por sección censal y año, proyectado a EPSG:4326.
    """

    # ── A. Carga del dataset de regresión ────────────────────────────────────────
    try:
        csv_path = find_file(
            os.path.join(DATOS_DIR, "modeloRegresionLineal.csv"),
            "modeloRegresionLineal.csv"
        )
        df = pd.read_csv(csv_path, encoding="utf-8", sep=";", decimal=",",
                         dtype={'SSCC': str})
        df['SSCC'] = df['SSCC'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
        df['Fecha_ano'] = pd.to_numeric(df['Fecha_ano'], errors='coerce')
        for col in ['Em2', '%VUT', 'VUT.Formula', 'Viviendas.Formula']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.'), errors='coerce')
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%y', errors='coerce')
            # El dato de Q4 se publica el 1 de enero del año siguiente.
            # Ejemplo: fecha 01/01/2024 → datos de Q4 2023 → Fecha_ano corregida a 2023.
            mask_enero = df['Fecha'].dt.month == 1
            df.loc[mask_enero, 'Fecha_ano'] = df.loc[mask_enero, 'Fecha'].dt.year - 1
    except FileNotFoundError:
        st.error("No se encuentra 'modeloRegresionLineal.csv'. Colócalo en la misma carpeta que app.py.")
        st.stop()

    # ── B. Carga de GeoPackages por ciudad ─────────────────────────────────────
    nombres_gpkg = ["MALAGA_FINAL_CENSO2021.gpkg", "SEVILLA_FINAL_CENSO.gpkg",
                    "JAEN_FINAL_CENSO.gpkg", "Teruel_FINAL_CENSO.gpkg"]
    mapas = []
    for f in nombres_gpkg:
        ruta = find_file(os.path.join(CENSO_DIR, f), f)
        if os.path.exists(ruta):
            try:
                mapas.append(gpd.read_file(ruta))
            except Exception as e:
                st.sidebar.warning(f"No se pudo leer {f}: {e}")

    if not mapas:
        st.error("No se encontró ningún archivo .gpkg.")
        st.stop()

    gdf_mapa = gpd.GeoDataFrame(pd.concat(mapas, ignore_index=True))
    gdf_mapa['CUSEC'] = gdf_mapa['CUSEC'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)

    # ── C. Join espacial: mapa + datos tabulares ───────────────────────────────
    gdf = gdf_mapa.merge(df, left_on='CUSEC', right_on='SSCC', how='inner')
    gdf = gdf.rename(columns={'CUSEC': 'Seccion_Censal'})

    # Normalizar la columna municipio (puede llegar como NMUN en algunos .gpkg)
    if 'municipio' not in gdf.columns:
        gdf = gdf.rename(columns={'NMUN': 'municipio'})
    if isinstance(gdf['municipio'], pd.DataFrame):
        gdf['municipio'] = gdf['municipio'].iloc[:, 0]
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    gdf['municipio'] = gdf['municipio'].astype(str).str.title()

    # Calcular el área de cada sección en km² usando UTM Zona 30N (EPSG:25830) antes
    # de reproyectar a WGS84. Es necesario un CRS métrico para obtener áreas precisas
    # en la Península Ibérica.
    gdf_utm = gdf.to_crs(epsg=25830)
    gdf['area_km2'] = (gdf_utm.geometry.area / 1_000_000).round(4)

    gdf = gdf.to_crs(epsg=4326)

    # Informar de secciones sin cruzar en la barra lateral (informativo, no bloqueante)
    sin_cruzar = len(gdf_mapa) - len(gdf)
    if sin_cruzar > 0:
        st.sidebar.caption(f"{sin_cruzar} secciones sin datos numéricos (secciones no incluidas en el dataset IATUR para ese año).")

    # ── D. Enriquecimiento con indicadores del Censo 2021 ─────────────────────────────
    df_censo = load_census_indicators()
    if df_censo is not None:
        gdf = gdf.merge(df_censo, left_on='Seccion_Censal', right_on='CUSEC', how='left')

        # % de viviendas principales en régimen de alquiler (Censo 2021, dato estático).
        # Las secciones con mayor proporción de inquilinos son las más expuestas a la
        # presión sobre los precios derivada de las VUT.
        gdf['Pct_viv_alquiler'] = (
            gdf['viv_alquiler'] / gdf['viv_principales'].replace(0, np.nan) * 100
        ).round(2)

    # Densidad de VUT por km² (dinámica, derivada de la geometría — no requiere censo).
    # Captura la intensidad territorial con independencia del tamaño del parque inmobiliario.
    if 'area_km2' in gdf.columns:
        gdf['VUT_km2'] = (
            gdf['VUT.Formula'] / gdf['area_km2'].replace(0, np.nan)
        ).round(2)

    # ── E. Deduplicación trimestral → conservar el último snapshot por año ────────
    # El CSV contiene hasta 4 filas por sección y año (datos trimestrales).
    # Conservar el último registro trimestral (Q4) da el stock máximo anual
    # de VUT para cada sección censal.
    if 'Fecha' in gdf.columns:
        crs_original = gdf.crs
        gdf = (gdf.sort_values('Fecha')
                  .groupby(['Seccion_Censal', 'Fecha_ano'], as_index=False)
                  .last())
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=crs_original)

    # ── F. Tasa de crecimiento anual del %VUT (Tasa_crec_VUT) ─────────────────────────
    # Fórmula: (%VUT_t − %VUT_{t−1}) / %VUT_{t−1} × 100
    # NaN en 2016 (año base, sin referencia previa). Valores negativos indican
    # contracción de VUT (claramente visible durante el período pandémico 2020–2021).
    # Las secciones con menos de 3 VUTs registradas se excluyen para evitar tasas
    # artificialmente altas causadas por la incorporación de una sola unidad.
    if '%VUT' in gdf.columns:
        gdf = gdf.sort_values(['Seccion_Censal', 'Fecha_ano'])
        gdf['Tasa_crec_VUT'] = (
            gdf.groupby('Seccion_Censal', group_keys=False)['%VUT']
            .pct_change() * 100
        ).round(2)
        gdf['Tasa_crec_VUT'] = gdf['Tasa_crec_VUT'].clip(-100, 300)
        gdf['Tasa_crec_VUT'] = np.where(
            gdf['VUT.Formula'] >= 3,
            gdf['Tasa_crec_VUT'],
            np.nan
        )
        crs_orig = gdf.crs
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=crs_orig)

    # ── G. Índice de Difusión Espacial (IDS_VUT) ──────────────────────────────
    # IDS_i = Tasa_crec_VUT_i − retardo_espacial_Queen(Tasa_crec_VUT)_i
    # Valores positivos: la sección crece más rápido que sus vecinas (foco emisor).
    # Valores negativos: la sección crece más lento que sus vecinas (zona receptora).
    # Operacionaliza la hipótesis de difusión espacial "mancha de aceite" de las VUT.
    if 'Tasa_crec_VUT' in gdf.columns:
        ids_vals = pd.Series(np.nan, index=gdf.index)
        for (_, _), subset in gdf.groupby(['municipio', 'Fecha_ano']):
            valid = subset.dropna(subset=['Tasa_crec_VUT'])
            if len(valid) < 4:
                continue
            try:
                orig_idx = valid.index.values
                valid_r  = valid.reset_index(drop=True)
                w_ids = libpysal.weights.Queen.from_dataframe(valid_r)
                w_ids.transform = 'r'
                lag  = libpysal.weights.lag_spatial(w_ids, valid_r['Tasa_crec_VUT'].values)
                ids_vals.loc[orig_idx] = np.round(
                    valid_r['Tasa_crec_VUT'].values - lag, 2)
            except Exception:
                pass
        gdf['IDS_VUT'] = ids_vals
        crs_orig2 = gdf.crs
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=crs_orig2)

    return gdf


# ===========================================================================
# ETL — SERIES TRIMESTRALES PARA FORECASTING
# ===========================================================================
@st.cache_data
def load_series_forecasting():
    """Construye DataFrames de series temporales trimestrales para el tab de Forecasting.

    A diferencia de load_and_merge_data(), esta función NO deduplica las filas
    trimestrales. Se conservan los 4 snapshots por año para maximizar el número
    de observaciones de entrenamiento del modelo (aprox. 31 trimestres × 4 ciudades).

    Retorna
    -------
    series_df : DataFrame
        Precio medio trimestral del alquiler (€/m²) por ciudad, indexado por fecha (freq QS).
    exog_df : DataFrame
        %VUT medio trimestral por ciudad — usado como variable exógena en SARIMAX.
    vut_df : DataFrame
        Número medio de VUTs registradas por ciudad y trimestre.
    """
    csv_path = find_file(
        os.path.join(DATOS_DIR, "modeloRegresionLineal.csv"),
        "modeloRegresionLineal.csv"
    )
    df = pd.read_csv(csv_path, encoding="utf-8", sep=";", decimal=",", dtype={'SSCC': str})
    for col in ['Em2', '%VUT', 'VUT.Formula']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%y', errors='coerce')
    # Las fechas fuente siguen el patrón QS-JAN (1 de ene/abr/jul/oct).
    # La entrada 01/01/2024 representa Q4 2023 publicado en enero de 2024.
    # Se conserva la fecha original para mantener la regularidad de la serie en SARIMAX.

    # Agregación: media de todas las secciones censales por ciudad y trimestre
    df_q = (df.groupby(['municipio', 'Fecha'])[['Em2', '%VUT', 'VUT.Formula']]
              .mean().round(4).reset_index()
              .dropna(subset=['Fecha', 'Em2'])
              .sort_values('Fecha'))

    series_df = df_q.pivot(index='Fecha', columns='municipio', values='Em2')
    exog_df   = df_q.pivot(index='Fecha', columns='municipio', values='%VUT')
    vut_df    = df_q.pivot(index='Fecha', columns='municipio', values='VUT.Formula')

    series_df.index = pd.DatetimeIndex(series_df.index, freq='QS')
    exog_df.index   = pd.DatetimeIndex(exog_df.index,   freq='QS')
    vut_df.index    = pd.DatetimeIndex(vut_df.index,    freq='QS')

    return series_df, exog_df, vut_df


# ---------------------------------------------------------------------------
# CARGA INICIAL DE DATOS
# ---------------------------------------------------------------------------
gdf = load_and_merge_data()

# Límites globales de la escala de color, calculados una vez al inicio.
# Usar el percentil 98 evita que las secciones con valores extremos compriman
# el rango de colores del resto del mapa choropleth.
MAX_PRECIO = gdf['Em2'].quantile(0.98)
MAX_VUT    = gdf['%VUT'].quantile(0.98)

TIENE_CENSO = 'pob_total' in gdf.columns


# ===========================================================================
# BARRA LATERAL
# ===========================================================================
st.sidebar.title("Panel de Control")

st.sidebar.markdown("### Análisis Espacial")
mostrar_moran = st.sidebar.checkbox(
    "Activar Clústeres (Moran Bivariado)",
    help="Calcula la correlación espacial VUT↔Alquiler. Identifica el efecto 'Mancha de Aceite'."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Añadir nueva ciudad")
st.sidebar.info(
    "Para replicar en otra provincia:\n\n"
    "1. Añade su `.gpkg` a la carpeta\n"
    "2. Incorpora sus secciones al dataset IATUR\n"
    "3. La app lo detecta automáticamente"
)


# ===========================================================================
# CABECERA
# ===========================================================================
st.title("Dashboard de Mercado Inmobiliario y Presión Turística")
st.markdown(
    "Análisis espacial de la relación entre Viviendas de Uso Turístico (VUT) "
    "y el precio del alquiler residencial · Málaga, Sevilla, Jaén y Teruel (2016–2023) · "
    "Datos facilitados por el equipo investigador del IATUR (Universidades de Granada, Málaga y Sevilla) · "
    "Último dato disponible: Q4 2023"
)
st.markdown("---")


# ===========================================================================
# FILTROS GLOBALES (compartidos entre todos los tabs)
# ===========================================================================
c1, c2, c3 = st.columns(3)

with c1:
    ciudades = sorted(gdf['municipio'].dropna().unique().tolist())
    ciudad_focal = st.selectbox("Ciudad:", ciudades)

with c2:
    anos = sorted(gdf[gdf['municipio'] == ciudad_focal]['Fecha_ano'].dropna().unique().astype(int))
    ano_sel = st.select_slider("Año:", options=anos, value=max(anos))

with c3:
    indicadores_dict = {
        "Precio Alquiler (€/m²)":              "Em2",
        "Saturación Turística (%VUT)":          "%VUT",
        "Densidad VUT (VUTs/km²)":              "VUT_km2",
        "Crecimiento anual %VUT (%)":           "Tasa_crec_VUT",
        "Total VUTs registradas":               "VUT.Formula",
    }
    if 'IDS_VUT' in gdf.columns:
        indicadores_dict["Difusión Espacial VUT (IDS)"] = "IDS_VUT"
    if not mostrar_moran:
        nombre_ind = st.selectbox("Indicador:", list(indicadores_dict.keys()))
        col_ind    = indicadores_dict[nombre_ind]
    else:
        nombre_ind = "Clústeres Moran"
        col_ind    = "Cluster_Moran"
        st.info("Modo Moran activo: %VUT vs €/m²")


# ===========================================================================
# TABS PRINCIPALES
# ===========================================================================
tab_mapa, tab_evol, tab_datos, tab_forecast = st.tabs([
    "Mapa interactivo",
    "Evolución comparada",
    "Tabla de datos",
    "Forecasting",
])


# ---------------------------------------------------------------------------
# TAB 1 — MAPA INTERACTIVO
# ---------------------------------------------------------------------------
with tab_mapa:

    df_f = gdf[(gdf['municipio'] == ciudad_focal) & (gdf['Fecha_ano'] == ano_sel)].copy()

    col_map, col_kpi = st.columns([3, 1])

    # Posiciones de cámara por defecto para cada ciudad (calibradas manualmente para el zoom inicial óptimo)
    CAM = {
        'málaga':  {"lat": 36.7213, "lon": -4.4214, "zoom": 11.5},
        'sevilla': {"lat": 37.3891, "lon": -5.9845, "zoom": 11.5},
        'jaén':    {"lat": 37.7796, "lon": -3.7849, "zoom": 12.5},
        'teruel':  {"lat": 40.3457, "lon": -1.1065, "zoom": 13.5},
    }
    ciudad_key = ciudad_focal.lower().replace('á','a').replace('é','e').replace('ó','o')
    cam = CAM.get(ciudad_key, {
        "lat": df_f.geometry.centroid.y.mean(),
        "lon": df_f.geometry.centroid.x.mean(),
        "zoom": 11
    })

    with col_map:
        if df_f.empty:
            st.warning(f"Sin datos para {ciudad_focal} en {ano_sel}.")
        elif mostrar_moran:
            with st.spinner("Calculando permutaciones espaciales (999 iteraciones)…"):
                df_f, m_i, m_z, m_p = calcular_moran_bivariado(df_f, ciudad=ciudad_focal)

            # Interpretación automática de la significancia según los umbrales del Z-score
            if m_z >= 2.58:
                st.success(f"I Global de Moran: **{m_i:.4f}** | Z = **{m_z:.2f}** — Significancia extrema (99,9 %): el efecto de contagio espacial es estadísticamente robusto.")
            elif m_z >= 1.96:
                st.info(f"I Global de Moran: **{m_i:.4f}** | Z = **{m_z:.2f}** — Significancia alta (95 %): correlación espacial real.")
            else:
                st.warning(f"I Global de Moran: **{m_i:.4f}** | Z = **{m_z:.2f}** — Patrón no concluyente.")

            hover_moran = {'Em2': ':.2f', '%VUT': ':.4f', 'Cluster_Moran': True}
            if 'VUT_km2' in df_f.columns:
                hover_moran['VUT_km2'] = ':.2f'
            fig = px.choropleth_mapbox(
                df_f, geojson=df_f.geometry, locations=df_f.index,
                color='Cluster_Moran',
                color_discrete_map={
                    'Alto-Alto (HH)': '#b91c1c',
                    'Bajo-Alto (LH)': '#fca5a5',
                    'Bajo-Bajo (LL)': '#93c5fd',
                    'Alto-Bajo (HL)': '#1d4ed8',
                    'No Significativo': '#e5e7eb'
                },
                mapbox_style="carto-positron",
                center={"lat": cam['lat'], "lon": cam['lon']},
                zoom=cam['zoom'], opacity=0.85,
                hover_name='Seccion_Censal',
                hover_data=hover_moran,
                labels={
                    'Em2': 'Alquiler (€/m²)',
                    '%VUT': 'Saturación VUT (%)',
                    'Cluster_Moran': 'Clúster espacial',
                    'Presion_1000hab': 'Densidad VUT (VUTs/1000 hab)',
                }
            )
        else:
            # Seleccionar paleta de color y rango según el indicador activo
            es_tasa   = col_ind == 'Tasa_crec_VUT'
            es_ids    = col_ind == 'IDS_VUT'
            es_precio = col_ind == 'Em2'
            if es_tasa:
                paleta = "RdYlGn_r"   # divergente: rojo = crecimiento, verde = contracción
                vals = df_f['Tasa_crec_VUT'].dropna()
                lim = max(abs(vals.quantile(0.02)), abs(vals.quantile(0.98))) if len(vals) > 0 else 50
                rango = [-lim, lim]
            elif es_ids:
                paleta = "RdYlGn_r"   # divergente: rojo = sección emisora, verde = receptora
                vals = df_f['IDS_VUT'].dropna()
                lim = max(abs(vals.quantile(0.02)), abs(vals.quantile(0.98))) if len(vals) > 0 else 30
                rango = [-lim, lim]
            elif es_precio:
                paleta = "YlOrRd"
                rango  = [0, MAX_PRECIO]
            else:
                paleta = "OrRd"
                rango  = [0, MAX_VUT]

            # El tooltip siempre muestra precio y %VUT junto al indicador activo
            hover_cols = {'Em2': ':.2f', '%VUT': ':.4f'}
            for col_extra in ['VUT_km2', 'Tasa_crec_VUT', 'IDS_VUT']:
                if col_extra in df_f.columns:
                    hover_cols[col_extra] = ':.2f'

            # Los indicadores de tasa de crecimiento e IDS no están definidos para el año base (2016)
            if (es_tasa or es_ids) and ano_sel == min(anos):
                st.info("ℹ️ Este indicador no está disponible para 2016 (es el año base de la serie).")

            fig = px.choropleth_mapbox(
                df_f, geojson=df_f.geometry, locations=df_f.index,
                color=col_ind,
                color_continuous_scale=paleta, range_color=rango,
                mapbox_style="carto-positron",
                center={"lat": cam['lat'], "lon": cam['lon']},
                zoom=cam['zoom'], opacity=0.75,
                hover_name='Seccion_Censal',
                hover_data={k: v for k, v in hover_cols.items() if k in df_f.columns},
                labels={
                    'Em2':             'Alquiler (€/m²)',
                    '%VUT':            'Saturación VUT (%)',
                    'VUT.Formula':     'VUTs registradas',
                    'VUT_km2':         'VUTs por km²',
                    'Tasa_crec_VUT':   'Crecimiento anual %VUT (%)',
                    'IDS_VUT':         'Difusión Espacial VUT (IDS)',
                }
            )

        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            height=600,
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#d1d5db",
                font_size=13,
                font_family="Inter, sans-serif",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_kpi:
        st.subheader(f"KPIs · {ano_sel}")

        col_precio = 'Em2'
        col_vut    = '%VUT'

        # Detectar si %VUT está almacenado como proporción [0, 1] o como porcentaje [0, 100]
        factor_vut = 100 if df_f[col_vut].max() <= 1.0 else 1

        st.metric("Alquiler Medio", f"{df_f[col_precio].mean():.2f} €/m²")
        st.metric("Saturación VUT", f"{df_f[col_vut].mean() * factor_vut:.2f} %")
        st.metric("Total VUTs registradas", f"{int(df_f['VUT.Formula'].sum()):,}")
        if 'VUT_km2' in df_f.columns:
            st.metric("Concentración VUT", f"{df_f['VUT_km2'].mean():.2f} VUTs/km²")

        st.markdown("---")
        if mostrar_moran:
            st.markdown("**Leyenda de clústeres:**")
            st.markdown("**HH** — Centro saturado y caro")
            st.markdown("**LH** — Efecto de contagio espacial")
            st.markdown("**LL** — Zonas residenciales con baja presión")
            st.markdown("**HL** — Anomalías / outliers")

        # Exportar CSV de la vista actual del mapa
        st.markdown("---")
        st.markdown("**Exportar datos**")
        cols_export = ['Seccion_Censal', 'Em2', '%VUT', 'VUT.Formula']
        for c in ['VUT_km2', 'Tasa_crec_VUT', 'IDS_VUT', 'area_km2']:
            if c in df_f.columns: cols_export.append(c)

        csv_out = df_f[[c for c in cols_export if c in df_f.columns]].to_csv(
            index=False, decimal='.', sep=';').encode('utf-8')
        st.download_button(
            "Descargar vista (.csv)",
            data=csv_out,
            file_name=f"{ciudad_focal}_{ano_sel}_datos.csv",
            mime="text/csv"
        )


# ---------------------------------------------------------------------------
# TAB 2 — EVOLUCIÓN COMPARADA
# ---------------------------------------------------------------------------
with tab_evol:
    st.markdown("### Evolución temporal comparada entre ciudades")
    st.caption(
        "Permite verificar visualmente la divergencia entre destinos de alta intensidad "
        "turística (Málaga, Sevilla) y ciudades con baja presión VUT (Jaén, Teruel). "
    )

    ind_evol = st.selectbox(
        "Indicador a comparar:",
        ["Precio Alquiler (€/m²)", "Saturación Turística (%VUT)", "Total VUTs registradas"],
        key="ind_evol"
    )
    col_evol = indicadores_dict.get(ind_evol, "Em2")

    # Agregar a nivel de ciudad: media de todas las secciones censales por ciudad y año
    df_evol = (
        gdf.groupby(['municipio', 'Fecha_ano'])[col_evol]
        .mean()
        .reset_index()
        .rename(columns={col_evol: ind_evol, 'Fecha_ano': 'Año'})
    )

    paleta_ciudades = {
        'Málaga':  '#b91c1c',
        'Sevilla': '#d97706',
        'Jaén':    '#1d4ed8',
        'Teruel':  '#6b7280',
    }

    fig_line = px.line(
        df_evol, x='Año', y=ind_evol, color='municipio',
        markers=True,
        color_discrete_map=paleta_ciudades,
        labels={'municipio': 'Ciudad', 'Año': 'Año'},
        title=f"Evolución de {ind_evol} · 2016–2023"
    )
    fig_line.update_layout(
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig_line.add_vline(x=2020, line_dash="dot", line_color="#6b7280",
                       annotation_text="COVID-19", annotation_position="top right")
    st.plotly_chart(fig_line, use_container_width=True)

    # Tabla resumen: años clave seleccionados + variación acumulada 2016→2023
    df_resumen = df_evol[df_evol['Año'].isin([2016, 2019, 2021, 2023])]
    df_pivot = df_resumen.pivot(index='municipio', columns='Año', values=ind_evol).round(2)
    if 2016 in df_pivot.columns and 2023 in df_pivot.columns:
        df_pivot['Var. 2016→2023'] = ((df_pivot[2023] - df_pivot[2016]) / df_pivot[2016] * 100).round(1).astype(str) + '%'
    st.dataframe(df_pivot, use_container_width=True)

    csv_evol = df_evol.to_csv(index=False, decimal='.', sep=';').encode('utf-8')
    st.download_button(
        "Descargar serie comparada (.csv)",
        data=csv_evol,
        file_name=f"evolucion_comparada_{col_evol}.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------------------------
# TAB 3 — TABLA DE DATOS
# ---------------------------------------------------------------------------
with tab_datos:
    st.markdown("### Tabla de datos por sección censal")

    c_fil1, c_fil2, c_fil3 = st.columns(3)
    with c_fil1:
        ciudad_tabla = st.selectbox("Ciudad:", ciudades, key="ciudad_tabla")
    with c_fil2:
        anos_tabla = sorted(gdf[gdf['municipio'] == ciudad_tabla]['Fecha_ano'].dropna().unique().astype(int))
        ano_tabla = st.selectbox("Año:", anos_tabla, index=len(anos_tabla)-1, key="ano_tabla")
    with c_fil3:
        st.markdown("&nbsp;", unsafe_allow_html=True)

    cols_tabla = ['Seccion_Censal', 'Em2', '%VUT', 'VUT.Formula', 'Viviendas.Formula']
    for c in ['VUT_km2', 'Tasa_crec_VUT', 'IDS_VUT']:
        if c in gdf.columns:
            cols_tabla.append(c)

    df_tabla = gdf[
        (gdf['municipio'] == ciudad_tabla) & (gdf['Fecha_ano'] == ano_tabla)
    ][[c for c in cols_tabla if c in gdf.columns]].copy()

    # Mostrar %VUT como porcentaje independientemente de cómo esté almacenado en el archivo fuente
    if '%VUT' in df_tabla.columns:
        factor = 100 if df_tabla['%VUT'].max() <= 1.0 else 1
        df_tabla['%VUT_display'] = (df_tabla['%VUT'] * factor).round(2)

    st.dataframe(df_tabla.reset_index(drop=True), use_container_width=True, height=420)
    st.caption(
        f"{len(df_tabla)} secciones censales · {ciudad_tabla} · {ano_tabla} · "
    )

    csv_tabla = df_tabla.to_csv(index=False, decimal='.', sep=';').encode('utf-8')
    st.download_button(
        "Descargar tabla completa (.csv)",
        data=csv_tabla,
        file_name=f"tabla_{ciudad_tabla}_{ano_tabla}.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------------------------
# TAB 4 — FORECASTING (SARIMAX)
# ---------------------------------------------------------------------------

with tab_forecast:

    st.markdown("### ¿A cuánto podría haber llegado el alquiler?")
    st.caption(
        "El modelo aprende la tendencia histórica del precio del alquiler (2016–2023) "
        "y estima cómo podría haber evolucionado en 2024 y 2025. "
    )

    if not FORECASTING_DISPONIBLE:
        st.error("Instala las dependencias: pip install statsmodels")
        st.stop()

    series_df, exog_df, vut_df = load_series_forecasting()
    ciudades_fc = series_df.columns.tolist()

    paleta_ciudades = {
        'Málaga': '#b91c1c', 'Sevilla': '#d97706',
        'Jaén': '#1d4ed8', 'Teruel': '#6b7280'
    }

    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        ciudad_fc = st.selectbox(
            "¿Para qué ciudad?", ciudades_fc, key="ciudad_forecast"
        )
    with col_c2:
        ano_consulta = st.select_slider(
            "¿Qué año te interesa?",
            options=[2024, 2025],
            value=2025
        )

    horizonte = 8  # 8 trimestres para cubrir 2024 y 2025 completos

    serie = series_df[ciudad_fc].dropna()
    exog  = exog_df[ciudad_fc].loc[serie.index].fillna(0)
    ultima_fecha  = serie.index[-1]
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.DateOffset(months=3),
        periods=horizonte, freq='QS'
    )

    with st.spinner("Calculando predicción..."):
        try:
            # Modelo principal: SARIMAX(1,1,1) con %VUT como variable exógena.
            # La proyección exógena usa la media de los últimos 4 trimestres observados
            # como estimación de estado estacionario de la saturación VUT en 2024–2025.
            model = SARIMAX(
                serie,
                exog=exog,
                order=(1, 1, 1),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False, method='lbfgs', maxiter=200)
            exog_fut = np.full((horizonte, 1), exog.iloc[-4:].mean())
            fc = res.get_forecast(steps=horizonte, exog=exog_fut)
            pred_vals = fc.predicted_mean.values
            ci_lower  = fc.conf_int().iloc[:, 0].values
            ci_upper  = fc.conf_int().iloc[:, 1].values
        except Exception:
            # Fallback: SARIMA(1,1,0) sin variable exógena si el modelo principal no converge
            model = SARIMAX(serie, order=(1, 1, 0))
            res   = model.fit(disp=False)
            fc    = res.get_forecast(steps=horizonte)
            pred_vals = fc.predicted_mean.values
            ci_lower  = fc.conf_int().iloc[:, 0].values
            ci_upper  = fc.conf_int().iloc[:, 1].values

    # Métricas resumen para el año de forecast seleccionado
    idx_ano    = [i for i, f in enumerate(fechas_futuras) if f.year == ano_consulta]
    precio_ano = pred_vals[idx_ano].mean() if idx_ano else pred_vals[-1]
    precio_min = ci_lower[idx_ano].mean()  if idx_ano else ci_lower[-1]
    precio_max = ci_upper[idx_ano].mean()  if idx_ano else ci_upper[-1]
    ultimo_real = serie.iloc[-1]
    variacion   = (precio_ano - ultimo_real) / ultimo_real * 100

    st.markdown("---")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric(
        f"Precio estimado en {ano_consulta}",
        f"{precio_ano:.2f} €/m²",
        delta=f"{variacion:+.1f}% respecto a Q4 2023"
    )
    col_r2.metric("Escenario optimista", f"{precio_min:.2f} €/m²")
    col_r3.metric("Escenario pesimista", f"{precio_max:.2f} €/m²")

    st.caption(
        f"Precio real en Q4 2023: **{ultimo_real:.2f} €/m²** · "
        f"El modelo estima que en {ano_consulta} el precio medio en {ciudad_fc} "
        f"podría situarse entre **{precio_min:.2f}** y **{precio_max:.2f} €/m²**."
    )

    # ── Gráfico de forecast ────────────────────────────────────────────────────
    st.markdown("---")
    color = paleta_ciudades.get(ciudad_fc, '#1d4ed8')
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig_fc = go.Figure()

    # Banda del intervalo de confianza al 95% (área rellena)
    fig_fc.add_trace(go.Scatter(
        x=list(fechas_futuras) + list(fechas_futuras[::-1]),
        y=list(ci_upper) + list(ci_lower[::-1]),
        fill='toself',
        fillcolor=f'rgba({r},{g},{b},0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Intervalo de confianza 95%',
        hoverinfo='skip'
    ))

    # Serie histórica
    fig_fc.add_trace(go.Scatter(
        x=serie.index, y=serie.values,
        mode='lines+markers',
        name='Precio real (histórico)',
        line=dict(color=color, width=2.5),
        marker=dict(size=5)
    ))

    # Línea de predicción
    fig_fc.add_trace(go.Scatter(
        x=fechas_futuras, y=pred_vals,
        mode='lines+markers',
        name='Precio estimado (SARIMAX)',
        line=dict(color=color, width=2.5, dash='dash'),
        marker=dict(size=7, symbol='diamond')
    ))

    # Línea vertical que marca el último dato real disponible
    fig_fc.add_vline(
        x=ultima_fecha.timestamp() * 1000,
        line_dash='dot', line_color='#9ca3af',
        annotation_text='Último dato real (Q4 2023)',
        annotation_position='top left',
        annotation_font_color='#6b7280'
    )

    # Sombrear el año de forecast seleccionado con un rectángulo
    if idx_ano:
        fig_fc.add_vrect(
            x0=fechas_futuras[idx_ano[0]].timestamp() * 1000,
            x1=(fechas_futuras[idx_ano[-1]] + pd.DateOffset(months=3)).timestamp() * 1000,
            fillcolor='rgba(250,204,21,0.10)',
            line_width=0,
            annotation_text=str(ano_consulta),
            annotation_position='top right',
            annotation_font_color='#92400e'
        )

    fig_fc.update_layout(
        title=f"Precio del alquiler en {ciudad_fc} — histórico y estimación SARIMAX",
        xaxis_title="",
        yaxis_title="€/m²",
        height=430,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor='white',
        yaxis=dict(gridcolor='#f3f4f6')
    )

    st.plotly_chart(fig_fc, use_container_width=True)

    st.caption(
        "La zona sombreada representa el intervalo de confianza al 95% del modelo SARIMAX (1,1,1). "
        "Cuanto más nos alejamos del último dato real, mayor es la incertidumbre. "
        "Datos base: IATUR 2024 (Universidades de Granada, Málaga y Sevilla)."
    )
