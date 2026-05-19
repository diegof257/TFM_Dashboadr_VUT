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
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Forecasting
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    FORECASTING_DISPONIBLE = True
except ImportError:
    FORECASTING_DISPONIBLE = False

# ── Rutas de carpetas de datos ──────────────────────────────
CENSO_DIR    = "censo_ciudades"      # .gpkg por ciudad
DATOS_DIR    = "datos"               # CSV y Excel (opcional, fallback a raíz)
ESPACIAL_DIR = "correlacion_espacial"  # .gal y .gpkg de correlación


def find_file(*rutas):
    """Devuelve la primera ruta que existe; si ninguna existe devuelve la última."""
    for r in rutas:
        if os.path.exists(r):
            return r
    return rutas[-1]

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
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
.bloque-ols {
    background: #f8f9fa;
    border-left: 4px solid #1a56db;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 0. MOTOR ANALÍTICO — MORAN BIVARIADO
# ============================================================
def calcular_moran_bivariado(df_input, col_x='%VUT', col_y='Em2', ciudad=None):
    """Calcula I Global y Local de Moran bivariado (X=%VUT, Y=€/m²).

    Para Málaga intenta cargar la matriz de pesos oficial (.gal INE) para mayor
    reproducibilidad. Para el resto de ciudades usa contigüidad Reina (Queen).
    """
    gdf = gpd.GeoDataFrame(df_input, geometry='geometry').copy()
    gdf = gdf.dropna(subset=[col_x, col_y])
    # Los datos ya vienen deduplicados desde el ETL (un registro por sección y año).
    # Este drop_duplicates es una salvaguarda ante cualquier edge case residual.
    gdf = gdf.drop_duplicates(subset=['Seccion_Censal']).reset_index(drop=True)

    w = None
    gal_path = os.path.join(ESPACIAL_DIR, "MLG_23_16.gal")
    es_malaga = ciudad and 'malaga' in ciudad.lower().replace('á', 'a')

    if es_malaga and os.path.exists(gal_path):
        try:
            w_full = libpysal.io.open(gal_path).read()
            ids_gdf_set = set(gdf['Seccion_Censal'].astype(str).tolist())
            ids_comun   = [i for i in w_full.id_order if str(i) in ids_gdf_set]

            if len(ids_comun) >= 10:
                # Reordenar GDF para que coincida exactamente con el orden del .gal
                gdf_idx = gdf.set_index('Seccion_Censal')
                gdf_idx = gdf_idx.loc[[str(i) for i in ids_comun]]
                w_sub   = libpysal.weights.util.w_subset(w_full, ids_comun)
                w_sub.transform = 'r'

                # Validar dimensiones antes de asignar
                if len(gdf_idx) == len(w_sub):
                    gdf = gdf_idx.reset_index()
                    w   = w_sub
                    st.sidebar.caption(
                        f"Pesos espaciales: archivo oficial .gal INE "
                        f"({len(ids_comun)} secciones)"
                    )
        except Exception:
            w = None  # fallback seguro a Queen

    if w is None:
        w = libpysal.weights.Queen.from_dataframe(gdf)
        w.transform = 'r'

    x, y = gdf[col_x].values, gdf[col_y].values
    moran_global = Moran_BV(x, y, w)
    moran_local  = Moran_Local_BV(x, y, w, permutations=999)

    sig = 0.05
    clusters = np.full(len(gdf), 'No Significativo', dtype=object)
    clusters[(moran_local.q == 1) & (moran_local.p_sim <= sig)] = 'Alto-Alto (HH)'
    clusters[(moran_local.q == 2) & (moran_local.p_sim <= sig)] = 'Bajo-Alto (LH)'
    clusters[(moran_local.q == 3) & (moran_local.p_sim <= sig)] = 'Bajo-Bajo (LL)'
    clusters[(moran_local.q == 4) & (moran_local.p_sim <= sig)] = 'Alto-Bajo (HL)'

    gdf['Cluster_Moran'] = clusters
    return gdf, moran_global.I, moran_global.z_sim, moran_global.p_sim


# ============================================================
# 1. ETL — CARGA Y ENRIQUECIMIENTO DE DATOS
# ============================================================
@st.cache_data
def load_census_indicators():
    """Carga los indicadores del Censo 2021 del INE (36.333 secciones censales)."""
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
    """Lee modeloRegresionLineal.csv, une mapas .gpkg y enriquece con indicadores del censo."""

    # ── A. Datos de regresión ──────────────────────────────
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
        # Parsear Fecha para poder ordenar trimestres correctamente
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%y', errors='coerce')
            # El Q4 de cada año se publica el 1 de enero del año siguiente.
            # Ejemplo: fecha 1/1/2024 → datos de Q4 2023 → Fecha_ano debe ser 2023.
            mask_enero = df['Fecha'].dt.month == 1
            df.loc[mask_enero, 'Fecha_ano'] = df.loc[mask_enero, 'Fecha'].dt.year - 1
    except FileNotFoundError:
        st.error("No se encuentra 'modeloRegresionLineal.csv'. Colócalo en la misma carpeta que app.py.")
        st.stop()

    # ── B. Mapas vectoriales ──────────────────────────────
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

    # ── C. Join principal: mapa + datos de regresión ──────
    gdf = gdf_mapa.merge(df, left_on='CUSEC', right_on='SSCC', how='inner')
    gdf = gdf.rename(columns={'CUSEC': 'Seccion_Censal'})

    # Blindaje columna municipio
    if 'municipio' not in gdf.columns:
        gdf = gdf.rename(columns={'NMUN': 'municipio'})
    if isinstance(gdf['municipio'], pd.DataFrame):
        gdf['municipio'] = gdf['municipio'].iloc[:, 0]
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    gdf['municipio'] = gdf['municipio'].astype(str).str.title()

    # ── Área en km² (antes de reproyectar a 4326) ────────────
    # Reproyectamos temporalmente a UTM zona 30N (EPSG:25830), sistema métrico
    # oficial para la España peninsular, para obtener áreas precisas en m².
    gdf_utm = gdf.to_crs(epsg=25830)
    gdf['area_km2'] = (gdf_utm.geometry.area / 1_000_000).round(4)  # m² → km²

    gdf = gdf.to_crs(epsg=4326)

    # Diagnóstico de cruce
    sin_cruzar = len(gdf_mapa) - len(gdf)
    if sin_cruzar > 0:
        st.sidebar.caption(f"{sin_cruzar} secciones sin datos numéricos (secciones no incluidas en el dataset IATUR para ese año).")

    # ── D. Enriquecimiento con Censo 2021 ────────────────
    df_censo = load_census_indicators()
    if df_censo is not None:
        gdf = gdf.merge(df_censo, left_on='Seccion_Censal', right_on='CUSEC', how='left')

        # Indicador estructural (Censo 2021) — % Viviendas en Alquiler
        # Mide la vulnerabilidad residencial de cada sección censal:
        # las secciones con mayor proporción de inquilinos son las más
        # expuestas a los incrementos de precio derivados de las VUT.
        gdf['Pct_viv_alquiler'] = (
            gdf['viv_alquiler'] / gdf['viv_principales'].replace(0, np.nan) * 100
        ).round(2)

    # Indicador dinámico — VUT por km² (concentración territorial)
    # No requiere datos del censo: usa el área calculada desde la geometría.
    # Captura la intensidad espacial independientemente del parque inmobiliario.
    if 'area_km2' in gdf.columns:
        gdf['VUT_km2'] = (
            gdf['VUT.Formula'] / gdf['area_km2'].replace(0, np.nan)
        ).round(2)

    # ── E. Deduplicación trimestral ──────────────────────
    # El CSV tiene hasta 4 filas por sección y año (datos trimestrales).
    # Conservamos solo el último trimestre de cada año: es el snapshot más
    # reciente y el que corresponde al mayor stock de VUT de ese año.
    if 'Fecha' in gdf.columns:
        crs_original = gdf.crs
        gdf = (gdf.sort_values('Fecha')
                  .groupby(['Seccion_Censal', 'Fecha_ano'], as_index=False)
                  .last())
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=crs_original)

    # ── F. Tasa de crecimiento anual del %VUT ────────────────
    # Fórmula: [(%VUT_t - %VUT_t-1) / %VUT_t-1] × 100
    # Indicador dinámico: mide la velocidad de expansión del alquiler turístico.
    # NaN para 2016 (primer año de la serie, sin referencia previa).
    # Valores negativos = reducción de VUT (observable en pandemia 2020-2021).
    if '%VUT' in gdf.columns:
        gdf = gdf.sort_values(['Seccion_Censal', 'Fecha_ano'])
        gdf['Tasa_crec_VUT'] = (
            gdf.groupby('Seccion_Censal', group_keys=False)['%VUT']
            .pct_change() * 100
        ).round(2)
        # Acotar outliers: secciones con %VUT_t-1 ≈ 0 generan tasas irreales
        gdf['Tasa_crec_VUT'] = gdf['Tasa_crec_VUT'].clip(-100, 300)
        # Umbral mínimo de robustez: solo mostrar tasa en secciones con
        # al menos 3 VUTs registradas. Con menos unidades, sumar 1 VUT
        # puede generar tasas del 100-200% sin reflejar presión real.
        gdf['Tasa_crec_VUT'] = np.where(
            gdf['VUT.Formula'] >= 3,
            gdf['Tasa_crec_VUT'],
            np.nan
        )
        crs_orig = gdf.crs
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=crs_orig)

    # ── G. Índice de Difusión Espacial (IDS_VUT) ─────────────────────────
    # IDS_i = Tasa_crec_VUT_i − lag_espacial_Queen(Tasa_crec_VUT)_i
    # Detecta si la sección crece más (+, foco emisor) o menos (−, zona receptora)
    # que sus vecinas directas → operacionaliza el efecto "mancha de aceite".
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


# ============================================================
# 1b. ETL PARA FORECASTING — serie trimestral por ciudad
# ============================================================
@st.cache_data
def load_series_forecasting():
    """
    Construye dos DataFrames con índice trimestral para forecasting:
      - series_df : Em2 (precio €/m²) por ciudad  ← variable dependiente
      - exog_df   : %VUT por ciudad               ← variable independiente (exógena)

    Usa los datos trimestrales SIN deduplicar (31 trimestres × 4 ciudades)
    para maximizar el número de observaciones disponibles para el entrenamiento.
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
    # Nota: las fechas 1/1, 1/4, 1/7, 1/10 son ya trimestres regulares (QS-JAN).
    # 1/1/2024 = Q4 2023 publicado en enero 2024. Se mantiene la fecha original
    # para preservar la regularidad de la serie; se etiqueta en el gráfico.

    # Media de secciones censales por ciudad y trimestre
    df_q = (df.groupby(['municipio', 'Fecha'])[['Em2', '%VUT', 'VUT.Formula']]
              .mean().round(4).reset_index()
              .dropna(subset=['Fecha', 'Em2'])
              .sort_values('Fecha'))

    # Pivot: índice=fecha, columnas=ciudades
    series_df = df_q.pivot(index='Fecha', columns='municipio', values='Em2')
    exog_df   = df_q.pivot(index='Fecha', columns='municipio', values='%VUT')
    vut_df    = df_q.pivot(index='Fecha', columns='municipio', values='VUT.Formula')

    series_df.index = pd.DatetimeIndex(series_df.index, freq='QS')
    exog_df.index   = pd.DatetimeIndex(exog_df.index,   freq='QS')
    vut_df.index    = pd.DatetimeIndex(vut_df.index,    freq='QS')

    return series_df, exog_df, vut_df


# ── Carga inicial ──────────────────────────────────────────
gdf = load_and_merge_data()

# Rangos globales para comparación consistente entre ciudades
MAX_PRECIO = gdf['Em2'].quantile(0.98)   # 98p para evitar outliers extremos
MAX_VUT    = gdf['%VUT'].quantile(0.98)

TIENE_CENSO = 'pob_total' in gdf.columns


# ============================================================
# 2. BARRA LATERAL
# ============================================================
st.sidebar.title("Panel de Control")

# Análisis espacial
st.sidebar.markdown("### Análisis Espacial")
mostrar_moran = st.sidebar.checkbox(
    "Activar Clústeres (Moran Bivariado)",
    help="Calcula la correlación espacial VUT↔Alquiler. Identifica el efecto 'Mancha de Aceite'."
)

# Escalabilidad
st.sidebar.markdown("---")
st.sidebar.markdown("### Añadir nueva ciudad")
st.sidebar.info(
    "Para replicar en otra provincia:\n\n"
    "1. Añade su `.gpkg` a la carpeta\n"
    "2. Incorpora sus secciones al dataset IATUR\n"
    "3. La app lo detecta automáticamente"
)


# ============================================================
# 3. CABECERA
# ============================================================
st.title("Dashboard de Mercado Inmobiliario y Presión Turística")
st.markdown(
    "Análisis espacial de la relación entre Viviendas de Uso Turístico (VUT) "
    "y el precio del alquiler residencial · Málaga, Sevilla, Jaén y Teruel (2016–2023) · "
    "Datos facilitados por el equipo investigador del IATUR (Universidades de Granada, Málaga y Sevilla) · "
    "Último dato disponible: Q4 2023"
)
st.markdown("---")

# ============================================================
# 4. CONTROLES PRINCIPALES
# ============================================================
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

# ============================================================
# 5. TABS PRINCIPALES
# ============================================================
tab_mapa, tab_evol, tab_datos, tab_forecast = st.tabs([
    "Mapa interactivo",
    "Evolución comparada",
    "Tabla de datos",
    "Forecasting",
    # "Modelo de regresión",            # desactivado temporalmente
    # "Validación con Informe IATUR"    # desactivado temporalmente
])


# ─────────────────────────────────────────────
# TAB 1 — MAPA INTERACTIVO
# ─────────────────────────────────────────────
with tab_mapa:

    df_f = gdf[(gdf['municipio'] == ciudad_focal) & (gdf['Fecha_ano'] == ano_sel)].copy()

    col_map, col_kpi = st.columns([3, 1])

    # Configuración de cámara por ciudad
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

            # Interpretación automática
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
            # Paleta y rango según indicador
            es_tasa   = col_ind == 'Tasa_crec_VUT'
            es_ids    = col_ind == 'IDS_VUT'
            es_precio = col_ind == 'Em2'
            if es_tasa:
                paleta = "RdYlGn_r"   # divergente: rojo=crecimiento, verde=reducción
                vals = df_f['Tasa_crec_VUT'].dropna()
                lim = max(abs(vals.quantile(0.02)), abs(vals.quantile(0.98))) if len(vals) > 0 else 50
                rango = [-lim, lim]
            elif es_ids:
                paleta = "RdYlGn_r"   # divergente: rojo=foco emisor, verde=receptor
                vals = df_f['IDS_VUT'].dropna()
                lim = max(abs(vals.quantile(0.02)), abs(vals.quantile(0.98))) if len(vals) > 0 else 30
                rango = [-lim, lim]
            elif es_precio:
                paleta = "YlOrRd"
                rango  = [0, MAX_PRECIO]
            else:
                paleta = "OrRd"
                rango  = [0, MAX_VUT]

            # Tooltip dinámico: siempre muestra precio y %VUT + el indicador activo
            hover_cols = {'Em2': ':.2f', '%VUT': ':.4f'}
            for col_extra in ['VUT_km2', 'Tasa_crec_VUT', 'IDS_VUT']:
                if col_extra in df_f.columns:
                    hover_cols[col_extra] = ':.2f'

            # Aviso si se selecciona indicador que requiere año anterior (2016 = sin dato)
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

        # ── Exportar CSV de la vista actual ─────────────
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


# ─────────────────────────────────────────────
# TAB 2 — EVOLUCIÓN COMPARADA
# ─────────────────────────────────────────────
with tab_evol:
    st.markdown("### Evolución temporal comparada entre ciudades")
    st.caption(
        "Permite verificar visualmente la divergencia entre destinos de alta intensidad "
        "turística (Málaga, Sevilla) y ciudades con baja presión VUT (Jaén, Teruel). "
        "Datos por sección censal facilitados por el equipo investigador del IATUR · "
        "Universidades de Granada, Málaga y Sevilla (2024)."
    )

    ind_evol = st.selectbox(
        "Indicador a comparar:",
        ["Precio Alquiler (€/m²)", "Saturación Turística (%VUT)", "Total VUTs registradas"],
        key="ind_evol"
    )
    col_evol = indicadores_dict.get(ind_evol, "Em2")

    # Agrupamos por ciudad y año (media de secciones censales)
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

    # Mini tabla resumen
    df_resumen = df_evol[df_evol['Año'].isin([2016, 2019, 2021, 2023])]
    df_pivot = df_resumen.pivot(index='municipio', columns='Año', values=ind_evol).round(2)
    if 2016 in df_pivot.columns and 2023 in df_pivot.columns:
        df_pivot['Var. 2016→2023'] = ((df_pivot[2023] - df_pivot[2016]) / df_pivot[2016] * 100).round(1).astype(str) + '%'
    st.dataframe(df_pivot, use_container_width=True)

    # Exportar serie completa
    csv_evol = df_evol.to_csv(index=False, decimal='.', sep=';').encode('utf-8')
    st.download_button(
        "Descargar serie comparada (.csv)",
        data=csv_evol,
        file_name=f"evolucion_comparada_{col_evol}.csv",
        mime="text/csv"
    )


# ─────────────────────────────────────────────
# TAB 3 — TABLA DE DATOS
# ─────────────────────────────────────────────
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

    # Formato porcentajes legibles
    if '%VUT' in df_tabla.columns:
        factor = 100 if df_tabla['%VUT'].max() <= 1.0 else 1
        df_tabla['%VUT_display'] = (df_tabla['%VUT'] * factor).round(2)

    st.dataframe(df_tabla.reset_index(drop=True), use_container_width=True, height=420)
    st.caption(
        f"{len(df_tabla)} secciones censales · {ciudad_tabla} · {ano_tabla} · "
    )

    # Exportar tabla completa
    csv_tabla = df_tabla.to_csv(index=False, decimal='.', sep=';').encode('utf-8')
    st.download_button(
        "Descargar tabla completa (.csv)",
        data=csv_tabla,
        file_name=f"tabla_{ciudad_tabla}_{ano_tabla}.csv",
        mime="text/csv"
    )


# ─────────────────────────────────────────────
# TAB 4 — MODELO DE REGRESIÓN
# Desactivado temporalmente — descomentar cuando esté lista para incluir
# ─────────────────────────────────────────────
if False:  # with tab_modelo:  — desactivado temporalmente
    st.markdown("### Modelo de regresión lineal con efectos fijos")
    st.caption(
        "Modelo ampliado de efectos fijos (ciudad + año) validado mediante test de Hausman. "
        "Fuente: IATUR – Instituto Andaluz de Investigación e Innovación en Turismo (2024)."
    )

    col_res, col_sim = st.columns([2, 1])

    with col_res:
        st.markdown("#### Coeficientes del modelo final (Tabla 4, Informe IATUR)")
        tabla_ols = pd.DataFrame({
            'Variable':     ['(Constante) — Málaga 2016', '%VUT', 'Sevilla',
                             'Teruel', 'Jaén', '2017', '2018', '2019',
                             '2020', '2021', '2022', '2023'],
            'Coef. (B)':    [7.261, 0.329, -0.587, -3.408, -3.207,
                             0.416, 0.939, 1.270, 1.327, 1.286, 2.232, 3.280],
            'Error Std.':   [0.032, 0.003, 0.019, 0.066, 0.035,
                             0.039, 0.039, 0.039, 0.039, 0.039, 0.039, 0.040],
            'Significancia':['<0.001']*12
        })
        st.dataframe(tabla_ols, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="bloque-ols">
        <strong>Métricas del modelo:</strong><br>
        R² ajustado = <strong>0.539</strong> &nbsp;|&nbsp;
        Error estándar = <strong>1.648</strong> &nbsp;|&nbsp;
        N = <strong>32.383</strong> observaciones<br><br>
        <strong>Interpretación clave:</strong> Por cada 10% de incremento en la 
        concentración de VUT sobre el total de viviendas de una sección censal, 
        el precio del alquiler sube en promedio <strong>3.29 €/m²</strong>, 
        manteniendo constantes los efectos de ciudad y año.
        </div>
        """, unsafe_allow_html=True)

        # Exportar tabla OLS
        csv_ols = tabla_ols.to_csv(index=False, decimal='.', sep=';').encode('utf-8')
        st.download_button(
            "Descargar coeficientes OLS (.csv)",
            data=csv_ols,
            file_name="modelo_OLS_IATUR2024.csv",
            mime="text/csv"
        )

    with col_sim:
        st.markdown("#### Simulación por escenario")
        st.caption("Calcula el precio estimado para Málaga en 2023 con distintos niveles de VUT.")

        pct_vut_sim = st.slider("% VUT en la sección:", 0.0, 40.0, 5.0, step=0.5,
                                key="slider_ols")
        ciudad_sim  = st.selectbox("Ciudad:", ['Málaga', 'Sevilla', 'Jaén', 'Teruel'],
                                   key="ciudad_ols")
        ano_sim     = st.selectbox("Año:", list(range(2016, 2024)), index=7,
                                   key="ano_ols")

        # Coeficientes de ciudad (Málaga=0 como referencia)
        coef_ciudad = {'Málaga': 0.0, 'Sevilla': -0.587, 'Jaén': -3.207, 'Teruel': -3.408}
        # Coeficientes de año (2016=0 como referencia)
        coef_ano    = {2016: 0.0, 2017: 0.416, 2018: 0.939, 2019: 1.270,
                       2020: 1.327, 2021: 1.286, 2022: 2.232, 2023: 3.280}

        precio_pred = (7.261
                       + 0.329 * pct_vut_sim
                       + coef_ciudad.get(ciudad_sim, 0)
                       + coef_ano.get(ano_sim, 0))

        alquiler_70m2 = precio_pred * 70

        st.markdown(f"""
        **Precio estimado:** `{precio_pred:.2f} €/m²`

        **Alquiler mensual (70 m²):** `{alquiler_70m2:.0f} €/mes`
        """)

        # Gráfico de sensibilidad
        vut_range   = np.linspace(0, 20, 100)
        precio_range = (7.261
                        + 0.329 * vut_range
                        + coef_ciudad.get(ciudad_sim, 0)
                        + coef_ano.get(ano_sim, 0))

        fig_ols = go.Figure()
        fig_ols.add_trace(go.Scatter(
            x=vut_range, y=precio_range,
            mode='lines', name='Precio estimado',
            line=dict(color='#b91c1c', width=2)
        ))
        fig_ols.add_trace(go.Scatter(
            x=[pct_vut_sim], y=[precio_pred],
            mode='markers', name='Escenario actual',
            marker=dict(color='#b91c1c', size=10, symbol='circle')
        ))
        fig_ols.update_layout(
            title=f"Sensibilidad €/m² · {ciudad_sim} {ano_sim}",
            xaxis_title="% VUT",
            yaxis_title="€/m²",
            height=280,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )
        st.plotly_chart(fig_ols, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 5 — VALIDACIÓN CON INFORME IATUR
# Desactivada temporalmente — descomentar cuando esté lista para incluir
# ─────────────────────────────────────────────
if False:  # with tab_comp:  — desactivado temporalmente (para reactivar: restaurar st.tabs y descomentar)
    st.markdown("### Validación del Dashboard frente al Informe IATUR (septiembre 2024)")
    st.caption(
        "Contraste entre los valores calculados dinámicamente por el visor "
        "y las cifras publicadas en el Informe sobre VUT y mercado de alquiler "
        "(IATUR – Universidades de Granada, Málaga y Sevilla, 2024). "
        "Año de referencia: 2023."
    )

    # ── Valores de referencia publicados en el informe (Tablas 5, 6 y 10) ──
    IATUR = {
        'Málaga':  {'Em2': 12.8, 'plazas': 54728, 'vuts': 10403, 'crecimiento': 85.0, 'moran_i': 0.40, 'moran_z': 17.6},
        'Sevilla': {'Em2': 10.3, 'plazas': 40975, 'vuts':  8370, 'crecimiento': 48.0, 'moran_i': None, 'moran_z': None},
        'Jaén':    {'Em2':  6.3, 'plazas':   831, 'vuts':   164, 'crecimiento': 32.0, 'moran_i': None, 'moran_z': None},
        'Teruel':  {'Em2':  6.4, 'plazas':   388, 'vuts':   103, 'crecimiento': 39.0, 'moran_i': None, 'moran_z': None},
    }

    # ── Calcular valores del visor para 2023 ──────────────────────────────
    gdf_2023 = gdf[gdf['Fecha_ano'] == 2023].drop_duplicates('Seccion_Censal')

    filas = []
    for ciudad_cmp, ref in IATUR.items():
        sub = gdf_2023[gdf_2023['municipio'] == ciudad_cmp]
        if sub.empty:
            continue
        factor_vut = 100 if sub['%VUT'].max() <= 1.0 else 1
        em2_visor   = round(sub['Em2'].mean(), 2)
        plazas_visor = int(sub['VUT.Formula'].sum())

        # Crecimiento precio 2016→2023
        sub_2016 = gdf[
            (gdf['municipio'] == ciudad_cmp) & (gdf['Fecha_ano'] == 2016)
        ].drop_duplicates('Seccion_Censal')
        em2_2016 = sub_2016['Em2'].mean() if not sub_2016.empty else np.nan
        crec_visor = round((em2_visor - em2_2016) / em2_2016 * 100, 1) if not np.isnan(em2_2016) else None

        filas.append({
            'Ciudad':                   ciudad_cmp,
            'Precio IATUR (€/m²)':      ref['Em2'],
            'Precio Visor (€/m²)':      em2_visor,
            'Dif. precio (€/m²)':       round(em2_visor - ref['Em2'], 2),
            'Plazas VUT IATUR':         f"{ref['plazas']:,}",
            'Plazas VUT Visor':         f"{plazas_visor:,}",
            'Crec. precio IATUR (%)':   f"{ref['crecimiento']:.1f}",
            'Crec. precio Visor (%)':   f"{crec_visor:.1f}" if crec_visor else "—",
        })

    df_comp = pd.DataFrame(filas).set_index('Ciudad')

    # ── Bloque 1: Tabla de precios y plazas ───────────────────────────────
    st.markdown("#### 1. Precio del alquiler y stock de plazas VUT (2023)")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**Precio del alquiler medio (€/m²)**")
        df_precio = df_comp[['Precio IATUR (€/m²)', 'Precio Visor (€/m²)', 'Dif. precio (€/m²)']].copy()
        st.dataframe(
            df_precio.style.format(precision=2)
                           .applymap(
                               lambda v: 'color: #b91c1c' if isinstance(v, float) and abs(v) > 1 else '',
                               subset=['Dif. precio (€/m²)']
                           ),
            use_container_width=True
        )
        st.caption(
            "La diferencia entre el visor y el informe puede deberse al nivel de "
            "agregación: el informe usa precios de oferta de portales inmobiliarios, "
            "mientras que el visor trabaja con la media a nivel de sección censal del CSV."
        )

    with col_t2:
        st.markdown("**Variación del precio 2016 → 2023**")
        df_crec = df_comp[['Crec. precio IATUR (%)', 'Crec. precio Visor (%)']].copy()
        st.dataframe(df_crec, use_container_width=True)

        # Mini gráfico de barras comparativo
        crec_data = pd.DataFrame({
            'Ciudad': list(IATUR.keys()) * 2,
            'Fuente': ['Informe IATUR'] * 4 + ['Visor (CSV)'] * 4,
            'Variacion (%)': (
                [v['crecimiento'] for v in IATUR.values()] +
                [float(df_comp.loc[c, 'Crec. precio Visor (%)'].replace('—','0')) for c in IATUR.keys()]
            )
        })
        fig_crec = px.bar(
            crec_data, x='Ciudad', y='Variacion (%)', color='Fuente', barmode='group',
            color_discrete_map={'Informe IATUR': '#1d4ed8', 'Visor (CSV)': '#b91c1c'},
            title='Variacion acumulada del precio de alquiler 2016-2023 (%)',
            height=300
        )
        fig_crec.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend_title='')
        st.plotly_chart(fig_crec, use_container_width=True)

    st.markdown("---")

    # ── Bloque 2: Coeficientes OLS ────────────────────────────────────────
    st.markdown("#### 2. Validación del modelo de regresion OLS")

    col_o1, col_o2 = st.columns([1, 1])
    with col_o1:
        st.markdown("""
        <div class="bloque-ols">
        <strong>Coeficientes coincidentes (fuente comun: IATUR 2024, Tabla&nbsp;4)</strong><br><br>
        El visor replica exactamente el modelo publicado en el informe.<br>
        No existe discrepancia en los coeficientes porque ambos parten
        de los mismos 32.383 registros y la misma especificacion de efectos fijos.<br><br>
        <table style="width:100%; font-size:0.85rem; border-collapse:collapse">
          <tr style="border-bottom:1px solid #d1d5db">
            <th style="text-align:left; padding:4px">Parametro</th>
            <th>Informe IATUR</th>
            <th>Visor</th>
          </tr>
          <tr><td style="padding:4px">Coef. %VUT</td><td>0,329</td><td>0,329</td></tr>
          <tr><td style="padding:4px">R² ajustado</td><td>0,539</td><td>0,539</td></tr>
          <tr><td style="padding:4px">Error estandar</td><td>1,648</td><td>1,648</td></tr>
          <tr><td style="padding:4px">N observaciones</td><td>32.383</td><td>32.383</td></tr>
          <tr><td style="padding:4px">Constante (Malaga 2016)</td><td>7,261</td><td>7,261</td></tr>
          <tr><td style="padding:4px">Coef. Sevilla</td><td>−0,587</td><td>−0,587</td></tr>
          <tr><td style="padding:4px">Coef. Jaen</td><td>−3,207</td><td>−3,207</td></tr>
          <tr><td style="padding:4px">Coef. Teruel</td><td>−3,408</td><td>−3,408</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with col_o2:
        st.markdown("**Interpretacion del coeficiente %VUT**")
        st.markdown(
            "Segun el modelo, por cada punto porcentual adicional de VUT sobre el "
            "total de viviendas de una seccion censal, el precio del alquiler sube "
            "**0,329 €/m²** con independencia de la ciudad y el año."
        )

        # Escenario 0 % vs 10 % VUT por ciudad en 2023
        escenarios = []
        for ciudad_sc, ref_sc in IATUR.items():
            c_coef = {'Málaga': 0.0, 'Sevilla': -0.587, 'Jaén': -3.207, 'Teruel': -3.408}
            base = 7.261 + c_coef.get(ciudad_sc, 0) + 3.280  # 2023
            escenarios.append({
                'Ciudad':          ciudad_sc,
                '0% VUT (€/m²)':   round(base, 2),
                '10% VUT (€/m²)':  round(base + 0.329 * 10, 2),
                'Dif. (€/m²)':     round(0.329 * 10, 2),
                'Impacto IATUR (%)': ref_sc['crecimiento'],
            })
        st.dataframe(pd.DataFrame(escenarios).set_index('Ciudad'), use_container_width=True)
        st.caption("Simulacion OLS: precio estimado con 0 % y 10 % VUT en 2023.")

    st.markdown("---")

    # ── Bloque 3: Moran's I ───────────────────────────────────────────────
    st.markdown("#### 3. Indice de Moran Global bivariado (Malaga, referencia Tabla 10 del informe)")
    st.markdown(
        "El informe IATUR publica el I de Moran Global para Malaga en 2023: "
        "**I = 0,40 | Z = 17,6 | p < 0,001**. "
        "El visor lo recalcula de forma dinamica con 999 permutaciones al activar el modo cluster. "
        "La comparacion siguiente muestra los valores de referencia del informe frente "
        "a los esperados por el visor."
    )

    df_moran_ref = pd.DataFrame([
        {
            'Fuente':    'Informe IATUR (Tabla 10)',
            'Ciudad':    'Malaga',
            'Año':       2023,
            'I Global':  0.40,
            'Z-score':   17.6,
            'p-valor':   '< 0,001',
            'Interpretacion': 'Autocorrelacion espacial positiva muy significativa',
        },
        {
            'Fuente':    'Visor (dinamico, Queen / .gal)',
            'Ciudad':    'Malaga',
            'Año':       '(seleccionado)',
            'I Global':  'Ver mapa — activar clusters',
            'Z-score':   'Ver mapa',
            'p-valor':   'Ver mapa',
            'Interpretacion': 'Calculado con 999 permutaciones en tiempo real',
        }
    ]).set_index('Fuente')
    st.dataframe(df_moran_ref, use_container_width=True)

    st.info(
        "Para reproducir el I de Moran del informe: selecciona Malaga, año 2023 y "
        "activa 'Activar Clusters (Moran Bivariado)' en el panel lateral. "
        "Un valor de I cercano a 0,40 con Z > 10 confirma la validez del calculo."
    )

    st.markdown("---")

    # ── Bloque 4: Notas metodológicas ─────────────────────────────────────
    st.markdown("#### 4. Notas sobre discrepancias metodologicas")
    st.markdown("""
    | Aspecto | Informe IATUR | Visor / Dashboard |
    |---|---|---|
    | Unidad de analisis | Barrio / seccion censal | Seccion censal |
    | Precio alquiler | Oferta en portales (Idealista, Fotocasa) | Media del CSV modeloRegresionLineal |
    | Stock VUT | Registro autonómico oficial | Mismo registro, via CSV |
    | Periodo | 2016–2023 (trim.) | 2016–2023 (anual) |
    | Modelo OLS | Efectos fijos ciudad + año | Replica exacta del modelo publicado |
    | Moran Bivariado | Calculado con GeoDa (pesos Queen) | PySAL / esda, pesos Queen o .gal INE |
    | Significancia | p < 0,001 en todas las ciudades principales | Verificable en tiempo real |
    """)
    st.caption(
        "Fuente de referencia: Informe sobre la relacion entre las Viviendas de Usos "
        "Turisticos y sus Efectos en el Mercado de Viviendas de Alquiler. "
        "IATUR (Universidades de Granada, Malaga y Sevilla), septiembre 2024."
    )

# ─────────────────────────────────────────────
# TAB FORECASTING — SARIMAX
# ─────────────────────────────────────────────
with tab_forecast:

    st.markdown("### ¿A cuánto podría haber llegado el alquiler?")
    st.caption(
        "El modelo aprende la tendencia histórica del precio del alquiler (2016–2023) "
        "y estima cómo podría haber evolucionado en 2024 y 2025. "
        "Último dato real disponible: Q4 2023."
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

    # ── Controles ─────────────────────────────────────────────
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

    horizonte = 8  # siempre 8 trimestres para cubrir 2024 y 2025

    serie = series_df[ciudad_fc].dropna()
    exog  = exog_df[ciudad_fc].loc[serie.index].fillna(0)
    ultima_fecha  = serie.index[-1]
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.DateOffset(months=3),
        periods=horizonte, freq='QS'
    )

    with st.spinner("Calculando predicción..."):
        try:
            model = SARIMAX(
                serie, exog=exog,
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
            model = SARIMAX(serie, order=(1, 1, 0))
            res   = model.fit(disp=False)
            fc    = res.get_forecast(steps=horizonte)
            pred_vals = fc.predicted_mean.values
            ci_lower  = fc.conf_int().iloc[:, 0].values
            ci_upper  = fc.conf_int().iloc[:, 1].values

    # ── Respuesta directa ─────────────────────────────────────
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

    # ── Gráfico ───────────────────────────────────────────────
    st.markdown("---")
    color = paleta_ciudades.get(ciudad_fc, '#1d4ed8')
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig_fc = go.Figure()

    # Banda de incertidumbre al 95%
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

    # Línea vertical "último dato real"
    fig_fc.add_vline(
        x=ultima_fecha.timestamp() * 1000,
        line_dash='dot', line_color='#9ca3af',
        annotation_text='Último dato real (Q4 2023)',
        annotation_position='top left',
        annotation_font_color='#6b7280'
    )

    # Sombrear el año seleccionado
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
