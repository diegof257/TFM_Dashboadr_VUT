import pandas as pd
import geopandas as gpd

print("🚀 Iniciando proceso ETL Espacial...")

# 1. CARGAR Y LIMPIAR DATOS TABULARES
print("1️⃣ Cargando datos.csv...")
df_datos = pd.read_csv("datos.csv", encoding="latin1", sep=";", decimal=",")

# Aseguramos que el código de sección censal sea texto de 10 dígitos
df_datos['SSCC'] = df_datos['SSCC'].astype(str).str.zfill(10)

# Forzamos numéricos por si acaso
columnas_numericas = ['Em2', '%VUT', 'VUT.Formula', 'Viviendas.Formula']
for col in columnas_numericas:
    if col in df_datos.columns and df_datos[col].dtype == object:
        df_datos[col] = df_datos[col].astype(str).str.replace(',', '.').astype(float)

# 2. CARGAR MAPA BASE
print("2️⃣ Cargando mapa base GeoJSON...")
gdf_mapa = gpd.read_file("malaga_procesado.geojson")
gdf_mapa['CUSEC'] = gdf_mapa['CUSEC'].astype(str).str.zfill(10)

# 3. FUSIÓN (MERGE)
print("3️⃣ Fusionando datos espaciales y estadísticos...")
gdf_final = gdf_mapa.merge(df_datos, left_on='CUSEC', right_on='SSCC', how='inner')

# 4. GUARDAR ARCHIVO OPTIMIZADO PARA STREAMLIT
print("4️⃣ Guardando archivo final optimizado...")
# Lo guardamos en formato GeoJSON listo para consumir
gdf_final.to_file("dashboard_data.geojson", driver="GeoJSON")

print("✅ ETL Completado con éxito. Ya puedes lanzar app.py")