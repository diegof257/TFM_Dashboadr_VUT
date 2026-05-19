# Contexto TFM — Dashboard VUT · Estado Actual

## Archivos clave
- **Documento Word**: `/Users/diego/Downloads/TFM_DIEGO_v4.docx` (output final)
- **Unpacked XML**: `/tmp/tfm_v4/word/document.xml` (edición directa)
- **Original base**: `/Users/diego/Downloads/files/TFM_DIEGO_v3.docx`
- **App dashboard**: `/Users/diego/Documents/UMA/TFM/TFM_Dashboadr_VUT/app.py`
- **Literatura VUT**: `/Users/diego/Downloads/Literatura_VUT_TFM.docx`
- **Guía SARIMAX**: `/Users/diego/Downloads/Guia_SARIMAX_TFM.docx`

## Comandos de trabajo
```bash
# Desempaquetar
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 \
  "/Users/diego/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/8b133caa-fc24-484e-8b25-7876d3c4862e/9e618464-c99c-40a5-beb8-66e66cf0fb99/skills/docx/scripts/office/unpack.py" \
  /Users/diego/Downloads/TFM_DIEGO_v4.docx /tmp/tfm_v4/

# Empaquetar
SKILLS_DIR="/Users/diego/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/8b133caa-fc24-484e-8b25-7876d3c4862e/9e618464-c99c-40a5-beb8-66e66cf0fb99/skills/docx"
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 "$SKILLS_DIR/scripts/office/pack.py" \
  /tmp/tfm_v4/ /Users/diego/Downloads/TFM_DIEGO_v4.docx \
  --original /Users/diego/Downloads/files/TFM_DIEGO_v3.docx

# Node.js para .docx nuevos
NODE_PATH=/Users/diego/.nvm/versions/node/v20.19.0/lib/node_modules node script.js
```

---

## Estudio
**Título**: Impacto de las VUT en el precio del alquiler — Málaga, Sevilla, Jaén y Teruel (2016–2023)
**Modelo principal**: SARIMAX(1,1,1) con %VUT como variable exógena
**Análisis espacial**: Índice de Moran Bivariado (libpysal/esda)
**Dashboard**: Streamlit + GeoPandas + Plotly

---

## Estructura del documento (capítulos relevantes)

### Marco Teórico (Cap. 2) — jerarquía actual
- **2.1** Subcapitulo: "El mercado de vivienda en España: de la burbuja inmobiliaria a la crisis del alquiler (1990–2023)"
  - **2.1.1** Subcapitulo-Hijo: "La respuesta legal y el Marco Normativo: El Decreto 31/2024"
  - **2.1.2** Subcapitulo-Hijo: "El Efecto 'Mancha de Aceite' (Spillover Effect)"
- **2.2** Subcapitulo: "Estado del Arte: Visores y Herramientas de Análisis Territorial"
  - **2.2.1** Subcapitulo-Hijo: "Literatura Académica sobre el Impacto de las VUT en los Precios"
  - **2.2.2** Subcapitulo-Hijo: "Visores Institucionales y de Administración Pública"
  - **2.2.3** Subcapitulo-Hijo: "Medios Periodísticos y Plataformas de Divulgación"
  - **2.2.4** Subcapitulo-Hijo: "El Vacío Metodológico que Justifica este Trabajo"

### Metodología (Cap. 3) — sección de indicadores
- Subcapitulo: "Indicadores de Presión Turística Empleados en el Análisis"
  - Párrafo intro: explica 4 indicadores (3 dinámicos + 1 estructural Censo 2021)
  - Párrafo IND00001: VUT/km²
  - Párrafo IND00002: Tasa crecimiento anual %VUT (con umbral mínimo 3 VUTs)
  - Párrafo IND00003: % Viviendas en alquiler (Censo 2021, vulnerabilidad)
  - **FALTA**: 4º indicador geoespacial (pendiente de decisión)
- Subcapitulo: "Arquitectura del Sistema y Tecnologías Empleadas"

---

## Indicadores del dashboard — estado actual en app.py

### Implementados ✅
| Variable en código | Nombre UI | Tipo | Fuente |
|---|---|---|---|
| `%VUT` | Saturación Turística (%VUT) | Dinámico | Registro VUT |
| `VUT_km2` | Densidad VUT (VUTs/km²) | Dinámico | VUT + shapefile EPSG:25830 |
| `Tasa_crec_VUT` | Crecimiento anual %VUT (%) | Dinámico | Registro VUT |
| `Em2` | Precio Alquiler (€/m²) | Dinámico | IATUR 2024 |
| `VUT.Formula` | Total VUTs registradas | Dinámico | Registro VUT |
| `Pct_viv_alquiler` | Vulnerabilidad: Viv. en Alquiler (%) | Estático | Censo 2021 |

### Implementado en sesión 2026-05-15 ✅
- **IDS_VUT (Índice de Difusión Espacial)**: `IDS_i = Tasa_crec_VUT_i − Lag_Queen(Tasa_crec_VUT)_i`
  - Calculado en `load_and_merge_data()` (bloque G), después de Tasa_crec_VUT
  - Paleta divergente RdYlGn_r (rojo=foco emisor, verde=receptor)
  - Disponible en selector, tooltip, exportación CSV
  - Aviso año base 2016 compartido con Tasa_crec_VUT

### Indicadores eliminados (y por qué)
| Eliminado | Motivo |
|---|---|
| `Presion_1000hab` (VUT/1000 hab) | Redundante con %VUT, denominador Censo estático |
| `Ratio_turistif` (VUT/viv. no principales) | Denominador incorrecto: incluye segundas residencias |

---

## Detalles técnicos del app.py

### Cálculo Tasa_crec_VUT (después de deduplicación)
```python
if '%VUT' in gdf.columns:
    gdf = gdf.sort_values(['Seccion_Censal', 'Fecha_ano'])
    gdf['Tasa_crec_VUT'] = (
        gdf.groupby('Seccion_Censal', group_keys=False)['%VUT']
        .pct_change() * 100
    ).round(2)
    gdf['Tasa_crec_VUT'] = gdf['Tasa_crec_VUT'].clip(-100, 300)
    # Umbral mínimo: solo secciones con ≥ 3 VUTs
    gdf['Tasa_crec_VUT'] = np.where(
        gdf['VUT.Formula'] >= 3,
        gdf['Tasa_crec_VUT'],
        np.nan
    )
```

### Cálculo VUT_km2
```python
gdf_utm = gdf.to_crs(epsg=25830)
gdf['area_km2'] = (gdf_utm.geometry.area / 1_000_000).round(4)
gdf['VUT_km2'] = (gdf['VUT.Formula'] / gdf['area_km2'].replace(0, np.nan)).round(2)
```

### Escala de color mapa (Tasa_crec_VUT es divergente)
```python
es_tasa = col_ind == 'Tasa_crec_VUT'
if es_tasa:
    paleta = "RdYlGn_r"  # divergente: rojo=crecimiento, verde=reducción
    vals = df_f['Tasa_crec_VUT'].dropna()
    lim = max(abs(vals.quantile(0.02)), abs(vals.quantile(0.98)))
    rango = [-lim, lim]
```

### Datos del Censo 2021
```python
# Archivo: INE_C2021_Indicadores.xlsx en carpeta datos/
# Columnas relevantes:
# t20_2 → viv_alquiler
# t19_1 → viv_principales
# Pct_viv_alquiler = viv_alquiler / viv_principales * 100
```

---

## Comentarios del tutor pendientes en el documento
- [ ] Metodología: *"Si te es posible, te sumará puntos si en el tribunal hay un INFORMÁTICO — referenciar herramientas, librerías o código, incluso un enlace a repo de GitHub"*
- [ ] Posible sección de Discusión
- [ ] Apéndices con Lorem ipsum (especificaciones técnicas)

---

## Estado tras sesión 2026-05-15
**5 indicadores implementados** (4 dinámicos + 1 Censo 2021):
1. VUT/km² (IND00001) ✅
2. Tasa_crec_VUT (IND00002) ✅
3. IDS_VUT — Difusión Espacial (IND00004) ✅ ← NUEVO
4. Pct_viv_alquiler — Censo 2021 (IND00003) ✅
5. (más Em2, %VUT, VUT.Formula ya presentes)

**Documento**: TFM_DIEGO_v4.docx actualizado y reempaquetado en `/Users/diego/Downloads/`.

## Tareas pendientes para próximas sesiones
- [ ] Comentario tutor: referenciar herramientas/librerías, enlace a repo GitHub
- [ ] Posible sección de Discusión
- [ ] Apéndices con especificaciones técnicas (Lorem ipsum → contenido real)
