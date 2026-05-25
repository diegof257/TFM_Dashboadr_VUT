# Dashboard de Vivienda Turística y Mercado de Alquiler

Dashboard interactivo para el análisis espacial de la relación entre Viviendas de Uso Turístico (VUT) y el precio del alquiler residencial en cuatro ciudades españolas: **Málaga, Sevilla, Jaén y Teruel** (2016–2023).

Desarrollado como parte del Trabajo Fin de Máster en Gestión Estratégica y Dirección de Marketing en la Universidad de Málaga.

---

## Funcionalidades

- **Mapa interactivo** por sección censal con 6 indicadores seleccionables
- **Análisis de clústeres espaciales** mediante el Índice de Moran Bivariado (LISA)
- **Evolución temporal comparada** entre ciudades (2016–2023)
- **Forecasting** del precio del alquiler para 2024–2025 con modelo SARIMAX(1,1,1)
- **Exportación de datos** en CSV para cada vista

## Indicadores disponibles

| Indicador | Descripción |
|---|---|
| Precio Alquiler (€/m²) | Precio medio del alquiler por sección censal |
| Saturación Turística (%VUT) | Proporción de VUT sobre el total de viviendas |
| Densidad VUT (VUTs/km²) | Concentración territorial de VUT |
| Crecimiento anual %VUT | Variación interanual de la saturación turística |
| Difusión Espacial VUT (IDS) | Efecto "mancha de aceite" respecto a secciones vecinas |
| Clústeres Moran (LISA) | Alto-Alto / Bajo-Alto / Bajo-Bajo / Alto-Bajo |

---

## Instalación y arranque

```bash
# 1. Clonar el repositorio
git clone https://github.com/diegof257/TFM_Dashboadr_VUT
cd TFM_Dashboadr_VUT

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Lanzar la aplicación
streamlit run app.py
```

La app abre en `http://localhost:8501`.

**Requisito:** Python 3.11 o superior.

---

## Estructura del proyecto

```
├── app.py                      # Aplicación principal
├── requirements.txt            # Dependencias
├── datos/
│   ├── modeloRegresionLineal.csv   # Datos VUT y precios por sección censal
│   └── INE_C2021_Indicadores.xlsx  # Indicadores del Censo 2021
├── censo_ciudades/             # Geometrías vectoriales (.gpkg) por ciudad
└── correlacion_espacial/       # Pesos espaciales (.gal) para análisis de Moran
```

---

## Tecnologías

- [Streamlit](https://streamlit.io) — interfaz web
- [GeoPandas](https://geopandas.org) + [PySAL](https://pysal.org) — análisis espacial
- [Plotly](https://plotly.com) — visualización interactiva
- [statsmodels](https://www.statsmodels.org) — modelo SARIMAX

---

## Fuente de datos

Datos facilitados por el equipo investigador del **IATUR** (Instituto Andaluz de Investigación e Innovación en Turismo) — Universidades de Granada, Málaga y Sevilla (2024).
