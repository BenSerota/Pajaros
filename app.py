"""
Panel de Siniestralidad de Avifauna en Fuerteventura — punto de entrada Streamlit.

Proporciona:
  - Configuración de página y diseño global
  - Página de inicio con descripción del proyecto y métricas clave
  - Guía de navegación (Streamlit detecta automáticamente las páginas en pages/)
"""

import streamlit as st
from pathlib import Path
import sys

# Ensure project root is on the path so all modules resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_data

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Siniestralidad Avifauna — Fuerteventura",
    page_icon="\U0001F985",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cargar datos (cacheados en todas las páginas) ───────────────────────────
df = load_data()

# ── Página de inicio ─────────────────────────────────────────────────────────
st.title("Panel de Siniestralidad de Avifauna en Líneas Eléctricas — Fuerteventura")

st.markdown(
    """
    Análisis de la mortalidad de aves causada por colisión y electrocución
    en líneas eléctricas de **Fuerteventura, Islas Canarias** (2018--2025).

    Fuente de datos: muestreos de campo realizados por **BIOSFERA XXI** en
    líneas de transporte de 66 kV y 132 kV a lo largo de tres corredores
    de estudio (Norte, Medio, Sur).
    """
)

# ── Métricas clave ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total registros", f"{len(df):,}")

with col2:
    if "date" in df.columns and df["date"].notna().any():
        date_min = df["date"].min().strftime("%b %Y")
        date_max = df["date"].max().strftime("%b %Y")
        st.metric("Rango de fechas", f"{date_min} -- {date_max}")
    else:
        st.metric("Rango de fechas", "N/D")

with col3:
    if "species_clean" in df.columns:
        n_species = df["species_clean"].nunique()
        st.metric("Especies únicas", f"{n_species}")

with col4:
    if "line_label" in df.columns:
        n_lines = df["line_label"].nunique()
        st.metric("Líneas eléctricas", f"{n_lines}")

st.divider()

# ── Páginas del panel ────────────────────────────────────────────────────────
st.subheader("Páginas del panel")

pages = {
    "Resumen": (
        "Resumen general de cifras de mortalidad, tipos de evento y tendencias. "
        "Indicadores clave y comparaciones interanuales."
    ),
    "Espacial": (
        "Análisis cartográfico de puntos críticos de colisión. Mapas interactivos "
        "con ubicaciones de mortalidad, agrupamiento por densidad y proximidad a carreteras."
    ),
    "Señalización": (
        "Efectividad de los distintos dispositivos de señalización (Triple Aspa, Aspa Corta, "
        "Espirales Amarillas) para reducir las colisiones. Comparaciones antes/después."
    ),
    "Temporal": (
        "Patrones estacionales y mensuales de mortalidad. Detección de picos "
        "migratorios, efectos del ciclo lunar y tendencias a largo plazo."
    ),
    "Especies": (
        "Desglose de víctimas por especie. Análisis por patrón de actividad "
        "(vulnerabilidad nocturna vs. diurna) y agregaciones a nivel de género."
    ),
    "Conservación": (
        "Evaluación del impacto sobre la conservación. Estado en la Lista Roja de la UICN "
        "de las especies afectadas, puntuación de prioridad y análisis de especies focales."
    ),
    "Metodología": (
        "Diseño del estudio, protocolos de recogida de datos, proceso de limpieza, "
        "métodos estadísticos y limitaciones conocidas del conjunto de datos."
    ),
}

for page_name, description in pages.items():
    st.markdown(f"**{page_name}** --- {description}")

st.divider()

# ── Calidad de datos ─────────────────────────────────────────────────────────
with st.expander("Resumen de calidad de datos"):
    total = len(df)

    coord_valid = 0
    if "latitude" in df.columns:
        coord_valid = df["latitude"].notna().sum()

    species_id = 0
    if "species_clean" in df.columns:
        species_id = (df["species_clean"] != "Unknown / No identificada").sum()

    signal_known = 0
    if "signal_type" in df.columns:
        signal_known = df["signal_type"].notna().sum()

    st.markdown(
        f"""
        | Indicador | Cantidad | % del total |
        |-----------|--------:|-----------:|
        | Registros con coordenadas válidas | {coord_valid:,} | {coord_valid/total*100:.0f}% |
        | Registros con especie identificada | {species_id:,} | {species_id/total*100:.0f}% |
        | Registros con información de señalización | {signal_known:,} | {signal_known/total*100:.0f}% |
        """
    )
