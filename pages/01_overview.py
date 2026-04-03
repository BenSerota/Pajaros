"""
Pagina 1: Resumen General
KPIs, graficos resumen y exploracion de datos de alto nivel.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COLORS, CHART_TEMPLATE, SOURCE_ANNOTATION, SIGNAL_TYPES_ORDERED
from src.data_loader import load_data
from src.filters import apply_filters, get_filter_summary

# -- Constantes de diseno -----------------------------------------------------
CHART_MARGINS = dict(l=20, r=20, t=40, b=60)
CHART_H_BAR = 450
CHART_H_SMALL = 350

# -- Configuracion de pagina --------------------------------------------------
st.header("Resumen General")
st.caption("Resumen de alto nivel de la siniestralidad de avifauna en las lineas electricas de Fuerteventura")

# -- Carga y filtrado ---------------------------------------------------------
df_full = load_data()
df = apply_filters(df_full)
subtitle = get_filter_summary(df, df_full)

# -- KPIs ---------------------------------------------------------------------
st.subheader("Indicadores Clave")

col1, col2, col3, col4, col5 = st.columns(5)

# Total victimas
total = len(df)
with col1:
    st.metric("Total Victimas", f"{total:,}")

# Victimas por ano (media)
if "year" in df.columns and df["year"].notna().any():
    year_counts_kpi = df.groupby("year").size()
    # Excluir anos parciales (primero y ultimo) si hay al menos 3 anos
    if len(year_counts_kpi) >= 3:
        full_year_counts = year_counts_kpi.iloc[1:-1]
        avg_annual = full_year_counts.mean()
    else:
        avg_annual = year_counts_kpi.mean()
else:
    avg_annual = 0
with col2:
    st.metric("Victimas por Ano (media)", f"{avg_annual:.1f}")

# Especies afectadas
n_species = df["species_clean"].nunique()
with col3:
    st.metric("Especies Afectadas", n_species)

# Victimas de especies amenazadas
threatened_mask = (
    df["iucn_status"].isin(["Vulnerable", "En peligro de extincion", "Riesgo menor-casi amenazada"]) |
    df["spanish_catalog"].isin(["Vulnerable", "En peligro de extincion"])
)
threatened = threatened_mask.sum()
with col4:
    st.metric("Victimas Especies Amenazadas", f"{threatened:,}")

# Victimas nocturnas
nocturnal = (df["activity_pattern"] == "Nocturna").sum() if "activity_pattern" in df.columns else 0
noct_pct = nocturnal / total * 100 if total > 0 else 0
with col5:
    st.metric("Victimas Especies Nocturnas", f"{nocturnal:,}", f"{noct_pct:.1f}% del total")

st.divider()

# =============================================================================
# Victimas por Linea Electrica (full-width)
# =============================================================================
st.subheader("Victimas por Linea Electrica")
line_counts = (
    df.groupby(["line_label", "voltage"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=True)
)
fig_line = px.bar(
    line_counts,
    x="count",
    y="line_label",
    color="voltage",
    color_discrete_map={66: COLORS["66kV"], 132: COLORS["132kV"]},
    orientation="h",
    labels={"count": "Victimas", "line_label": "", "voltage": "Tension"},
    template=CHART_TEMPLATE,
    height=CHART_H_BAR,
)
fig_line.update_layout(
    legend_title_text="Tension",
    margin=CHART_MARGINS,
    xaxis_title="Numero de victimas",
    title_font_size=16,
    hovermode="x unified",
)
fig_line.add_annotation(
    text=f"{SOURCE_ANNOTATION} | n={total:,}",
    xref="paper", yref="paper", x=0, y=-0.12,
    showarrow=False, font=dict(size=10, color="gray"),
)
st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# =============================================================================
# Victimas por Ano (full-width)
# =============================================================================
st.subheader("Victimas por Ano")
year_counts = df.groupby("year").size().reset_index(name="count")
year_counts["year"] = year_counts["year"].astype(int)

# Marcar anos parciales
min_year = int(df["year"].min()) if len(df) > 0 else 2018
max_year = int(df["year"].max()) if len(df) > 0 else 2025
year_counts["partial"] = year_counts["year"].isin([min_year, max_year])

fig_year = go.Figure()
fig_year.add_trace(go.Bar(
    x=year_counts["year"],
    y=year_counts["count"],
    marker_color=[
        "#BDBDBD" if p else "#D32F2F"
        for p in year_counts["partial"]
    ],
    text=year_counts["count"],
    textposition="outside",
    hovertemplate="Ano: %{x}<br>Victimas: %{y}<extra></extra>",
))

# Linea de tendencia (excluyendo anos parciales)
full_years = year_counts[~year_counts["partial"]]
if len(full_years) >= 3:
    z = np.polyfit(full_years["year"], full_years["count"], 1)
    trend_x = year_counts["year"]
    trend_y = np.polyval(z, trend_x)
    fig_year.add_trace(go.Scatter(
        x=trend_x, y=trend_y,
        mode="lines",
        line=dict(color="rgba(0,0,0,0.3)", dash="dash", width=2),
        name="Tendencia (anos completos)",
        hoverinfo="skip",
    ))

fig_year.update_layout(
    template=CHART_TEMPLATE,
    height=CHART_H_BAR,
    showlegend=False,
    margin=CHART_MARGINS,
    xaxis_title="Ano",
    yaxis_title="Victimas",
    xaxis=dict(dtick=1),
    title_font_size=16,
    hovermode="x unified",
)
fig_year.add_annotation(
    text=f"*Ano parcial | {SOURCE_ANNOTATION} | n={total:,}",
    xref="paper", yref="paper", x=0, y=-0.12,
    showarrow=False, font=dict(size=10, color="gray"),
)
st.plotly_chart(fig_year, use_container_width=True)

st.divider()

# =============================================================================
# Top 15 Especies (full-width)
# =============================================================================
st.subheader("Top 15 Especies")
sp_counts = (
    df.groupby(["species_clean", "activity_pattern"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=True)
    .tail(15)
)
activity_color_map = {
    "Nocturna": COLORS.get("Nocturna", "#1a237e"),
    "Crepuscular": COLORS.get("Crepuscular", "#6a1b9a"),
    "Diurna": COLORS.get("Diurna", "#e65100"),
    "Desconocida": COLORS.get("Desconocida", "#757575"),
}
fig_sp = px.bar(
    sp_counts,
    x="count",
    y="species_clean",
    color="activity_pattern",
    color_discrete_map=activity_color_map,
    orientation="h",
    labels={"count": "Victimas", "species_clean": "", "activity_pattern": "Actividad"},
    template=CHART_TEMPLATE,
    height=CHART_H_BAR,
)
fig_sp.update_layout(
    margin=CHART_MARGINS,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    title_font_size=16,
    hovermode="x unified",
)
fig_sp.add_annotation(
    text=f"{SOURCE_ANNOTATION} | n={total:,}",
    xref="paper", yref="paper", x=0, y=-0.1,
    showarrow=False, font=dict(size=10, color="gray"),
)
st.plotly_chart(fig_sp, use_container_width=True)

st.divider()

# =============================================================================
# Causante + Tipo de Senalizacion (1:1 columns)
# =============================================================================
col_ev, col_sig = st.columns(2)

# Causante (donut)
with col_ev:
    st.subheader("Causante")
    if "causante" in df.columns and df["causante"].notna().any():
        caus_counts = df["causante"].value_counts().reset_index()
        caus_counts.columns = ["causante", "count"]
        causante_colors = {
            "Propio": "#1976D2",
            "Ajeno": "#FF5722",
        }
        fig_caus = px.pie(
            caus_counts,
            values="count",
            names="causante",
            color="causante",
            color_discrete_map=causante_colors,
            hole=0.5,
            template=CHART_TEMPLATE,
            height=CHART_H_SMALL,
        )
        fig_caus.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        fig_caus.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )
        st.plotly_chart(fig_caus, use_container_width=True)
        st.caption(
            "**Propio** = colision con la linea electrica propia del estudio. "
            "**Ajeno** = causa externa."
        )
    else:
        st.info("Datos de causante no disponibles.")

# Tipo de Senalizacion
with col_sig:
    st.subheader("Tipo de Senalizacion")
    sig_counts = (
        df[df["signal_type"].notna()]
        .groupby("signal_type")
        .size()
        .reset_index(name="count")
    )
    order = [s for s in SIGNAL_TYPES_ORDERED if s in sig_counts["signal_type"].values]
    sig_counts["signal_type"] = pd.Categorical(sig_counts["signal_type"], categories=order, ordered=True)
    sig_counts = sig_counts.sort_values("signal_type")

    fig_sig = px.bar(
        sig_counts,
        x="signal_type",
        y="count",
        color="signal_type",
        color_discrete_map={s: COLORS.get(s, "#999") for s in SIGNAL_TYPES_ORDERED},
        template=CHART_TEMPLATE,
        height=CHART_H_SMALL,
        labels={"signal_type": "", "count": "Victimas"},
    )
    fig_sig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        title_font_size=16,
        hovermode="x unified",
    )
    fig_sig.update_traces(text=sig_counts["count"], textposition="outside")
    fig_sig.add_annotation(
        text=SOURCE_ANNOTATION,
        xref="paper", yref="paper", x=0, y=-0.12,
        showarrow=False, font=dict(size=10, color="gray"),
    )
    st.plotly_chart(fig_sig, use_container_width=True)

    null_sig = df["signal_type"].isna().sum()
    if null_sig > 0:
        st.caption(f"{null_sig} registros con tipo de senalizacion desconocido excluidos.")

st.divider()

# =============================================================================
# Distribucion de Senalizacion por Linea (full-width)
# =============================================================================
st.subheader("Distribucion de Senalizacion por Linea")
st.caption("Muestra la combinacion de dispositivos anticolision instalados en cada linea donde se registraron victimas")

sig_line = (
    df[df["signal_type"].notna()]
    .groupby(["line_label", "signal_type"])
    .size()
    .reset_index(name="count")
)
fig_sig_line = px.bar(
    sig_line,
    x="line_label",
    y="count",
    color="signal_type",
    color_discrete_map={s: COLORS.get(s, "#999") for s in SIGNAL_TYPES_ORDERED},
    barmode="stack",
    labels={"line_label": "", "count": "Victimas", "signal_type": "Senalizacion"},
    template=CHART_TEMPLATE,
    height=CHART_H_BAR,
    category_orders={"signal_type": SIGNAL_TYPES_ORDERED},
)
fig_sig_line.update_layout(
    margin=CHART_MARGINS,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    title_font_size=16,
    hovermode="x unified",
)
fig_sig_line.add_annotation(
    text=f"{SOURCE_ANNOTATION} | n={total:,}",
    xref="paper", yref="paper", x=0, y=-0.12,
    showarrow=False, font=dict(size=10, color="gray"),
)
st.plotly_chart(fig_sig_line, use_container_width=True)

st.divider()

# =============================================================================
# Distribucion Mensual + Por Zona (side-by-side)
# =============================================================================
col_m, col_z = st.columns(2)

with col_m:
    st.subheader("Distribucion Mensual")
    month_names = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
    month_counts = df.groupby("month").size().reindex(range(1, 13), fill_value=0).reset_index()
    month_counts.columns = ["month", "count"]
    month_counts["month_name"] = month_counts["month"].map(month_names)

    fig_month = px.bar(
        month_counts,
        x="month_name",
        y="count",
        template=CHART_TEMPLATE,
        height=CHART_H_SMALL,
        labels={"month_name": "", "count": "Victimas"},
        color="count",
        color_continuous_scale="YlOrRd",
    )
    fig_month.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_showscale=False,
        title_font_size=16,
        hovermode="x unified",
    )
    fig_month.update_traces(text=month_counts["count"], textposition="outside")
    fig_month.add_annotation(
        text=SOURCE_ANNOTATION,
        xref="paper", yref="paper", x=0, y=-0.12,
        showarrow=False, font=dict(size=10, color="gray"),
    )
    st.plotly_chart(fig_month, use_container_width=True)

with col_z:
    st.subheader("Distribucion por Zona")
    zone_counts = df.groupby("zone").size().reset_index(name="count")
    fig_zone = px.pie(
        zone_counts,
        values="count",
        names="zone",
        color="zone",
        color_discrete_map={z: COLORS.get(z, "#999") for z in ["Sur", "Norte", "Medio"]},
        hole=0.5,
        template=CHART_TEMPLATE,
        height=CHART_H_SMALL,
    )
    fig_zone.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    fig_zone.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_zone, use_container_width=True)

# -- Descarga de datos --------------------------------------------------------
st.divider()
st.subheader("Descargar datos filtrados")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Descargar datos filtrados ({len(df):,} registros) como CSV",
    data=csv,
    file_name="fuerteventura_siniestralidad_avifauna_filtrado.csv",
    mime="text/csv",
)
