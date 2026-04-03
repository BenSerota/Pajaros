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
from config import COLORS, CHART_HEIGHT, CHART_TEMPLATE, SOURCE_ANNOTATION, SIGNAL_TYPES_ORDERED
from src.data_loader import load_data
from src.filters import apply_filters, get_filter_summary

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

# Accidentes confirmados
accidents = (df["event_type"] == "Accidente").sum()
acc_pct = accidents / total * 100 if total > 0 else 0
with col2:
    st.metric("Accidentes Confirmados", f"{accidents:,}", f"{acc_pct:.1f}% del total")

# Especies afectadas
n_species = df["species_clean"].nunique()
with col3:
    st.metric("Especies Afectadas", n_species)

# Victimas de especies amenazadas (UICN Vulnerable o peor, o catalogo espanol Vulnerable+)
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

# -- Fila 2: Graficos resumen principales ------------------------------------
col_left, col_right = st.columns(2)

# Victimas por linea electrica (barras horizontales)
with col_left:
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
        height=350,
    )
    fig_line.update_layout(
        legend_title_text="Tension",
        margin=dict(l=0, r=20, t=10, b=40),
        xaxis_title="Numero de victimas",
    )
    fig_line.add_annotation(
        text=f"{SOURCE_ANNOTATION} | n={total:,}",
        xref="paper", yref="paper", x=0, y=-0.15,
        showarrow=False, font=dict(size=10, color="gray"),
    )
    st.plotly_chart(fig_line, use_container_width=True)

# Victimas por ano (barras + tendencia)
with col_right:
    st.subheader("Victimas por Ano")
    year_counts = df.groupby("year").size().reset_index(name="count")
    year_counts["year"] = year_counts["year"].astype(int)

    # Marcar anos parciales
    min_year = int(df["year"].min()) if len(df) > 0 else 2018
    max_year = int(df["year"].max()) if len(df) > 0 else 2025
    year_counts["partial"] = year_counts["year"].isin([min_year, max_year])
    year_counts["label"] = year_counts.apply(
        lambda r: f"{r['year']}*" if r["partial"] else str(r["year"]), axis=1
    )

    fig_year = go.Figure()
    fig_year.add_trace(go.Bar(
        x=year_counts["year"],
        y=year_counts["count"],
        marker_color=[
            "#BDBDBD" if p else COLORS["Accidente"]
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
        height=350,
        showlegend=False,
        margin=dict(l=0, r=20, t=10, b=60),
        xaxis_title="Ano",
        yaxis_title="Victimas",
        xaxis=dict(dtick=1),
    )
    fig_year.add_annotation(
        text=f"*Ano parcial | {SOURCE_ANNOTATION} | n={total:,}",
        xref="paper", yref="paper", x=0, y=-0.22,
        showarrow=False, font=dict(size=10, color="gray"),
    )
    st.plotly_chart(fig_year, use_container_width=True)

st.divider()

# -- Fila 3: Especies + Tipo de evento + Senalizacion ------------------------
col_sp, col_ev, col_sig = st.columns([2, 1, 1])

# Top 15 Especies
with col_sp:
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
        height=CHART_HEIGHT,
    )
    fig_sp.update_layout(
        margin=dict(l=0, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_sp, use_container_width=True)

# Tipo de Evento (donut)
with col_ev:
    st.subheader("Tipo de Evento")
    ev_counts = df["event_type"].value_counts().reset_index()
    ev_counts.columns = ["event_type", "count"]
    fig_ev = px.pie(
        ev_counts,
        values="count",
        names="event_type",
        color="event_type",
        color_discrete_map={
            "Accidente": COLORS["Accidente"],
            "Incidente": COLORS["Incidente"],
        },
        hole=0.5,
        template=CHART_TEMPLATE,
        height=350,
    )
    fig_ev.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    fig_ev.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )
    st.plotly_chart(fig_ev, use_container_width=True)

    st.caption(
        "**Accidente** = evento de mortalidad confirmado. "
        "**Incidente** = evidencia de colision encontrada (restos)."
    )

# Tipo de Senalizacion
with col_sig:
    st.subheader("Tipo de Senalizacion")
    sig_counts = (
        df[df["signal_type"].notna()]
        .groupby("signal_type")
        .size()
        .reset_index(name="count")
    )
    # Orden
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
        height=350,
        labels={"signal_type": "", "count": "Victimas"},
    )
    fig_sig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=20, t=10, b=10),
    )
    fig_sig.update_traces(text=sig_counts["count"], textposition="outside")
    st.plotly_chart(fig_sig, use_container_width=True)

    null_sig = df["signal_type"].isna().sum()
    if null_sig > 0:
        st.caption(f"{null_sig} registros con tipo de senalizacion desconocido excluidos.")

st.divider()

# -- Fila 4: Senalizacion por linea (barras apiladas) ------------------------
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
    height=400,
    category_orders={"signal_type": SIGNAL_TYPES_ORDERED},
)
fig_sig_line.update_layout(
    margin=dict(l=0, r=20, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_sig_line, use_container_width=True)

st.divider()

# -- Fila 5: Distribucion mensual + por zona ---------------------------------
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
        height=300,
        labels={"month_name": "", "count": "Victimas"},
        color="count",
        color_continuous_scale="YlOrRd",
    )
    fig_month.update_layout(
        margin=dict(l=0, r=20, t=10, b=10),
        coloraxis_showscale=False,
    )
    fig_month.update_traces(text=month_counts["count"], textposition="outside")
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
        height=300,
    )
    fig_zone.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    fig_zone.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_zone, use_container_width=True)

# -- Descarga de datos --------------------------------------------------------
st.divider()
with st.expander("Descargar datos filtrados"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Descargar datos filtrados ({len(df):,} registros) como CSV",
        data=csv,
        file_name="fuerteventura_siniestralidad_avifauna_filtrado.csv",
        mime="text/csv",
    )
