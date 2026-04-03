"""
Pagina 6: Impacto en Conservacion
Analisis en profundidad de la mortalidad de especies amenazadas y protegidas
en las lineas electricas de Fuerteventura.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COLORS, CHART_TEMPLATE, SOURCE_ANNOTATION, MAP_CENTER_LAT, MAP_CENTER_LON
from src.data_loader import load_data
from src.filters import apply_filters, get_filter_summary
from src.statistics import binomial_test, stat_badge, format_p_value

# -- Constantes de diseno -----------------------------------------------------
CHART_MARGINS = dict(l=20, r=20, t=40, b=60)
CHART_H_DEFAULT = 450
CHART_H_MAP = 500
CHART_H_BUBBLE = 550
CHART_H_SMALL = 350

# -- Configuracion de pagina --------------------------------------------------
st.header("Impacto en Conservacion")
st.caption("Analisis de mortalidad de especies amenazadas, protegidas y focales en las lineas electricas de Fuerteventura")

# -- Cargar y filtrar ----------------------------------------------------------
df_full = load_data()
df = apply_filters(df_full)
subtitle = get_filter_summary(df, df_full)

if len(df) == 0:
    st.warning("Ningun registro coincide con los filtros actuales. Ajuste los filtros para ver resultados.")
    st.stop()

total = len(df)

# =============================================================================
# SECCION 1: Fila de KPIs de Conservacion
# =============================================================================
st.subheader("Metricas Clave de Conservacion")

col1, col2, col3, col4, col5 = st.columns(5)

# Victimas UICN Vulnerable+
iucn_vuln_plus = df[df["iucn_status"].isin(["Vulnerable", "En peligro de extincion"])]
with col1:
    st.metric("Victimas UICN Vulnerable+", f"{len(iucn_vuln_plus):,}")

# Victimas en peligro nacional
nat_endangered = df[df["spanish_catalog"] == "En peligro de extincion"]
with col2:
    st.metric("Victimas en Peligro Nacional", f"{len(nat_endangered):,}")

# Victimas de especies focales
focal_mask = (df["is_focal"] == True) | (df["is_focal"] == "Si")
focal_kills = df[focal_mask]
with col3:
    st.metric("Victimas Especies Focales", f"{len(focal_kills):,}")

# Especies protegidas unicas
protected_mask = (
    df["iucn_status"].notna() & (df["iucn_status"] != "Riesgo menor-preocupacion menor")
) | (
    df["spanish_catalog"].notna() & (df["spanish_catalog"] != "")
) | (
    df["regional_catalog"].notna() & (df["regional_catalog"] != "")
)
protected_species = df.loc[protected_mask, "species_clean"].nunique()
with col4:
    st.metric("Especies Protegidas Unicas", protected_species)

# % de mortalidad total por especies protegidas
protected_kills = protected_mask.sum()
pct_protected = protected_kills / total * 100 if total > 0 else 0
with col5:
    st.metric("% Mortalidad Especies Protegidas", f"{pct_protected:.1f}%")

st.divider()

# =============================================================================
# SECCION 2: Tabla de Prioridad de Conservacion (full-width)
# =============================================================================
st.subheader("Tabla de Prioridad de Conservacion")
st.caption("Todas las especies con puntuacion de conservacion > 0, ordenadas por prioridad")

df_conserv = df[df["conservation_score"] > 0].copy()

if len(df_conserv) > 0:
    priority_table = (
        df_conserv.groupby("species_clean")
        .agg(
            count=("species_clean", "size"),
            iucn_status=("iucn_status", "first"),
            spanish_catalog=("spanish_catalog", "first"),
            regional_catalog=("regional_catalog", "first"),
            is_focal=("is_focal", "first"),
            conservation_score=("conservation_score", "first"),
        )
        .reset_index()
        .sort_values("conservation_score", ascending=False)
    )
    priority_table.columns = [
        "Especie", "Victimas", "Estado UICN", "Catalogo Espanol",
        "Catalogo Regional", "Especie Focal", "Puntuacion de Conservacion",
    ]

    st.dataframe(
        priority_table,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "Puntuacion de Conservacion": st.column_config.NumberColumn(format="%.1f"),
            "Victimas": st.column_config.NumberColumn(format="%d"),
        },
    )
else:
    st.info("No hay especies con puntuacion de conservacion > 0 en la seleccion de filtros actual.")

st.divider()

# =============================================================================
# SECCION 3: Analisis Detallado: Avutarda Hubara (map full-width 500px)
# =============================================================================
st.subheader("Analisis Detallado: Avutarda Hubara (*Chlamydotis undulata*)")
st.warning(
    "**UICN Vulnerable** | **En peligro de extincion (Catalogo Nacional)** | "
    "La Hubara canaria es una especie iconica de zonas aridas con una poblacion insular "
    "pequena y fragmentada. La mortalidad por lineas electricas es una amenaza documentada "
    "para esta subespecie."
)

houbara_mask = df["species_scientific"].str.contains("Chlamydotis", case=False, na=False)
df_houbara = df[houbara_mask].copy()

if len(df_houbara) > 0:
    # Mapa de victimas de Hubara (full-width, 500px)
    df_houbara_map = df_houbara.dropna(subset=["latitude", "longitude"])
    if len(df_houbara_map) > 0:
        fig_houbara_map = px.scatter_mapbox(
            df_houbara_map,
            lat="latitude",
            lon="longitude",
            hover_data=["date", "line_label", "signal_type", "event_type", "zone"],
            color_discrete_sequence=[COLORS["En peligro de extincion"]],
            mapbox_style="carto-positron",
            zoom=9,
            center={"lat": MAP_CENTER_LAT, "lon": MAP_CENTER_LON},
            height=CHART_H_MAP,
        )
        fig_houbara_map.update_layout(
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig_houbara_map, use_container_width=True)

    # Linea temporal (full-width)
    st.markdown("**Victimas de Hubara a lo largo del tiempo**")
    fig_timeline = px.scatter(
        df_houbara,
        x="date",
        y="line_label",
        color_discrete_sequence=[COLORS["En peligro de extincion"]],
        hover_data=["signal_type", "event_type", "zone"],
        template=CHART_TEMPLATE,
        height=CHART_H_SMALL,
        labels={"date": "Fecha", "line_label": "Linea electrica"},
    )
    fig_timeline.update_traces(marker=dict(size=12, symbol="x"))
    fig_timeline.update_layout(
        margin=CHART_MARGINS,
        showlegend=False,
        title_font_size=16,
        hovermode="closest",
    )
    fig_timeline.add_annotation(
        text=SOURCE_ANNOTATION,
        xref="paper", yref="paper", x=0, y=-0.12,
        showarrow=False, font=dict(size=10, color="gray"),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Tabla de registros (full-width)
    st.markdown("**Todos los registros de Hubara**")
    display_cols = ["date", "line_label", "signal_type", "event_type", "zone", "sex", "age"]
    available_cols = [c for c in display_cols if c in df_houbara.columns]
    houbara_display = df_houbara[available_cols].copy()
    if "date" in houbara_display.columns:
        houbara_display["date"] = houbara_display["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(houbara_display, use_container_width=True, hide_index=True, height=400)

    st.info(
        f"**{len(df_houbara)} victimas** representan una mortalidad significativa para una pequena "
        "poblacion insular. La Hubara canaria (*Chlamydotis undulata fuertaventurae*) tiene una "
        "poblacion estimada de solo unos pocos cientos de individuos en Fuerteventura."
    )
else:
    st.info("No hay registros de Avutarda Hubara en la seleccion de filtros actual.")

st.divider()

# =============================================================================
# SECCION 4: Nota: Alimoche Comun
# =============================================================================
st.subheader("Nota: Alimoche Comun (*Neophron percnopterus*)")

vulture_mask = df["species_scientific"].str.contains("Neophron", case=False, na=False)
df_vulture = df[vulture_mask].copy()

if len(df_vulture) > 0:
    st.warning(
        f"**{len(df_vulture)} victima(s)** de una especie **UICN En Peligro**. "
        "El Alimoche comun (*Neophron percnopterus*) esta globalmente en peligro con una poblacion "
        "en declive. Incluso un unico evento de mortalidad es significativo para esta especie "
        "en las Islas Canarias."
    )
    display_cols = ["date", "line_label", "signal_type", "event_type", "zone", "sex", "age",
                    "iucn_status", "spanish_catalog", "regional_catalog"]
    available_cols = [c for c in display_cols if c in df_vulture.columns]
    vulture_display = df_vulture[available_cols].copy()
    if "date" in vulture_display.columns:
        vulture_display["date"] = vulture_display["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(vulture_display, use_container_width=True, hide_index=True, height=400)
else:
    st.info("No hay registros de Alimoche comun en la seleccion de filtros actual.")

st.divider()

# =============================================================================
# SECCION 5: Mortalidad de Aves Marinas Nocturnas (full-width charts)
# =============================================================================
st.subheader("Mortalidad de Aves Marinas Nocturnas")
st.caption(
    "Las aves marinas Procellariiformes (pardelas, petreles, painos) son estrictamente nocturnas "
    "en tierra y fuertemente atraidas por la luz artificial. No pueden ver las marcas anticolision "
    "visuales durante la noche."
)

seabird_mask = df["species_scientific"].str.contains(
    "Calonectris|Bulweria|Pelagodroma", case=False, na=False
)
df_seabirds = df[seabird_mask].copy()

if len(df_seabirds) > 0:
    # KPIs
    col_s1, col_s2, col_s3 = st.columns(3)
    seabird_total = len(df_seabirds)
    seabird_pct = seabird_total / total * 100 if total > 0 else 0
    seabird_species = df_seabirds["species_clean"].nunique()

    with col_s1:
        st.metric("Total victimas aves marinas", f"{seabird_total:,}")
    with col_s2:
        st.metric("% de toda la mortalidad", f"{seabird_pct:.1f}%")
    with col_s3:
        st.metric("Especies de aves marinas unicas", seabird_species)

    # Grafico de barras estacional (full-width)
    st.markdown("**Distribucion mensual de victimas de aves marinas**")
    month_names = {
        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
    }
    seabird_months = (
        df_seabirds.groupby("month")
        .size()
        .reindex(range(1, 13), fill_value=0)
        .reset_index()
    )
    seabird_months.columns = ["month", "count"]
    seabird_months["month_name"] = seabird_months["month"].map(month_names)

    fig_seasonal = px.bar(
        seabird_months,
        x="month_name",
        y="count",
        template=CHART_TEMPLATE,
        height=CHART_H_DEFAULT,
        labels={"month_name": "", "count": "Victimas aves marinas"},
        color="count",
        color_continuous_scale="Blues",
    )
    fig_seasonal.update_layout(
        margin=CHART_MARGINS,
        coloraxis_showscale=False,
        title_font_size=16,
        hovermode="x unified",
    )
    fig_seasonal.update_traces(text=seabird_months["count"], textposition="outside")
    fig_seasonal.add_annotation(
        text=f"{SOURCE_ANNOTATION} | n={seabird_total:,}",
        xref="paper", yref="paper", x=0, y=-0.1,
        showarrow=False, font=dict(size=10, color="gray"),
    )
    st.plotly_chart(fig_seasonal, use_container_width=True)

    st.divider()

    # Tipo de senalizacion: aves marinas vs resto (full-width)
    st.markdown("**Distribucion por tipo de senalizacion: aves marinas vs otras especies**")
    df_other = df[~seabird_mask].copy()

    seabird_sig = (
        df_seabirds[df_seabirds["signal_type"].notna()]
        .groupby("signal_type").size()
        .reset_index(name="count")
    )
    seabird_sig["group"] = "Aves marinas"

    other_sig = (
        df_other[df_other["signal_type"].notna()]
        .groupby("signal_type").size()
        .reset_index(name="count")
    )
    other_sig["group"] = "Otras especies"

    sig_combined = pd.concat([seabird_sig, other_sig], ignore_index=True)

    group_totals = sig_combined.groupby("group")["count"].transform("sum")
    sig_combined["proportion"] = sig_combined["count"] / group_totals * 100

    fig_signal_cmp = px.bar(
        sig_combined,
        x="signal_type",
        y="proportion",
        color="group",
        barmode="group",
        template=CHART_TEMPLATE,
        height=CHART_H_DEFAULT,
        labels={"signal_type": "", "proportion": "% del grupo", "group": ""},
        color_discrete_map={"Aves marinas": COLORS["Nocturna"], "Otras especies": "#BDBDBD"},
    )
    fig_signal_cmp.update_layout(
        margin=CHART_MARGINS,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_font_size=16,
        hovermode="x unified",
    )
    fig_signal_cmp.add_annotation(
        text=SOURCE_ANNOTATION,
        xref="paper", yref="paper", x=0, y=-0.1,
        showarrow=False, font=dict(size=10, color="gray"),
    )
    st.plotly_chart(fig_signal_cmp, use_container_width=True)

    st.info(
        "**Mensaje clave:** Las aves marinas nocturnas no pueden ver las marcas anticolision visuales. "
        "Su alta tasa de mortalidad en todos los tipos de senalizacion respalda el argumento a favor de "
        "dispositivos reflectantes UV o iluminados en los vanos cercanos a las colonias de cria."
    )
else:
    st.info("No hay registros de aves marinas nocturnas (Calonectris, Bulweria, Pelagodroma) en la seleccion de filtros actual.")

st.divider()

# =============================================================================
# SECCION 6: Matriz de Prioridad de Conservacion (full-width, 550px)
# =============================================================================
st.subheader("Matriz de Prioridad de Conservacion")
st.caption("Las especies en el cuadrante superior derecho son las de mayor prioridad para mitigacion")

if "conservation_score" in df.columns and "activity_pattern" in df.columns:
    bubble_data = (
        df[df["conservation_score"] > 0]
        .groupby(["species_clean", "activity_pattern"])
        .agg(
            kills=("species_clean", "size"),
            conservation_score=("conservation_score", "first"),
        )
        .reset_index()
    )

    if len(bubble_data) > 0:
        bubble_data["label"] = bubble_data.apply(
            lambda r: r["species_clean"].split(" / ")[0] if r["conservation_score"] > 3 else "",
            axis=1,
        )

        activity_color_map = {
            "Nocturna": COLORS.get("Nocturna", "#1a237e"),
            "Crepuscular": COLORS.get("Crepuscular", "#6a1b9a"),
            "Diurna": COLORS.get("Diurna", "#e65100"),
            "Desconocida": COLORS.get("Desconocida", "#757575"),
        }

        fig_bubble = px.scatter(
            bubble_data,
            x="kills",
            y="conservation_score",
            size="kills",
            color="activity_pattern",
            color_discrete_map=activity_color_map,
            text="label",
            hover_data=["species_clean", "kills", "conservation_score"],
            template=CHART_TEMPLATE,
            height=CHART_H_BUBBLE,
            labels={
                "kills": "Numero de victimas",
                "conservation_score": "Puntuacion de conservacion",
                "activity_pattern": "Patron de actividad",
            },
        )
        fig_bubble.update_traces(
            textposition="top center",
            textfont=dict(size=10),
        )
        fig_bubble.update_layout(
            margin=CHART_MARGINS,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title_font_size=16,
            hovermode="closest",
        )
        fig_bubble.add_annotation(
            text=SOURCE_ANNOTATION,
            xref="paper", yref="paper", x=0, y=-0.1,
            showarrow=False, font=dict(size=10, color="gray"),
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.info("No hay especies con puntuacion de conservacion > 0 para mostrar.")
