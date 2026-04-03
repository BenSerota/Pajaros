"""
Pagina 2: Analisis Espacial
Mapas interactivos, clustering DBSCAN de puntos calientes, vanos mas afectados
y proximidad a carreteras.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pydeck as pdk
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    COLORS, CHART_TEMPLATE, SOURCE_ANNOTATION,
    MAP_CENTER_LAT, MAP_CENTER_LON, MAP_DEFAULT_ZOOM,
    ROAD_DISTANCE_BINS, ROAD_DISTANCE_LABELS,
)
from src.data_loader import load_data
from src.filters import apply_filters, get_filter_summary
from src.statistics import (
    dbscan_clusters, poisson_test_per_span, kruskal_wallis_test,
    stat_badge, format_p_value,
)

# -- Constantes de diseno -----------------------------------------------------
MAP_HEIGHT = 600
CHART_H_BAR = 450
CHART_H_BOX = 450
CHART_H_HIST = 450
CHART_H_SMALL = 350
CHART_MARGINS = dict(l=20, r=20, t=40, b=60)

# -- Colores de patron de actividad (tuplas RGB para pydeck) -------------------
ACTIVITY_COLORS_RGB = {
    "Nocturna": [26, 35, 126],
    "Crepuscular": [106, 27, 154],
    "Diurna": [230, 81, 0],
    "Desconocida": [117, 117, 117],
}

# -- Configuracion de pagina --------------------------------------------------
st.header("Analisis Espacial")
st.caption("Mapas interactivos, puntos calientes de mortalidad y patrones geograficos en las lineas electricas de Fuerteventura")

# -- Carga y filtrado ---------------------------------------------------------
df_full = load_data()
df = apply_filters(df_full)
subtitle = get_filter_summary(df, df_full)

if len(df) == 0:
    st.warning("Ningun registro coincide con los filtros actuales. Ajuste los filtros para ver resultados.")
    st.stop()

# Restringir a registros con coordenadas validas
df_geo = df[df["latitude"].notna() & df["longitude"].notna()].copy()
n_no_coords = len(df) - len(df_geo)

if n_no_coords > 0:
    st.info(f"{n_no_coords} registros sin coordenadas validas excluidos de los mapas. Mostrando {len(df_geo):,} registros geolocalizados.")

if len(df_geo) == 0:
    st.warning("No hay registros geolocalizados disponibles tras el filtrado. Ajuste los filtros para ver resultados.")
    st.stop()

# -- Asignar columna de color RGB segun patron de actividad -------------------
df_geo["_color"] = df_geo["activity_pattern"].map(
    lambda ap: ACTIVITY_COLORS_RGB.get(ap, [117, 117, 117])
)

# Radio del punto escalado por conservation_score (min 50, max 300)
if "conservation_score" in df_geo.columns and df_geo["conservation_score"].notna().any():
    cs = df_geo["conservation_score"]
    cs_min, cs_max = cs.min(), cs.max()
    if cs_max > cs_min:
        df_geo["_radius"] = 50 + (cs - cs_min) / (cs_max - cs_min) * 250
    else:
        df_geo["_radius"] = 150.0
else:
    df_geo["_radius"] = 150.0

# Formatear fecha para tooltip
if "date" in df_geo.columns:
    df_geo["_date_str"] = df_geo["date"].dt.strftime("%Y-%m-%d").fillna("Desconocida")
else:
    df_geo["_date_str"] = "Desconocida"

st.divider()

# =============================================================================
# SECCION 1: Mapa Interactivo (full-width, 600px)
# =============================================================================
st.subheader("1. Mapa Interactivo")
st.caption(
    "Cada punto representa una victima registrada. El color indica el patron de actividad; "
    "el tamano del punto refleja la puntuacion de prioridad de conservacion (mayor = mas prioritario)."
)

# Leyenda
legend_cols = st.columns(4)
for i, (pattern, rgb) in enumerate(ACTIVITY_COLORS_RGB.items()):
    hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
    legend_cols[i].markdown(
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{hex_color};border-radius:50%;vertical-align:middle;'
        f'margin-right:6px;"></span> {pattern}',
        unsafe_allow_html=True,
    )

# Selector de modo de mapa
map_mode = st.radio(
    "Modo de mapa",
    ["Dispersion (pydeck)", "Mapa plotly", "Mapa de calor"],
    horizontal=True,
    key="spatial_map_mode",
)

view_state = pdk.ViewState(
    latitude=MAP_CENTER_LAT,
    longitude=MAP_CENTER_LON,
    zoom=MAP_DEFAULT_ZOOM,
    pitch=0,
)

if map_mode == "Dispersion (pydeck)":
    deck_df = df_geo[["latitude", "longitude", "_color", "_radius",
                       "species_clean", "_date_str", "activity_pattern"]].copy()
    for col in ["line_label", "signal_type", "event_type"]:
        if col in df_geo.columns:
            deck_df[col] = df_geo[col].fillna("N/D")
        else:
            deck_df[col] = "N/D"
    # Add vano_label to tooltip
    if "vano_label" in df_geo.columns:
        deck_df["vano_label"] = df_geo["vano_label"].fillna("N/D")
    else:
        deck_df["vano_label"] = "N/D"

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=deck_df,
        get_position=["longitude", "latitude"],
        get_fill_color="_color",
        get_radius="_radius",
        radius_min_pixels=3,
        radius_max_pixels=15,
        pickable=True,
        opacity=0.7,
    )

    tooltip = {
        "html": (
            "<b>{species_clean}</b><br>"
            "Fecha: {_date_str}<br>"
            "Linea: {line_label}<br>"
            "Vano: {vano_label}<br>"
            "Senalizacion: {signal_type}<br>"
            "Evento: {event_type}<br>"
            "Actividad: {activity_pattern}"
        ),
        "style": {"backgroundColor": "steelblue", "color": "white", "fontSize": "13px"},
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[scatter_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/light-v10",
        ),
        use_container_width=True,
        height=MAP_HEIGHT,
    )

elif map_mode == "Mapa plotly":
    hover_data_map = {
        "_date_str": True,
        "line_label": True,
        "signal_type": True,
        "event_type": True,
        "_radius": False,
        "latitude": ":.4f",
        "longitude": ":.4f",
    }
    if "vano_label" in df_geo.columns:
        hover_data_map["vano_label"] = True

    labels_map = {
        "_date_str": "Fecha",
        "line_label": "Linea",
        "signal_type": "Senalizacion",
        "event_type": "Evento",
        "activity_pattern": "Actividad",
        "vano_label": "Vano",
    }

    fig_map = px.scatter_mapbox(
        df_geo,
        lat="latitude",
        lon="longitude",
        color="activity_pattern",
        color_discrete_map={
            k: "#{:02x}{:02x}{:02x}".format(*v)
            for k, v in ACTIVITY_COLORS_RGB.items()
        },
        size="_radius",
        size_max=12,
        hover_name="species_clean",
        hover_data=hover_data_map,
        labels=labels_map,
        mapbox_style="open-street-map",
        zoom=MAP_DEFAULT_ZOOM,
        center={"lat": MAP_CENTER_LAT, "lon": MAP_CENTER_LON},
        height=MAP_HEIGHT,
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_map, use_container_width=True)

else:  # Mapa de calor
    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=df_geo[["latitude", "longitude"]],
        get_position=["longitude", "latitude"],
        aggregation="MEAN",
        threshold=0.05,
        radius_pixels=30,
        opacity=0.6,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[heat_layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v10",
        ),
        use_container_width=True,
        height=MAP_HEIGHT,
    )

st.caption(f"{SOURCE_ANNOTATION} | {subtitle} | n={len(df_geo):,} registros geolocalizados")

st.divider()

# =============================================================================
# SECCION 2: Analisis de Puntos Calientes
# =============================================================================
st.subheader("2. Analisis de Puntos Calientes")
st.caption(
    "El clustering DBSCAN basado en densidad sobre coordenadas UTM identifica "
    "clusteres espaciales de mortalidad. Los puntos a menos de 500 m de al menos "
    "2 otros eventos forman un cluster."
)

# -- Clustering DBSCAN (full-width) -------------------------------------------
st.markdown("**Clusteres DBSCAN**")

df_utm = df_geo[df_geo["utm_x"].notna() & df_geo["utm_y"].notna()].copy()

if len(df_utm) >= 3:
    coords = df_utm[["utm_x", "utm_y"]].values
    labels = dbscan_clusters(coords, eps=500, min_samples=3)
    df_utm["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_in_cluster = (labels != -1).sum()
    n_noise = (labels == -1).sum()

    # Metricas resumen
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Clusteres encontrados", n_clusters)
    mc2.metric("Puntos en clusteres", f"{n_in_cluster:,}")
    mc3.metric("Puntos aislados", f"{n_noise:,}")

    # Mapa de clusteres (full-width, 600px)
    df_utm["cluster_label"] = df_utm["cluster"].apply(
        lambda c: f"Cluster {c}" if c >= 0 else "Ruido"
    )

    fig_cluster = px.scatter_mapbox(
        df_utm,
        lat="latitude",
        lon="longitude",
        color="cluster_label",
        hover_name="species_clean",
        hover_data={"line_label": True, "cluster": True},
        mapbox_style="open-street-map",
        zoom=MAP_DEFAULT_ZOOM,
        center={"lat": MAP_CENTER_LAT, "lon": MAP_CENTER_LON},
        height=MAP_HEIGHT,
        opacity=0.7,
    )
    for trace in fig_cluster.data:
        if trace.name == "Ruido":
            trace.marker.size = 5
            trace.marker.opacity = 0.3
        else:
            trace.marker.size = 9

    fig_cluster.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=10),
        ),
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    # Detalles del cluster (full-width table, no expander)
    if n_clusters > 0:
        st.markdown("**Detalles de clusteres**")
        cluster_summary = (
            df_utm[df_utm["cluster"] >= 0]
            .groupby("cluster")
            .agg(
                casualties=("cluster", "size"),
                species=("species_clean", "nunique"),
                top_species=("species_clean", lambda x: x.value_counts().index[0]),
                lines=("line_label", "nunique"),
            )
            .sort_values("casualties", ascending=False)
            .reset_index()
        )
        cluster_summary.columns = ["Cluster", "Victimas", "Especies", "Especie principal", "Lineas"]
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True, height=400)
else:
    st.info("No hay suficientes registros geolocalizados para el clustering DBSCAN (minimo 3 requeridos).")

st.divider()

# =============================================================================
# SECCION 3: Top 20 Vanos Mas Afectados (full-width + clickable)
# =============================================================================
st.subheader("3. Top 20 Vanos Mas Afectados")

# Determine the vano column to use
vano_col = "vano_label" if "vano_label" in df.columns else "span_id"
has_vano = vano_col in df.columns and df[vano_col].notna().any()

if has_vano:
    # Build span-level aggregation using vano_label as primary display
    agg_dict = {
        "casualties": (vano_col, "size"),
    }
    if "line_label" in df.columns:
        agg_dict["line"] = ("line_label", "first")
    if "signal_type" in df.columns:
        agg_dict["signal_type"] = ("signal_type", "first")
    if "nearest_pylon" in df.columns:
        agg_dict["nearest_pylon"] = ("nearest_pylon", "first")
    if "span_id" in df.columns and vano_col != "span_id":
        agg_dict["span_id_val"] = ("span_id", "first")

    span_data = (
        df[df[vano_col].notna()]
        .groupby(vano_col)
        .agg(**agg_dict)
        .sort_values("casualties", ascending=False)
        .head(20)
        .reset_index()
    )

    # Build display columns
    col_rename = {vano_col: "Vano", "casualties": "Victimas"}
    if "line" in span_data.columns:
        col_rename["line"] = "Linea"
    if "signal_type" in span_data.columns:
        col_rename["signal_type"] = "Senalizacion"
    if "nearest_pylon" in span_data.columns:
        col_rename["nearest_pylon"] = "Apoyo mas cercano"
    if "span_id_val" in span_data.columns:
        col_rename["span_id_val"] = "ID Sistema"

    span_data = span_data.rename(columns=col_rename)

    # Full-width table
    st.dataframe(span_data, use_container_width=True, hide_index=True, height=400)

    # -- Clickable span selector for map zoom ---------------------------------
    st.markdown("**Ubicacion del vano seleccionado**")

    top_vanos = span_data["Vano"].tolist()
    selected_vano = st.selectbox(
        "Seleccionar vano para ver ubicacion:",
        options=["(ninguno)"] + top_vanos,
        index=0,
        key="spatial_vano_select",
    )

    if selected_vano != "(ninguno)":
        # Find records for this vano
        vano_records = df_geo[df_geo[vano_col] == selected_vano]

        if len(vano_records) > 0 and vano_records["latitude"].notna().any():
            avg_lat = vano_records["latitude"].mean()
            avg_lon = vano_records["longitude"].mean()

            fig_vano_map = px.scatter_mapbox(
                vano_records,
                lat="latitude",
                lon="longitude",
                color_discrete_sequence=["#D32F2F"],
                hover_name="species_clean",
                hover_data={
                    "_date_str": True,
                    "line_label": True,
                    "signal_type": True,
                },
                labels={
                    "_date_str": "Fecha",
                    "line_label": "Linea",
                    "signal_type": "Senalizacion",
                },
                mapbox_style="open-street-map",
                zoom=14,
                center={"lat": avg_lat, "lon": avg_lon},
                height=400,
            )
            fig_vano_map.update_traces(marker=dict(size=14))
            fig_vano_map.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig_vano_map, use_container_width=True)
            st.caption(
                f"Vano **{selected_vano}** — {len(vano_records)} victima(s) registrada(s). "
                f"Coordenadas medias: {avg_lat:.4f}, {avg_lon:.4f}"
            )
        else:
            st.info(f"El vano {selected_vano} no tiene coordenadas geolocalizadas disponibles.")

    st.divider()

    # -- Test de Poisson (full-width) ------------------------------------------
    st.markdown("**Test de Poisson para Puntos Calientes**")
    st.caption("Que vanos tienen significativamente mas eventos de los esperados bajo un modelo uniforme?")

    df_spans = df[df[vano_col].notna()].copy()
    poisson_df = poisson_test_per_span(df_spans, span_col=vano_col)

    if len(poisson_df) > 0:
        n_sig = poisson_df["significant"].sum()
        n_total_spans = len(poisson_df)
        expected_rate = poisson_df["expected"].iloc[0] if len(poisson_df) > 0 else 0

        st.markdown(
            f"**{n_sig}** de **{n_total_spans}** vanos tienen significativamente mas "
            f"eventos que la tasa esperada de **{expected_rate:.2f}** por vano "
            f"(test de Poisson unilateral, p < 0,05)."
        )

        # Mostrar vanos significativos (full-width)
        sig_spans = (
            poisson_df[poisson_df["significant"]]
            .sort_values("observed", ascending=False)
            .head(10)
        )
        if len(sig_spans) > 0:
            display_sig = sig_spans[["span_id", "observed", "expected", "p_value"]].copy()
            display_sig["p_value"] = display_sig["p_value"].apply(format_p_value)
            display_sig.columns = ["Vano", "Observado", "Esperado", "Valor p"]
            st.dataframe(display_sig, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("Ningun vano supera significativamente la tasa esperada.")
    else:
        st.info("No se pudo calcular el test de Poisson.")
else:
    st.info("Los datos de identificacion de vano no estan disponibles en el conjunto de datos actual.")

st.divider()

# =============================================================================
# SECCION 4: Proximidad a Carreteras (condicional)
# =============================================================================
st.subheader("4. Proximidad a Carreteras")

has_road_distance = "road_distance_m" in df.columns and df["road_distance_m"].notna().any()

if not has_road_distance:
    st.info(
        "El analisis de proximidad a carreteras requiere ejecutar "
        "`python scripts/precompute_roads.py` primero. "
        "Este script calcula la distancia de cada ubicacion de victima a la carretera "
        "mas cercana utilizando datos de OpenStreetMap."
    )
else:
    df_road = df[df["road_distance_m"].notna()].copy()
    n_road = len(df_road)
    st.caption(f"Analizando {n_road:,} registros con datos de proximidad a carreteras.")

    # -- Histograma de distancias a carretera (full-width) ---------------------
    st.markdown("**Distribucion de distancia a carretera**")

    df_road["road_bin"] = pd.cut(
        df_road["road_distance_m"],
        bins=ROAD_DISTANCE_BINS,
        labels=ROAD_DISTANCE_LABELS,
        right=False,
    )

    fig_hist = px.histogram(
        df_road,
        x="road_distance_m",
        nbins=50,
        template=CHART_TEMPLATE,
        height=CHART_H_HIST,
        labels={"road_distance_m": "Distancia a la carretera mas cercana (m)", "count": "Victimas"},
        color_discrete_sequence=[COLORS.get("Accidente", "#D32F2F")],
    )
    fig_hist.update_layout(
        margin=CHART_MARGINS,
        yaxis_title="Numero de victimas",
        title_font_size=16,
        hovermode="x unified",
    )
    fig_hist.add_annotation(
        text=f"{SOURCE_ANNOTATION} | n={n_road:,}",
        xref="paper", yref="paper", x=0, y=-0.15,
        showarrow=False, font=dict(size=10, color="gray"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # -- Distancia por patron de actividad (full-width) ------------------------
    st.markdown("**Distancia por patron de actividad**")

    if "activity_pattern" in df_road.columns:
        fig_box_act = px.box(
            df_road,
            x="activity_pattern",
            y="road_distance_m",
            color="activity_pattern",
            color_discrete_map={
                "Nocturna": "#{:02x}{:02x}{:02x}".format(*ACTIVITY_COLORS_RGB["Nocturna"]),
                "Crepuscular": "#{:02x}{:02x}{:02x}".format(*ACTIVITY_COLORS_RGB["Crepuscular"]),
                "Diurna": "#{:02x}{:02x}{:02x}".format(*ACTIVITY_COLORS_RGB["Diurna"]),
                "Desconocida": "#{:02x}{:02x}{:02x}".format(*ACTIVITY_COLORS_RGB["Desconocida"]),
            },
            template=CHART_TEMPLATE,
            height=CHART_H_BOX,
            labels={
                "activity_pattern": "",
                "road_distance_m": "Distancia a carretera (m)",
            },
            category_orders={
                "activity_pattern": ["Nocturna", "Crepuscular", "Diurna", "Desconocida"]
            },
        )
        fig_box_act.update_layout(
            showlegend=False,
            margin=CHART_MARGINS,
            title_font_size=16,
            hovermode="closest",
        )
        st.plotly_chart(fig_box_act, use_container_width=True)

        # Kruskal-Wallis badge BELOW chart
        groups_act = [
            df_road[df_road["activity_pattern"] == p]["road_distance_m"].dropna().values
            for p in ["Nocturna", "Crepuscular", "Diurna", "Desconocida"]
            if p in df_road["activity_pattern"].values
            and len(df_road[df_road["activity_pattern"] == p]["road_distance_m"].dropna()) > 0
        ]
        if len(groups_act) >= 2:
            kw_act = kruskal_wallis_test(groups_act)
            badge_act = stat_badge(
                "Kruskal-Wallis (Actividad)",
                kw_act["statistic"],
                kw_act["p_value"],
                effect_size=kw_act["effect_size"],
                n=n_road,
            )
            st.markdown(badge_act, unsafe_allow_html=True)
    else:
        st.info("Los datos de patron de actividad no estan disponibles.")

    st.divider()

    # -- Distancia por tipo de senalizacion (full-width) -----------------------
    st.markdown("**Distancia por tipo de senalizacion**")

    if "signal_type" in df_road.columns and df_road["signal_type"].notna().any():
        df_road_sig = df_road[df_road["signal_type"].notna()]

        fig_box_sig = px.box(
            df_road_sig,
            x="signal_type",
            y="road_distance_m",
            color="signal_type",
            color_discrete_map={
                s: COLORS.get(s, "#999")
                for s in df_road_sig["signal_type"].unique()
            },
            template=CHART_TEMPLATE,
            height=CHART_H_BOX,
            labels={
                "signal_type": "",
                "road_distance_m": "Distancia a carretera (m)",
            },
        )
        fig_box_sig.update_layout(
            showlegend=False,
            margin=CHART_MARGINS,
            title_font_size=16,
            hovermode="closest",
        )
        st.plotly_chart(fig_box_sig, use_container_width=True)

        # Kruskal-Wallis badge BELOW chart
        groups_sig = [
            df_road_sig[df_road_sig["signal_type"] == s]["road_distance_m"].dropna().values
            for s in df_road_sig["signal_type"].unique()
            if len(df_road_sig[df_road_sig["signal_type"] == s]["road_distance_m"].dropna()) > 0
        ]
        if len(groups_sig) >= 2:
            kw_sig = kruskal_wallis_test(groups_sig)
            badge_sig = stat_badge(
                "Kruskal-Wallis (Senalizacion)",
                kw_sig["statistic"],
                kw_sig["p_value"],
                effect_size=kw_sig["effect_size"],
                n=len(df_road_sig),
            )
            st.markdown(badge_sig, unsafe_allow_html=True)
    else:
        st.info("Los datos de tipo de senalizacion no estan disponibles para los registros de proximidad a carreteras.")

    st.divider()

    # -- Tabla resumen por rango de distancia (full-width, no expander) --------
    st.markdown("**Desglose por rango de distancia**")
    agg_dict_road = {
        "casualties": ("road_bin", "size"),
        "species": ("species_clean", "nunique"),
    }
    if "conservation_score" in df_road.columns:
        agg_dict_road["median_conservation"] = ("conservation_score", "median")

    bin_summary = (
        df_road.groupby("road_bin", observed=True)
        .agg(**agg_dict_road)
        .reset_index()
    )

    if "median_conservation" in bin_summary.columns:
        bin_summary.columns = ["Rango de distancia", "Victimas", "Especies", "Mediana puntuacion conservacion"]
    else:
        bin_summary.columns = ["Rango de distancia", "Victimas", "Especies"]

    bin_summary["% del total"] = (bin_summary["Victimas"] / bin_summary["Victimas"].sum() * 100).round(1)
    st.dataframe(bin_summary, use_container_width=True, hide_index=True, height=400)
