"""
Pagina 5: Analisis de Especies
Desglose taxonomico, perfiles de patron de actividad, curvas de acumulacion,
mapa de calor de zonificacion ecologica y sesgo de deteccion basado en restos.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COLORS, CHART_TEMPLATE, SOURCE_ANNOTATION
from src.data_loader import load_data
from src.filters import apply_filters, get_filter_summary
from src.statistics import chi_squared_test, stat_badge, format_p_value

# -- Constantes de diseno -----------------------------------------------------
CHART_MARGINS = dict(l=20, r=20, t=40, b=60)
CHART_H_DEFAULT = 450
CHART_H_TREEMAP = 550
CHART_H_SUNBURST = 550
CHART_H_SMALL = 350

# -- Configuracion de pagina --------------------------------------------------
st.header("Analisis de Especies")
st.caption("Desglose taxonomico de la mortalidad de aves — patrones de actividad, zonificacion ecologica y sesgo de deteccion")

# -- Cargar y filtrar ----------------------------------------------------------
df_full = load_data()
df = apply_filters(df_full)
subtitle = get_filter_summary(df, df_full)
total = len(df)

if total == 0:
    st.warning("Ningun registro coincide con los filtros actuales. Ajuste los filtros para ver resultados.")
    st.stop()

# -- Mapa de colores por patron de actividad
ACTIVITY_COLOR_MAP = {
    "Nocturna": COLORS.get("Nocturna", "#1a237e"),
    "Crepuscular": COLORS.get("Crepuscular", "#6a1b9a"),
    "Diurna": COLORS.get("Diurna", "#e65100"),
    "Desconocida": COLORS.get("Desconocida", "#757575"),
}

# =============================================================================
# SECCION 1: Mapa de Arbol de Especies (full-width, 550px)
# =============================================================================
st.subheader("1. Mapa de Arbol de Especies")
st.caption(
    "Vista jerarquica de victimas por genero y especie. "
    "El color indica el patron de actividad de cada especie."
)

if "species_genus" in df.columns and "species_clean" in df.columns:
    treemap_df = (
        df.groupby(["species_genus", "species_clean", "activity_pattern"])
        .size()
        .reset_index(name="count")
    )

    if len(treemap_df) > 0:
        fig_tree = px.treemap(
            treemap_df,
            path=["species_genus", "species_clean"],
            values="count",
            color="activity_pattern",
            color_discrete_map=ACTIVITY_COLOR_MAP,
            template=CHART_TEMPLATE,
            height=CHART_H_TREEMAP,
        )
        fig_tree.update_layout(
            margin=CHART_MARGINS,
            title_font_size=16,
        )
        fig_tree.update_traces(
            hovertemplate="<b>%{label}</b><br>Victimas: %{value}<extra></extra>",
        )
        fig_tree.add_annotation(
            text=f"{SOURCE_ANNOTATION} | n={total:,}",
            xref="paper", yref="paper", x=0, y=-0.05,
            showarrow=False, font=dict(size=10, color="gray"),
        )
        st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("No hay datos de especies disponibles para el mapa de arbol.")
else:
    st.info("Las columnas de genero/especie no estan disponibles en el conjunto de datos.")

st.divider()

# =============================================================================
# SECCION 2: Resumen por Patron de Actividad (KPIs + Sunburst full-width 550px)
# =============================================================================
st.subheader("2. Resumen por Patron de Actividad")
st.caption("Cuando son mas activas las especies afectadas? El predominio nocturno sugiere que las marcas visuales pueden ser insuficientes.")

if "activity_pattern" in df.columns:
    activity_counts = df["activity_pattern"].value_counts()

    # -- Fila de KPIs
    kpi_patterns = ["Nocturna", "Crepuscular", "Diurna"]
    kpi_labels = {
        "Nocturna": "Victimas nocturnas",
        "Crepuscular": "Victimas crepusculares",
        "Diurna": "Victimas diurnas",
    }
    kpi_cols = st.columns(len(kpi_patterns))

    for col_target, pattern in zip(kpi_cols, kpi_patterns):
        count = int(activity_counts.get(pattern, 0))
        pct = count / total * 100 if total > 0 else 0
        with col_target:
            st.metric(
                kpi_labels[pattern],
                f"{count:,}",
                f"{pct:.1f}% del total",
                delta_color="off",
            )

    # -- Sunburst (full-width, 550px)
    sunburst_df = (
        df.groupby(["activity_pattern", "species_clean"])
        .size()
        .reset_index(name="count")
    )

    if len(sunburst_df) > 0:
        fig_sun = px.sunburst(
            sunburst_df,
            path=["activity_pattern", "species_clean"],
            values="count",
            color="activity_pattern",
            color_discrete_map=ACTIVITY_COLOR_MAP,
            template=CHART_TEMPLATE,
            height=CHART_H_SUNBURST,
        )
        fig_sun.update_layout(
            margin=CHART_MARGINS,
            title_font_size=16,
        )
        fig_sun.add_annotation(
            text=f"{SOURCE_ANNOTATION} | n={total:,}",
            xref="paper", yref="paper", x=0, y=-0.05,
            showarrow=False, font=dict(size=10, color="gray"),
        )
        st.plotly_chart(fig_sun, use_container_width=True)
    else:
        st.info("No hay datos disponibles para el grafico sunburst.")
else:
    st.info("La columna de patron de actividad no esta disponible en el conjunto de datos.")

st.divider()

# =============================================================================
# SECCION 3: Perfil de Colision Nocturna vs Diurna (side-by-side bars, chi2 badge below)
# =============================================================================
st.subheader("3. Perfil de Colision Nocturna vs Diurna")
st.caption(
    "Comparacion lado a lado de las especies mas afectadas por patron de actividad. "
    "Debajo: prueba chi-cuadrado sobre si el patron de actividad y el tipo de senalizacion son independientes."
)

if "activity_pattern" in df.columns:
    col_noct, col_diur = st.columns(2)

    for pattern, col_target, bar_color, label in [
        ("Nocturna", col_noct, COLORS.get("Nocturna", "#1a237e"), "Especies nocturnas"),
        ("Diurna", col_diur, COLORS.get("Diurna", "#e65100"), "Especies diurnas"),
    ]:
        with col_target:
            st.markdown(f"**{label} — Principales victimas**")
            subset = df[df["activity_pattern"] == pattern]
            if len(subset) > 0:
                sp_top = (
                    subset["species_clean"]
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                sp_top.columns = ["species_clean", "count"]
                sp_top = sp_top.sort_values("count", ascending=True)

                fig_bar = px.bar(
                    sp_top,
                    x="count",
                    y="species_clean",
                    orientation="h",
                    template=CHART_TEMPLATE,
                    height=CHART_H_SMALL,
                    labels={"count": "Victimas", "species_clean": ""},
                )
                fig_bar.update_traces(marker_color=bar_color)
                fig_bar.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20),
                    title_font_size=16,
                    hovermode="x unified",
                )
                fig_bar.add_annotation(
                    text=SOURCE_ANNOTATION,
                    xref="paper", yref="paper", x=0, y=-0.12,
                    showarrow=False, font=dict(size=10, color="gray"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.caption(f"n = {len(subset):,}")
            else:
                st.info(f"No hay registros de especies {pattern.lower()}s.")

    # -- Chi-cuadrado badge BELOW charts
    if "signal_type" in df.columns:
        df_test = df[
            df["activity_pattern"].isin(["Nocturna", "Crepuscular", "Diurna"])
            & df["signal_type"].notna()
        ].copy()

        if len(df_test) > 0:
            ct_activity_signal = pd.crosstab(df_test["activity_pattern"], df_test["signal_type"])
            if ct_activity_signal.shape[0] >= 2 and ct_activity_signal.shape[1] >= 2:
                result = chi_squared_test(ct_activity_signal)
                badge_html = stat_badge(
                    "Chi-cuadrado (Actividad x Senalizacion)",
                    result["statistic"],
                    result["p_value"],
                    effect_size=result["effect_size"],
                    n=len(df_test),
                )
                st.markdown(badge_html, unsafe_allow_html=True)

                st.markdown("**Tabla de contingencia: Patron de Actividad x Tipo de Senalizacion**")
                st.dataframe(ct_activity_signal, use_container_width=True)
                st.markdown(f"""
                - **chi-cuadrado** = {result['statistic']:.2f}, gl = {result['dof']}
                - **p** = {format_p_value(result['p_value'])}
                - **V de Cramer** = {result['effect_size']:.3f}
                """)
            else:
                st.info("No hay suficientes categorias para la prueba chi-cuadrado.")
        else:
            st.info("No hay registros con patron de actividad y tipo de senalizacion para la prueba estadistica.")

st.divider()

# =============================================================================
# SECCION 4: Curva de Acumulacion de Especies (full-width)
# =============================================================================
st.subheader("4. Curva de Acumulacion de Especies")
st.caption(
    "Indica si el muestreo ha alcanzado la saturacion de especies o si se siguen "
    "descubriendo nuevas especies."
)

if "date" in df.columns and "species_clean" in df.columns:
    df_sorted = df.dropna(subset=["date", "species_clean"]).sort_values("date").reset_index(drop=True)

    if len(df_sorted) > 0:
        seen_species = set()
        cumulative = []
        for _, row in df_sorted.iterrows():
            seen_species.add(row["species_clean"])
            cumulative.append(len(seen_species))

        accum_df = pd.DataFrame({
            "record_number": range(1, len(cumulative) + 1),
            "cumulative_species": cumulative,
        })

        fig_accum = px.line(
            accum_df,
            x="record_number",
            y="cumulative_species",
            template=CHART_TEMPLATE,
            height=CHART_H_DEFAULT,
            labels={
                "record_number": "Numero de registro (orden cronologico)",
                "cumulative_species": "Especies unicas acumuladas",
            },
        )
        fig_accum.update_traces(
            line=dict(color=COLORS.get("Accidente", "#D32F2F"), width=2.5),
            fill="tozeroy",
            fillcolor="rgba(211, 47, 47, 0.08)",
        )
        fig_accum.update_layout(
            margin=CHART_MARGINS,
            title_font_size=16,
            hovermode="x unified",
        )

        final_count = cumulative[-1] if cumulative else 0
        fig_accum.add_annotation(
            text=f"{final_count} especies detectadas",
            x=len(cumulative),
            y=final_count,
            showarrow=True,
            arrowhead=2,
            font=dict(size=12, color=COLORS.get("Accidente", "#D32F2F")),
        )

        fig_accum.add_annotation(
            text=f"{SOURCE_ANNOTATION} | n={len(df_sorted):,}",
            xref="paper", yref="paper", x=0, y=-0.1,
            showarrow=False, font=dict(size=10, color="gray"),
        )
        st.plotly_chart(fig_accum, use_container_width=True)

        # Evaluacion de saturacion
        if len(cumulative) > 20:
            last_quarter = cumulative[int(len(cumulative) * 0.75):]
            new_in_last_quarter = last_quarter[-1] - last_quarter[0]
            if new_in_last_quarter <= 2:
                st.success(
                    f"La curva se aproxima a la saturacion: solo {new_in_last_quarter} nuevas especies "
                    f"descubiertas en el ultimo 25% de los registros."
                )
            else:
                st.info(
                    f"{new_in_last_quarter} nuevas especies descubiertas en el ultimo 25% de los registros. "
                    f"Es posible que el muestreo aun no haya alcanzado la saturacion de especies."
                )
    else:
        st.info("No hay registros con fechas y especies validas para la curva de acumulacion.")
else:
    st.info("Las columnas de fecha o especie no estan disponibles para la curva de acumulacion.")

st.divider()

# =============================================================================
# SECCION 5: Mapa de Calor Especie x Linea (full-width)
# =============================================================================
st.subheader("5. Mapa de Calor Especie x Linea")
st.caption("Que especies mueren en que lineas — revela zonificacion ecologica y asociaciones de habitat.")

if "species_clean" in df.columns and "line_label" in df.columns:
    top15 = df["species_clean"].value_counts().head(15).index.tolist()
    df_top = df[df["species_clean"].isin(top15)]

    if len(df_top) > 0:
        heatmap_ct = pd.crosstab(df_top["species_clean"], df_top["line_label"])
        row_order = (
            df_top["species_clean"]
            .value_counts()
            .reindex(heatmap_ct.index)
            .sort_values(ascending=True)
            .index.tolist()
        )
        heatmap_ct = heatmap_ct.reindex(row_order)

        fig_heat = px.imshow(
            heatmap_ct.values,
            x=heatmap_ct.columns.tolist(),
            y=heatmap_ct.index.tolist(),
            color_continuous_scale="Viridis",
            template=CHART_TEMPLATE,
            height=max(CHART_H_DEFAULT, len(top15) * 35),
            labels=dict(x="Linea electrica", y="Especie", color="Victimas"),
            aspect="auto",
        )
        fig_heat.update_layout(
            margin=CHART_MARGINS,
            xaxis_title="",
            yaxis_title="",
            title_font_size=16,
        )
        # Anotaciones de texto
        for i, row_name in enumerate(heatmap_ct.index):
            for j, col_name in enumerate(heatmap_ct.columns):
                val = heatmap_ct.loc[row_name, col_name]
                if val > 0:
                    fig_heat.add_annotation(
                        x=j, y=i,
                        text=str(val),
                        showarrow=False,
                        font=dict(
                            color="white" if val > heatmap_ct.values.max() * 0.5 else "black",
                            size=10,
                        ),
                    )

        fig_heat.add_annotation(
            text=f"{SOURCE_ANNOTATION} | Top 15 especies | n={len(df_top):,}",
            xref="paper", yref="paper", x=0, y=-0.08,
            showarrow=False, font=dict(size=10, color="gray"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No hay datos para el mapa de calor especie x linea.")
else:
    st.info("Las columnas de especie o linea no estan disponibles para el mapa de calor.")

st.divider()

# =============================================================================
# SECCION 6: Analisis de Restos por Patron de Actividad (full-width charts)
# =============================================================================
st.subheader("6. Analisis de Restos por Patron de Actividad")
st.caption(
    "Sesgo de deteccion: las especies nocturnas producen patrones de restos diferentes? "
    "Las diferencias pueden indicar carroneo antes de las visitas de muestreo diurnas."
)

if "activity_pattern" in df.columns:
    df_activity = df[df["activity_pattern"].isin(["Nocturna", "Crepuscular", "Diurna"])].copy()

    if len(df_activity) > 0:
        # -- Tipo de restos por patron de actividad (full-width) ---------------
        st.markdown("**Tipo de restos por patron de actividad**")
        if "remains_type" in df_activity.columns:
            rt_df = (
                df_activity[df_activity["remains_type"].notna()]
                .groupby(["activity_pattern", "remains_type"])
                .size()
                .reset_index(name="count")
            )
            if len(rt_df) > 0:
                remains_type_colors = {
                    k: v for k, v in COLORS.items()
                    if k in rt_df["remains_type"].unique()
                }
                fig_rt = px.bar(
                    rt_df,
                    x="activity_pattern",
                    y="count",
                    color="remains_type",
                    barmode="stack",
                    template=CHART_TEMPLATE,
                    height=CHART_H_DEFAULT,
                    color_discrete_map=remains_type_colors if remains_type_colors else None,
                    labels={
                        "activity_pattern": "Patron de actividad",
                        "count": "Victimas",
                        "remains_type": "Tipo de restos",
                    },
                )
                fig_rt.update_layout(
                    margin=CHART_MARGINS,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    title_font_size=16,
                    hovermode="x unified",
                )
                fig_rt.add_annotation(
                    text=SOURCE_ANNOTATION,
                    xref="paper", yref="paper", x=0, y=-0.1,
                    showarrow=False, font=dict(size=10, color="gray"),
                )
                st.plotly_chart(fig_rt, use_container_width=True)
            else:
                st.info("No hay datos de tipo de restos disponibles.")
        else:
            st.info("La columna de tipo de restos no esta disponible.")

        st.divider()

        # -- Antiguedad de restos por patron de actividad (full-width) ---------
        st.markdown("**Antiguedad de restos por patron de actividad**")
        if "remains_age" in df_activity.columns:
            ra_df = (
                df_activity[df_activity["remains_age"].notna()]
                .groupby(["activity_pattern", "remains_age"])
                .size()
                .reset_index(name="count")
            )
            if len(ra_df) > 0:
                fig_ra = px.bar(
                    ra_df,
                    x="activity_pattern",
                    y="count",
                    color="remains_age",
                    barmode="stack",
                    template=CHART_TEMPLATE,
                    height=CHART_H_DEFAULT,
                    labels={
                        "activity_pattern": "Patron de actividad",
                        "count": "Victimas",
                        "remains_age": "Antiguedad de restos",
                    },
                )
                fig_ra.update_layout(
                    margin=CHART_MARGINS,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    title_font_size=16,
                    hovermode="x unified",
                )
                fig_ra.add_annotation(
                    text=SOURCE_ANNOTATION,
                    xref="paper", yref="paper", x=0, y=-0.1,
                    showarrow=False, font=dict(size=10, color="gray"),
                )
                st.plotly_chart(fig_ra, use_container_width=True)
            else:
                st.info("No hay datos de antiguedad de restos disponibles.")
        else:
            st.info("La columna de antiguedad de restos no esta disponible.")

        # -- Nota sobre carroneo
        if "scavenging" in df.columns:
            scav_positive = df["scavenging"].notna() & (df["scavenging"] != "No") & (df["scavenging"] != "")
            scav_count = scav_positive.sum()
            scav_pct = scav_count / total * 100 if total > 0 else 0
            st.caption(
                f"{scav_pct:.1f}% de los registros muestran evidencias de carroneo "
                f"({scav_count:,} registros). El carroneo puede eliminar o alterar los restos antes "
                f"de que lleguen los muestreadores, afectando especialmente la deteccion de victimas "
                f"nocturnas encontradas durante los muestreos diurnos."
            )
    else:
        st.info("No hay registros con patrones de actividad conocidos para el analisis de restos.")
else:
    st.info("La columna de patron de actividad no esta disponible para el analisis de restos.")
