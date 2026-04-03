"""
Pagina 4: Analisis Temporal
Deteccion de tendencias, patrones estacionales, curvas acumulativas y esfuerzo de muestreo.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COLORS, CHART_HEIGHT, CHART_TEMPLATE, SOURCE_ANNOTATION
from src.data_loader import load_data
from src.filters import apply_filters, get_filter_summary
from src.statistics import (
    mann_kendall_test, chi_squared_test, rayleigh_test,
    watson_u2_test, months_to_radians, stat_badge, format_p_value,
)

MONTH_NAMES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}

# -- Configuracion de pagina -------------------------------------------------
st.header("Analisis Temporal")
st.caption(
    "Tendencias, estacionalidad y esfuerzo de muestreo a lo largo del periodo de seguimiento. "
    "Las pruebas estadisticas identifican patrones significativos teniendo en cuenta el sesgo de anos parciales."
)

# -- Cargar y filtrar ---------------------------------------------------------
df_full = load_data()
df = apply_filters(df_full)
subtitle = get_filter_summary(df, df_full)

if len(df) == 0:
    st.warning("Ningun registro coincide con los filtros actuales. Ajuste los filtros para ver resultados.")
    st.stop()

n_total = len(df)

st.divider()

# =============================================================================
# SECCION 1: Tendencia Anual
# =============================================================================
st.subheader("1. Tendencia Anual")
st.caption(
    "Esta cambiando la mortalidad con el tiempo? "
    "Conteos anuales de victimas con prueba de tendencia Mann-Kendall y estimador de pendiente de Sen. "
    "Los anos parciales (primero y ultimo) se marcan con asterisco."
)

year_counts = df.groupby("year").size().reset_index(name="count")
year_counts["year"] = year_counts["year"].astype(int)

min_year = int(year_counts["year"].min())
max_year = int(year_counts["year"].max())
year_counts["partial"] = year_counts["year"].isin([min_year, max_year])

# Mann-Kendall solo sobre anos completos
full_years = year_counts[~year_counts["partial"]]
mk_series = year_counts["count"].values  # todos los anos para ajuste de linea de tendencia
mk_result = mann_kendall_test(full_years["count"].values)

col_trend, col_stat = st.columns([3, 1])

with col_trend:
    fig_trend = go.Figure()

    # Barras: anos parciales con color atenuado
    fig_trend.add_trace(go.Bar(
        x=year_counts["year"],
        y=year_counts["count"],
        marker_color=[
            "#BDBDBD" if p else COLORS["Accidente"]
            for p in year_counts["partial"]
        ],
        text=[
            f"{c}{'*' if p else ''}"
            for c, p in zip(year_counts["count"], year_counts["partial"])
        ],
        textposition="outside",
        hovertemplate="Ano: %{x}<br>Victimas: %{y}<extra></extra>",
        name="Conteo anual",
    ))

    # Linea de tendencia de pendiente de Sen (ajustada sobre anos completos, dibujada en todos)
    if not np.isnan(mk_result["slope"]) and len(full_years) >= 3:
        first_full_year = int(full_years["year"].iloc[0])
        trend_x = year_counts["year"].values
        trend_y = mk_result["intercept"] + mk_result["slope"] * (trend_x - first_full_year)

        fig_trend.add_trace(go.Scatter(
            x=trend_x,
            y=trend_y,
            mode="lines",
            line=dict(color="rgba(0,0,0,0.4)", dash="dash", width=2),
            name=f"Tendencia (anos completos) ({mk_result['slope']:+.1f}/ano)",
            hoverinfo="skip",
        ))

    fig_trend.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=20, t=40, b=60),
        xaxis_title="Ano",
        yaxis_title="Victimas",
        xaxis=dict(dtick=1),
    )
    fig_trend.add_annotation(
        text=f"*Ano parcial | {SOURCE_ANNOTATION} | n={n_total:,}",
        xref="paper", yref="paper", x=0, y=-0.18,
        showarrow=False, font=dict(size=10, color="gray"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_stat:
    badge_mk = stat_badge(
        "Mann-Kendall",
        mk_result["tau"],
        mk_result["p_value"],
        n=len(full_years),
    )
    st.markdown(badge_mk, unsafe_allow_html=True)

    st.markdown(f"""
    - **tau** = {mk_result['tau']:.3f}
    - **{format_p_value(mk_result['p_value'])}**
    - **Pendiente de Sen** = {mk_result['slope']:+.2f} victimas/ano
    """)

    st.caption(
        "Prueba de tendencia calculada solo sobre anos completos (anos parciales primero/ultimo excluidos). "
        "La pendiente de Sen es un estimador de tendencia robusto no parametrico."
    )

st.divider()

# =============================================================================
# SECCION 2: Estacionalidad Mensual (Grafico Polar)
# =============================================================================
st.subheader("2. Estacionalidad Mensual")
st.caption("Cuando son mas frecuentes las colisiones? Grafico polar de la distribucion mensual de victimas con prueba de Rayleigh para no uniformidad.")

month_counts = (
    df.groupby("month").size()
    .reindex(range(1, 13), fill_value=0)
    .reset_index()
)
month_counts.columns = ["month", "count"]
month_counts["month_name"] = month_counts["month"].map(MONTH_NAMES)

# Angulos de barras polares: enero arriba (90 grados), sentido horario
month_counts["theta"] = 90 - (month_counts["month"] - 1) * 30

col_polar, col_rayleigh = st.columns([3, 1])

with col_polar:
    max_count = month_counts["count"].max() if month_counts["count"].max() > 0 else 1

    fig_polar = go.Figure()
    fig_polar.add_trace(go.Barpolar(
        r=month_counts["count"],
        theta=month_counts["theta"],
        width=28,
        marker=dict(
            color=month_counts["count"],
            colorscale="YlOrRd",
            cmin=0,
            cmax=max_count,
            colorbar=dict(title="Conteo", len=0.6),
            line=dict(color="white", width=1),
        ),
        text=[f"{name}: {c}" for name, c in zip(month_counts["month_name"], month_counts["count"])],
        hovertemplate="%{text}<extra></extra>",
    ))

    fig_polar.update_layout(
        template=CHART_TEMPLATE,
        height=500,
        polar=dict(
            angularaxis=dict(
                tickmode="array",
                tickvals=[90 - i * 30 for i in range(12)],
                ticktext=[MONTH_NAMES[m] for m in range(1, 13)],
                direction="clockwise",
                rotation=90,
            ),
            radialaxis=dict(
                showticklabels=True,
                tickfont=dict(size=9),
            ),
        ),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_polar, use_container_width=True)

with col_rayleigh:
    # Prueba de Rayleigh: crear array de angulos repitiendo el radian de cada mes por su conteo
    angles_list = []
    for _, row in month_counts.iterrows():
        if row["count"] > 0:
            month_rad = months_to_radians(pd.Series([row["month"]]))[0]
            angles_list.extend([month_rad] * int(row["count"]))

    if len(angles_list) > 0:
        angles_arr = np.array(angles_list)
        ray_result = rayleigh_test(angles_arr)

        badge_ray = stat_badge(
            "Rayleigh",
            ray_result["Z"],
            ray_result["p_value"],
            n=len(angles_arr),
        )
        st.markdown(badge_ray, unsafe_allow_html=True)

        # Convertir direccion media a mes
        mean_month = (ray_result["mean_direction"] / (2 * np.pi)) * 12 + 1
        mean_month_name = MONTH_NAMES.get(int(round(mean_month)) % 12 or 12, "?")

        st.markdown(f"""
        - **Z** = {ray_result['Z']:.2f}
        - **{format_p_value(ray_result['p_value'])}**
        - **Direccion media** = {mean_month_name} ({np.degrees(ray_result['mean_direction']):.0f} grados)
        - **R-bar** = {ray_result['mean_resultant_length']:.3f}
        """)

        if ray_result["p_value"] < 0.05:
            st.success("Se detecta no uniformidad significativa: las victimas se concentran en meses especificos.")
        else:
            st.info("No se detecta concentracion estacional significativa a alfa = 0,05.")
    else:
        st.info("Datos insuficientes para la prueba de Rayleigh.")

st.divider()

# =============================================================================
# SECCION 3: Mapa de Calor Ano x Mes
# =============================================================================
st.subheader("3. Mapa de Calor Ano x Mes")
st.caption("Son consistentes los patrones estacionales entre anos? Conteos mensuales de victimas por ano. Las celdas mas oscuras indican mas victimas.")

heatmap_data = (
    df.groupby(["year", "month"]).size()
    .reset_index(name="count")
)
heatmap_data["year"] = heatmap_data["year"].astype(int)
heatmap_data["month"] = heatmap_data["month"].astype(int)

# Construir cuadricula completa (todas las combinaciones ano-mes)
all_years = sorted(heatmap_data["year"].unique())
all_months = list(range(1, 13))

pivot = heatmap_data.pivot_table(
    index="year", columns="month", values="count", fill_value=0, aggfunc="sum"
)
# Asegurar que todos los meses estan presentes
for m in all_months:
    if m not in pivot.columns:
        pivot[m] = 0
pivot = pivot[all_months]
pivot = pivot.sort_index()

fig_heat = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=[MONTH_NAMES[m] for m in all_months],
    y=[str(y) for y in pivot.index],
    colorscale="YlOrRd",
    text=pivot.values,
    texttemplate="%{text}",
    textfont=dict(size=12),
    hovertemplate="Ano: %{y}<br>Mes: %{x}<br>Conteo: %{z}<extra></extra>",
    colorbar=dict(title="Conteo"),
))

fig_heat.update_layout(
    template=CHART_TEMPLATE,
    height=max(300, len(all_years) * 50),
    xaxis_title="Mes",
    yaxis_title="Ano",
    yaxis=dict(autorange="reversed", dtick=1),
    margin=dict(l=0, r=20, t=10, b=40),
)
fig_heat.add_annotation(
    text=f"{SOURCE_ANNOTATION} | n={n_total:,}",
    xref="paper", yref="paper", x=0, y=-0.12,
    showarrow=False, font=dict(size=10, color="gray"),
)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# =============================================================================
# SECCION 4: Curvas Acumulativas por Ano
# =============================================================================
st.subheader("4. Curvas Acumulativas por Ano")
st.caption(
    "Victimas acumuladas a lo largo del ano calendario. "
    "Las pendientes mas pronunciadas indican periodos de mayor acumulacion de mortalidad."
)

# Construir conteos acumulativos por ano y mes
cum_data = (
    df.groupby(["year", "month"]).size()
    .reset_index(name="count")
)
cum_data["year"] = cum_data["year"].astype(int)
cum_data["month"] = cum_data["month"].astype(int)

# Para cada ano, completar meses faltantes con 0 y calcular suma acumulada
cum_records = []
for yr in sorted(cum_data["year"].unique()):
    yr_data = cum_data[cum_data["year"] == yr].set_index("month")["count"]
    yr_data = yr_data.reindex(range(1, 13), fill_value=0)
    cumsum = yr_data.cumsum()
    for m, val in cumsum.items():
        cum_records.append({"year": yr, "month": m, "cumulative": val})

cum_df = pd.DataFrame(cum_records)
cum_df["month_name"] = cum_df["month"].map(MONTH_NAMES)
cum_df["year_str"] = cum_df["year"].astype(str)

# Paleta de colores: un color por ano
years_list = sorted(cum_df["year"].unique())
year_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set1
year_color_map = {str(y): year_palette[i % len(year_palette)] for i, y in enumerate(years_list)}

fig_cum = px.line(
    cum_df,
    x="month",
    y="cumulative",
    color="year_str",
    color_discrete_map=year_color_map,
    markers=True,
    labels={"month": "Mes", "cumulative": "Victimas acumuladas", "year_str": "Ano"},
    template=CHART_TEMPLATE,
    height=CHART_HEIGHT,
)
fig_cum.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=[MONTH_NAMES[m] for m in range(1, 13)],
    ),
    margin=dict(l=0, r=20, t=10, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig_cum.add_annotation(
    text=f"{SOURCE_ANNOTATION} | n={n_total:,}",
    xref="paper", yref="paper", x=0, y=-0.12,
    showarrow=False, font=dict(size=10, color="gray"),
)
st.plotly_chart(fig_cum, use_container_width=True)

st.divider()

# =============================================================================
# SECCION 5: Cronologia del Esfuerzo de Muestreo
# =============================================================================
st.subheader("5. Cronologia del Esfuerzo de Muestreo")
st.caption(
    "Cada punto representa una fecha unica de muestreo. Los vacios en el seguimiento son periodos "
    "sin muestreos registrados; las victimas durante esas ventanas pueden pasar desapercibidas."
)

if "date" in df.columns and "study" in df.columns:
    # Fechas unicas de muestreo por estudio
    survey_dates = df.dropna(subset=["date"]).groupby("study")["date"].apply(
        lambda x: x.dt.date.unique()
    ).explode().reset_index()
    survey_dates.columns = ["study", "date"]
    survey_dates["date"] = pd.to_datetime(survey_dates["date"])
    survey_dates["y"] = 1  # linea plana

    fig_effort = px.scatter(
        survey_dates,
        x="date",
        y="y",
        color="study",
        labels={"date": "Fecha", "y": "", "study": "Estudio"},
        template=CHART_TEMPLATE,
        height=250,
    )
    fig_effort.update_traces(marker=dict(size=6, opacity=0.7))
    fig_effort.update_layout(
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0.5, 1.5]),
        margin=dict(l=0, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_effort.add_annotation(
        text=SOURCE_ANNOTATION,
        xref="paper", yref="paper", x=0, y=-0.25,
        showarrow=False, font=dict(size=10, color="gray"),
    )
    st.plotly_chart(fig_effort, use_container_width=True)

    st.caption(
        "El esfuerzo de muestreo no es uniforme en el tiempo ni entre zonas de estudio. "
        "Los periodos sin puntos pueden reflejar vacios de seguimiento en lugar de mortalidad cero. "
        "Esto debe considerarse al interpretar las tendencias anuales y estacionales anteriores."
    )
elif "date" in df.columns:
    # Alternativa: sin columna de estudio
    survey_dates = df.dropna(subset=["date"])["date"].dt.date.unique()
    survey_dates = pd.DataFrame({"date": pd.to_datetime(survey_dates), "y": 1})

    fig_effort = px.scatter(
        survey_dates,
        x="date",
        y="y",
        template=CHART_TEMPLATE,
        height=200,
        labels={"date": "Fecha", "y": ""},
    )
    fig_effort.update_traces(marker=dict(size=6, color=COLORS["Accidente"], opacity=0.7))
    fig_effort.update_layout(
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        margin=dict(l=0, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_effort, use_container_width=True)
else:
    st.info("Columna de fecha no disponible para la visualizacion de esfuerzo de muestreo.")

st.divider()

# =============================================================================
# SECCION 6: Estacionalidad: Nocturnas vs Diurnas
# =============================================================================
st.subheader("6. Estacionalidad: Especies Nocturnas vs Diurnas")
st.caption(
    "Comparacion de la distribucion mensual entre especies nocturnas y diurnas. "
    "La zona sombreada marca la temporada de cria de pardelas (Calonectris) (marzo-octubre)."
)

has_activity = "activity_pattern" in df.columns

if not has_activity:
    st.info("Columna de patron de actividad no disponible. Esta seccion requiere datos de clasificacion de especies.")
else:
    df_nd = df[df["activity_pattern"].isin(["Nocturna", "Diurna"])].copy()

    if len(df_nd) == 0:
        st.warning("No hay registros de especies nocturnas o diurnas en los datos filtrados.")
    else:
        # Conteos mensuales por patron de actividad
        noct_monthly = (
            df_nd[df_nd["activity_pattern"] == "Nocturna"]
            .groupby("month").size()
            .reindex(range(1, 13), fill_value=0)
        )
        diur_monthly = (
            df_nd[df_nd["activity_pattern"] == "Diurna"]
            .groupby("month").size()
            .reindex(range(1, 13), fill_value=0)
        )

        col_chart6, col_stat6 = st.columns([3, 1])

        with col_chart6:
            fig_nd = go.Figure()

            # Sombreado de temporada de cria de pardelas (marzo=3 a octubre=10)
            fig_nd.add_vrect(
                x0=2.5, x1=10.5,
                fillcolor="rgba(255, 193, 7, 0.12)",
                line_width=0,
                annotation_text="Temporada de cria de pardelas",
                annotation_position="top left",
                annotation_font=dict(size=10, color="#F9A825"),
            )

            # Linea nocturnas
            fig_nd.add_trace(go.Scatter(
                x=list(range(1, 13)),
                y=noct_monthly.values,
                mode="lines+markers",
                name="Nocturna",
                line=dict(color=COLORS["Nocturna"], width=3),
                marker=dict(size=8),
                hovertemplate="Mes: %{x}<br>Nocturnas: %{y}<extra></extra>",
            ))

            # Linea diurnas
            fig_nd.add_trace(go.Scatter(
                x=list(range(1, 13)),
                y=diur_monthly.values,
                mode="lines+markers",
                name="Diurna",
                line=dict(color=COLORS["Diurna"], width=3),
                marker=dict(size=8),
                hovertemplate="Mes: %{x}<br>Diurnas: %{y}<extra></extra>",
            ))

            fig_nd.update_layout(
                template=CHART_TEMPLATE,
                height=CHART_HEIGHT,
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(1, 13)),
                    ticktext=[MONTH_NAMES[m] for m in range(1, 13)],
                    title="Mes",
                ),
                yaxis_title="Victimas",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=20, t=40, b=60),
            )
            fig_nd.add_annotation(
                text=f"{SOURCE_ANNOTATION} | n={len(df_nd):,}",
                xref="paper", yref="paper", x=0, y=-0.18,
                showarrow=False, font=dict(size=10, color="gray"),
            )
            st.plotly_chart(fig_nd, use_container_width=True)

        with col_stat6:
            # Prueba circular Watson U2
            noct_angles = []
            for m in range(1, 13):
                c = int(noct_monthly.get(m, 0))
                if c > 0:
                    rad = months_to_radians(pd.Series([m]))[0]
                    noct_angles.extend([rad] * c)

            diur_angles = []
            for m in range(1, 13):
                c = int(diur_monthly.get(m, 0))
                if c > 0:
                    rad = months_to_radians(pd.Series([m]))[0]
                    diur_angles.extend([rad] * c)

            if len(noct_angles) >= 2 and len(diur_angles) >= 2:
                watson_result = watson_u2_test(
                    np.array(noct_angles),
                    np.array(diur_angles),
                )

                badge_watson = stat_badge(
                    "Watson U\u00B2",
                    watson_result["U2"],
                    watson_result["p_value"],
                    n=len(noct_angles) + len(diur_angles),
                )
                st.markdown(badge_watson, unsafe_allow_html=True)

                st.markdown(f"""
                - **U\u00B2** = {watson_result['U2']:.4f}
                - **{format_p_value(watson_result['p_value'])}**
                - Nocturnas n = {len(noct_angles):,}
                - Diurnas n = {len(diur_angles):,}
                """)

                if watson_result["p_value"] < 0.05:
                    st.info(
                        "Los patrones estacionales difieren significativamente entre especies nocturnas y diurnas. "
                        "Las muertes de nocturnas pueden alcanzar su pico durante la cria/emancipacion de pardelas, "
                        "mientras que las diurnas siguen ciclos migratorios o de alimentacion diferentes."
                    )
                else:
                    st.success(
                        "No se detecta diferencia significativa en la distribucion estacional entre especies "
                        "nocturnas y diurnas. Ambos grupos siguen patrones mensuales similares."
                    )
            else:
                st.info("Datos insuficientes para la prueba Watson U\u00B2 (se necesitan al menos 2 observaciones por grupo).")

        # Suplementario: tabla de desglose mensual
        with st.expander("Tabla de desglose mensual"):
            table_df = pd.DataFrame({
                "Mes": [MONTH_NAMES[m] for m in range(1, 13)],
                "Nocturnas": noct_monthly.values,
                "Diurnas": diur_monthly.values,
                "Total": noct_monthly.values + diur_monthly.values,
            })
            st.dataframe(table_df, use_container_width=True, hide_index=True)
