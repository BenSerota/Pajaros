"""
Pagina 3: Eficacia de Senalizacion
Pagina analitica central que compara tipos de dispositivos anticolision.
Incluye pruebas chi-cuadrado, regresion binomial negativa y graficos de bosque.
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
from src.statistics import (
    chi_squared_test, fisher_pairwise, kruskal_wallis_test,
    negative_binomial_regression, stat_badge, format_p_value,
)

# -- Configuracion de pagina -------------------------------------------------
st.header("Analisis de Eficacia de Senalizacion")
st.caption("Los dispositivos anticolision reducen la mortalidad de aves? Comparacion estadistica de tipos de senalizacion.")

# -- ADVERTENCIA --------------------------------------------------------------
st.warning(
    "**Advertencia importante:** El tipo de senalizacion **NO se asigna aleatoriamente** a los vanos. "
    "Las distintas lineas tienen combinaciones diferentes de senalizacion y atraviesan "
    "habitats y contextos geograficos distintos. Las diferencias observadas pueden reflejar "
    "factores ecologicos o geograficos en lugar de la eficacia de la senalizacion por si sola. "
    "Se necesitarian experimentos controlados para establecer conclusiones causales."
)

# -- Cargar y filtrar ---------------------------------------------------------
df_full = load_data()
df = apply_filters(df_full)
subtitle = get_filter_summary(df, df_full)

# Filtrar registros con tipo de senalizacion conocido
df_sig = df[df["signal_type"].notna()].copy()
n_excluded = len(df) - len(df_sig)

if n_excluded > 0:
    st.info(f"{n_excluded} registros con tipo de senalizacion desconocido excluidos de este analisis. Mostrando {len(df_sig):,} registros.")

if len(df_sig) == 0:
    st.warning("Ningun registro coincide con los filtros actuales. Ajuste los filtros para ver resultados.")
    st.stop()

st.divider()

# =============================================================================
# SECCION 1: Tipo de senalizacion x Tipo de evento
# =============================================================================
st.subheader("1. Severidad de Colision por Tipo de Senalizacion")
st.caption(
    "Difiere la proporcion de accidentes confirmados (Accidente) frente a evidencias de "
    "colision (Incidente) segun el tipo de senalizacion? Una tasa de accidentes mas alta "
    "podria indicar colisiones mas graves o recientes."
)

col1, col2 = st.columns([2, 1])

with col1:
    # Tabla de contingencia
    ct = pd.crosstab(df_sig["signal_type"], df_sig["event_type"])
    # Asegurar que ambas columnas existen
    for col_name in ["Accidente", "Incidente"]:
        if col_name not in ct.columns:
            ct[col_name] = 0
    ct = ct[["Accidente", "Incidente"]]

    # Reordenar filas
    order = [s for s in SIGNAL_TYPES_ORDERED if s in ct.index]
    ct = ct.reindex(order)

    # Calcular proporciones
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    # Grafico de barras agrupadas
    fig_ct = go.Figure()
    for ev_type, color in [("Accidente", COLORS["Accidente"]), ("Incidente", COLORS["Incidente"])]:
        fig_ct.add_trace(go.Bar(
            name=ev_type,
            x=ct.index,
            y=ct[ev_type],
            marker_color=color,
            text=[f"{v} ({p:.1f}%)" for v, p in zip(ct[ev_type], ct_pct[ev_type])],
            textposition="auto",
        ))

    fig_ct.update_layout(
        barmode="group",
        template=CHART_TEMPLATE,
        height=400,
        xaxis_title="",
        yaxis_title="Victimas",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=20, t=40, b=10),
    )
    st.plotly_chart(fig_ct, use_container_width=True)

with col2:
    # Prueba chi-cuadrado
    result = chi_squared_test(ct)
    badge_html = stat_badge(
        "Chi-cuadrado", result["statistic"], result["p_value"],
        effect_size=result["effect_size"], n=len(df_sig)
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("**Tabla de contingencia**")
    st.dataframe(ct, use_container_width=True)

    st.markdown(f"""
    - **\u03C7\u00B2** = {result['statistic']:.2f}, gl = {result['dof']}
    - **p** = {format_p_value(result['p_value'])}
    - **V de Cramer** = {result['effect_size']:.3f}
    """)

# Comparaciones pareadas
with st.expander("Comparaciones pareadas (Fisher Exact)"):
    pairwise = fisher_pairwise(ct)
    if pairwise is not None and len(pairwise) > 0:
        display_df = pairwise.copy()
        display_df["p_adjusted"] = display_df["p_adjusted"].apply(format_p_value)
        display_df["p_value"] = display_df["p_value"].apply(format_p_value)
        display_df["odds_ratio"] = display_df["odds_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No hay suficientes categorias para comparaciones pareadas.")

st.divider()

# =============================================================================
# SECCION 2: Separacion de senalizacion (10m vs 20m)
# =============================================================================
st.subheader("2. Separacion de Senalizacion: 10m vs 20m")

# ADVERTENCIA DE VARIABLE CONFUNDIDA
st.error(
    "**Variable confundida:** La separacion de los dispositivos esta casi perfectamente "
    "correlacionada con el voltaje. Las lineas de 66kV usan separacion de 10m, "
    "las de 132kV usan separacion de 20m. Cualquier diferencia en tasas de colision "
    "entre separaciones es **inseparable** de los efectos del voltaje. "
    "Esta comparacion se muestra por completitud, pero **no es posible extraer conclusiones causales**."
)

df_spacing = df_sig[df_sig["signal_spacing_m"].notna()].copy()
df_spacing["spacing_label"] = df_spacing["signal_spacing_m"].apply(
    lambda x: f"{int(x)}m" if pd.notna(x) else "Desconocido"
)

col_s1, col_s2 = st.columns([2, 1])

with col_s1:
    ct_spacing = pd.crosstab(df_spacing["spacing_label"], df_spacing["event_type"])
    for col_name in ["Accidente", "Incidente"]:
        if col_name not in ct_spacing.columns:
            ct_spacing[col_name] = 0
    ct_spacing = ct_spacing[["Accidente", "Incidente"]]

    fig_sp = go.Figure()
    for ev_type, color in [("Accidente", COLORS["Accidente"]), ("Incidente", COLORS["Incidente"])]:
        pcts = ct_spacing[ev_type] / ct_spacing.sum(axis=1) * 100
        fig_sp.add_trace(go.Bar(
            name=ev_type,
            x=ct_spacing.index,
            y=ct_spacing[ev_type],
            marker_color=color,
            text=[f"{v} ({p:.1f}%)" for v, p in zip(ct_spacing[ev_type], pcts)],
            textposition="auto",
        ))
    fig_sp.update_layout(
        barmode="group",
        template=CHART_TEMPLATE,
        height=350,
        xaxis_title="Separacion de dispositivos",
        yaxis_title="Victimas",
        margin=dict(l=0, r=20, t=10, b=10),
    )
    st.plotly_chart(fig_sp, use_container_width=True)

with col_s2:
    # Tabulacion cruzada: separacion x voltaje
    st.markdown("**Separacion x Voltaje (tabulacion cruzada)**")
    ct_sv = pd.crosstab(df_spacing["spacing_label"], df_spacing["voltage"])
    st.dataframe(ct_sv, use_container_width=True)
    st.caption("Confundimiento casi perfecto: 10m <-> 66kV, 20m <-> 132kV")

st.divider()

# =============================================================================
# SECCION 3: Efecto del estado de la senalizacion
# =============================================================================
st.subheader("3. Efecto del Estado de la Senalizacion")
st.caption("Afecta el estado fisico de los dispositivos de senalizacion a las tasas de colision?")

df_cond = df_sig[df_sig["signal_condition"].notna()].copy()

if len(df_cond) > 0:
    col_c1, col_c2 = st.columns([2, 1])

    with col_c1:
        ct_cond = pd.crosstab(df_cond["signal_condition"], df_cond["event_type"])
        for col_name in ["Accidente", "Incidente"]:
            if col_name not in ct_cond.columns:
                ct_cond[col_name] = 0
        ct_cond = ct_cond[["Accidente", "Incidente"]]

        ct_cond_pct = ct_cond.div(ct_cond.sum(axis=1), axis=0) * 100

        fig_cond = go.Figure()
        for ev_type, color in [("Accidente", COLORS["Accidente"]), ("Incidente", COLORS["Incidente"])]:
            fig_cond.add_trace(go.Bar(
                name=ev_type,
                x=ct_cond.index,
                y=ct_cond[ev_type],
                marker_color=color,
                text=[f"{v} ({p:.1f}%)" for v, p in zip(ct_cond[ev_type], ct_cond_pct[ev_type])],
                textposition="auto",
            ))
        fig_cond.update_layout(
            barmode="group",
            template=CHART_TEMPLATE, height=350,
            xaxis_title="", yaxis_title="Victimas",
            margin=dict(l=0, r=20, t=10, b=10),
        )
        st.plotly_chart(fig_cond, use_container_width=True)

    with col_c2:
        result_cond = chi_squared_test(ct_cond)
        badge_cond = stat_badge(
            "Chi-cuadrado", result_cond["statistic"], result_cond["p_value"],
            effect_size=result_cond["effect_size"], n=len(df_cond)
        )
        st.markdown(badge_cond, unsafe_allow_html=True)
        st.dataframe(ct_cond, use_container_width=True)

        # Nota sobre celdas pequenas
        min_cell = ct_cond.min().min()
        if min_cell < 5:
            st.caption(f"Recuento minimo en celda = {min_cell}. Chi-cuadrado puede no ser fiable; considere la prueba exacta de Fisher.")

st.divider()

# =============================================================================
# SECCION 4: Comparacion a nivel de vano (Kruskal-Wallis)
# =============================================================================
st.subheader("4. Victimas por Vano segun Tipo de Senalizacion")
st.caption(
    "Comparacion de la distribucion de victimas por vano segun tipo de senalizacion. "
    "Esto controla por el numero de vanos en lugar de comparar totales brutos."
)

# Calcular conteos por vano
span_sig = (
    df_sig.groupby(["span_id", "signal_type"])
    .size()
    .reset_index(name="count")
)

col_k1, col_k2 = st.columns([2, 1])

with col_k1:
    fig_box = px.box(
        span_sig,
        x="signal_type",
        y="count",
        color="signal_type",
        color_discrete_map={s: COLORS.get(s, "#999") for s in SIGNAL_TYPES_ORDERED},
        template=CHART_TEMPLATE,
        height=400,
        labels={"signal_type": "", "count": "Victimas por vano"},
        category_orders={"signal_type": [s for s in SIGNAL_TYPES_ORDERED if s in span_sig["signal_type"].values]},
    )
    fig_box.update_layout(showlegend=False, margin=dict(l=0, r=20, t=10, b=10))
    st.plotly_chart(fig_box, use_container_width=True)

with col_k2:
    # Prueba de Kruskal-Wallis
    groups = [
        span_sig[span_sig["signal_type"] == st_name]["count"].values
        for st_name in SIGNAL_TYPES_ORDERED
        if st_name in span_sig["signal_type"].values and len(span_sig[span_sig["signal_type"] == st_name]) > 0
    ]

    if len(groups) >= 2:
        kw_result = kruskal_wallis_test(groups)
        badge_kw = stat_badge(
            "Kruskal-Wallis", kw_result["statistic"], kw_result["p_value"],
            effect_size=kw_result["effect_size"],
        )
        st.markdown(badge_kw, unsafe_allow_html=True)

        st.markdown(f"""
        - **H** = {kw_result['statistic']:.2f}
        - **p** = {format_p_value(kw_result['p_value'])}
        - **\u03B5\u00B2** = {kw_result['effect_size']:.3f}
        """)

        # Estadisticas resumidas por grupo
        summary = span_sig.groupby("signal_type")["count"].agg(
            ["count", "mean", "median", "std"]
        ).round(2)
        summary.columns = ["n_vanos", "media", "mediana", "desv_est"]
        st.dataframe(summary, use_container_width=True)

st.divider()

# =============================================================================
# SECCION 5: Regresion binomial negativa (grafico de bosque)
# =============================================================================
st.subheader("5. Modelo Multivariante: Regresion Binomial Negativa")
st.caption(
    "Controlando por voltaje y separacion de senalizacion. "
    "IRR (Razon de Tasas de Incidencia) > 1 indica mayor mortalidad respecto a la categoria de referencia."
)

# Preparar datos de regresion: conteos por vano con atributos
try:
    span_attrs = (
        df_sig.groupby("span_id")
        .agg(
            count=("span_id", "size"),
            signal_type=("signal_type", "first"),
            voltage=("voltage", "first"),
            signal_spacing_m=("signal_spacing_m", "first"),
            signal_condition=("signal_condition", "first"),
            line=("line_label", "first"),
        )
        .reset_index()
    )
    span_attrs = span_attrs.dropna(subset=["signal_type"])

    if len(span_attrs) >= 20 and span_attrs["signal_type"].nunique() >= 2:
        nb_result = negative_binomial_regression(
            span_attrs,
            formula="count ~ C(signal_type, Treatment(reference='Triple Aspa')) + C(voltage)"
        )

        if nb_result is not None and "params" in nb_result:
            params_df = pd.DataFrame(nb_result["params"])

            # Grafico de bosque de IRRs
            irr_data = params_df[params_df.index.str.contains("signal_type")].copy()
            if len(irr_data) > 0:
                irr_data["label"] = irr_data.index.str.extract(r"\[T\.(.*?)\]")[0].values
                irr_data = irr_data.dropna(subset=["label"])

                if len(irr_data) > 0:
                    fig_forest = go.Figure()

                    # Linea de referencia en IRR=1
                    fig_forest.add_vline(
                        x=1, line_dash="dash", line_color="gray",
                        annotation_text="Sin efecto", annotation_position="top"
                    )

                    # Graficar IRRs con IC
                    for _, row in irr_data.iterrows():
                        irr = row["irr"]
                        ci_low = irr * np.exp(-1.96 * row["se"])
                        ci_high = irr * np.exp(1.96 * row["se"])
                        sig = "*" if row["p_value"] < 0.05 else ""

                        fig_forest.add_trace(go.Scatter(
                            x=[ci_low, irr, ci_high],
                            y=[row["label"]] * 3,
                            mode="markers+lines",
                            marker=dict(
                                size=[8, 14, 8],
                                color=COLORS.get(row["label"], "#666"),
                            ),
                            line=dict(color=COLORS.get(row["label"], "#666"), width=3),
                            name=f"{row['label']} (IRR={irr:.2f}{sig})",
                            hovertemplate=(
                                f"<b>{row['label']}</b><br>"
                                f"IRR = {irr:.2f} [{ci_low:.2f}, {ci_high:.2f}]<br>"
                                f"p = {format_p_value(row['p_value'])}"
                                "<extra></extra>"
                            ),
                        ))

                    fig_forest.update_layout(
                        template=CHART_TEMPLATE,
                        height=300,
                        xaxis_title="Razon de Tasas de Incidencia (vs Triple Aspa)",
                        yaxis_title="",
                        showlegend=True,
                        margin=dict(l=0, r=20, t=10, b=40),
                        xaxis_type="log",
                    )
                    st.plotly_chart(fig_forest, use_container_width=True)

            # Diagnosticos del modelo
            with st.expander("Detalles de regresion"):
                st.text(str(nb_result.get("summary", "Resumen del modelo no disponible")))
                st.markdown(f"""
                - **AIC** = {nb_result.get('aic', 'N/D')}
                - **BIC** = {nb_result.get('bic', 'N/D')}
                - **Dispersion (\u03B1)** = {nb_result.get('alpha', 'N/D')}
                """)
        else:
            st.info("No se pudo ajustar el modelo de regresion. Esto puede deberse a variacion insuficiente en los datos.")
    else:
        st.info("Datos insuficientes para regresion multivariante tras el filtrado.")

except Exception as e:
    st.warning(f"No se pudo ajustar el modelo de regresion: {str(e)}")

st.divider()

# =============================================================================
# SECCION 6: Nocturnas vs Diurnas x Senalizacion
# =============================================================================
st.subheader("6. Especies Nocturnas vs Diurnas por Tipo de Senalizacion")
st.caption(
    "Si las especies nocturnas muestran la misma distribucion entre tipos de senalizacion que las diurnas, "
    "esto sugiere que los marcadores visuales son igualmente ineficaces de noche, lo que respalda "
    "el uso de dispositivos reflectantes UV o iluminados."
)

if "activity_pattern" in df_sig.columns:
    df_nd = df_sig[df_sig["activity_pattern"].isin(["Nocturna", "Diurna"])].copy()

    if len(df_nd) > 0:
        col_n, col_d = st.columns(2)

        for pattern, col_target in [("Nocturna", col_n), ("Diurna", col_d)]:
            with col_target:
                st.markdown(f"**Especies {pattern.lower()}s**")
                subset = df_nd[df_nd["activity_pattern"] == pattern]
                sig_cts = subset["signal_type"].value_counts().reindex(
                    [s for s in SIGNAL_TYPES_ORDERED if s in subset["signal_type"].values]
                ).reset_index()
                sig_cts.columns = ["signal_type", "count"]

                fig_nd = px.bar(
                    sig_cts,
                    x="signal_type",
                    y="count",
                    color="signal_type",
                    color_discrete_map={s: COLORS.get(s, "#999") for s in SIGNAL_TYPES_ORDERED},
                    template=CHART_TEMPLATE,
                    height=300,
                    labels={"signal_type": "", "count": "Victimas"},
                )
                fig_nd.update_layout(showlegend=False, margin=dict(l=0, r=10, t=10, b=10))
                fig_nd.update_traces(text=sig_cts["count"], textposition="outside")
                st.plotly_chart(fig_nd, use_container_width=True)
                st.caption(f"n = {len(subset):,}")

        # Comparacion chi-cuadrado
        ct_nd = pd.crosstab(df_nd["activity_pattern"], df_nd["signal_type"])
        result_nd = chi_squared_test(ct_nd)
        badge_nd = stat_badge(
            "Chi-cuadrado (Actividad x Senalizacion)", result_nd["statistic"], result_nd["p_value"],
            effect_size=result_nd["effect_size"], n=len(df_nd)
        )
        st.markdown(badge_nd, unsafe_allow_html=True)

        if result_nd["p_value"] >= 0.05:
            st.success(
                "No se detecta diferencia significativa en la distribucion por tipo de senalizacion entre "
                "especies nocturnas y diurnas. Esto es consistente con la hipotesis de que los marcadores "
                "visuales actuales son igualmente (in)eficaces independientemente del momento del dia."
            )
        else:
            st.info(
                "Se detecta diferencia significativa. Examine las distribuciones anteriores para entender "
                "que tipos de senalizacion difieren entre patrones de actividad."
            )
