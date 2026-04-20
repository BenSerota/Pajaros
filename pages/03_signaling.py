"""
Pagina 3: Eficacia de Senalizacion
Pagina analitica central que compara tipos de dispositivos anticolision.
Incluye pruebas chi-cuadrado, regresion binomial negativa, analisis UV
y comparaciones nocturnas/diurnas.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    COLORS, CHART_TEMPLATE, SIGNAL_TYPES_ORDERED,
    UV_LINE, UV_LINE_LABEL, UV_VANOS_RAW, UV_INSTALL_DATE, UV_SIGNAL_LABEL,
)
from src.data_loader import load_data
from src.filters import apply_filters, get_filter_summary
from src.statistics import (
    chi_squared_test, chi_squared_gof, kruskal_wallis_test,
    negative_binomial_regression, poisson_rate_test, stat_badge, format_p_value,
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

# Ordered signal types present in data
sig_types_present = [s for s in SIGNAL_TYPES_ORDERED if s in df_sig["signal_type"].values]

st.divider()

# =============================================================================
# SECCION 1: Victimas por Tipo de Senalizacion
# =============================================================================
st.subheader("1. Victimas por Tipo de Senalizacion")
st.caption(
    "Distribucion total de victimas segun el tipo de dispositivo anticolision. "
    "Se aplica una prueba chi-cuadrado de bondad de ajuste para evaluar si las "
    "victimas se distribuyen uniformemente entre los tipos de senalizacion."
)

col1, col2 = st.columns([2, 1])

with col1:
    sig_counts = (
        df_sig.groupby("signal_type")
        .size()
        .reindex(sig_types_present, fill_value=0)
        .reset_index(name="count")
    )
    sig_counts["signal_type"] = pd.Categorical(
        sig_counts["signal_type"], categories=sig_types_present, ordered=True
    )
    sig_counts = sig_counts.sort_values("signal_type")

    fig_sig = px.bar(
        sig_counts,
        x="signal_type",
        y="count",
        color="signal_type",
        color_discrete_map={s: COLORS.get(s, "#999") for s in SIGNAL_TYPES_ORDERED},
        template=CHART_TEMPLATE,
        height=400,
        labels={"signal_type": "", "count": "Victimas"},
    )
    fig_sig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=20, t=10, b=10),
    )
    fig_sig.update_traces(
        text=sig_counts["count"], textposition="outside"
    )
    st.plotly_chart(fig_sig, use_container_width=True)

with col2:
    # Chi-squared goodness-of-fit (uniform distribution)
    observed = sig_counts["count"].values
    gof_result = chi_squared_gof(observed)
    badge_gof = stat_badge(
        "Chi-cuadrado (bondad de ajuste)", gof_result["statistic"], gof_result["p_value"],
        n=int(observed.sum())
    )
    st.markdown(badge_gof, unsafe_allow_html=True)

    st.markdown("**Recuento por tipo de senalizacion**")
    display_sig = sig_counts[["signal_type", "count"]].copy()
    display_sig.columns = ["Senalizacion", "Victimas"]
    st.dataframe(display_sig, use_container_width=True, hide_index=True)

    st.markdown(f"""
    - **chi-cuadrado** = {gof_result['statistic']:.2f}, gl = {gof_result['dof']}
    - **p** = {format_p_value(gof_result['p_value'])}
    - Hipotesis nula: las victimas se distribuyen uniformemente
    """)

st.divider()

# =============================================================================
# SECCION 2: Analisis de Dispositivos Ultravioleta (UV)
# =============================================================================
st.subheader("2. Analisis de Dispositivos Ultravioleta (UV)")
st.caption(
    "Los marcadores UV estan instalados en los vanos 89-90, 90-91 y 91-92 "
    f"de la linea {UV_LINE_LABEL} desde junio de 2024."
)

st.info(
    "El analisis UV se basa en solo 3 vanos y un periodo corto (desde junio 2024). "
    "Los resultados deben interpretarse con precaucion debido al tamano muestral limitado."
)

uv_install_date = pd.Timestamp(UV_INSTALL_DATE)

# Helper: identify UV vanos on the GT-MB 132kV line
df_gt_mb = df[df["line"] == UV_LINE].copy()
has_vano_label = "vano_label" in df_gt_mb.columns
vano_col = "vano_label" if has_vano_label else "vano_raw_2" if "vano_raw_2" in df_gt_mb.columns else None

if vano_col is not None and len(df_gt_mb) > 0:
    df_gt_mb["is_uv_vano"] = df_gt_mb[vano_col].isin(UV_VANOS_RAW)

    # -------------------------------------------------------------------------
    # Sub-seccion A: Antes/Despues en vanos UV
    # -------------------------------------------------------------------------
    st.markdown("#### A. Comparacion Antes/Despues en Vanos UV")

    df_uv_vanos = df_gt_mb[df_gt_mb["is_uv_vano"]].copy()

    if len(df_uv_vanos) > 0:
        before = df_uv_vanos[df_uv_vanos["date"] < uv_install_date]
        after = df_uv_vanos[df_uv_vanos["date"] >= uv_install_date]

        count_before = len(before)
        count_after = len(after)

        # Calculate exposure in months
        if len(df_gt_mb) > 0 and df_gt_mb["date"].notna().any():
            study_start = df_gt_mb["date"].min()
            study_end = df_gt_mb["date"].max()
        else:
            study_start = pd.Timestamp("2018-03-01")
            study_end = pd.Timestamp("2025-09-30")

        months_before = max(1, (uv_install_date - study_start).days / 30.44)
        months_after = max(1, (study_end - uv_install_date).days / 30.44)

        rate_before = count_before / months_before
        rate_after = count_after / months_after

        col_a1, col_a2 = st.columns([2, 1])

        with col_a1:
            # Bar chart before/after
            ba_data = pd.DataFrame({
                "Periodo": ["Antes de UV", "Despues de UV"],
                "Victimas": [count_before, count_after],
                "Tasa mensual": [rate_before, rate_after],
                "Meses": [round(months_before, 1), round(months_after, 1)],
            })

            fig_ba = go.Figure()
            fig_ba.add_trace(go.Bar(
                x=ba_data["Periodo"],
                y=ba_data["Victimas"],
                marker_color=["#9E9E9E", COLORS.get("UV (Ultravioleta)", "#7B1FA2")],
                text=[f"{v} ({m:.1f} meses)" for v, m in zip(ba_data["Victimas"], ba_data["Meses"])],
                textposition="outside",
            ))
            fig_ba.update_layout(
                template=CHART_TEMPLATE,
                height=300,
                yaxis_title="Victimas",
                xaxis_title="",
                margin=dict(l=0, r=20, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_ba, use_container_width=True)

        with col_a2:
            # Poisson rate test
            rate_result = poisson_rate_test(count_before, months_before, count_after, months_after)
            badge_ba = stat_badge(
                "Comparacion de tasas Poisson",
                rate_result["rate_ratio"],
                rate_result["p_value"],
                n=count_before + count_after,
            )
            st.markdown(badge_ba, unsafe_allow_html=True)

            st.markdown(f"""
            - **Antes**: {count_before} victimas en {months_before:.1f} meses ({rate_before:.2f}/mes)
            - **Despues**: {count_after} victimas en {months_after:.1f} meses ({rate_after:.2f}/mes)
            - **Razon de tasas** = {rate_result['rate_ratio']:.2f}
            - **p** = {format_p_value(rate_result['p_value'])}
            """)
    else:
        st.info("No se encontraron registros en los vanos UV de la linea GT-MB 132kV.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Sub-seccion B: UV vs No-UV en la misma linea (desde junio 2024)
    # -------------------------------------------------------------------------
    st.markdown("#### B. Vanos UV vs No-UV en la Misma Linea (desde junio 2024)")

    df_gt_mb_post = df_gt_mb[df_gt_mb["date"] >= uv_install_date].copy()

    if len(df_gt_mb_post) > 0 and vano_col is not None:
        uv_post = df_gt_mb_post[df_gt_mb_post["is_uv_vano"]]
        non_uv_post = df_gt_mb_post[~df_gt_mb_post["is_uv_vano"]]

        count_uv = len(uv_post)
        count_non_uv = len(non_uv_post)

        # Number of distinct vanos
        n_uv_vanos = 3  # known
        n_non_uv_vanos = max(1, df_gt_mb_post[~df_gt_mb_post["is_uv_vano"]][vano_col].nunique())

        months_post = max(1, (study_end - uv_install_date).days / 30.44)

        rate_uv = count_uv / (n_uv_vanos * months_post)
        rate_non_uv = count_non_uv / (n_non_uv_vanos * months_post)

        col_b1, col_b2 = st.columns([2, 1])

        with col_b1:
            comp_data = pd.DataFrame({
                "Grupo": ["UV (3 vanos)", f"No-UV ({n_non_uv_vanos} vanos)"],
                "Victimas": [count_uv, count_non_uv],
                "Tasa (victimas/vano/mes)": [rate_uv, rate_non_uv],
            })

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=comp_data["Grupo"],
                y=comp_data["Tasa (victimas/vano/mes)"],
                marker_color=[COLORS.get("UV (Ultravioleta)", "#7B1FA2"), "#9E9E9E"],
                text=[f"{r:.3f}" for r in comp_data["Tasa (victimas/vano/mes)"]],
                textposition="outside",
            ))
            fig_comp.update_layout(
                template=CHART_TEMPLATE,
                height=300,
                yaxis_title="Victimas / vano / mes",
                xaxis_title="",
                margin=dict(l=0, r=20, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_b2:
            # Poisson rate comparison (exposure = n_vanos * months)
            exp_uv = n_uv_vanos * months_post
            exp_non_uv = n_non_uv_vanos * months_post
            rate_result_b = poisson_rate_test(count_uv, exp_uv, count_non_uv, exp_non_uv)
            badge_b = stat_badge(
                "Comparacion de tasas",
                rate_result_b["rate_ratio"],
                rate_result_b["p_value"],
                n=count_uv + count_non_uv,
            )
            st.markdown(badge_b, unsafe_allow_html=True)

            st.markdown(f"""
            - **UV**: {count_uv} victimas en {n_uv_vanos} vanos ({rate_uv:.3f}/vano/mes)
            - **No-UV**: {count_non_uv} victimas en {n_non_uv_vanos} vanos ({rate_non_uv:.3f}/vano/mes)
            - **p** = {format_p_value(rate_result_b['p_value'])}
            """)
    else:
        st.info("No hay registros suficientes en la linea GT-MB 132kV despues de junio 2024.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Sub-seccion C: UV vs Todos los Tipos de Senalizacion (junio 2024+)
    # -------------------------------------------------------------------------
    st.markdown("#### C. UV vs Otros Tipos de Senalizacion (todas las lineas, desde junio 2024)")

    df_post_uv = df_sig[df_sig["date"] >= uv_install_date].copy()

    if len(df_post_uv) > 0:
        sig_cts_post = df_post_uv["signal_type"].value_counts()
        types_present = [s for s in SIGNAL_TYPES_ORDERED if s in sig_cts_post.index]

        if UV_SIGNAL_LABEL in types_present and len(types_present) >= 2:
            col_c1, col_c2 = st.columns([2, 1])

            with col_c1:
                cts_df = sig_cts_post.reindex(types_present, fill_value=0).reset_index()
                cts_df.columns = ["signal_type", "count"]
                cts_df["signal_type"] = pd.Categorical(
                    cts_df["signal_type"], categories=types_present, ordered=True
                )
                cts_df = cts_df.sort_values("signal_type")

                fig_c = px.bar(
                    cts_df,
                    x="signal_type",
                    y="count",
                    color="signal_type",
                    color_discrete_map={s: COLORS.get(s, "#999") for s in SIGNAL_TYPES_ORDERED},
                    template=CHART_TEMPLATE,
                    height=350,
                    labels={"signal_type": "", "count": "Victimas (desde jun 2024)"},
                )
                fig_c.update_layout(showlegend=False, margin=dict(l=0, r=20, t=10, b=10))
                fig_c.update_traces(text=cts_df["count"], textposition="outside")
                st.plotly_chart(fig_c, use_container_width=True)

            with col_c2:
                # Overall chi-squared goodness-of-fit
                obs_c = cts_df["count"].values
                gof_c = chi_squared_gof(obs_c)
                badge_c = stat_badge(
                    "Chi-cuadrado (bondad de ajuste)",
                    gof_c["statistic"], gof_c["p_value"],
                    n=int(obs_c.sum()),
                )
                st.markdown(badge_c, unsafe_allow_html=True)

                st.markdown("**Recuento por tipo (desde jun 2024)**")
                st.dataframe(
                    cts_df[["signal_type", "count"]].rename(
                        columns={"signal_type": "Senalizacion", "count": "Victimas"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            # Pairwise Fisher exact tests: UV vs each other type
            with st.expander("Comparaciones pareadas UV vs otros (Fisher, Bonferroni)"):
                uv_count = int(sig_cts_post.get(UV_SIGNAL_LABEL, 0))
                other_types = [t for t in types_present if t != UV_SIGNAL_LABEL]

                pairwise_results = []
                for ot in other_types:
                    ot_count = int(sig_cts_post.get(ot, 0))
                    # 2x2 table: [[UV, non-UV in UV], [other, non-other in other]]
                    # Simpler: test if UV proportion differs from other type proportion
                    total_pair = uv_count + ot_count
                    if total_pair > 0:
                        table_2x2 = np.array([
                            [uv_count, total_pair - uv_count],
                            [ot_count, total_pair - ot_count],
                        ])
                        try:
                            odds, p = fisher_exact(table_2x2)
                        except Exception:
                            odds, p = np.nan, np.nan
                        pairwise_results.append({
                            "UV vs": ot,
                            "Victimas UV": uv_count,
                            f"Victimas {ot[:15]}": ot_count,
                            "p_value": p,
                        })

                if pairwise_results:
                    pw_df = pd.DataFrame(pairwise_results)
                    n_tests = len(pw_df)
                    pw_df["p_ajustado (Bonf.)"] = np.minimum(pw_df["p_value"] * n_tests, 1.0)
                    pw_df["p_value"] = pw_df["p_value"].apply(format_p_value)
                    pw_df["p_ajustado (Bonf.)"] = pw_df["p_ajustado (Bonf.)"].apply(
                        lambda x: format_p_value(x) if not isinstance(x, str) else x
                    )
                    st.dataframe(pw_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No hay suficientes datos para comparaciones pareadas.")
        else:
            st.info(
                "No se encontraron registros con senalizacion UV despues de junio 2024. "
                "Verifique que los datos incluyen registros en los vanos UV."
            )
    else:
        st.info("No hay registros despues de junio 2024 con los filtros actuales.")

else:
    st.info("No hay datos disponibles para la linea GT-MB 132kV o no se dispone del identificador de vano.")

st.divider()

# =============================================================================
# SECCION 3: Separacion de senalizacion (10m vs 20m)
# =============================================================================
st.subheader("3. Separacion de Senalizacion: 10m vs 20m")

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
    spacing_counts = df_spacing.groupby("spacing_label").size().reset_index(name="count")

    fig_sp = go.Figure()
    fig_sp.add_trace(go.Bar(
        x=spacing_counts["spacing_label"],
        y=spacing_counts["count"],
        marker_color="#1976D2",
        text=[f"{v}" for v in spacing_counts["count"]],
        textposition="outside",
    ))
    fig_sp.update_layout(
        template=CHART_TEMPLATE,
        height=350,
        xaxis_title="Separacion de dispositivos",
        yaxis_title="Victimas",
        margin=dict(l=0, r=20, t=10, b=10),
        showlegend=False,
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
# SECCION 4: Efecto del estado de la senalizacion
# =============================================================================
st.subheader("4. Efecto del Estado de la Senalizacion")
st.caption("Afecta el estado fisico de los dispositivos de senalizacion a las tasas de colision?")

df_cond = df_sig[df_sig["signal_condition"].notna()].copy()

if len(df_cond) > 0:
    col_c1, col_c2 = st.columns([2, 1])

    with col_c1:
        cond_counts = df_cond.groupby("signal_condition").size().reset_index(name="count")

        fig_cond = go.Figure()
        fig_cond.add_trace(go.Bar(
            x=cond_counts["signal_condition"],
            y=cond_counts["count"],
            marker_color="#1976D2",
            text=[f"{v}" for v in cond_counts["count"]],
            textposition="outside",
        ))
        fig_cond.update_layout(
            template=CHART_TEMPLATE, height=350,
            xaxis_title="Estado de la senalizacion", yaxis_title="Victimas",
            margin=dict(l=0, r=20, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_cond, use_container_width=True)

    with col_c2:
        # Chi-squared goodness-of-fit on condition
        obs_cond = cond_counts["count"].values
        gof_cond = chi_squared_gof(obs_cond)
        badge_cond = stat_badge(
            "Chi-cuadrado (bondad de ajuste)", gof_cond["statistic"], gof_cond["p_value"],
            n=int(obs_cond.sum())
        )
        st.markdown(badge_cond, unsafe_allow_html=True)

        st.markdown("**Recuento por estado**")
        st.dataframe(
            cond_counts.rename(columns={"signal_condition": "Estado", "count": "Victimas"}),
            use_container_width=True, hide_index=True,
        )

        # Nota sobre celdas pequenas
        min_cell = obs_cond.min()
        if min_cell < 5:
            st.caption(f"Recuento minimo = {min_cell}. Los resultados del chi-cuadrado pueden no ser fiables con celdas pequenas.")

st.divider()

# =============================================================================
# SECCION 5: Comparacion a nivel de vano (Kruskal-Wallis)
# =============================================================================
st.subheader("5. Victimas por Vano segun Tipo de Senalizacion")
st.caption(
    "Comparacion de la distribucion de victimas por vano segun tipo de senalizacion. "
    "Esto controla por el numero de vanos en lugar de comparar totales brutos."
)

# Calcular conteos por vano
span_col = (
    "vano_label" if "vano_label" in df_sig.columns
    else "vano_raw_2" if "vano_raw_2" in df_sig.columns
    else "span_id" if "span_id" in df_sig.columns
    else None
)

if span_col is None:
    st.info("No hay identificador de vano disponible para esta comparacion.")
    span_sig = pd.DataFrame(columns=["span_id", "signal_type", "count"])
else:
    span_sig = (
        df_sig.dropna(subset=[span_col])
        .groupby([span_col, "signal_type"])
        .size()
        .reset_index(name="count")
        .rename(columns={span_col: "span_id"})
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
        - **epsilon cuadrado** = {kw_result['effect_size']:.3f}
        """)

        # Estadisticas resumidas por grupo
        summary = span_sig.groupby("signal_type")["count"].agg(
            ["count", "mean", "median", "std"]
        ).round(2)
        summary.columns = ["n_vanos", "media", "mediana", "desv_est"]
        st.dataframe(summary, use_container_width=True)

st.divider()

# =============================================================================
# SECCION 6: Regresion binomial negativa (grafico de bosque)
# =============================================================================
st.subheader("6. Modelo Multivariante: Regresion Binomial Negativa")
st.caption(
    "Modelo: count ~ signal_type + voltage. "
    "IRR (Razon de Tasas de Incidencia) > 1 indica mayor mortalidad respecto a la categoria de referencia."
)

# Preparar datos de regresion: conteos por vano con atributos
try:
    if span_col is None:
        raise KeyError("span identifier")

    span_attrs = (
        df_sig.dropna(subset=[span_col])
        .groupby(span_col)
        .agg(
            count=(span_col, "size"),
            signal_type=("signal_type", "first"),
            voltage=("voltage", "first"),
            signal_spacing_m=("signal_spacing_m", "first"),
            signal_condition=("signal_condition", "first"),
            line=("line_label", "first"),
        )
        .reset_index()
        .rename(columns={span_col: "span_id"})
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
                        sig = "*" if row["pvalue"] < 0.05 else ""

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
                                f"p = {format_p_value(row['pvalue'])}"
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
                - **Dispersion (alfa)** = {nb_result.get('alpha', 'N/D')}
                """)
        else:
            st.info("No se pudo ajustar el modelo de regresion. Esto puede deberse a variacion insuficiente en los datos.")
    else:
        st.info("Datos insuficientes para regresion multivariante tras el filtrado.")

except Exception as e:
    st.warning(f"No se pudo ajustar el modelo de regresion: {str(e)}")

st.divider()

# =============================================================================
# SECCION 7: Nocturnas vs Diurnas x Senalizacion
# =============================================================================
st.subheader("7. Especies Nocturnas vs Diurnas por Tipo de Senalizacion")
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
