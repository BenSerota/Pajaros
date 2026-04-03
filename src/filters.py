"""
Sistema global de filtros en la barra lateral del Panel de Siniestralidad
de Avifauna de Fuerteventura.

Renderiza filtros interactivos en la barra lateral de Streamlit y devuelve
una copia filtrada del DataFrame.  Todas las páginas del panel importan
``apply_filters`` para que las selecciones persistan entre navegaciones.
"""

import streamlit as st
import pandas as pd


# ---------------------------------------------------------------------------
# Main filter function
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered DataFrame.

    Filters are applied sequentially; only rows matching ALL active filters
    are returned.  Widget state is automatically persisted in
    ``st.session_state`` across page navigations.
    """
    filtered = df.copy()

    # ------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------
    st.sidebar.subheader("Temporal")

    if "year" in df.columns and df["year"].notna().any():
        year_min = int(df["year"].min())
        year_max = int(df["year"].max())
        year_range = st.sidebar.slider(
            "Rango de años",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            key="filter_year_range",
        )
        filtered = filtered[
            filtered["year"].between(year_range[0], year_range[1])
        ]

    # ------------------------------------------------------------------
    # Infraestructura
    # ------------------------------------------------------------------
    st.sidebar.subheader("Infraestructura")

    # Línea eléctrica
    if "line_label" in df.columns:
        line_options = sorted(df["line_label"].dropna().unique().tolist())
        selected_lines = st.sidebar.multiselect(
            "Línea eléctrica",
            options=line_options,
            default=line_options,
            key="filter_power_line",
        )
        if selected_lines:
            filtered = filtered[filtered["line_label"].isin(selected_lines)]

    # Tensión
    if "voltage" in df.columns:
        voltage_choice = st.sidebar.radio(
            "Tensión",
            options=["Todas", "66kV", "132kV"],
            index=0,
            key="filter_voltage",
            horizontal=True,
        )
        if voltage_choice == "66kV":
            filtered = filtered[filtered["voltage"] == 66]
        elif voltage_choice == "132kV":
            filtered = filtered[filtered["voltage"] == 132]

    # Tipo de señalización
    if "signal_type" in df.columns:
        signal_options = sorted(
            df["signal_type"].dropna().unique().tolist()
        )
        if signal_options:
            selected_signals = st.sidebar.multiselect(
                "Tipo de señalización",
                options=signal_options,
                default=signal_options,
                key="filter_signal_type",
            )
            if selected_signals:
                filtered = filtered[
                    filtered["signal_type"].isin(selected_signals)
                    | filtered["signal_type"].isna()
                ]

    # ------------------------------------------------------------------
    # Especies
    # ------------------------------------------------------------------
    st.sidebar.subheader("Especies")

    # Grupo de especies — top 10 + Otros + Desconocida
    if "species_clean" in df.columns:
        top_species = (
            df["species_clean"]
            .value_counts()
            .head(10)
            .index.tolist()
        )
        has_unknown = "Unknown / No identificada" in df["species_clean"].values
        group_options = top_species.copy()
        if "Unknown / No identificada" not in group_options and has_unknown:
            group_options.append("Desconocida")
        group_options.append("Otros")

        selected_groups = st.sidebar.multiselect(
            "Grupo de especies",
            options=group_options,
            default=group_options,
            key="filter_species_group",
        )
        if selected_groups:
            keep_species = set()
            for g in selected_groups:
                if g == "Otros":
                    other = set(df["species_clean"].unique()) - set(top_species)
                    if has_unknown:
                        other.discard("Unknown / No identificada")
                    keep_species.update(other)
                elif g == "Desconocida":
                    keep_species.add("Unknown / No identificada")
                else:
                    keep_species.add(g)
            filtered = filtered[filtered["species_clean"].isin(keep_species)]

    # Patrón de actividad
    if "activity_pattern" in df.columns:
        activity_options = ["Nocturna", "Crepuscular", "Diurna", "Desconocida"]
        available_activities = [
            a for a in activity_options
            if a in df["activity_pattern"].values
        ]
        selected_activities = st.sidebar.multiselect(
            "Patrón de actividad",
            options=available_activities,
            default=available_activities,
            key="filter_activity_pattern",
        )
        if selected_activities:
            filtered = filtered[
                filtered["activity_pattern"].isin(selected_activities)
            ]

    # ------------------------------------------------------------------
    # Evento
    # ------------------------------------------------------------------
    st.sidebar.subheader("Evento")

    # Estado de conservación
    if "iucn_status" in df.columns:
        iucn_options = sorted(
            df["iucn_status"].dropna().unique().tolist()
        )
        if iucn_options:
            selected_iucn = st.sidebar.multiselect(
                "Estado de conservación (UICN)",
                options=iucn_options,
                default=iucn_options,
                key="filter_iucn_status",
            )
            if selected_iucn:
                filtered = filtered[
                    filtered["iucn_status"].isin(selected_iucn)
                    | filtered["iucn_status"].isna()
                ]

    # Evidencia
    st.sidebar.subheader("Evidencia")

    # Tipo de restos
    if "remains_type" in df.columns:
        remains_options = sorted(
            df["remains_type"].dropna().unique().tolist()
        )
        if remains_options:
            selected_remains = st.sidebar.multiselect(
                "Tipo de restos",
                options=remains_options,
                default=remains_options,
                key="filter_remains_type",
            )
            if selected_remains:
                filtered = filtered[
                    filtered["remains_type"].isin(selected_remains)
                    | filtered["remains_type"].isna()
                ]

    # ------------------------------------------------------------------
    # Resumen de registros
    # ------------------------------------------------------------------
    st.sidebar.divider()
    total = len(df)
    shown = len(filtered)
    pct = (shown / total * 100) if total > 0 else 0
    st.sidebar.metric(
        "Mostrando",
        f"{shown:,} de {total:,} registros",
        f"{pct:.0f}%",
        delta_color="off",
    )

    return filtered


# ---------------------------------------------------------------------------
# Filter summary for chart subtitles
# ---------------------------------------------------------------------------

def get_filter_summary(df_filtered: pd.DataFrame, df_full: pd.DataFrame) -> str:
    """Devuelve un resumen legible de los filtros activos.

    Compara el DataFrame filtrado con el conjunto completo para detectar
    qué dimensiones se han acotado.  Se usa como subtítulo de gráficos.
    """
    parts: list[str] = []

    # Rango de años
    if "year" in df_filtered.columns and df_filtered["year"].notna().any():
        filt_min = int(df_filtered["year"].min())
        filt_max = int(df_filtered["year"].max())
        full_min = int(df_full["year"].min())
        full_max = int(df_full["year"].max())
        if filt_min != full_min or filt_max != full_max:
            parts.append(f"{filt_min}--{filt_max}")

    # Tensión
    if "voltage" in df_filtered.columns:
        voltages = df_filtered["voltage"].dropna().unique()
        if len(voltages) == 1:
            parts.append(f"solo {int(voltages[0])}kV")

    # Línea eléctrica
    if "line_label" in df_filtered.columns:
        filt_lines = set(df_filtered["line_label"].dropna().unique())
        full_lines = set(df_full["line_label"].dropna().unique())
        if filt_lines and filt_lines != full_lines:
            if len(filt_lines) <= 2:
                parts.append(", ".join(sorted(filt_lines)))
            else:
                parts.append(f"{len(filt_lines)} de {len(full_lines)} líneas")

    # Patrón de actividad
    if "activity_pattern" in df_filtered.columns:
        filt_act = set(df_filtered["activity_pattern"].dropna().unique())
        full_act = set(df_full["activity_pattern"].dropna().unique())
        if filt_act and filt_act != full_act:
            if len(filt_act) <= 2:
                parts.append(", ".join(sorted(filt_act)))
            else:
                parts.append(f"{len(filt_act)} patrones de actividad")

    n = len(df_filtered)
    total = len(df_full)
    if n < total:
        parts.append(f"{n:,} de {total:,} registros")

    if not parts:
        return "Todos los datos"

    return " | ".join(parts)
