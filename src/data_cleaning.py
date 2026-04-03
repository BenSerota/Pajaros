"""
Data cleaning pipeline for Fuerteventura bird mortality data.
Handles: column renaming, species parsing, normalization, derived columns.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    COLUMN_MAP, SIGNAL_TYPE_NORM, SIGNAL_CONDITION_NORM,
    MUNICIPIO_NORM, LINE_META, MONTH_TO_SEASON,
)


def load_raw(filepath: Path) -> pd.DataFrame:
    """Load the raw XLS file using xlrd (handles old .xls format)."""
    df = pd.read_excel(filepath, sheet_name="export", engine="xlrd")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns using COLUMN_MAP.
    Handles the duplicate 'Vano' columns by positional index.
    """
    # The raw file has two columns named "Vano" at indices 21 and 22.
    # pandas auto-suffixes them as "Vano" and "Vano.1" when reading.
    # We rename those specifically before applying the general map.
    cols = list(df.columns)

    # Find duplicate Vano columns
    vano_indices = [i for i, c in enumerate(cols) if c.startswith("Vano") and "Completado" not in c]
    if len(vano_indices) >= 2:
        cols[vano_indices[0]] = "vano_raw_1"
        cols[vano_indices[1]] = "vano_raw_2"
        df.columns = cols

    # Apply the general column map
    df = df.rename(columns=COLUMN_MAP)

    # Drop columns not in our keep list that weren't renamed (e.g. vano_raw_1, vano_raw_2)
    return df


def parse_species(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the species_raw column into structured components.

    Raw format examples:
        "Columba livia / paloma bravía"
        "Calonectris diomedea / pardela cenicienta - 1"
        "Burhinus oedicnemus insularum / Alcaraván canario oriental o majorero"
        "_no identificada"

    Produces:
        species_scientific: "Columba livia"
        species_common: "paloma bravía"
        species_genus: "Columba"
        species_variant: 1 (or NaN)
        species_clean: "Columba livia / paloma bravía" (without variant suffix)
    """
    sci = []
    common = []
    genus = []
    variant = []
    clean = []

    for raw in df["species_raw"].fillna("_no identificada"):
        raw = raw.strip()

        if raw.startswith("_") or raw == "":
            sci.append("Unknown")
            common.append("No identificada")
            genus.append("Unknown")
            variant.append(np.nan)
            clean.append("Unknown / No identificada")
            continue

        # Extract trailing variant number: " - 1", " - 2", etc.
        var_match = re.search(r"\s*-\s*(\d+)\s*$", raw)
        if var_match:
            var_num = int(var_match.group(1))
            raw_stripped = raw[:var_match.start()].strip()
        else:
            var_num = np.nan
            raw_stripped = raw

        # Split on " / "
        parts = raw_stripped.split(" / ", 1)
        if len(parts) == 2:
            s = parts[0].strip()
            c = parts[1].strip()
        else:
            # No common name separator — treat whole thing as scientific
            s = parts[0].strip()
            c = s

        # Extract genus (first word of scientific name)
        g = s.split()[0] if s else "Unknown"

        sci.append(s)
        common.append(c)
        genus.append(g)
        variant.append(var_num)
        clean.append(f"{s} / {c}")

    df["species_scientific"] = sci
    df["species_common"] = common
    df["species_genus"] = genus
    df["species_variant"] = variant
    df["species_clean"] = clean

    return df


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical columns: signal types, municipio, etc."""

    # Signal type
    df["signal_type"] = (
        df["signal_type"]
        .fillna("")
        .str.strip()
        .map(SIGNAL_TYPE_NORM)
    )
    # Records with empty signal type stay NaN (excluded from signaling analysis)

    # Signal condition
    df["signal_condition"] = (
        df["signal_condition"]
        .fillna("")
        .str.strip()
        .replace("", np.nan)
        .map(SIGNAL_CONDITION_NORM)
    )

    # Municipio
    df["municipio"] = (
        df["municipio"]
        .fillna("")
        .str.strip()
        .replace("", np.nan)
        .map(lambda x: MUNICIPIO_NORM.get(x, x) if pd.notna(x) else x)
    )

    # Paraje normalization (title case, fix known issues)
    paraje_norm = {
        "Natural Protegido": "Natural protegido",
        "RURAL": "Rural",
        "Interurbana": "Interurbano",
    }
    df["paraje"] = (
        df["paraje"]
        .fillna("")
        .str.strip()
        .replace("", np.nan)
        .map(lambda x: paraje_norm.get(x, x) if pd.notna(x) else x)
    )

    # Boolean columns
    bool_map = {"Sí": True, "No": False}
    for col in ["is_collision", "has_signaling", "scavenging", "is_focal"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .str.strip()
                .map(bool_map)
            )

    # Provincia normalization
    df["provincia"] = (
        df["provincia"]
        .fillna("")
        .str.strip()
        .replace("", np.nan)
        .map(lambda x: "Las Palmas" if pd.notna(x) else x)
    )

    return df


def derive_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from the date column."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year.astype("Int64")
    df["month"] = df["date"].dt.month.astype("Int64")
    df["day_of_week"] = df["date"].dt.dayofweek.astype("Int64")  # 0=Mon
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("Int64")
    df["quarter"] = df["date"].dt.quarter.astype("Int64")
    df["season"] = df["month"].map(MONTH_TO_SEASON)
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    return df


def derive_line_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add voltage, corridor, and zone from LINE_META lookup."""
    df["voltage"] = df["line"].map(lambda x: LINE_META.get(x, {}).get("voltage"))
    df["corridor"] = df["line"].map(lambda x: LINE_META.get(x, {}).get("corridor"))
    df["zone"] = df["line"].map(lambda x: LINE_META.get(x, {}).get("zone"))
    return df


def derive_conservation_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a numeric conservation priority score.
    Higher = more urgent conservation concern.
    """
    score = pd.Series(0, index=df.index, dtype=float)

    # IUCN component (0–4)
    iucn_scores = {
        "En peligro de extinción": 4,
        "Vulnerable": 3,
        "Riesgo menor-casi amenazada": 2,
        "Riesgo menor-preocupación menor": 0,
    }
    score += df["iucn_status"].map(iucn_scores).fillna(0)

    # Spanish catalog (0–3)
    cat_scores = {
        "En peligro de extinción": 3,
        "Vulnerable": 2,
        "Listado de Especies Silvestres en Régimen de Protección Especial": 1,
    }
    score += df["spanish_catalog"].map(cat_scores).fillna(0)

    # Regional catalog (0–2)
    reg_scores = {
        "En peligro de extinción": 2,
        "Vulnerable": 1.5,
        "De interés especial": 1,
    }
    score += df["regional_catalog"].map(reg_scores).fillna(0)

    # Focal species bonus
    score += df["is_focal"].astype(float).fillna(0) * 1

    df["conservation_score"] = score
    return df


def derive_line_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create shorter line labels for charts."""
    label_map = {
        "L/Gran Tarajal-Matas Blancas 132 kV": "GT–MB 132kV",
        "66 kV GRAN TARAJAL MATAS BLANCAS": "GT–MB 66kV",
        "L/ La Oliva-Pto. del Rosario 132 kV": "OL–PR 132kV",
        "66 kV CORRALEJO SALINAS": "CO–SA 66kV",
        "L/Pto. del Rosario-Gran Tarajal 132 kV": "PR–GT 132kV",
        "66 kV SALINAS GRAN TARAJAL": "SA–GT 66kV",
    }
    df["line_label"] = df["line"].map(label_map).fillna(df["line"])
    return df


def clean_data(filepath: Path) -> pd.DataFrame:
    """
    Full cleaning pipeline: load → rename → parse → normalize → derive.
    Returns a clean DataFrame ready for analysis.
    """
    df = load_raw(filepath)
    df = rename_columns(df)
    df = parse_species(df)
    df = normalize_categoricals(df)
    df = derive_date_features(df)
    df = derive_line_metadata(df)
    df = derive_conservation_score(df)
    df = derive_line_label(df)

    # Ensure numeric columns are proper types
    for col in ["utm_x", "utm_y", "observer_distance_m", "signal_spacing_m", "specimen_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Force mixed-type columns to string for parquet compatibility
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace("nan", np.nan).replace("None", np.nan)

    # Fix boolean columns that got stringified
    for col in ["is_collision", "scavenging"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False})

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df


if __name__ == "__main__":
    """Run standalone to verify cleaning pipeline."""
    from config import RAW_XLS, CLEAN_PARQUET

    print("Loading and cleaning data...")
    df = clean_data(RAW_XLS)

    print(f"\nShape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for c in df.columns:
        non_null = df[c].notna().sum()
        print(f"  {c:35s}  {non_null:4d} non-null  {df[c].dtype}")

    print(f"\nSample species parsing:")
    for _, row in df[["species_raw", "species_scientific", "species_common", "species_genus", "species_variant"]].drop_duplicates("species_raw").head(10).iterrows():
        print(f"  {str(row['species_raw']):60s} → {row['species_scientific']} | {row['species_common']} | genus={row['species_genus']} | var={row['species_variant']}")

    print(f"\nSignal type distribution:")
    print(df["signal_type"].value_counts(dropna=False))

    print(f"\nMunicipio distribution:")
    print(df["municipio"].value_counts(dropna=False))

    print(f"\nConservation score distribution:")
    print(df["conservation_score"].describe())

    # Save
    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)
    print(f"\nSaved to {CLEAN_PARQUET}")
