"""
Cached data loading for the Fuerteventura Bird Mortality Streamlit Dashboard.

Responsibilities:
  - Load cleaned parquet (or run cleaning pipeline if parquet doesn't exist)
  - Convert UTM coordinates to WGS84 lat/lon
  - Merge species activity classifications
  - Load precomputed road distances if available
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_XLS, CLEAN_PARQUET, ROAD_DISTANCES_PARQUET,
    SRC_CRS, DST_CRS,
)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def convert_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Add latitude/longitude columns by converting UTM Zone 28N to WGS84.

    Uses pyproj if available; falls back to an approximate formula otherwise.
    Rows with missing utm_x or utm_y get NaN for both lat and lon.
    """
    mask = df["utm_x"].notna() & df["utm_y"].notna()

    if mask.sum() == 0:
        df["latitude"] = np.nan
        df["longitude"] = np.nan
        return df

    x = df.loc[mask, "utm_x"].values
    y = df.loc[mask, "utm_y"].values

    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            SRC_CRS,   # EPSG:32628 — UTM Zone 28N
            DST_CRS,   # EPSG:4326  — WGS84
            always_xy=True,
        )
        # always_xy: input (easting, northing) -> output (lon, lat)
        lon, lat = transformer.transform(x, y)

    except ImportError:
        # Approximate fallback — good enough for map rendering
        # Central meridian for UTM zone 28N = -15 degrees
        central_meridian = -15.0
        lat = y / 111_320.0
        lat_rad = np.radians(lat)
        lon = (x - 500_000) / (111_320.0 * np.cos(lat_rad)) + central_meridian

    df["latitude"] = np.nan
    df["longitude"] = np.nan
    df.loc[mask, "latitude"] = lat
    df.loc[mask, "longitude"] = lon

    return df


# ---------------------------------------------------------------------------
# Main data loader
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading bird mortality data...")
def load_data() -> pd.DataFrame:
    """Load, enrich, and cache the cleaned dataset.

    Pipeline:
      1. Load parquet if it exists, otherwise run the cleaning pipeline
      2. Convert UTM coordinates to WGS84 lat/lon
      3. Classify species by activity pattern (Nocturnal/Crepuscular/Diurnal)
      4. Return the full DataFrame
    """
    # Step 1 — load or generate clean parquet
    if CLEAN_PARQUET.exists():
        df = pd.read_parquet(CLEAN_PARQUET)
    else:
        from src.data_cleaning import clean_data

        df = clean_data(RAW_XLS)
        CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CLEAN_PARQUET, index=False)

    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Step 2 — coordinate conversion
    if "utm_x" in df.columns and "utm_y" in df.columns:
        df = convert_coordinates(df)

    # Step 3 — species activity classification
    from src.species_classifier import classify_species

    if "species_clean" in df.columns and "activity_pattern" not in df.columns:
        df = classify_species(df)

    return df


# ---------------------------------------------------------------------------
# Road distances (optional precomputed data)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading road distance data...")
def load_road_distances():
    """Load precomputed road distances if the file exists.

    Returns None when no precomputed distances are available.
    """
    if ROAD_DISTANCES_PARQUET.exists():
        return pd.read_parquet(ROAD_DISTANCES_PARQUET)
    return None
