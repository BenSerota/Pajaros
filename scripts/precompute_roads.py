#!/usr/bin/env python3
"""
Precompute road distances for all bird casualty records.

Fetches the Fuerteventura road network and power lines from OpenStreetMap,
then computes the nearest-road distance for each of the ~1,026 casualty
points. Results are saved to data/processed/road_distances.parquet.

Run as:
    python scripts/precompute_roads.py

Typical runtime: 30-60 seconds (OSM download + spatial operations).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    CLEAN_PARQUET, ROAD_DISTANCES_PARQUET,
    ROADS_GPKG, POWER_LINES_GPKG,
    BBOX_LAT_MIN, BBOX_LAT_MAX, BBOX_LON_MIN, BBOX_LON_MAX,
)

try:
    from src.gis_pipeline import (
        fetch_road_network,
        fetch_power_lines,
        casualties_to_geodataframe,
        compute_road_distances,
        classify_road,
    )
except ImportError as e:
    print(f"Error: Could not import GIS pipeline: {e}")
    print(
        "Install required packages with:\n"
        "  pip install osmnx geopandas shapely pyproj"
    )
    sys.exit(1)


def main() -> None:
    t0 = time.time()

    # ── 1. Load cleaned data ──────────────────────────────────────────────
    if not CLEAN_PARQUET.exists():
        print(f"Error: Cleaned data not found at {CLEAN_PARQUET}")
        print("Run the data cleaning pipeline first.")
        sys.exit(1)

    df = pd.read_parquet(CLEAN_PARQUET)
    n_total = len(df)
    n_with_coords = df.dropna(subset=["utm_x", "utm_y"]).shape[0]
    print(f"Loaded {n_total} casualty records ({n_with_coords} with coordinates)")

    # ── 2. Add a 2km buffer around data extent for road queries ───────────
    # Convert UTM extent to approximate lat/lon buffer
    buffer_deg = 0.02  # roughly 2km at this latitude
    bbox = (
        BBOX_LAT_MAX + buffer_deg,   # north
        BBOX_LAT_MIN - buffer_deg,   # south
        BBOX_LON_MAX + buffer_deg,   # east
        BBOX_LON_MIN - buffer_deg,   # west
    )

    # ── 3. Fetch road network ─────────────────────────────────────────────
    print("Fetching road network from OpenStreetMap...")
    try:
        roads_gdf = fetch_road_network(bbox=bbox, cache_path=ROADS_GPKG)
    except Exception as e:
        print(f"\nError fetching road network: {e}")
        print(
            "This may be a network issue. Try again later or check your "
            "internet connection."
        )
        sys.exit(1)

    if roads_gdf is None or roads_gdf.empty:
        print("Error: No road data retrieved. Cannot continue.")
        sys.exit(1)

    print(f"Found {len(roads_gdf)} road segments")

    # ── 4. Fetch power lines ──────────────────────────────────────────────
    print("Fetching power lines from OpenStreetMap...")
    try:
        power_gdf = fetch_power_lines(bbox=bbox, cache_path=POWER_LINES_GPKG)
    except Exception as e:
        print(f"\nWarning: Could not fetch power lines: {e}")
        print("Continuing without power line data.")
        power_gdf = None

    if power_gdf is not None and not power_gdf.empty:
        print(f"Found {len(power_gdf)} power line ways")
    else:
        print("No power line data available (non-critical, continuing)")

    # ── 5. Convert casualties to GeoDataFrame ─────────────────────────────
    casualties_gdf = casualties_to_geodataframe(df)
    if casualties_gdf is None or casualties_gdf.empty:
        print("Error: Could not create GeoDataFrame from casualty data.")
        sys.exit(1)

    n_geo = len(casualties_gdf)
    print(f"Computing nearest road distances for {n_geo} casualty records...")

    # ── 6. Compute nearest road distances ─────────────────────────────────
    try:
        distances_df = compute_road_distances(casualties_gdf, roads_gdf)
    except Exception as e:
        print(f"\nError computing road distances: {e}")
        print("This may be due to a spatial indexing issue. Check your data.")
        sys.exit(1)

    if distances_df.empty:
        print("Error: No distances computed.")
        sys.exit(1)

    # ── 7. Save results ───────────────────────────────────────────────────
    ROAD_DISTANCES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    distances_df.to_parquet(ROAD_DISTANCES_PARQUET, index=False)

    elapsed = time.time() - t0

    # ── 8. Print summary ──────────────────────────────────────────────────
    print(f"\nDone. Summary:")
    print(f"  Records processed: {len(distances_df)}")
    print(f"  Time elapsed: {elapsed:.1f}s")

    valid = distances_df["road_distance_m"].dropna()
    if len(valid) > 0:
        print(f"\n  Road distance distribution:")
        print(f"    Min:    {valid.min():.1f} m")
        print(f"    Max:    {valid.max():.1f} m")
        print(f"    Median: {valid.median():.1f} m")
        print(f"    Mean:   {valid.mean():.1f} m")

    if "nearest_road_type" in distances_df.columns:
        type_counts = distances_df["nearest_road_type"].value_counts()
        if len(type_counts) > 0:
            print(f"\n  Distribution by road type:")
            for road_type, count in type_counts.items():
                pct = count / len(distances_df) * 100
                print(f"    {road_type:20s}  {count:4d}  ({pct:.1f}%)")

    print(f"\nSaved to {ROAD_DISTANCES_PARQUET}")


if __name__ == "__main__":
    main()
