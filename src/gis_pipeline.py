"""
GIS analysis module for road proximity and power line overlay.

Uses OpenStreetMap data via osmnx to:
  - Fetch the road network for Fuerteventura
  - Fetch high-voltage power line geometries
  - Compute nearest-road distance for each casualty point
  - Classify road types from OSM highway tags

All spatial computations use UTM 28N (EPSG:32628) for metric accuracy.
Gracefully degrades when osmnx / geopandas are not installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional, Tuple
import warnings
import pandas as pd
import numpy as np

from config import (
    BBOX_LAT_MIN, BBOX_LAT_MAX, BBOX_LON_MIN, BBOX_LON_MAX,
    SRC_CRS, DST_CRS, ROADS_GPKG, POWER_LINES_GPKG,
    ROAD_DISTANCES_PARQUET, ROAD_CLASSIFICATION,
)

# ---------------------------------------------------------------------------
# Optional GIS imports — dashboard still works without them
# ---------------------------------------------------------------------------
try:
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import Point
    from shapely.ops import nearest_points

    _HAS_GIS = True
except ImportError:
    _HAS_GIS = False
    warnings.warn(
        "osmnx / geopandas not installed. GIS features (road distances, "
        "power line overlay) will be unavailable. Install with: "
        "pip install osmnx geopandas",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Road classification
# ---------------------------------------------------------------------------

def classify_road(highway_tag: str) -> str:
    """Map an OSM highway tag to a simplified category.

    Categories:
        motorway / trunk       -> "Major highway"
        primary / secondary    -> "Main road"
        tertiary               -> "Local road"
        residential / unclassified / track / service -> "Minor road/track"

    Unrecognised tags default to "Minor road/track".
    """
    if not isinstance(highway_tag, str):
        return "Minor road/track"
    # osmnx sometimes stores lists; take the first element
    tag = highway_tag.split(",")[0].strip().strip("['\" ]")
    return ROAD_CLASSIFICATION.get(tag, "Minor road/track")


# ---------------------------------------------------------------------------
# Fetching OSM data
# ---------------------------------------------------------------------------

def fetch_road_network(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    cache_path: Optional[Path] = None,
) -> "Optional[gpd.GeoDataFrame]":
    """Download the road network from OpenStreetMap using osmnx.

    Parameters
    ----------
    bbox : tuple (north, south, east, west) in WGS84, optional
        Defaults to the Fuerteventura bounding box from config.
    cache_path : Path, optional
        GeoPackage file to cache the result. Defaults to ``ROADS_GPKG``.

    Returns
    -------
    geopandas.GeoDataFrame of road edges, or None if GIS libs missing.
    """
    if not _HAS_GIS:
        print("GIS libraries not available. Skipping road network fetch.")
        return None

    if cache_path is None:
        cache_path = ROADS_GPKG

    cache_path = Path(cache_path)
    if cache_path.exists():
        return gpd.read_file(cache_path)

    if bbox is None:
        bbox = (BBOX_LAT_MAX, BBOX_LAT_MIN, BBOX_LON_MAX, BBOX_LON_MIN)

    north, south, east, west = bbox

    G = ox.graph_from_bbox(
        north=north, south=south, east=east, west=west,
        network_type="drive",
        retain_all=True,
    )
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # Keep useful columns
    keep_cols = ["geometry", "highway", "name", "ref", "length"]
    available = [c for c in keep_cols if c in edges.columns]
    edges = edges[available].copy()

    # Persist
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_file(cache_path, driver="GPKG")

    return edges


def fetch_power_lines(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    cache_path: Optional[Path] = None,
) -> "Optional[gpd.GeoDataFrame]":
    """Download power lines from OpenStreetMap via Overpass.

    Parameters
    ----------
    bbox : tuple (north, south, east, west) in WGS84, optional
        Defaults to the Fuerteventura bounding box from config.
    cache_path : Path, optional
        GeoPackage file to cache the result. Defaults to ``POWER_LINES_GPKG``.

    Returns
    -------
    geopandas.GeoDataFrame with geometry, voltage, name, operator columns,
    or None if GIS libs are missing.
    """
    if not _HAS_GIS:
        print("GIS libraries not available. Skipping power lines fetch.")
        return None

    if cache_path is None:
        cache_path = POWER_LINES_GPKG

    cache_path = Path(cache_path)
    if cache_path.exists():
        return gpd.read_file(cache_path)

    if bbox is None:
        bbox = (BBOX_LAT_MAX, BBOX_LAT_MIN, BBOX_LON_MAX, BBOX_LON_MIN)

    north, south, east, west = bbox

    # Overpass query for power=line within bounding box
    query = f"""
    [out:json][timeout:60];
    (
      way["power"="line"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """

    gdf = ox.features_from_bbox(
        north=north, south=south, east=east, west=west,
        tags={"power": "line"},
    )

    if gdf.empty:
        print("No power lines found in bounding box.")
        return gdf

    # Keep useful columns
    keep_cols = ["geometry", "voltage", "name", "operator"]
    available = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[available].copy()

    # Persist
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache_path, driver="GPKG")

    return gdf


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def casualties_to_geodataframe(df: pd.DataFrame) -> "Optional[gpd.GeoDataFrame]":
    """Convert a DataFrame with utm_x, utm_y columns to a GeoDataFrame.

    Points are created in EPSG:32628 (UTM 28N).

    Returns None if geopandas is not installed or if utm_x/utm_y are missing.
    """
    if not _HAS_GIS:
        print("GIS libraries not available. Cannot create GeoDataFrame.")
        return None

    if "utm_x" not in df.columns or "utm_y" not in df.columns:
        raise ValueError("DataFrame must contain 'utm_x' and 'utm_y' columns.")

    # Drop rows without coordinates
    valid = df.dropna(subset=["utm_x", "utm_y"]).copy()

    geometry = [Point(x, y) for x, y in zip(valid["utm_x"], valid["utm_y"])]
    gdf = gpd.GeoDataFrame(valid, geometry=geometry, crs=SRC_CRS)

    return gdf


# ---------------------------------------------------------------------------
# Nearest-road computation
# ---------------------------------------------------------------------------

def compute_road_distances(
    casualties_gdf: "gpd.GeoDataFrame",
    roads_gdf: "gpd.GeoDataFrame",
) -> pd.DataFrame:
    """Compute the distance from each casualty to the nearest road segment.

    Both inputs must be GeoDataFrames. Roads are reprojected to UTM 28N
    (EPSG:32628) if necessary so distances are in metres.

    Parameters
    ----------
    casualties_gdf : gpd.GeoDataFrame
        Casualty points in EPSG:32628.
    roads_gdf : gpd.GeoDataFrame
        Road edges (typically in EPSG:4326 from OSM).

    Returns
    -------
    pd.DataFrame with columns:
        index, road_distance_m, nearest_road_type, nearest_road_name,
        nearest_road_ref
    """
    if not _HAS_GIS:
        print("GIS libraries not available.")
        return pd.DataFrame()

    # Ensure both are in the projected CRS for metric accuracy
    if casualties_gdf.crs is None:
        casualties_gdf = casualties_gdf.set_crs(SRC_CRS)
    if casualties_gdf.crs.to_epsg() != 32628:
        casualties_gdf = casualties_gdf.to_crs(SRC_CRS)

    if roads_gdf.crs is None:
        roads_gdf = roads_gdf.set_crs(DST_CRS)
    if roads_gdf.crs.to_epsg() != 32628:
        roads_gdf = roads_gdf.to_crs(SRC_CRS)

    # Build spatial index on roads for fast lookup
    roads_sindex = roads_gdf.sindex

    results = []
    for idx, row in casualties_gdf.iterrows():
        pt = row.geometry
        # Find candidate roads near the point (within ~5km buffer)
        candidate_idx = list(roads_sindex.nearest(pt, max_distance=5000))
        if len(candidate_idx) == 0:
            # Fallback: brute-force nearest
            candidate_idx = list(range(len(roads_gdf)))

        # Handle different rtree/pygeos return formats
        if isinstance(candidate_idx, np.ndarray) and candidate_idx.ndim == 2:
            candidate_idx = candidate_idx[1].tolist()
        elif hasattr(candidate_idx, '__len__') and len(candidate_idx) > 0:
            if isinstance(candidate_idx[0], (list, tuple, np.ndarray)):
                candidate_idx = [c for sublist in candidate_idx for c in sublist]

        if len(candidate_idx) == 0:
            results.append({
                "index": idx,
                "road_distance_m": np.nan,
                "nearest_road_type": None,
                "nearest_road_name": None,
                "nearest_road_ref": None,
            })
            continue

        candidates = roads_gdf.iloc[candidate_idx]
        distances = candidates.geometry.distance(pt)
        nearest_idx = distances.idxmin()
        nearest_road = roads_gdf.loc[nearest_idx]

        highway_tag = nearest_road.get("highway", "") if hasattr(nearest_road, "get") else ""
        road_name = nearest_road.get("name", None) if hasattr(nearest_road, "get") else None
        road_ref = nearest_road.get("ref", None) if hasattr(nearest_road, "get") else None

        results.append({
            "index": idx,
            "road_distance_m": round(distances.min(), 1),
            "nearest_road_type": classify_road(str(highway_tag) if highway_tag else ""),
            "nearest_road_name": road_name if pd.notna(road_name) else None,
            "nearest_road_ref": road_ref if pd.notna(road_ref) else None,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_road_distances(cache_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load precomputed road distances from parquet if the file exists.

    Parameters
    ----------
    cache_path : Path, optional
        Defaults to ``ROAD_DISTANCES_PARQUET`` from config.

    Returns
    -------
    pd.DataFrame or None if the file does not exist.
    """
    if cache_path is None:
        cache_path = ROAD_DISTANCES_PARQUET

    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    return None
