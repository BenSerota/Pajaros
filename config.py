"""
Central configuration for the Fuerteventura Bird Mortality Dashboard.
Column mappings, color palettes, constants, and metadata.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GEO_DIR = DATA_DIR / "geo"

RAW_XLS = RAW_DIR / "Info_colisiones_SACORP_hasta_0326.xlsx"
CLEAN_PARQUET = PROCESSED_DIR / "casualties_clean.parquet"
SPECIES_PARQUET = PROCESSED_DIR / "species_lookup.parquet"
ROAD_DISTANCES_PARQUET = PROCESSED_DIR / "road_distances.parquet"
ROADS_GPKG = GEO_DIR / "fuerteventura_roads.gpkg"
POWER_LINES_GPKG = GEO_DIR / "fuerteventura_power_lines.gpkg"

# ── Coordinate System ────────────────────────────────────────────────────────
SRC_CRS = "EPSG:32628"   # ETRS89 UTM Zone 28N (Canary Islands)
DST_CRS = "EPSG:4326"    # WGS84 lat/lon

# Fuerteventura bounding box (lat/lon) — for OSM queries
BBOX_LAT_MIN = 28.0
BBOX_LAT_MAX = 28.8
BBOX_LON_MIN = -14.6
BBOX_LON_MAX = -13.7

# Map center
MAP_CENTER_LAT = 28.40
MAP_CENTER_LON = -14.10
MAP_DEFAULT_ZOOM = 10

# ── Column Mapping ───────────────────────────────────────────────────────────
# Raw Excel column name → clean Python column name
COLUMN_MAP = {
    "Empresa del Estudio": "company",
    "Estudio": "study",
    "Muestreo": "survey",
    "Línea": "line",
    "Accidentes relacionados": "related_accidents",
    "Event \\ Tipo": "event_type",
    "Causante": "causante",
    "Título": "title",
    "Tipo de Impacto": "impact_type",
    "Fecha": "date",
    "Fecha local": "date_local",
    "Zona horaria": "timezone",
    "SinFL \\ Tipología": "tipologia",
    "Provincia": "provincia",
    "Municipio": "municipio",
    "Comuna": "comuna",
    "Distrito": "distrito",
    "Paraje": "paraje",
    "Localidad": "localidad",
    "Centro Poblado": "centro_poblado",
    "Observador": "observer",
    # There are two "Vano" columns (index 21 and 22) — use positional renaming
    "Vano Completado": "span_id",
    "Apoyo más cercano": "nearest_pylon",
    "Situación de los restos": "remains_position",
    "Coordenada X": "utm_x",
    "Coordenada Y": "utm_y",
    "Sistema de Coordenadas": "coord_system",
    "Distancia del observador al localizar los restos (m)": "observer_distance_m",
    "Tamaño relativo de los restos": "relative_size",
    "Separación de los dispositivos de señalización (m)": "signal_spacing_m",
    "Estado de la señalización": "signal_condition",
    "Vano accidente / Zona de la subestación": "accident_zone",
    "Colisión": "is_collision",
    "Señalización": "has_signaling",
    "Tipo de Señalización": "signal_type",
    "Electrocución": "electrocution",
    "¿Está el apoyo aislado? ": "pylon_isolated",
    "Otra Causa": "other_cause",
    "¿Conoce la especie?": "species_known",
    "List Victim Cause \\ Causa": "victim_cause",
    "Ave Colisionada": "species_raw",
    "Cantidad de Ejemplares": "specimen_count",
    "EURING": "euring_code",
    "Lista Roja UICN": "iucn_status",
    "Catálogo Español": "spanish_catalog",
    "Catálogo Autonómico": "regional_catalog",
    "MINAGRI": "minagri",
    "CITES": "cites",
    "CHILE": "chile",
    "Especie focal": "is_focal",
    "Descripción si se desconoce la especie": "unknown_species_desc",
    "Sexo": "sex",
    "Edad": "age",
    "Tipo de restos": "remains_type",
    "Estado de los restos": "remains_state",
    "Antigüedad de los restos": "remains_age",
    "Evidencias de Carroñeo": "scavenging",
    "Marcas": "bands",
    "Infraestructuras cercanas": "nearby_infrastructure",
    "Fotografías": "photos",
}

# Columns to keep after renaming (drop unused)
COLUMNS_KEEP = [
    "study", "line", "event_type", "causante",
    "date",
    "observer", "vano_raw_2", "nearest_pylon", "remains_position",
    "utm_x", "utm_y", "coord_system",
    "observer_distance_m", "relative_size",
    "signal_spacing_m", "signal_condition",
    "is_collision", "has_signaling", "signal_type",
    "species_raw", "euring_code",
    "iucn_status", "spanish_catalog", "regional_catalog",
    "is_focal", "unknown_species_desc",
    "sex", "age", "remains_type", "remains_state", "remains_age",
    "scavenging", "bands", "nearby_infrastructure",
]

# ── Signaling Types ──────────────────────────────────────────────────────────
SIGNAL_TYPE_NORM = {
    "Triple Aspa": "Triple Aspa",
    "Aspa Corta": "Aspa Corta",
    "Espirales  Amarillas": "Espirales Amarillas",  # double space in raw
    "Espirales": "Espirales Amarillas",              # merge single record
    "Sin Señalización": "Sin Señalización",
    "Tiras en X": "Tiras en X",
}
SIGNAL_TYPES_ORDERED = ["Triple Aspa", "Aspa Corta", "Espirales Amarillas", "Tiras en X", "Sin Señalización", "UV (Ultravioleta)"]

# ── Signal Condition ─────────────────────────────────────────────────────────
SIGNAL_CONDITION_NORM = {
    "Bueno": "Bueno",
    "Parcialmente ausente": "Parcialmente ausente",
    "Deteriorado": "Degradado",
    "Decolorado": "Degradado",
}

# ── Municipio Normalization ──────────────────────────────────────────────────
MUNICIPIO_NORM = {
    "LA OLIVA": "La Oliva",
    "La oliva": "La Oliva",
    "La Oilva": "La Oliva",
    "la Oliva": "La Oliva",
    "PUERTO DEL ROSARIO": "Puerto del Rosario",
    "Puerto del Rosario": "Puerto del Rosario",
    "Puerto Del Rosario": "Puerto del Rosario",
    "TUINEJE": "Tuineje",
    "Tuineje": "Tuineje",
    "Pajara": "Pájara",
    "Pájara": "Pájara",
    "PAJARA": "Pájara",
    "ANTIGUA": "Antigua",
    "Antigua": "Antigua",
    "BETANCURIA": "Betancuria",
    "Betancuria": "Betancuria",
    "LAS PALMAS": "Las Palmas",
    "Las Palmas": "Las Palmas",
    "Las palmas": "Las Palmas",
    "las Palmas": "Las Palmas",
}

# ── Line Metadata ────────────────────────────────────────────────────────────
LINE_META = {
    "L/Gran Tarajal-Matas Blancas 132 kV": {
        "voltage": 132, "corridor": "Gran Tarajal – Matas Blancas", "zone": "Sur",
    },
    "66 kV GRAN TARAJAL MATAS BLANCAS": {
        "voltage": 66, "corridor": "Gran Tarajal – Matas Blancas", "zone": "Sur",
    },
    "L/ La Oliva-Pto. del Rosario 132 kV": {
        "voltage": 132, "corridor": "La Oliva – Pto. del Rosario", "zone": "Norte",
    },
    "66 kV CORRALEJO SALINAS": {
        "voltage": 66, "corridor": "Corralejo – Salinas", "zone": "Norte",
    },
    "L/Pto. del Rosario-Gran Tarajal 132 kV": {
        "voltage": 132, "corridor": "Pto. del Rosario – Gran Tarajal", "zone": "Medio",
    },
    "66 kV SALINAS GRAN TARAJAL": {
        "voltage": 66, "corridor": "Salinas – Gran Tarajal", "zone": "Medio",
    },
}

# ── Study Metadata ───────────────────────────────────────────────────────────
STUDY_META = {
    "Seguimiento PVA_FV_sur": {
        "label": "PVA Sur", "zone": "Sur", "start": "2018-03", "end": "2025-09",
    },
    "Seguimiento PVA_FV_norte": {
        "label": "PVA Norte", "zone": "Norte", "start": "2019-03", "end": "2025-09",
    },
    "Seguimiento PVA_FV_medio": {
        "label": "PVA Medio", "zone": "Medio", "start": "2024-09", "end": "2025-07",
    },
    "PVA_CRJ-LSL_19-24": {
        "label": "PVA CRJ-LSL", "zone": "Norte", "start": "2019-03", "end": "2019-10",
    },
}

# ── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    # Patrones de actividad
    "Nocturna": "#1a237e",
    "Crepuscular": "#6a1b9a",
    "Diurna": "#e65100",
    "Desconocida": "#757575",

    # Voltage
    "66kV": "#2196F3",
    "132kV": "#FF5722",

    # Event type
    "Accidente": "#D32F2F",
    "Incidente": "#FFA726",

    # Signaling types
    "Triple Aspa": "#1976D2",
    "Aspa Corta": "#388E3C",
    "Espirales Amarillas": "#FBC02D",
    "Sin Señalización": "#9E9E9E",
    "UV (Ultravioleta)": "#7B1FA2",

    # Conservation status
    "En peligro de extinción": "#B71C1C",
    "Vulnerable": "#E65100",
    "Riesgo menor-casi amenazada": "#F9A825",
    "Riesgo menor-preocupación menor": "#4CAF50",

    # Remains freshness
    "Fresco": "#C62828",
    "En putrefacción": "#EF6C00",
    "Esqueleto expuesto": "#78909C",
    "Cerco de plumas": "#B0BEC5",
    "Seco": "#A1887F",

    # Zones
    "Sur": "#E91E63",
    "Norte": "#00BCD4",
    "Medio": "#8BC34A",
}

# ── Significance Thresholds ──────────────────────────────────────────────────
SIG_LEVELS = [
    (0.001, "***", "Altamente significativo", "#4CAF50"),
    (0.01,  "**",  "Muy significativo",       "#FFC107"),
    (0.05,  "*",   "Significativo",           "#FF9800"),
    (1.0,   "ns",  "No significativo",        "#9E9E9E"),
]

# ── Road Classification (OSM highway tags) ───────────────────────────────────
ROAD_CLASSIFICATION = {
    "motorway": "Autopista/autovía",
    "trunk": "Autopista/autovía",
    "primary": "Carretera principal",
    "secondary": "Carretera principal",
    "tertiary": "Carretera local",
    "residential": "Camino/pista",
    "unclassified": "Camino/pista",
    "track": "Camino/pista",
    "service": "Camino/pista",
}
ROAD_TYPES_ORDERED = ["Autopista/autovía", "Carretera principal", "Carretera local", "Camino/pista"]

# ── Road Distance Bins ───────────────────────────────────────────────────────
ROAD_DISTANCE_BINS = [0, 50, 200, 500, 1000, float("inf")]
ROAD_DISTANCE_LABELS = ["0–50m", "50–200m", "200–500m", "500m–1km", ">1km"]

# ── Spanish Labels ───────────────────────────────────────────────────────────
ACTIVITY_LABELS = {
    "Nocturna": "Nocturna",
    "Crepuscular": "Crepuscular",
    "Diurna": "Diurna",
    "Desconocida": "Desconocida",
}

# ── Remains Age Ordering ─────────────────────────────────────────────────────
REMAINS_AGE_ORDER = ["1-2 días", "1 semana", "1 mes", "más de 1 mes", "Indeterminado"]

# ── Season Definition (Northern Hemisphere, Canary climate) ──────────────────
MONTH_TO_SEASON = {
    1: "Invierno", 2: "Invierno", 3: "Primavera",
    4: "Primavera", 5: "Primavera", 6: "Verano",
    7: "Verano", 8: "Verano", 9: "Otoño",
    10: "Otoño", 11: "Otoño", 12: "Invierno",
}

# ── UV Signal Configuration ──────────────────────────────────────────────────
UV_LINE = "L/Gran Tarajal-Matas Blancas 132 kV"
UV_LINE_LABEL = "GT–MB 132kV"
UV_VANOS_RAW = ["89-90", "90-91", "91-92"]  # pylon-to-pylon numbers
UV_INSTALL_DATE = "2024-06-01"
UV_SIGNAL_LABEL = "UV (Ultravioleta)"

# ── Dashboard Layout ─────────────────────────────────────────────────────────
PAGE_ICON = "\U0001F985"  # eagle emoji
PAGE_TITLE = "Siniestralidad Avifauna — Fuerteventura"
LAYOUT = "wide"

# Chart defaults
CHART_HEIGHT = 450
CHART_FONT_FAMILY = "Inter, system-ui, -apple-system, sans-serif"
CHART_BG = "rgba(0,0,0,0)"
CHART_TEMPLATE = "plotly_white"
SOURCE_ANNOTATION = "Datos: BIOSFERA XXI, 2018–2026"
