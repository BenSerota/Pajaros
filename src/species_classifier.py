"""
Species activity-pattern classifier for Fuerteventura power line mortality data.

Classifies each species_clean value into one of four categories:
  - Nocturnal: active primarily at night, attracted to artificial light
  - Crepuscular: active at dawn/dusk, often active at night too
  - Diurnal: active during daylight hours
  - Unknown: unidentified taxa or indeterminate specimens

Used to test whether nocturnal/crepuscular birds are disproportionately
killed by power lines, supporting the case for UV-reflective markers.

Reference dataset: 62 unique species from Fuerteventura mortality surveys.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Nocturnal species
# Procellariiformes and storm-petrels — strictly nocturnal on land during
# breeding season; strongly attracted to artificial light sources.
# ---------------------------------------------------------------------------
_NOCTURNAL = {
    # Cory's shearwater (Atlantic) — nocturnal colony visits during breeding
    "Calonectris borealis / Pardela cenicienta atlántica/Pardela atlántica",
    # Cory's shearwater (sensu lato) — nocturnal colony visits
    "Calonectris diomedea / pardela cenicienta",
    # Cory's shearwater (Mediterranean form label) — nocturnal colony visits
    "Calonectris diomedea / Pardela cenicienta/Pardela cenicienta  mediterránea",
    # Bulwer's petrel — small nocturnal Procellariiform seabird
    "Bulweria bulwerii / petrel de Bulwer",
    # White-faced storm-petrel — nocturnal at breeding colonies
    "Pelagodroma marina / paíño pechialbo",
}

# ---------------------------------------------------------------------------
# Crepuscular species
# Active primarily at dawn and dusk; several are also active at night,
# making them vulnerable during low-light periods.
# ---------------------------------------------------------------------------
_CREPUSCULAR = {
    # Eurasian stone-curlew — classically crepuscular-nocturnal wader
    "Burhinus oedicnemus / alcaraván común",
    # Canary Islands stone-curlew (insularum ssp.) — same crepuscular-nocturnal behavior
    "Burhinus oedicnemus insularum / Alcaraván canario oriental o majorero",
    # Cream-colored courser — crepuscular in arid desert habitats
    "Cursorius cursor / corredor sahariano",
    # Black-bellied sandgrouse — crepuscular flights to water at dawn/dusk
    "Pterocles orientalis / ganga ortega",
    # Common quail — migrates at night, generally crepuscular activity pattern
    "Coturnix coturnix / codorniz común",
}

# ---------------------------------------------------------------------------
# Unknown / indeterminate taxa
# Either unidentified specimens or genus-level IDs where activity pattern
# cannot be reliably assigned.
# ---------------------------------------------------------------------------
_UNKNOWN = {
    # Unidentified specimen
    "Unknown / No identificada",
    # Indeterminate pigeon/dove — could be feral pigeon or other Columba sp.
    "Columba sp. / paloma indeterminada",
    # Indeterminate duck/teal
    "Anas sp. / ánade/cerceta indeterminada",
    # Indeterminate gull
    "Larus sp. / gaviota/gavión indeterminado",
    # Indeterminate pipit
    "Anthus spp. / bisbita indeterminado",
}

# ---------------------------------------------------------------------------
# Diurnal species
# All remaining identified species — active primarily during daylight.
# ---------------------------------------------------------------------------
_DIURNAL = {
    # Barbary partridge — diurnal gamebird
    "Alectoris barbara / perdiz moruna",
    # Greylag goose — diurnal waterfowl
    "Anser anser / ánsar común",
    # Berthelot's pipit — diurnal Macaronesian passerine
    "Anthus berthelotii / bisbita caminero",
    # Common swift — diurnal aerial forager (migrates at night but collisions
    # in Fuerteventura are daytime aerial foraging events)
    "Apus apus / vencejo común",
    # Plain swift — same rationale as common swift
    "Apus unicolor / vencejo unicolor",
    # Grey heron — diurnal wading bird
    "Ardea cinerea / garza real",
    # Cattle egret — diurnal, follows livestock
    "Bubulcus ibis / garcilla bueyera",
    # Trumpeter finch — diurnal desert passerine
    "Bucanetes githagineus / camachuelo trompetero",
    # Houbara bustard — diurnal, open-ground species
    "Chlamydotis undulata / avutarda hubara",
    # Rock dove / feral pigeon — diurnal
    "Columba livia / paloma bravía",
    # Common raven — diurnal corvid
    "Corvus corax / cuervo grande",
    # Canarian raven (canariensis ssp.) — diurnal corvid
    "Corvus corax canariensis / Cuervo canario",
    # Little egret — diurnal wading bird
    "Egretta garzetta / garceta común",
    # Common kestrel — diurnal raptor
    "Falco tinnunculus / cernícalo vulgar",
    # European pied flycatcher — diurnal migrant passerine
    "Ficedula hypoleuca / papamoscas cerrojillo",
    # Common moorhen — diurnal waterbird
    "Gallinula chloropus / gallineta común",
    # Great grey shrike — diurnal predatory passerine
    "Lanius excubitor / alcaudón norteño",
    # Iberian grey shrike — diurnal predatory passerine
    "Lanius meridionalis / alcaudón real",
    # Lesser black-backed gull — diurnal gull
    "Larus fuscus / gaviota sombría",
    # Yellow-legged gull — diurnal gull
    "Larus michahellis / gaviota patiamarilla",
    # Marabou stork — diurnal soaring bird
    "Leptoptilos crumenifer / marabú africano",
    # Egyptian vulture — diurnal raptor/scavenger
    "Neophron percnopterus / alimoche común",
    # Whimbrel — primarily diurnal migrant wader
    "Numenius phaeopus / zarapito trinador",
    # Spanish sparrow — diurnal granivorous passerine
    "Passer hispaniolensis / gorrión moruno",
    # Black redstart — diurnal passerine
    "Phoenicurus ochruros / colirrojo tizón",
    # Eurasian spoonbill — diurnal wading bird
    "Platalea leucorodia / espátula común",
    # Canary Islands chat — diurnal endemic passerine
    "Saxicola dacotiae / tarabilla canaria",
    # Eurasian collared dove — diurnal
    "Streptopelia decaocto / tórtola turca",
    # Laughing dove — diurnal
    "Streptopelia senegalensis / tórtola senegalesa",
    # European turtle dove — diurnal
    "Streptopelia turtur / tórtola europea",
    # Eurasian blackcap — diurnal warbler
    "Sylvia atricapilla / curruca capirotada",
    # Spectacled warbler — diurnal warbler
    "Sylvia conspicillata / curruca tomillera",
    # Sardinian warbler — diurnal warbler
    "Sylvia melanocephala / curruca cabecinegra",
    # Ruddy shelduck — diurnal waterfowl
    "Tadorna ferruginea / tarro canelo",
    # Eurasian hoopoe — diurnal insectivore
    "Upupa epops / abubilla",
}


# ---------------------------------------------------------------------------
# Build the master lookup: species_clean -> activity_pattern
# ---------------------------------------------------------------------------
SPECIES_ACTIVITY: dict[str, str] = {}
for sp in _NOCTURNAL:
    SPECIES_ACTIVITY[sp] = "Nocturna"
for sp in _CREPUSCULAR:
    SPECIES_ACTIVITY[sp] = "Crepuscular"
for sp in _DIURNAL:
    SPECIES_ACTIVITY[sp] = "Diurna"
for sp in _UNKNOWN:
    SPECIES_ACTIVITY[sp] = "Desconocida"


# ---------------------------------------------------------------------------
# Convenience sets for summary / reporting
# ---------------------------------------------------------------------------
NOCTURNAL_SPECIES: set[str] = _NOCTURNAL
CREPUSCULAR_SPECIES: set[str] = _CREPUSCULAR
DIURNAL_SPECIES: set[str] = _DIURNAL
UNKNOWN_SPECIES: set[str] = _UNKNOWN


# ---------------------------------------------------------------------------
# Genus-level fallback lookup
# Extracted from the species_clean strings (text before the first space).
# Used when an exact match is not found (e.g. a new species_clean variant).
# ---------------------------------------------------------------------------
_GENUS_ACTIVITY: dict[str, str] = {}
for species_clean, category in SPECIES_ACTIVITY.items():
    genus = species_clean.split()[0]
    # Only set genus if not already present (first match wins, which is fine
    # because all species within a genus share the same category here).
    if genus not in _GENUS_ACTIVITY:
        _GENUS_ACTIVITY[genus] = category


def _classify_one(species_clean: str) -> str:
    """Classify a single species_clean value.

    Lookup order:
      1. Exact match in SPECIES_ACTIVITY
      2. Genus-level fallback (first word of the binomial)
      3. "Unknown"
    """
    if species_clean in SPECIES_ACTIVITY:
        return SPECIES_ACTIVITY[species_clean]

    genus = species_clean.split()[0] if isinstance(species_clean, str) else ""
    if genus in _GENUS_ACTIVITY:
        return _GENUS_ACTIVITY[genus]

    return "Desconocida"


def classify_species(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``activity_pattern`` column to *df* based on ``species_clean``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``species_clean`` column (produced by the cleaning
        pipeline in ``data_cleaning.py``).

    Returns
    -------
    pd.DataFrame
        A copy of the input with an additional ``activity_pattern`` column
        containing one of: Nocturnal, Crepuscular, Diurnal, Unknown.
    """
    if "species_clean" not in df.columns:
        raise ValueError("DataFrame must contain a 'species_clean' column")

    out = df.copy()
    out["activity_pattern"] = out["species_clean"].map(_classify_one)
    return out
