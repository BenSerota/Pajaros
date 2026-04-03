"""
Pagina 7: Metodologia
Diccionario de datos, metodos estadisticos, notas de calidad de datos, limitaciones y referencias.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_data
from src.filters import apply_filters, get_filter_summary

# -- Configuracion de pagina --------------------------------------------------
st.header("Metodologia")
st.caption("Fuentes de datos, definiciones, metodos estadisticos y limitaciones")

# -- Cargar y filtrar ----------------------------------------------------------
df_full = load_data()
df = apply_filters(df_full)
subtitle = get_filter_summary(df, df_full)

# =============================================================================
# SECCION 1: Fuente de Datos
# =============================================================================
st.subheader("Fuente de Datos")

total_records = len(df_full)
year_min = int(df_full["year"].min()) if df_full["year"].notna().any() else "N/D"
year_max = int(df_full["year"].max()) if df_full["year"].notna().any() else "N/D"
n_species = df_full["species_clean"].nunique()

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown(f"""
    | Elemento | Detalle |
    |---|---|
    | **Estudio** | Plan de Vigilancia Ambiental (PVA) de BIOSFERA XXI |
    | **Periodo** | Marzo 2018 -- Septiembre 2025 |
    | **Ubicacion** | Fuerteventura, Islas Canarias, Espana |
    | **Lineas monitorizadas** | 6 (3 a 132kV, 3 a 66kV) |
    | **Registros totales** | {total_records:,} |
    | **Especies identificadas** | {n_species} |
    """)

with col_info2:
    st.markdown("""
    | Elemento | Detalle |
    |---|---|
    | **CRS de origen** | ETRS89 UTM Zona 28N (EPSG:32628) |
    | **CRS de visualizacion** | WGS84 (EPSG:4326) |
    | **Zonas de monitoreo** | Sur, Norte, Medio |
    | **Operador** | Red Electrica de Espana (REE) |
    | **Contratista** | BIOSFERA XXI, S.L. |
    | **Formato de datos** | Excel (.xls), limpiado a Parquet |
    """)

st.divider()

# =============================================================================
# SECCION 2: Diccionario de Datos
# =============================================================================
st.subheader("Diccionario de Datos")
st.caption("Columnas clave del conjunto de datos limpio con descripciones y valores de ejemplo")

# Construir diccionario de datos desde el dataframe real
key_columns = {
    "species_clean": {
        "description": "Nombre de especie normalizado: Cientifico / Comun",
        "type": "string",
        "example": "Calonectris borealis / Pardela cenicienta",
    },
    "species_scientific": {
        "description": "Nombre cientifico binomial",
        "type": "string",
        "example": "Calonectris borealis",
    },
    "species_common": {
        "description": "Nombre comun en espanol",
        "type": "string",
        "example": "Pardela cenicienta",
    },
    "activity_pattern": {
        "description": "Clasificacion de actividad de la especie: Nocturna, Crepuscular, Diurna, Desconocida",
        "type": "string",
        "example": "Nocturna",
    },
    "conservation_score": {
        "description": "Puntuacion compuesta de prioridad de conservacion (UICN + nacional + regional + focal)",
        "type": "float",
        "example": "7.0",
    },
    "iucn_status": {
        "description": "Categoria de la Lista Roja de la UICN",
        "type": "string",
        "example": "Vulnerable",
    },
    "spanish_catalog": {
        "description": "Estado en el Catalogo Nacional de Especies Amenazadas",
        "type": "string",
        "example": "En peligro de extincion",
    },
    "regional_catalog": {
        "description": "Estado en el Catalogo Regional de Especies Protegidas de Canarias",
        "type": "string",
        "example": "De interes especial",
    },
    "is_focal": {
        "description": "Si la especie esta designada como focal para el seguimiento",
        "type": "boolean",
        "example": "True / False",
    },
    "line_label": {
        "description": "Etiqueta corta de la linea electrica (p.ej. GT-MB 132kV)",
        "type": "string",
        "example": "GT-MB 132kV",
    },
    "voltage": {
        "description": "Tension de la linea: 66 o 132 kV",
        "type": "integer",
        "example": "132",
    },
    "signal_type": {
        "description": "Tipo de dispositivo anticolision instalado en el vano",
        "type": "string",
        "example": "Triple Aspa",
    },
    "signal_spacing_m": {
        "description": "Separacion entre dispositivos anticolision en metros",
        "type": "float",
        "example": "10, 20",
    },
    "signal_condition": {
        "description": "Estado fisico de los dispositivos de senalizacion: Bueno, Parcialmente ausente, Degradado",
        "type": "string",
        "example": "Bueno",
    },
    "event_type": {
        "description": "Accidente (mortalidad confirmada) o Incidente (evidencia de colision)",
        "type": "string",
        "example": "Accidente",
    },
    "date": {
        "description": "Fecha del registro",
        "type": "datetime",
        "example": "2022-10-15",
    },
    "year": {
        "description": "Ano extraido de la fecha",
        "type": "integer",
        "example": "2022",
    },
    "month": {
        "description": "Numero de mes (1-12)",
        "type": "integer",
        "example": "10",
    },
    "latitude": {
        "description": "Latitud WGS84 (convertida desde UTM)",
        "type": "float",
        "example": "28.35",
    },
    "longitude": {
        "description": "Longitud WGS84 (convertida desde UTM)",
        "type": "float",
        "example": "-14.05",
    },
    "zone": {
        "description": "Zona geografica de la linea electrica: Sur, Norte, Medio",
        "type": "string",
        "example": "Sur",
    },
    "span_id": {
        "description": "Identificador del vano especifico (entre dos apoyos)",
        "type": "string",
        "example": "V-23",
    },
    "observer": {
        "description": "Nombre del observador de campo que registro el evento",
        "type": "string",
        "example": "(nombre del observador)",
    },
    "remains_type": {
        "description": "Tipo de restos encontrados (cadaver, plumas, esqueleto, etc.)",
        "type": "string",
        "example": "Cadaver",
    },
    "remains_state": {
        "description": "Estado de los restos encontrados",
        "type": "string",
        "example": "Fresco",
    },
    "remains_age": {
        "description": "Antiguedad estimada de los restos: 1-2 dias, 1 semana, 1 mes, mas de 1 mes",
        "type": "string",
        "example": "1 semana",
    },
    "scavenging": {
        "description": "Si se encontraron evidencias de carroneo",
        "type": "boolean",
        "example": "True / False",
    },
    "sex": {
        "description": "Sexo del ejemplar si fue determinado",
        "type": "string",
        "example": "Macho, Hembra, Indeterminado",
    },
    "age": {
        "description": "Clase de edad del ejemplar si fue determinada",
        "type": "string",
        "example": "Adulto, Juvenil, Indeterminado",
    },
}

dict_rows = []
for col_name, info in key_columns.items():
    null_count = int(df_full[col_name].isna().sum()) if col_name in df_full.columns else "N/D"
    dict_rows.append({
        "Columna": col_name,
        "Descripcion": info["description"],
        "Tipo": info["type"],
        "Valores de ejemplo": info["example"],
        "Nulos": null_count,
    })

dict_df = pd.DataFrame(dict_rows)
st.dataframe(dict_df, use_container_width=True, hide_index=True, height=600)

st.divider()

# =============================================================================
# SECCION 3: Clasificacion de Actividad de Especies
# =============================================================================
st.subheader("Clasificacion de Actividad de Especies")

st.markdown("""
Cada especie del conjunto de datos se clasifica en uno de cuatro patrones de actividad
basados en la literatura ornitologica establecida:

- **Nocturna**: Activa principalmente de noche. Incluye Procellariiformes (pardelas, petreles,
  paiños) que son estrictamente nocturnos en tierra durante la temporada de cria y fuertemente
  atraidos por las fuentes de luz artificial.
- **Crepuscular**: Activa principalmente al amanecer y al anochecer, a menudo tambien activa
  de noche. Incluye alcaravanes, corredores, gangas y codornices.
- **Diurna**: Activa durante las horas de luz. La mayoria de las especies del conjunto de datos.
- **Desconocida**: Especimenes no identificados o identificaciones a nivel de genero donde el
  patron de actividad no puede asignarse de forma fiable.
""")

col_noct, col_crep, col_diurn = st.columns(3)

with col_noct:
    st.markdown("**Especies nocturnas**")
    nocturnal_spp = df_full[df_full["activity_pattern"] == "Nocturna"]["species_clean"].unique()
    for sp in sorted(nocturnal_spp):
        sci = sp.split(" / ")[0] if " / " in sp else sp
        st.markdown(f"- *{sci}*")

with col_crep:
    st.markdown("**Especies crepusculares**")
    crep_spp = df_full[df_full["activity_pattern"] == "Crepuscular"]["species_clean"].unique()
    for sp in sorted(crep_spp):
        sci = sp.split(" / ")[0] if " / " in sp else sp
        st.markdown(f"- *{sci}*")

with col_diurn:
    st.markdown("**Especies diurnas (muestra)**")
    diurnal_spp = df_full[df_full["activity_pattern"] == "Diurna"]["species_clean"].unique()
    for sp in sorted(diurnal_spp)[:10]:
        sci = sp.split(" / ")[0] if " / " in sp else sp
        st.markdown(f"- *{sci}*")
    remaining = len(diurnal_spp) - 10
    if remaining > 0:
        st.caption(f"...y {remaining} especies diurnas mas")

st.markdown("""
**Nota sobre Procellariiformes:** Las pardelas, petreles y paiños son estrictamente nocturnos
al visitar las colonias de cria terrestres. Su vulnerabilidad a la luz artificial y a las
colisiones con lineas electricas durante la noche esta bien documentada en la literatura
ornitologica. Esta clasificacion es fundamental para el argumento de que las marcas anticolision
visuales son insuficientes para estas especies.
""")

st.divider()

# =============================================================================
# SECCION 4: Pruebas Estadisticas
# =============================================================================
st.subheader("Pruebas Estadisticas")
st.caption("Metodos utilizados en las paginas del panel, con supuestos y guia de interpretacion")

with st.expander("Prueba chi-cuadrado de independencia"):
    st.markdown("""
    **Que evalua:** Si dos variables categoricas son estadisticamente independientes.

    **Supuestos:**
    - Las observaciones son independientes
    - Las frecuencias esperadas en las celdas deben ser >= 5 (cuando se viola, se prefiere la prueba exacta de Fisher)

    **Cuando se usa:** Para comparar distribuciones de tipo de senalizacion entre tipos de evento,
    patrones de actividad entre tipos de senalizacion y otras tabulaciones cruzadas categoricas.

    **Tamano del efecto:** V de Cramer (0 = sin asociacion, 1 = asociacion perfecta).
    """)

with st.expander("Prueba exacta de Fisher (por pares)"):
    st.markdown("""
    **Que evalua:** Si dos grupos difieren significativamente en un resultado categorico, utilizando
    calculos de probabilidad exacta en lugar de la aproximacion chi-cuadrado.

    **Supuestos:**
    - Totales marginales fijos
    - Observaciones independientes

    **Cuando se usa:** Comparaciones post-hoc por pares despues de una prueba chi-cuadrado significativa,
    especialmente cuando los recuentos en las celdas son pequenos. Se aplica la correccion de Bonferroni
    para comparaciones multiples.

    **Resultado:** Razon de probabilidades (odds ratio) y valor p ajustado para cada par.
    """)

with st.expander("Prueba H de Kruskal-Wallis"):
    st.markdown("""
    **Que evalua:** Si las distribuciones de una variable continua difieren entre tres o mas grupos
    independientes. Alternativa no parametrica al ANOVA de un factor.

    **Supuestos:**
    - Observaciones independientes
    - Variable dependiente ordinal o continua
    - Formas de distribucion similares entre grupos (evalua desplazamiento de ubicacion)

    **Cuando se usa:** Para comparar recuentos de victimas por vano entre tipos de senalizacion,
    donde los datos de conteo suelen tener una distribucion asimetrica a la derecha y no normal.

    **Tamano del efecto:** Epsilon-cuadrado (0 = sin efecto, 1 = separacion completa).
    """)

with st.expander("Prueba de tendencia de Mann-Kendall + pendiente de Sen"):
    st.markdown("""
    **Que evalua:** Si existe una tendencia monotonica (ascendente o descendente) en una serie
    temporal. No parametrica: no asume linealidad ni normalidad.

    **Supuestos:**
    - Las observaciones estan ordenadas temporalmente
    - Sin autocorrelacion serial fuerte (puede inflar la significacion)

    **Cuando se usa:** Para evaluar si los recuentos anuales de mortalidad muestran una tendencia
    significativa creciente o decreciente a lo largo del periodo de estudio.

    **Resultado:** Tau de Kendall (-1 a +1), valor p y pendiente de Sen (mediana de todas las
    pendientes por pares, robusta ante valores atipicos).
    """)

with st.expander("Regresion binomial negativa"):
    st.markdown("""
    **Que evalua:** Modela datos de conteo (p.ej., victimas por vano) en funcion de variables
    predictoras, permitiendo sobredispersion (varianza > media).

    **Supuestos:**
    - Variable dependiente de conteo
    - Sobredispersion relativa a Poisson (alfa > 0)
    - Independencia de observaciones condicionada a los predictores

    **Cuando se usa:** Modelo multivariante que controla por tension y separacion de senalizacion
    al comparar los efectos del tipo de senalizacion sobre la mortalidad por vano.

    **Resultado:** Razones de tasas de incidencia (IRR) con intervalos de confianza. IRR > 1
    significa mayor mortalidad respecto a la categoria de referencia.
    """)

with st.expander("Prueba de Rayleigh (uniformidad circular)"):
    st.markdown("""
    **Que evalua:** Si una muestra de datos circulares (direccionales) esta distribuida uniformemente
    alrededor del circulo, o si tiene una direccion preferente.

    **Supuestos:**
    - Los datos son angulares/circulares (aqui, mes mapeado a 0-2pi)
    - Hipotesis alternativa unimodal

    **Cuando se usa:** Para evaluar si la mortalidad de aves tiene un pico estacional (distribucion
    mensual no uniforme) en lugar de estar repartida uniformemente a lo largo del ano.

    **Resultado:** Estadistico Z de Rayleigh, valor p, direccion media, longitud resultante media
    (R-bar; 0 = uniforme, 1 = perfectamente concentrada).
    """)

with st.expander("Prueba U-cuadrado de Watson (circular, dos muestras)"):
    st.markdown("""
    **Que evalua:** Si dos muestras de datos circulares provienen de la misma distribucion.

    **Supuestos:**
    - Ambas muestras son circulares/angulares
    - Distribuciones circulares continuas

    **Cuando se usa:** Para comparar los patrones estacionales (mensuales) de mortalidad de dos
    grupos, como especies nocturnas vs diurnas, para determinar si sus meses de mayor mortalidad
    difieren.

    **Resultado:** Estadistico U-cuadrado con valor p aproximado a partir de tablas de valores criticos.
    """)

with st.expander("Prueba binomial"):
    st.markdown("""
    **Que evalua:** Si una proporcion observada difiere significativamente de una proporcion
    esperada bajo la hipotesis nula.

    **Supuestos:**
    - Resultado binario (exito/fracaso)
    - Ensayos independientes
    - Numero fijo de ensayos

    **Cuando se usa:** Para evaluar si un grupo especifico de especies (p.ej., aves nocturnas)
    representa una proporcion significativamente mayor de la mortalidad de lo esperado por su
    representacion en la avifauna local.

    **Resultado:** Proporcion observada, proporcion esperada y valor p bilateral.
    """)

with st.expander("Agrupacion espacial DBSCAN"):
    st.markdown("""
    **Que evalua:** Identifica agrupaciones espaciales de eventos de mortalidad sin requerir un
    numero predefinido de clusters.

    **Parametros:**
    - **eps** (por defecto 500m): Distancia maxima entre dos puntos para ser considerados vecinos
    - **min_samples** (por defecto 3): Puntos minimos para formar una region densa (nucleo del cluster)

    **Cuando se usa:** Para detectar puntos negros de mortalidad a lo largo de los corredores de
    lineas electricas utilizando coordenadas UTM. Los puntos que no pertenecen a ningun cluster
    se etiquetan como ruido (-1).

    **Resultado:** Etiquetas de cluster para cada punto. Los puntos de ruido se excluyen de las
    estadisticas de cluster.
    """)

st.divider()

# =============================================================================
# SECCION 5: Notas de Calidad de Datos
# =============================================================================
st.subheader("Notas de Calidad de Datos")

# Conteo de nulos por columna clave
st.markdown("**Datos faltantes por columna clave:**")

null_cols = [
    "species_clean", "date", "signal_type", "signal_spacing_m", "signal_condition",
    "event_type", "iucn_status", "spanish_catalog", "regional_catalog",
    "utm_x", "utm_y", "observer", "sex", "age", "remains_type", "remains_state",
]
null_data = []
for col in null_cols:
    if col in df_full.columns:
        null_count = int(df_full[col].isna().sum())
        null_pct = null_count / len(df_full) * 100
        null_data.append({
            "Columna": col,
            "Nulos": null_count,
            "% Nulos": f"{null_pct:.1f}%",
            "Completos": f"{len(df_full) - null_count:,}",
        })

null_df = pd.DataFrame(null_data)
st.dataframe(null_df, use_container_width=True, hide_index=True)

# Distribucion de observadores
st.markdown("**Distribucion de observadores:**")
if "observer" in df_full.columns:
    observer_counts = (
        df_full["observer"]
        .value_counts(dropna=False)
        .reset_index()
    )
    observer_counts.columns = ["Observador", "Registros"]
    observer_counts["Observador"] = observer_counts["Observador"].fillna("(sin datos)")
    st.dataframe(observer_counts, use_container_width=True, hide_index=True)

st.markdown("""
**Notas sobre la completitud de los datos:**
- **Anos parciales:** 2018 (el monitoreo comenzo en marzo) y 2025 (datos hasta septiembre)
  tienen datos anuales incompletos. Estos se senalan en los analisis de tendencias.
- **Periodos de monitoreo desiguales:** La zona Sur ha sido monitorizada desde 2018, Norte desde
  2019 y Medio solo desde finales de 2024. Las comparaciones directas de recuentos absolutos entre
  zonas deben tener en cuenta las diferentes ventanas de observacion.
- **Confundido separacion-tension:** La separacion de senalizacion (10m vs 20m) esta casi
  perfectamente confundida con la tension (66kV vs 132kV). Cualquier diferencia observada entre
  separaciones no puede separarse de los efectos de tension.
""")

st.divider()

# =============================================================================
# SECCION 6: Limitaciones
# =============================================================================
st.subheader("Limitaciones")

st.markdown("""
1. **Sin asignacion aleatoria de tipos de senalizacion.** Diferentes lineas electricas tienen
   distintas combinaciones de dispositivos anticolision, y estas lineas atraviesan diferentes
   habitats, altitudes y contextos geograficos. Las diferencias observadas en las tasas de
   mortalidad entre tipos de senalizacion pueden reflejar factores ecologicos o geograficos
   en lugar de la eficacia del dispositivo. Se necesitarian experimentos controlados con
   asignacion aleatoria para realizar afirmaciones causales.

2. **Sin ensayos de retirada de cadaveres.** Sin ensayos de retirada, no podemos estimar la
   proporcion de cadaveres retirados por carroneros antes de la deteccion. La mortalidad real
   esta subestimada.

3. **Alta tasa de carroneo.** Aproximadamente el 95% de los restos encontrados muestran
   evidencias de carroneo o degradacion, lo que significa que la mayoria de los cadaveres se
   descubren como cercos de plumas, esqueletos parciales o restos dispersos en lugar de
   especimenes intactos.

4. **Sin datos de hora del dia.** El conjunto de datos registra la fecha de descubrimiento,
   no la hora de la colision. Los patrones de actividad de las especies (nocturna/crepuscular/
   diurna) de la literatura ornitologica se utilizan como indicador aproximado del momento
   probable de la colision.

5. **Los anos parciales afectan al analisis de tendencias.** El primer ano (2018) y el ultimo
   (2025) tienen datos incompletos. Estos se excluyen o senalan en los calculos de tendencias,
   pero la serie temporal efectiva para el analisis de tendencias es mas corta que el periodo
   nominal de estudio.

6. **La probabilidad de deteccion del observador varia.** Diferentes observadores pueden tener
   diferentes tasas de deteccion dependiendo del terreno, clima, experiencia y esfuerzo de
   busqueda. Esta es una fuente de variabilidad no controlada.

7. **Sin correccion por deteccion imperfecta.** No se ha aplicado ninguna correccion estadistica
   por probabilidad de deteccion. Los recuentos reportados son estimaciones minimas.

8. **Incertidumbre en la identificacion de especies.** Algunos especimenes se identifican solo
   a nivel de genero o se registran como desconocidos. Estos se clasifican con patron de
   actividad "Desconocida" y se excluyen de los analisis especificos por especie.
""")

st.divider()

# =============================================================================
# SECCION 7: Referencias
# =============================================================================
st.subheader("Referencias")

st.markdown("""
**Infraestructura:**
- Red Electrica de Espana (REE). Nota de prensa sobre el proyecto de refuerzo de la linea de
  transmision de 132kV en Fuerteventura que conecta Gran Tarajal, Puerto del Rosario y La Oliva.

**Marco legal:**
- Real Decreto 1432/2008 que establece los requisitos de seguimiento ambiental (Plan de
  Vigilancia Ambiental, PVA) para infraestructuras de lineas electricas.
- Legislacion regional de Canarias sobre especies y habitats protegidos (Catalogo Canario de
  Especies Protegidas).

**Clasificacion de actividad de especies:**
- Rodriguez, A. & Rodriguez, B. (2009). Attraction of petrels to artificial lights in the
  Canary Islands: effects of the moon phase and age class. *Ibis*, 151(2), 299-310.
- Rodda, G.H. & Dean-Bradley, K. (2002). Nocturnal activity in recovering populations of a
  diurnal species. *Animal Conservation*, 5(3), 217-222.
- Cramp, S. & Simmons, K.E.L. (Eds.) (1977-1994). *Handbook of the Birds of Europe, the
  Middle East and North Africa* (Vols. 1-9). Oxford University Press.

**Metodologia de seguimiento de mortalidad:**
- Ferrer, M. et al. (2012). Weak relationship between risk assessment studies and recorded
  mortality in wind farms. *Journal of Applied Ecology*, 49(1), 38-46.
- Martin, G.R. & Shaw, J.M. (2010). Bird collisions with power lines: Failing to see the
  way ahead? *Biological Conservation*, 143(11), 2695-2702.

**Estado de conservacion:**
- Lista Roja de Especies Amenazadas de la UICN. Version 2024-2.
- Catalogo Nacional de Especies Amenazadas (CNEA), Ministerio para la Transicion Ecologica.
""")
