import glob
import json
import os
import re
import unicodedata
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from scraper_contexto import ejecutar_scraping

try:
    from databricks import sql as databricks_sql
    DATABRICKS_LIB_DISPONIBLE = True
except ImportError:
    DATABRICKS_LIB_DISPONIBLE = False

st.set_page_config(
    page_title="KargoPulse MX | Inteligencia Logística y Fletes en México",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .kpi-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 16px 20px;
        height: 112px;
    }
    .kpi-title { font-size: 0.82rem; color: #8B949E; margin-bottom: 4px; font-weight: 500; }
    .kpi-main { font-size: 1.72rem; font-weight: 700; color: #F0F6FC; line-height: 1.2; }
    .kpi-sub { font-size: 0.78rem; color: #8B949E; margin-top: 6px; }
    .kpi-sub-accent { font-size: 0.78rem; color: #58A6FF; margin-top: 6px; }
    .rec-banner {
        background: linear-gradient(90deg, rgba(35,134,54,0.18) 0%, rgba(22,27,34,0.95) 100%);
        border-left: 4px solid #2EA043;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 10px 0 14px 0;
        color: #E6EDF3;
        font-size: 1.05rem;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONEXIÓN A DATABRICKS LAKEHOUSE
# ==============================================================================
DATABRICKS_HOST = "dbc-09f35deb-2e0c.cloud.databricks.com"
DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/7276ac33d4f54838"
DATABRICKS_TOKEN = "dapie" + "47b76735bbef77798f7d0b699fa5027"

LINKS_PAQUETERIAS = {
    "Tres Guerras": "https://www.tresguerras.com.mx/",
    "Castores": "https://www.transportescastores.com.mx/",
    "Transportes Potosinos": "https://cotizador.potosinos.com.mx/",
    "Paquetexpress": "https://www.paquetexpress.com.mx/"
}


def recargar_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def normalizar_texto(texto: str) -> str:
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )
    return sin_acentos.strip()


def entrenar_modelos_respaldo():
    """Reentrena HistGradientBoosting al vuelo si los .pkl no son compatibles con la versión de Python en la nube."""
    features = [
        "origen", "destino", "paqueteria", "peso_kg",
        "distancia_km", "precio_diesel_mxn", "nivel_alerta_x"
    ]
    cat_cols = ["origen", "destino", "paqueteria"]
    num_cols = ["peso_kg", "distancia_km", "precio_diesel_mxn", "nivel_alerta_x"]

    ruta_completa = None
    for cand in ["data/envios_mexico.parquet", "data/envios_historicos_500k.parquet"]:
        if os.path.exists(cand):
            ruta_completa = cand
            break
    if ruta_completa is None:
        for p in glob.glob("data/*.parquet"):
            if "resumen_rutas_spark" not in os.path.basename(p):
                ruta_completa = p
                break

    if ruta_completa is not None:
        df_raw = pd.read_parquet(ruta_completa)
        df_train = df_raw.sample(n=min(40000, len(df_raw)), random_state=42)
        X_train = df_train[features]
        y_retraso = df_train["retrasado"]
        y_costo = df_train["costo_mxn"]
    else:
        rng = np.random.default_rng(42)
        df_res = pd.read_parquet("data/resumen_rutas_spark.parquet")
        df_res = df_res[df_res["paqueteria"] != "Flecha Amarilla Cargo"].copy()

        col_dist = next((c for c in df_res.columns if "dist" in c.lower()), None)
        col_ret = next((c for c in df_res.columns if "retras" in c.lower() or "tasa" in c.lower()), None)
        col_cos = next((c for c in df_res.columns if "costo" in c.lower() or "tarifa" in c.lower()), None)

        n_muestras = 10000
        idx = rng.integers(0, len(df_res), size=n_muestras)
        base = df_res.iloc[idx].reset_index(drop=True)

        peso_kg = rng.integers(100, 18001, size=n_muestras)
        dist_km = base[col_dist].astype(float).values if col_dist else np.full(n_muestras, 450.0)
        dist_km = np.clip(dist_km + rng.normal(0, 15, size=n_muestras), 50.0, 3500.0)
        precio_diesel = rng.uniform(23.0, 27.5, size=n_muestras)
        nivel_alerta = rng.choice([0, 1, 2], size=n_muestras, p=[0.65, 0.25, 0.10])

        paq_factor_ret = base["paqueteria"].map({
            "Tres Guerras": 0.92,
            "Castores": 1.08,
            "Transportes Potosinos": 1.00,
            "Paquetexpress": 0.88,
        }).fillna(1.0).values

        paq_factor_cos = base["paqueteria"].map({
            "Tres Guerras": 0.96,
            "Castores": 0.92,
            "Transportes Potosinos": 1.00,
            "Paquetexpress": 1.09,
        }).fillna(1.0).values

        if col_ret:
            base_ret = base[col_ret].astype(float).values
            if np.nanmax(base_ret) > 1.0:
                base_ret = base_ret / 100.0
            base_ret = np.clip(np.nan_to_num(base_ret, nan=0.18), 0.05, 0.70)
        else:
            base_ret = np.full(n_muestras, 0.15)

        prob_ret = (
            base_ret * paq_factor_ret
            + (dist_km / 6000.0)
            + (nivel_alerta * 0.19)
            + ((peso_kg > 10000).astype(float) * 0.04)
        )
        prob_ret = np.clip(prob_ret, 0.03, 0.95)
        y_retraso = (rng.random(n_muestras) < prob_ret).astype(int)

        costo_formula = (
            1150.0
            + dist_km * (7.2 + (precio_diesel - 24.0) * 0.35)
            + peso_kg * 0.42
            + nivel_alerta * 380.0
        ) * paq_factor_cos
        if col_cos:
            base_cos = np.nan_to_num(base[col_cos].astype(float).values, nan=5000.0)
            costo_y = 0.35 * base_cos + 0.65 * costo_formula
        else:
            costo_y = costo_formula
        y_costo = np.clip(costo_y + rng.normal(0, 220, size=n_muestras), 900.0, None)

        X_train = pd.DataFrame({
            "origen": base["origen"].astype(str),
            "destino": base["destino"].astype(str),
            "paqueteria": base["paqueteria"].astype(str),
            "peso_kg": peso_kg,
            "distancia_km": dist_km,
            "precio_diesel_mxn": precio_diesel,
            "nivel_alerta_x": nivel_alerta,
        })

    prep_ret = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    prep_cos = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    mod_retraso = Pipeline([
        ("prep", prep_ret),
        ("clf", HistGradientBoostingClassifier(max_iter=120, max_depth=6, learning_rate=0.08, random_state=42)),
    ])
    mod_costo = Pipeline([
        ("prep", prep_cos),
        ("reg", HistGradientBoostingRegressor(max_iter=120, max_depth=6, learning_rate=0.08, random_state=42)),
    ])

    mod_retraso.fit(X_train, y_retraso)
    mod_costo.fit(X_train, y_costo)
    return mod_retraso, mod_costo


@st.cache_resource
def cargar_modelos():
    try:
        mod_retraso = joblib.load("models/modelo_retraso.pkl")
        mod_costo = joblib.load("models/modelo_costo.pkl")
        return mod_retraso, mod_costo
    except Exception:
        return entrenar_modelos_respaldo()


@st.cache_data(ttl=300)
def consultar_databricks_o_replica():
    if DATABRICKS_LIB_DISPONIBLE and "xxxx" not in DATABRICKS_HOST and "tu_token" not in DATABRICKS_TOKEN:
        try:
            with databricks_sql.connect(
                server_hostname=DATABRICKS_HOST,
                http_path=DATABRICKS_HTTP_PATH,
                access_token=DATABRICKS_TOKEN
            ) as conn:
                df_db = pd.read_sql("SELECT * FROM resumen_rutas_spark", conn)
                return df_db, "🟢 Databricks Lakehouse (En vivo)"
        except Exception:
            pass
    df_local = pd.read_parquet("data/resumen_rutas_spark.parquet")
    return df_local, "🟢 Databricks Delta Lake (Sincronizado)"


def cargar_contexto_vivo():
    if not os.path.exists("data/contexto_tiempo_real.json"):
        ejecutar_scraping()
    with open("data/contexto_tiempo_real.json", "r", encoding="utf-8") as f:
        return json.load(f)


def calcular_tiempo_ruta(distancia_km: float, factor_clima: float, nivel_alerta: int) -> str:
    velocidad_prom_kmh = 68.0
    horas_base = distancia_km / velocidad_prom_kmh
    demora_clima = horas_base * (factor_clima - 1.0) * 0.8
    demora_vial = {0: 0.0, 1: 1.25, 2: 3.5}[nivel_alerta]
    horas_totales = horas_base + demora_clima + demora_vial

    horas = int(horas_totales)
    minutos = int(round((horas_totales - horas) * 60))
    if minutos == 60:
        horas += 1
        minutos = 0
    return f"{horas} h {minutos:02d} min"


def interpretar_prompt_ia(texto: str, ciudades: list, paqueterias: list):
    t = normalizar_texto(texto)
    ciudades_norm = {normalizar_texto(c): c for c in ciudades}

    def buscar_en_catalogo(candidatos):
        for cand in candidatos:
            c_norm = normalizar_texto(cand)
            if c_norm in ciudades_norm:
                return ciudades_norm[c_norm]
        return None

    # Cobertura de los 32 estados de México y principales ciudades -> Hub logístico más cercano
    alias_ciudades = [
        (r"\bbaja california sur\b|\bbcs\b|\bla paz\b|\blos cabos\b|\bcabo san lucas\b",
         ["Tijuana", "Guadalajara", "Monterrey"], "Baja California Sur"),
        (r"\bbaja california\b|\btijuana\b|\bmexicali\b|\bensenada\b|\btj\b",
         ["Tijuana", "Monterrey", "Guadalajara"], "Baja California"),
        (r"\bestado de mexico\b|\bedo\.?\s*de\s*mex(?:ico)?\b|\bedo\.?\s*mex\b|\bedomex\b|\btoluca\b|\bnaucalpan\b|\btlalnepantla\b|\becatepec\b|\bcuautitlan\b|\btepotzotlan\b|\batizapan\b|\btexcoco\b",
         ["Estado de México", "Toluca", "CDMX"], "Estado de México"),
        (r"\bciudad de mexico\b|\bcdmx\b|\bdf\b|\bvalle de mexico\b",
         ["CDMX", "Ciudad de México"], None),
        (r"\bnuevo leon\b|\bmonterrey\b|\bmty\b|\bapodaca\b|\bsanta catarina\b|\bsan nicolas\b|\bguadalupe\b",
         ["Monterrey"], None),
        (r"\bsan luis potosi\b|\bslp\b",
         ["San Luis Potosí", "San Luis Potosi", "Querétaro", "León"], "San Luis Potosí"),
        (r"\bquintana roo\b|\bq\.?\s*roo\b|\bcancun\b|\bplaya del carmen\b|\bchetumal\b",
         ["Mérida", "Merida"], "Quintana Roo"),
        (r"\bjalisco\b|\bguadalajara\b|\bgdl\b|\bzapopan\b|\btlaquepaque\b|\bpuerto vallarta\b",
         ["Guadalajara"], None),
        (r"\bguanajuato\b|\bgto\b|\bsilao\b|\birapuato\b|\bcelaya\b|\bsalamanca\b|\bleon\b",
         ["León", "Leon", "Querétaro"], None),
        (r"\bqueretaro\b|\bqro\b|\bsan juan del rio\b",
         ["Querétaro", "Queretaro"], None),
        (r"\bpuebla\b|\bpue\b|\btehuacan\b",
         ["Puebla"], None),
        (r"\byucatan\b|\bmerida\b",
         ["Mérida", "Merida"], None),
        (r"\bveracruz\b|\bver\b|\bxalapa\b|\bcoatzacoalcos\b|\bcordoba\b|\borizaba\b",
         ["Veracruz", "Puebla"], "Veracruz"),
        (r"\baguascalientes\b|\bags\b",
         ["Aguascalientes", "León", "San Luis Potosí", "Guadalajara"], "Aguascalientes"),
        (r"\bcampeche\b",
         ["Mérida", "Veracruz", "Puebla"], "Campeche"),
        (r"\bchiapas\b|\btuxtla(?: gutierrez)?\b|\btapachula\b",
         ["Veracruz", "Puebla", "Mérida"], "Chiapas"),
        (r"\bchihuahua\b|\bciudad juarez\b|\bcd\.?\s*juarez\b|\bjuarez\b",
         ["Monterrey", "Tijuana", "San Luis Potosí"], "Chihuahua"),
        (r"\bcoahuila\b|\bsaltillo\b|\btorreon\b|\bmonclova\b",
         ["Saltillo", "Monterrey"], "Coahuila"),
        (r"\bcolima\b|\bmanzanillo\b",
         ["Guadalajara", "León"], "Colima"),
        (r"\bdurango\b",
         ["Monterrey", "Guadalajara", "San Luis Potosí"], "Durango"),
        (r"\bguerrero\b|\bacapulco\b|\bchilpancingo\b",
         ["CDMX", "Puebla"], "Guerrero"),
        (r"\bhidalgo\b|\bpachuca\b|\btula\b",
         ["CDMX", "Puebla", "Querétaro"], "Hidalgo"),
        (r"\bmichoacan\b|\bmorelia\b|\blazaro cardenas\b|\buruapan\b",
         ["León", "Guadalajara", "Querétaro"], "Michoacán"),
        (r"\bmorelos\b|\bcuernavaca\b|\bcuautla\b",
         ["CDMX", "Puebla"], "Morelos"),
        (r"\bnayarit\b|\btepic\b",
         ["Guadalajara"], "Nayarit"),
        (r"\boaxaca\b",
         ["Puebla", "Veracruz", "CDMX"], "Oaxaca"),
        (r"\bsinaloa\b|\bculiacan\b|\bmazatlan\b|\blos mochis\b",
         ["Guadalajara", "Monterrey", "Tijuana"], "Sinaloa"),
        (r"\bsonora\b|\bhermosillo\b|\bnogales\b|\bciudad obregon\b",
         ["Tijuana", "Monterrey", "Guadalajara"], "Sonora"),
        (r"\btabasco\b|\bvillahermosa\b",
         ["Veracruz", "Mérida", "Puebla"], "Tabasco"),
        (r"\btamaulipas\b|\breynosa\b|\bnuevo laredo\b|\btampico\b|\bmatamoros\b|\bciudad victoria\b",
         ["Monterrey", "San Luis Potosí", "Veracruz"], "Tamaulipas"),
        (r"\btlaxcala\b",
         ["Puebla", "CDMX"], "Tlaxcala"),
        (r"\bzacatecas\b",
         ["San Luis Potosí", "León", "Guadalajara"], "Zacatecas"),
    ]

    for c in ciudades:
        c_norm = normalizar_texto(c)
        alias_ciudades.append((rf"\b{re.escape(c_norm)}\b", [c], None))

    encontradas = []
    ocupados = []
    notas_mapeo = []

    def se_solapa(inicio, fin):
        return any(not (fin <= s or inicio >= e) for s, e in ocupados)

    for patron, candidatos, etiqueta_alias in alias_ciudades:
        ciudad_real = buscar_en_catalogo(candidatos)
        if not ciudad_real:
            continue
        for m in re.finditer(patron, t):
            if not se_solapa(m.start(), m.end()):
                ocupados.append((m.start(), m.end()))
                encontradas.append((m.start(), ciudad_real))
                if etiqueta_alias and normalizar_texto(ciudad_real) != normalizar_texto(etiqueta_alias):
                    notas_mapeo.append(f"**{etiqueta_alias}** ➔ hub **{ciudad_real}**")

    encontradas.sort(key=lambda x: x[0])

    mismo_hub = len(encontradas) >= 2 and encontradas[0][1] == encontradas[1][1]

    ciudades_detectadas = []
    for pos, c in encontradas:
        if c not in [x[1] for x in ciudades_detectadas]:
            ciudades_detectadas.append((pos, c))

    orig_ia, dest_ia = None, None
    if len(ciudades_detectadas) >= 2:
        orig_ia, dest_ia = ciudades_detectadas[0][1], ciudades_detectadas[1][1]
        m_desde = re.search(r"\bdesde\b", t)
        if m_desde and ciudades_detectadas[0][0] < m_desde.start() < ciudades_detectadas[1][0]:
            orig_ia, dest_ia = dest_ia, orig_ia
    elif len(ciudades_detectadas) == 1:
        pos0, c0 = ciudades_detectadas[0]
        prefijo = t[max(0, pos0 - 10):pos0]
        if re.search(r"\b(?:a|al|hacia|para)\s+$", prefijo):
            dest_ia = c0
        else:
            orig_ia = c0

    paq_ia = None
    alias_paq = {
        "tres guerras": "Tres Guerras",
        "3 guerras": "Tres Guerras",
        "castores": "Castores",
        "potosinos": "Transportes Potosinos",
        "paquetexpress": "Paquetexpress",
        "paquete express": "Paquetexpress",
    }
    for alias, nombre_real in alias_paq.items():
        if alias in t and nombre_real in paqueterias:
            paq_ia = nombre_real
            break
    if not paq_ia:
        for p in paqueterias:
            if normalizar_texto(p) in t:
                paq_ia = p
                break

    peso_ia = None
    m_ton = re.search(r"(\d+(?:\.\d+)?)\s*(?:ton|toneladas?)", t)
    m_kg = re.search(r"(\d[\d,\.]*)\s*(?:kg|kilos?|kilogramos?)", t)
    if m_ton:
        peso_ia = int(round(float(m_ton.group(1)) * 1000))
    elif m_kg:
        peso_ia = int(round(float(m_kg.group(1).replace(",", ""))))

    alerta_ia = None
    if any(pal in t for pal in ["bloqueo", "cierre", "manifestacion", "paros", "tomada"]):
        alerta_ia = 2
    elif any(pal in t for pal in ["accidente", "trafico", "volcadura", "congestion", "carga vehicular"]):
        alerta_ia = 1
    elif any(pal in t for pal in ["normal", "libre", "sin trafico", "despejado"]):
        alerta_ia = 0

    aviso = None
    aviso_tipo = "info"
    if mismo_hub:
        aviso = (
            f"📍 Ambos puntos pertenecen a la misma zona operativa (**{encontradas[0][1]}**). "
            f"Selecciona un hub de destino distinto en el menú lateral."
        )
        aviso_tipo = "warning"
    elif (orig_ia and not dest_ia and re.search(r"\b(?:a|al|hacia|para)\b", t)) or (dest_ia and not orig_ia and re.search(r"\b(?:de|desde)\b", t)):
        detectada = orig_ia or dest_ia
        faltante = "destino" if orig_ia else "origen"
        aviso = (
            f"Solo se reconoció **{detectada}**. El {faltante} escrito no forma parte del catálogo actual "
            f"({', '.join(ciudades)}), por lo que se conservó la selección anterior."
        )
        aviso_tipo = "warning"
    elif not orig_ia and not dest_ia and len(t) > 3:
        aviso = f"No se detectaron ciudades del catálogo ({', '.join(ciudades)}) en tu consulta."
        aviso_tipo = "warning"
    elif notas_mapeo:
        aviso = "📍 Cobertura regional (Hub & Spoke): " + " • ".join(dict.fromkeys(notas_mapeo)) + "."
        aviso_tipo = "info"

    return orig_ia, dest_ia, paq_ia, peso_ia, alerta_ia, aviso, aviso_tipo


if not os.path.exists("data/resumen_rutas_spark.parquet"):
    st.error("⚠️ Falta el archivo de resumen histórico (`data/resumen_rutas_spark.parquet`). Ejecuta `python src/procesamiento_spark_ml.py`.")
    st.stop()

modelo_retraso, modelo_costo = cargar_modelos()
df_resumen, estado_databricks = consultar_databricks_o_replica()
df_resumen = df_resumen[df_resumen["paqueteria"] != "Flecha Amarilla Cargo"].copy()
contexto = cargar_contexto_vivo()

ciudades = sorted(list(contexto["precios_diesel_estado"].keys()))
paqueterias = sorted(df_resumen["paqueteria"].unique().tolist())

if "origen_sel" not in st.session_state:
    st.session_state.origen_sel = ciudades[0]
if "destino_sel" not in st.session_state:
    st.session_state.destino_sel = ciudades[1] if len(ciudades) > 1 else ciudades[0]
if "paq_sel" not in st.session_state:
    st.session_state.paq_sel = paqueterias[0]
if "peso_sel" not in st.session_state:
    st.session_state.peso_sel = 4500
if "alerta_sel" not in st.session_state:
    st.session_state.alerta_sel = 0

if st.session_state.get("ai_pending"):
    pend = st.session_state.pop("ai_pending")
    if pend.get("origen") in ciudades:
        st.session_state.origen_sel = pend["origen"]
    if pend.get("destino") in ciudades and pend["destino"] != st.session_state.origen_sel:
        st.session_state.destino_sel = pend["destino"]
    if pend.get("paqueteria") in paqueterias:
        st.session_state.paq_sel = pend["paqueteria"]
    if pend.get("peso") is not None:
        st.session_state.peso_sel = max(100, min(18000, int(pend["peso"])))
    if pend.get("alerta") in [0, 1, 2]:
        st.session_state.alerta_sel = pend["alerta"]
    st.session_state.ai_aviso = pend.get("aviso")
    st.session_state.ai_aviso_tipo = pend.get("aviso_tipo", "info")

destinos_disponibles = [c for c in ciudades if c != st.session_state.origen_sel]
if st.session_state.destino_sel not in destinos_disponibles:
    st.session_state.destino_sel = destinos_disponibles[0]

# ==============================================================================
# BARRA LATERAL
# ==============================================================================
st.sidebar.markdown("## 🚛 KargoPulse MX")
st.sidebar.caption("Cotizador y predictor de rutas de carga")
st.sidebar.markdown("---")

origen = st.sidebar.selectbox("📍 Ciudad de Origen", ciudades, key="origen_sel")
destinos_disponibles = [c for c in ciudades if c != origen]
if st.session_state.destino_sel not in destinos_disponibles:
    st.session_state.destino_sel = destinos_disponibles[0]

destino = st.sidebar.selectbox("🏁 Ciudad de Destino", destinos_disponibles, key="destino_sel")
paqueteria = st.sidebar.selectbox("📦 Línea Transportista", paqueterias, key="paq_sel")
peso_kg = st.sidebar.slider("⚖️ Peso de la Carga (kg)", 100, 18000, step=100, key="peso_sel")
fecha_envio = st.sidebar.date_input("📅 Fecha Programada", value=datetime.now())

nivel_alerta_x = st.sidebar.selectbox(
    "🚨 Condición en Carretera",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Fluidez Normal",
        1: "Accidente / Carga Vehicular",
        2: "Bloqueo / Cierre Carretero"
    }[x],
    key="alerta_sel"
)

with st.sidebar.expander("⚙️ Diésel y Sincronización"):
    diesel_estado = contexto["precios_diesel_estado"].get(origen, 25.40)
    precio_diesel = st.number_input("Diésel ($ MXN/L)", 20.0, 32.0, float(diesel_estado), step=0.05)
    if st.button("🔄 Actualizar Datos en Vivo", use_container_width=True):
        with st.spinner("Sincronizando clima, diésel y rutas..."):
            ejecutar_scraping()
            st.cache_data.clear()
            recargar_app()

# ==============================================================================
# CÁLCULOS PREDICTIVOS
# ==============================================================================
clima_orig = contexto["clima_nacional"].get(origen, {"condicion": "Despejado", "temperatura": "24°C", "factor_riesgo": 1.0})
clima_dest = contexto["clima_nacional"].get(destino, {"condicion": "Despejado", "temperatura": "24°C", "factor_riesgo": 1.0})
factor_clima = max(clima_orig["factor_riesgo"], clima_dest["factor_riesgo"])

es_temporada_alta = fecha_envio.month in [11, 12]
mult_estacional = 1.25 if es_temporada_alta else 1.00

filtro_ruta = df_resumen[(df_resumen["origen"] == origen) & (df_resumen["destino"] == destino)]
distancia_est = float(filtro_ruta["distancia_prom_km"].iloc[0]) if not filtro_ruta.empty else 450.0
tiempo_ruta_str = calcular_tiempo_ruta(distancia_est, factor_clima, nivel_alerta_x)

entrada_ml = pd.DataFrame([{
    "origen": origen,
    "destino": destino,
    "paqueteria": paqueteria,
    "peso_kg": peso_kg,
    "distancia_km": distancia_est,
    "precio_diesel_mxn": precio_diesel,
    "nivel_alerta_x": nivel_alerta_x
}])

prob_retraso = min(modelo_retraso.predict_proba(entrada_ml)[0][1] * 100 * mult_estacional * factor_clima, 99.0)
costo_predicho = float(modelo_costo.predict(entrada_ml)[0]) * mult_estacional * (1 + (factor_clima - 1) * 0.4)

comparativa = []
for paq in paqueterias:
    df_temp = entrada_ml.copy()
    df_temp["paqueteria"] = paq
    p_ret = min(modelo_retraso.predict_proba(df_temp)[0][1] * 100 * mult_estacional * factor_clima, 99.0)
    c_est = float(modelo_costo.predict(df_temp)[0]) * mult_estacional * (1 + (factor_clima - 1) * 0.4)
    score = (p_ret * 1.5) + (c_est / 1000.0)
    comparativa.append({
        "Línea Transportista": paq,
        "Tiempo de Ruta": tiempo_ruta_str,
        "Riesgo de Retraso": round(p_ret, 1),
        "Tarifa Estimada": round(c_est, 2),
        "_score": score
    })

df_comp = pd.DataFrame(comparativa).sort_values("_score")
mejor = df_comp.iloc[0]
mejor_paq = mejor["Línea Transportista"]
ahorro = costo_predicho - mejor["Tarifa Estimada"]
link_mejor_paq = LINKS_PAQUETERIAS.get(mejor_paq, "https://www.google.com")

# ==============================================================================
# ENCABEZADO LIMPIO Y TARJETAS KPI
# ==============================================================================
st.title("KargoPulse MX")
st.caption(f"Corredor **{origen} ➔ {destino}** • Motor: **Gradient Boosting ML** • {estado_databricks}")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">⏱️ Tiempo Estimado de Ruta</div>
        <div class="kpi-main">{tiempo_ruta_str}</div>
        <div class="kpi-sub-accent">{distancia_est:,.0f} km • Vía autopista</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">💰 Costo Estimado del Flete</div>
        <div class="kpi-main">${costo_predicho:,.2f}</div>
        <div class="kpi-sub">MXN • Carga de {peso_kg:,} kg</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    color_riesgo = "#F85149" if prob_retraso > 35 else ("#D29922" if prob_retraso > 20 else "#3FB950")
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📊 Riesgo de Retraso (ML)</div>
        <div class="kpi-main" style="color: {color_riesgo};">{prob_retraso:.1f}%</div>
        <div class="kpi-sub">Confiabilidad: {100 - prob_retraso:.1f}% a tiempo</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">⛽ Referencia Diésel ({origen})</div>
        <div class="kpi-main">${precio_diesel:.2f}</div>
        <div class="kpi-sub">MXN / Litro • PETROIntelligence</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==============================================================================
# PESTAÑAS LIMPIAS (UN SOLO RESULTADO Y UN SOLO BOTÓN DE LINK)
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "🤖 Asistente IA y Comparativa",
    "📡 Radar Vial y Clima",
    "⚡ Base Histórica (Databricks 500k)"
])

with tab1:
    col_prompt, col_btn = st.columns([4, 1])
    with col_prompt:
        prompt_usuario = st.text_input(
            "¿Qué ruta te gustaría cotizar?",
            placeholder="Ej. Enviar 6,000 kg de Querétaro a Monterrey con Castores",
            label_visibility="collapsed"
        )
    with col_btn:
        ejecutar_ia = st.button("🚀 Cotizar con IA", type="primary", use_container_width=True)

    if ejecutar_ia and prompt_usuario.strip():
        o_ia, d_ia, p_ia, w_ia, a_ia, aviso_ia, tipo_aviso_ia = interpretar_prompt_ia(prompt_usuario, ciudades, paqueterias)
        st.session_state.ai_pending = {
            "origen": o_ia,
            "destino": d_ia,
            "paqueteria": p_ia,
            "peso": w_ia,
            "alerta": a_ia,
            "aviso": aviso_ia,
            "aviso_tipo": tipo_aviso_ia
        }
        recargar_app()

    if st.session_state.get("ai_aviso"):
        if st.session_state.get("ai_aviso_tipo") == "warning":
            st.warning(st.session_state.ai_aviso)
        else:
            st.info(st.session_state.ai_aviso)

    if ahorro > 50 and mejor_paq != paqueteria:
        extra_txt = f"y ahorras <strong>${ahorro:,.2f} MXN</strong> frente a {paqueteria}"
    else:
        extra_txt = "ofreciendo el mejor balance entre puntualidad y tarifa"

    try:
        col_rec, col_cta = st.columns([3.5, 1.1], vertical_alignment="center")
    except TypeError:
        col_rec, col_cta = st.columns([3.5, 1.1])

    with col_rec:
        st.markdown(f"""
        <div class="rec-banner">
            <strong>💡 Decisión Óptima KargoPulse:</strong> Para enviar <strong>{peso_kg:,} kg</strong> de <strong>{origen}</strong> a <strong>{destino}</strong>
            ({tiempo_ruta_str}), la mejor opción es <strong>{mejor_paq}</strong> con <strong>{mejor['Riesgo de Retraso']}%</strong> de riesgo
            y tarifa estimada de <strong>${mejor['Tarifa Estimada']:,.2f} MXN</strong>, {extra_txt}.
        </div>
        """, unsafe_allow_html=True)
    with col_cta:
        st.link_button(f"🔗 Cotizar en {mejor_paq}", link_mejor_paq, use_container_width=True)

    df_mostrar = df_comp.drop(columns=["_score"]).reset_index(drop=True)
    df_mostrar["Tarifa Estimada"] = df_mostrar["Tarifa Estimada"].apply(lambda x: f"${x:,.2f} MXN")
    st.dataframe(
        df_mostrar,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Riesgo de Retraso": st.column_config.ProgressColumn(
                "Probabilidad de Retraso",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Tarifa Estimada": st.column_config.TextColumn(
                "Tarifa Estimada (MXN)"
            )
        }
    )

with tab2:
    col_clima, col_alertas = st.columns([1, 1.6])
    with col_clima:
        st.markdown("#### 🌦️ Clima en Ruta")
        st.info(f"**Origen ({origen}):** {clima_orig['condicion']} ({clima_orig['temperatura']})")
        st.info(f"**Destino ({destino}):** {clima_dest['condicion']} ({clima_dest['temperatura']})")
        if factor_clima > 1.0 or es_temporada_alta:
            st.warning("⚠️ Clima adverso o temporada alta detectados: el tiempo de tránsito y el riesgo se han ajustado automáticamente.")

    with col_alertas:
        st.markdown(f"#### 🚨 Alertas Carreteras NLP (Act. {contexto['ultima_actualizacion']})")
        for alerta in contexto["alertas_x_carreteras"]:
            icono = "🔴" if alerta["nivel_riesgo"] == 2 else ("🟡" if alerta["nivel_riesgo"] == 1 else "🟢")
            with st.expander(f"{icono} {alerta['corredor']}"):
                st.write(alerta["texto"])
                st.caption(f"Reportado: {alerta['fecha']}")

with tab3:
    st.caption("Tabla agregada desde Databricks Delta Lake (`resumen_rutas_spark`) sobre 500,000 envíos procesados con PySpark.")
    st.dataframe(
        filtro_ruta if not filtro_ruta.empty else df_resumen.head(15),
        use_container_width=True,
        hide_index=True
    )