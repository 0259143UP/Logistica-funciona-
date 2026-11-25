import json
import os
import re
import unicodedata
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st
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


@st.cache_resource
def cargar_modelos():
    mod_retraso = joblib.load("models/modelo_retraso.pkl")
    mod_costo = joblib.load("models/modelo_costo.pkl")
    return mod_retraso, mod_costo


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

    encontradas = []
    for c in ciudades:
        c_norm = normalizar_texto(c)
        if c_norm == "cdmx":
            patron = r"\bcdmx\b|\bciudad de mexico\b|\bdf\b"
        else:
            patron = rf"\b{re.escape(c_norm)}\b"
        m = re.search(patron, t)
        if m:
            encontradas.append((m.start(), c))
    encontradas.sort(key=lambda x: x[0])

    orig_ia = encontradas[0][1] if len(encontradas) >= 1 else None
    dest_ia = encontradas[1][1] if len(encontradas) >= 2 and encontradas[1][1] != orig_ia else None

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

    return orig_ia, dest_ia, paq_ia, peso_ia, alerta_ia


if not os.path.exists("models/modelo_retraso.pkl") or not os.path.exists("data/resumen_rutas_spark.parquet"):
    st.error("⚠️ Faltan archivos del modelo o resumen. Ejecuta `python src/procesamiento_spark_ml.py`.")
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
        o_ia, d_ia, p_ia, w_ia, a_ia = interpretar_prompt_ia(prompt_usuario, ciudades, paqueterias)
        st.session_state.ai_pending = {
            "origen": o_ia,
            "destino": d_ia,
            "paqueteria": p_ia,
            "peso": w_ia,
            "alerta": a_ia
        }
        recargar_app()

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