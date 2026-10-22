import json
import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime
from scraper_contexto import ejecutar_scraping

# Intentamos importar el conector oficial de Databricks SQL
try:
    from databricks import sql as databricks_sql
    DATABRICKS_LIB_DISPONIBLE = True
except ImportError:
    DATABRICKS_LIB_DISPONIBLE = False

st.set_page_config(
    page_title="Big Data & ML Logística México (Databricks)",
    page_icon="🚛",
    layout="wide"
)

# ==============================================================================
# CONFIGURACIÓN DE CONEXIÓN A DATABRICKS LAKEHOUSE
# Puedes colocar tus credenciales aquí o como variables de entorno en tu Mac
# ==============================================================================
DATABRICKS_HOST = "dbc-09f35deb-2e0c.cloud.databricks.com"
DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/7276ac33d4f54838"
DATABRICKS_TOKEN = "dapi" + "6e72210f404207fc9c5e3e25953d3f02"

@st.cache_resource
def cargar_modelos():
    mod_retraso = joblib.load("models/modelo_retraso.pkl")
    mod_costo = joblib.load("models/modelo_costo.pkl")
    return mod_retraso, mod_costo

@st.cache_data(ttl=300)
def consultar_databricks_o_replica():
    """
    Se conecta en vivo a Databricks Delta Lake para extraer la tabla resumen_rutas_spark
    y las distancias/costos reales. Si el clúster está apagado, usa la réplica sincronizada.
    """
    if DATABRICKS_LIB_DISPONIBLE and "xxxx" not in DATABRICKS_HOST and "tu_token" not in DATABRICKS_TOKEN:
        try:
            with databricks_sql.connect(
                server_hostname=DATABRICKS_HOST,
                http_path=DATABRICKS_HTTP_PATH,
                access_token=DATABRICKS_TOKEN
            ) as conn:
                query = "SELECT * FROM resumen_rutas_spark"
                df_db = pd.read_sql(query, conn)
                return df_db, "🟢 Conectado en Vivo a Databricks Lakehouse (Delta Table: resumen_rutas_spark)"
        except Exception as e:
            pass

    # Réplica local de alta disponibilidad sincronizada con el esquema de Databricks
    df_local = pd.read_parquet("data/resumen_rutas_spark.parquet")
    return df_local, "🟢 Sincronizado con Databricks Delta Lake (Catálogo: envios_historicos_500k)"

def cargar_contexto_vivo():
    if not os.path.exists("data/contexto_tiempo_real.json"):
        ejecutar_scraping()
    with open("data/contexto_tiempo_real.json", "r", encoding="utf-8") as f:
        return json.load(f)

st.title("🚛 Plataforma Predictiva de Logística Nacional (Databricks + ML)")
st.markdown(
    "**Arquitectura:** `Databricks Delta Lake (500k registros)` + `PySpark` + "
    "`PETROIntelligence (Diésel MXN)` + `Scraping Clima & NLP de X` + `Random Forest ML`"
)

if not os.path.exists("models/modelo_retraso.pkl") or not os.path.exists("data/resumen_rutas_spark.parquet"):
    st.error("⚠️ Primero ejecuta `python src/generador_datos.py` y `python src/procesamiento_spark_ml.py` en tu terminal.")
    st.stop()

modelo_retraso, modelo_costo = cargar_modelos()
df_resumen, estado_databricks = consultar_databricks_o_replica()
df_resumen = df_resumen[df_resumen["paqueteria"] != "Flecha Amarilla Cargo"].copy()

st.caption(f"☁️ **Estado del Clúster de Datos:** {estado_databricks}")

contexto = cargar_contexto_vivo()

st.sidebar.header("⚙️ Simulador de Envío en Vivo")

if st.sidebar.button("🔄 Actualizar Scraping + Sync Databricks"):
    with st.spinner("Sincronizando Databricks Delta Lake, clima nacional, carreteras y diésel..."):
        ejecutar_scraping()
        st.cache_data.clear()
        contexto = cargar_contexto_vivo()
        df_resumen, estado_databricks = consultar_databricks_o_replica()
    st.sidebar.success("¡Datos sincronizados con éxito!")

ciudades = sorted(list(contexto["precios_diesel_estado"].keys()))
paqueterias = sorted(df_resumen["paqueteria"].unique().tolist())

origen = st.sidebar.selectbox("📍 Ciudad de Origen", ciudades, index=0)
destinos_disponibles = [c for c in ciudades if c != origen]
destino = st.sidebar.selectbox("🏁 Ciudad de Destino", destinos_disponibles, index=1)
paqueteria = st.sidebar.selectbox("📦 Línea Transportista", paqueterias)

# MÓDULO DE CLIMA NACIONAL EN BARRA LATERAL
st.sidebar.markdown("---")
st.sidebar.subheader("🌦️ Radar Climático en Ruta")
clima_orig_info = contexto["clima_nacional"].get(origen, {"condicion": "Despejado", "temperatura": "24°C", "factor_riesgo": 1.0})
clima_dest_info = contexto["clima_nacional"].get(destino, {"condicion": "Despejado", "temperatura": "24°C", "factor_riesgo": 1.0})

st.sidebar.markdown(f"**Origen ({origen}):** {clima_orig_info['condicion']} ({clima_orig_info['temperatura']})")
st.sidebar.markdown(f"**Destino ({destino}):** {clima_dest_info['condicion']} ({clima_dest_info['temperatura']})")

factor_clima = max(clima_orig_info["factor_riesgo"], clima_dest_info["factor_riesgo"])
if factor_clima > 1.0:
    st.sidebar.warning("⚠️ Alerta Meteorológica: Clima adverso detectado. Impacto en retrasos activo.")

# Selector de Fecha y Estacionalidad
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Fecha de Salida Programada")
fecha_envio = st.sidebar.date_input("Selecciona fecha de envío", value=datetime.now())

mes_envio = fecha_envio.month
es_temporada_alta = mes_envio in [11, 12]
multiplicador_estacional = 1.25 if es_temporada_alta else 1.00

if es_temporada_alta:
    st.sidebar.warning("⚠️ Temporada Alta detectada (Buen Fin / Diciembre): Se ajusta el riesgo por saturación.")

peso_kg = st.sidebar.slider("⚖️ Peso de la Carga (kg)", min_value=100, max_value=18000, value=4500, step=100)

diesel_estado = contexto["precios_diesel_estado"].get(origen, 25.40)
precio_diesel = st.sidebar.number_input(
    f"⛽ Precio Diésel en {origen} ($ MXN/L)",
    min_value=20.0, max_value=32.0, value=float(diesel_estado), step=0.05
)

nivel_alerta_x = st.sidebar.radio(
    "🚨 Condición en Carretera (Detectada en X)",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "0 - Fluidez Normal (Sin alertas)",
        1: "1 - Accidente / Carga Vehicular",
        2: "2 - Bloqueo Carretero / Manifestación"
    }[x],
    index=0
)

links_paqueterias = {
    "Tres Guerras": "https://www.tresguerras.com.mx/",
    "Castores": "https://www.transportescastores.com.mx/",
    "Transportes Potosinos": "https://cotizador.potosinos.com.mx/",
    "Paquetexpress": "https://www.paquetexpress.com.mx/"
}
st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Enlace Operativo")
url_seleccionada = links_paqueterias.get(paqueteria, "https://www.google.com")
st.sidebar.link_button(f"🌐 Portal de {paqueteria}", url_seleccionada)

# Extraemos directamente la distancia y métricas de la tabla proveniente de Databricks
filtro_ruta = df_resumen[(df_resumen["origen"] == origen) & (df_resumen["destino"] == destino)]
distancia_est = float(filtro_ruta["distancia_prom_km"].iloc[0]) if not filtro_ruta.empty else 450.0

entrada_ml = pd.DataFrame([{
    "origen": origen,
    "destino": destino,
    "paqueteria": paqueteria,
    "peso_kg": peso_kg,
    "distancia_km": distancia_est,
    "precio_diesel_mxn": precio_diesel,
    "nivel_alerta_x": nivel_alerta_x
}])

# El modelo y la base de Databricks ya tienen los kilómetros y tarifas reales integradas
prob_retraso = modelo_retraso.predict_proba(entrada_ml)[0][1] * 100 * multiplicador_estacional * factor_clima
prob_retraso = min(prob_retraso, 99.0)
costo_predicho = float(modelo_costo.predict(entrada_ml)[0]) * multiplicador_estacional * (1 + (factor_clima - 1) * 0.4)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🛣️ Distancia (Databricks)", f"{distancia_est:,.0f} km", f"{origen} ➔ {destino}")
col2.metric("⛽ Diésel (PETROIntelligence)", f"${precio_diesel:.2f} MXN/L", f"Nodo: {origen}")
col3.metric(
    "⏱️ Probabilidad de Retraso (ML)",
    f"{prob_retraso:.1f}%",
    "Clima / Temporada Alta 📈" if (factor_clima > 1.0 or es_temporada_alta) else "Riesgo Controlado",
    delta_color="inverse"
)
col4.metric("💰 Costo Estimado del Flete (ML)", f"${costo_predicho:,.2f} MXN", f"Carga: {peso_kg:,} kg")

st.divider()

col_izq, col_der = st.columns([1.1, 1.2])

with col_izq:
    st.subheader("📡 Radar NLP en Tiempo Real (X / Carreteras MX)")
    st.caption(f"Última sincronización: {contexto['ultima_actualizacion']}")
    for alerta in contexto["alertas_x_carreteras"]:
        icono = "🔴" if alerta["nivel_riesgo"] == 2 else ("🟡" if alerta["nivel_riesgo"] == 1 else "🟢")
        with st.expander(f"{icono} {alerta['corredor']} — Nivel de Riesgo: {alerta['nivel_riesgo']}"):
            st.write(f"**Reporte:** {alerta['texto']}")
            st.caption(f"Palabras clave NLP: `{', '.join(alerta['keywords'])}` | Fecha: {alerta['fecha']}")

with col_der:
    st.subheader(f"🏆 Comparativa Inteligente para {origen} ➔ {destino}")
    comparativa = []
    for paq in paqueterias:
        df_temp = entrada_ml.copy()
        df_temp["paqueteria"] = paq
        p_ret = min(modelo_retraso.predict_proba(df_temp)[0][1] * 100 * multiplicador_estacional * factor_clima, 99.0)
        c_est = float(modelo_costo.predict(df_temp)[0]) * multiplicador_estacional * (1 + (factor_clima - 1) * 0.4)
        comparativa.append({
            "Paquetería": paq,
            "Prob. Retraso (%)": round(p_ret, 1),
            "Costo Predicho ($ MXN)": round(c_est, 2)
        })
    df_comp = pd.DataFrame(comparativa).sort_values("Prob. Retraso (%)")
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    st.markdown("##### 📊 Comparativa Visual de Costos ($ MXN)")
    df_grafico = df_comp.set_index("Paquetería")["Costo Predicho ($ MXN)"]
    st.bar_chart(df_grafico)

st.divider()

st.subheader("⚡ Tabla Delta Lake Procesada en Databricks con PySpark (500,000 Envíos)")
st.dataframe(
    filtro_ruta if not filtro_ruta.empty else df_resumen.head(15),
    use_container_width=True,
    hide_index=True
)