import os
import numpy as np
import pandas as pd

DISTANCIAS_REALES_KM = {
    ("CDMX", "Querétaro"): 215, ("CDMX", "Monterrey"): 905, ("CDMX", "Guadalajara"): 535,
    ("CDMX", "Puebla"): 130, ("CDMX", "Veracruz"): 400, ("CDMX", "León"): 385,
    ("CDMX", "San Luis Potosí"): 415, ("CDMX", "Tijuana"): 2760, ("CDMX", "Mérida"): 1320,
    ("Querétaro", "Monterrey"): 700, ("Querétaro", "Guadalajara"): 365, ("Querétaro", "Puebla"): 335,
    ("Querétaro", "Veracruz"): 605, ("Querétaro", "León"): 175, ("Querétaro", "San Luis Potosí"): 205,
    ("Querétaro", "Tijuana"): 2560, ("Querétaro", "Mérida"): 1525, ("Monterrey", "Guadalajara"): 805,
    ("Monterrey", "Puebla"): 1025, ("Monterrey", "Veracruz"): 1110, ("Monterrey", "León"): 710,
    ("Monterrey", "San Luis Potosí"): 510, ("Monterrey", "Tijuana"): 2420, ("Monterrey", "Mérida"): 2050,
    ("Guadalajara", "Puebla"): 655, ("Guadalajara", "Veracruz"): 925, ("Guadalajara", "León"): 220,
    ("Guadalajara", "San Luis Potosí"): 335, ("Guadalajara", "Tijuana"): 2230, ("Guadalajara", "Mérida"): 1845,
    ("Puebla", "Veracruz"): 275, ("Puebla", "León"): 505, ("Puebla", "San Luis Potosí"): 535,
    ("Puebla", "Tijuana"): 2880, ("Puebla", "Mérida"): 1195, ("Veracruz", "León"): 775,
    ("Veracruz", "San Luis Potosí"): 805, ("Veracruz", "Tijuana"): 3150, ("Veracruz", "Mérida"): 935,
    ("León", "San Luis Potosí"): 180, ("León", "Tijuana"): 2430, ("León", "Mérida"): 1695,
    ("San Luis Potosí", "Tijuana"): 2410, ("San Luis Potosí", "Mérida"): 1725, ("Tijuana", "Mérida"): 4070,
}

def obtener_km(o: str, d: str) -> int:
    if (o, d) in DISTANCIAS_REALES_KM:
        return DISTANCIAS_REALES_KM[(o, d)]
    return DISTANCIAS_REALES_KM.get((d, o), 450)

def generar_dataset_masivo(n_registros: int = 500_000):
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)

    ciudades = [
        "CDMX", "Querétaro", "Monterrey", "Guadalajara", "Puebla",
        "Veracruz", "León", "San Luis Potosí", "Tijuana", "Mérida"
    ]
    paqueterias = ["Castores", "Tres Guerras", "Transportes Potosinos", "Paquetexpress"]

    origenes = np.random.choice(ciudades, n_registros)
    destinos = np.random.choice(ciudades, n_registros)
    mask = origenes == destinos
    while mask.any():
        destinos[mask] = np.random.choice(ciudades, mask.sum())
        mask = origenes == destinos

    df = pd.DataFrame({
        "id_envio": np.arange(1, n_registros + 1),
        "origen": origenes,
        "destino": destinos,
        "paqueteria": np.random.choice(paqueterias, n_registros),
        "peso_kg": np.random.randint(100, 18000, n_registros),
        "precio_diesel_mxn": np.round(np.random.uniform(24.50, 26.80, n_registros), 2),
        "nivel_alerta_x": np.random.choice([0, 1, 2], n_registros, p=[0.70, 0.20, 0.10])
    })

    pares = {(o, d): obtener_km(o, d) for o in ciudades for d in ciudades if o != d}
    df["distancia_km"] = [pares[(o, d)] for o, d in zip(df["origen"], df["destino"])]

    factor_paq = df["paqueteria"].map({
        "Paquetexpress": 1.06,
        "Castores": 1.02,
        "Tres Guerras": 0.98,
        "Transportes Potosinos": 0.93
    })

    banderazo = 4800.0
    costo_km = df["distancia_km"] * 22.5
    costo_ton = (df["peso_kg"] / 1000.0) * df["distancia_km"] * (df["precio_diesel_mxn"] / 25.0) * 1.85
    ruido = np.random.normal(0, 350, n_registros)

    df["costo_mxn"] = np.round((banderazo + costo_km + costo_ton) * factor_paq + ruido, 2)
    df["costo_flete_mxn"] = df["costo_mxn"]

    prob_retraso = np.clip(
        0.08 + (df["distancia_km"] / 12000.0) + (df["nivel_alerta_x"] * 0.22)
        + np.where(df["paqueteria"] == "Transportes Potosinos", 0.05, 0.0),
        0.05, 0.95
    )
    df["retrasado"] = np.random.binomial(1, prob_retraso)
    df["retraso"] = df["retrasado"]
    df["horas_retraso"] = np.where(df["retrasado"] == 1, np.round(np.random.exponential(4.5, n_registros) + 1.0, 1), 0.0)

    df.to_parquet("data/envios_mexico.parquet", index=False)
    df.to_parquet("data/envios_historicos_500k.parquet", index=False)
    print("✅ Dataset de 500,000 envíos con KM y precios reales sincronizado para Databricks y PySpark.")

if __name__ == "__main__":
    generar_dataset_masivo()