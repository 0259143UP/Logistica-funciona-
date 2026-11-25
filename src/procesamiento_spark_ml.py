import glob
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

RUTA_RESUMEN = "data/resumen_rutas_spark.parquet"


def obtener_ruta_dataset():
    candidatos = [
        "data/envios_mexico.parquet",
        "data/envios_historicos_500k.parquet",
    ]
    for ruta in candidatos:
        if os.path.exists(ruta):
            return ruta
    for ruta in glob.glob("data/*.parquet"):
        if os.path.basename(ruta) != os.path.basename(RUTA_RESUMEN):
            return ruta
    raise FileNotFoundError(
        "No se encontró el archivo .parquet de 500,000 envíos en 'data/'. Ejecuta primero: python src/generador_datos.py"
    )


def procesar_con_pyspark(ruta_parquet):
    """Procesa los 500,000 registros usando PySpark SQL y DataFrames distribuidos."""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F

        print("⚡ Iniciando clúster local de PySpark...")
        spark = (
            SparkSession.builder
            .appName("KargoPulseMX_BigData")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        df_spark = spark.read.parquet(ruta_parquet)
        total_filas = df_spark.count()
        print(f"✅ PySpark leyó exitosamente {total_filas:,} registros distribuidos.")

        resumen_spark = (
            df_spark.groupBy("origen", "destino", "paqueteria")
            .agg(
                F.count("id_envio").alias("total_envios"),
                F.round(F.avg("costo_mxn"), 2).alias("costo_promedio_mxn"),
                F.round(F.avg("retrasado") * 100, 1).alias("porcentaje_retraso"),
                F.round(F.avg("horas_retraso"), 1).alias("horas_retraso_prom"),
                F.round(F.avg("distancia_km"), 1).alias("distancia_prom_km")
            )
            .orderBy(F.desc("total_envios"))
        )

        print("\n📊 Top 5 Rutas Procesadas por PySpark:")
        resumen_spark.show(5, truncate=False)

        resumen_pd = resumen_spark.toPandas()
        resumen_pd.to_parquet(RUTA_RESUMEN, index=False)

        df_ml = df_spark.sample(withReplacement=False, fraction=0.20, seed=42).toPandas()
        spark.stop()
        return df_ml

    except Exception as e:
        print(f"⚠️ Aviso de entorno Java/PySpark ({e}). Ejecutando motor columnar PyArrow sobre los 500,000 registros...")
        df_full = pd.read_parquet(ruta_parquet)
        resumen_pd = (
            df_full.groupby(["origen", "destino", "paqueteria"], as_index=False)
            .agg(
                total_envios=("id_envio", "count"),
                costo_promedio_mxn=("costo_mxn", lambda x: round(x.mean(), 2)),
                porcentaje_retraso=("retrasado", lambda x: round(x.mean() * 100, 1)),
                horas_retraso_prom=("horas_retraso", lambda x: round(x.mean(), 1)),
                distancia_prom_km=("distancia_km", lambda x: round(x.mean(), 1))
            )
            .sort_values("total_envios", ascending=False)
        )
        print("\n📊 Top 5 Rutas Procesadas (500,000 registros):")
        print(resumen_pd.head(5).to_string(index=False))
        resumen_pd.to_parquet(RUTA_RESUMEN, index=False)
        return df_full.sample(n=min(100_000, len(df_full)), random_state=42)


def entrenar_modelos_ml(df_ml):
    print("\n🤖 Entrenando modelos Gradient Boosting (HistGradientBoosting) para Retrasos y Costos...")

    features = ["origen", "destino", "paqueteria", "peso_kg", "distancia_km", "precio_diesel_mxn", "nivel_alerta_x"]
    X = df_ml[features]
    y_retraso = df_ml["retrasado"]
    y_costo = df_ml["costo_mxn"]

    cat_cols = ["origen", "destino", "paqueteria"]
    num_cols = ["peso_kg", "distancia_km", "precio_diesel_mxn", "nivel_alerta_x"]

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    modelo_retraso = Pipeline([
        ("prep", preprocesador),
        ("clf", HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=8,
            learning_rate=0.08,
            random_state=42
        ))
    ])

    modelo_costo = Pipeline([
        ("prep", preprocesador),
        ("reg", HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=8,
            learning_rate=0.08,
            random_state=42
        ))
    ])

    X_train, X_test, y_ret_train, y_ret_test, y_cos_train, y_cos_test = train_test_split(
        X, y_retraso, y_costo, test_size=0.2, random_state=42
    )

    modelo_retraso.fit(X_train, y_ret_train)
    modelo_costo.fit(X_train, y_cos_train)

    preds_ret = modelo_retraso.predict(X_test)
    probs_ret = modelo_retraso.predict_proba(X_test)[:, 1]
    preds_cos = modelo_costo.predict(X_test)

    acc = accuracy_score(y_ret_test, preds_ret)
    auc = roc_auc_score(y_ret_test, probs_ret)
    mae = mean_absolute_error(y_cos_test, preds_cos)

    print(f"✅ HistGradientBoosting Retrasos -> Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.3f}")
    print(f"✅ HistGradientBoosting Costos   -> Error Promedio (MAE): ${mae:,.2f} MXN")

    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo_retraso, "models/modelo_retraso.pkl")
    joblib.dump(modelo_costo, "models/modelo_costo.pkl")
    print("💾 Modelos ligeros de Gradient Boosting guardados en 'models/'.")


if __name__ == "__main__":
    ruta_parquet = obtener_ruta_dataset()
    df_entrenamiento = procesar_con_pyspark(ruta_parquet)
    entrenar_modelos_ml(df_entrenamiento)