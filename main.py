from fastapi import FastAPI
from pydantic import BaseModel
from etl.transformextra import obtener_ventas_recientes
from estadisticas.general import resumen_modelo, alerta_tendencia_anomala
from etl.transformgeneral import extract_and_transform_general_timeseries
from etl.transformproductos import extract_and_transform_product_timeseries
from tensorflow.modelgeneral import train_general_timeseries_model, load_general_timeseries_model
from tensorflow.modelproductos import train_product_timeseries_models, load_product_timeseries_model
from tensorflow.prediction import predict_general_timeseries, predict_product_timeseries
import os
import pandas as pd

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O puedes limitarlo a tu frontend: ["https://tudominio.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODELO_GENERAL_PATH = "modelos/general/modelo_general_timeseries.pkl"
MODELO_PRODUCTOS_DIR = "modelos/productos/"

class TimeseriesRequest(BaseModel):
    tenant_id: str
    dias_historial: int = 90
    dias_prediccion: int = 90

class TimeseriesProductoRequest(TimeseriesRequest):
    producto_id: str

@app.post("/train_general_timeseries/")
def train_general_timeseries_api(req: TimeseriesRequest):
    tenant_id = req.tenant_id
    dias_historial = req.dias_historial

    print(f"[LOG][API] Entrenamiento general: tenant_id={tenant_id}, dias_historial={dias_historial}")
    df = extract_and_transform_general_timeseries(tenant_id)
    print(f"[LOG][API] DataFrame general shape: {df.shape}")
    if df.empty:
        print("[LOG][API] No hay datos para entrenamiento general.")
        return {"mensaje": "No hay datos para entrenar el modelo.", "metrica": {}}
    df_hist = df.sort_values("ds").tail(dias_historial)
    os.makedirs(os.path.dirname(MODELO_GENERAL_PATH), exist_ok=True)
    model = train_general_timeseries_model(df_hist, MODELO_GENERAL_PATH)
    print("[LOG][API] Modelo general entrenado y guardado.")
    return {"mensaje": "Modelo de series temporales entrenado y guardado."}

@app.post("/predict_general_timeseries/")
def predict_general_timeseries_api(req: TimeseriesRequest):
    tenant_id = req.tenant_id
    dias_prediccion = req.dias_prediccion  # normalmente 90

    print(f"[LOG][API] Predicción general: tenant_id={tenant_id}, dias_prediccion={dias_prediccion}")
    df = extract_and_transform_general_timeseries(tenant_id)  # ¡Ya trae 90 días!

    print(f"[LOG][API] DataFrame general shape: {df.shape}")
    if df.empty:
        print("[LOG][API] No hay datos para predicción general.")
        return {"historial": [], "predicciones": []}

    if os.path.exists(MODELO_GENERAL_PATH):
        print(f"[LOG][API] Cargando modelo general de {MODELO_GENERAL_PATH}")
        model = load_general_timeseries_model(MODELO_GENERAL_PATH)
    else:
        print("[LOG][API] Modelo general no existe, entrenando...")
        model = train_general_timeseries_model(df, MODELO_GENERAL_PATH)

    historial = [
        {"fecha": row["ds"].strftime("%Y-%m-%d"), "venta_total": float(row["y"])}
        for _, row in df.iterrows()
    ]
    print(f"[LOG][API] Historial general: {historial if len(historial)<10 else '...'}")
    predicciones_df = predict_general_timeseries(df, model, dias_prediccion=dias_prediccion)
    predicciones = predicciones_df.to_dict(orient="records")
    print(f"[LOG][API] Predicciones general: {predicciones if len(predicciones)<10 else '...'}")

    # Prints para depurar
    print(f"[LOG][API] Días en historial: {len(historial)} ({historial[0]['fecha']} a {historial[-1]['fecha']})")
    print(f"[LOG][API] Días en predicción: {len(predicciones)} ({predicciones[0]['fecha']} a {predicciones[-1]['fecha']})")

    return {
        "historial": historial,
        "predicciones": predicciones
    }

@app.post("/train_product_timeseries/")
def train_product_timeseries_api(req: TimeseriesRequest):
    tenant_id = req.tenant_id
    dias_historial = req.dias_historial

    print(f"[LOG][API] Entrenamiento productos: tenant_id={tenant_id}, dias_historial={dias_historial}")
    df = extract_and_transform_product_timeseries(tenant_id)
    print(f"[LOG][API] DataFrame productos shape: {df.shape}")
    if df.empty:
        print("[LOG][API] No hay datos para entrenar modelos por producto.")
        return {"mensaje": "No hay datos para entrenar modelos por producto.", "metrica": {}}
    df_hist = df.sort_values("ds").groupby("producto_id").tail(dias_historial)
    os.makedirs(MODELO_PRODUCTOS_DIR, exist_ok=True)
    modelos = train_product_timeseries_models(df_hist, MODELO_PRODUCTOS_DIR)

    from bd.database import get_mongo_db
    db = get_mongo_db()
    producto_ids = list(modelos.keys())
    print(f"[LOG][API] Modelos entrenados para producto_ids: {producto_ids}")
    productos = list(db.productos.find({
        "_id": {"$in": [__import__('bson').ObjectId(pid) for pid in producto_ids]}
    }))
    id2titulo = {str(p["_id"]): p.get("titulo", "") for p in productos}
    log_entrenados = [
        {"producto_id": pid, "titulo": id2titulo.get(pid, "")}
        for pid in producto_ids
    ]
    print(f"[LOG][API] Productos entrenados: {log_entrenados}")

    return {
        "mensaje": f"Modelos entrenados para {len(modelos)} productos.",
        "productos_entrenados": log_entrenados
    }

@app.post("/predict_product_timeseries/")
def predict_product_timeseries_api(req: TimeseriesProductoRequest):
    tenant_id = req.tenant_id
    producto_id = req.producto_id
    dias_historial = req.dias_historial
    dias_prediccion = req.dias_prediccion

    print(f"\n[LOG][API] --- INICIO ENDPOINT /predict_product_timeseries/ ---")
    print(f"[LOG][API] Request: tenant_id={tenant_id}, producto_id={producto_id}, dias_historial={dias_historial}, dias_prediccion={dias_prediccion}")

    df = extract_and_transform_product_timeseries(tenant_id)
    print(f"[LOG][API] DataFrame extraído shape: {df.shape}")
    print(f"[LOG][API] Columnas: {df.columns}")
    if not df.empty:
        print(f"[LOG][API] Cantidad de producto_id únicos en DF: {df['producto_id'].nunique()}")
        print(f"[LOG][API] Ejemplo de producto_ids: {df['producto_id'].unique()[:10]}")
    else:
        print("[LOG][API] DataFrame de ventas por producto VACÍO tras extracción.")

    producto_id_str = str(producto_id)
    df['producto_id'] = df['producto_id'].astype(str)
    df_producto = df[df["producto_id"] == producto_id_str].sort_values("ds")

    # Rellenar días faltantes hasta hoy en historial
    if not df_producto.empty:
        hoy = pd.to_datetime(pd.Timestamp.now().date())
        fechas_historial = pd.date_range(start=df_producto["ds"].min(), end=hoy, freq="D")
        df_producto = df_producto.set_index("ds").reindex(fechas_historial, fill_value=0).reset_index()
        df_producto = df_producto.rename(columns={"index": "ds"})
        df_producto = df_producto.sort_values("ds").tail(dias_historial)
        print(f"[LOG][API] Historial tras reindex y tail: {df_producto.shape}")

    if df_producto.empty:
        print("[LOG][API] DataFrame filtrado por producto está VACÍO. No hay historial para predecir.")
        print(f"[LOG][API] --- FIN ENDPOINT /predict_product_timeseries/ ---\n")
        return {"historial": [], "predicciones": []}

    modelo_path = os.path.join(MODELO_PRODUCTOS_DIR, f"{producto_id}.pkl")
    if os.path.exists(modelo_path):
        print(f"[LOG][API] Cargando modelo existente en: {modelo_path}")
        model = load_product_timeseries_model(MODELO_PRODUCTOS_DIR, producto_id)
    else:
        print(f"[LOG][API] No existe modelo en disco, entrenando nuevo modelo Prophet para producto_id={producto_id_str}")
        from prophet import Prophet
        model = Prophet(daily_seasonality=True)
        model.fit(df_producto[["ds", "y"]])
        with open(modelo_path, "wb") as f:
            import pickle
            pickle.dump(model, f)
        print(f"[LOG][API] Modelo Prophet entrenado y guardado en {modelo_path}")

    historial = [
        {"fecha": row["ds"].strftime("%Y-%m-%d"), "monto_vendido": float(row["y"])}
        for _, row in df_producto.iterrows()
    ]
    print(f"[LOG][API] Historial generado: {historial if len(historial) < 10 else '...'} (total {len(historial)})")

    predicciones_df = predict_product_timeseries(df_producto, model, dias_prediccion=dias_prediccion)
    predicciones = predicciones_df.to_dict(orient="records")
    print(f"[LOG][API] Predicciones generadas: {predicciones if len(predicciones) < 10 else '...'} (total {len(predicciones)})")

    print(f"[LOG][API] --- FIN ENDPOINT /predict_product_timeseries/ ---\n")
    return {
        "producto_id": producto_id,
        "historial": historial,
        "predicciones": predicciones
    }

@app.post("/ventas_recientes/")
def endpoint_ventas_recientes(req: TimeseriesRequest):
    df = obtener_ventas_recientes(req.tenant_id)
    return df.to_dict(orient="records")

@app.post("/resumen_modelo/")
def endpoint_resumen_modelo(req: TimeseriesRequest):
    predicciones = predict_general_timeseries(req.tenant_id, req.dias_historial, req.dias_prediccion)
    if not isinstance(predicciones, list) or not predicciones:
        return {"mensaje": "No se generaron predicciones"}
    return resumen_modelo(predicciones, req.dias_historial, req.dias_prediccion)

@app.post("/alerta_tendencia/")
def endpoint_alerta_tendencia(req: TimeseriesRequest):
    predicciones = predict_general_timeseries(req.tenant_id, req.dias_historial, req.dias_prediccion)
    if not isinstance(predicciones, list) or not predicciones:
        return {"alerta": False, "detalle": "No se generaron predicciones"}
    return alerta_tendencia_anomala(predicciones)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)